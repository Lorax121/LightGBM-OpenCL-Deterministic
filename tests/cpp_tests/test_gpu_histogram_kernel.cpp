/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Raw OpenCL histogram kernel tests for the deterministic (int64 fixed-point)
 * path, covering the 16-, 64- and 256-bin kernel families. Each kernel is
 * compiled directly from its .cl file with -D DETERMINISTIC_GPU_HIST=1 (and
 * without any fast-math flags), driven with synthetic feature data and real
 * GPUHistogramQuantizer quantized values, and compared against an int64 CPU
 * reference using exact equality. Every configuration is repeated three times
 * and must produce byte-identical int64 histograms.
 */
#include <gtest/gtest.h>

#include <boost/compute/core.hpp>

#include "../../src/treelearner/gpu_histogram_quantizer.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// The kernel files double as C++ raw-string literals; the host skips the
// ")\n#endif\n" prefix (9 characters) exactly like gpu_tree_learner.cpp does.
static const char* kKernel16SrcRaw = {
#include "../../src/treelearner/ocl/histogram16.cl"
};
static const char* kKernel64SrcRaw = {
#include "../../src/treelearner/ocl/histogram64.cl"
};
static const char* kKernel256SrcRaw = {
#include "../../src/treelearner/ocl/histogram256.cl"
};
static const char* kStatGatherSrcRaw = {
#include "../../src/treelearner/ocl/stat_gather.cl"
};

namespace {

using LightGBM::GPUHistogramQuantizer;
using LightGBM::data_size_t;
using LightGBM::score_t;

constexpr int kNumExamples = 2048;
constexpr size_t kPad = 2048;             // prefetch padding (kernel reads [i + subglobal_size])
constexpr size_t kPadded = kNumExamples + kPad;
constexpr int kNumTuples = 2;             // >= 2 so per-tuple output offsets are exercised

// One kernel family
struct KernelSpec {
  const char* source;      // kernel source (already offset by the +9 skip)
  const char* name;        // kernel name
  int num_bins;            // NUM_BINS of the kernel
  int dword_features;      // features per feature4 tuple (8 for the 16-bin kernel, else 4)
  int num_features;        // kNumTuples * dword_features
};

const KernelSpec kKernels[] = {
    {kKernel16SrcRaw + 9, "histogram16", 16, 8, kNumTuples * 8},
    {kKernel64SrcRaw + 9, "histogram64", 64, 4, kNumTuples * 4},
    {kKernel256SrcRaw + 9, "histogram256", 256, 4, kNumTuples * 4},
};

// Gradient sign pattern for the synthetic data
enum class GradSign { kMixed, kPositive, kNegative };

struct KernelParams {
  int kernel_index;     // index into kKernels
  int pwf;              // log2 of the number of workgroups per feature (0, 1, 2)
  bool const_hessian;   // CONST_HESSIAN
  bool all_features;    // ENABLE_ALL_FEATURES == 1 (no feature mask)
  bool ignore_indices;  // IGNORE_INDICES == 1 (full-data kernel)
  int mult;             // device bin multiplier (redistribution), 1 or 2
  GradSign sign;        // gradient sign pattern
};

const KernelSpec& Spec(const KernelParams& p) { return kKernels[p.kernel_index]; }

std::string ParamsToString(const KernelParams& p) {
  std::ostringstream ss;
  ss << Spec(p).name << ", pwf=" << p.pwf << ", const=" << p.const_hessian
     << ", allfeats=" << p.all_features << ", fulldata=" << p.ignore_indices
     << ", mult=" << p.mult << ", sign=" << static_cast<int>(p.sign);
  return ss.str();
}

// ---------------------------------------------------------------------------
// Build options for one kernel variant (mirrors GPUTreeLearner::BuildGPUKernels)
// ---------------------------------------------------------------------------
std::string BuildOpts(const KernelParams& p) {
  std::ostringstream opts;
  opts << " -D DETERMINISTIC_GPU_HIST=1 -D USE_CONSTANT_BUF=0"
       << " -D POWER_FEATURE_WORKGROUPS=" << p.pwf
       << " -D CONST_HESSIAN=" << (p.const_hessian ? 1 : 0)
       << " -D ENABLE_ALL_FEATURES=" << (p.all_features ? 1 : 0);
  if (p.ignore_indices) {
    opts << " -D IGNORE_INDICES=1";
  }
  // NOTE: deliberately no -cl-mad-enable / -cl-no-signed-zeros / -cl-fast-relaxed-math
  return opts.str();
}

// ---------------------------------------------------------------------------
// OpenCL driver
// ---------------------------------------------------------------------------
struct Dev {
  boost::compute::device device;
  boost::compute::context context;
  boost::compute::command_queue queue;

  Dev()
      : device(boost::compute::system::default_device()),
        context(device),
        queue(context, device) {}
};

// Run the deterministic kernel once; returns the final int64 output:
// num_features x 2 x num_bins, interleaved [grad, hess] per feature.
std::vector<int64_t> RunKernel(Dev* dev, const KernelParams& p,
                               const std::vector<unsigned char>& packed,
                               const std::vector<unsigned char>& mask,
                               const std::vector<uint32_t>& indices,
                               const std::vector<int64_t>& ordered_grads,
                               const std::vector<int64_t>& ordered_hess,
                               int64_t quantized_const_hessian) {
  const KernelSpec& spec = Spec(p);
  const int num_wg = (1 << p.pwf) * kNumTuples;
  const int block_size = 2 * spec.num_bins;
  const size_t kOutBytes = spec.num_features * block_size * sizeof(int64_t);
  const size_t kSubhistBytes = static_cast<size_t>(num_wg) * spec.dword_features * block_size * sizeof(int64_t);

  // programs are cached per (kernel, build-option); gtest runs are single-threaded
  static std::map<std::string, boost::compute::program> program_cache;
  const std::string cache_key = std::string(spec.name) + "|" + BuildOpts(p);
  auto it = program_cache.find(cache_key);
  if (it == program_cache.end()) {
    it = program_cache.emplace(cache_key, boost::compute::program::build_with_source(spec.source, dev->context, BuildOpts(p))).first;
  }
  boost::compute::kernel kernel = it->second.create_kernel(spec.name);

  // feature data: tuple-major [tuple][example] uchar4, like the host device_features_
  boost::compute::buffer feature_data(dev->context, packed.size(),
                                      boost::compute::memory_object::read_only);
  dev->queue.enqueue_write_buffer(feature_data, 0, packed.size(), packed.data());

  boost::compute::buffer feature_masks(dev->context, mask.size(),
                                       boost::compute::memory_object::read_only);
  dev->queue.enqueue_write_buffer(feature_masks, 0, mask.size(), mask.data());

  boost::compute::buffer data_indices(dev->context, indices.size() * sizeof(uint32_t),
                                      boost::compute::memory_object::read_only);
  dev->queue.enqueue_write_buffer(data_indices, 0, indices.size() * sizeof(uint32_t), indices.data());

  boost::compute::buffer grads(dev->context, ordered_grads.size() * sizeof(int64_t),
                               boost::compute::memory_object::read_only);
  dev->queue.enqueue_write_buffer(grads, 0, ordered_grads.size() * sizeof(int64_t), ordered_grads.data());

  boost::compute::buffer hess(dev->context, ordered_hess.size() * sizeof(int64_t),
                              boost::compute::memory_object::read_only);
  if (!p.const_hessian) {
    dev->queue.enqueue_write_buffer(hess, 0, ordered_hess.size() * sizeof(int64_t), ordered_hess.data());
  }

  boost::compute::buffer subhist(dev->context, kSubhistBytes,
                                 boost::compute::memory_object::read_write);
  std::vector<int64_t> zero_subhist(kSubhistBytes / sizeof(int64_t), 0);
  dev->queue.enqueue_write_buffer(subhist, 0, kSubhistBytes, zero_subhist.data());

  boost::compute::buffer sync_counters(dev->context, kNumTuples * sizeof(uint32_t),
                                       boost::compute::memory_object::read_write);
  std::vector<uint32_t> zero_counters(kNumTuples, 0);
  dev->queue.enqueue_write_buffer(sync_counters, 0, zero_counters.size() * sizeof(uint32_t), zero_counters.data());

  boost::compute::buffer hist_out(dev->context, kOutBytes,
                                  boost::compute::memory_object::read_write);
  std::vector<int64_t> zero_out(spec.num_features * block_size, 0);
  dev->queue.enqueue_write_buffer(hist_out, 0, kOutBytes, zero_out.data());

  kernel.set_arg(0, feature_data);
  kernel.set_arg(1, feature_masks);
  kernel.set_arg(2, static_cast<uint32_t>(kNumExamples));  // feature_size
  kernel.set_arg(3, data_indices);
  kernel.set_arg(4, static_cast<uint32_t>(kNumExamples));  // num_data
  kernel.set_arg(5, grads);
  if (p.const_hessian) {
    kernel.set_arg(6, quantized_const_hessian);
  } else {
    kernel.set_arg(6, hess);
  }
  kernel.set_arg(7, subhist);
  kernel.set_arg(8, sync_counters);
  kernel.set_arg(9, hist_out);

  dev->queue.enqueue_1d_range_kernel(kernel, 0, static_cast<size_t>(num_wg) * 256, 256);
  dev->queue.finish();

  std::vector<int64_t> out(spec.num_features * block_size, 0);
  dev->queue.enqueue_read_buffer(hist_out, 0, kOutBytes, out.data());
  dev->queue.finish();
  return out;
}

// ---------------------------------------------------------------------------
// int64 CPU reference: per feature slot, sum the quantized values grouped by
// the device bin (the packed value). For constant hessian the hessian bin
// accumulates quantized_const_hessian per row (count * quantized_const_hessian).
// ---------------------------------------------------------------------------
std::vector<int64_t> ComputeReference(const KernelParams& p,
                                      const std::vector<unsigned char>& packed,
                                      const std::vector<unsigned char>& mask,
                                      const std::vector<int64_t>& quantized_grads,
                                      const std::vector<int64_t>& quantized_hess,
                                      int64_t quantized_const_hessian) {
  const KernelSpec& spec = Spec(p);
  const int block_size = 2 * spec.num_bins;
  const int per_tuple = spec.dword_features;
  const int packed_per_example = per_tuple == 8 ? 4 : per_tuple;
  std::vector<int64_t> out(spec.num_features * block_size, 0);
  for (int j = 0; j < kNumExamples; ++j) {
    for (int s = 0; s < spec.num_features; ++s) {
      if (!p.all_features && mask[s] == 0) {
        // disabled feature: the kernel never publishes its block
        continue;
      }
      const int t = s / per_tuple;
      const int slot = s % per_tuple;
      int dev_bin;
      if (per_tuple == 8) {
        // unpack the 4-bit nibble
        dev_bin = (packed[(t * kNumExamples + j) * 4 + (slot >> 1)] >> ((slot & 1) << 2)) & 0xf;
      } else {
        dev_bin = packed[(t * kNumExamples + j) * per_tuple + slot];
      }
      out[s * block_size + 2 * dev_bin] += quantized_grads[j];
      out[s * block_size + 2 * dev_bin + 1] += p.const_hessian ? quantized_const_hessian : quantized_hess[j];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The actual test
// ---------------------------------------------------------------------------
class GpuDeterministicHistogramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (boost::compute::system::platforms().empty()) {
      GTEST_SKIP() << "No OpenCL platform available";
    }
    dev_.reset(new Dev());
    // deterministic kernels must fit into the device local memory
    device_local_mem_ = dev_->device.get_info<size_t>(CL_DEVICE_LOCAL_MEM_SIZE);
  }

  std::unique_ptr<Dev> dev_;
  size_t device_local_mem_ = 0;
};

TEST_F(GpuDeterministicHistogramTest, RawInt64HistogramMatchesCpuReference) {
  const std::vector<KernelParams> params = {
      // kernel, pwf, const, allfeats, fulldata, mult, sign
      {0, 0, false, true, true, 1, GradSign::kMixed},
      {0, 0, false, true, false, 1, GradSign::kMixed},
      {0, 0, false, false, true, 1, GradSign::kMixed},
      {0, 0, false, false, false, 1, GradSign::kMixed},
      {0, 0, true, true, true, 1, GradSign::kMixed},
      {0, 0, true, true, false, 1, GradSign::kMixed},
      {0, 0, true, false, true, 1, GradSign::kMixed},
      {0, 0, true, false, false, 1, GradSign::kMixed},
      {0, 1, false, true, true, 1, GradSign::kMixed},
      {0, 1, false, true, false, 1, GradSign::kMixed},
      {0, 1, false, false, true, 1, GradSign::kMixed},
      {0, 1, false, false, false, 1, GradSign::kMixed},
      {0, 1, true, true, true, 1, GradSign::kMixed},
      {0, 1, true, true, false, 1, GradSign::kMixed},
      {0, 1, true, false, true, 1, GradSign::kMixed},
      {0, 1, true, false, false, 1, GradSign::kMixed},
      {0, 2, false, true, true, 1, GradSign::kMixed},
      {0, 2, false, true, false, 1, GradSign::kMixed},
      {0, 2, false, false, true, 1, GradSign::kMixed},
      {0, 2, false, false, false, 1, GradSign::kMixed},
      {0, 2, true, true, true, 1, GradSign::kMixed},
      {0, 2, true, true, false, 1, GradSign::kMixed},
      {0, 2, true, false, true, 1, GradSign::kMixed},
      {0, 2, true, false, false, 1, GradSign::kMixed},
      // redistribution (mult = 2)
      {0, 1, false, true, false, 2, GradSign::kMixed},
      {0, 1, false, false, false, 2, GradSign::kMixed},
      {0, 1, true, true, false, 2, GradSign::kMixed},
      {0, 2, false, true, false, 2, GradSign::kMixed},
      {0, 2, true, false, false, 2, GradSign::kMixed},
      // gradient sign patterns on a representative subset
      {0, 1, false, true, false, 1, GradSign::kPositive},
      {0, 1, false, false, false, 1, GradSign::kNegative},
      {0, 2, true, true, false, 1, GradSign::kPositive},
      {0, 2, true, false, true, 1, GradSign::kNegative},
  };

  for (int k = 0; k < static_cast<int>(sizeof(kKernels) / sizeof(kKernels[0])); ++k) {
    const KernelSpec& spec = kKernels[k];
    SCOPED_TRACE(std::string("kernel family: ") + spec.name);

    const int max_feature_value = spec.num_bins - 1;
    const int num_bin = spec.num_bins;
    const int per_tuple = spec.dword_features;

    // masks: one feature-mask vector per tuple (uchar4 for 64/256, uchar8 for 16)
    std::vector<unsigned char> mask_all(spec.num_features, 0xff);
    std::vector<unsigned char> mask_partial(spec.num_features, 0xff);
    // disable feature 1 of tuple 0 and feature 2 of tuple 1
    mask_partial[1] = 0x00;
    mask_partial[per_tuple + 2] = 0x00;

    // synthetic per-example statistics (score_t); hessians are always positive
    std::vector<score_t> grads_f(kNumExamples), hess_f(kNumExamples);
    for (int j = 0; j < kNumExamples; ++j) {
      grads_f[j] = static_cast<score_t>(((j % 17) - 8) * 0.5);
      hess_f[j] = static_cast<score_t>((j % 5) + 1);
    }

    for (const KernelParams& base : params) {
      KernelParams p = base;
      p.kernel_index = k;

      SCOPED_TRACE(ParamsToString(p));

      // quantize with the real quantizer (fixed-point factors are powers of two)
      GPUHistogramQuantizer quantizer;
      quantizer.ComputeState(static_cast<data_size_t>(kNumExamples), grads_f.data(),
                             hess_f.data(), p.const_hessian, 0);
      std::vector<int64_t> quantized_grads(kNumExamples), quantized_hess(kNumExamples, 0);
      quantizer.QuantizeGradients(grads_f.data(), quantized_grads.data(), kNumExamples);
      int64_t quantized_const_hessian = 0;
      if (p.const_hessian) {
        quantized_const_hessian = quantizer.QuantizeConstHessian(hess_f[0], 0);
      } else {
        quantizer.QuantizeHessians(hess_f.data(), quantized_hess.data(), kNumExamples);
      }

      // build the feature data with the AllocateGPUMemory packing formula:
      // value = RawGet * mult + ((j + s) & (mult - 1)); device bins are within
      // the kernel's feature range. Layout is TUPLE-MAJOR: [tuple][example]
      // uchar4. The 16-bin kernel packs 8 features as 4-bit nibbles (2 per byte);
      // the 64/256 kernels use one byte per feature.
      const int num_bin_here = p.mult == 1 ? num_bin : num_bin / p.mult;
      const int packed_per_example = per_tuple == 8 ? 4 : per_tuple;
      std::vector<unsigned char> packed(kNumTuples * kNumExamples * packed_per_example, 0);
      for (int t = 0; t < kNumTuples; ++t) {
        for (int j = 0; j < kNumExamples; ++j) {
          for (int s = 0; s < per_tuple; ++s) {
            const int feat = t * per_tuple + s;
            const int raw = (j * 37 + feat * 13) % num_bin_here;
            const int value = raw * p.mult + ((j + s) & (p.mult - 1));
            ASSERT_LE(value, max_feature_value);
            if (per_tuple == 8) {
              // nibble-packed: feature s -> byte (s >> 1), nibble ((s & 1) << 2)
              packed[(t * kNumExamples + j) * 4 + (s >> 1)] |=
                  static_cast<unsigned char>(value << ((s & 1) << 2));
            } else {
              packed[(t * kNumExamples + j) * per_tuple + s] =
                  static_cast<unsigned char>(value);
            }
          }
        }
      }
      const std::vector<unsigned char>& mask = p.all_features ? mask_all : mask_partial;

      // indices: reversal permutation, padded with zeros for the prefetch reads
      std::vector<uint32_t> indices(kPadded, 0);
      for (int j = 0; j < kNumExamples; ++j) {
        indices[j] = static_cast<uint32_t>(kNumExamples - 1 - j);
      }

      // ordered arrays: gathered by indices for the indexed kernel, identity for fulldata
      std::vector<int64_t> ordered_grads(kPadded, 0), ordered_hess(kPadded, 0);
      for (int j = 0; j < kNumExamples; ++j) {
        const uint32_t idx = p.ignore_indices ? static_cast<uint32_t>(j) : indices[j];
        ordered_grads[j] = quantized_grads[idx];
        if (!p.const_hessian) {
          ordered_hess[j] = quantized_hess[idx];
        }
      }

      const std::vector<int64_t> reference = ComputeReference(
          p, packed, mask, quantized_grads, quantized_hess, quantized_const_hessian);

      // kernel local memory validation
      const std::string opts = BuildOpts(p);
      static std::map<std::string, boost::compute::program> prog_cache;
      const std::string cache_key = std::string(spec.name) + "|" + opts;
      auto pit = prog_cache.find(cache_key);
      if (pit == prog_cache.end()) {
        pit = prog_cache.emplace(cache_key, boost::compute::program::build_with_source(spec.source, dev_->context, opts)).first;
      }
      const size_t kernel_local_mem =
          pit->second.create_kernel(spec.name).get_work_group_info<size_t>(dev_->device, CL_KERNEL_LOCAL_MEM_SIZE);
      EXPECT_LE(kernel_local_mem, device_local_mem_);

      // run three times; all runs must be byte-identical and equal to the CPU reference
      std::vector<int64_t> first;
      const int block_size = 2 * spec.num_bins;
      for (int run = 0; run < 3; ++run) {
        const std::vector<int64_t> gpu = RunKernel(
            dev_.get(), p, packed, mask, indices, ordered_grads, ordered_hess, quantized_const_hessian);
        if (run == 0) {
          first = gpu;
        } else {
          EXPECT_TRUE(first == gpu) << "run " << run << " differs from run 0";
        }
        // enabled features must match the CPU int64 reference exactly
        for (int s = 0; s < spec.num_features; ++s) {
          if (!p.all_features && mask[s] == 0) {
            // disabled feature: its final block must stay zero
            for (int b = 0; b < block_size; ++b) {
              EXPECT_EQ(gpu[s * block_size + b], 0) << "disabled feature slot " << s << " bin " << b << " not zero";
            }
          } else {
            for (int b = 0; b < block_size; ++b) {
              EXPECT_EQ(gpu[s * block_size + b], reference[s * block_size + b])
                  << "slot " << s << " bin " << b;
            }
          }
        }
      }
    }
  }
}

TEST_F(GpuDeterministicHistogramTest, StatGatherPreservesPartitionOrder) {
  constexpr int kRows = 4099;
  constexpr int kSelected = 1025;
  std::vector<int64_t> full_gradients(kRows), full_hessians(kRows);
  std::vector<int32_t> indices(kSelected);
  for (int i = 0; i < kRows; ++i) {
    full_gradients[i] = static_cast<int64_t>(i) * 7919 - 1234567;
    full_hessians[i] = static_cast<int64_t>(i) * 104729 + 17;
  }
  for (int i = 0; i < kSelected; ++i) {
    indices[i] = (i * 3571 + 97) % kRows;
  }

  boost::compute::program program = boost::compute::program::build_with_source(
      kStatGatherSrcRaw + 9, dev_->context);
  boost::compute::kernel kernel =
      program.create_kernel("gather_ordered_stats_i64");
  boost::compute::buffer full_gradients_buffer(
      dev_->context, full_gradients.size() * sizeof(int64_t));
  boost::compute::buffer full_hessians_buffer(
      dev_->context, full_hessians.size() * sizeof(int64_t));
  boost::compute::buffer indices_buffer(
      dev_->context, indices.size() * sizeof(int32_t));
  boost::compute::buffer ordered_gradients_buffer(
      dev_->context, indices.size() * sizeof(int64_t));
  boost::compute::buffer ordered_hessians_buffer(
      dev_->context, indices.size() * sizeof(int64_t));
  dev_->queue.enqueue_write_buffer(full_gradients_buffer, 0,
      full_gradients.size() * sizeof(int64_t), full_gradients.data());
  dev_->queue.enqueue_write_buffer(full_hessians_buffer, 0,
      full_hessians.size() * sizeof(int64_t), full_hessians.data());
  dev_->queue.enqueue_write_buffer(indices_buffer, 0,
      indices.size() * sizeof(int32_t), indices.data());

  kernel.set_args(full_gradients_buffer, full_hessians_buffer, indices_buffer,
                  ordered_gradients_buffer, ordered_hessians_buffer,
                  kSelected, 1);
  const size_t global_size = (static_cast<size_t>(kSelected) + 255u) & ~size_t{255u};
  dev_->queue.enqueue_1d_range_kernel(kernel, 0, global_size, 256);
  std::vector<int64_t> gathered_gradients(kSelected), gathered_hessians(kSelected);
  dev_->queue.enqueue_read_buffer(ordered_gradients_buffer, 0,
      gathered_gradients.size() * sizeof(int64_t), gathered_gradients.data());
  dev_->queue.enqueue_read_buffer(ordered_hessians_buffer, 0,
      gathered_hessians.size() * sizeof(int64_t), gathered_hessians.data());

  for (int i = 0; i < kSelected; ++i) {
    EXPECT_EQ(gathered_gradients[i], full_gradients[indices[i]]);
    EXPECT_EQ(gathered_hessians[i], full_hessians[indices[i]]);
  }
}

}  // namespace
