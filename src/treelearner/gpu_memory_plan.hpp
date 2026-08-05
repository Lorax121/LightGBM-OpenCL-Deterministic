/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_GPU_MEMORY_PLAN_HPP_
#define LIGHTGBM_SRC_TREELEARNER_GPU_MEMORY_PLAN_HPP_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace LightGBM {

struct GPUMemoryPlanInput {
  uint64_t num_data;
  uint64_t num_feature_tuples;
  uint64_t dword_features;
  uint64_t device_bin_size;
  uint64_t histogram_entry_size;
  uint64_t preallocated_workgroups;
  uint64_t score_size;
  uint64_t data_index_size;
  bool deterministic;
  bool constant_hessian;
  uint64_t global_memory_size;
  uint64_t max_memory_allocation_size;
};

struct GPUMemoryPlan {
  uint64_t feature_bytes;
  uint64_t device_working_bytes;
  uint64_t host_pinned_bytes;
  uint64_t stat_gather_bytes;
  uint64_t in_core_bytes;
  uint64_t in_core_with_stat_gather_bytes;
  uint64_t largest_in_core_allocation_bytes;
  uint64_t usable_device_bytes;
  bool in_core_fits_hard_limits;
  bool in_core_fits_with_headroom;
  bool stat_gather_fits_with_headroom;
};

namespace gpu_memory_plan {

inline uint64_t CheckedAdd(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::overflow_error("OpenCL memory plan addition overflow");
  }
  return left + right;
}

inline uint64_t CheckedMultiply(uint64_t left, uint64_t right) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    throw std::overflow_error("OpenCL memory plan multiplication overflow");
  }
  return left * right;
}

inline uint64_t AddTo(uint64_t total, uint64_t value) {
  return CheckedAdd(total, value);
}

}  // namespace gpu_memory_plan

inline GPUMemoryPlan MakeGPUMemoryPlan(const GPUMemoryPlanInput& input) {
  using gpu_memory_plan::AddTo;
  using gpu_memory_plan::CheckedAdd;
  using gpu_memory_plan::CheckedMultiply;

  if (input.num_data == 0 || input.num_feature_tuples == 0 ||
      input.dword_features == 0 || input.device_bin_size == 0 ||
      input.histogram_entry_size == 0 || input.score_size == 0 ||
      input.data_index_size == 0 || input.global_memory_size == 0 ||
      input.max_memory_allocation_size == 0) {
    throw std::invalid_argument("OpenCL memory plan inputs must be non-zero");
  }

  constexpr uint64_t kFeatureTupleSize = 4;
  constexpr uint64_t kFixedPointStatSize = sizeof(int64_t);
  constexpr uint64_t kPrefetchRows = 256 * (uint64_t{1} << 10);
  const uint64_t allocated_num_data = CheckedAdd(input.num_data, kPrefetchRows);
  const uint64_t feature_tuple_bytes =
      CheckedMultiply(input.num_data, kFeatureTupleSize);
  const uint64_t feature_bytes =
      CheckedMultiply(input.num_feature_tuples, feature_tuple_bytes);

  const uint64_t score_buffers = CheckedMultiply(
      CheckedMultiply(allocated_num_data, input.score_size), 2);
  const uint64_t quantized_buffers = input.deterministic
      ? CheckedMultiply(
            CheckedMultiply(allocated_num_data, kFixedPointStatSize), 2)
      : 0;
  const uint64_t index_buffer =
      CheckedMultiply(allocated_num_data, input.data_index_size);
  const uint64_t feature_mask_bytes =
      CheckedMultiply(input.num_feature_tuples, input.dword_features);
  const uint64_t subhistogram_bytes = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(input.preallocated_workgroups, input.dword_features),
          input.device_bin_size),
      input.histogram_entry_size);
  const uint64_t histogram_output_bytes = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(input.num_feature_tuples, input.dword_features),
          input.device_bin_size),
      input.histogram_entry_size);
  const uint64_t sync_counter_bytes =
      CheckedMultiply(input.num_feature_tuples, sizeof(int32_t));

  uint64_t device_working_bytes = 0;
  for (const uint64_t bytes : {score_buffers, quantized_buffers, index_buffer,
                               feature_mask_bytes, subhistogram_bytes,
                               histogram_output_bytes, sync_counter_bytes}) {
    device_working_bytes = AddTo(device_working_bytes, bytes);
  }

  // CL_MEM_USE_HOST_PTR / CL_MEM_ALLOC_HOST_PTR buffers are kept separate from
  // the VRAM budget. Drivers may map or stage them differently, but reporting
  // their host-side footprint prevents the plan from hiding that cost.
  uint64_t host_pinned_bytes = score_buffers;
  host_pinned_bytes = AddTo(host_pinned_bytes, feature_mask_bytes);
  host_pinned_bytes = AddTo(host_pinned_bytes, histogram_output_bytes);

  const uint64_t stat_gather_bytes = input.deterministic
      ? CheckedMultiply(
            input.num_data,
            input.constant_hessian ? kFixedPointStatSize
                                   : 2 * kFixedPointStatSize)
      : 0;
  const uint64_t in_core_bytes = CheckedAdd(feature_bytes, device_working_bytes);
  const uint64_t in_core_with_stat_gather_bytes =
      CheckedAdd(in_core_bytes, stat_gather_bytes);

  uint64_t largest_in_core_allocation_bytes = feature_bytes;
  for (const uint64_t bytes : {
           CheckedMultiply(allocated_num_data, input.score_size),
           input.deterministic
               ? CheckedMultiply(allocated_num_data, kFixedPointStatSize)
               : uint64_t{0},
           index_buffer, subhistogram_bytes, histogram_output_bytes,
           input.deterministic
               ? CheckedMultiply(input.num_data, kFixedPointStatSize)
               : uint64_t{0}}) {
    largest_in_core_allocation_bytes =
        std::max(largest_in_core_allocation_bytes, bytes);
  }

  // Reserve 25% for the OpenCL runtime, kernel code, driver allocations and
  // allocator fragmentation. This is a policy threshold, not a hard limit.
  const uint64_t usable_device_bytes =
      input.global_memory_size - input.global_memory_size / 4;
  const bool allocation_fits = largest_in_core_allocation_bytes <=
                               input.max_memory_allocation_size;
  const bool in_core_fits_hard_limits =
      allocation_fits && in_core_bytes <= input.global_memory_size;
  const bool in_core_fits_with_headroom =
      allocation_fits && in_core_bytes <= usable_device_bytes;
  const bool stat_gather_fits_with_headroom =
      in_core_fits_with_headroom &&
      in_core_with_stat_gather_bytes <= usable_device_bytes &&
      (!input.deterministic ||
       CheckedMultiply(input.num_data, kFixedPointStatSize) <=
           input.max_memory_allocation_size);

  return GPUMemoryPlan{
      feature_bytes,
      device_working_bytes,
      host_pinned_bytes,
      stat_gather_bytes,
      in_core_bytes,
      in_core_with_stat_gather_bytes,
      largest_in_core_allocation_bytes,
      usable_device_bytes,
      in_core_fits_hard_limits,
      in_core_fits_with_headroom,
      stat_gather_fits_with_headroom};
}

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_GPU_MEMORY_PLAN_HPP_
