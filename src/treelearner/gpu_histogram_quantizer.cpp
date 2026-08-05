/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "gpu_histogram_quantizer.hpp"

#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace LightGBM {

namespace {

// 2^62 as a double: the largest safe aggregate sum.
constexpr double kMaxIntDouble = 4611686018427387904.0;

}  // namespace

void GPUHistogramQuantizer::ComputeState(data_size_t num_data,
                                         const score_t* gradients,
                                         const score_t* hessians,
                                         bool is_constant_hessian,
                                         int iteration) {
  num_data_ = num_data;
  iteration_ = iteration;
  is_constant_hessian_ = is_constant_hessian;

  gradient_bound_ = ComputeBound(gradients, num_data, "gradient", iteration);

  if (is_constant_hessian) {
    if (hessians == nullptr) {
      Log::Fatal(
          "Constant hessian requires a non-null hessian array (iteration %d).",
          iteration);
    }
    const double hessian_value = static_cast<double>(hessians[0]);
    if (!std::isfinite(hessian_value)) {
      Log::Fatal("hessian value is not finite at index 0 (iteration %d): %g",
                 iteration, hessian_value);
    }
    hessian_bound_ = std::fabs(hessian_value) * static_cast<double>(num_data);
    if (!std::isfinite(hessian_bound_)) {
      Log::Fatal(
          "hessian bound overflows double for constant hessian %g with %d rows "
          "(iteration %d).",
          hessian_value, num_data, iteration);
    }
  } else {
    hessian_bound_ = ComputeBound(hessians, num_data, "hessian", iteration);
  }

  ComputeFactors(gradient_bound_, num_data, "gradient", iteration,
                 &gradient_rounding_, &gradient_to_fixed_,
                 &gradient_to_floating_);
  ComputeFactors(hessian_bound_, num_data, "hessian", iteration,
                 &hessian_rounding_, &hessian_to_fixed_,
                 &hessian_to_floating_);
}

double GPUHistogramQuantizer::ComputeBound(const score_t* values,
                                           data_size_t n, const char* name,
                                           int iteration) {
  double positive_sum = 0.0;
  double negative_sum = 0.0;
  for (data_size_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(values[i]);
    if (!std::isfinite(v)) {
      Log::Fatal("%s value is not finite at index %d (iteration %d): %g", name,
                 i, iteration, v);
    }
    positive_sum += std::fmax(v, 0.0);
    negative_sum += std::fmax(-v, 0.0);
  }
  return std::fmax(positive_sum, negative_sum);
}

void GPUHistogramQuantizer::ComputeFactors(double bound, data_size_t n,
                                           const char* name, int iteration,
                                           double* rounding, double* to_fixed,
                                           double* to_floating) {
  if (bound == 0.0) {
    // All quantized values are zero; factors must still have safe defined
    // values so that dequantization is well-defined.
    *rounding = 1.0;
    *to_floating = 1.0 / kMaxIntDouble;
    *to_fixed = kMaxIntDouble;
    return;
  }

  const double epsilon = std::numeric_limits<double>::epsilon();
  const double denominator = 1.0 - 2.0 * static_cast<double>(n) * epsilon;
  if (!(denominator > 0.0)) {
    Log::Fatal("%s scale denominator is not positive for %d rows (iteration %d).",
               name, n, iteration);
  }
  const double delta = bound / denominator;
  if (!std::isfinite(delta) || delta <= 0.0) {
    Log::Fatal("%s delta is not a positive finite value (iteration %d): %g",
               name, iteration, delta);
  }
  const int exponent = static_cast<int>(std::ceil(std::log2(delta)));
  *rounding = std::ldexp(1.0, exponent);
  if (!std::isfinite(*rounding) || *rounding <= 0.0) {
    Log::Fatal(
        "%s rounding is not a positive finite power of two (iteration %d): %g",
        name, iteration, *rounding);
  }
  *to_floating = *rounding / kMaxIntDouble;
  if (!std::isfinite(*to_floating) || *to_floating <= 0.0) {
    Log::Fatal(
        "%s to_floating is not a positive finite value (iteration %d): %g",
        name, iteration, *to_floating);
  }
  *to_fixed = 1.0 / *to_floating;
  if (!std::isfinite(*to_fixed) || *to_fixed <= 0.0) {
    Log::Fatal("%s to_fixed is not a positive finite value (iteration %d): %g",
               name, iteration, *to_fixed);
  }
}

int64_t GPUHistogramQuantizer::QuantizeValue(double value, double to_fixed,
                                             const char* name,
                                             data_size_t index,
                                             int iteration) {
  if (!std::isfinite(value)) {
    Log::Fatal("%s value is not finite at index %d (iteration %d): %g", name,
               index, iteration, value);
  }
  const double product = value * to_fixed;
  if (!std::isfinite(product) ||
      std::fabs(product) > static_cast<double>(kMaxInt)) {
    Log::Fatal(
        "%s value * to_fixed is out of the safe int64 range at index %d "
        "(iteration %d): %g",
        name, index, iteration, value);
  }
  // Truncation toward zero.
  return static_cast<int64_t>(product);
}

void GPUHistogramQuantizer::QuantizeGradients(const score_t* gradients,
                                              int64_t* out,
                                              data_size_t n) const {
  // Element-wise and order-independent: parallelization cannot change the
  // quantized values, so the deterministic contract is preserved.
  OMP_INIT_EX();
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < n; ++i) {
    OMP_LOOP_EX_BEGIN();
    out[i] = QuantizeValue(static_cast<double>(gradients[i]),
                           gradient_to_fixed_, "gradient", i, iteration_);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
}

void GPUHistogramQuantizer::QuantizeHessians(const score_t* hessians,
                                             int64_t* out,
                                             data_size_t n) const {
  // Element-wise and order-independent: parallelization cannot change the
  // quantized values, so the deterministic contract is preserved.
  OMP_INIT_EX();
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < n; ++i) {
    OMP_LOOP_EX_BEGIN();
    const double v = static_cast<double>(hessians[i]);
    int64_t q = QuantizeValue(v, hessian_to_fixed_, "hessian", i, iteration_);
    if (v > 0.0 && q == 0) {
      q = 1;  // preserve positive curvature.
    }
    out[i] = q;
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
}

int64_t GPUHistogramQuantizer::QuantizeConstHessian(double hessian_value,
                                                    int iteration) const {
  int64_t q =
      QuantizeValue(hessian_value, hessian_to_fixed_, "hessian", 0, iteration);
  if (hessian_value > 0.0 && q == 0) {
    q = 1;
  }
  return q;
}

}  // namespace LightGBM
