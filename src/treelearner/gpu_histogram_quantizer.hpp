/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_GPU_HISTOGRAM_QUANTIZER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_GPU_HISTOGRAM_QUANTIZER_HPP_

#include <LightGBM/meta.h>

#include <cstdint>

namespace LightGBM {

/*!
 * \brief Fixed-point quantizer for deterministic GPU histogram accumulation.
 *
 * Pure CPU module with no OpenCL dependency. Computes a deterministic int64
 * fixed-point representation of gradient and hessian values for the current
 * boosting iteration. Conversion factors follow the XGBoost
 * CreateRoundingFactor scheme and are exact powers of two; gradient and
 * hessian always have independent factors.
 *
 * Only the scale state of the current boosting iteration is stored.
 */
class GPUHistogramQuantizer {
 public:
  /*! Largest safe absolute int64 aggregate sum (2^62). */
  static constexpr int64_t kMaxInt = INT64_C(4611686018427387904);

  /*!
   * \brief Compute bounds and conversion factors for one boosting iteration.
   *
   * When \c is_constant_hessian is true the hessian bound is derived from a
   * single value and the row count without scanning the full hessian array.
   * Invalid input (NaN/Inf, non-positive denominator, impossible scale)
   * terminates with Log::Fatal.
   */
  void ComputeState(data_size_t num_data, const score_t* gradients,
                    const score_t* hessians, bool is_constant_hessian,
                    int iteration);

  /*! Quantize the full gradient array (n == num_data of ComputeState). */
  void QuantizeGradients(const score_t* gradients, int64_t* out,
                         data_size_t n) const;

  /*! Quantize the full hessian array (n == num_data of ComputeState). */
  void QuantizeHessians(const score_t* hessians, int64_t* out,
                        data_size_t n) const;

  /*!
   * \brief Quantize the single constant hessian value.
   * \note Only valid after ComputeState with is_constant_hessian == true.
   */
  int64_t QuantizeConstHessian(double hessian_value, int iteration) const;

  /*! Inverse factor for histogram readback (dequantization). */
  double gradient_to_floating() const { return gradient_to_floating_; }

  /*! Inverse factor for histogram readback (dequantization). */
  double hessian_to_floating() const { return hessian_to_floating_; }

  /*! Host-side multiplier for gradient quantization. */
  double gradient_to_fixed() const { return gradient_to_fixed_; }

  /*! Host-side multiplier for hessian quantization. */
  double hessian_to_fixed() const { return hessian_to_fixed_; }

  double gradient_bound() const { return gradient_bound_; }

  double hessian_bound() const { return hessian_bound_; }

  double gradient_rounding() const { return gradient_rounding_; }

  double hessian_rounding() const { return hessian_rounding_; }

  bool is_constant_hessian() const { return is_constant_hessian_; }

  int iteration() const { return iteration_; }

  data_size_t num_data() const { return num_data_; }

 private:
  /*!
   * \brief Sequential double accumulation in dataset row order.
   * \return max(positive_sum, negative_sum); deterministic by construction.
   */
  static double ComputeBound(const score_t* values, data_size_t n,
                             const char* name, int iteration);

  /*!
   * \brief XGBoost CreateRoundingFactor adaptation.
   *
   * delta = bound / (1 - 2 * n * epsilon)
   * rounding = 2 ^ ceil(log2(delta))
   * to_floating = rounding / kMaxInt
   * to_fixed = 1 / to_floating
   *
   * Fatal on non-positive denominator or non-finite results.
   */
  static void ComputeFactors(double bound, data_size_t n, const char* name,
                             int iteration, double* rounding, double* to_fixed,
                             double* to_floating);

  /*! Validate one value and convert it (truncation toward zero). */
  static int64_t QuantizeValue(double value, double to_fixed,
                               const char* name, data_size_t index,
                               int iteration);

  data_size_t num_data_ = 0;
  int iteration_ = -1;
  bool is_constant_hessian_ = false;

  double gradient_bound_ = 0.0;
  double hessian_bound_ = 0.0;
  double gradient_rounding_ = 1.0;
  double hessian_rounding_ = 1.0;
  double gradient_to_fixed_ = 0.0;
  double hessian_to_fixed_ = 0.0;
  double gradient_to_floating_ = 0.0;
  double hessian_to_floating_ = 0.0;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_GPU_HISTOGRAM_QUANTIZER_HPP_
