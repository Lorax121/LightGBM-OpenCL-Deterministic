/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <gtest/gtest.h>

#include "../../src/treelearner/gpu_histogram_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using LightGBM::GPUHistogramQuantizer;
using LightGBM::data_size_t;
using LightGBM::score_t;

constexpr int64_t kTwoTo62 = INT64_C(4611686018427387904);

std::vector<score_t> ToScores(const std::vector<double>& values) {
  std::vector<score_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = static_cast<score_t>(values[i]);
  }
  return out;
}

GPUHistogramQuantizer MakeState(const std::vector<double>& grads,
                                const std::vector<double>& hess,
                                bool const_hessian = false,
                                int iteration = 0) {
  const auto g = ToScores(grads);
  const auto h = ToScores(hess);
  GPUHistogramQuantizer q;
  q.ComputeState(static_cast<data_size_t>(g.size()), g.data(), h.data(),
                 const_hessian, iteration);
  return q;
}

std::vector<int64_t> QuantizeGrads(const GPUHistogramQuantizer& q,
                                   const std::vector<double>& grads) {
  const auto g = ToScores(grads);
  std::vector<int64_t> out(g.size());
  q.QuantizeGradients(g.data(), out.data(), static_cast<data_size_t>(g.size()));
  return out;
}

std::vector<int64_t> QuantizeHess(const GPUHistogramQuantizer& q,
                                  const std::vector<double>& hess) {
  const auto h = ToScores(hess);
  std::vector<int64_t> out(h.size());
  q.QuantizeHessians(h.data(), out.data(), static_cast<data_size_t>(h.size()));
  return out;
}

// Log::Fatal throws std::runtime_error; verify the message contains needle.
void ExpectFatalContains(const std::function<void()>& fn,
                         const std::string& needle) {
  try {
    fn();
    ADD_FAILURE() << "expected Log::Fatal (std::runtime_error) containing: "
                  << needle;
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
        << "actual message: " << e.what();
  }
}

bool IsPowerOfTwo(double x) {
  if (x <= 0.0 || !std::isfinite(x)) {
    return false;
  }
  const double log2_x = std::log2(x);
  return std::fabs(log2_x - std::round(log2_x)) < 1e-12;
}

}  // namespace

TEST(GPUHistogramQuantizerTest, AllZeroValues) {
  const std::vector<double> zeros(100, 0.0);
  const auto q = MakeState(zeros, zeros);
  // Safe defined factors for a zero bound.
  EXPECT_DOUBLE_EQ(q.gradient_to_fixed(), static_cast<double>(kTwoTo62));
  EXPECT_DOUBLE_EQ(q.hessian_to_fixed(), static_cast<double>(kTwoTo62));
  EXPECT_DOUBLE_EQ(q.gradient_to_floating(), 1.0 / static_cast<double>(kTwoTo62));
  EXPECT_DOUBLE_EQ(q.hessian_to_floating(), 1.0 / static_cast<double>(kTwoTo62));
  EXPECT_DOUBLE_EQ(q.gradient_rounding(), 1.0);
  EXPECT_DOUBLE_EQ(q.hessian_rounding(), 1.0);
  EXPECT_DOUBLE_EQ(q.gradient_bound(), 0.0);
  EXPECT_DOUBLE_EQ(q.hessian_bound(), 0.0);

  const auto qg = QuantizeGrads(q, zeros);
  const auto qh = QuantizeHess(q, zeros);
  for (int64_t v : qg) {
    EXPECT_EQ(v, 0);
  }
  for (int64_t v : qh) {
    EXPECT_EQ(v, 0);
  }
}

TEST(GPUHistogramQuantizerTest, OnlyPositiveValues) {
  const std::vector<double> grads = {0.5, 1.0, 2.0, 4.0};
  const std::vector<double> hess = {1.0, 1.0, 1.0, 1.0};
  const auto q = MakeState(grads, hess);
  // positive_sum == 7.5, negative_sum == 0.
  EXPECT_DOUBLE_EQ(q.gradient_bound(), 7.5);
  const auto qg = QuantizeGrads(q, grads);
  for (int64_t v : qg) {
    EXPECT_GT(v, 0);
  }
  const auto qh = QuantizeHess(q, hess);
  for (int64_t v : qh) {
    EXPECT_GT(v, 0);
  }
}

TEST(GPUHistogramQuantizerTest, MixedPositiveNegative) {
  const std::vector<double> grads = {-3.0, 1.0, -2.0, 4.0};
  const std::vector<double> hess = {1.0, 1.0, 1.0, 1.0};
  const auto q = MakeState(grads, hess);
  EXPECT_DOUBLE_EQ(q.gradient_bound(), 5.0);  // max(5, 5)
  const auto qg = QuantizeGrads(q, grads);
  // Sign must be preserved for non-zero quantized values.
  for (size_t i = 0; i < grads.size(); ++i) {
    if (grads[i] > 0.0) {
      EXPECT_GT(qg[i], 0);
    } else if (grads[i] < 0.0) {
      EXPECT_LT(qg[i], 0);
    }
  }
}

TEST(GPUHistogramQuantizerTest, AsymmetricPositiveNegativeSums) {
  // positive_sum = 7, negative_sum = 3 -> bound = 7.
  const std::vector<double> grads = {4.0, 2.0, 1.0, -3.0};
  const std::vector<double> hess = {1.0, 1.0, 1.0, 1.0};
  const auto q = MakeState(grads, hess);
  EXPECT_DOUBLE_EQ(q.gradient_bound(), 7.0);
  // And the reverse asymmetry.
  const std::vector<double> grads2 = {-4.0, -2.0, -1.0, 3.0};
  const auto q2 = MakeState(grads2, hess);
  EXPECT_DOUBLE_EQ(q2.gradient_bound(), 7.0);
}

TEST(GPUHistogramQuantizerTest, ConstantHessian) {
  const std::vector<double> grads(64, 1.0);
  const std::vector<double> hess(64, 2.0);
  const auto q = MakeState(grads, hess, /*const_hessian=*/true);
  EXPECT_TRUE(q.is_constant_hessian());
  // bound = |hessian| * n without scanning the array.
  EXPECT_DOUBLE_EQ(q.hessian_bound(), 128.0);
  // The single quantized constant must match the full-array quantization.
  const auto qh = QuantizeHess(q, hess);
  const int64_t single = q.QuantizeConstHessian(2.0, q.iteration());
  for (int64_t v : qh) {
    EXPECT_EQ(v, single);
  }
  EXPECT_GT(single, 0);
}

TEST(GPUHistogramQuantizerTest, VerySmallNormalFloatValues) {
  // Small but normal float values; must quantize without overflow and
  // dequantize back close to the original sign/magnitude. The quantizer
  // sees score_t values, so the reference is the float-rounded value.
  const std::vector<double> grads = {1e-30, -2e-30, 3e-30, -1e-25};
  const std::vector<double> hess = {1e-30, 1e-30, 1e-30, 1e-30};
  const auto q = MakeState(grads, hess);
  const auto g = ToScores(grads);
  const auto qg = QuantizeGrads(q, grads);
  for (size_t i = 0; i < grads.size(); ++i) {
    const double deq = static_cast<double>(qg[i]) * q.gradient_to_floating();
    const double ref = static_cast<double>(g[i]);
    EXPECT_LE(std::fabs(deq - ref), q.gradient_to_floating())
        << "index " << i;
  }
}

TEST(GPUHistogramQuantizerTest, VeryLargeFiniteFloatValues) {
  // FLT_MAX as input: bound is huge, factors stay finite, products stay in
  // the safe int64 range.
  const float flt_max = std::numeric_limits<float>::max();
  const std::vector<double> grads = {flt_max, -flt_max, flt_max, flt_max};
  const std::vector<double> hess = {1.0, 1.0, 1.0, 1.0};
  const auto q = MakeState(grads, hess);
  const auto qg = QuantizeGrads(q, grads);
  for (size_t i = 0; i < grads.size(); ++i) {
    EXPECT_NE(qg[i], 0);
    EXPECT_LE(std::llabs(qg[i]), kTwoTo62);
  }
}

TEST(GPUHistogramQuantizerTest, TinyPositiveHessianBecomesOne) {
  // A large hessian dominates the bound; a tiny positive hessian would
  // truncate to zero and must be forced to 1 to preserve curvature.
  const std::vector<double> grads = {1.0, 1.0, 1.0, 1.0};
  const std::vector<double> hess = {1000.0, 1e-20, 1e-20, 1e-20};
  const auto q = MakeState(grads, hess);
  const auto qh = QuantizeHess(q, hess);
  // 1000 * to_fixed must stay well above 1, tiny ones become 1.
  EXPECT_GT(qh[0], 1);
  EXPECT_EQ(qh[1], 1);
  EXPECT_EQ(qh[2], 1);
  EXPECT_EQ(qh[3], 1);
}

TEST(GPUHistogramQuantizerTest, NegativeTinyHessianNotForcedToOne) {
  // Negative hessian must NOT be forced to 1.
  const std::vector<double> grads = {1.0, 1.0, 1.0, 1.0};
  const std::vector<double> hess = {1000.0, -1e-20, -1e-20, -1e-20};
  const auto q = MakeState(grads, hess);
  const auto qh = QuantizeHess(q, hess);
  EXPECT_GT(qh[0], 1);
  EXPECT_EQ(qh[1], 0);
  EXPECT_EQ(qh[2], 0);
  EXPECT_EQ(qh[3], 0);
}

TEST(GPUHistogramQuantizerTest, PowerOfTwoBoundary) {
  // Values on exact powers of two must quantize to exact powers of two.
  // bound = 3.5 -> delta ~3.5 -> rounding = 4 -> to_fixed = 2^62 / 4 = 2^60.
  const std::vector<double> grads = {0.5, 1.0, 2.0};
  const std::vector<double> hess = {1.0, 1.0, 1.0};
  const auto q = MakeState(grads, hess);
  EXPECT_DOUBLE_EQ(q.gradient_rounding(), 4.0);
  EXPECT_DOUBLE_EQ(q.gradient_to_fixed(), static_cast<double>(INT64_C(1) << 60));
  const auto qg = QuantizeGrads(q, grads);
  EXPECT_EQ(qg[0], INT64_C(1) << 59);
  EXPECT_EQ(qg[1], INT64_C(1) << 60);
  EXPECT_EQ(qg[2], INT64_C(1) << 61);
}

TEST(GPUHistogramQuantizerTest, NaNThrowsWithContext) {
  const std::vector<double> grads = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0};
  const std::vector<double> hess = {1.0, 1.0, 1.0, 1.0};
  ExpectFatalContains(
      [&]() { MakeState(grads, hess, false, 7); }, "gradient");
  // Message must contain the index of the first problem row and iteration.
  try {
    MakeState(grads, hess, false, 7);
    ADD_FAILURE() << "expected fatal";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("index 2"), std::string::npos)
        << "actual: " << e.what();
    EXPECT_NE(std::string(e.what()).find("iteration 7"), std::string::npos)
        << "actual: " << e.what();
  }

  // NaN in hessian.
  const std::vector<double> grads_ok = {1.0, 1.0, 1.0, 1.0};
  const std::vector<double> hess_nan = {1.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0};
  ExpectFatalContains([&]() { MakeState(grads_ok, hess_nan); }, "hessian");
}

TEST(GPUHistogramQuantizerTest, InfThrows) {
  const std::vector<double> hess = {1.0, 1.0, 1.0, 1.0};
  const std::vector<double> pos_inf = {1.0, std::numeric_limits<double>::infinity(), 1.0, 1.0};
  const std::vector<double> neg_inf = {1.0, -std::numeric_limits<double>::infinity(), 1.0, 1.0};
  ExpectFatalContains([&]() { MakeState(pos_inf, hess); }, "gradient");
  ExpectFatalContains([&]() { MakeState(neg_inf, hess); }, "gradient");
  // Constant hessian NaN also fatal.
  const std::vector<double> grads = {1.0, 1.0, 1.0, 1.0};
  const std::vector<double> hess_nan = {std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0, 1.0};
  ExpectFatalContains([&]() { MakeState(grads, hess_nan, true); }, "hessian");
}

TEST(GPUHistogramQuantizerTest, DenominatorAndScaleValidation) {
  // The denominator 1 - 2*n*epsilon is positive for the maximal int32 row
  // count (defensive requirement of the plan).
  constexpr double epsilon = std::numeric_limits<double>::epsilon();
  const double max_n = static_cast<double>(std::numeric_limits<data_size_t>::max());
  EXPECT_GT(1.0 - 2.0 * max_n * epsilon, 0.0);

  // Factors for a huge (but finite) bound must remain finite powers of two.
  const std::vector<double> grads = {1e38, -1e38, 1e38};
  const std::vector<double> hess = {1.0, 1.0, 1.0};
  const auto q = MakeState(grads, hess);
  EXPECT_TRUE(IsPowerOfTwo(q.gradient_rounding()));
  EXPECT_TRUE(IsPowerOfTwo(q.gradient_to_fixed()));
  EXPECT_TRUE(IsPowerOfTwo(q.gradient_to_floating()));
  EXPECT_TRUE(IsPowerOfTwo(q.hessian_rounding()));
  EXPECT_TRUE(IsPowerOfTwo(q.hessian_to_fixed()));
  EXPECT_TRUE(IsPowerOfTwo(q.hessian_to_floating()));
}

TEST(GPUHistogramQuantizerTest, DeterministicEquality) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const std::vector<double> grads(1000, 0.0);
  std::vector<double> hess(1000, 0.0);
  for (size_t i = 0; i < hess.size(); ++i) {
    hess[i] = std::fabs(dist(rng)) + 0.001;
  }
  const auto q = MakeState(grads, hess);
  const auto qh1 = QuantizeHess(q, hess);
  const auto qh2 = QuantizeHess(q, hess);
  EXPECT_EQ(qh1, qh2);

  // Repeated state construction gives identical factors.
  const auto q2 = MakeState(grads, hess);
  EXPECT_DOUBLE_EQ(q.gradient_bound(), q2.gradient_bound());
  EXPECT_DOUBLE_EQ(q.hessian_bound(), q2.hessian_bound());
  EXPECT_DOUBLE_EQ(q.gradient_to_fixed(), q2.gradient_to_fixed());
  EXPECT_DOUBLE_EQ(q.hessian_to_fixed(), q2.hessian_to_fixed());
}

TEST(GPUHistogramQuantizerTest, SubsetSumWithinSafeRange) {
  // Property: full positive and negative sums of quantized values never
  // exceed the reserved headroom (2^62).
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  const size_t n = 4096;
  std::vector<double> grads(n);
  std::vector<double> hess(n);
  for (size_t i = 0; i < n; ++i) {
    grads[i] = dist(rng);
    hess[i] = std::fabs(dist(rng)) + 1e-6;
  }
  const auto q = MakeState(grads, hess);
  const auto qg = QuantizeGrads(q, grads);
  const auto qh = QuantizeHess(q, hess);

  int64_t pos_sum = 0;
  int64_t neg_sum = 0;
  for (size_t i = 0; i < n; ++i) {
    if (qg[i] > 0) {
      pos_sum += qg[i];
    } else {
      neg_sum += qg[i];
    }
  }
  EXPECT_LE(pos_sum, kTwoTo62);
  EXPECT_GE(neg_sum, -kTwoTo62);
  EXPECT_LE(std::llabs(pos_sum + neg_sum), kTwoTo62);

  // Sum of any contiguous subset stays in the safe range.
  const size_t half = n / 2;
  int64_t subset = 0;
  for (size_t i = half - 100; i < half + 100; ++i) {
    subset += qh[i];
  }
  EXPECT_LE(std::llabs(subset), kTwoTo62);

  int64_t pos_h = 0;
  int64_t neg_h = 0;
  for (int64_t v : qh) {
    if (v > 0) {
      pos_h += v;
    } else {
      neg_h += v;
    }
  }
  EXPECT_LE(pos_h, kTwoTo62);
  EXPECT_GE(neg_h, -kTwoTo62);
}

TEST(GPUHistogramQuantizerTest, HeadroomForMaxDataSize) {
  // Reserve between 2^62 and INT64_MAX must cover adding one unit per row
  // (the tiny-positive-hessian fix) for the maximal data_size_t.
  const int64_t max_rows = static_cast<int64_t>(std::numeric_limits<data_size_t>::max());
  const int64_t worst_case = kTwoTo62 + max_rows;
  EXPECT_GT(std::numeric_limits<int64_t>::max(), worst_case);
}

TEST(GPUHistogramQuantizerTest, DequantizationPreservesSignAndPrecision) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  const size_t n = 2048;
  std::vector<double> grads(n);
  std::vector<double> hess(n);
  for (size_t i = 0; i < n; ++i) {
    grads[i] = dist(rng);
    hess[i] = std::fabs(dist(rng)) + 0.01;
  }
  const auto q = MakeState(grads, hess);
  const auto g = ToScores(grads);
  const auto qg = QuantizeGrads(q, grads);

  for (size_t i = 0; i < n; ++i) {
    const double deq = static_cast<double>(qg[i]) * q.gradient_to_floating();
    // Sign is preserved.
    if (grads[i] > 0.0) {
      EXPECT_GT(qg[i], 0) << "index " << i;
    } else if (grads[i] < 0.0) {
      EXPECT_LT(qg[i], 0) << "index " << i;
    }
    // One-unit quantization error relative to the score_t value the
    // quantizer actually sees.
    const double ref = static_cast<double>(g[i]);
    EXPECT_LE(std::fabs(deq - ref), q.gradient_to_floating())
        << "index " << i;
  }
}

TEST(GPUHistogramQuantizerTest, DifferentScalesPerGradHess) {
  // Gradient and hessian must have independent factors when bounds differ.
  const std::vector<double> grads = {100.0, 100.0, 100.0};
  const std::vector<double> hess = {1.0, 1.0, 1.0};
  const auto q = MakeState(grads, hess);
  EXPECT_DOUBLE_EQ(q.gradient_bound(), 300.0);
  EXPECT_DOUBLE_EQ(q.hessian_bound(), 3.0);
  EXPECT_NE(q.gradient_to_fixed(), q.hessian_to_fixed());
}

TEST(GPUHistogramQuantizerTest, StateGetters) {
  const std::vector<double> grads = {1.0, -1.0};
  const std::vector<double> hess = {0.5, 0.5};
  const auto q = MakeState(grads, hess, false, 42);
  EXPECT_EQ(q.iteration(), 42);
  EXPECT_EQ(q.num_data(), 2);
  EXPECT_FALSE(q.is_constant_hessian());
}
