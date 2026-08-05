/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "../../src/treelearner/gpu_memory_plan.hpp"

namespace LightGBM {

namespace {

constexpr uint64_t kGiB = uint64_t{1024} * 1024 * 1024;

GPUMemoryPlanInput DeterministicPlanInput(uint64_t rows,
                                          uint64_t feature_tuples,
                                          uint64_t global_memory,
                                          uint64_t max_allocation) {
  return GPUMemoryPlanInput{
      rows,
      feature_tuples,
      4,
      256,
      2 * sizeof(int64_t),
      1024,
      sizeof(float),
      sizeof(int32_t),
      true,
      true,
      global_memory,
      max_allocation};
}

}  // namespace

TEST(GPUMemoryPlan, InCoreFiveMillionRowWorkloadFitsFourGiBDevice) {
  const auto plan = MakeGPUMemoryPlan(
      DeterministicPlanInput(5000000, 72, 4 * kGiB, 3 * kGiB));

  EXPECT_EQ(plan.feature_bytes, uint64_t{72} * 5000000 * 4);
  EXPECT_TRUE(plan.in_core_fits_hard_limits);
  EXPECT_TRUE(plan.in_core_fits_with_headroom);
  EXPECT_TRUE(plan.stat_gather_fits_with_headroom);
}

TEST(GPUMemoryPlan, RejectsWorkloadBeyondGlobalMemory) {
  const auto plan = MakeGPUMemoryPlan(
      DeterministicPlanInput(30000000, 75, 8 * kGiB, 6745 * kGiB / 1024));

  EXPECT_EQ(plan.feature_bytes, uint64_t{75} * 30000000 * 4);
  EXPECT_FALSE(plan.in_core_fits_hard_limits);
  EXPECT_FALSE(plan.in_core_fits_with_headroom);
}

TEST(GPUMemoryPlan, RejectsWorkloadBeyondPerAllocationLimit) {
  const auto plan = MakeGPUMemoryPlan(
      DeterministicPlanInput(30000000, 75, 16 * kGiB, 2 * kGiB));

  EXPECT_LT(plan.in_core_bytes, plan.usable_device_bytes);
  EXPECT_FALSE(plan.in_core_fits_hard_limits);
}

TEST(GPUMemoryPlan, RejectsArithmeticOverflow) {
  auto input = DeterministicPlanInput(
      std::numeric_limits<uint64_t>::max(), 75, 16 * kGiB, 2 * kGiB);
  EXPECT_THROW(MakeGPUMemoryPlan(input), std::overflow_error);
}

}  // namespace LightGBM
