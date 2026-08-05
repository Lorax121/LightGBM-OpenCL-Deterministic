/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "../../src/io/sparse_bin.hpp"

namespace LightGBM {

TEST(SparseBin, ReleasesPushBuffersAfterFinishLoad) {
  constexpr data_size_t kNumData = 4096;
  SparseBin<uint8_t> bin(kNumData);
  for (data_size_t i = 0; i < kNumData; ++i) {
    bin.Push(0, i, static_cast<uint32_t>((i % 15) + 1));
  }
  const size_t construction_bytes = bin.MemoryUsage();

  bin.FinishLoad();

  EXPECT_LT(bin.MemoryUsage(), construction_bytes);
  const size_t finished_bytes = bin.MemoryUsage();
  bin.FinishLoad();
  EXPECT_EQ(bin.MemoryUsage(), finished_bytes);
  std::unique_ptr<BinIterator> iterator(bin.GetIterator(1, 15, 0));
  for (data_size_t i = 0; i < kNumData; ++i) {
    EXPECT_EQ(iterator->RawGet(i),
              static_cast<uint32_t>((i % 15) + 1));
  }
}

TEST(SparseBin, SubrowCopyPreservesValuesAfterConstructionBuffersRelease) {
  constexpr data_size_t kSourceSize = 4096;
  constexpr data_size_t kSubsetSize = 512;
  SparseBin<uint8_t> source(kSourceSize);
  for (data_size_t i = 0; i < kSourceSize; i += 3) {
    source.Push(0, i, static_cast<uint32_t>((i % 15) + 1));
  }
  source.FinishLoad();

  std::vector<data_size_t> indices(kSubsetSize);
  for (data_size_t i = 0; i < kSubsetSize; ++i) {
    indices[i] = i * 5;
  }
  SparseBin<uint8_t> subset(kSubsetSize);
  subset.CopySubrow(&source, indices.data(), kSubsetSize);

  std::unique_ptr<BinIterator> iterator(subset.GetIterator(1, 15, 0));
  for (data_size_t i = 0; i < kSubsetSize; ++i) {
    const data_size_t source_index = indices[i];
    const uint32_t expected = source_index % 3 == 0
        ? static_cast<uint32_t>((source_index % 15) + 1)
        : 0;
    EXPECT_EQ(iterator->RawGet(i), expected);
  }
}

}  // namespace LightGBM
