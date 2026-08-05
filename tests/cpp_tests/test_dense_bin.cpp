/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "../../src/io/dense_bin.hpp"

namespace LightGBM {

TEST(Dense4BitBin, ReleasesPushBufferAfterFinishLoad) {
  constexpr data_size_t kNumData = 1024;
  DenseBin<uint8_t, true> bin(kNumData);
  const size_t packed_bytes = (kNumData + 1) / 2;

  EXPECT_EQ(bin.MemoryUsage(), 2 * packed_bytes);
  for (data_size_t i = 0; i < kNumData; ++i) {
    bin.Push(0, i, static_cast<uint32_t>(i % 16));
  }
  bin.FinishLoad();

  EXPECT_EQ(bin.MemoryUsage(), packed_bytes);
  std::unique_ptr<BinIterator> iterator(bin.GetIterator(0, 15, 0));
  for (data_size_t i = 0; i < kNumData; ++i) {
    EXPECT_EQ(iterator->RawGet(i), static_cast<uint32_t>(i % 16));
  }
}

TEST(Dense4BitBin, ReleasesPushBufferAfterBinaryLoad) {
  constexpr data_size_t kNumData = 1024;
  const size_t packed_bytes = (kNumData + 1) / 2;
  std::vector<uint8_t> packed(packed_bytes);
  for (data_size_t i = 0; i < kNumData; i += 2) {
    packed[i / 2] = static_cast<uint8_t>(
        (i % 16) | (((i + 1) % 16) << 4));
  }

  DenseBin<uint8_t, true> bin(kNumData);
  bin.LoadFromMemory(packed.data(), {});

  EXPECT_EQ(bin.MemoryUsage(), packed_bytes);
  std::unique_ptr<BinIterator> iterator(bin.GetIterator(0, 15, 0));
  for (data_size_t i = 0; i < kNumData; ++i) {
    EXPECT_EQ(iterator->RawGet(i), static_cast<uint32_t>(i % 16));
  }
}

TEST(Dense4BitBin, ReleasesPushBufferAfterSubrowCopy) {
  constexpr data_size_t kSourceSize = 1024;
  constexpr data_size_t kSubsetSize = 256;
  std::vector<uint8_t> packed((kSourceSize + 1) / 2);
  for (data_size_t i = 0; i < kSourceSize; i += 2) {
    packed[i / 2] = static_cast<uint8_t>(
        (i % 16) | (((i + 1) % 16) << 4));
  }
  DenseBin<uint8_t, true> source(kSourceSize);
  source.LoadFromMemory(packed.data(), {});
  std::vector<data_size_t> indices(kSubsetSize);
  for (data_size_t i = 0; i < kSubsetSize; ++i) {
    indices[i] = i * 3;
  }

  DenseBin<uint8_t, true> subset(kSubsetSize);
  subset.CopySubrow(&source, indices.data(), kSubsetSize);

  EXPECT_EQ(subset.MemoryUsage(), static_cast<size_t>(kSubsetSize) / 2);
  std::unique_ptr<BinIterator> iterator(subset.GetIterator(0, 15, 0));
  for (data_size_t i = 0; i < kSubsetSize; ++i) {
    EXPECT_EQ(iterator->RawGet(i),
              static_cast<uint32_t>(indices[i] % 16));
  }
}

}  // namespace LightGBM
