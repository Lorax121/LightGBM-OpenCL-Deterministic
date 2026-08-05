/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef __OPENCL_VERSION__
R""()
#endif

#ifndef _STAT_GATHER_KERNEL_
#define _STAT_GATHER_KERNEL_

__kernel void gather_ordered_stats_i64(
    __global const long* full_gradients,
    __global const long* full_hessians,
    __global const int* indices,
    __global long* ordered_gradients,
    __global long* ordered_hessians,
    const int num_data,
    const int gather_hessians) {
  const int i = (int)get_global_id(0);
  if (i >= num_data) {
    return;
  }
  const int row = indices[i];
  ordered_gradients[i] = full_gradients[row];
  if (gather_hessians != 0) {
    ordered_hessians[i] = full_hessians[row];
  }
}

// )"" "\n#endif"
#endif
