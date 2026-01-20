#pragma once

#include <string>
#include <vector>

#include <map_optimizer/voxel_key.hpp>

// Minimal POD (avoid pulling CUDA headers into non-CUDA translation units).
struct Float3 {
  float x;
  float y;
  float z;
};

// Computes pass-through counts for each occupied voxel (indexed by keys_by_index).
// - keys_by_index: size N, where keys_by_index[i] is the voxel key for voxel index i
// - frame_origins: size F, origin (t) per frame in world coordinates
// - frame_rotations_rowmajor: size F*9, row-major world_R_local per frame
// - points_local: concatenated local points for all frames
// - frame_offsets: size F+1, prefix sum offsets into points_local
// Output:
// - out_counts: resized to N and filled
// Returns false on any CUDA error; error (if non-null) will contain a message.
bool raycasting_cuda_compute_counts(const std::vector<VoxelKey>& keys_by_index,
                                   float voxel_res,
                                   const std::vector<Float3>& frame_origins,
                                   const std::vector<float>& frame_rotations_rowmajor,
                                   const std::vector<Float3>& points_local,
                                   const std::vector<int>& frame_offsets,
                                   std::vector<int>& out_counts,
                                   std::string* error);
