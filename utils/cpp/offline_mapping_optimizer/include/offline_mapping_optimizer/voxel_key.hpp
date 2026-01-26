#pragma once

#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
    #define MAP_OPTIMIZER_HOSTDEV __host__ __device__
#else
    #define MAP_OPTIMIZER_HOSTDEV
#endif

struct VoxelKey {
    int x;
    int y;
    int z;

    MAP_OPTIMIZER_HOSTDEV bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const {
        return (k.x * 73856093u) ^ (k.y * 19349663u) ^ (k.z * 83492791u);
    }
};

inline VoxelKey voxel_key_from_xyz(double x, double y, double z, double inv_voxel_res) {
    return VoxelKey {
        (int)std::floor(x * inv_voxel_res),
        (int)std::floor(y * inv_voxel_res),
        (int)std::floor(z * inv_voxel_res)
    };
}

#undef MAP_OPTIMIZER_HOSTDEV
