#include <offline_mapping_optimizer/raycasting_cuda.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kMaxDdaSteps = 4096;

inline std::string cuda_err_to_string(cudaError_t err) {
    return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
}

inline std::size_t next_pow2(std::size_t v) {
    if (v <= 1)
        return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if constexpr (sizeof(std::size_t) == 8) {
        v |= v >> 32;
    }
    return v + 1;
}

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    // splitmix64
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t voxel_hash(int x, int y, int z) {
    // Mix signed ints into 64-bit.
    uint64_t ux = static_cast<uint32_t>(x);
    uint64_t uy = static_cast<uint32_t>(y);
    uint64_t uz = static_cast<uint32_t>(z);
    uint64_t h = (ux * 73856093ULL) ^ (uy * 19349663ULL) ^ (uz * 83492791ULL);
    return mix64(h);
}

__device__ __forceinline__ uint64_t pack_hi(int x, int y) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(y);
}

__device__ __forceinline__ uint64_t pack_lo(int z) {
    return static_cast<uint32_t>(z);
}

__device__ __forceinline__ int floor_to_int(float v) {
    // round down
    return __float2int_rd(v);
}

__device__ __forceinline__ void mat3_mul_vec3_rowmajor(const float* R9, const Float3& v, Float3* out) {
    // R row-major: [r00 r01 r02 r10 r11 r12 r20 r21 r22]
    out->x = R9[0] * v.x + R9[1] * v.y + R9[2] * v.z;
    out->y = R9[3] * v.x + R9[4] * v.y + R9[5] * v.z;
    out->z = R9[6] * v.x + R9[7] * v.y + R9[8] * v.z;
}

__device__ __forceinline__ float norm3(const Float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ VoxelKey voxel_key_from_xyz_f(const Float3& p, float inv_voxel_res) {
    return VoxelKey {
        floor_to_int(p.x * inv_voxel_res),
        floor_to_int(p.y * inv_voxel_res),
        floor_to_int(p.z * inv_voxel_res)
    };
}

__device__ __forceinline__ float next_boundary(int voxel_coord, int step, float voxel_res) {
    return (step > 0) ? (static_cast<float>(voxel_coord + 1) * voxel_res)
                      : (static_cast<float>(voxel_coord) * voxel_res);
}

__global__ void build_hash_table_kernel(
    const VoxelKey* keys_by_index,
    int num_keys,
    uint32_t* occupied,
    uint64_t* key_hi,
    uint64_t* key_lo,
    int* values,
    int table_size_mask
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_keys)
        return;

    const VoxelKey k = keys_by_index[i];
    const uint64_t hi = pack_hi(k.x, k.y);
    const uint64_t lo = pack_lo(k.z);

    uint64_t h = voxel_hash(k.x, k.y, k.z);
    int idx = static_cast<int>(h) & table_size_mask;

    for (int probe = 0; probe < 256; probe++) {
        // Claim slot
        if (atomicCAS(reinterpret_cast<unsigned int*>(&occupied[idx]), 0u, 1u) == 0u) {
            key_hi[idx] = hi;
            key_lo[idx] = lo;
            values[idx] = i;
            return;
        }

        // If occupied, check if it matches (shouldn't for unique keys, but be safe)
        if (key_hi[idx] == hi && key_lo[idx] == lo) {
            return;
        }

        idx = (idx + 1) & table_size_mask;
    }
}

__device__ __forceinline__ int hashmap_lookup(
    const VoxelKey& k,
    const uint32_t* occupied,
    const uint64_t* key_hi,
    const uint64_t* key_lo,
    const int* values,
    int table_size_mask
) {
    const uint64_t hi = pack_hi(k.x, k.y);
    const uint64_t lo = pack_lo(k.z);

    uint64_t h = voxel_hash(k.x, k.y, k.z);
    int idx = static_cast<int>(h) & table_size_mask;

    for (int probe = 0; probe < 256; probe++) {
        if (!occupied[idx]) {
            return -1;
        }
        if (key_hi[idx] == hi && key_lo[idx] == lo) {
            return values[idx];
        }
        idx = (idx + 1) & table_size_mask;
    }

    return -1;
}

__device__ __forceinline__ int find_frame_id(const int* frame_offsets, int num_frames, int p) {
    // Binary search in [0, num_frames) for offsets[f] <= p < offsets[f+1]
    int lo = 0;
    int hi = num_frames - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int a = frame_offsets[mid];
        int b = frame_offsets[mid + 1];
        if (p < a) {
            hi = mid - 1;
        } else if (p >= b) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    // Should never happen
    return 0;
}

__global__ void raycast_counts_kernel(
    const Float3* points_local,
    int total_points,
    const int* frame_offsets,
    int num_frames,
    const Float3* frame_origins,
    const float* frame_rotations_rowmajor,
    float voxel_res,
    float inv_voxel_res,
    const uint32_t* occupied,
    const uint64_t* key_hi,
    const uint64_t* key_lo,
    const int* values,
    int table_size_mask,
    int* counts
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total_points)
        return;

    const int frame_id = find_frame_id(frame_offsets, num_frames, p);
    const Float3 origin = frame_origins[frame_id];

    const Float3 pl = points_local[p];
    const float* R9 = frame_rotations_rowmajor + frame_id * 9;

    // delta = R * pl, end = origin + delta
    Float3 delta;
    mat3_mul_vec3_rowmajor(R9, pl, &delta);

    const float distance = norm3(delta);
    if (distance <= 1e-9f)
        return;
    if (distance < voxel_res * 0.5f)
        return;

    const float inv_dist = 1.0f / distance;
    Float3 dir {delta.x * inv_dist, delta.y * inv_dist, delta.z * inv_dist};

    VoxelKey v = voxel_key_from_xyz_f(origin, inv_voxel_res);
    Float3 end_pt {origin.x + delta.x, origin.y + delta.y, origin.z + delta.z};
    const VoxelKey vend = voxel_key_from_xyz_f(end_pt, inv_voxel_res);
    if (v == vend)
        return;

    const int step_x = (dir.x > 0.0f) ? 1 : ((dir.x < 0.0f) ? -1 : 0);
    const int step_y = (dir.y > 0.0f) ? 1 : ((dir.y < 0.0f) ? -1 : 0);
    const int step_z = (dir.z > 0.0f) ? 1 : ((dir.z < 0.0f) ? -1 : 0);

    const float inf = INFINITY;

    float t_max_x = inf, t_max_y = inf, t_max_z = inf;
    float t_delta_x = inf, t_delta_y = inf, t_delta_z = inf;

    if (step_x != 0) {
        float bx = next_boundary(v.x, step_x, voxel_res);
        t_max_x = (bx - origin.x) / dir.x;
        t_delta_x = voxel_res / fabsf(dir.x);
    }
    if (step_y != 0) {
        float by = next_boundary(v.y, step_y, voxel_res);
        t_max_y = (by - origin.y) / dir.y;
        t_delta_y = voxel_res / fabsf(dir.y);
    }
    if (step_z != 0) {
        float bz = next_boundary(v.z, step_z, voxel_res);
        t_max_z = (bz - origin.z) / dir.z;
        t_delta_z = voxel_res / fabsf(dir.z);
    }

    // Traverse, excluding end voxel. Stop after distance - voxel_res.
    const float t_stop = distance - voxel_res;

    int steps = 0;
    while (!(v == vend) && steps++ < kMaxDdaSteps) {
        float t_next;
        if (t_max_x < t_max_y) {
            if (t_max_x < t_max_z) {
                t_next = t_max_x;
                v.x += step_x;
                t_max_x += t_delta_x;
            } else {
                t_next = t_max_z;
                v.z += step_z;
                t_max_z += t_delta_z;
            }
        } else {
            if (t_max_y < t_max_z) {
                t_next = t_max_y;
                v.y += step_y;
                t_max_y += t_delta_y;
            } else {
                t_next = t_max_z;
                v.z += step_z;
                t_max_z += t_delta_z;
            }
        }

        if (t_next > t_stop)
            break;
        if (v == vend)
            break;

        int voxel_index = hashmap_lookup(v, occupied, key_hi, key_lo, values, table_size_mask);
        if (voxel_index >= 0) {
            atomicAdd(&counts[voxel_index], 1);
        }
    }
}

} // namespace

bool raycasting_cuda_compute_counts(
    const std::vector<VoxelKey>& keys_by_index,
    float voxel_res,
    const std::vector<Float3>& frame_origins,
    const std::vector<float>& frame_rotations_rowmajor,
    const std::vector<Float3>& points_local,
    const std::vector<int>& frame_offsets,
    std::vector<int>& out_counts,
    std::string* error
) {
    if (voxel_res <= 0.0f) {
        if (error)
            *error = "voxel_res must be > 0";
        return false;
    }
    if (frame_offsets.empty() || frame_offsets.size() != frame_origins.size() + 1) {
        if (error)
            *error = "frame_offsets size must be num_frames + 1";
        return false;
    }
    if (frame_rotations_rowmajor.size() != frame_origins.size() * 9) {
        if (error)
            *error = "frame_rotations_rowmajor size must be num_frames * 9";
        return false;
    }

    const int num_keys = static_cast<int>(keys_by_index.size());
    const int num_frames = static_cast<int>(frame_origins.size());
    const int total_points = static_cast<int>(points_local.size());

    out_counts.assign(num_keys, 0);

    // Build a simple open-addressing hash table.
    const std::size_t table_size = next_pow2(static_cast<std::size_t>(std::max(1024, num_keys * 2)));
    const int table_size_mask = static_cast<int>(table_size - 1);

    VoxelKey* d_keys_by_index = nullptr;
    Float3* d_points_local = nullptr;
    int* d_frame_offsets = nullptr;
    Float3* d_frame_origins = nullptr;
    float* d_frame_rot = nullptr;

    uint32_t* d_occupied = nullptr;
    uint64_t* d_key_hi = nullptr;
    uint64_t* d_key_lo = nullptr;
    int* d_values = nullptr;

    int* d_counts = nullptr;

    auto fail = [&](const char* where, cudaError_t err) {
        if (error) {
            *error = std::string(where) + ": " + cuda_err_to_string(err);
        }
    };

    cudaError_t err;

    err = cudaMalloc(&d_keys_by_index, sizeof(VoxelKey) * num_keys);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_keys_by_index)", err);
        return false;
    }
    err = cudaMalloc(&d_points_local, sizeof(Float3) * total_points);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_points_local)", err);
        cudaFree(d_keys_by_index);
        return false;
    }
    err = cudaMalloc(&d_frame_offsets, sizeof(int) * frame_offsets.size());
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_frame_offsets)", err);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        return false;
    }
    err = cudaMalloc(&d_frame_origins, sizeof(Float3) * num_frames);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_frame_origins)", err);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        cudaFree(d_frame_offsets);
        return false;
    }
    err = cudaMalloc(&d_frame_rot, sizeof(float) * frame_rotations_rowmajor.size());
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_frame_rot)", err);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        cudaFree(d_frame_offsets);
        cudaFree(d_frame_origins);
        return false;
    }

    err = cudaMalloc(&d_occupied, sizeof(uint32_t) * table_size);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_occupied)", err);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        cudaFree(d_frame_offsets);
        cudaFree(d_frame_origins);
        cudaFree(d_frame_rot);
        return false;
    }
    err = cudaMalloc(&d_key_hi, sizeof(uint64_t) * table_size);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_key_hi)", err);
        cudaFree(d_occupied);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        cudaFree(d_frame_offsets);
        cudaFree(d_frame_origins);
        cudaFree(d_frame_rot);
        return false;
    }
    err = cudaMalloc(&d_key_lo, sizeof(uint64_t) * table_size);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_key_lo)", err);
        cudaFree(d_key_hi);
        cudaFree(d_occupied);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        cudaFree(d_frame_offsets);
        cudaFree(d_frame_origins);
        cudaFree(d_frame_rot);
        return false;
    }
    err = cudaMalloc(&d_values, sizeof(int) * table_size);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_values)", err);
        cudaFree(d_key_lo);
        cudaFree(d_key_hi);
        cudaFree(d_occupied);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        cudaFree(d_frame_offsets);
        cudaFree(d_frame_origins);
        cudaFree(d_frame_rot);
        return false;
    }

    err = cudaMalloc(&d_counts, sizeof(int) * num_keys);
    if (err != cudaSuccess) {
        fail("cudaMalloc(d_counts)", err);
        cudaFree(d_values);
        cudaFree(d_key_lo);
        cudaFree(d_key_hi);
        cudaFree(d_occupied);
        cudaFree(d_keys_by_index);
        cudaFree(d_points_local);
        cudaFree(d_frame_offsets);
        cudaFree(d_frame_origins);
        cudaFree(d_frame_rot);
        return false;
    }

    err = cudaMemcpy(d_keys_by_index, keys_by_index.data(), sizeof(VoxelKey) * num_keys, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fail("cudaMemcpy(d_keys_by_index)", err);
        goto cleanup;
    }
    err = cudaMemcpy(d_points_local, points_local.data(), sizeof(Float3) * total_points, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fail("cudaMemcpy(d_points_local)", err);
        goto cleanup;
    }
    err = cudaMemcpy(d_frame_offsets, frame_offsets.data(), sizeof(int) * frame_offsets.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fail("cudaMemcpy(d_frame_offsets)", err);
        goto cleanup;
    }
    err = cudaMemcpy(d_frame_origins, frame_origins.data(), sizeof(Float3) * num_frames, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fail("cudaMemcpy(d_frame_origins)", err);
        goto cleanup;
    }
    err = cudaMemcpy(
        d_frame_rot,
        frame_rotations_rowmajor.data(),
        sizeof(float) * frame_rotations_rowmajor.size(),
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        fail("cudaMemcpy(d_frame_rot)", err);
        goto cleanup;
    }

    err = cudaMemset(d_occupied, 0, sizeof(uint32_t) * table_size);
    if (err != cudaSuccess) {
        fail("cudaMemset(d_occupied)", err);
        goto cleanup;
    }
    err = cudaMemset(d_counts, 0, sizeof(int) * num_keys);
    if (err != cudaSuccess) {
        fail("cudaMemset(d_counts)", err);
        goto cleanup;
    }

    {
        const int threads = 256;
        const int blocks = (num_keys + threads - 1) / threads;
        build_hash_table_kernel<<<blocks, threads>>>(
            d_keys_by_index,
            num_keys,
            d_occupied,
            d_key_hi,
            d_key_lo,
            d_values,
            table_size_mask
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fail("build_hash_table_kernel launch", err);
            goto cleanup;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fail("build_hash_table_kernel sync", err);
            goto cleanup;
        }
    }

    {
        const int threads = 256;
        const int blocks = (total_points + threads - 1) / threads;
        const float inv_voxel_res = 1.0f / voxel_res;
        raycast_counts_kernel<<<blocks, threads>>>(
            d_points_local,
            total_points,
            d_frame_offsets,
            num_frames,
            d_frame_origins,
            d_frame_rot,
            voxel_res,
            inv_voxel_res,
            d_occupied,
            d_key_hi,
            d_key_lo,
            d_values,
            table_size_mask,
            d_counts
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fail("raycast_counts_kernel launch", err);
            goto cleanup;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fail("raycast_counts_kernel sync", err);
            goto cleanup;
        }
    }

    err = cudaMemcpy(out_counts.data(), d_counts, sizeof(int) * num_keys, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fail("cudaMemcpy(out_counts)", err);
        goto cleanup;
    }

    // Success
    cudaFree(d_counts);
    cudaFree(d_values);
    cudaFree(d_key_lo);
    cudaFree(d_key_hi);
    cudaFree(d_occupied);
    cudaFree(d_frame_rot);
    cudaFree(d_frame_origins);
    cudaFree(d_frame_offsets);
    cudaFree(d_points_local);
    cudaFree(d_keys_by_index);
    return true;

cleanup:
    cudaFree(d_counts);
    cudaFree(d_values);
    cudaFree(d_key_lo);
    cudaFree(d_key_hi);
    cudaFree(d_occupied);
    cudaFree(d_frame_rot);
    cudaFree(d_frame_origins);
    cudaFree(d_frame_offsets);
    cudaFree(d_points_local);
    cudaFree(d_keys_by_index);
    return false;
}
