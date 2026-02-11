#pragma once

#include "rasterization_config.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <cstdint>

namespace faster_gs::rasterization {

    inline __host__ int extract_end_bit(uint n) {
        int leading_zeros = 0;
        if ((n & 0xffff0000) == 0) { leading_zeros += 16; n <<= 16; }
        if ((n & 0xff000000) == 0) { leading_zeros += 8; n <<= 8; }
        if ((n & 0xf0000000) == 0) { leading_zeros += 4; n <<= 4; }
        if ((n & 0xc0000000) == 0) { leading_zeros += 2; n <<= 2; }
        if ((n & 0x80000000) == 0) { leading_zeros += 1; }
        return 32 - leading_zeros;
    }

    struct mat3x3 {
        float m11, m12, m13;
        float m21, m22, m23;
        float m31, m32, m33;
    };

    struct __align__(8) mat3x3_triu {
        float m11, m12, m13, m22, m23, m33;
    };

    template <typename T>
    static void obtain(char*& blob, T*& ptr, size_t count) {
        constexpr size_t alignment = 128;
        size_t offset = (reinterpret_cast<size_t>(blob) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        blob = reinterpret_cast<char*>(ptr + count);
    }

    template <typename T, typename... Args>
    size_t required(size_t P, Args... args) {
        char* size = nullptr;
        T::from_blob(size, P, args...);
        return ((size_t)size);
    }

    struct PrimitiveBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        uint* n_touched_tiles;
        uint* offset;
        ushort4* screen_bounds;
        float2* mean2d;
        float4* conic_opacity;
        float3* color;
        float* depth;

        static PrimitiveBuffers from_blob(char*& blob, int n_primitives) {
            PrimitiveBuffers buffers;
            obtain(blob, buffers.n_touched_tiles, n_primitives);
            obtain(blob, buffers.offset, n_primitives);
            obtain(blob, buffers.screen_bounds, n_primitives);
            obtain(blob, buffers.mean2d, n_primitives);
            obtain(blob, buffers.conic_opacity, n_primitives);
            obtain(blob, buffers.color, n_primitives);
            obtain(blob, buffers.depth, n_primitives);
            cub::DeviceScan::InclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.n_touched_tiles, buffers.offset,
                n_primitives
            );
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size);
            return buffers;
        }
    };

    struct InstanceBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<uint64_t> keys;
        cub::DoubleBuffer<uint> primitive_indices;

        static InstanceBuffers from_blob(char*& blob, int n_instances, int end_bit) {
            InstanceBuffers buffers;
            uint64_t* keys_current;
            obtain(blob, keys_current, n_instances);
            uint64_t* keys_alternate;
            obtain(blob, keys_alternate, n_instances);
            buffers.keys = cub::DoubleBuffer<uint64_t>(keys_current, keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_instances);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_instances);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            cub::DeviceRadixSort::SortPairs(
                nullptr, buffers.cub_workspace_size,
                buffers.keys, buffers.primitive_indices,
                n_instances,
                0, end_bit
            );
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size);
            return buffers;
        }
    };

    struct TileBuffers {
        uint2* instance_ranges;
        float* final_transmittances;

        static TileBuffers from_blob(char*& blob, int n_tiles) {
            TileBuffers buffers;
            obtain(blob, buffers.instance_ranges, n_tiles);
            obtain(blob, buffers.final_transmittances, n_tiles * config::block_size_blend);
            return buffers;
        }
    };

}
