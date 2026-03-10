// metal_ops.mm — Metal GPU compute for attention + RMSNorm
// Uses runtime shader compilation (no separate .metallib needed)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_ops.h"
#include <cstdio>
#include <cstring>
#include <mutex>

namespace ane_lm {

// ============ Metal shader source (compiled at runtime) ============
static const char* g_shader_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

// GQA Attention kernel: one thread-group per head
// Each thread in the group processes a subset of KV positions
kernel void gqa_attention_kernel(
    device const float* q          [[buffer(0)]],   // [n_heads * head_dim]
    device const half*  k_cache    [[buffer(1)]],   // [capacity * n_kv_heads * head_dim]
    device const half*  v_cache    [[buffer(2)]],   // [capacity * n_kv_heads * head_dim]
    device float*       output     [[buffer(3)]],   // [n_heads * head_dim]
    constant int&       n_heads    [[buffer(4)]],
    constant int&       n_kv_heads [[buffer(5)]],
    constant int&       head_dim   [[buffer(6)]],
    constant int&       q_stride   [[buffer(7)]],
    constant int&       cache_start[[buffer(8)]],
    constant int&       cache_len  [[buffer(9)]],
    constant int&       capacity   [[buffer(10)]],
    constant float&     scale      [[buffer(11)]],
    uint  gid   [[threadgroup_position_in_grid]],     // head index
    uint  tid   [[thread_position_in_threadgroup]],
    uint  tcount[[threads_per_threadgroup]])
{
    int h = gid;
    if (h >= n_heads) return;

    int groups = n_heads / n_kv_heads;
    int kv_h = h / groups;
    int kv_step = n_kv_heads * head_dim;

    // Q vector for this head
    device const float* qh = q + h * q_stride;
    device float* oh = output + h * head_dim;

    // Phase 1: compute attention scores (parallel across KV positions)
    // Each thread handles a subset of positions
    threadgroup float shared_scores[2048];  // max cache_len
    threadgroup float shared_max[1];
    threadgroup float shared_sum[1];

    if (tid == 0) { shared_max[0] = -INFINITY; shared_sum[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute scores for this thread's positions
    for (int t = (int)tid; t < cache_len; t += (int)tcount) {
        int pos = cache_start + t;
        if (pos >= capacity) pos -= capacity;  // ring buffer wrap

        device const half* kh = k_cache + pos * kv_step + kv_h * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qh[d] * float(kh[d]);
        }
        shared_scores[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: find max (reduction)
    float local_max = -INFINITY;
    for (int t = (int)tid; t < cache_len; t += (int)tcount) {
        local_max = max(local_max, shared_scores[t]);
    }
    // Atomic max (simple approach for correctness)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (int t = 0; t < cache_len; t++) m = max(m, shared_scores[t]);
        shared_max[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_val = shared_max[0];

    // Phase 3: exp and sum
    for (int t = (int)tid; t < cache_len; t += (int)tcount) {
        shared_scores[t] = exp(shared_scores[t] - max_val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < cache_len; t++) s += shared_scores[t];
        shared_sum[0] = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = shared_sum[0];
    for (int t = (int)tid; t < cache_len; t += (int)tcount) {
        shared_scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: weighted sum of V
    for (int d = (int)tid; d < head_dim; d += (int)tcount) {
        float acc = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            int pos = cache_start + t;
            if (pos >= capacity) pos -= capacity;
            acc += shared_scores[t] * float(v_cache[pos * kv_step + kv_h * head_dim + d]);
        }
        oh[d] = acc;
    }
}

// RMSNorm kernel
kernel void rmsnorm_kernel(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant int&       dim    [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tcount[[threads_per_threadgroup]])
{
    // Parallel reduction for sum of squares
    threadgroup float shared_ss[256];
    float local_ss = 0.0f;
    for (int i = (int)tid; i < dim; i += (int)tcount) {
        float v = input[i];
        local_ss += v * v;
    }
    shared_ss[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tcount / 2; s > 0; s >>= 1) {
        if (tid < s) shared_ss[tid] += shared_ss[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float ss = 1.0f / sqrt(shared_ss[0] / float(dim) + eps);

    // Apply normalization + weight
    for (int i = (int)tid; i < dim; i += (int)tcount) {
        output[i] = input[i] * ss * weight[i];
    }
}
)METAL";

// ============ Metal state ============
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_attn_pipeline = nil;
static id<MTLComputePipelineState> g_rmsnorm_pipeline = nil;
static bool g_metal_ok = false;
static std::once_flag g_metal_once;

bool metal_init() {
    std::call_once(g_metal_once, []() {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "Metal: no device available\n");
            return;
        }
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "Metal: failed to create command queue\n");
            return;
        }

        // Compile shaders at runtime
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:g_shader_source];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;

        id<MTLLibrary> library = [g_device newLibraryWithSource:source options:opts error:&error];
        if (!library) {
            fprintf(stderr, "Metal: shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        id<MTLFunction> attn_fn = [library newFunctionWithName:@"gqa_attention_kernel"];
        id<MTLFunction> rmsnorm_fn = [library newFunctionWithName:@"rmsnorm_kernel"];

        if (attn_fn) {
            g_attn_pipeline = [g_device newComputePipelineStateWithFunction:attn_fn error:&error];
            if (!g_attn_pipeline) {
                fprintf(stderr, "Metal: attn pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        if (rmsnorm_fn) {
            g_rmsnorm_pipeline = [g_device newComputePipelineStateWithFunction:rmsnorm_fn error:&error];
            if (!g_rmsnorm_pipeline) {
                fprintf(stderr, "Metal: rmsnorm pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        g_metal_ok = (g_attn_pipeline != nil);
        if (g_metal_ok) {
            fprintf(stderr, "Metal GPU compute: initialized (device=%s)\n",
                    [[g_device name] UTF8String]);
        }
    });
    return g_metal_ok;
}

bool metal_available() { return g_metal_ok; }

// ============ Metal GQA attention ============
void metal_gqa_attention(float* out, const float* q,
                         const uint16_t* k_cache_f16, const uint16_t* v_cache_f16,
                         int n_heads, int n_kv_heads, int head_dim, int q_head_stride,
                         int cache_start, int cache_len, int cache_capacity) {
    if (!g_metal_ok || cache_len <= 0) {
        memset(out, 0, (size_t)n_heads * head_dim * sizeof(float));
        return;
    }

    @autoreleasepool {
        size_t q_size = (size_t)n_heads * q_head_stride * sizeof(float);
        size_t kv_size = (size_t)cache_capacity * n_kv_heads * head_dim * sizeof(uint16_t);
        size_t out_size = (size_t)n_heads * head_dim * sizeof(float);
        float scale = 1.0f / sqrtf((float)head_dim);

        // Wrap existing memory as Metal buffers (zero-copy on Apple Silicon)
        id<MTLBuffer> q_buf = [g_device newBufferWithBytesNoCopy:(void*)q
            length:q_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> k_buf = [g_device newBufferWithBytesNoCopy:(void*)k_cache_f16
            length:kv_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> v_buf = [g_device newBufferWithBytesNoCopy:(void*)v_cache_f16
            length:kv_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> o_buf = [g_device newBufferWithBytesNoCopy:(void*)out
            length:out_size options:MTLResourceStorageModeShared deallocator:nil];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_attn_pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:k_buf offset:0 atIndex:1];
        [enc setBuffer:v_buf offset:0 atIndex:2];
        [enc setBuffer:o_buf offset:0 atIndex:3];
        [enc setBytes:&n_heads    length:sizeof(int)   atIndex:4];
        [enc setBytes:&n_kv_heads length:sizeof(int)   atIndex:5];
        [enc setBytes:&head_dim   length:sizeof(int)   atIndex:6];
        [enc setBytes:&q_head_stride length:sizeof(int) atIndex:7];
        [enc setBytes:&cache_start length:sizeof(int)   atIndex:8];
        [enc setBytes:&cache_len  length:sizeof(int)    atIndex:9];
        [enc setBytes:&cache_capacity length:sizeof(int) atIndex:10];
        [enc setBytes:&scale      length:sizeof(float)  atIndex:11];

        // One threadgroup per head, 256 threads per group
        int threads_per_group = std::min(256, cache_len);
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ============ Metal RMSNorm ============
void metal_rmsnorm(float* out, const float* x, const float* weight, int dim, float eps) {
    if (!g_metal_ok) return;

    @autoreleasepool {
        size_t buf_size = (size_t)dim * sizeof(float);

        id<MTLBuffer> x_buf = [g_device newBufferWithBytesNoCopy:(void*)x
            length:buf_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> w_buf = [g_device newBufferWithBytesNoCopy:(void*)weight
            length:buf_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> o_buf = [g_device newBufferWithBytesNoCopy:(void*)out
            length:buf_size options:MTLResourceStorageModeShared deallocator:nil];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_rmsnorm_pipeline];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:0 atIndex:1];
        [enc setBuffer:o_buf offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(int) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];

        int threads = std::min(256, dim);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace ane_lm
