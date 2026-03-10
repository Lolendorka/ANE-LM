// metal_ops.mm — Metal GPU compute engine for LLM inference
// Replaces ANE for matmuls with lower dispatch overhead on Apple Silicon
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_ops.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <mutex>
#include <algorithm>

namespace ane_lm {

static const char* g_shader_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Matrix-vector multiply: y[i] = sum_j(W[i*in_dim+j] * x[j])
// W is fp16, x is fp32, y is fp32. One thread per output element.
kernel void matvec_f16(
    device const half*  W      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint&      in_dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    device const half* row = W + tid * in_dim;
    float sum = 0.0f;
    uint j = 0;
    // Vectorized: process 8 elements per iteration
    for (; j + 7 < in_dim; j += 8) {
        half8 w8 = *((device const half8*)(row + j));
        sum += float(w8[0]) * x[j]   + float(w8[1]) * x[j+1]
             + float(w8[2]) * x[j+2] + float(w8[3]) * x[j+3]
             + float(w8[4]) * x[j+4] + float(w8[5]) * x[j+5]
             + float(w8[6]) * x[j+6] + float(w8[7]) * x[j+7];
    }
    for (; j < in_dim; j++) sum += float(row[j]) * x[j];
    y[tid] = sum;
}

// Fused gate+up+SiLU: out[i] = SiLU(gate_row_i . x) * (up_row_i . x)
// Two matvecs + SiLU + mul in one dispatch (2x weight bandwidth but 1 dispatch)
kernel void fused_gate_up_silu(
    device const half*  W_gate [[buffer(0)]],
    device const half*  W_up   [[buffer(1)]],
    device const float* x      [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant uint&      in_dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    device const half* g_row = W_gate + tid * in_dim;
    device const half* u_row = W_up   + tid * in_dim;
    float g = 0.0f, u = 0.0f;
    uint j = 0;
    for (; j + 7 < in_dim; j += 8) {
        half8 gw = *((device const half8*)(g_row + j));
        half8 uw = *((device const half8*)(u_row + j));
        g += float(gw[0])*x[j]   + float(gw[1])*x[j+1]
           + float(gw[2])*x[j+2] + float(gw[3])*x[j+3]
           + float(gw[4])*x[j+4] + float(gw[5])*x[j+5]
           + float(gw[6])*x[j+6] + float(gw[7])*x[j+7];
        u += float(uw[0])*x[j]   + float(uw[1])*x[j+1]
           + float(uw[2])*x[j+2] + float(uw[3])*x[j+3]
           + float(uw[4])*x[j+4] + float(uw[5])*x[j+5]
           + float(uw[6])*x[j+6] + float(uw[7])*x[j+7];
    }
    for (; j < in_dim; j++) {
        g += float(g_row[j]) * x[j];
        u += float(u_row[j]) * x[j];
    }
    // SiLU(gate) * up
    out[tid] = g / (1.0f + exp(-g)) * u;
}

// GQA Attention: one threadgroup per head
kernel void gqa_attention_kernel(
    device const float* q          [[buffer(0)]],
    device const half*  k_cache    [[buffer(1)]],
    device const half*  v_cache    [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    constant int&       n_heads    [[buffer(4)]],
    constant int&       n_kv_heads [[buffer(5)]],
    constant int&       head_dim   [[buffer(6)]],
    constant int&       q_stride   [[buffer(7)]],
    constant int&       cache_start[[buffer(8)]],
    constant int&       cache_len  [[buffer(9)]],
    constant int&       capacity   [[buffer(10)]],
    constant float&     scale      [[buffer(11)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tcount[[threads_per_threadgroup]])
{
    int h = gid;
    if (h >= n_heads) return;
    int groups = n_heads / n_kv_heads;
    int kv_h = h / groups;
    int kv_step = n_kv_heads * head_dim;
    device const float* qh = q + h * q_stride;
    device float* oh = output + h * head_dim;

    threadgroup float scores[2048];
    threadgroup float s_max[1];
    threadgroup float s_sum[1];

    if (tid == 0) { s_max[0] = -INFINITY; s_sum[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = (int)tid; t < cache_len; t += (int)tcount) {
        int pos = cache_start + t;
        if (pos >= capacity) pos -= capacity;
        device const half* kh = k_cache + pos * kv_step + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qh[d] * float(kh[d]);
        scores[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float m = -INFINITY;
        for (int t = 0; t < cache_len; t++) m = max(m, scores[t]);
        s_max[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = s_max[0];
    for (int t = (int)tid; t < cache_len; t += (int)tcount) {
        scores[t] = exp(scores[t] - mx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < cache_len; t++) s += scores[t];
        s_sum[0] = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = s_sum[0];
    for (int t = (int)tid; t < cache_len; t += (int)tcount) scores[t] *= inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tcount) {
        float acc = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            int pos = cache_start + t;
            if (pos >= capacity) pos -= capacity;
            acc += scores[t] * float(v_cache[pos * kv_step + kv_h * head_dim + d]);
        }
        oh[d] = acc;
    }
}
)METAL";

// ============ Metal state ============
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_matvec_pipeline = nil;
static id<MTLComputePipelineState> g_fused_gu_pipeline = nil;
static id<MTLComputePipelineState> g_attn_pipeline = nil;
static bool g_metal_ok = false;
static std::once_flag g_metal_once;

// Reusable command buffer + scratch buffers for zero-copy
static id<MTLBuffer> g_x_buf = nil;      // input vector buffer
static id<MTLBuffer> g_y_buf = nil;      // output vector buffer
static id<MTLBuffer> g_inter_buf = nil;  // intermediate FFN buffer
static int g_x_buf_size = 0;
static int g_y_buf_size = 0;
static int g_inter_buf_size = 0;

struct MetalWeight {
    id<MTLBuffer> buffer;
    int out_dim;
    int in_dim;
};

static void ensure_buf(id<MTLBuffer>& buf, int& cur_size, int need_size) {
    if (need_size > cur_size) {
        buf = [g_device newBufferWithLength:need_size * sizeof(float)
                       options:MTLResourceStorageModeShared];
        cur_size = need_size;
    }
}

bool metal_init() {
    std::call_once(g_metal_once, []() {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return;
        g_queue = [g_device newCommandQueue];
        if (!g_queue) return;

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:g_shader_source];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;

        id<MTLLibrary> lib = [g_device newLibraryWithSource:source options:opts error:&error];
        if (!lib) {
            fprintf(stderr, "Metal shader compile: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        auto make_pipe = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) return nil;
            return [g_device newComputePipelineStateWithFunction:fn error:&error];
        };

        g_matvec_pipeline = make_pipe(@"matvec_f16");
        g_fused_gu_pipeline = make_pipe(@"fused_gate_up_silu");
        g_attn_pipeline = make_pipe(@"gqa_attention_kernel");

        g_metal_ok = (g_matvec_pipeline != nil);
        if (g_metal_ok) {
            fprintf(stderr, "Metal GPU compute: initialized (%s)\n",
                    [[g_device name] UTF8String]);
        }
    });
    return g_metal_ok;
}

bool metal_available() { return g_metal_ok; }

MetalWeight* metal_create_weight(const uint16_t* fp16_data, int out_dim, int in_dim) {
    if (!g_metal_ok) return nullptr;
    size_t size = (size_t)out_dim * in_dim * sizeof(uint16_t);
    MetalWeight* w = new MetalWeight();
    w->out_dim = out_dim;
    w->in_dim = in_dim;
    // Zero-copy wrap (Apple Silicon unified memory)
    w->buffer = [g_device newBufferWithBytesNoCopy:(void*)fp16_data
                          length:size
                          options:MTLResourceStorageModeShared
                          deallocator:nil];
    if (!w->buffer) {
        // Fallback: copy
        w->buffer = [g_device newBufferWithBytes:fp16_data length:size
                              options:MTLResourceStorageModeShared];
    }
    return w;
}

void metal_free_weight(MetalWeight* w) {
    if (w) { w->buffer = nil; delete w; }
}

void metal_matvec(MetalWeight* w, float* output, const float* input) {
    if (!w || !g_metal_ok) return;

    @autoreleasepool {
        ensure_buf(g_x_buf, g_x_buf_size, w->in_dim);
        ensure_buf(g_y_buf, g_y_buf_size, w->out_dim);

        // Copy input to Metal buffer
        memcpy([g_x_buf contents], input, w->in_dim * sizeof(float));

        uint in_dim = (uint)w->in_dim;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matvec_pipeline];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:g_x_buf   offset:0 atIndex:1];
        [enc setBuffer:g_y_buf   offset:0 atIndex:2];
        [enc setBytes:&in_dim    length:sizeof(uint) atIndex:3];

        NSUInteger tpg = std::min((NSUInteger)g_matvec_pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        [enc dispatchThreads:MTLSizeMake(w->out_dim, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(output, [g_y_buf contents], w->out_dim * sizeof(float));
    }
}

void metal_fused_ffn(MetalWeight* gate_w, MetalWeight* up_w, MetalWeight* down_w,
                     float* output, const float* input,
                     int in_dim, int inter_dim, int out_dim) {
    if (!g_metal_ok || !gate_w || !up_w || !down_w) return;

    @autoreleasepool {
        ensure_buf(g_x_buf, g_x_buf_size, std::max(in_dim, inter_dim));
        ensure_buf(g_inter_buf, g_inter_buf_size, inter_dim);
        ensure_buf(g_y_buf, g_y_buf_size, out_dim);

        memcpy([g_x_buf contents], input, in_dim * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        // Dispatch 1: gate + up + SiLU (fused)
        {
            uint udim = (uint)in_dim;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_fused_gu_pipeline];
            [enc setBuffer:gate_w->buffer offset:0 atIndex:0];
            [enc setBuffer:up_w->buffer   offset:0 atIndex:1];
            [enc setBuffer:g_x_buf        offset:0 atIndex:2];
            [enc setBuffer:g_inter_buf    offset:0 atIndex:3];
            [enc setBytes:&udim           length:sizeof(uint) atIndex:4];

            NSUInteger tpg = std::min((NSUInteger)g_fused_gu_pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
            [enc dispatchThreads:MTLSizeMake(inter_dim, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        }

        // Dispatch 2: down projection
        {
            uint udim = (uint)inter_dim;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_matvec_pipeline];
            [enc setBuffer:down_w->buffer offset:0 atIndex:0];
            [enc setBuffer:g_inter_buf    offset:0 atIndex:1];
            [enc setBuffer:g_y_buf        offset:0 atIndex:2];
            [enc setBytes:&udim           length:sizeof(uint) atIndex:3];

            NSUInteger tpg = std::min((NSUInteger)g_matvec_pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
            [enc dispatchThreads:MTLSizeMake(out_dim, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(output, [g_y_buf contents], out_dim * sizeof(float));
    }
}

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
        [enc setBytes:&n_heads     length:sizeof(int)   atIndex:4];
        [enc setBytes:&n_kv_heads  length:sizeof(int)   atIndex:5];
        [enc setBytes:&head_dim    length:sizeof(int)   atIndex:6];
        [enc setBytes:&q_head_stride length:sizeof(int) atIndex:7];
        [enc setBytes:&cache_start length:sizeof(int)   atIndex:8];
        [enc setBytes:&cache_len   length:sizeof(int)   atIndex:9];
        [enc setBytes:&cache_capacity length:sizeof(int) atIndex:10];
        [enc setBytes:&scale       length:sizeof(float) atIndex:11];

        int tpg = std::min(256, cache_len);
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace ane_lm
