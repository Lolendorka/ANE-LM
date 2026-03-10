#pragma once

#include <cstdint>

namespace ane_lm {

// Initialize Metal compute pipeline. Call once at startup.
// Returns true if Metal is available and initialized.
bool metal_init();
bool metal_available();

// Metal-accelerated GQA attention (replaces CPU gqa_attention for long contexts)
// Uses GPU massive parallelism for dot products + softmax + weighted sum.
// KV cache is in fp16 (uint16_t*) for bandwidth efficiency.
void metal_gqa_attention(float* out, const float* q,
                         const uint16_t* k_cache_f16, const uint16_t* v_cache_f16,
                         int n_heads, int n_kv_heads, int head_dim, int q_head_stride,
                         int cache_start, int cache_len, int cache_capacity);

// Metal-accelerated RMSNorm
void metal_rmsnorm(float* out, const float* x, const float* weight, int dim, float eps);

} // namespace ane_lm
