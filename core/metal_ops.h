#pragma once

#include <cstdint>

namespace ane_lm {

// Metal GPU compute engine for matrix-vector operations
// Can replace ANE for all matmuls with potentially lower dispatch overhead

bool metal_init();
bool metal_available();

// Opaque handle for Metal weight buffer
struct MetalWeight;

// Create Metal weight buffer from fp16 data (zero-copy on Apple Silicon)
// fp16_data must remain valid for the lifetime of the MetalWeight
MetalWeight* metal_create_weight(const uint16_t* fp16_data, int out_dim, int in_dim);
void metal_free_weight(MetalWeight* w);

// Matrix-vector multiply: y = W * x (fp16 weights, fp32 input/output)
void metal_matvec(MetalWeight* w, float* output, const float* input);

// Fused FFN: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
// gate_w, up_w: [inter_dim x in_dim], down_w: [out_dim x inter_dim]
void metal_fused_ffn(MetalWeight* gate_w, MetalWeight* up_w, MetalWeight* down_w,
                     float* output, const float* input,
                     int in_dim, int inter_dim, int out_dim);

// Metal GQA attention (fp16 KV cache, direct GPU processing)
void metal_gqa_attention(float* out, const float* q,
                         const uint16_t* k_cache_f16, const uint16_t* v_cache_f16,
                         int n_heads, int n_kv_heads, int head_dim, int q_head_stride,
                         int cache_start, int cache_len, int cache_capacity);

} // namespace ane_lm
