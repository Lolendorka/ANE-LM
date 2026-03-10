#include "cpu_ops.h"
#include <alloca.h>
#include <dispatch/dispatch.h>
#include "metal_ops.h"
#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

namespace ane_lm {

void silu_vec_inplace(float* x, int n, float* tmp) {
    int vn = n;
    float one = 1.0f;
    vDSP_vneg(x, 1, tmp, 1, (vDSP_Length)n);
    vvexpf(tmp, tmp, &vn);
    vDSP_vsadd(tmp, 1, &one, tmp, 1, (vDSP_Length)n);
    vDSP_vdiv(tmp, 1, x, 1, x, 1, (vDSP_Length)n);
}

void mul_sigmoid_inplace(float* y, const float* z, int n, float* tmp) {
    int vn = n;
    float one = 1.0f;
    vDSP_vneg(z, 1, tmp, 1, (vDSP_Length)n);
    vvexpf(tmp, tmp, &vn);
    vDSP_vsadd(tmp, 1, &one, tmp, 1, (vDSP_Length)n);
    vDSP_vdiv(tmp, 1, y, 1, y, 1, (vDSP_Length)n);
}

void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps) {
    float ss = 0.0f;
    vDSP_svesq(x, 1, &ss, (vDSP_Length)dim);
    ss = 1.0f / sqrtf(ss / dim + eps);
    vDSP_vsmul(x, 1, &ss, out, 1, (vDSP_Length)dim);
    vDSP_vmul(out, 1, weight, 1, out, 1, (vDSP_Length)dim);
}

void rmsnorm_gated(float* out, const float* x, const float* z,
                   const float* weight, int dim) {
    rmsnorm(out, x, weight, dim);
    float* tmp = (float*)alloca((size_t)dim * sizeof(float));
    mul_sigmoid_inplace(out, z, dim, tmp);
    vDSP_vmul(out, 1, z, 1, out, 1, (vDSP_Length)dim);
}

// ============ PERF: NEON alternating-pairs RoPE via vld2q_f32/vst2q_f32 ============
// Correct for (v[2j], v[2j+1]) pairs. vld2 de-interleaves: val[0]=even, val[1]=odd.
// Process 4 pairs (8 floats) per iteration using vmlsq/vmlaq SIMD ops.
#if defined(__aarch64__) || defined(__arm64__)
static void apply_rope_neon_pairs(float* v, const float* cos_row,
                                   const float* sin_row, int rot_dim) {
    int half = rot_dim / 2;  // number of pairs
    int j = 0;
    for (; j + 3 < half; j += 4) {
        float32x4x2_t vp = vld2q_f32(&v[j * 2]);
        float32x4_t cos4 = vld1q_f32(&cos_row[j]);
        float32x4_t sin4 = vld1q_f32(&sin_row[j]);
        // new_even = even * cos - odd * sin
        float32x4_t new_even = vmlsq_f32(vmulq_f32(vp.val[0], cos4), vp.val[1], sin4);
        // new_odd  = even * sin + odd * cos
        float32x4_t new_odd  = vmlaq_f32(vmulq_f32(vp.val[0], sin4), vp.val[1], cos4);
        float32x4x2_t result = {new_even, new_odd};
        vst2q_f32(&v[j * 2], result);
    }
    // Scalar tail
    for (; j < half; j++) {
        float v0 = v[j * 2], v1 = v[j * 2 + 1];
        v[j * 2]     = v0 * cos_row[j] - v1 * sin_row[j];
        v[j * 2 + 1] = v0 * sin_row[j] + v1 * cos_row[j];
    }
}
#endif

void apply_rope_cached(float* q, float* k, int n_q_heads, int n_kv_heads,
                       int head_dim, int q_head_stride, int k_head_stride,
                       int rot_dim, int pos, float theta,
                       const float* cos_row, const float* sin_row) {
#if defined(__aarch64__) || defined(__arm64__)
    if (cos_row && sin_row) {
        // NEON fast path: correct alternating-pairs rotation with vld2/vst2
        for (int h = 0; h < n_q_heads + n_kv_heads; h++) {
            float* v = (h < n_q_heads) ? q + h * q_head_stride : k + (h - n_q_heads) * k_head_stride;
            apply_rope_neon_pairs(v, cos_row, sin_row, rot_dim);
        }
        return;
    }
#endif
    // Scalar fallback (no precomputed trig tables)
    for (int h = 0; h < n_q_heads + n_kv_heads; h++) {
        float* v = (h < n_q_heads) ? q + h * q_head_stride : k + (h - n_q_heads) * k_head_stride;
        for (int i = 0, j = 0; i < rot_dim; i += 2, j++) {
            float cos_a, sin_a;
            if (cos_row && sin_row) { cos_a = cos_row[j]; sin_a = sin_row[j]; }
            else {
                float freq = 1.0f / powf(theta, (float)i / (float)rot_dim);
                float angle = pos * freq; cos_a = cosf(angle); sin_a = sinf(angle);
            }
            float v0 = v[i], v1 = v[i + 1];
            v[i]     = v0 * cos_a - v1 * sin_a;
            v[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

// ============ PATCH 3: Vectorized softmax via vvexpf ============
void softmax(float* x, int n) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, (vDSP_Length)n);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, (vDSP_Length)n);
    int vn = n;
    vvexpf(x, x, &vn);
    float sum;
    vDSP_sve(x, 1, &sum, (vDSP_Length)n);
    float inv = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv, x, 1, (vDSP_Length)n);
}

void matvec(float* y, const float* W, const float* x, int out_dim, int in_dim) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim, 1.0f,
                W, in_dim, x, 1, 0.0f, y, 1);
}

void l2_normalize(float* x, int dim) {
    float norm = 0.0f;
    vDSP_svesq(x, 1, &norm, (vDSP_Length)dim);
    norm = 1.0f / sqrtf(norm + 1e-12f);
    vDSP_vsmul(x, 1, &norm, x, 1, (vDSP_Length)dim);
}

void conv1d_update(float* y, float* conv_state, int* state_pos, const float* x,
                   const float* w, int channels, int kernel_size) {
    int state_len = kernel_size - 1;
    int pos = *state_pos;

    if (kernel_size == 4) {
        int p0 = pos;
        int p1 = (pos + 1); if (p1 == 3) p1 = 0;
        int p2 = (p1 + 1);  if (p2 == 3) p2 = 0;

        for (int c = 0; c < channels; c++) {
            const int sbase = c * 3, wbase = c * 4;
            float s0 = conv_state[sbase + p0];
            float s1 = conv_state[sbase + p1];
            float s2 = conv_state[sbase + p2];
            float xc = x[c];
            y[c] = s0 * w[wbase] + s1 * w[wbase+1] + s2 * w[wbase+2] + xc * w[wbase+3];
            conv_state[sbase + p0] = xc;
        }
    } else {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            int base = c * state_len;
            for (int j = 0; j < state_len; j++) {
                int idx = pos + j;
                if (idx >= state_len) idx -= state_len;
                sum += conv_state[base + idx] * w[c * kernel_size + j];
            }
            float xc = x[c];
            y[c] = sum + xc * w[c * kernel_size + state_len];
            conv_state[base + pos] = xc;
        }
    }
    pos++;
    if (pos == state_len) pos = 0;
    *state_pos = pos;
}

void ssm_step(float* y, float* state, const float* q, const float* k,
              const float* v, float decay, float beta, int key_dim, int value_dim) {
    float* Sk = (float*)alloca(value_dim * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasTrans, key_dim, value_dim, 1.0f,
                state, value_dim, k, 1, 0.0f, Sk, 1);
    float* delta = (float*)alloca(value_dim * sizeof(float));
    vDSP_vsub(Sk, 1, v, 1, delta, 1, (vDSP_Length)value_dim);
    cblas_sscal(key_dim * value_dim, decay, state, 1);
    cblas_sger(CblasRowMajor, key_dim, value_dim, beta,
               k, 1, delta, 1, state, value_dim);
    cblas_sgemv(CblasRowMajor, CblasTrans, key_dim, value_dim, 1.0f,
                state, value_dim, q, 1, 0.0f, y, 1);
}

// ============ PATCH 8: Parallel GQA attention heads via dispatch_apply ============
void gqa_attention(float* out, const float* q,
                   const float* k_cache, const float* v_cache,
                   int n_heads, int n_kv_heads, int head_dim, int q_head_stride,
                   int cache_start, int cache_len, int cache_capacity) {
    if (cache_len <= 0) {
        memset(out, 0, (size_t)n_heads * head_dim * sizeof(float));
        return;
    }

    int groups = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_step = (size_t)n_kv_heads * head_dim;

    int first_span = cache_capacity - cache_start;
    if (first_span > cache_len) first_span = cache_len;
    int second_span = cache_len - first_span;

    float* scores_buf = (float*)malloc((size_t)n_heads * cache_len * sizeof(float));

    auto head_fn = [&](int h) {
        int kv_h = h / groups;
        const float* qh = q + (size_t)h * q_head_stride;
        float* oh = out + h * head_dim;
        size_t kv_head_off = (size_t)kv_h * head_dim;
        float* scores = scores_buf + (size_t)h * cache_len;

        int t = 0;
        for (int s = 0; s < first_span; s++, t++) {
            const float* kh = k_cache + ((size_t)(cache_start + s) * kv_step + kv_head_off);
            float dot = 0.0f;
            vDSP_dotpr(qh, 1, kh, 1, &dot, (vDSP_Length)head_dim);
            scores[t] = dot * scale;
        }
        for (int s = 0; s < second_span; s++, t++) {
            const float* kh = k_cache + ((size_t)s * kv_step + kv_head_off);
            float dot = 0.0f;
            vDSP_dotpr(qh, 1, kh, 1, &dot, (vDSP_Length)head_dim);
            scores[t] = dot * scale;
        }

        softmax(scores, cache_len);
        memset(oh, 0, head_dim * sizeof(float));
        t = 0;
        for (int s = 0; s < first_span; s++, t++) {
            const float* vh = v_cache + ((size_t)(cache_start + s) * kv_step + kv_head_off);
            cblas_saxpy(head_dim, scores[t], vh, 1, oh, 1);
        }
        for (int s = 0; s < second_span; s++, t++) {
            const float* vh = v_cache + ((size_t)s * kv_step + kv_head_off);
            cblas_saxpy(head_dim, scores[t], vh, 1, oh, 1);
        }
    };

    if (cache_len >= 128 && n_heads >= 4) {
        dispatch_apply(n_heads,
            dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
            ^(size_t h) { head_fn((int)h); });
    } else {
        for (int h = 0; h < n_heads; h++) head_fn(h);
    }

    free(scores_buf);
}

} // namespace ane_lm
