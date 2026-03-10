#include "sampling.h"
#include "cpu_ops.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <Accelerate/Accelerate.h>

namespace ane_lm {

int sample_token(const float* logits, int vocab_size,
                 const SamplingParams& params,
                 const std::vector<int>& recent_tokens) {
    float* adjusted = (float*)malloc(vocab_size * sizeof(float));
    memcpy(adjusted, logits, vocab_size * sizeof(float));

    if (!recent_tokens.empty()) {
        int start = std::max(0, (int)recent_tokens.size() - params.repetition_context_size);
        std::unordered_map<int, int> freq;
        for (int j = start; j < (int)recent_tokens.size(); j++) {
            int tok = recent_tokens[j];
            if (tok >= 0 && tok < vocab_size) freq[tok]++;
        }
        for (auto& [tok, count] : freq) {
            if (params.repetition_penalty > 1.0f) {
                if (adjusted[tok] > 0.0f) adjusted[tok] /= params.repetition_penalty;
                else                       adjusted[tok] *= params.repetition_penalty;
            }
            if (params.frequency_penalty > 0.0f)
                adjusted[tok] -= params.frequency_penalty * count;
        }
    }

    if (params.temperature <= 0.0f) {
        // ============ PERF: vDSP_maxvi for greedy argmax ============
        // Replaces O(vocab) scalar loop with SIMD max search.
        float max_val;
        vDSP_Length max_idx = 0;
        vDSP_maxvi(adjusted, 1, &max_val, &max_idx, (vDSP_Length)vocab_size);
        free(adjusted);
        return (int)max_idx;
    }

    // ============ PERF: vDSP_vsmul for temperature scaling ============
    // Replaces scalar loop with single SIMD multiply.
    float inv_t = 1.0f / params.temperature;
    vDSP_vsmul(adjusted, 1, &inv_t, adjusted, 1, (vDSP_Length)vocab_size);

    softmax(adjusted, vocab_size);

    float r = (float)drand48();
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cum += adjusted[i];
        if (cum >= r) { free(adjusted); return i; }
    }
    free(adjusted);
    return vocab_size - 1;
}

} // namespace ane_lm
