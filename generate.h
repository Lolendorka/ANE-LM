#pragma once

#include "models/llm/qwen3_5.h"
#include "core/tokenizer.h"
#include "core/sampling.h"
#include <functional>
#include <string>
#include <vector>
#include <utility>

namespace ane_lm {

struct GenerationResponse {
    std::string text;
    int token = 0;
    int prompt_tokens = 0;
    double prompt_tps = 0.0;
    int generation_tokens = 0;
    double generation_tps = 0.0;
};

// Multi-turn: accepts full message history
void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& messages,
    int max_tokens = 0,
    bool enable_thinking = false,
    const SamplingParams& sampling = {},
    std::function<void(const GenerationResponse&)> callback = nullptr,
    int prompt_cache_len = 0,  // skip first N tokens (already in KV cache)
    std::vector<int>* out_all_tokens = nullptr);  // output: all processed token IDs

// Single-prompt convenience overload
void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::string& prompt,
    int max_tokens = 0,
    bool enable_thinking = false,
    const SamplingParams& sampling = {},
    std::function<void(const GenerationResponse&)> callback = nullptr,
    int prompt_cache_len = 0,
    std::vector<int>* out_all_tokens = nullptr);

} // namespace ane_lm
