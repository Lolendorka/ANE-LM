#include "generate.h"
#include "core/sampling.h"
#include <ane_lm/common.h>
#include <climits>

namespace ane_lm {

void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& messages,
    int max_tokens,
    bool enable_thinking,
    const SamplingParams& sampling,
    std::function<void(const GenerationResponse&)> callback)
{
    // Tokenize with chat template
    std::vector<int> prompt_tokens;
    if (tokenizer.has_chat_template()) {
        std::string formatted = tokenizer.apply_chat_template(messages, true, enable_thinking);
        prompt_tokens = tokenizer.encode(formatted);
    } else {
        // Fallback: concatenate all message contents
        std::string combined;
        for (auto& [role, content] : messages) {
            combined += content + "\n";
        }
        prompt_tokens = tokenizer.encode(combined);
    }

    // Prefill
    Timer prefill_timer;
    float* logits = nullptr;
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        logits = model.forward(prompt_tokens[i], i);
        if (!logits) {
            fprintf(stderr, "Forward failed during prefill at token index %d\n", i);
            return;
        }
    }
    double prefill_ms = prefill_timer.elapsed_ms();
    double prompt_tps = prompt_tokens.size() / (prefill_ms / 1000.0);

    // Decode
    Timer gen_timer;
    int n_generated = 0;
    std::vector<int> generated_tokens;
    int next_token = sample_token(logits, model.vocab_size(), sampling, generated_tokens);

    int limit = (max_tokens > 0) ? max_tokens : INT_MAX;
    for (int i = 0; i < limit; i++) {
        if (next_token == tokenizer.eos_id() || next_token == tokenizer.im_end_id()) {
            break;
        }

        n_generated++;
        generated_tokens.push_back(next_token);
        std::string piece = tokenizer.decode(next_token);

        if (callback) {
            GenerationResponse r;
            r.text = piece;
            r.token = next_token;
            r.prompt_tokens = (int)prompt_tokens.size();
            r.prompt_tps = prompt_tps;
            r.generation_tokens = n_generated;
            r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
            callback(r);
        }

        int pos = (int)prompt_tokens.size() + i;
        logits = model.forward(next_token, pos);
        if (!logits) {
            fprintf(stderr, "Forward failed during generation at step %d\n", i);
            return;
        }
        next_token = sample_token(logits, model.vocab_size(), sampling, generated_tokens);
    }

    // Final stats callback (token = -1 signals end)
    if (callback) {
        GenerationResponse r;
        r.token = -1;
        r.prompt_tokens = (int)prompt_tokens.size();
        r.prompt_tps = prompt_tps;
        r.generation_tokens = n_generated;
        r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
        callback(r);
    }
}

// Single-prompt overload wraps into messages vector
void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::string& prompt,
    int max_tokens,
    bool enable_thinking,
    const SamplingParams& sampling,
    std::function<void(const GenerationResponse&)> callback)
{
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    stream_generate(model, tokenizer, messages, max_tokens, enable_thinking, sampling, std::move(callback));
}

} // namespace ane_lm
