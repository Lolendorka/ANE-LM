#include "qwen3_5.h"
#include <ane_lm/common.h>
#include "../../core/cpu_ops.h"
#include "../../core/metal_ops.h"
#include <atomic>
#include <cmath>
#include <dispatch/dispatch.h>
#include <fstream>
#include <sys/stat.h>

namespace ane_lm {

using json = nlohmann::json;

// --- Qwen35Args::from_json ---

Qwen35Args Qwen35Args::from_json(const json& j) {
    Qwen35Args args;

    const json& tc = j.contains("text_config") ? j["text_config"] : j;

    args.hidden_size = tc.value("hidden_size", args.hidden_size);
    args.num_hidden_layers = tc.value("num_hidden_layers", args.num_hidden_layers);
    args.num_attention_heads = tc.value("num_attention_heads", args.num_attention_heads);
    args.num_key_value_heads = tc.value("num_key_value_heads", args.num_key_value_heads);
    args.head_dim = tc.value("head_dim", args.head_dim);
    args.intermediate_size = tc.value("intermediate_size", args.intermediate_size);
    args.vocab_size = tc.value("vocab_size", args.vocab_size);
    args.full_attention_interval = tc.value("full_attention_interval", args.full_attention_interval);
    args.rms_norm_eps = tc.value("rms_norm_eps", args.rms_norm_eps);
    args.tie_word_embeddings = tc.value("tie_word_embeddings", j.value("tie_word_embeddings", args.tie_word_embeddings));
    args.attn_output_gate = tc.value("attn_output_gate", args.attn_output_gate);
    args.linear_num_key_heads = tc.value("linear_num_key_heads", args.linear_num_key_heads);
    args.linear_key_head_dim = tc.value("linear_key_head_dim", args.linear_key_head_dim);
    args.linear_value_head_dim = tc.value("linear_value_head_dim", args.linear_value_head_dim);
    args.linear_num_value_heads = tc.value("linear_num_value_heads", args.linear_num_value_heads);
    args.linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", args.linear_conv_kernel_dim);

    if (tc.contains("rope_parameters")) {
        auto& rp = tc["rope_parameters"];
        args.rope_theta = rp.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    } else {
        args.rope_theta = tc.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    }

    if (tc.contains("layer_types")) {
        for (auto& lt : tc["layer_types"]) {
            std::string s = lt.get<std::string>();
            if (s == "linear_attention") {
                args.layer_types.push_back(LayerType::LinearAttention);
            } else {
                args.layer_types.push_back(LayerType::FullAttention);
            }
        }
    } else {
        for (int i = 0; i < args.num_hidden_layers; i++) {
            if ((i + 1) % args.full_attention_interval == 0) {
                args.layer_types.push_back(LayerType::FullAttention);
            } else {
                args.layer_types.push_back(LayerType::LinearAttention);
            }
        }
    }

    return args;
}

// --- Qwen35Model ---

Qwen35Model::~Qwen35Model() {
    free(embed_tokens_);
    free(final_norm_);
    free(x_);
    free(x_norm_);
    free(logits_);
    free(scratch_qkv_);
    free(scratch_conv_);
    free(scratch_y_);
    free(scratch_attn_);
    free(scratch_tmp_);
    free(rope_cos_);
    free(rope_sin_);

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        free(lw.input_layernorm);
        free(lw.post_attention_layernorm);

        if (lw.type == LayerType::LinearAttention) {
            free(lw.deltanet.in_proj_a);
            free(lw.deltanet.in_proj_b);
            free(lw.deltanet.conv1d_w);
            free(lw.deltanet.A);
            free(lw.deltanet.dt_bias);
            free(lw.deltanet.norm_w);
        } else {
            free(lw.full_attn.q_norm);
            free(lw.full_attn.k_norm);
        }
    }

    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            free(kv_caches_[L].k_cache);
            free(kv_caches_[L].v_cache);
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            free(delta_states_[L].ssm_state);
            free(delta_states_[L].conv_state);
        }
        ane_free_layer(&ane_layers_[L]);
    }

    free_lm_head_ane();
}

void Qwen35Model::reset() {
    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            kv_caches_[L].len = 0;
            kv_caches_[L].start = 0;
            memset(kv_caches_[L].k_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(uint16_t));
            memset(kv_caches_[L].v_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(uint16_t));
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            memset(delta_states_[L].ssm_state, 0, (size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_ * sizeof(float));
            memset(delta_states_[L].conv_state, 0, (size_t)lin_qkv_dim_ * (conv_kernel_ - 1) * sizeof(float));
            delta_states_[L].conv_pos = 0;
        }
    }
}

void Qwen35Model::apply_args(const Qwen35Args& args) {
    hidden_size_ = args.hidden_size;
    intermediate_size_ = args.intermediate_size;
    vocab_size_ = args.vocab_size;
    num_layers_ = args.num_hidden_layers;
    num_q_heads_ = args.num_attention_heads;
    num_kv_heads_ = args.num_key_value_heads;
    head_dim_ = args.head_dim;
    rot_dim_ = args.rotation_dim();
    rope_theta_ = args.rope_theta;
    rms_eps_ = args.rms_norm_eps;
    lin_num_heads_ = args.linear_num_key_heads;
    lin_num_val_heads_ = args.linear_num_value_heads;
    lin_key_dim_ = args.linear_key_head_dim;
    lin_val_dim_ = args.linear_value_head_dim;
    lin_total_key_ = lin_num_heads_ * lin_key_dim_;
    lin_total_val_ = lin_num_val_heads_ * lin_val_dim_;
    lin_qkv_dim_ = lin_total_key_ * 2 + lin_total_val_;
    conv_kernel_ = args.linear_conv_kernel_dim;
    full_q_dim_ = num_q_heads_ * head_dim_ * 2;
    full_kv_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = num_q_heads_ * head_dim_;
    attn_output_gate_ = args.attn_output_gate;
    layer_types_ = args.layer_types;
}

bool Qwen35Model::load(const std::string& model_dir) {
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", config_path.c_str());
        return false;
    }
    json j = json::parse(f);
    Qwen35Args args = Qwen35Args::from_json(j);
    apply_args(args);

    auto sf = ModelWeights::open(model_dir);
    if (!sf) {
        fprintf(stderr, "Failed to open model weights in %s\n", model_dir.c_str());
        return false;
    }

    const SFTensor* embed = sf->find("model.language_model.embed_tokens.weight");
    if (!embed || embed->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid embed_tokens.weight\n");
        return false;
    }
    const SFTensor* gate = sf->find("model.language_model.layers.0.mlp.gate_proj.weight");
    if (!gate || gate->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid gate_proj.weight\n");
        return false;
    }

    hidden_size_ = (int)embed->shape[1];
    vocab_size_ = (int)embed->shape[0];
    intermediate_size_ = (int)gate->shape[0];

    LOG("Model dims: hidden=%d intermediate=%d vocab=%d layers=%d\n",
        hidden_size_, intermediate_size_, vocab_size_, num_layers_);

    ane_init();

    x_ = (float*)calloc(hidden_size_, sizeof(float));
    x_norm_ = (float*)calloc(hidden_size_, sizeof(float));
    logits_ = (float*)calloc(vocab_size_, sizeof(float));
    scratch_qkv_ = (float*)calloc(lin_qkv_dim_ + lin_total_val_, sizeof(float));
    scratch_conv_ = (float*)calloc(lin_qkv_dim_, sizeof(float));
    scratch_y_ = (float*)calloc(lin_total_val_, sizeof(float));
    scratch_attn_ = (float*)calloc(full_out_dim_, sizeof(float));
    scratch_tmp_ = (float*)calloc((size_t)lin_num_val_heads_ * 2 + lin_qkv_dim_, sizeof(float));
    rope_cos_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));
    rope_sin_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));

    if (rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        float inv_freq[half_rot];
        for (int j2 = 0, i = 0; i < rot_dim_; i += 2, j2++) {
            inv_freq[j2] = 1.0f / powf(rope_theta_, (float)i / (float)rot_dim_);
        }
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
            float* cos_row = rope_cos_ + (size_t)pos * half_rot;
            float* sin_row = rope_sin_ + (size_t)pos * half_rot;
            for (int j2 = 0; j2 < half_rot; j2++) {
                float angle = pos * inv_freq[j2];
                cos_row[j2] = cosf(angle);
                sin_row[j2] = sinf(angle);
            }
        }
    }

    layers_.resize(num_layers_);
    delta_states_.resize(num_layers_);
    kv_caches_.resize(num_layers_);
    ane_layers_.resize(num_layers_);

    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            auto& kv = kv_caches_[L];
            kv.k_cache = (uint16_t*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(uint16_t));
            kv.v_cache = (uint16_t*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(uint16_t));
            kv.len = 0;
            kv.start = 0;
            kv.capacity = KV_CACHE_CAPACITY;
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            auto& ds = delta_states_[L];
            ds.ssm_state = (float*)calloc((size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_, sizeof(float));
            ds.conv_state = (float*)calloc((size_t)lin_qkv_dim_ * (conv_kernel_ - 1), sizeof(float));
            ds.conv_pos = 0;
        }
    }

    if (!load_weights(sf.get())) { return false; }
    std::string blob_dir = model_dir + "/ane_weights";
    struct stat st_blob;
    bool has_blobs = (stat(blob_dir.c_str(), &st_blob) == 0 && S_ISDIR(st_blob.st_mode));
    if (has_blobs) {
        LOG("Using pre-converted ANE blobs from %s\n", blob_dir.c_str());
    }

    if (!compile_ane(sf.get(), has_blobs ? blob_dir : "")) { return false; }

    return true;
}

bool Qwen35Model::load_weights(ModelWeights* sf) {
    char name[256];

    embed_tokens_ = sf->load_bf16_to_f32("model.language_model.embed_tokens.weight",
                                           (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;

    final_norm_ = sf->load_norm_weight("model.language_model.norm.weight", hidden_size_);
    if (!final_norm_) return false;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        lw.type = layer_types_[L];

        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", L);
        lw.input_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.input_layernorm) return false;

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", L);
        lw.post_attention_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.post_attention_layernorm) return false;

        if (lw.type == LayerType::LinearAttention) {
            auto& dw = lw.deltanet;

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L);
            dw.in_proj_a = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L);
            dw.in_proj_b = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.conv1d.weight", L);
            dw.conv1d_w = sf->load_bf16_to_f32(name, (int64_t)lin_qkv_dim_ * conv_kernel_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.A_log", L);
            dw.A = sf->load_f32_direct(name, lin_num_val_heads_);
            if (dw.A) {
                for (int i = 0; i < lin_num_val_heads_; i++) dw.A[i] = expf(dw.A[i]);
            }

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.dt_bias", L);
            dw.dt_bias = sf->load_bf16_to_f32(name, lin_num_val_heads_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.norm.weight", L);
            dw.norm_w = sf->load_f32_direct(name, lin_val_dim_);

            if (!dw.in_proj_a || !dw.in_proj_b || !dw.conv1d_w ||
                !dw.A || !dw.dt_bias || !dw.norm_w) {
                fprintf(stderr, "Failed to load DeltaNet weights for layer %d\n", L);
                return false;
            }
        } else {
            auto& fw = lw.full_attn;

            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_norm.weight", L);
            fw.q_norm = sf->load_norm_weight(name, head_dim_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_norm.weight", L);
            fw.k_norm = sf->load_norm_weight(name, head_dim_);

            if (!fw.q_norm || !fw.k_norm) {
                fprintf(stderr, "Failed to load FullAttn weights for layer %d\n", L);
                return false;
            }
        }
    }

    LOG("All weights loaded successfully\n");
    return true;
}

static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/";
    for (const char* c = tensor_name; *c; c++) {
        p += (*c == '.') ? '/' : *c;
    }
    p += ".bin";
    return p;
}

bool Qwen35Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) {
        fprintf(stderr, "ANE not available, cannot run\n");
        return false;
    }

    bool use_blobs = !blob_dir.empty();
    LOG("Compiling ANE kernels%s...\n", use_blobs ? " (from blobs)" : "");
    char name[256], name2[256], name3[256];

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d (%s)...\r", L+1, num_layers_,
            layer_types_[L] == LayerType::LinearAttention ? "deltanet" : "full_attn");

        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", L);
            snprintf(name2, sizeof(name2), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_2_blob(
                    blob_path(blob_dir, name), lin_qkv_dim_,
                    blob_path(blob_dir, name2), lin_total_val_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_2(
                    sf->get_bf16_ptr(name), lin_qkv_dim_,
                    sf->get_bf16_ptr(name2), lin_total_val_, hidden_size_);
            }
        } else {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", L);
            snprintf(name2, sizeof(name2), "model.language_model.layers.%d.self_attn.k_proj.weight", L);
            snprintf(name3, sizeof(name3), "model.language_model.layers.%d.self_attn.v_proj.weight", L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_3_blob(
                    blob_path(blob_dir, name), full_q_dim_,
                    blob_path(blob_dir, name2), full_kv_dim_,
                    blob_path(blob_dir, name3), full_kv_dim_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_3(
                    sf->get_bf16_ptr(name), full_q_dim_,
                    sf->get_bf16_ptr(name2), full_kv_dim_,
                    sf->get_bf16_ptr(name3), full_kv_dim_, hidden_size_);
            }
        }

        if (!ane_layers_[L].first_proj) {
            fprintf(stderr, "ANE first_proj compile failed for layer %d\n", L);
            return false;
        }

        int attn_dim;
        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.out_proj.weight", L);
            attn_dim = lin_total_val_;
        } else {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", L);
            attn_dim = full_out_dim_;
        }
        if (use_blobs) {
            ane_layers_[L].o_proj = ane_compile_matmul_blob(blob_path(blob_dir, name), hidden_size_, attn_dim);
        } else {
            ane_layers_[L].o_proj = ane_compile_matmul(sf->get_bf16_ptr(name), hidden_size_, attn_dim);
        }
        if (!ane_layers_[L].o_proj) {
            fprintf(stderr, "ANE o_proj compile failed for layer %d\n", L);
            return false;
        }

        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.language_model.layers.%d.mlp.up_proj.weight", L);
        snprintf(name3, sizeof(name3), "model.language_model.layers.%d.mlp.down_proj.weight", L);

        if (use_blobs) {
            ane_layers_[L].fused_ffn = ane_compile_fused_ffn_blob(
                blob_path(blob_dir, name), blob_path(blob_dir, name2),
                blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
        } else {
            ane_layers_[L].fused_ffn = ane_compile_fused_ffn(
                sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
        }
        if (!ane_layers_[L].fused_ffn) {
            fprintf(stderr, "ANE fused_ffn compile failed for layer %d\n", L);
            return false;
        }
    }

        int compiled = ane_compile_count();
    int cached = ane_cache_loads();
    LOG("  %d ANE layer kernels ready (compiled=%d, cached=%d)\n",
        compiled + cached, compiled, cached);

    // Pre-compile Metal weights (available for --use-metal mode)
    if (metal_available()) {
        compile_metal_weights(sf);
    }

    if (!compile_lm_head_ane(sf, blob_dir)) {
        LOG("ANE LM head disabled, falling back to CPU\n");
    } else {
        LOG("  LM head ANE enabled (%d chunks)\n", (int)lm_head_kernels_.size());
    }

    return true;
}

bool Qwen35Model::compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir) {
    bool use_blobs = !blob_dir.empty();

    const uint16_t* embed_bf16 = nullptr;
    if (!use_blobs) {
        embed_bf16 = sf->get_bf16_ptr("model.language_model.embed_tokens.weight");
        if (!embed_bf16) {
            fprintf(stderr, "ANE LM head: missing embed_tokens BF16 weights\n");
            return false;
        }
    }

    int chunk = lm_head_chunk_;
    if (chunk > vocab_size_) chunk = vocab_size_;

    int chunks = (vocab_size_ + chunk - 1) / chunk;
    lm_head_kernels_.resize(chunks, nullptr);

    LOG("  LM head ANE: compiling %d chunks (chunk=%d)\n", chunks, chunk);
    for (int c = 0; c < chunks; c++) {
        int offset = c * chunk;
        int rows = vocab_size_ - offset;
        if (rows > chunk) rows = chunk;

        LOG("    LM head chunk %d/%d...\r", c + 1, chunks);

        if (use_blobs) {
            embed_bf16 = sf->get_bf16_ptr("model.language_model.embed_tokens.weight");
            if (!embed_bf16) return false;
            const uint16_t* chunk_w = embed_bf16 + (int64_t)offset * hidden_size_;
            lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        } else {
            const uint16_t* chunk_w = embed_bf16 + (int64_t)offset * hidden_size_;
            lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        }
        if (!lm_head_kernels_[c]) {
            fprintf(stderr, "\nANE LM head: compile failed at chunk %d/%d\n", c + 1, chunks);
            free_lm_head_ane();
            // Auto-retry with half chunk size if ANE can't handle large chunks
            if (chunk > 4096) {
                lm_head_chunk_ = chunk / 2;
                LOG("  Retrying LM head with chunk=%d...\n", lm_head_chunk_);
                return compile_lm_head_ane(sf, blob_dir);
            }
            return false;
        }
    }
    LOG("    LM head chunk %d/%d done          \n", chunks, chunks);
    ane_lm_head_enabled_ = true;
    lm_head_chunk_ = chunk;
    return true;
}

void Qwen35Model::compile_metal_weights(ModelWeights* sf) {
    if (!metal_available()) return;
    for (int L = 0; L < num_layers_; L++) {
        char mn[256], mn2[256], mn3[256];
        // O projection
        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(mn, sizeof(mn), "model.language_model.layers.%d.linear_attn.out_proj.weight", L);
        } else {
            snprintf(mn, sizeof(mn), "model.language_model.layers.%d.self_attn.o_proj.weight", L);
        }
        const uint16_t* o_bf16 = sf ? sf->get_bf16_ptr(mn) : nullptr;
        int o_out = hidden_size_;
        int o_in = (layer_types_[L] == LayerType::LinearAttention) ? lin_total_val_ : full_out_dim_;
        if (o_bf16) {
            uint16_t* fp16 = (uint16_t*)malloc((size_t)o_out * o_in * 2);
            bf16_to_f16_vec(fp16, o_bf16, o_out * o_in);
            metal_layers_[L].o_proj = metal_create_weight(fp16, o_out, o_in);
        }
        // FFN
        snprintf(mn, sizeof(mn), "model.language_model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(mn2, sizeof(mn2), "model.language_model.layers.%d.mlp.up_proj.weight", L);
        snprintf(mn3, sizeof(mn3), "model.language_model.layers.%d.mlp.down_proj.weight", L);
        const uint16_t* g = sf ? sf->get_bf16_ptr(mn) : nullptr;
        const uint16_t* u = sf ? sf->get_bf16_ptr(mn2) : nullptr;
        const uint16_t* d = sf ? sf->get_bf16_ptr(mn3) : nullptr;
        if (g && u && d) {
            uint16_t* gf = (uint16_t*)malloc((size_t)intermediate_size_ * hidden_size_ * 2);
            uint16_t* uf = (uint16_t*)malloc((size_t)intermediate_size_ * hidden_size_ * 2);
            uint16_t* df = (uint16_t*)malloc((size_t)hidden_size_ * intermediate_size_ * 2);
            bf16_to_f16_vec(gf, g, intermediate_size_ * hidden_size_);
            bf16_to_f16_vec(uf, u, intermediate_size_ * hidden_size_);
            bf16_to_f16_vec(df, d, hidden_size_ * intermediate_size_);
            metal_layers_[L].gate_proj = metal_create_weight(gf, intermediate_size_, hidden_size_);
            metal_layers_[L].up_proj = metal_create_weight(uf, intermediate_size_, hidden_size_);
            metal_layers_[L].down_proj = metal_create_weight(df, hidden_size_, intermediate_size_);
        }
    }
    LOG("  Metal GPU weights created for %d layers\n", num_layers_);
}

void Qwen35Model::free_lm_head_ane() {
    for (auto* k : lm_head_kernels_) ane_free(k);
    lm_head_kernels_.clear();
    ane_lm_head_enabled_ = false;
}

bool Qwen35Model::forward_deltanet_core(int L, float* x, float* pre_oproj) {
    auto& dw = layers_[L].deltanet;
    auto& st = delta_states_[L];

    float* qkv_z = scratch_qkv_;
    if (!ane_matvec(ane_layers_[L].first_proj, qkv_z, x,
                    hidden_size_, lin_qkv_dim_ + lin_total_val_)) {
        fprintf(stderr, "ANE first_proj eval failed at layer %d (DeltaNet)\n", L);
        return false;
    }

    float* mixed_qkv = qkv_z;
    float* z = qkv_z + lin_qkv_dim_;

    float* a_vec = scratch_tmp_;
    float* b_vec = scratch_tmp_ + lin_num_val_heads_;
    matvec(a_vec, dw.in_proj_a, x, lin_num_val_heads_, hidden_size_);
    matvec(b_vec, dw.in_proj_b, x, lin_num_val_heads_, hidden_size_);

    float* conv_out = scratch_conv_;
    conv1d_update(conv_out, st.conv_state, &st.conv_pos, mixed_qkv, dw.conv1d_w, lin_qkv_dim_, conv_kernel_);
    silu_vec_inplace(conv_out, lin_qkv_dim_, scratch_tmp_ + lin_num_val_heads_ * 2);

    float* Q = conv_out;
    float* K = conv_out + lin_total_key_;
    float* V = conv_out + lin_total_key_ * 2;

    float* y = scratch_y_;
    float q_scale = 1.0f / sqrtf((float)lin_key_dim_);
    int val_heads_per_key = lin_num_val_heads_ / lin_num_heads_;

    for (int kh = 0; kh < lin_num_heads_; kh++) {
        float* qh = Q + kh * lin_key_dim_;
        float* kh_ptr = K + kh * lin_key_dim_;

        l2_normalize(qh, lin_key_dim_);
        l2_normalize(kh_ptr, lin_key_dim_);
        float qs = q_scale;
        vDSP_vsmul(qh, 1, &qs, qh, 1, (vDSP_Length)lin_key_dim_);

        for (int vsub = 0; vsub < val_heads_per_key; vsub++) {
            int vh = kh * val_heads_per_key + vsub;
            float* vh_ptr = V + vh * lin_val_dim_;
            float* yh = y + vh * lin_val_dim_;
            float* state = st.ssm_state + vh * lin_key_dim_ * lin_val_dim_;

            float beta = sigmoid_f(b_vec[vh]);
            float decay = expf(-dw.A[vh] * softplus_f(a_vec[vh] + dw.dt_bias[vh]));
            ssm_step(yh, state, qh, kh_ptr, vh_ptr, decay, beta, lin_key_dim_, lin_val_dim_);
        }
    }

    for (int h = 0; h < lin_num_val_heads_; h++) {
        rmsnorm_gated(pre_oproj + h * lin_val_dim_,
                      y + h * lin_val_dim_,
                      z + h * lin_val_dim_,
                      dw.norm_w, lin_val_dim_);
    }
    return true;
}

bool Qwen35Model::forward_full_attn_core(int L, float* x, float* pre_oproj, int pos) {
    auto& fw = layers_[L].full_attn;
    auto& cache = kv_caches_[L];

    float* qkv_buf = scratch_qkv_;
    if (!ane_matvec(ane_layers_[L].first_proj, qkv_buf, x,
                    hidden_size_, full_q_dim_ + full_kv_dim_ * 2)) {
        fprintf(stderr, "ANE first_proj eval failed at layer %d (FullAttn)\n", L);
        return false;
    }

    float* q_gate_raw = qkv_buf;
    float* k_raw = qkv_buf + full_q_dim_;
    float* v_raw = qkv_buf + full_q_dim_ + full_kv_dim_;

    // ============ PATCH 6: Parallel Q/K per-head RMSNorm via dispatch_apply ============
    int nqh = num_q_heads_, nkh = num_kv_heads_;
    float eps = rms_eps_;
    float* q_n = fw.q_norm;
    float* k_n = fw.k_norm;
    int hd = head_dim_;
    // Note: q_gate_raw stride is head_dim*2 (interleaved with gate)
    dispatch_apply(nqh + nkh,
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t h) {
            if (h < (size_t)nqh) {
                float* qh = q_gate_raw + h * hd * 2;  // stride=head_dim*2 for q+gate
                rmsnorm(qh, qh, q_n, hd, eps);
            } else {
                float* kh = k_raw + (h - nqh) * hd;
                rmsnorm(kh, kh, k_n, hd, eps);
            }
        });

    const float* rope_cos_row = nullptr;
    const float* rope_sin_row = nullptr;
    if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
    }
    apply_rope_cached(q_gate_raw, k_raw, num_q_heads_, num_kv_heads_,
                      head_dim_, head_dim_ * 2, head_dim_, rot_dim_, pos, rope_theta_,
                      rope_cos_row, rope_sin_row);

    int slot;
    if (cache.len < cache.capacity) {
        slot = cache.start + cache.len;
        if (slot >= cache.capacity) slot -= cache.capacity;
        cache.len++;
    } else {
        slot = cache.start;
        cache.start++;
        if (cache.start >= cache.capacity) cache.start = 0;
    }
    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    // fp32 -> fp16 on write (halves KV cache memory bandwidth during attention)
    {
        uint16_t* kslot = cache.k_cache + (size_t)slot * kv_stride;
        uint16_t* vslot = cache.v_cache + (size_t)slot * kv_stride;
#if defined(__aarch64__) || defined(__arm64__)
        for (size_t _i = 0; _i < kv_stride; _i++) {
            ((__fp16*)kslot)[_i] = (__fp16)k_raw[_i];
            ((__fp16*)vslot)[_i] = (__fp16)v_raw[_i];
        }
#else
        for (size_t _i = 0; _i < kv_stride; _i++) {
            kslot[_i] = f32_to_f16(k_raw[_i]);
            vslot[_i] = f32_to_f16(v_raw[_i]);
        }
#endif
    }

    // fp16 -> fp32 for gqa_attention (convert entire active region)
    {
        size_t active_kv = (size_t)cache.len * num_kv_heads_ * head_dim_;
        float* k_f32 = (float*)alloca(active_kv * sizeof(float));
        float* v_f32 = (float*)alloca(active_kv * sizeof(float));
        // Copy from ring buffer, unwrapping if needed
        int first_span = cache.capacity - cache.start;
        if (first_span > cache.len) first_span = cache.len;
        int second_span = cache.len - first_span;
        size_t kv_row = (size_t)num_kv_heads_ * head_dim_;
        size_t fs = (size_t)first_span * kv_row;
#if defined(__aarch64__) || defined(__arm64__)
        const __fp16* ksrc = (const __fp16*)cache.k_cache;
        const __fp16* vsrc = (const __fp16*)cache.v_cache;
        for (size_t _i = 0; _i < fs; _i++) {
            k_f32[_i] = (float)ksrc[(size_t)cache.start * kv_row + _i];
            v_f32[_i] = (float)vsrc[(size_t)cache.start * kv_row + _i];
        }
        for (size_t _i = 0; _i < (size_t)second_span * kv_row; _i++) {
            k_f32[fs + _i] = (float)ksrc[_i];
            v_f32[fs + _i] = (float)vsrc[_i];
        }
#else
        for (size_t _i = 0; _i < fs; _i++) {
            k_f32[_i] = f16_to_f32(cache.k_cache[(size_t)cache.start * kv_row + _i]);
            v_f32[_i] = f16_to_f32(cache.v_cache[(size_t)cache.start * kv_row + _i]);
        }
        for (size_t _i = 0; _i < (size_t)second_span * kv_row; _i++) {
            k_f32[fs + _i] = f16_to_f32(cache.k_cache[_i]);
            v_f32[fs + _i] = f16_to_f32(cache.v_cache[_i]);
        }
#endif
        // Use Metal GPU for attention when available (faster for long contexts)
        if (metal_available() && cache.len >= 64) {
            // Metal path: operates directly on fp16 KV cache (no conversion needed!)
            metal_gqa_attention(pre_oproj, q_gate_raw,
                                cache.k_cache, cache.v_cache,
                                num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                                cache.start, cache.len, cache.capacity);
        } else {
            // CPU fallback with fp32 converted cache
            gqa_attention(pre_oproj, q_gate_raw, k_f32, v_f32,
                          num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                          0, cache.len, cache.len);
        }
    }

    if (attn_output_gate_) {
        for (int h = 0; h < num_q_heads_; h++) {
            float* oh = pre_oproj + h * head_dim_;
            const float* gh = q_gate_raw + (size_t)h * head_dim_ * 2 + head_dim_;
            mul_sigmoid_inplace(oh, gh, head_dim_, scratch_tmp_);
        }
    }
    return true;
}

float* Qwen35Model::forward(int token, int pos) {
    memcpy(x_, embed_tokens_ + (int64_t)token * hidden_size_, hidden_size_ * sizeof(float));

    float* pre_oproj = scratch_attn_;

    for (int L = 0; L < num_layers_; L++) {
        rmsnorm(x_norm_, x_, layers_[L].input_layernorm, hidden_size_, rms_eps_);

        if (layer_types_[L] == LayerType::LinearAttention) {
            if (!forward_deltanet_core(L, x_norm_, pre_oproj)) return nullptr;
        } else {
            if (!forward_full_attn_core(L, x_norm_, pre_oproj, pos)) return nullptr;
        }

        int attn_dim = (layer_types_[L] == LayerType::LinearAttention) ? lin_total_val_ : full_out_dim_;
        float* attn_out = x_norm_;
        if (use_metal_matmul_ && metal_layers_[L].o_proj) {
            metal_matvec(metal_layers_[L].o_proj, attn_out, pre_oproj);
        } else if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, attn_dim, hidden_size_)) {
            fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
            return nullptr;
        }

        // ============ PATCH 2: vDSP_vadd for residual (was scalar loop) ============
        vDSP_vadd(x_, 1, attn_out, 1, x_, 1, (vDSP_Length)hidden_size_);

        rmsnorm(x_norm_, x_, layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);

        float* mlp_out = scratch_attn_;
        if (use_metal_matmul_ && metal_layers_[L].gate_proj) {
            metal_fused_ffn(metal_layers_[L].gate_proj, metal_layers_[L].up_proj,
                           metal_layers_[L].down_proj, mlp_out, x_norm_,
                           hidden_size_, intermediate_size_, hidden_size_);
        } else if (!ane_matvec(ane_layers_[L].fused_ffn, mlp_out, x_norm_, hidden_size_, hidden_size_)) {
            fprintf(stderr, "ANE fused_ffn eval failed at layer %d\n", L);
            return nullptr;
        }

        // ============ PATCH 2: vDSP_vadd for residual (was scalar loop) ============
        vDSP_vadd(x_, 1, mlp_out, 1, x_, 1, (vDSP_Length)hidden_size_);
    }

    rmsnorm(x_, x_, final_norm_, hidden_size_, rms_eps_);

    // Skip LM head during prefill_step (saves ~10 ANE dispatches per token)
    if (skip_lm_head_) { skip_lm_head_ = false; return logits_; }

    // ============ PATCH 4: Parallel LM head chunks via dispatch_apply ============
    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        int chunks = (int)lm_head_kernels_.size();
        std::atomic<bool> lm_ok{true};
        auto* lm_ok_ptr = &lm_ok;
        int chunk_sz = lm_head_chunk_;
        int vocab = vocab_size_;
        float* x_ptr = x_;
        int hsz = hidden_size_;
        float* logits_ptr = logits_;
        dispatch_apply(chunks,
            dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
            ^(size_t c) {
                if (!lm_ok_ptr->load(std::memory_order_relaxed)) return;
                int offset = (int)c * chunk_sz;
                int rows = vocab - offset;
                if (rows > chunk_sz) rows = chunk_sz;
                if (!ane_matvec(lm_head_kernels_[c],
                                logits_ptr + offset, x_ptr, hsz, rows)) {
                    lm_ok_ptr->store(false, std::memory_order_relaxed);
                }
            });
        if (!lm_ok.load()) {
            free_lm_head_ane();
            matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
        }
    } else {
        matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
    }

    return logits_;
}

void Qwen35Model::set_use_metal(bool v) {
    use_metal_matmul_ = v;
    if (v && metal_layers_.empty()) {
        // Need to initialize Metal weight buffers - but ModelWeights is gone
        // The weights will be compiled lazily if sf is available during load
        LOG("Metal mode enabled (weights will be compiled if available)\n");
    }
}

bool Qwen35Model::prefill_step(int token_id, int pos) {
    skip_lm_head_ = true;
    return forward(token_id, pos) != nullptr;
}

} // namespace ane_lm
