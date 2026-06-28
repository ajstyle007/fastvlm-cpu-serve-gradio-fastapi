// fastvlm_server_v2.c
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAX_NEW_TOKENS 512
#define MAX_PROMPT_LEN 4096
#define MAX_PATH_LEN   1024

float* load_embeddings(const char* path, int* n_tokens, int* n_embd) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[server] Cannot open embeddings: %s\n", path);
        return NULL;
    }
    fread(n_tokens, sizeof(int32_t), 1, f);
    fread(n_embd,   sizeof(int32_t), 1, f);
    float* data = malloc((*n_tokens) * (*n_embd) * sizeof(float));
    fread(data, sizeof(float), (*n_tokens) * (*n_embd), f);
    fclose(f);
    return data;
}

llama_token* tokenize_str(const struct llama_vocab* vocab,
                          const char* text, int* n_out) {
    int n = -llama_tokenize(vocab, text, strlen(text), NULL, 0, false, true);
    if (n <= 0) { *n_out = 0; return NULL; }
    llama_token* toks = malloc(n * sizeof(llama_token));
    llama_tokenize(vocab, text, strlen(text), toks, n, false, true);
    *n_out = n;
    return toks;
}

int run_inference(
    struct llama_context* ctx,
    const struct llama_vocab* vocab,
    const char* embd_path,
    const char* user_prompt,
    int seq_id          // unique sequence slot for this request
) {
    clock_t t0 = clock();

    // Load embeddings
    int n_img, n_embd;
    float* img_embd = load_embeddings(embd_path, &n_img, &n_embd);
    if (!img_embd) return -1;

    // Build prompt parts
    char prefix[1024];
    snprintf(prefix, sizeof(prefix),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
    );
    char suffix[MAX_PROMPT_LEN];
    snprintf(suffix, sizeof(suffix),
        "\n%s<|im_end|>\n<|im_start|>assistant\n",
        user_prompt
    );

    int n_prefix, n_suffix;
    llama_token* prefix_toks = tokenize_str(vocab, prefix, &n_prefix);
    llama_token* suffix_toks = tokenize_str(vocab, suffix, &n_suffix);

    int total_input = n_prefix + n_img + n_suffix;
    fprintf(stderr, "[seq %d] prefix=%d img=%d suffix=%d total=%d\n",
            seq_id, n_prefix, n_img, n_suffix, total_input);

    int cur_pos = 0;

    // ── Batch 1: prefix ───────────────────────────────────────────────────────
    {
        struct llama_batch b = llama_batch_init(n_prefix, 0, 1);
        b.n_tokens = n_prefix;
        for (int i = 0; i < n_prefix; i++) {
            b.token[i]     = prefix_toks[i];
            b.pos[i]       = cur_pos++;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq_id;   // ← use this request's seq slot
            b.logits[i]    = 0;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "[seq %d] Failed prefix decode\n", seq_id);
            llama_batch_free(b);
            goto cleanup;
        }
        llama_batch_free(b);
    }

    // ── Batch 2: image embeddings ─────────────────────────────────────────────
    {
        struct llama_batch b = llama_batch_init(n_img, n_embd, 1);
        b.n_tokens = n_img;
        for (int i = 0; i < n_img; i++) {
            memcpy(b.embd + i * n_embd,
                   img_embd + i * n_embd,
                   n_embd * sizeof(float));
            b.pos[i]       = cur_pos++;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq_id;
            b.logits[i]    = 0;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "[seq %d] Failed image decode\n", seq_id);
            llama_batch_free(b);
            goto cleanup;
        }
        llama_batch_free(b);
    }

    // ── Batch 3: suffix ───────────────────────────────────────────────────────
    {
        struct llama_batch b = llama_batch_init(n_suffix, 0, 1);
        b.n_tokens = n_suffix;
        for (int i = 0; i < n_suffix; i++) {
            b.token[i]     = suffix_toks[i];
            b.pos[i]       = cur_pos++;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq_id;
            b.logits[i]    = (i == n_suffix - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "[seq %d] Failed suffix decode\n", seq_id);
            llama_batch_free(b);
            goto cleanup;
        }
        llama_batch_free(b);
    }

    clock_t t1 = clock();
    fprintf(stderr, "[seq %d] Prefill done in %.2fs. Generating...\n",
            seq_id, (double)(t1-t0)/CLOCKS_PER_SEC);

    // Stop tokens
    llama_token eos = llama_vocab_eos(vocab);
    int n_tmp;
    llama_token* tmp = tokenize_str(vocab, "<|im_end|>", &n_tmp);
    llama_token im_end = tmp[0];
    free(tmp);

    // Sampler
    struct llama_sampler* sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // ── Generation loop ───────────────────────────────────────────────────────
    int n_generated = 0;
    for (int i = 0; i < MAX_NEW_TOKENS; i++) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (tok == eos || tok == im_end) break;

        char piece[128] = {0};
        llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
        printf("%s", piece);
        fflush(stdout);
        n_generated++;

        llama_sampler_accept(sampler, tok);

        struct llama_batch next = llama_batch_init(1, 0, 1);
        next.n_tokens      = 1;
        next.token[0]      = tok;
        next.pos[0]        = cur_pos++;
        next.n_seq_id[0]   = 1;
        next.seq_id[0][0]  = seq_id;
        next.logits[0]     = 1;
        if (llama_decode(ctx, next) != 0) {
            llama_batch_free(next);
            break;
        }
        llama_batch_free(next);
    }

    clock_t t2 = clock();
    fprintf(stderr, "\n[seq %d] Generated %d tokens in %.2fs (%.1f t/s)\n",
            seq_id, n_generated,
            (double)(t2-t1)/CLOCKS_PER_SEC,
            n_generated / ((double)(t2-t1)/CLOCKS_PER_SEC));

    // End sentinel
    printf("\n---END---\n");
    fflush(stdout);

    llama_sampler_free(sampler);

    // ── Free only THIS request's KV entries ───────────────────────────────────
    // This frees the seq_id slot so it can be reused
    // Other sequences (future requests reusing slot 0 etc.) are unaffected
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
    fprintf(stderr, "[seq %d] KV slot freed\n", seq_id);

    free(img_embd);
    free(prefix_toks);
    free(suffix_toks);
    return 0;

cleanup:
    printf("\n---END---\n");
    fflush(stdout);
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
    free(img_embd);
    if (prefix_toks) free(prefix_toks);
    if (suffix_toks) free(suffix_toks);
    return -1;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "[server] Loading: %s\n", argv[1]);
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    struct llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "[server] Load failed\n"); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 4096;  // larger ctx to handle multiple concurrent requests
    cp.n_batch  = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 4;    // support up to 4 concurrent sequence slots
    struct llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "[server] Context failed\n"); return 1; }

    const struct llama_vocab* vocab = llama_model_get_vocab(model);

    fprintf(stderr, "[server] READY\n");
    fflush(stderr);

    // ── Request loop ──────────────────────────────────────────────────────────
    char embd_path[MAX_PATH_LEN];
    char user_prompt[MAX_PROMPT_LEN];
    int  seq_id = 0;   // cycle through sequence slots 0-3

    while (1) {
        if (!fgets(embd_path,   sizeof(embd_path),   stdin)) break;
        embd_path[strcspn(embd_path, "\n")] = 0;
        if (strlen(embd_path) == 0) continue;

        if (!fgets(user_prompt, sizeof(user_prompt), stdin)) break;
        user_prompt[strcspn(user_prompt, "\n")] = 0;

        run_inference(ctx, vocab, embd_path, user_prompt, seq_id);

        // Cycle to next slot (0,1,2,3,0,1,2,3...)
        seq_id = (seq_id + 1) % 4;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}