#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "utilities.h"

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void element_wise_mul(float *x, float *y, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] * y[i];
    }
}

void normalize(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

long time_in_ms() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec * 1000 + time.tv_usec / 1000;
}

void silu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] * (1.0f / (1.0f + expf(-x[i])));
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

void position_embedding(float *q, float *k, LLMWeight * w, int pos, int dim, int n_heads) {
    int head_size = dim / n_heads;
    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;
    for (int h = 0; h < n_heads; h++) {
        // get the q and k vectors for this head
        float* q = q + h * head_size;
        float* k = k + h * head_size;
        // rotate q and k by the freq_cis_real and freq_cis_imag
        for (int i = 0; i < head_size; i+=2) {
            float q0 = q[0], q1 = q[1];
            float k0 = k[0], k1 = k[1];
            float fcr = freq_cis_real_row[i/2];
            float fci = freq_cis_imag_row[i/2];
            q[i]  = q0 * fcr - q1 * fci;
            q[i+1] = q0 * fci + q1 * fcr;
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
    }
}

void attention(int layer, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    int head_size = p->dim / p->n_heads, layer_offset = layer * p->seq_len * p->dim;;
    // multihead attention. iterate over all heads
    for (int h = 0; h < p->n_heads; h++) {
        // get the query vector for this head
        float* q = s->q + h * head_size;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = s->key_cache + layer_offset + t * p->dim + h * head_size;
            // attention score is dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            // scaled by sqrt(head_size) to avoid floating number overflow
            s->att[t] = score / sqrtf(head_size);
        }
        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(s->att, pos + 1);    
        // weighted sum of the values
        for (int i = 0; i < head_size; i++) {
            float val = 0.0f;
            for (int t = 0; t <= pos; t++) {
                val += s->att[t] * s->value_cache[layer_offset + t * p->dim + h * head_size + i];
            }
            s->xb[h * head_size + i] = val;
        }
    }
}

void key_value_cache(int layer, int pos, LLMConfig* p, LLMRuntime* s) {
    int dim = p->dim, layer_offset = layer * p->seq_len * dim; // kv cache layer offset for convenience
    memcpy(s->key_cache + layer_offset + pos * dim, s->k, dim * sizeof(float));
    memcpy(s->value_cache + layer_offset + pos * dim, s->v, dim * sizeof(float));
}

void malloc_LLMRuntime(LLMRuntime* s, LLMConfig* p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = (float *) calloc(p->dim, sizeof(float));
    s->xb = (float *) calloc(p->dim, sizeof(float));
    s->xb2 = (float *) calloc(p->dim, sizeof(float));
    s->h1 = (float *) calloc(p->hidden_dim, sizeof(float));
    s->h2 = (float *) calloc(p->hidden_dim, sizeof(float));
    s->q = (float *) calloc(p->dim, sizeof(float));
    s->k = (float *) calloc(p->dim, sizeof(float));
    s->v = (float *) calloc(p->dim, sizeof(float));
    s->att = (float *) calloc(p->seq_len, sizeof(float));
    s->logits = (float *) calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float *) calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = (float *) calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->h1 || !s->h2 || !s->q 
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache 
     || !s->value_cache) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_LLMRuntime(LLMRuntime* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->h1);
    free(s->h2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void malloc_LLMWeight(LLMWeight* w, LLMConfig* p) {
    // we calloc instead of malloc to keep valgrind happy
    w->token_embedding_table = (float *) calloc(p->vocab_size * p->dim, sizeof(float));
    w->rms_att_weight = (float *) calloc(p->n_layers * p->dim, sizeof(float));
    w->rms_ffn_weight = (float *) calloc(p->n_layers * p->dim, sizeof(float));
    w->wq = (float *) calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wk = (float *) calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wv = (float *) calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wo = (float *) calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->w1 = (float *) calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    w->w2 = (float *) calloc(p->n_layers * p->dim * p->hidden_dim, sizeof(float));
    w->w3 = (float *) calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    w->rms_final_weight = (float *) calloc(p->dim, sizeof(float));
    w->freq_cis_real = (float *) calloc(p->seq_len * p->dim / 2, sizeof(float));
    w->freq_cis_imag = (float *) calloc(p->seq_len * p->dim / 2, sizeof(float));
    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight 
     || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 || 
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_LLMWeight(LLMWeight* w) {
    free(w->token_embedding_table);
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->rms_final_weight);
    free(w->freq_cis_real);
    free(w->freq_cis_imag);
}

int load_LLMWeight(LLMWeight *w, LLMConfig* p, FILE* f) {
    if (fread(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, f) != p->vocab_size * p->dim) return 1;
    if (fread(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, f) != p->n_layers * p->dim) return 1;
    if (fread(w->wq, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    if (fread(w->wk, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    if (fread(w->wv, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    if (fread(w->wo, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    if (fread(w->rms_ffn_weight, sizeof(float), p->n_layers * p->dim, f) != p->n_layers * p->dim) return 1;
    if (fread(w->w1, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != p->n_layers * p->dim * p->hidden_dim) return 1;
    if (fread(w->w2, sizeof(float), p->n_layers * p->hidden_dim * p->dim, f) != p->n_layers * p->hidden_dim * p->dim) return 1;
    if (fread(w->w3, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != p->n_layers * p->dim * p->hidden_dim) return 1;
    if (fread(w->rms_final_weight, sizeof(float), p->dim, f) != p->dim) return 1;
    int head_size = p->dim / p->n_heads;
    if (fread(w->freq_cis_real, sizeof(float), p->seq_len * head_size / 2, f) != p->seq_len * head_size / 2) return 1;
    if (fread(w->freq_cis_imag, sizeof(float), p->seq_len * head_size / 2, f) != p->seq_len * head_size / 2) return 1;
    return 0;
}

int load_LLM_Config_Weight(LLMConfig *config, LLMWeight *weights) {
    FILE *file = fopen("model.bin", "rb");
    if (!file) {
        printf("Unable to open the model file!\n");
        return 1;
    }
    if (fread(config, sizeof(LLMConfig), 1, file) != 1) { return 1; }
    malloc_LLMWeight(weights, config);
    if (load_LLMWeight(weights, config, file)) { return 1; }
    fclose(file);
    return 0;
}

int load_tokenizer(char **vocab, int vocab_size) {
    FILE *file = fopen("tokenizer.bin", "rb");
    if (!file) {
        printf("Unable to open the tokenizer file\n");
        return 1;
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) { 
            printf("Can not read length\n");
            return 1;
        }
        vocab[i] = (char *) malloc(len + 1);
        if (fread(vocab[i], len, 1, file) != 1) { 
            printf("Length doesn't match\n");
            return 1; 
        }
        vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
    return 0;
}