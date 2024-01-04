#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef struct {
    // table for token embedding
    float* token_embedding_table; // (vocab_size, dim), token idx -> token embedding
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // weights for normalizations
    float* rms_att_weight; // (layer, dim)
    float* rms_ffn_weight; // (layer, dim)
    float* rms_final_weight; // (dim,)
    // weights for positional embedding
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
} LLMWeight;

/**
 * Runtime State of the current inference
*/
typedef struct {
    // Important Runtime States
    float* x; // current state (dim,)
    float* q; // query (dim,)
    float* k; // key (dim,)
    float* v; // value (dim,)
    float* att; // attention scores (seq_len,)
    float* logits; // output probability distribution of tokens  (vocab_size,)
    // Additional Buffers for Convenience
    float* xb; // same, but inside a residual branch (dim,)
    float* xb2; // an additional buffer just for convenience (dim,)
    float* h1; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* h2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* key_cache;   // key cache of previous word (layer, seq_len, dim)
    float* value_cache; // value cache of previous word (layer, seq_len, dim)
} LLMRuntime;

typedef struct {
    int dim; // transformer feature dimension
    int hidden_dim; // hidden dimension of ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} LLMConfig;

void malloc_LLMRuntime(LLMRuntime *s, LLMConfig *p);
void free_LLMRuntime(LLMRuntime *s);
void malloc_LLMWeight(LLMWeight *w, LLMConfig *p);
void free_LLMWeight(LLMWeight *w);
int load_LLMWeight(LLMWeight *w, LLMConfig *p, FILE *f);
int load_LLM_Config_Weight(LLMConfig *p, LLMWeight *w);
int load_tokenizer(char **vocab, int vocab_size);
void accum(float *a, float *b, int size);
long time_in_ms();

void normalize(float *o, float *x, float *weight, int size);
void element_wise_mul(float *x, float *y, int size);
void softmax(float *x, int size);
void silu(float *x, int size);
int sample(float *probabilities, int n);
void position_embedding(float *k, float *v, LLMWeight *w, int pos, int dim, int n_heads); // Position Embedding
void attention(int layer, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w);
void key_value_cache(int layer, int pos, LLMConfig* p, LLMRuntime* s);