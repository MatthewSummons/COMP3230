/*
Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

Include utilities.c in compilation:
$ gcc -o seq seq.c utilities.c -O2 -lm

Then Run with:
$ ./seq <seed>
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-Vector Multiplication, used in Attention and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each row, so we can use Multi-Threading for acceleration.
 * 
 * Please use <pthread.h> and your favorite synchronization method,
 * semaphore / mutex lock + conditional variable
 * 
 * A sequential version is provided below.
*/

// Global Variable
struct rusage main_usage;        // get usage for main thread

// Sequential Implementation Starts Here

void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    for (int i = 0; i < row; i++) {
        float val = 0.0f;
        for (int j = 0; j < col; j++) {
            val += mat[i * col + j] * vec[j];
        }
        out[i] = val;
    }
}

// Sequential Implementation Ends Here

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;

    if (argc == 2) {
        seed = atoi(argv[1]);
    } else {
        printf("Usage: ./llama2_seq <seed>\n");
        return 1;
    }

    srand(seed);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // print out self usage information
    getrusage(RUSAGE_SELF, &main_usage);
    printf("main thread - user: %.4f s, system: %.4f s\n",
    (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec/1000000.0),
    (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec/1000000.0));

    // cleanup
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}