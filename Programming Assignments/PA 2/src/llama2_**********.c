/*
PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: llama2_**********
* NAME: Shaheer Ziya
* UID : **********
* Development Platform: Ubuntu 20.04.6 LTS (GNU/Linux 5.4.0-155-generic x86_64)
* Remark: 
    The provided code performs matrix multiplication by reusing multiple thread and its output agrees
    with the sequential version provided to us. All requirements in assignment met.
    
    It should be noted that the assignment of rows to threads as mentioned in the assignment makes the assumption
    that numRows >> numThreads. The logic breaks down for example with 5 threads working on a matrix consisting
    of 6 rows. In this case ⌈6/5⌉ * (5-1) = 2 * 4 = 8. Thus the total number of rows assigned to the first 4 
    (or n-1) threads already exceeds the total number of rows. Although better logic can be implemented (in 
    my humble opinion) and some math can be done to calculate the values for which this logic does indeed work
    robustly, the reward far outweighs the time investment. If only there was some incentive...

* How to compile: (gcc -o llama2_3035946760 llama2_3035946760.c utilities.c -O2 -pthread -lm)

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthread` to link library:
$ gcc -o llama2_3035946760 llama2_3035946760.c utilities.c -O2 -pthread -lm

Then Run with:
$ ./llama2_3035946760 <seed> <thr_count>
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
 * A sequential version is provided in seq.c, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// Addtional Header File Here
#include <pthread.h>

// Global Variables
struct rusage main_usage;        // get usage for main thread

// Struct to hold the parameters for each thread
typedef struct {
    pthread_t thr;
    int haveWork;
    struct rusage usage;
} matThread;

typedef struct {
    float* out;
    float* vec;
    float* mat;
    int col;
    int row;
} thrParameters;

pthread_mutex_t mutex;
pthread_cond_t state;
pthread_cond_t mainSleep;

int threadsRunning;
int thrCount;

matThread* thrArr;
thrParameters thrParam;


/* Sleep upon initialization and work upon being woken up. Continue working until signalled to stop,
    after which collect resource usage and terminate */
void* thr_func(void* arg) {
    // Get the id of this thread and free the memory assigned on the heap
    int thrNum = *((int *)arg);
    free(arg);

    while (thrArr[thrNum].haveWork) { // Exit loop when no more work for thread left (i.e. thrArr[i].haveWork = 0)
        // Go to sleep (and wait to be awoken)
        pthread_mutex_lock(&mutex);
        threadsRunning--;
        if (threadsRunning == 0) { // Wake the "main" thread if no more threads need to be put to sleep
            pthread_cond_signal(&mainSleep); 
        }
        pthread_cond_wait(&state, &mutex);
        pthread_mutex_unlock(&mutex);

        if (thrArr[thrNum].haveWork) { // If work left to do, get it done; Check essential for last iteration
            int startRow, endRow;
            // Reading thrCount is fine since no writers to thrCount
            if (thrParam.row % thrCount == 0) { // Case 1: The number of rows are divisible by the number of threads
                startRow = thrNum * (thrParam.row / thrCount);
                endRow = (thrNum + 1) * (thrParam.row / thrCount); 
            } else { // Case 2: The number of rows is NOT divisible by the number of threads
                // In this case we make the assumption numRows >> numThreads, otherwise the behaviour is ill-defined
                startRow = thrNum * ((thrParam.row / thrCount) + 1);
                if (thrNum == thrCount - 1) {
                    endRow = thrParam.row + 1;
                } else {
                    endRow = startRow + (thrParam.row / thrCount) + 1;
                }
            }

            // Do the multiplication for startRow to (endRow - 1)
            for (int i = startRow; i < endRow; i++) { // Final row is [(k+1) x d/n] - 1
                float val = 0.0f;
                for (int j = 0; j < thrParam.col; j++) {
                    val += thrParam.mat[i * thrParam.col + j] * thrParam.vec[j]; // mat[i * col + j] := mat[i][j]
                } thrParam.out[i] = val;
            }
        } // Else continue
    }

    // Collect and save usage on this thread, then terminate
    int ret = getrusage(RUSAGE_THREAD, &(thrArr[thrNum].usage));
    if (ret != 0) {
        perror("Error: Failed to get resource usage\n");
    }

    // Terminate the Thread
    pthread_exit(NULL);
}

/* Creates "thr_count" worker threads and waits until all of them have been put to sleep. 
    Place each thread into thrArr for access in other functions */
int init_mat_vec_mul(int thr_count) { 
    if (thr_count <= 0) {
        perror("Error: The number of child threads must be positive!"); exit(EXIT_FAILURE);
    }
    
    // Initialise the lock and condition variables that enable us to block/wake the threads
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&state, NULL);
    pthread_cond_init(&mainSleep, NULL);
    
    // Initialise the array to hold the thread and stop condition
    thrArr = malloc(thr_count * sizeof(matThread));
    if (thrArr == NULL) {
        perror("Error: Couldn't allocate memory for thrArr\n"); return(EXIT_FAILURE);
    }

    // Initialise the number of thrCount (Globally)
    thrCount = thr_count;

    // Create "thr_count" worker threads, pass in its id to thr_func and store thread info in thrArr
    for (int i = 0; i < thr_count; i++) {
        // Allocate memory on the heap to tell the thread its number (Freed in the i-th thread)
        int *thrNum = malloc(sizeof(thrNum));
        if (thrNum == NULL) { // Check if malloc failed
            perror("Error: Couldn't allocate memory for thrNum\n"); return(EXIT_FAILURE);
        }
        
        *thrNum = i; // Store the value of the iteration variable
        if (pthread_create(&(thrArr[i].thr), NULL, thr_func, thrNum) != 0) { // Check if thread creation failed
            perror("Error: Thread creation unsuccessful\n"); return(EXIT_FAILURE);
        } threadsRunning++;
        // Tell each thread that it has work to do
        thrArr[i].haveWork = 1;
    }
    
    // Have this thread wait until all child threads have been put to sleep
    pthread_mutex_lock(&mutex);
    while (threadsRunning > 0) {
        pthread_cond_wait(&mainSleep, &mutex);
    } pthread_mutex_unlock(&mutex);

    return 0;
}


/* Change the paramters, wake the threads up to perform the computation and wait until they're done */
void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    // Assign the parameters to the paramStruct, no race condition since all other threads are sleep
    thrParam.out = out;
    thrParam.vec = vec;
    thrParam.mat = mat;
    thrParam.col = col;
    thrParam.row = row;

    // Wake up all the threads and wait for all of them to finish
    threadsRunning = thrCount;
    pthread_cond_broadcast(&state);

    pthread_mutex_lock(&mutex);
    while (threadsRunning > 0) {
        pthread_cond_wait(&mainSleep, &mutex);
    } pthread_mutex_unlock(&mutex);
}

/* Wake the threads up, inform them to collect usage and wait for all threads to terminate. Print threads and main usage and
    clear all initialized variables not needed anymore */
int close_mat_vec_mul() {
    // Inform the threads that all the work is done
    for (int i = 0; i < thrCount; i++) { 
        thrArr[i].haveWork = 0;
    }

    // Wake up all the threads and wait for all of them to finish
    threadsRunning = thrCount;
    pthread_cond_broadcast(&state);

    for (int i = 0; i < thrCount; i++) {
        pthread_join(thrArr[i].thr, NULL);
    }

    // Collect and print system usage for all threads (including main)
    for (int i = 0; i < thrCount; i++) {
        printf("Thread %d has completed - user: %.4f s, system %.4f s\n", i, 
        thrArr[i].usage.ru_utime.tv_sec + thrArr[i].usage.ru_utime.tv_usec / 1000000.0 ,
        thrArr[i].usage.ru_stime.tv_sec + thrArr[i].usage.ru_stime.tv_usec / 1000000.0
        );
    }

    int ret = getrusage(RUSAGE_SELF, &main_usage);
    if (ret != 0) {
        perror("Error: Failed to get resource usage on main");
    }

    printf("Main Thread - user: %ld.%04ld s, system: %ld.%04ld s\n",
        main_usage.ru_utime.tv_sec, main_usage.ru_utime.tv_usec,
        main_usage.ru_stime.tv_sec, main_usage.ru_stime.tv_usec 
    );

    free(thrArr);
    
    // Destroy mutex and CVs
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&state);
    pthread_cond_destroy(&mainSleep);

    return 0;
}

// YOUR CODE ENDS HERE

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
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

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

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}