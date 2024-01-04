#include <semaphore.h>

#define N 8

typedef struct {
    int id;
    int track;
} request_t;

typedef struct {
    request_t req;
    int track;
} buffer_t;

// Declare the semaphores here
sem_t pq_empty, pq_full, pq_mutex;
sem_t pBuff_empty, pBuff_full, pBuff_mutex;

// Initialize the semaphores here

void init() {
    // The Locks (Binary Semaphores)
    sem_init(&pq_mutex, 0, 1);
    sem_init(&pBuff_mutex, 0, 1);

    // The Counters
    sem_init(&pq_empty, 0, N);
    sem_init(&pq_full, 0, 0);

    sem_init(&pBuff_empty, 0, 1);       // Buffer Size = 1
    sem_init(&pBuff_full, 0, 0);
}


void printfile(request_t req) {
    sem_wait(&pq_empty);
    sem_wait(&pq_mutex);

    enqueue(req);

    sem_post(&pq_mutex);
    sem_post(&pq_full);
}


void Spooler() {
    while(1) {
        // For retrieving the request from the print queue buffer
        request_t req;
        buffer_t output;       // For the output file

        // Consume the request from the print queue buffer
        sem_wait(&pq_full);
        sem_wait(&pq_mutex);

        dequeue(&req);

        sem_post(&pq_mutex);
        sem_post(&pq_empty);

        // Process the request
        process(req, &output);

        // Produce the output file
        sem_wait(&pBuff_empty);
        sem_wait(&pBuff_mutex);

        Output(output);

        sem_post(&pBuff_mutex);
        sem_post(&pBuff_full);
    }
}


void Printer() {
    while(1) {
        buffer_t output; // For the output file

        sem_wait(&pBuff_full);
        sem_wait(&pBuff_mutex);

        retrieve(&output);

        sem_post(&pBuff_mutex);
        sem_post(&pBuff_empty);

        // Print the output file
        printing(output);

    }
}