/*  COMP3230 Operating Systems
    Shaheer Ziya (3033594769)
    PS2 Q1
*/

#include <pthread.h>

int readerCount, writerCount;

pthread_mutex_t mutex;
pthread_cond_t read;
pthread_cond_t write;

void init() {
    readerCount = 0;
    writerCount = 0;
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&read, NULL);
    pthread_cond_init(&write, NULL);
}

void reader() {
    // Wait for writers to finish
    pthread_mutex_lock(&mutex);
    if (writerCount > 0) {
        pthread_cond_wait(&read, &mutex);
    }
    readerCount++;
    pthread_mutex_unlock(&mutex);
    
    do_reading();

    // Signal that all readers have finished
    pthread_mutex_lock(&mutex);
    readerCount--;
    if (readerCount == 0) {
        pthread_cond_signal(&write);
    }
    pthread_mutex_unlock(&mutex);
}

void writer() {
    // Wait for readers and writers to finish
    pthread_mutex_lock(&mutex);
    if (readerCount > 0 || writerCount > 0) {
        pthread_cond_wait(&write, &mutex);
    }
    writerCount++;
    pthread_mutex_unlock(&mutex);

    do_writing();

    // Signal that all writers have finished
    pthread_mutex_lock(&mutex);
    writerCount--;
    pthread_cond_signal(&write);
    pthread_cond_broadcast(&read);
    pthread_mutex_unlock(&mutex);
}

