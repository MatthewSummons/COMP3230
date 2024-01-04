# include <stdio.h>         // For I/O
# include <stdlib.h>        // For rand() & srand()
# include <time.h>          // To provide the seed for rand
# include <string.h>        // For strlen()
#include "randStr.h"

#define ASCII_LENGTH 128
#define ASCII_A 65

/*

*/
void randStr(char *str, int strLen) {

    srand(time(0));         // Set the seed for rand()

    for (size_t i = 0; i < strLen; i++) {
        str[i] = (char) (rand() % ((ASCII_LENGTH - ASCII_A) + ASCII_A));
    }
}

// int main(int argc, char** argv) {

//     char *x = malloc(sizeof(char) * 100);
//     randStr(x, 100);

//     printf("%s\n", x);

//     free(x);
// }