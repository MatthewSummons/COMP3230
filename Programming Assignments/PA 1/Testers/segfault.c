#include <stdio.h>
#include <stdlib.h>

int max(int a, int b) {
	if (a > b) return a;
	else return b;
}

int main(int argc, char *argv[]) {
	char * buffer;

	int i, j, loop;

	if (argc == 2) loop = atoi(argv[1]);
	else loop = 128;
	
	buffer = (char *) malloc(1048576);
	for (j=0; j<loop; j++)
		for (i=0; i<1048576; i++) {
			buffer[i] = i%256;
		}

	buffer = (char *) (&max);
	
	buffer[100] = 100;
	
	return 0;
}