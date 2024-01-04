#include <stdlib.h>
#include <stdio.h>
#include <signal.h>		/* To handle various signals */
#include <unistd.h>

volatile int timer = 0;

void sigalrm_handler(int signum) {	
	printf("Time is up: %d seconds\nProgram terminated.\n", timer);
	exit(timer);
}

void main(int argc, char ** argv) {
	unsigned long long i, x;

	/* Install signal handler for SIGALRM */
	signal(SIGALRM, sigalrm_handler);
	
	if (argc > 2) {
		printf("Usage: ./loopf [second]\n");
		exit(0);
	} else if (argc == 2) {
		timer = atoi(argv[1]);
		alarm(timer);
	}

	x = 1;
	while (1) {
		for (i=0; i<1073741824; i += 2)
			x *= i;
			if (x > 8589934589)
				x = 1;
	}	

}
