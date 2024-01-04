#include <stdlib.h>
#include <stdio.h>
#include <signal.h>		/* To handle various signals */
#include <unistd.h>

void sigint_handler(int signum) {
	printf("Receives SIGINT!! IGNORE IT :)\n");
  return;
}

int main(int argc, char ** argv) {
  unsigned long long i, x;
  
	/* Install signal handler for SIGALRM */
	signal(SIGINT, sigint_handler);

	x = 1;
	while (1) {
		for (i=0; i<1073741824; i += 2)
			x *= i;
			if (x > 8589934589)
				x = 1;
	}	
	return 0;
}
