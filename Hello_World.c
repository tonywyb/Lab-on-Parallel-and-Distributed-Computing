#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[]) {
int nthreads, tid;
	/* Fork a team of threads giving them their own copies of variables */
	#pragma omp parallel private(nthreads, tid)
	{
	 	/* Obtain thread number */
		tid = omp_get_thread_num();
		/* Only master thread does this */ if (tid == 0) {
			nthreads = omp_get_num_threads();
			printf("Number of threads = %d; ", nthreads); 
		}
		printf("Hello World from thread = %d; ", tid);
	} /* All threads join master thread and disband */
}