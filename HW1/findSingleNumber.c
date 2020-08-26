#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 99999999
int a[N];
int main (int argc, char *argv[]) {
	int i;
	int tmp = 0;
	#pragma omp parallel for
	for(i = 0; i < N; ++i) {
		a[i] = i >> 1;
	}
	a[65536 << 1] = 99999999 >> 1;
	#pragma omp parallel for reduction(^:tmp)
	for(i = 0; i < N; ++i) {
		tmp = tmp^a[i];
	}
	printf("%d\n", tmp);
}