#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 100000000
float a[N], b[N], c[N];
int main (int argc, char *argv[]) {
	int i;
	float tmp;
	#pragma omp parallel for
	for(i = 0; i < N; ++i) {
		a[i] = i % 3;
		b[i] = i + 1;
	}
	#pragma omp parallel for private(tmp)
	for(i = 0; i < N; ++i) {
		tmp = a[i]/b[i];
		c[i] = tmp * tmp;
	}
}