#include <cstdio>
#include <algorithm>
#include <iostream>
#include <omp.h>
using namespace std;
int a[5000000];
int t[2][5000000];

void task(int start, int end, int threshold, int p, int* s) {
    int lp = 0, rp = end - start;
    for (int i = start; i <= end; ++i) {
        if (a[i] < threshold) {
            t[p][lp++] = a[i];
        } else {
            t[p][rp--] = a[i];
        }
    }
    *s = lp;
}

void k_find(int high, int k, int* num) {
    int middle = high / 2;
    int s1, s2;
	#pragma omp task shared(s1)
    task(1, middle, a[0], 0, &s1);
    #pragma omp task shared(s2)
    task(middle + 1, high, a[0], 1, &s2);
    #pragma omp taskwait

    if (s1 + s2 >= k) {
        for (int i = 0; i < s1; ++i) {
            a[i] = t[0][i];
        }
        for (int i = 0; i < s2; ++i) {
            a[i + s1] = t[1][i];
        }
        k_find(s1 + s2 - 1, k, num);
    }
    if (s1 + s2 < k - 1) {
        for (int i = s1; i <= middle - 1; ++i) {
            a[i - s1] = t[0][i];
        }
        for (int i = s2; i <= high - middle - 1; ++i) {
            a[i - s2 + middle - s1] = t[1][i];
        }
        k_find(high - s1 - s2 - 1, k - s1 - s2 - 1, num);
    }
    *num = a[0];
}

int main() {
    int n, k;
    int result;
	 
    scanf("%d%d", &n, &k);
    for (int i = 0; i < n; i++) 
	{
        scanf("%d", &a[i]);
    }
    #pragma omp parallel
    {
        #pragma omp single
		k_find(n - 1, k, &result);
    }
    printf("%d\n", result);
    return 0;
}
