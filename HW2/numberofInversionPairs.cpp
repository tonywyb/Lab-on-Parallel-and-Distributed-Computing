#include <cstdio>
#include <omp.h>
using namespace std;
int num[20000], tmp[20000];
int merge(int beg, int end) {
    if (beg == end)
        return 0;
    int mid = (beg + end) >> 1, lpos = beg, rpos = mid + 1, i;
    int a, b;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            a = merge(beg, mid);
            #pragma omp task
            b = merge(mid + 1, end);
        }
    }
    int ret = a + b;
    #pragma omp parallel for
    for (i = beg; i <= end; ++i)
        tmp[i] = num[i];

    i = beg;
    while (lpos <= mid && rpos <= end)              
        if (tmp[lpos] <= tmp[rpos])
            num[i++] = tmp[lpos++];
        else {
            num[i++] = tmp[rpos++];
            ret += mid - lpos + 1;
        }
    

    while (lpos <= mid)
        num[i++] = tmp[lpos++];
    while (rpos <= end)
        num[i++] = tmp[rpos++];
    return ret;
}
int main() {
    int N;
    scanf("%d", &N);
    for (int i = 0; i < N; ++i)
        scanf("%d", num + i);
    printf("%d\n", merge(0, N - 1));
    return 0;
}