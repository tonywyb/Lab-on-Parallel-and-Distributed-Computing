#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#define MPICH_SKIP_MPICXX
using namespace std;
bool isPrime(long n) {
	if (n < 2) {
		return false;
	}
	long t = sqrt(n);
	for (long i = 2;i <= t;++i) {
		if (n % i == 0) {
			return false;
		}
	}
	return true;
}
int main(int argc, char * argv[]) {
	int p, k, myrank, size;
	k = atoi(argv[1]);
	MPI_Status status;
	vector<long> ans;
	long scope = 1 << k;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	long * vThread = new long[scope/p+1];
	int num = 0;
	for (long i = myrank* scope/p; i < (myrank+1)*scope/p;++i) {
		if (isPrime(i)) {
			vThread[num++] = i;
		}
	}

	if (myrank != 0) {
		MPI_Send(vThread, num, MPI_LONG, 0, 0, MPI_COMM_WORLD);
	}else {
		ans.insert(ans.end(), vThread, vThread+num);
		for (int i = 1;i < p;++i) {
			MPI_Recv(vThread, scope/p+1, MPI_LONG, i, 0, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_LONG, &num);
			ans.insert(ans.end(), vThread, vThread+num);
		}
		sort(ans.begin(), ans.end());
		ofstream of("ref.out", ios::binary|ios::out);
		for (int i = 0;i < ans.size();++i) {
			of << ans[i] << " ";
		}
		for (int i = 0;i < ans.size();++i) {
			printf("%d ", ans[i]);
		}
	}

	MPI_Finalize();
}