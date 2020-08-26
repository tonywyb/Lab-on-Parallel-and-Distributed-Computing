#include<iostream>
#include<fstream>
#include<cstdio>
#include<cmath>
#include<map>
#include<vector>
#include<string>
#include<algorithm>
#include<cstdlib>
#include<time.h>
#include<mpi.h>
#include<CL/cl.h>

using namespace std;

ifstream fin;

int numprocs, myid;

//openCL kernel
const char *source =
"__kernel void clGetDistance(int dimension, __global double *p1, __global double *p2, __global double *distSquare)\n"
"{\n"
"	int tid = get_local_id(0);\n"
"	int tsize = get_local_size(0);\n"
"\n"
"	double sumSquare = 0;\n"
"	for(int i = tid; i < dimension; i += tsize) {\n"
"		sumSquare += (p1[i] - p2[i]) * (p1[i] - p2[i]);\n"
"	}\n"
"	__local double tmp[128];\n"
"	tmp[tid] = sumSquare;\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	if (tid == 0){\n"
"		for (int i = 1; i < 128; i++){\n"
"			tmp[0] += tmp[i];\n"
"		}\n"
"	*distSquare = tmp[0];\n"
"	}\n"
"}\n";

double dataset[50000 * 200];
string label[50000];

//数据归一化
void normalize(int rowLen, int colLen)
{
	double *maxValue = new double[colLen];
	double *minValue = new double[colLen];
	double *range = new double[colLen];

	for (int i = 0; i < colLen; i++)
	{
		maxValue[i] = max(dataset[i], dataset[colLen + i]);
		minValue[i] = min(dataset[i], dataset[colLen + i]);
	}
	for (int i = 2; i < rowLen; i++)
	{
		for (int j = 0; j < colLen; j++)
		{
			if (dataset[i * colLen + j] > maxValue[j])
				maxValue[j] = dataset[i * colLen + j];
			if (dataset[i * colLen + j] < minValue[j])
				minValue[j] = dataset[i * colLen + j];
		}
	}
	for (int i = 0; i < colLen; i++)
	{
		range[i] = maxValue[i] - minValue[i];
	}

	for (int i = 0; i < rowLen; i++)
	{
		for (int j = 0; j < colLen; j++)
			dataset[i * colLen + j] = (dataset[i * colLen + j] - minValue[j]) / range[j];
	}
	delete[] maxValue;
	delete[] minValue;
	delete[] range;
}

//计算距离
double getDistance(double *p1, double *p2, int colLen)
{
	double sum = 0;
	for (int i = 0; i < colLen; i++)
		sum += pow((p1[i] - p2[i]), 2);
	return sqrt(sum);
}

//用于pair_vector排序
struct CmpByValue
{
	bool operator() (const pair<int, double>& p1, const pair<int, double>& p2)
	{
		return p1.second < p2.second;
	}
};

struct result
{
	int index;
	double distance;
};
//用于pair_vector排序
struct CmpByValue2
{
	bool operator() (const result & r1, const result & r2)
	{
		return r1.distance < r2.distance;
	}
};
result allResults[64 * 20];
int main(int argc, char **argv)
{
	//k: "k"nn  size: data size  dimension: vector dimension
	int k, size, dimension, testSize;
	char *filename;
	int startTime, endTime;

	//OpenCL初始化
	cl_platform_id *platforms;
	cl_uint num_platforms;
	cl_int err;
	size_t ext_size;
	cl_device_id device = 0;
	cl_platform_id platform = 0;
	bool cuda = false;
	size_t maxWorkGroup;
	size_t maxWorkItemPerGroup;
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	if (err < 0)
	{
		printf("don't support openCL\n");
		exit(0);
	}

	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	clGetPlatformIDs(num_platforms, platforms, NULL);
	for (int i = 0; i < num_platforms; i++)
	{
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,
			0, NULL, &ext_size);
		char *name = (char *)malloc(ext_size);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
			ext_size, name, NULL);
		if (strcmp(name, "NVIDIA CUDA") == 0)
		{
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
			platform = platforms[i];
			cuda = true;
			break;
			/*clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &maxWorkGroup, NULL);
			printf("CL_DEVICE_MAX_WORK_GROUP=%d\n", (int)maxWorkGroup);
			clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkItemPerGroup, NULL);
			printf("CL_DEVICE_MAX_WORK_GROUP_SIZE=%d\n", (int)maxWorkItemPerGroup);*/
		}
	}
	if (!cuda)
	{
		printf("no nvidia cuda!\n");
		exit(0);
	}
	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0 };
	cl_context mContext = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
	cl_program myprog = clCreateProgramWithSource(mContext, 1, &source, NULL, &err);
	err = clBuildProgram(myprog, 1, &device, 0, 0, 0);
	if (err != CL_SUCCESS)
	{
		char buffer[20000];
		size_t len;

		clGetProgramBuildInfo(myprog, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		puts(buffer);
		return -1;
	}
	cl_kernel mykernel = clCreateKernel(myprog, "clGetDistance", &err);

	//MPI初始化
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

	//自定义主从进程传送数据类型
	MPI_Datatype oldTypes[2] = {MPI_INT, MPI_DOUBLE};
	MPI_Datatype mpi_result;
	int blockLength[2] = { 1, 1 };
	MPI_Aint offset[2] = { 0, sizeof(double) };
	MPI_Type_create_struct(2, blockLength, offset, oldTypes, &mpi_result);
	MPI_Type_commit(&mpi_result);

	if (argc != 5)
	{
		printf("need to be ./xx k size dimension filename\n");
		MPI_Finalize();
		exit(0);
	}
	k = atoi(argv[1]);
	size = atoi(argv[2]);
	dimension = atoi(argv[3]);
	filename = argv[4];

	if (myid == 0)
	{
		fin.open(filename);
		if (!fin)
		{
			printf("cannot open file\n");
			MPI_Finalize();
			exit(0);
		}
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < dimension; j++)
			{
				fin >> dataset[i * dimension + j];
			}
			fin >> label[i];
		}
		normalize(size, dimension);
	}
	testSize = 0.1 * size;
	int testCount = 0;
	int row_each_proc = ceil(size / numprocs);
	int restSize = size - row_each_proc * (numprocs - 1);
	if (myid == 0)
	{
		startTime = clock();
		for (int i = 1; i < numprocs - 1; i++)
			MPI_Send(dataset + i * row_each_proc * dimension, row_each_proc * dimension, 
				MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		if (numprocs != 1)
			MPI_Send(dataset + (numprocs - 1) * row_each_proc * dimension, restSize * dimension,
				MPI_DOUBLE, numprocs - 1, 0, MPI_COMM_WORLD);
	}
	else
	{
		if (myid != numprocs - 1)
			MPI_Recv(dataset, row_each_proc * dimension, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		else
			MPI_Recv(dataset, restSize * dimension, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		
		int myRow = row_each_proc;
		if (myid == numprocs - 1)
			myRow = restSize;
	}
	double testData[200];
	map<int, double> index_dis;
	for (int i = 0; i < testSize; i++)
	{
		if (myid == 0)
		{
			//printf("test point %d\n", i);
			memcpy(testData, dataset + i * dimension, sizeof(double) * dimension);
			MPI_Bcast(testData, dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			int myRow = row_each_proc;
			if (myid == numprocs - 1)
				myRow = restSize;
			double distance;
			for (int j = testSize; j < myRow; j++)
			{
				//distance = getDistance(dataset + j * dimension, testData, dimension);

				//try openCL
				cl_command_queue mQueue;
				mQueue = clCreateCommandQueue(mContext, device, 0, &err);
				cl_mem p1 = clCreateBuffer(mContext, CL_MEM_READ_ONLY, dimension * sizeof(double), NULL, &err);
				cl_mem p2 = clCreateBuffer(mContext, CL_MEM_READ_ONLY, dimension * sizeof(double), NULL, &err);
				cl_mem res = clCreateBuffer(mContext, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &err);
				err = clEnqueueWriteBuffer(mQueue, p1, CL_TRUE, 0, sizeof(double) * dimension, (void *)testData, 0, NULL, NULL);
				err = clEnqueueWriteBuffer(mQueue, p2, CL_TRUE, 0, sizeof(double) * dimension, (void *)(dataset + j * dimension), 0, NULL, NULL);
				double square;
				clSetKernelArg(mykernel, 0, sizeof(int), &dimension);
				clSetKernelArg(mykernel, 1, sizeof(cl_mem), &p1);
				clSetKernelArg(mykernel, 2, sizeof(cl_mem), &p2);
				clSetKernelArg(mykernel, 3, sizeof(cl_mem), &res);
				size_t globalWorkSize[1];
				size_t localWorkSize[1];
				globalWorkSize[0] = 128;
				localWorkSize[0] = 128;
				err = clEnqueueNDRangeKernel(mQueue, mykernel, 1, NULL, globalWorkSize,
					localWorkSize, 0, NULL, NULL);
				clFinish(mQueue);
				clEnqueueReadBuffer(mQueue, res, CL_TRUE, 0, sizeof(double), &square, 0, NULL, NULL);
				distance = sqrt(square);

				//double distance2 = getDistance(dataset + j * dimension, testData, dimension);
				index_dis[myid * row_each_proc + j] = distance;
				clReleaseCommandQueue(mQueue);
				clReleaseMemObject(p1);
				clReleaseMemObject(p2);
				clReleaseMemObject(res);
			}
			vector<pair<int, double>> vec_index_dis(index_dis.begin(), index_dis.end());
			sort(vec_index_dis.begin(), vec_index_dis.end(), CmpByValue());
			for (int j = 0; j < k; j++)
			{
				allResults[j].index = vec_index_dis[j].first;
				allResults[j].distance = vec_index_dis[j].second;
			}
			//接收各进程的离测试点前k近的点
			for (int j = 1; j < numprocs; j++)
			{
				MPI_Recv(allResults + k * j, k, mpi_result, j, 0, MPI_COMM_WORLD, &status);
			}
			sort(allResults, allResults + k * numprocs, CmpByValue2());
			
			/*for (int j = 0; j < numprocs * k; j++)
			{
				printf("index: %d, dis: %lf\n", allResults[j].index, allResults[j].distance);
			}*/
			
			map<string, int> label_freq;

			for (int j = 0; j < k; j++)
			{
				//printf("index: %d dis: %lf\n", allResults[j].index, allResults[j].distance);
				label_freq[label[allResults[j].index]]++;
			}

			map<string, int>::iterator p = label_freq.begin();
			string tempLabel;
			int max_freq = 0;
			while (p != label_freq.end())
			{
				if (p->second > max_freq)
				{
					max_freq = p->second;
					tempLabel = p->first;
				}
				p++;
			}
			if (tempLabel != label[i])
				testCount++;
			label_freq.clear();
			index_dis.clear();
		}
		else
		{
			MPI_Bcast(testData, dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			int myRow = row_each_proc;
			if (myid == numprocs - 1)
				myRow = restSize;
			double distance;
			for (int j = 0; j < myRow; j++)
			{
				//distance = getDistance(dataset + j * dimension, testData, dimension);

				//try openCL
				cl_command_queue mQueue;
				mQueue = clCreateCommandQueue(mContext, device, 0, &err);
				cl_mem p1 = clCreateBuffer(mContext, CL_MEM_READ_ONLY, dimension * sizeof(double), NULL, &err);
				cl_mem p2 = clCreateBuffer(mContext, CL_MEM_READ_ONLY, dimension * sizeof(double), NULL, &err);
				cl_mem res = clCreateBuffer(mContext, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &err);
				err = clEnqueueWriteBuffer(mQueue, p1, CL_TRUE, 0, sizeof(double) * dimension, (void *)testData, 0, NULL, NULL);
				err = clEnqueueWriteBuffer(mQueue, p2, CL_TRUE, 0, sizeof(double) * dimension, (void *)(dataset + j * dimension), 0, NULL, NULL);
				double square;
				clSetKernelArg(mykernel, 0, sizeof(int), &dimension);
				clSetKernelArg(mykernel, 1, sizeof(cl_mem), &p1);
				clSetKernelArg(mykernel, 2, sizeof(cl_mem), &p2);
				clSetKernelArg(mykernel, 3, sizeof(cl_mem), &res);
				size_t globalWorkSize[1];
				size_t localWorkSize[1];
				globalWorkSize[0] = 128;
				localWorkSize[0] = 128;
				err = clEnqueueNDRangeKernel(mQueue, mykernel, 1, NULL, globalWorkSize,
					localWorkSize, 0, NULL, NULL);
				clFinish(mQueue);
				clEnqueueReadBuffer(mQueue, res, CL_TRUE, 0, sizeof(double), &square, 0, NULL, NULL);
				distance = sqrt(square);

				index_dis[myid * row_each_proc + j] = distance;
				clReleaseCommandQueue(mQueue);
				clReleaseMemObject(p1);
				clReleaseMemObject(p2);
				clReleaseMemObject(res);
			}
			vector<pair<int, double>> vec_index_dis(index_dis.begin(), index_dis.end());
			sort(vec_index_dis.begin(), vec_index_dis.end(), CmpByValue());
			for (int j = 0; j < k; j++)
			{
				allResults[j].index = vec_index_dis[j].first;
				allResults[j].distance = vec_index_dis[j].second;
			}
			MPI_Send(allResults, k, mpi_result, 0, 0, MPI_COMM_WORLD);
			index_dis.clear();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (myid == 0)
	{
		endTime = clock();
		printf("error rate: %lf\n", (testCount * 1.0) / testSize);
		printf("time: %d s\n", (endTime - startTime) / CLOCKS_PER_SEC);
	}
	MPI_Finalize();
	return 0;
}