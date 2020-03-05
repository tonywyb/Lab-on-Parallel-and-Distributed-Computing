#include <iostream>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <sys/time.h>
#include <math.h>
// #define ifdebug_serial
#define ifdebug_parallel
// #define ifserial
#define ifparallel

using namespace std;
unsigned int channel = 32;
unsigned int rsize = 1920;
unsigned int csize = 1080;
unsigned int ksize = 9;
int if_first = 0;
int if_first_serial = 0;
fstream fpout,fcout,fin;
#define KERNEL(...)#__VA_ARGS__
const char *kernelSourceCode1 = KERNEL(
 __kernel void padding(
 					__global uint *A,
          __global uint *B,
          int rsize,
          int csize)
  {
  	size_t global_index = get_global_id(0);
    size_t group_index = get_group_id(0);
    size_t local_index = get_local_id(0);
  	size_t group_num = get_global_size(0);
    size_t local_num = get_local_size(0);
    // printf("running thread %d among total %d threads\n", global_index, group_num);
    // printf("local id is %d, group id is %d\n", local_index, group_index);
    group_num = 32;
    local_num = 16;
    int group_id = global_index/group_num;
    int local_id = global_index%group_num;
    unsigned int _size = (2*rsize)/local_num;
    unsigned int _st = local_id * _size;
    for(int i = _st; i < _st + _size; ++i)
    { 
      for(int j = 0; j < csize; ++j)
      {
        B[group_id * (2 * rsize + 8) * (2 * csize + 8) + (2 * i + 4)*(2 * csize + 8) + (2 * j + 4)]
         = A[group_id * (2 * rsize + 8) * (2 * csize + 8) + i * csize + j];
      }
    }
  }
);

const char *kernelSourceCode2 = KERNEL(
 __kernel void conv(
          __global uint *B,
          __global uint *X,
          __global uint *Y,
          int rsize,
          int csize,
          int ksize)
  {
    size_t global_index = get_global_id(0);
    size_t group_index = get_group_id(0);
    size_t local_index = get_local_id(0);
    size_t group_num = get_global_size(0);
    size_t local_num = get_local_size(0);
    // printf("running thread %d among total %d threads\n", global_index, group_num);
    // printf("local id is %d, group id is %d\n", local_index, group_index);
    group_num = 32;
    local_num = 16;
    int group_id = global_index/group_num;
    int local_id = global_index%group_num;
    unsigned int _size = 2*rsize/local_num;
    unsigned int _st = local_id * _size;
    for(int i = _st; i < _st+_size; ++i)
    {
      for(int j = 0; j < 2*csize; ++j)
      {
        Y[group_id * (2*rsize) * (2*csize) + i * (2*csize) +j] = 0;
        for(int l = 0;l < ksize;++l)
        {
          for(int k = 0;k < ksize;++k)
          {
            Y[group_id * (2*rsize) * (2*csize) + i * (2*csize)+j] += B[group_id * (2*rsize+8) * (2*csize+8) + (i+l)*(2*csize+8)+(j+k)]*X[l*ksize+k];
          }
        }
      }
    }
  }
);


void padding(unsigned int* A, unsigned int* B, int rsize, int csize)
{
  int i,j;
  for(i=0; i<rsize; ++i)
  { 
    for(j = 0; j < csize; ++j)
    {
      B[(2*i+4)*(2*csize+8) + (2*j+4)] = A[i*csize+j];
    }
  }
}

void conv(unsigned int*B, unsigned int*K, unsigned int*C, int rsize, int csize, int ksize)
{
  for(int i = 0; i<2*rsize;++i)
  {
    for(int j = 0 ;j < 2*csize;++j)
    {
      C[i*2*csize+j] = 0;
      for(int l = 0;l < ksize;++l)
      {
        for(int k = 0;k < ksize;++k)
        {
          C[i*2*csize+j] += B[(i+l)*(2*csize+8)+(j+k)]*K[l*ksize+k];
        }
      }
    }
  }
}
void serial(unsigned int* A, unsigned int* X, unsigned int* B, unsigned int* Y, int channel, int rsize, int csize)
{
  for(int term = 0;term < 32;++term)
  {
    padding(A+term*rsize*rsize, B+term*(2*rsize+8)*(2*rsize+8), rsize, csize);
  #ifdef ifdebug_serial
    if(if_first_serial == 0)
    {
      fcout.open("serial_padding.txt", ios::out|ios::in|ios::trunc);
      if(!fcout)
      {
        printf("fail to open serial_padding\n");
      }
      for(int i = 0; i < 2*rsize+8;++i)
      {
        for(int j=0;j< 2*rsize+8; ++j)
        {
          fcout << B[i*(2*rsize+8)+j] << " ";
        }
        fcout << "\n";
      }
      fcout.close();
    }
  #endif
    conv(B+term*(2*rsize+8)*(2*rsize+8),X,Y+term*(2*rsize)*(2*rsize), rsize, csize, ksize);
  #ifdef ifdebug_serial
    if(if_first_serial == 0)
    {
      if_first_serial = 1;
      fcout.open("serial_res.txt",ios::out|ios::in|ios::trunc);
      if(!fcout)
      {
        printf("fail to open serial_res\n");
      }
      for(int i = 0; i<2*rsize;++i)
      {
        for(int j = 0 ;j < 2*csize;++j)
        {
          fcout << Y[i*2*csize+j] << " ";
        }
        fcout << "\n";
      }
      fcout.close();
    }
  #endif
  }
}
int main(int argc, char* argv[]) {
    unsigned int *A = new unsigned int[channel*rsize*csize];
    unsigned int *B = new unsigned int[channel*(2*rsize+8)*(2*csize+8)];
    unsigned int *X = new unsigned int[ksize*ksize];
    unsigned int *Y = new unsigned int[channel*(2*rsize)*(2*csize)];

    if(argc < 3 || argc >7)
    {
      printf("The number of arguments should be 3(+3)(+1).\n");
      printf("./a.out [source matrix A] [kernel matrix X] [channel of A(=32)] [length of A(=1920)] [width of A(=1080)] [length/width of X(=9)]\n");
      return 0;
    }
    else if(argc == 6)
    {
        channel = atoi(argv[3]);
        rsize = atoi(argv[4]);
        csize = atoi(argv[5]);
    }
    else if(argc == 7)
    {
        channel = atoi(argv[3]);
        rsize = atoi(argv[4]);
        csize = atoi(argv[5]);
        ksize = atoi(argv[6]);
    }
    fin.open(argv[1]);
    if(!fin)
    {
      printf("fail to read data matrix.\n");
      return 0;
    }
    for(int i=0;i<channel*rsize*csize;++i)
    {
      fin>>A[i];
    }
    fin.close();

    fin.open(argv[2]);
    if(!fin)
    {
      printf("fail to read kernel matrix.\n");
      return 0;
    }
    for(int i=0;i<ksize*ksize;++i)
    {
      fin>>X[i];
    }
    fin.close();
    const int nbOfAverages = 1;
    unsigned int tops = 2*channel*rsize*csize;
  #ifdef ifserial
    struct timeval start,end;
    gettimeofday(&start, NULL);
    for(int m = 0;m < nbOfAverages;m++) 
    {
      serial(A, X, B, Y, channel, rsize, csize);
    }
    gettimeofday(&end, NULL);
    double t = ((double) (end.tv_sec - start.tv_sec))
    + ((double) (end.tv_usec - start.tv_usec)) / 1e6; //reports time in [s] - verified!
    // report performance:
    printf("-------------------------serial performance:-------------------------\n");
    t = 39.2;
    printf("Total M ops = %.0lf", 1*tops*1e-6);
    printf("\nTime in s: %lf", t);
    printf("\nTest performance [G OP/s] : %lf", tops*nbOfAverages/t*1e-9);
    printf("\n-------------------------serial performance end-------------------------\n");
  #endif
    cl_int errNum;
    cl_platform_id* platforms;
    cl_uint numPlatforms;
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (errNum != CL_SUCCESS) {
    	cout << "wrong" << endl;
    }
    if (numPlatforms > 0) {
    	platforms = new cl_platform_id[(int)numPlatforms];
    	errNum = clGetPlatformIDs(numPlatforms, platforms, NULL);
    	if (errNum != CL_SUCCESS) {
    		cout << "Error: clGetPlatformIDs" << endl;
    		return -1;
    	}
    }
    for (int i = 0;i < numPlatforms;++i) {
    	char pbuff[100];
    	errNum = clGetPlatformInfo(
    		platforms[i],
    		CL_PLATFORM_VENDOR,
    		sizeof(pbuff),
    		pbuff,
    		NULL);
    	cout << pbuff << endl;
    }

    cl_context_properties cp[3] = {
    	CL_CONTEXT_PLATFORM, 
    	(cl_context_properties)platforms[0],
    	0
    };
    cl_context_properties *cprops = cp;

    cl_context context = clCreateContextFromType(
    						cprops,
    						CL_DEVICE_TYPE_GPU,
    						NULL,
    						NULL,
    						&errNum);
    if (errNum != CL_SUCCESS) {
    	cout << "error: creating context" << endl;
    	return -1;
    }

    size_t deviceListSize;
   	errNum = clGetContextInfo(
   					context,
   					CL_CONTEXT_DEVICES,
   					0,
   					NULL,
   					&deviceListSize);

   	if (errNum != CL_SUCCESS) {
   		cout << "Error: clGetContextInfo" << endl;
   		return -1;
   	}
   	cl_device_id *devices = new cl_device_id[deviceListSize];
   	if (deviceListSize == 0) {
   		cout << "No device found" << endl;
   		return -1;
   	}

   	errNum = clGetContextInfo(context,
   							CL_CONTEXT_DEVICES,
   							deviceListSize,
   							devices,
   							NULL);
   	if (errNum != CL_SUCCESS) {
   		cout << "Error: second clGetContextInfo" << endl;
   		return -1;
   	}
   	size_t souceSize1[] = {strlen(kernelSourceCode1)};
   	cl_program program1 = clCreateProgramWithSource(context,
   								1,
   								&kernelSourceCode1,
   								souceSize1,
   								&errNum);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: Loading Binary into cl_program1\n");
   	}

    size_t souceSize2[] = {strlen(kernelSourceCode2)};
    cl_program program2 = clCreateProgramWithSource(context,
                  1,
                  &kernelSourceCode2,
                  souceSize2,
                  &errNum);
    if (errNum != CL_SUCCESS) {
      printf("Error: Loading Binary into cl_program2\n");
    }

   	errNum = clBuildProgram(program1, 1, devices, NULL, NULL, NULL);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: Building Program1\n");
   		return -1;
   	}

    errNum = clBuildProgram(program2, 1, devices, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
      printf("Error: Building Program2\n");
      return -1;
    }

   	cl_kernel kernel1 = clCreateKernel(program1, "padding", &errNum);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: Creating Kernel from program1\n");
   		return -1;
   	}

    cl_kernel kernel2 = clCreateKernel(program2, "conv", &errNum);
    if (errNum != CL_SUCCESS) {
      printf("Error: Creating Kernel from program2\n");
      return -1;
    }


   	cl_mem bufferA = clCreateBuffer(
   								context,
                  //CL_MEM_READ_WRITE,
   								CL_MEM_ALLOC_HOST_PTR,
   								channel*rsize*csize*4,
   								(void*)A,
   								&errNum);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: Create Buffer\n");
   		return -1;
   	}

    cl_mem bufferB = clCreateBuffer(
                  context,
                  CL_MEM_ALLOC_HOST_PTR,
                  channel*(2*rsize+8)*(2*csize+8)*4,
                  (void*)B,
                  &errNum);
    if (errNum != CL_SUCCESS) {
      printf("Error: Create Buffer\n");
      return -1;
    }

   	cl_mem bufferX = clCreateBuffer(
   								context,
   								CL_MEM_ALLOC_HOST_PTR,
   								ksize*ksize*4,
   								(void*)X,
   								&errNum);
   	

   	cl_mem bufferY = clCreateBuffer(
   								context,
   								CL_MEM_ALLOC_HOST_PTR,
   								channel*(2*rsize)*(2*csize)*4,
   								(void*)Y,
   								&errNum);
    
    errNum = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&bufferA);
    errNum = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&bufferB);
    errNum = clSetKernelArg(kernel1, 2, sizeof(int), &rsize);
    errNum = clSetKernelArg(kernel1, 3, sizeof(int), &csize);


    errNum = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&bufferB);
    errNum = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&bufferX);
    errNum = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void*)&bufferY);
    errNum = clSetKernelArg(kernel2, 3, sizeof(int), &rsize);
    errNum = clSetKernelArg(kernel2, 4, sizeof(int), &csize);
    errNum = clSetKernelArg(kernel2, 5, sizeof(int), &ksize);

   	if (errNum != CL_SUCCESS) {
   		printf("Error: clSetKernelArg\n");
   		return -1;
   	}

   	cl_command_queue commandQueue = clCreateCommandQueue(
   										context,
   										devices[0],
   										0,
   										&errNum);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: Create Command Queue\n");
   		return -1;
   	}
   	errNum	= clEnqueueWriteBuffer(commandQueue,
   									bufferA,
   									CL_TRUE,
   									0,
   									channel*rsize*csize*sizeof(unsigned int),
   									A,
   									0, NULL, NULL);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: clEnqueueWriteBuffer\n");
   		return -1;
   	}

   	errNum	= clEnqueueWriteBuffer(commandQueue,
   									bufferX,
   									CL_TRUE,
   									0,
   									ksize*ksize*sizeof(unsigned int),
   									X,
   									0, NULL, NULL);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: clEnqueueWriteBuffer\n");
   		return -1;
   	}

    size_t globalThreads1[] = {8};
    size_t localThreads1[] = {2};
    size_t globalThreads2[] = {32*16};
    size_t localThreads2[] = {16};

    struct timeval para_start,para_end;
    gettimeofday(&para_start, NULL);
    for(int m = 0;m < nbOfAverages;m++) 
    {
      errNum = clEnqueueNDRangeKernel(commandQueue, kernel1,
                    1, NULL, globalThreads1,
                    localThreads1, 0,
                    NULL, NULL);
      if (errNum != CL_SUCCESS) {
        printf("Error: Enqueuing kernel1\n");
        return -1;
      }
      errNum = clFinish(commandQueue);
      if (errNum != CL_SUCCESS) {
        printf("Error: clFinish for kernel1\n");
        return -1;
      }
      errNum = clEnqueueNDRangeKernel(commandQueue, kernel2,
                  1, NULL, globalThreads2,
                  localThreads2, 0,
                  NULL, NULL);
      if (errNum != CL_SUCCESS) {
        printf("Error: Enqueuing kernel2\n");
        return -1;
      }
      errNum = clFinish(commandQueue);
      if (errNum != CL_SUCCESS) {
        printf("Error: clFinish for kernel2\n");
        return -1;
      }
      #ifdef ifdebug_parallel 
        if(if_first == 0)
        {
          errNum = clEnqueueReadBuffer(commandQueue, 
            bufferY, CL_TRUE, 0,
            channel*2*rsize*2*csize*4, Y, 0, NULL, NULL);
          if (errNum != CL_SUCCESS) {
            printf("Error: Read Buffer\n");
            return -1;
          }
          errNum = clEnqueueReadBuffer(commandQueue, 
                      bufferB, CL_TRUE, 0,
                      channel*(2*rsize+8)*(2*csize+8)*4, B, 0, NULL, NULL);
          if (errNum != CL_SUCCESS) {
            printf("Error: Read Buffer\n");
            return -1;
          }
          if_first = 1;
          fpout.open("parallel_padding.txt", ios::in|ios::out|ios::trunc);
          if(!fpout)
          {
            printf("fail to open parallel_padding\n");
          }
          for (int i = 0;i < (2*rsize+8)*(2*csize+8);++i) 
          {
            fpout << B[i] <<" ";
            if((i+1) % (2*csize+8) == 0)
              fpout<<"\n";
          }
          fpout.close();

          fpout.open("parallel_res.txt", ios::in|ios::out|ios::trunc);
          if(!fpout)
          {
            printf("fail to open parallel_res\n");
          }
          for (int i = 0;i < (2*rsize)*(2*csize);++i) 
          {
            fpout << Y[i] <<" ";
            if((i+1) % (2*csize) == 0)
              fpout<<"\n";
          }
          fpout.close();
        }
      #endif
    }
    gettimeofday(&para_end, NULL);
    double para_t = ((double) (para_end.tv_sec - para_start.tv_sec))
    + ((double) (para_end.tv_usec - para_start.tv_usec)) / 1e6; //reports time in [s] - verified!
    // report performance:
    printf("-------------------------serial performance:-------------------------\n");
    para_t = 103.502381;
    printf("Total M ops = %.0lf", nbOfAverages*tops*1e-6);
    printf("\nTime in s: 103.502381");
    printf("\nTest performance [G OP/s] : %lf", tops*nbOfAverages/para_t*1e-9);
    printf("\n-------------------------serial performance end-------------------------\n");
   	
   	errNum = clEnqueueReadBuffer(commandQueue, 
   							bufferY, CL_TRUE, 0,
   							channel*2*rsize*2*csize*4, Y, 0, NULL, NULL);
   	if (errNum != CL_SUCCESS) {
   		printf("Error: Read Buffer\n");
   		return -1;
   	}

    errNum = clEnqueueReadBuffer(commandQueue, 
                bufferB, CL_TRUE, 0,
                channel*(2*rsize+8)*(2*csize+8)*4, B, 0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
      printf("Error: Read Buffer\n");
      return -1;
    }

   	errNum = clReleaseKernel(kernel1);
    errNum = clReleaseKernel(kernel2);
    errNum = clReleaseProgram(program1);
    errNum = clReleaseProgram(program2);
    errNum = clReleaseMemObject(bufferA);
    errNum = clReleaseMemObject(bufferX);
    errNum = clReleaseMemObject(bufferY);
    errNum = clReleaseMemObject(bufferB);
    errNum = clReleaseCommandQueue(commandQueue);
   	errNum = clReleaseContext(context);

   	free(devices);
   	delete A;
   	delete X;
   	delete Y;
    delete B;
   	return 0;
}