#include <wb.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 512 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
	//@@ Modify the body of this kernel to generate the scanned blocks
	//@@ Make sure to use the workefficient version of the parallel scan
	//@@ Also make sure to store the block sum to the aux array 

	__shared__ float XY[20 * BLOCK_SIZE];

	int a = 2 * blockIdx.x*blockDim.x + threadIdx.x;
	int b = a;

	for (int i = threadIdx.x; i < len; i += 2 * BLOCK_SIZE) {  //Data Loading Phase Kernel Code PPT Slide 22     Highly Modified
		if (b < len) {
			XY[i] = input[b];
		}
		if (b + blockDim.x < len) {
			XY[i + BLOCK_SIZE] = input[b + blockDim.x];
		}
		b += 2 * BLOCK_SIZE;
	}

	__syncthreads();

	for (unsigned int stride = 1; stride < BLOCK_SIZE; stride *= 2) { //Reduction Phase Kernel Code PPT11 Slide 24
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index <= 2*BLOCK_SIZE) {
			XY[index] += XY[index - stride];
		}
	}
	
	for (int stride = 2*BLOCK_SIZE / 4; stride > 0; stride /= 2) { // Post Reduction Reverse Phavse Kernel Code PPT11 Slide 27
		__syncthreads();
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < 2*BLOCK_SIZE) {
			XY[index + stride] += XY[index];
		}
	}
	
	__syncthreads();

	int c = 2 * blockIdx.x*blockDim.x + blockDim.x-1;
	if (a < len) {
		output[a] = XY[threadIdx.x];
	}
	if (a + blockDim.x < len) {
		
		if (a+BLOCK_SIZE == c + BLOCK_SIZE) {
			output[a + BLOCK_SIZE] = XY[threadIdx.x + BLOCK_SIZE] - XY[threadIdx.x + BLOCK_SIZE];
			output[a + BLOCK_SIZE] = XY[threadIdx.x + BLOCK_SIZE-1] + input[a + BLOCK_SIZE]; //finds the last value of the block
			aux[blockIdx.x] = XY[threadIdx.x + BLOCK_SIZE - 1] + input[a + BLOCK_SIZE];      //saves it to the aux
		}
		else {
			output[a + BLOCK_SIZE] = XY[threadIdx.x + BLOCK_SIZE];
		}
	}
}

__global__ void addScannedBlockSums(float *input, float *aux, int len) {
	//@@ Modify the body of this kernel to add scanned block sums to 
	//@@ all values of the scanned blocks

	int a = 2 * blockIdx.x*blockDim.x + threadIdx.x; //approach is similar to Data Loading Phase of Scan Kernel

	if(blockIdx.x >= 1){
		if (a < len) {
			input[a] += aux[blockIdx.x - 1];
		}
		if (a + blockDim.x < len) {
			input[a + blockDim.x] += aux[blockIdx.x - 1];
		}
	}

	__syncthreads();
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output 1D list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray;
  float *deviceAuxScannedArray;
  int numElements; // number of elements in the input/output list
  int numBlocks;	//number of blocks needed to determine GridSize

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  //numBlocks = ceil(float(numElements) / BLOCK_SIZE);
  numBlocks = (float(numElements - 1)) / BLOCK_SIZE + 1;
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numElements);

  wbTime_start(GPU, "Allocating device memory.");
  //@@ Allocate device memory
  //you can assume that aux array size would not need to be more than BLOCK_SIZE*2 (i.e., 1024)
  cudaMalloc((void**)&deviceInput, numElements * sizeof(float));
  cudaMalloc((void**)&deviceOutput, numElements * sizeof(float));
  cudaMalloc((void**)&deviceAuxArray, 2*numBlocks * sizeof(float));
  cudaMalloc((void **)&deviceAuxScannedArray, 2*numBlocks * sizeof(float));
  wbTime_stop(GPU, "Allocating device memory.");

  wbTime_start(GPU, "Clearing output device memory.");
  cudaMemset(deviceOutput, 0, numElements * sizeof(float));
  wbTime_stop(GPU, "Clearing output device memory.");

  wbTime_start(GPU, "Copying input host memory to device.");
  //@@ Copy input host memory to device	
  cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input host memory to device.");

  //@@ Initialize the grid and block dimensions here
  dim3 GridDim(numBlocks, 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  //@@ You need to launch scan kernel twice: 1) for generating scanned blocks 
  //@@ (hint: pass deviceAuxArray to the aux parameter)
  //@@ and 2) for generating scanned aux array that has the scanned block sums. 
  //@@ (hint: pass NULL to the aux parameter)
  //@@ Then you should call addScannedBlockSums kernel.
  scan <<< GridDim, BlockDim >>> (deviceInput, deviceOutput, deviceAuxArray, numElements);
  cudaDeviceSynchronize();
  scan <<< GridDim, BlockDim >>> (deviceAuxArray, deviceAuxScannedArray, NULL, numBlocks);
  cudaDeviceSynchronize();
  addScannedBlockSums <<< GridDim, BlockDim >>>(deviceOutput, deviceAuxScannedArray, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  //@@ Copy results from device to host	
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  //@@ Deallocate device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
