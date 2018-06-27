#include <wb.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_BINS 4096
#define BLOCK_SIZE 512 

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

__global__ void histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) {
	//@@ Write the kernel that computes the histogram
	//@@ Make sure to use the privitization technique
	
	__shared__ unsigned int private_histo[NUM_BINS];

	for (unsigned int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += BLOCK_SIZE) { //reference to line 3 from figure 9.10 from pg 211 of the textbook.
		private_histo[binIdx] = 0;
	}

	__syncthreads();

	int tid = threadIdx.x + blockIdx.x * blockDim.x;	//reference PPT12 Slides 45-47
	int stride = blockDim.x * gridDim.x;

	while(tid < num_elements) {
		int numberValue = input[tid];
		if (numberValue >= 0 && numberValue < num_bins) {
			atomicAdd(&(private_histo[numberValue]), 1);
		}
		tid += stride;
	}
	
	__syncthreads();

	for (unsigned int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += BLOCK_SIZE) { //reference to line 9 from figure 9.10 from pg 211 of the textbook.
		atomicAdd(&(bins[binIdx]), private_histo[binIdx]);
	}
}

__global__ void saturate(unsigned int *bins, unsigned int num_bins) {
	//@@ Write the kernel that applies saturtion to counters (i.e., if the bin value is more than 127, make it equal to 127)

	for (int i = 0; i < num_bins; i++) {  //simple function for 127 value cap.
		if (bins[i] > 127) {
			bins[i] = 127;
		}
	}

}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;
	int numBlocks;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	numBlocks = (float(inputLength - 1)) / BLOCK_SIZE + 1;
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating device memory");
	//@@ Allocate device memory here
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMalloc((void**)&deviceInput, inputLength * sizeof(float));
	cudaMalloc((void**)&deviceBins, NUM_BINS * sizeof(float));
		wbTime_stop(GPU, "Allocating device memory");

	wbTime_start(GPU, "Copying input host memory to device");
	//@@ Copy input host memory to device
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input host memory to device");

	wbTime_start(GPU, "Clearing the bins on device");
	//@@ zero out the deviceBins using cudaMemset() 
	cudaMemset(deviceBins, 0, NUM_BINS * sizeof(float));
	wbTime_stop(GPU, "Clearing the bins on device");

	//@@ Initialize the grid and block dimensions here
	dim3 GridDim(numBlocks, 1, 1);
	dim3 BlockDim(BLOCK_SIZE, 1, 1);

	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Invoke kernels: first call histogram kernel and then call saturate kernel

	histogram <<< GridDim, BlockDim >>> (deviceInput, deviceBins, inputLength, NUM_BINS);

	CUDA_CHECK(cudaDeviceSynchronize());

	saturate <<< GridDim, BlockDim >>> (deviceBins, NUM_BINS);

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output device memory to host");
	
	//@@ Copy output device memory to host


	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost);

	wbTime_stop(Copy, "Copying output device memory to host");

	wbTime_start(GPU, "Freeing device memory");
	//@@ Free the device memory here
	cudaFree(deviceInput);
	cudaFree(deviceBins);
	wbTime_stop(GPU, "Freeing device memory");

	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);
	return 0;
}
