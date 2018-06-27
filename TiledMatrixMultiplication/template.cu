#include <wb.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
  //@@ Insert code to implement tiled matrix multiplication here
  //@@ You have to use shared memory to write this kernel
	#define TILE_WIDTH 16	//Chapter 4 page 98 from 3rd Edition of book.

	__shared__ float Ads[TILE_WIDTH][TILE_WIDTH]; //Chapter 4 page 90-91 from 3rd Edition of book.
	__shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Cvalue = 0;

	for(int ch = 0; ch < (TILE_WIDTH + numAColumns - 1) / TILE_WIDTH; ch++) {
		if (Row < numARows && ch * TILE_WIDTH + tx < numAColumns) {
			Ads[ty][tx] = A[Row * numAColumns + ch * TILE_WIDTH + tx];
		}
		else {
			Ads[ty][tx] = 0.0;
		}
		if (ch * TILE_WIDTH + ty < numBRows && Col < numBColumns) {
			Bds[ty][tx] = B[(ch * TILE_WIDTH + ty) * numBColumns + Col];
		}
		else {
			Bds[ty][tx] = 0.0;
		}

		__syncthreads();

		for(int k = 0; k < TILE_WIDTH; k++) {
			Cvalue += Ads[ty][k] * Bds[k][tx];
		}
		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) {
		C[((by*blockDim.y + ty)*numCColumns) + (bx*blockDim.x) + tx] = Cvalue;
	}
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  int allocSizeC = numCRows * numCColumns * sizeof(float);
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  int allocSizeA = sizeof(float) * numARows * numAColumns;
  int allocSizeB = sizeof(float) * numBRows * numBColumns;

  cudaMalloc((void **)&deviceA, allocSizeA);
  cudaMalloc((void **)&deviceB, allocSizeB);
  cudaMalloc((void **)&deviceC, allocSizeC);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceA, hostA, allocSizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, allocSizeB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(16, 16, 1);
  dim3 DimGrid((numBColumns - 1) / 16 + 1, (numARows - 1) / 16 + 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared <<<DimGrid, DimBlock >>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostC, deviceC, allocSizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
