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
  } while (0)  //copied from previous assignment for fail safe purposes. refer to basicMatrixMultiply.


#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH -1)  //PowerPoint9 Slide51
#define clamp(x) (min(max((x), 0.0), 1.0))

//__constant__ float M[MAX_MASK_WIDTH];
//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values
__global__ void convolution_2D_kernel(float *P, float *N, int height, int width, int channels, float* __restrict__ const M) { //from powerpoint9 slide 52
	//www.acceleware.com/blog/A-Simple-Trick-To-Pass-Constant-Arguments-Into-GPU-Kernels
	//The format as "const float __restrict__ *M" from the book and powerpoint
	//kept giving me an error. I used the format used in the link above to the error I was receiving.

	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH]; //shared memory. credit from figure 7.20 of textbook.

	for (int k = 0; k < channels; k++) {		//this is the loop for the channel outline as suggested from the A2-Instruction outline
		
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int row_o = blockIdx.y * O_TILE_WIDTH + ty;
		int col_o = blockIdx.x * O_TILE_WIDTH + tx;
		int row_i = row_o - MASK_RADIUS;
		int col_i = col_o - MASK_RADIUS;
		int index_i = (row_i * width + col_i) * channels + k;	//this is the input index of the interleave image as suggested from the A2-Instruction outline
		int index_o = (row_o * width + col_o) * channels + k;	//I created an index output as when the final image will be sent to the host through variable N.

		if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {  
			N_ds[ty][tx] = P[index_i];
		}
		else {
			N_ds[ty][tx] = 0.0f;
		}

		__syncthreads();

		float output = 0.0f;
		if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
			for (int i = 0; i < MASK_WIDTH; i++) {
				for (int j = 0; j < MASK_WIDTH; j++) {
					output += N_ds[i + ty][j + tx] * M[i * MASK_WIDTH + j]; 
				}
			}
		}

		if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {	//I have found that the parameters for the data output works correctly with
			N[index_o] = clamp(output);					//(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) as in the previous If-statement.
		}												//The suggested if(row_o < height && col_o < width) works for most of the inputs
														//but fails for a few. 
		__syncthreads();

	}
}
int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
	assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//@@ INSERT CODE HERE
	//allocate device memory
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//@@ INSERT CODE HERE
	//copy host memory to device

	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	//initialize thread block and kernel grid dimensions
	//invoke CUDA kernel	
	dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 DimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
	convolution_2D_kernel <<<DimGrid, DimBlock >>>(deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth, imageChannels, deviceMaskData);
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//@@ INSERT CODE HERE
	//copy results from device to host	
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//@@ INSERT CODE HERE
	//deallocate device memory	
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
