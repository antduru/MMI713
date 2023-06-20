#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <vector>


__global__ void calculateHistogram(unsigned char* inputImage, int imageWidth, int imageHeight, int tileWidth, int tileHeight, int* histograms
, int numTilesX, int numTilesY) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tidX = threadIdx.x + blockIdx.x * tileWidth;
    int tidY = threadIdx.y + blockIdx.y * tileHeight;
    int tileIndex = tileX + tileY * numTilesX;
    int pixelIndex = tidX + tidY * imageWidth;
    if (pixelIndex < imageWidth * imageHeight) {
        int pixelValue = inputImage[pixelIndex];
        if (tidX < tileWidth * (blockIdx.x+1) && tidY < tileHeight * (blockIdx.y+1) && tileIndex < numTilesX * numTilesY) {
            atomicAdd(&histograms[tileIndex * 256 + pixelValue], 1);
        }
    }  
}

__global__ void clip_limit(int* histograms, int numTilesX, int numTilesY, float clipLimit) {
	int tileIndex = blockIdx.x;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int shared[256];
    __shared__ int differences[256];
    shared[threadIdx.x] = histograms[tid];
    if (histograms[tid] > round(256.0f  * (clipLimit))) {
        differences[threadIdx.x] = histograms[tid] - round(256.0f * (clipLimit));
        histograms[tid] -= round(256*clipLimit);
    }
    else {
	    differences[threadIdx.x] = 0;
    }
    int result = thrust::reduce(thrust::device, differences, differences + 256, 1);
    float factor = static_cast<float>(result) / static_cast<float>(256);
    if(tileIndex < numTilesX * numTilesY && tid < 256 * numTilesX * numTilesY) {
		histograms[tid] = histograms[tid] + round(factor);
	}
  
}


void cdf_calculation(int* histograms, int numTilesX, int numTilesY) {
    for (int i = 0; i < numTilesX * numTilesY; i++) {
        for (int j = 1; j < 256; j++) {
            histograms[i*256 + j] += histograms[i * 256 + j-1];
        }
    }
}

__global__ void normalize_cdf_and_lut_gpu(int* histograms, int numTilesX, int numTilesY, int width, int height, float* normalized_histograms) {
    int tileIndex = blockIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int blockMin;
    int blockMax;    
    __shared__ int shared[256];
    shared[threadIdx.x] = histograms[tid];
    __syncthreads();
    thrust::pair<int*, int*> result = thrust::minmax_element(thrust::device, shared, shared + blockDim.x);
    blockMin = *result.first;
    blockMax = *result.second;
    if (tileIndex < numTilesX * numTilesY && tid < 256 * numTilesX * numTilesY) {
        normalized_histograms[tid] = 255.0f * static_cast<float>((histograms[tid] - blockMin)) / static_cast<float>((blockMax - blockMin));
    }


}


void normalize_cdf_and_lut(int* histograms, int numTilesX, int numTilesY, int width, int height, float* normalized_histograms) {
    int* mins = new int[numTilesX * numTilesY];
    int* maxs = new int[numTilesX * numTilesY];
    for (int i = 0; i < numTilesX * numTilesY; i++) {
		mins[i] = 256;
		maxs[i] = 0;
	}
    for (int s = 0; s < numTilesX * numTilesY; s++) {
        int* slice = new int[256];
        for (int j = 0; j < 256; j++) {
            slice[j] = histograms[s * 256 + j];
        }
        thrust::pair<int*, int*>(result) = thrust::minmax_element(thrust::host, slice, slice + 256);
        mins[s] = *result.first;
        maxs[s] = *result.second;

    }
    for(int i = 0; i < numTilesX * numTilesY; i++) {
		for (int j = 0; j < 256; j++) {
            histograms[i * 256 + j] += 1;
            if ((maxs[i] - mins[i]) == 0){
                normalized_histograms[i * 256 + j] = histograms[i * 256 + j];
            }	
            else {
                normalized_histograms[i * 256 + j] = (histograms[i * 256 + j] - mins[i]) * 255 / ((maxs[i] - mins[i]));
            }
                   
		}
	}  
}

__global__ void map_intensity(unsigned char* image, int numTilesX, int numTilesY, int width, int height, float* normalized_histograms, int tileHeight, int tileWidth) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tidX = threadIdx.x + blockIdx.x * tileWidth;
    int tidY = threadIdx.y + blockIdx.y * tileHeight;
    int tileIndex = tileX + tileY * numTilesX;
    int pixelIndex = tidX + tidY * width;
    if (pixelIndex < width*height && tileIndex < numTilesX*numTilesY && tidX < width && tidY < height) {
		int pixelValue = image[pixelIndex];
		unsigned char newPixelValue = static_cast<unsigned char>(normalized_histograms[tileIndex * 256 + pixelValue]);
		image[pixelIndex] = newPixelValue;
	}
}

void map_intensity_cpu(unsigned char* image, int numTilesX, int numTilesY, int width, int height, float* normalized_histograms, int tileHeight, int tileWidth) {

    for (int nY = 0; nY < numTilesY; nY++){
        for (int nX = 0; nX < numTilesX; nX++) {
            for (int y = 0; y < tileHeight; y++) {
				for (int x = 0; x < tileWidth; x++) {
					int pixelIndex = x + y * width + nX * tileWidth + nY * tileHeight * width;
					int tileIndex = nX + nY * numTilesX;
					int pixelValue = image[pixelIndex];
					unsigned char newPixelValue = static_cast<unsigned char>(normalized_histograms[tileIndex * 256 + pixelValue]);
					image[pixelIndex] = newPixelValue;
				}
			}

        }
    }
}

void averaging(unsigned char* image, int numTilesX, int numTilesY, int width, int height, int tileWidth, int tileHeight) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int pixelIndex = i + j * width;
            if (pixelIndex % tileWidth == 0 && pixelIndex % tileHeight != 0 && i != width - 1 && i != 0) {
                int pixelValue = static_cast<int>((image[pixelIndex-3] + image[pixelIndex - 2] + image[pixelIndex - 1] + image[pixelIndex + 1] + image[pixelIndex+2] + image[pixelIndex + 3]) / 6);
                image[pixelIndex] = pixelValue;
            }
            else if (pixelIndex % tileHeight == 0 && pixelIndex % tileWidth != 0 && j != 0 && j != height - 1) {
			    int pixelValue = static_cast<int>((image[pixelIndex - 2 * width] + image[pixelIndex - width] + image[pixelIndex + width] + image[pixelIndex + 2 * width]) / 4);
                image[pixelIndex] = pixelValue;
			}
            else if (pixelIndex % tileWidth == 0 && pixelIndex % tileHeight == 0 && i != width - 1 && i != 0 && j != 0 && j != height - 1) {
                int pixelValue = static_cast<int>((image[pixelIndex - 1] + image[pixelIndex + 1] + image[pixelIndex - width] + image[pixelIndex + width]) / 4);
                image[pixelIndex] = pixelValue;
            }
            
        }
    }
}



int main() {
    int width, height, channels;
    unsigned char* image = stbi_load("samples/input_samples/cell.png", &width, &height, &channels, STBI_default);


    if (image == NULL) {
        // Handle image loading error
        std::cout << "Error loading image" << std::endl;
        return 1;
    }

    if (channels > 1) {
        // Convert the image to grayscale
        unsigned char* grayscaleImage = new unsigned char[width * height];
        for (int i = 0; i < width * height; i++) {
            // Compute the grayscale value by taking the average of the color channels
            grayscaleImage[i] = (image[i * channels] + image[i * channels + 1] + image[i * channels + 2]) / 3;
        }
        stbi_image_free(image);  // Free the original image memory
        image = grayscaleImage;  // Set the grayscale image as the new image data
        channels = 1;            // Update the number of channels to 1
    }


    int numTilesX = 16;  
    int numTilesY = 16;

    // Allocate GPU memory for the input image
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, width * height * sizeof(unsigned char));
    cudaMemcpy(d_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch the kernel
    int tileWidth = static_cast<int>(std::floor(static_cast<double>(width) / numTilesX));
    int tileHeight = static_cast<int>(std::floor(static_cast<double>(height) / numTilesY));
    dim3 blockSize(32, 32);
    dim3 gridSize(numTilesX*numTilesX, numTilesY*numTilesY);

    // Allocate GPU memory for the histograms
    int* d_histograms;
    cudaMalloc((void**)&d_histograms, numTilesX * numTilesY * 256 * sizeof(int));

    // Initialize histograms on the host
    int* histograms = new int[numTilesX * numTilesY * 256];
    memset(histograms, 0, numTilesX * numTilesY * 256 * sizeof(int));

    // copy the histograms to the device
    cudaMemcpy(d_histograms, histograms, numTilesX * numTilesY * 256 * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    // Launch the kernel
    calculateHistogram << <gridSize, blockSize >> > (d_image, width, height, tileWidth, tileHeight, d_histograms, numTilesX, numTilesY);
    cudaError_t kernelLaunchError = cudaGetLastError();
    cudaDeviceSynchronize();

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    // Destroy events for kernel1
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    // Print the execution time for kernel1
    std::cout << "Histogram Calculation Execution Time: " << milliseconds1 << " ms" << std::endl;

    // call clip limit
    float clipLimit = 0.1f;
    dim3 gridSizeCLIP(numTilesX * numTilesY, 1);
    dim3 blockSizeCLIP(256, 1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    clip_limit<<<gridSizeCLIP , blockSizeCLIP>>>(d_histograms, numTilesX, numTilesY, clipLimit);
    cudaDeviceSynchronize();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    // Destroy events for kernel1
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    // Print the execution time for kernel1
    std::cout << "Clip Limit Execution Time: " << milliseconds1 << " ms" << std::endl;
   
    // Copy the output histogram from device to host memory
    cudaMemcpy(histograms, d_histograms, numTilesX * numTilesY * 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // call cdf calculation function
    cdf_calculation(histograms, numTilesX, numTilesY);
    printf("CDF Calculation done\n");


    //float* normalized_histograms = new float[numTilesX * numTilesY * 256];
    float* normalized_histograms = (float*)malloc(sizeof(float)* 256 * numTilesX*numTilesY);
    memset(normalized_histograms, 0, numTilesX * numTilesY * 256 * sizeof(float));

    float* d_normalized_histograms;
    cudaMalloc((void**)&d_normalized_histograms, numTilesX * numTilesY * 256 * sizeof(float));
    cudaMemcpy(normalized_histograms, d_normalized_histograms, numTilesX * numTilesY * 256 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_histograms, histograms, numTilesX * numTilesY * 256 * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 gridSizeLUT(numTilesX * numTilesY, 1);
    dim3 blockSizeLUT(256, 1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    normalize_cdf_and_lut_gpu << <gridSizeLUT, blockSizeLUT >> > (d_histograms, numTilesX, numTilesY, width, height, d_normalized_histograms);
    cudaDeviceSynchronize();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    // Destroy events for kernel1
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    // Print the execution time for kernel1
    std::cout << "CDF Normalization Execution Time: " << milliseconds1 << " ms" << std::endl;
    printf("CDF Normalization and LUT creation done\n");

    cudaError_t normalizecdf = cudaGetLastError();
    if (normalizecdf != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(normalizecdf));
        return 1;
    }

    dim3 blockSize2(32,32);
    dim3 gridSize2(numTilesX, numTilesY);
    //cudaMemcpy(d_normalized_histograms, normalized_histograms, numTilesX * numTilesY * 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    map_intensity <<<gridSize2, blockSize2>>> (d_image, numTilesX, numTilesY, width, height, d_normalized_histograms, tileHeight, tileWidth);
    cudaDeviceSynchronize();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    // Destroy events for kernel1
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    // Print the execution time for kernel1
    std::cout << "Intensity Mapping Execution Time: " << milliseconds1 << " ms" << std::endl;
    printf("Intensity Mapping done\n");

    cudaMemcpy(image, d_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaMemcpy(normalized_histograms, d_normalized_histograms, numTilesX * numTilesY * 256 * sizeof(int), cudaMemcpyDeviceToHost);

    averaging(image, numTilesX, numTilesY, width, height, tileWidth, tileHeight);
    
    
    
    // write image to file
    stbi_write_bmp("samples/outputs/cell_output_wo_clip_interpolated.png", width, height, channels, image);
    printf("%s\n", "ImageWritten");


    cudaFree(d_normalized_histograms);
    cudaFree(d_histograms);
	//Free host memory
    delete[] histograms;
    delete[] normalized_histograms;
    cudaFree(d_image);
    stbi_image_free(image);

    return 0;
}
