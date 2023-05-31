#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void threshold_kernel(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (index < width * height)
    {
        if (inputImage[3*index] > threshold)
        {
            outputImage[3*index] = 255;     
            outputImage[3*index+1] = 255;   
            outputImage[3*index+2] = 255; 
        }
        else
        {
            outputImage[3*index] = 0;       
            outputImage[3*index+1] = 0;     
            outputImage[3*index+2] = 0;     
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Usage: %s input.ppm output.ppm threshold\n", argv[0]);
        return 1;
    }

    FILE *inputFile = fopen(argv[1], "rb");
    if (!inputFile)
    {
        printf("Error : Could not open input file.\n");
        return 1;
    }

    int width, height, maxValue;
    if (fscanf(inputFile, "P6\n%d %d\n%d\n", &width, &height, &maxValue) != 3)
    {
        printf("Error: Invalid input file format.\n");
        fclose(inputFile);
        return 1;
    }

    unsigned char *inputImage = (unsigned char*)malloc(width * height * 3);
    unsigned char *outputImage = (unsigned char*)malloc(width * height * 3);
    if (!inputImage || !outputImage)
    {
        printf("Error: Could not allocate memory for input/output images.\n");
        fclose(inputFile);
        free(inputImage);
        free(outputImage);
        return 1;
    }

    if (fread(inputImage, sizeof(unsigned char), width * height * 3, inputFile) != width * height * 3)
    {
        printf("Error: Could not read image data from input file.\n");
        fclose(inputFile);
        free(inputImage);
        free(outputImage);
        return 1;
    }

    fclose(inputFile);

    int threshold = atoi(argv[3]);

    unsigned char *dev_inputImage, *dev_outputImage;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&dev_inputImage, width * height * 3);
    if (cudaStatus != cudaSuccess)
    {
        printf("Error: Could not allocate memory on device for input image.\n");
        free(inputImage);
        free(outputImage);
        return 1;
    }
    cudaStatus = cudaMalloc(&dev_outputImage, width * height * 3);
    if (cudaStatus != cudaSuccess)
    {
        printf("Error: Could not allocate memory on device for output image.\n");
        cudaFree(dev_inputImage);
        free(inputImage);
        free(outputImage);
        return 1;
    }
    cudaStatus = cudaMemcpy(dev_inputImage, inputImage, width * height * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Error: Could not copy input image to device memory.\n");
        cudaFree(dev_inputImage);
        cudaFree(dev_outputImage);
        free(inputImage);
        free(outputImage);
        return 1;
    }

    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    threshold_kernel<<<gridSize, blockSize>>>(dev_inputImage, dev_outputImage, width, height, threshold);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Error: Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_inputImage);
        cudaFree(dev_outputImage);
        free(inputImage);
        free(outputImage);
        return 1;
    }

    cudaStatus = cudaMemcpy(outputImage, dev_outputImage, width * height * 3, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("Error: Could not copy output image from device memory.\n");
        cudaFree(dev_inputImage);
        cudaFree(dev_outputImage);
        free(inputImage);
        free(outputImage);
        return 1;
    }

    FILE *outputFile = fopen(argv[2], "wb");
    if (!outputFile)
    {
        printf("Error: Could not open output file.\n");
        cudaFree(dev_inputImage);
        cudaFree(dev_outputImage);
        free(inputImage);
        free(outputImage);
        return 1;
    }

    fprintf(outputFile, "P6\n%d %d\n%d\n", width, height, maxValue);

    if (fwrite(outputImage, sizeof(unsigned char), width * height * 3, outputFile) != width * height * 3)
    {
        printf("Error: Could not write output image data to output file.\n");
        fclose(outputFile);
        cudaFree(dev_inputImage);
        cudaFree(dev_outputImage);
        free(inputImage);
        free(outputImage);
        return 1;
    }

    fclose(outputFile);

    cudaFree(dev_inputImage);
    cudaFree(dev_outputImage);
    free(inputImage);
    free(outputImage);

    return 0;
}