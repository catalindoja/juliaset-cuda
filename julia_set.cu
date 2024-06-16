#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERR(call) { cudaError_t err = call; if(err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err)); exit(1); } }

// Define the CUDA kernel for the Julia set computation
__global__ void julia_kernel(unsigned char *rgb, int w, int h, float xl, float xr, float yb, float yt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < w && j < h) {
        float ci = 0.156;
        float cr = -0.8;
        int k;
        float t;
        float x = ((float)(w - i - 1) * xl + (float)(i) * xr) / (float)(w - 1);
        float y = ((float)(h - j - 1) * yb + (float)(j) * yt) / (float)(h - 1);
        float ar = x;
        float ai = y;

        for (k = 0; k < 200; k++) {
            t  = ar * ar - ai * ai + cr;
            ai = ar * ai + ai * ar + ci;
            ar = t;
            if (1000 < ar * ar + ai * ai) {
                break;
            }
        }

        int juliaValue = (k == 200) ? 1 : 0;

        int index = 3 * (j * w + i);
        rgb[index] = 255 * (1 - juliaValue);
        rgb[index + 1] = 255 * (1 - juliaValue);
        rgb[index + 2] = 255;
    }
}

unsigned char *julia_rgb(int w, int h, float xl, float xr, float yb, float yt) {
    unsigned char *rgb;
    size_t size = w * h * 3 * sizeof(unsigned char);

    // Allocate memory on the host
    rgb = (unsigned char *)malloc(size);
    if (rgb == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(1);
    }

    // Allocate memory on the device
    unsigned char *d_rgb;
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_rgb, size));

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((w + threadsPerBlock.x - 1) / threadsPerBlock.x, (h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel
    julia_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, w, h, xl, xr, yb, yt);
    CHECK_CUDA_ERR(cudaGetLastError());

    // Copy the result back to the host
    CHECK_CUDA_ERR(cudaMemcpy(rgb, d_rgb, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERR(cudaFree(d_rgb));

    return rgb;
}

void tga_write(int w, int h, unsigned char rgb[], char *filename) {
    FILE *file_unit;
    unsigned char header1[12] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned char header2[6] = {w % 256, w / 256, h % 256, h / 256, 24, 0};

    file_unit = fopen(filename, "wb");
    if (file_unit == NULL) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }
    fwrite(header1, sizeof(unsigned char), 12, file_unit);
    fwrite(header2, sizeof(unsigned char), 6, file_unit);
    fwrite(rgb, sizeof(unsigned char), 3 * w * h, file_unit);
    fclose(file_unit);

    printf("\n");
    printf("TGA_WRITE:\n");
    printf("  Graphics data saved as '%s'\n", filename);
}

int main() {
    int size = 16;
    int h = 1000 * size;  // Reduced size for practical execution
    int w = 1000 * size;  // Reduced size for practical execution
    float xl = -1.5;
    float xr = 1.5;
    float yb = -1.5;
    float yt = 1.5;
    clock_t begin = clock();

    printf("\n");
    printf("JULIA_SET:\n");
    printf("  CUDA version.\n");
    printf("  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n");

    unsigned char *rgb = julia_rgb(w, h, xl, xr, yb, yt);

    tga_write(w, h, rgb, "julia_set.tga");

    free(rgb);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\n");
    printf("JULIA_SET:\n");
    printf("Normal end of execution.\n");
    printf("Execution time %f\n", time_spent);
    return 0;
}
