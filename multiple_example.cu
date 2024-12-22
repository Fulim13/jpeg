#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

using namespace std;

// Print utility functions remain the same
void print_with_tab(const vector<vector<float>> &mat)
{
    for (const auto &row : mat)
    {
        for (const auto &val : row)
        {
            cout << val << "\t";
        }
        cout << endl;
    }
}

__global__ void recenterKernel(int *input, int num_blocks)
{
    int tx = threadIdx.x;       // Position within the 8x8 block (0-7)
    int ty = threadIdx.y;       // Position within the 8x8 block (0-7)
    int block_idx = blockIdx.x; // Which block we're processing

    if (block_idx < num_blocks)
    {
        // Calculate position in global memory
        int idx = block_idx * 64 + ty * 8 + tx;
        // Subtract 128 from each value
        input[idx] -= 128;
    }
}

// Constants for DCT computation
__constant__ float c_dct_table[8][8];

// Initialize DCT coefficient table
void initDCTTable(float *h_dct_table)
{
    for (int i = 0; i < 8; i++)
    {
        float ci = (i == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
        for (int j = 0; j < 8; j++)
        {
            h_dct_table[i * 8 + j] = ci * cosf((2.0f * j + 1.0f) * i * M_PI / 16.0f);
        }
    }
}

// Modified CUDA kernel to process multiple blocks in parallel
__global__ void dctKernel(const int *input, float *output, int num_blocks)
{
    __shared__ float s_block[8][8];
    __shared__ float s_temp[8][8];

    // Thread and block indices
    int tx = threadIdx.x;       // Position within the 8x8 block (0-7)
    int ty = threadIdx.y;       // Position within the 8x8 block (0-7)
    int block_idx = blockIdx.x; // Which image block we're processing

    // Only process if we're within valid blocks
    if (block_idx < num_blocks)
    {
        // Load data into shared memory
        // Each thread loads one pixel from its corresponding image block
        s_block[ty][tx] = input[block_idx * 64 + ty * 8 + tx];
        __syncthreads();

        // Step 1: Row-wise 1D DCT
        float sum = 0.0f;
        for (int j = 0; j < 8; j++)
        {
            sum += s_block[ty][j] * c_dct_table[tx][j];
        }
        s_temp[ty][tx] = sum;
        __syncthreads();

        // Step 2: Column-wise 1D DCT
        sum = 0.0f;
        for (int i = 0; i < 8; i++)
        {
            sum += s_temp[i][tx] * c_dct_table[ty][i];
        }

        // Write result to global memory
        output[block_idx * 64 + ty * 8 + tx] = roundf(sum * 100.0f / 4.0f) / 100.0f;
    }
}

__global__ void quantizeKernel(float *dct_blocks, const int *quantization_table, int num_blocks)
{
    int block_idx = blockIdx.x; // Which block we're processing
    int tx = threadIdx.x;       // Position within the 8x8 block (0-63)

    if (block_idx < num_blocks)
    {
        int idx = block_idx * 64 + tx; // Position in global memory
        dct_blocks[idx] = roundf(dct_blocks[idx] / quantization_table[tx]);
        if (dct_blocks[idx] == -0.0f)
            dct_blocks[idx] = 0.0f; // Handle negative zero
    }
}

__global__ void zigzagKernel(const float *quantized_blocks, int *zigzag_blocks, int num_blocks)
{
    const int zigzag_order[64] = {
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63};

    int block_idx = blockIdx.x; // Which block we're processing
    int tx = threadIdx.x;       // Position within the zigzag array (0-63)

    if (block_idx < num_blocks)
    {
        int idx = block_idx * 64 + tx;
        zigzag_blocks[idx] = static_cast<int>(quantized_blocks[block_idx * 64 + zigzag_order[tx]]);
    }
}

__global__ void runLengthEncodeKernel(const int *zigzag_blocks, int *rle_blocks, int *block_lengths, int num_blocks)
{
    int block_idx = blockIdx.x;

    if (block_idx < num_blocks)
    {
        const int *zigzag = &zigzag_blocks[block_idx * 64];
        int *rle = &rle_blocks[block_idx * 128]; // Max size: 128 (pairs of value and run-length)
        int length = 0;

        int prev_val = zigzag[0];
        int run_length = 0;

        for (int i = 1; i < 64; i++)
        {
            if (zigzag[i] == prev_val)
            {
                run_length++;
            }
            else
            {
                rle[length++] = prev_val;
                rle[length++] = run_length;
                prev_val = zigzag[i];
                run_length = 1;
            }
        }
        rle[length++] = prev_val;
        rle[length++] = run_length;

        block_lengths[block_idx] = length; // Store RLE length
    }
}

void dctAndJpegParallel(const vector<vector<vector<int>>> &input_blocks, const vector<int> &quantization_table,
                        vector<vector<int>> &encoded_blocks)
{
    int num_blocks = input_blocks.size();
    int total_size = num_blocks * 64; // Total number of elements

    // Allocate device memory
    int *d_input, *d_zigzag_blocks, *d_rle_blocks, *d_block_lengths, *d_quantization_table;
    float *d_output;

    cudaMalloc(&d_input, total_size * sizeof(int));
    cudaMalloc(&d_output, total_size * sizeof(float));
    cudaMalloc(&d_zigzag_blocks, total_size * sizeof(int));
    cudaMalloc(&d_rle_blocks, num_blocks * 128 * sizeof(int));
    cudaMalloc(&d_block_lengths, num_blocks * sizeof(int));
    cudaMalloc(&d_quantization_table, 64 * sizeof(int));

    // Prepare quantization table
    cudaMemcpy(d_quantization_table, quantization_table.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);

    // Flatten input blocks for device transfer
    vector<int> input_linear(total_size);
    for (int block = 0; block < num_blocks; block++)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                input_linear[block * 64 + i * 8 + j] = input_blocks[block][i][j];
            }
        }
    }
    cudaMemcpy(d_input, input_linear.data(), total_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch recenter kernel
    dim3 blockDim(8, 8);      // Each thread block is 8x8
    dim3 gridDim(num_blocks); // One block per image
    recenterKernel<<<gridDim, blockDim>>>(d_input, num_blocks);

    // Initialize DCT table and copy to device
    float h_dct_table[64];
    initDCTTable(h_dct_table);
    cudaMemcpyToSymbol(c_dct_table, h_dct_table, 64 * sizeof(float));

    // Launch DCT kernel
    dctKernel<<<gridDim, blockDim>>>(d_input, d_output, num_blocks);

    // Launch quantization kernel
    quantizeKernel<<<num_blocks, 64>>>(d_output, d_quantization_table, num_blocks);

    // Launch zigzag kernel
    zigzagKernel<<<num_blocks, 64>>>(d_output, d_zigzag_blocks, num_blocks);

    // Launch RLE kernel
    runLengthEncodeKernel<<<num_blocks, 1>>>(d_zigzag_blocks, d_rle_blocks, d_block_lengths, num_blocks);

    // Copy results back to host
    vector<int> rle_flattened(num_blocks * 128);
    vector<int> block_lengths(num_blocks);
    cudaMemcpy(rle_flattened.data(), d_rle_blocks, num_blocks * 128 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_lengths.data(), d_block_lengths, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Reformat results
    encoded_blocks.resize(num_blocks);
    for (int b = 0; b < num_blocks; b++)
    {
        encoded_blocks[b].assign(rle_flattened.begin() + b * 128, rle_flattened.begin() + b * 128 + block_lengths[b]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_zigzag_blocks);
    cudaFree(d_rle_blocks);
    cudaFree(d_block_lengths);
    cudaFree(d_quantization_table);
}

int main()
{
    // Input image blocks
    vector<vector<vector<int>>> image_blocks = {
        {{52, 55, 61, 66, 70, 61, 64, 73},
         {63, 59, 55, 90, 109, 85, 69, 72},
         {62, 59, 68, 113, 144, 104, 66, 73},
         {63, 58, 71, 122, 154, 106, 70, 69},
         {67, 61, 68, 104, 126, 88, 68, 70},
         {79, 65, 60, 70, 77, 68, 58, 75},
         {85, 71, 64, 59, 55, 61, 65, 83},
         {87, 79, 69, 68, 65, 76, 78, 94}},
        {{65, 70, 72, 74, 76, 73, 70, 65},
         {68, 65, 60, 58, 56, 55, 60, 70},
         {72, 75, 80, 85, 87, 90, 88, 80},
         {60, 58, 55, 52, 50, 51, 55, 60},
         {70, 75, 80, 85, 90, 95, 100, 105},
         {110, 115, 120, 125, 130, 128, 125, 120},
         {100, 95, 90, 85, 80, 75, 70, 65},
         {60, 55, 50, 45, 40, 38, 35, 30}}};

    // Quantization table
    vector<int> quantization_table = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99};

    // Encode the image blocks
    vector<vector<int>> encoded_blocks;
    dctAndJpegParallel(image_blocks, quantization_table, encoded_blocks);

    // Print the encoded blocks
    for (const auto &block : encoded_blocks)
    {
        cout << "Block:" << endl;
        for (const auto &val : block)
        {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
