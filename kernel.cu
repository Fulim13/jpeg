#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

using namespace std;

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

__constant__ float c_dct_table[8][8];

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
    int thread_idx = threadIdx.x;

    if (block_idx < num_blocks)
    {
        const int *zigzag = &zigzag_blocks[block_idx * 64];
        int *rle = &rle_blocks[block_idx * 128]; // Max size: 128 (pairs of value and run-length)

        __shared__ int shared_values[64]; // We only need this for values

        // Load values into shared memory
        shared_values[thread_idx] = zigzag[thread_idx];
        __syncthreads();

        int length = 0;
        if (thread_idx == 0)
        {
            // Handle the first element of the block
            int prev_val = shared_values[0];
            int run_length = 1;

            for (int i = 1; i < 64; i++)
            {
                if (shared_values[i] == prev_val)
                {
                    run_length++;
                }
                else
                {
                    rle[length++] = prev_val;
                    rle[length++] = run_length;
                    prev_val = shared_values[i];
                    run_length = 1;
                }
            }
            rle[length++] = prev_val;
            rle[length++] = run_length;

            block_lengths[block_idx] = length; // Store RLE length
        }
    }
}

void encodeGPU(const vector<vector<vector<int>>> &input_blocks, const vector<vector<int>> &quantization_table,
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

    // Flatten quantization table
    vector<int> quantization_flat(64);
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            quantization_flat[i * 8 + j] = quantization_table[i][j];
        }
    }

    // Prepare quantization table
    cudaMemcpy(d_quantization_table, quantization_flat.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);

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
    runLengthEncodeKernel<<<num_blocks, 64>>>(d_zigzag_blocks, d_rle_blocks, d_block_lengths, num_blocks);

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

__global__ void runLengthDecodeKernel(const int *rle_blocks, const int *block_lengths, int *zigzag_blocks, int num_blocks)
{
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (block_idx < num_blocks)
    {
        const int *rle = &rle_blocks[block_idx * 128];
        int *zigzag = &zigzag_blocks[block_idx * 64];
        int length = block_lengths[block_idx];

        // Initialize output array to zeros - each thread handles its position
        if (thread_idx < 64)
        {
            zigzag[thread_idx] = 0;
        }

        // Ensure all threads have initialized their positions
        __syncthreads();

        // Only thread 0 determines the positions for all values
        if (thread_idx == 0)
        {
            int pos = 0;
            for (int i = 0; i < length; i += 2)
            {
                int value = rle[i];
                int run_length = rle[i + 1];
                for (int j = 0; j < run_length && pos < 64; j++)
                {
                    zigzag[pos++] = value;
                }
            }
        }
    }
}

__global__ void inverseZigzagKernel(const int *zigzag_blocks, float *reordered_blocks, int num_blocks)
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
        reordered_blocks[block_idx * 64 + zigzag_order[tx]] = static_cast<float>(zigzag_blocks[idx]);
    }
}

__global__ void dequantizeKernel(float *dct_blocks, const int *quantization_table, int num_blocks)
{
    int block_idx = blockIdx.x;
    int tx = threadIdx.x;

    if (block_idx < num_blocks)
    {
        int idx = block_idx * 64 + tx;
        dct_blocks[idx] = dct_blocks[idx] * quantization_table[tx];
    }
}

__global__ void idctKernel(float *input, int *output, int num_blocks)
{
    __shared__ float s_block[8][8];
    __shared__ float s_temp[8][8];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_idx = blockIdx.x;

    if (block_idx < num_blocks)
    {
        // Load data into shared memory
        s_block[ty][tx] = input[block_idx * 64 + ty * 8 + tx];
        __syncthreads();

        // Step 1: Row-wise 1D IDCT
        float sum = 0.0f;
        for (int j = 0; j < 8; j++)
        {
            float cj = (j == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
            sum += cj * s_block[ty][j] * cosf((2.0f * tx + 1.0f) * j * M_PI / 16.0f);
        }
        s_temp[ty][tx] = sum;
        __syncthreads();

        // Step 2: Column-wise 1D IDCT
        sum = 0.0f;
        for (int i = 0; i < 8; i++)
        {
            float ci = (i == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
            sum += ci * s_temp[i][tx] * cosf((2.0f * ty + 1.0f) * i * M_PI / 16.0f);
        }

        // Scale, round and add 128 back
        output[block_idx * 64 + ty * 8 + tx] = static_cast<int>(roundf(sum / 4.0f)) + 128;
    }
}

void decodeGPU(const vector<vector<int>> &encoded_blocks, const vector<vector<int>> &quantization_table,
               vector<vector<vector<int>>> &decoded_blocks)
{
    int num_blocks = encoded_blocks.size();
    int total_size = num_blocks * 64;

    // Allocate device memory
    int *d_rle_blocks, *d_block_lengths, *d_zigzag_blocks, *d_output;
    float *d_dequantized_blocks;
    int *d_quantization_table;

    cudaMalloc(&d_rle_blocks, num_blocks * 128 * sizeof(int));
    cudaMalloc(&d_block_lengths, num_blocks * sizeof(int));
    cudaMalloc(&d_zigzag_blocks, total_size * sizeof(int));
    cudaMalloc(&d_dequantized_blocks, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(int));
    cudaMalloc(&d_quantization_table, 64 * sizeof(int));

    // Prepare input data
    vector<int> rle_flattened(num_blocks * 128, 0);
    vector<int> block_lengths(num_blocks);

    for (int b = 0; b < num_blocks; b++)
    {
        copy(encoded_blocks[b].begin(), encoded_blocks[b].end(),
             rle_flattened.begin() + b * 128);
        block_lengths[b] = encoded_blocks[b].size();
    }

    // Flatten quantization table
    vector<int> quantization_flat(64);
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            quantization_flat[i * 8 + j] = quantization_table[i][j];
        }
    }

    // Copy data to device
    cudaMemcpy(d_rle_blocks, rle_flattened.data(), num_blocks * 128 * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_lengths, block_lengths.data(), num_blocks * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantization_table, quantization_flat.data(), 64 * sizeof(int),
               cudaMemcpyHostToDevice);

    // Launch kernels
    runLengthDecodeKernel<<<num_blocks, 64>>>(d_rle_blocks, d_block_lengths, d_zigzag_blocks,
                                              num_blocks);
    // print the zigzag blocks
    int *zigzag_blocks = new int[total_size];
    cudaMemcpy(zigzag_blocks, d_zigzag_blocks, total_size * sizeof(int), cudaMemcpyDeviceToHost);

    inverseZigzagKernel<<<num_blocks, 64>>>(d_zigzag_blocks, d_dequantized_blocks, num_blocks);

    // print the dequantized blocks
    float *dequantized_blocks = new float[total_size];
    cudaMemcpy(dequantized_blocks, d_dequantized_blocks, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    dequantizeKernel<<<num_blocks, 64>>>(d_dequantized_blocks, d_quantization_table, num_blocks);

    dim3 blockDim(8, 8);
    dim3 gridDim(num_blocks);
    idctKernel<<<gridDim, blockDim>>>(d_dequantized_blocks, d_output, num_blocks);

    // Copy results back to host
    vector<int> output_linear(total_size);
    cudaMemcpy(output_linear.data(), d_output, total_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Reshape output into blocks
    decoded_blocks.resize(num_blocks, vector<vector<int>>(8, vector<int>(8)));
    for (int block = 0; block < num_blocks; block++)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                decoded_blocks[block][i][j] = output_linear[block * 64 + i * 8 + j];
            }
        }
    }

    // Free device memory
    cudaFree(d_rle_blocks);
    cudaFree(d_block_lengths);
    cudaFree(d_zigzag_blocks);
    cudaFree(d_dequantized_blocks);
    cudaFree(d_output);
    cudaFree(d_quantization_table);
}
