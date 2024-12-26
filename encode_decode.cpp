#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <string>
#include <chrono>
#include "kernel.cuh"

using namespace std;
using namespace cv;
using namespace std::chrono;

Mat readImage(const string &image_name)
{
    Mat image = imread(image_name);
    if (image.empty())
    {
        cerr << "Could not open or find the image" << endl;
        return Mat();
    }
    return image;
}

void ensureMultipleOf16(Mat &image)
{
    if (image.rows % 16 != 0 || image.cols % 16 != 0)
    {
        int newWidth = (image.cols / 16) * 16;
        int newHeight = (image.rows / 16) * 16;
        resize(image, image, Size(newWidth, newHeight), 0, 0, INTER_AREA);
    }
}

Mat RGB2YCbCr(const Mat &image)
{
    const array<double, 3> offset = {0.0, 128.0, 128.0};
    const double ycbcr_transform[3][3] = {
        {0.299, 0.587, 0.114},
        {-0.1687, -0.3313, 0.5},
        {0.5, -0.4187, -0.0813}};

    Mat ycbcr_image = Mat::zeros(image.size(), CV_8UC3);

    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            Vec3b pixel = image.at<Vec3b>(i, j);
            array<double, 3> transformed_pixel = {0.0, 0.0, 0.0};

            for (int k = 0; k < 3; ++k)
            {
                transformed_pixel[k] = offset[k];
                for (int l = 0; l < 3; ++l)
                {
                    transformed_pixel[k] += ycbcr_transform[k][l] * pixel[l];
                }
                transformed_pixel[k] = round(transformed_pixel[k]);
            }

            // Set the transformed Y, Cb, Cr values in the ycbcr_image
            ycbcr_image.at<Vec3b>(i, j) = Vec3b(
                static_cast<uchar>(transformed_pixel[0]),
                static_cast<uchar>(transformed_pixel[1]),
                static_cast<uchar>(transformed_pixel[2]));
        }
    }

    return ycbcr_image;
}

// Function to perform chroma subsampling (4:2:0)
void chromaSubsampling(const Mat &input, Mat &Y, Mat &Cb, Mat &Cr)
{
    int width = input.cols;
    int height = input.rows;

    // Full resolution Y channel
    Y = Mat(height, width, CV_8UC1);
    // Quarter resolution Cb, Cr channels
    Cb = Mat(height / 2, width / 2, CV_8UC1);
    Cr = Mat(height / 2, width / 2, CV_8UC1);

    for (int i = 0; i < height; i += 2)
    {
        for (int j = 0; j < width; j += 2)
        {
            // Store full-res Y values
            Y.at<uchar>(i, j) = input.at<Vec3b>(i, j)[0];
            Y.at<uchar>(i + 1, j) = input.at<Vec3b>(i + 1, j)[0];
            Y.at<uchar>(i, j + 1) = input.at<Vec3b>(i, j + 1)[0];
            Y.at<uchar>(i + 1, j + 1) = input.at<Vec3b>(i + 1, j + 1)[0];

            // Store averaged Cb, Cr at quarter resolution
            Cb.at<uchar>(i / 2, j / 2) = (input.at<Vec3b>(i, j)[1] +
                                          input.at<Vec3b>(i + 1, j)[1] +
                                          input.at<Vec3b>(i, j + 1)[1] +
                                          input.at<Vec3b>(i + 1, j + 1)[1]) /
                                         4;

            Cr.at<uchar>(i / 2, j / 2) = (input.at<Vec3b>(i, j)[2] +
                                          input.at<Vec3b>(i + 1, j)[2] +
                                          input.at<Vec3b>(i, j + 1)[2] +
                                          input.at<Vec3b>(i + 1, j + 1)[2]) /
                                         4;
        }
    }
}

struct CompressionStats
{
    size_t original_size;        // Original image size (in bytes)
    size_t compressed_size;      // Compressed image size (in bytes)
    double compression_ratio;    // Compression ratio (original / compressed)
    double percentage_reduction; // Percentage reduction in size
    double original_size_kb;     // Original size in KB
    double compressed_size_kb;   // Compressed size in KB
};

CompressionStats calculateCompressionStats(const string &original_filename, const string &compressed_filename)
{
    CompressionStats stats;

    // Open the original file and calculate its size on disk
    ifstream original_file(original_filename, ios::binary | ios::ate); // Open file and move to the end
    if (!original_file)
    {
        throw runtime_error("Cannot open the original file: " + original_filename);
    }

    stats.original_size = original_file.tellg(); // Get the size of the original file

    // Convert original size to KB
    stats.original_size_kb = static_cast<double>(stats.original_size) / 1024.0;

    // Open the compressed file and calculate its size
    ifstream file(compressed_filename, ios::binary | ios::ate); // Open file and move to the end
    if (!file)
    {
        throw runtime_error("Cannot open the compressed file: " + compressed_filename);
    }

    stats.compressed_size = file.tellg(); // Get the size of the compressed file

    // Convert compressed size to KB
    stats.compressed_size_kb = static_cast<double>(stats.compressed_size) / 1024.0;

    // Calculate compression ratio
    if (stats.compressed_size == 0)
    {
        stats.compression_ratio = 0;
        stats.percentage_reduction = 0;
    }
    else
    {
        stats.compression_ratio = static_cast<double>(stats.original_size) / stats.compressed_size;
        stats.percentage_reduction = (1.0 - static_cast<double>(stats.compressed_size) / stats.original_size) * 100;
    }

    file.close();
    return stats;
}

void extractBlocks(const Mat &Y, vector<vector<vector<int>>> &blocks)
{
    int height = Y.rows;
    int width = Y.cols;

    for (int i = 0; i < height; i += 8)
    {
        for (int j = 0; j < width; j += 8)
        {
            vector<vector<int>> block(8, vector<int>(8, 0));

            for (int x = 0; x < 8; ++x)
            {
                for (int y = 0; y < 8; ++y)
                {
                    if (i + x < height && j + y < width)
                    {
                        block[x][y] = Y.at<uchar>(i + x, j + y);
                    }
                }
            }

            blocks.push_back(block);
        }
    }
}

void dct(const vector<vector<int>> &image_block, vector<vector<float>> &dct_block)
{
    // Iterate over each coefficient in the 8x8 DCT matrix
    for (int u = 0; u < 8; u++)
    {
        for (int v = 0; v < 8; v++)
        {
            float cu = (u == 0) ? 1 / sqrt(2) : 1.0;
            float cv = (v == 0) ? 1 / sqrt(2) : 1.0;

            float sum_val = 0;

            // Perform the summation of the DCT formula
            for (int x = 0; x < 8; x++)
            {
                for (int y = 0; y < 8; y++)
                {
                    sum_val += image_block[x][y] *
                               cos((2 * x + 1) * u * M_PI / 16) *
                               cos((2 * y + 1) * v * M_PI / 16);
                }
            }

            // 100 / 100 because of round to 2 decimal places
            dct_block[u][v] = round(0.25 * cu * cv * sum_val * 100) / 100;
        }
    }
}

void recenterAroundZero(vector<vector<int>> &mat)
{
    for (auto &row : mat)
    {
        for (auto &val : row)
        {
            val -= 128;
        }
    }
}

void quantize(vector<vector<float>> &dct_block, const vector<vector<int>> &quantization_table)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            float quantized_value = round(dct_block[i][j] / quantization_table[i][j]);
            dct_block[i][j] = (quantized_value == -0.0f) ? 0.0f : quantized_value;
        }
    }
}

void zigzag_scan(const vector<vector<float>> &block, vector<int> &zigzag)
{
    zigzag.clear();     // Clear the vector to ensure it's empty
    zigzag.reserve(64); // Reserve space for 64 elements

    for (int i = 0; i < 15; i++)
    {
        if (i < 8)
        {
            for (int j = 0; j <= i; j++)
            {
                if (i % 2 == 0)
                {
                    zigzag.push_back(static_cast<int>(block[i - j][j]));
                }
                else
                {
                    zigzag.push_back(static_cast<int>(block[j][i - j]));
                }
            }
        }
        else
        {
            for (int j = 0; j < 15 - i; j++)
            {
                if (i % 2 == 0)
                {
                    zigzag.push_back(static_cast<int>(block[7 - j][j + (i - 7)]));
                }
                else
                {
                    zigzag.push_back(static_cast<int>(block[j + (i - 7)][7 - j]));
                }
            }
        }
    }
}

vector<int> run_length_encode_ac(const vector<int> &ac_coefficients)
{
    vector<int> rle_encoded_ac;
    int run_length = 1;
    int prev_chr = ac_coefficients[0];
    for (size_t i = 1; i < ac_coefficients.size(); ++i)
    {
        if (ac_coefficients[i] == prev_chr)
        {
            run_length++;
        }
        else
        {
            rle_encoded_ac.push_back(prev_chr);
            rle_encoded_ac.push_back(run_length);
            prev_chr = ac_coefficients[i];
            run_length = 1;
        }
    }
    // Add the last entry
    rle_encoded_ac.push_back(prev_chr);
    rle_encoded_ac.push_back(run_length);
    return rle_encoded_ac;
}

unordered_map<int, int> build_frequency_dict(const vector<int> &data)
{
    unordered_map<int, int> frequency_dict;
    for (int val : data)
    {
        frequency_dict[val]++;
    }
    return frequency_dict;
}

struct HuffmanNode
{
    int value;
    int frequency;
    HuffmanNode *left;
    HuffmanNode *right;

    HuffmanNode(int val, int freq) : value(val), frequency(freq), left(nullptr), right(nullptr) {}
};

struct Compare
{
    bool operator()(HuffmanNode *left, HuffmanNode *right)
    {
        return left->frequency > right->frequency;
    }
};

HuffmanNode *build_huffman_tree(const unordered_map<int, int> &freq_dict)
{
    priority_queue<HuffmanNode *, vector<HuffmanNode *>, Compare> min_heap;

    // Create a leaf node for each value and push it into the priority queue
    for (const auto &pair : freq_dict)
    {
        min_heap.push(new HuffmanNode(pair.first, pair.second));
    }

    // Iterate until only one node remains in the priority queue
    while (min_heap.size() > 1)
    {
        HuffmanNode *left = min_heap.top();
        min_heap.pop();
        HuffmanNode *right = min_heap.top();
        min_heap.pop();

        // Create a new internal node with combined frequency
        HuffmanNode *internal = new HuffmanNode(99999, left->frequency + right->frequency);
        internal->left = left;
        internal->right = right;

        // Push the internal node back into the priority queue
        min_heap.push(internal);
    }

    // The last node in the queue is the root of the Huffman tree
    return min_heap.top();
}

void generate_huffman_codes(HuffmanNode *node, const string &code, unordered_map<int, string> &huffman_codes)
{
    if (!node)
        return;

    if (node->value != 99999)
    {
        huffman_codes[node->value] = code;
    }

    generate_huffman_codes(node->left, code + "0", huffman_codes);
    generate_huffman_codes(node->right, code + "1", huffman_codes);
}

string huffman_encode(const vector<int> &data, const unordered_map<int, string> &huffman_codes)
{
    string encoded_str;
    for (int val : data)
    {
        encoded_str += huffman_codes.at(val);
    }
    return encoded_str;
}

vector<int> huffman_decode(const string &encoded_str, HuffmanNode *root)
{
    vector<int> decoded_values;
    HuffmanNode *current = root;
    for (char bit : encoded_str)
    {
        current = (bit == '0') ? current->left : current->right;
        if (!current->left && !current->right)
        {
            decoded_values.push_back(current->value);
            current = root;
        }
    }
    return decoded_values;
}

vector<vector<int>> decodeRLE(const vector<int> &rle_encoded)
{
    vector<vector<int>> blocks;
    vector<int> current_block;
    int current_length = 0;

    for (size_t i = 0; i < rle_encoded.size(); i += 2)
    {
        int value = rle_encoded[i];         // Value at odd index
        int frequency = rle_encoded[i + 1]; // Frequency at even index

        // Add "frequency" occurrences of "value" to the current block
        for (int j = 0; j < frequency; ++j)
        {
            current_block.push_back(value);
            ++current_length;

            // If the block size reaches 64, save it and start a new block
            if (current_length == 64)
            {
                blocks.push_back(current_block);
                current_block.clear();
                current_length = 0;
            }
        }
    }

    // If there is remaining data in the current block, add it to blocks
    if (!current_block.empty())
    {
        blocks.push_back(current_block);
    }

    return blocks;
}

void inverse_zigzag(const vector<int> &zigzag, vector<vector<float>> &block)
{
    block.assign(8, vector<float>(8, 0.0)); // Initialize the 8x8 block with zeros
    int index = 0;

    for (int i = 0; i < 15; i++)
    {
        if (i < 8)
        {
            for (int j = 0; j <= i; j++)
            {
                if (i % 2 == 0)
                {
                    block[i - j][j] = zigzag[index++];
                }
                else
                {
                    block[j][i - j] = zigzag[index++];
                }
            }
        }
        else
        {
            for (int j = 0; j < 15 - i; j++)
            {
                if (i % 2 == 0)
                {
                    block[7 - j][j + (i - 7)] = zigzag[index++];
                }
                else
                {
                    block[j + (i - 7)][7 - j] = zigzag[index++];
                }
            }
        }
    }
}

void inverse_quantize(vector<vector<float>> &dct_block, const vector<vector<int>> &quantization_table)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            dct_block[i][j] *= quantization_table[i][j];
        }
    }
}

void idct(const vector<vector<float>> &dct_block, vector<vector<int>> &image_block)
{
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            float sum_val = 0.0;

            for (int u = 0; u < 8; u++)
            {
                for (int v = 0; v < 8; v++)
                {
                    float cu = (u == 0) ? 1 / sqrt(2) : 1.0;
                    float cv = (v == 0) ? 1 / sqrt(2) : 1.0;

                    sum_val += cu * cv * dct_block[u][v] *
                               cos((2 * x + 1) * u * M_PI / 16) *
                               cos((2 * y + 1) * v * M_PI / 16);
                }
            }

            // Scale by 1/4 and round to the nearest integer
            image_block[x][y] = round(0.25 * sum_val);
        }
    }
}

void add_back_128(vector<vector<int>> &image_block)
{
    for (auto &row : image_block)
    {
        for (auto &val : row)
        {
            val += 128;
        }
    }
}

struct EncodedData
{
    string huffman_encoded_str;
    HuffmanNode *huffman_tree;
};

EncodedData encodeChannel(const Mat &channel, const vector<vector<int>> &quantization_table)
{
    // Extract 8x8 blocks
    vector<vector<vector<int>>> blocks;
    extractBlocks(channel, blocks);

    // Vector to store encoded values for all blocks
    vector<int> combined_encoded_values;

    // Process each block
    for (auto &block : blocks)
    {
        // Recenter around zero
        recenterAroundZero(block);

        // Perform DCT
        vector<vector<float>> dct_block(8, vector<float>(8, 0.0));
        dct(block, dct_block);

        // Quantization
        quantize(dct_block, quantization_table);

        // Zigzag scan
        vector<int> zigzag;
        zigzag_scan(dct_block, zigzag);

        // Run-length encoding
        vector<int> rle_encoded = run_length_encode_ac(zigzag);

        // Add encoded values to combined vector
        combined_encoded_values.insert(combined_encoded_values.end(),
                                       rle_encoded.begin(),
                                       rle_encoded.end());
    }

    // Build Huffman tree and codes
    unordered_map<int, int> freq_dict = build_frequency_dict(combined_encoded_values);
    HuffmanNode *huffman_tree = build_huffman_tree(freq_dict);

    unordered_map<int, string> huffman_codes;
    generate_huffman_codes(huffman_tree, "", huffman_codes);

    // Perform Huffman encoding
    string huffman_encoded_str = huffman_encode(combined_encoded_values, huffman_codes);

    return {huffman_encoded_str, huffman_tree};
}

Mat decodeChannel(const EncodedData &encoded_data, int height, int width, const vector<vector<int>> &quantization_table)
{
    // Huffman decode
    vector<int> decoded_values = huffman_decode(encoded_data.huffman_encoded_str,
                                                encoded_data.huffman_tree);

    // RLE decode
    vector<vector<int>> decoded_blocks = decodeRLE(decoded_values);

    // Create output matrix
    Mat reconstructed = Mat::zeros(height, width, CV_8UC1);

    // Process each block
    int block_idx = 0;
    for (int i = 0; i < height; i += 8)
    {
        for (int j = 0; j < width; j += 8)
        {
            if (block_idx >= decoded_blocks.size())
                break;

            // Inverse zigzag
            vector<vector<float>> restored_block(8, vector<float>(8, 0.0));
            inverse_zigzag(decoded_blocks[block_idx], restored_block);

            // Inverse quantization
            inverse_quantize(restored_block, quantization_table);

            // Inverse DCT
            vector<vector<int>> reconstructed_block(8, vector<int>(8, 0));
            idct(restored_block, reconstructed_block);

            // Add back 128
            add_back_128(reconstructed_block);

            // Copy block to output matrix
            for (int x = 0; x < 8; x++)
            {
                for (int y = 0; y < 8; y++)
                {
                    if (i + x < height && j + y < width)
                    {
                        reconstructed.at<uchar>(i + x, j + y) =
                            static_cast<uchar>(reconstructed_block[x][y]);
                    }
                }
            }

            block_idx++;
        }
    }

    return reconstructed;
}

EncodedData encodeChannelGPU(const Mat &channel, const vector<vector<int>> &quantization_table)
{
    // Extract 8x8 blocks
    vector<vector<vector<int>>> blocks;
    extractBlocks(channel, blocks);

    // Vector to store encoded values for all blocks
    vector<vector<int>> encoded_values;

    encodeGPU(blocks, quantization_table, encoded_values);

    vector<int> combined_encoded_values;
    for (auto &block : encoded_values)
    {
        combined_encoded_values.insert(combined_encoded_values.end(),
                                       block.begin(),
                                       block.end());
    }
    // Build Huffman tree and codes
    unordered_map<int, int>
        freq_dict = build_frequency_dict(combined_encoded_values);
    HuffmanNode *huffman_tree = build_huffman_tree(freq_dict);

    unordered_map<int, string> huffman_codes;
    generate_huffman_codes(huffman_tree, "", huffman_codes);

    // Perform Huffman encoding
    string huffman_encoded_str = huffman_encode(combined_encoded_values, huffman_codes);

    return {huffman_encoded_str, huffman_tree};
}

Mat decodeChannelGPU(const EncodedData &encoded_data, int height, int width, const vector<vector<int>> &quantization_table)
{
    // Huffman decode
    vector<int> decoded_values = huffman_decode(encoded_data.huffman_encoded_str,
                                                encoded_data.huffman_tree);
    vector<vector<int>> blocks;
    vector<int> current_block;
    int current_length = 0;

    for (size_t i = 0; i < decoded_values.size(); i += 2)
    {
        int value = decoded_values[i];         // Value
        int frequency = decoded_values[i + 1]; // Frequency

        while (frequency > 0)
        {
            int to_add = min(64 - current_length, frequency);
            current_block.push_back(value);
            current_block.push_back(to_add);

            current_length += to_add;
            frequency -= to_add;

            if (current_length == 64)
            {
                blocks.push_back(current_block);
                current_block.clear();
                current_length = 0;
            }
        }
    }

    // Add any remaining elements to the last block
    if (!current_block.empty())
    {
        blocks.push_back(current_block);
    }

    vector<vector<vector<int>>> image_blocks;

    decodeGPU(blocks, quantization_table, image_blocks);

    // Create output matrix
    Mat reconstructed = Mat::zeros(height, width, CV_8UC1);

    // Fill the reconstructed Mat from image_blocks
    int block_size = 8;
    int blocks_per_row = (width + block_size - 1) / block_size;
    int blocks_per_col = (height + block_size - 1) / block_size;

    for (int block_row = 0; block_row < blocks_per_col; ++block_row)
    {
        for (int block_col = 0; block_col < blocks_per_row; ++block_col)
        {
            int block_index = block_row * blocks_per_row + block_col;
            if (block_index >= image_blocks.size())
                break;

            // Get the current block
            const auto &block = image_blocks[block_index];

            // Place the block into the Mat
            for (int i = 0; i < block_size; ++i)
            {
                for (int j = 0; j < block_size; ++j)
                {
                    int row = block_row * block_size + i;
                    int col = block_col * block_size + j;

                    // Ensure we don't go out of bounds
                    if (row < height && col < width)
                    {
                        reconstructed.at<uchar>(row, col) = static_cast<uchar>(block[i][j]);
                    }
                }
            }
        }
    }

    return reconstructed;
}

// Function to serialize Huffman tree for storage
void serializeHuffmanTree(HuffmanNode *root, ofstream &file)
{
    // Write whether this node is null (0) or not (1)
    bool isNotNull = (root != nullptr);
    file.write(reinterpret_cast<char *>(&isNotNull), sizeof(bool));

    if (root)
    {
        // Write node data
        file.write(reinterpret_cast<char *>(&root->value), sizeof(int));
        file.write(reinterpret_cast<char *>(&root->frequency), sizeof(int));

        // Recursively serialize left and right subtrees
        serializeHuffmanTree(root->left, file);
        serializeHuffmanTree(root->right, file);
    }
}

// Function to deserialize Huffman tree from storage
HuffmanNode *deserializeHuffmanTree(ifstream &file)
{
    bool isNotNull;
    file.read(reinterpret_cast<char *>(&isNotNull), sizeof(bool));

    if (!isNotNull)
    {
        return nullptr;
    }

    int value, frequency;
    file.read(reinterpret_cast<char *>(&value), sizeof(int));
    file.read(reinterpret_cast<char *>(&frequency), sizeof(int));

    HuffmanNode *node = new HuffmanNode(value, frequency);
    node->left = deserializeHuffmanTree(file);
    node->right = deserializeHuffmanTree(file);

    return node;
}

// Function to convert a Huffman-encoded string to a bitstream (vector of bytes)
vector<unsigned char> stringToBitstream(const string &huffman_data)
{
    vector<unsigned char> bitstream;
    unsigned char current_byte = 0;
    int bit_pos = 0;

    for (char bit : huffman_data)
    {
        // Set the corresponding bit in the current byte
        if (bit == '1')
        {
            current_byte |= (1 << (7 - bit_pos)); // Set bit to 1
        }
        // Else bit is '0', no action needed since the bit is already 0

        // Move to the next bit in the current byte
        bit_pos++;

        // If the current byte is full (8 bits), add it to the bitstream
        if (bit_pos == 8)
        {
            bitstream.push_back(current_byte);
            current_byte = 0; // Reset current byte
            bit_pos = 0;      // Reset bit position
        }
    }

    // If there are leftover bits in the current byte, push it to the bitstream
    if (bit_pos > 0)
    {
        bitstream.push_back(current_byte);
    }

    return bitstream;
}

// Function to convert a bitstream (vector of bytes) back to a Huffman-encoded string
string bitstreamToString(const vector<unsigned char> &bitstream, int total_bits)
{
    string huffman_data;
    int bit_count = 0;

    for (unsigned char byte : bitstream)
    {
        for (int i = 7; i >= 0; --i)
        {
            if (bit_count == total_bits)
                return huffman_data; // Stop when we've processed all bits

            huffman_data.push_back((byte & (1 << i)) ? '1' : '0');
            ++bit_count;
        }
    }

    return huffman_data; // Return the reconstructed Huffman string
}

void saveEncodedData(const string &filename,
                     const EncodedData &y_data, const EncodedData &cb_data, const EncodedData &cr_data,
                     int y_rows, int y_cols, int cb_rows, int cb_cols, int cr_rows, int cr_cols)
{
    ofstream file(filename, ios::binary);
    if (!file)
    {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    // Write dimensions for each channel
    file.write(reinterpret_cast<const char *>(&y_rows), sizeof(int));
    file.write(reinterpret_cast<const char *>(&y_cols), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cb_rows), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cb_cols), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cr_rows), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cr_cols), sizeof(int));

    // Write encoded strings lengths
    int y_len = y_data.huffman_encoded_str.length();
    int cb_len = cb_data.huffman_encoded_str.length();
    int cr_len = cr_data.huffman_encoded_str.length();

    file.write(reinterpret_cast<const char *>(&y_len), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cb_len), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cr_len), sizeof(int));

    // Convert Huffman strings to bitstreams
    vector<unsigned char> y_bitstream = stringToBitstream(y_data.huffman_encoded_str);
    vector<unsigned char> cb_bitstream = stringToBitstream(cb_data.huffman_encoded_str);
    vector<unsigned char> cr_bitstream = stringToBitstream(cr_data.huffman_encoded_str);

    // Write the bitstream data to file
    file.write(reinterpret_cast<const char *>(y_bitstream.data()), y_bitstream.size());
    file.write(reinterpret_cast<const char *>(cb_bitstream.data()), cb_bitstream.size());
    file.write(reinterpret_cast<const char *>(cr_bitstream.data()), cr_bitstream.size());

    // Write Huffman trees
    serializeHuffmanTree(y_data.huffman_tree, file);
    serializeHuffmanTree(cb_data.huffman_tree, file);
    serializeHuffmanTree(cr_data.huffman_tree, file);

    file.close();
}

void loadEncodedData(const string &filename,
                     EncodedData &y_data, EncodedData &cb_data, EncodedData &cr_data,
                     int &y_rows, int &y_cols, int &cb_rows, int &cb_cols, int &cr_rows, int &cr_cols)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        throw runtime_error("Cannot open file for reading: " + filename);
    }

    // Read dimensions
    file.read(reinterpret_cast<char *>(&y_rows), sizeof(int));
    file.read(reinterpret_cast<char *>(&y_cols), sizeof(int));
    file.read(reinterpret_cast<char *>(&cb_rows), sizeof(int));
    file.read(reinterpret_cast<char *>(&cb_cols), sizeof(int));
    file.read(reinterpret_cast<char *>(&cr_rows), sizeof(int));
    file.read(reinterpret_cast<char *>(&cr_cols), sizeof(int));

    // Read string lengths
    int y_len, cb_len, cr_len;
    file.read(reinterpret_cast<char *>(&y_len), sizeof(int));
    file.read(reinterpret_cast<char *>(&cb_len), sizeof(int));
    file.read(reinterpret_cast<char *>(&cr_len), sizeof(int));

    // Read the bitstream data from file
    vector<unsigned char> y_bitstream(y_len / 8 + 1), cb_bitstream(cb_len / 8 + 1), cr_bitstream(cr_len / 8 + 1);
    file.read(reinterpret_cast<char *>(y_bitstream.data()), y_bitstream.size());
    file.read(reinterpret_cast<char *>(cb_bitstream.data()), cb_bitstream.size());
    file.read(reinterpret_cast<char *>(cr_bitstream.data()), cr_bitstream.size());

    // Convert the bitstreams back to Huffman-encoded strings
    y_data.huffman_encoded_str = bitstreamToString(y_bitstream, y_len);
    cb_data.huffman_encoded_str = bitstreamToString(cb_bitstream, cb_len);
    cr_data.huffman_encoded_str = bitstreamToString(cr_bitstream, cr_len);

    // Read Huffman trees
    y_data.huffman_tree = deserializeHuffmanTree(file);
    cb_data.huffman_tree = deserializeHuffmanTree(file);
    cr_data.huffman_tree = deserializeHuffmanTree(file);

    file.close();
}

double mainEncode(const Mat &Y, const Mat &Cb, const Mat &Cr,
                  const vector<vector<int>> &quantization_table_Y,
                  const vector<vector<int>> &quantization_table_CbCr,
                  EncodedData &y_encoded, EncodedData &cb_encoded, EncodedData &cr_encoded, string platform)
{
    // Measure the start of encoding time
    auto start_encoding = high_resolution_clock::now();

    // Encode each channel
    if (platform == "CPU")
    {
        y_encoded = encodeChannel(Y, quantization_table_Y);
        cb_encoded = encodeChannel(Cb, quantization_table_CbCr);
        cr_encoded = encodeChannel(Cr, quantization_table_CbCr);
    }
    else
    {
        y_encoded = encodeChannelGPU(Y, quantization_table_Y);
        cb_encoded = encodeChannelGPU(Cb, quantization_table_CbCr);
        cr_encoded = encodeChannelGPU(Cr, quantization_table_CbCr);
    }
    // Measure the end of encoding time
    auto stop_encoding = high_resolution_clock::now();

    // Print encoding time
    auto duration_encoding = duration_cast<milliseconds>(stop_encoding - start_encoding);
    return duration_encoding.count();
}

double mainDecode(const EncodedData &y_loaded, const EncodedData &cb_loaded, const EncodedData &cr_loaded,
                  int y_rows, int y_cols, int cb_rows, int cb_cols, int cr_rows, int cr_cols,
                  const vector<vector<int>> &quantization_table_Y,
                  const vector<vector<int>> &quantization_table_CbCr,
                  Mat &Y_reconstructed, Mat &Cb_reconstructed, Mat &Cr_reconstructed, string platform)
{
    // Measure decoding time for GPU
    auto start_decoding = high_resolution_clock::now();

    // Decode each channel
    if (platform == "CPU")
    {
        Y_reconstructed = decodeChannel(y_loaded, y_rows, y_cols, quantization_table_Y);
        Cb_reconstructed = decodeChannel(cb_loaded, cb_rows, cr_rows, quantization_table_CbCr);
        Cr_reconstructed = decodeChannel(cr_loaded, cr_rows, cr_cols, quantization_table_CbCr);
    }
    else
    {
        Y_reconstructed = decodeChannelGPU(y_loaded, y_rows, y_cols, quantization_table_Y);
        Cb_reconstructed = decodeChannelGPU(cb_loaded, cb_rows, cr_rows, quantization_table_CbCr);
        Cr_reconstructed = decodeChannelGPU(cr_loaded, cr_rows, cr_cols, quantization_table_CbCr);
    }

    // Measure the end of decoding time
    auto end_decoding = high_resolution_clock::now();

    // Calculate decoding time
    auto duration_decoding = duration_cast<milliseconds>(end_decoding - start_decoding);
    return duration_decoding.count();
}

vector<vector<int>> quantization_table_Y = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99},
};

vector<vector<int>> quantization_table_CbCr = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
};

struct ImageMetric
{
    double PSNR;
    double MSE;
};

// Function to calculate Mean Squared Error (MSE)
double calculateMSE(const Mat &original, const Mat &reconstructed)
{
    if (original.size() != reconstructed.size() || original.type() != reconstructed.type())
    {
        throw runtime_error("Images must have the same size and type for MSE calculation.");
    }

    Mat diff;
    absdiff(original, reconstructed, diff); // Compute absolute difference
    diff.convertTo(diff, CV_32F);           // Convert to float for precision
    diff = diff.mul(diff);                  // Square the difference

    Scalar sumOfSquares = sum(diff); // Sum across all channels
    double mse = (sumOfSquares[0] + sumOfSquares[1] + sumOfSquares[2]) / (original.total() * original.channels());

    return mse;
}

// Function to calculate Peak Signal-to-Noise Ratio (PSNR)
double calculatePSNR(const Mat &original, const Mat &reconstructed)
{
    double mse = calculateMSE(original, reconstructed);
    if (mse == 0)
    {
        return INFINITY; // No error means infinite PSNR
    }
    double psnr = 10.0 * log10((255 * 255) / mse);
    return psnr;
}

void drawBarChart(const vector<double> &executionTimes, const vector<string> &labels, const string &title)
{
    // Parameters
    int width = 800;
    int height = 600;
    int margin = 50;
    int barWidth = 100;
    int barSpacing = 100;
    int yAxisInterval = 10;

    // Create a blank image with white background
    Mat chart(height, width, CV_8UC3, Scalar(255, 255, 255));

    // Find the maximum execution time for normalization
    double maxTime = *max_element(executionTimes.begin(), executionTimes.end());
    if (maxTime > 500)
        yAxisInterval = 100;
    else
        yAxisInterval = 10;
    double roundedMaxTime = ceil(maxTime / yAxisInterval) * yAxisInterval;

    // Find the index of the best performance (minimum time)
    auto minElementIter = min_element(executionTimes.begin(), executionTimes.end());
    size_t bestIndex = distance(executionTimes.begin(), minElementIter);

    // Draw bars
    for (size_t i = 0; i < executionTimes.size(); ++i)
    {
        int barHeight = static_cast<int>((executionTimes[i] / roundedMaxTime) * (height - 2 * margin));
        int x = margin + i * (barWidth + barSpacing);
        int y = height - margin - barHeight;
        Rect barRect(x, y, barWidth, barHeight);
        rectangle(chart, barRect, Scalar(0, 0, 255), -1); // Red bars
    }

    // Draw X and Y axes
    line(chart, Point(margin, height - margin), Point(width - margin, height - margin), Scalar(0, 0, 0), 2);
    line(chart, Point(margin, margin), Point(margin, height - margin), Scalar(0, 0, 0), 2);

    putText(chart, "Execution Time (ms)", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

    double adjustedTime;
    if ((int)maxTime % yAxisInterval != 0)
        adjustedTime = maxTime + yAxisInterval;
    else
        adjustedTime = maxTime;

    // Draw labels y
    int yAxisLabelsCount = static_cast<int>((adjustedTime) / yAxisInterval) + 1;
    for (int i = 0; i < yAxisLabelsCount; ++i)
    {
        int yLabel = height - margin - (i * (height - 2 * margin) / (yAxisLabelsCount - 1));
        int yValue = yAxisInterval * i;
        putText(chart, to_string(yValue) + "ms", Point(margin - 40, yLabel + 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // Draw labels x
    for (size_t i = 0; i < labels.size(); ++i)
    {
        int x = margin + i * (barWidth + barSpacing) + barWidth / 4;
        int y = height - margin + 20; // Place label below the X axis
        putText(chart, labels[i], Point(x - 10, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // Draw the title
    int titleX = (width - title.size() * 15) / 2; // Center the title horizontally
    int titleY = margin / 2;                      // Position the title above the chart
    putText(chart, title, Point(titleX, titleY), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);

    // Add text for the best performance with execution time in two lines
    int bestX = margin + bestIndex * (barWidth + barSpacing) + barWidth / 4;
    int bestY = height - margin - (executionTimes[bestIndex] / roundedMaxTime) * (height - 2 * margin);
    string bestPerformanceText1 = "Best Performance:";
    string bestPerformanceText2 = to_string((int)executionTimes[bestIndex]) + " ms";
    putText(chart, bestPerformanceText1, Point(bestX - 70, bestY - 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    putText(chart, bestPerformanceText2, Point(bestX, bestY - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

    // Display the image
    imshow("Bar Chart", chart);
    waitKey(0);
}
