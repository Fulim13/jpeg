#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath> // For cosine and sqrt functions
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

Mat readImage(const string &folder_path, const string &image_name)
{
    Mat image = imread(folder_path + image_name + ".png"); // Replace with your image path
    if (image.empty())
    {
        cerr << "Could not open or find the image" << endl;
        return Mat();
    }
    return image;
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

void saveProcessedImages(const Mat &Y, const Mat &Cb, const Mat &Cr)
{
    imwrite("output/Y_channel.png", Y);
    imwrite("output/Cb_channel.png", Cb);
    imwrite("output/Cr_channel.png", Cr);
}

void showImages(const Mat &image, const Mat &Y, const Mat &Cb, const Mat &Cr)
{
    imshow("Original Image", image);
    imshow("Y Channel", Y);
    imshow("Cb Channel (Subsampled)", Cb);
    imshow("Cr Channel (Subsampled)", Cr);
    waitKey(0);
}

// Calculate compression ratio and sizes
struct CompressionStats
{
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
};

CompressionStats calculateCompressionStats(const Mat &original_image,
                                           const string &y_encoded,
                                           const string &cb_encoded,
                                           const string &cr_encoded)
{
    CompressionStats stats;

    // Calculate original size (3 channels * width * height)
    stats.original_size = original_image.total() * original_image.elemSize();

    // Calculate compressed size (encoded strings in bytes + small overhead for dimensions)
    stats.compressed_size = (y_encoded.length() + cb_encoded.length() + cr_encoded.length()) / 8;
    stats.compressed_size += sizeof(int) * 2; // For storing width and height

    // Calculate compression ratio
    stats.compression_ratio = static_cast<double>(stats.original_size) / stats.compressed_size;

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

// const: This means that the function will not modify the mat parameter. It's a promise that the function only reads the data.

// & (reference): This allows the function to avoid copying the entire matrix. Instead, it works directly with the original matrix passed as an argument, making the function more efficient.
void print_with_tab(const vector<vector<int>> &mat)
{
    // auto lets the compiler deduce the type automatically.
    // & means we're accessing each row of the matrix by reference (not making a copy).
    for (const auto &row : mat) // Iterate over each row of the matrix
    {
        for (const auto &val : row) // Iterate over each value in the current row
        {
            cout << val << "\t";
        }
        cout << endl;
    }
}

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

// Function to perform Discrete Cosine Transform (DCT) on an 8x8 image block
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
        cout << endl;
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

// RLE for AC components
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

// Function to build frequency dictionary
unordered_map<int, int> build_frequency_dict(const vector<int> &data)
{
    unordered_map<int, int> frequency_dict;
    for (int val : data)
    {
        frequency_dict[val]++;
    }
    return frequency_dict;
}

// Node structure for Huffman tree
struct HuffmanNode
{
    int value;
    int frequency;
    HuffmanNode *left;
    HuffmanNode *right;

    HuffmanNode(int val, int freq) : value(val), frequency(freq), left(nullptr), right(nullptr) {}
};

// Comparator for priority queue
struct Compare
{
    bool operator()(HuffmanNode *left, HuffmanNode *right)
    {
        return left->frequency > right->frequency;
    }
};

// Function to build the Huffman tree
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

// Function to generate Huffman codes
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

// Function to encode data using Huffman codes
string huffman_encode(const vector<int> &data, const unordered_map<int, string> &huffman_codes)
{
    string encoded_str;
    // Print the values with their corresponding Huffman codes
    cout << "Huffman Codes: " << endl;
    for (const auto &pair : huffman_codes)
    {
        cout << pair.first << ": " << pair.second << endl;
    }

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

// Example quantization tables for Y, Cb, and Cr
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

// Function to print the Huffman tree structure
void print_huffman_tree(HuffmanNode *node, const string &prefix = "", bool is_left = true)
{
    if (!node)
        return;

    // Print the current node
    cout << prefix << (is_left ? "├── " : "└── ");
    if (!node->left && !node->right)
    {
        // Leaf node
        cout << "Value: " << node->value << ", Frequency: " << node->frequency << endl;
    }
    else
    {
        // Internal node
        cout << "Frequency: " << node->frequency << endl;
    }

    // Recursively print left and right subtrees
    print_huffman_tree(node->left, prefix + (is_left ? "│   " : "    "), true);
    print_huffman_tree(node->right, prefix + (is_left ? "│   " : "    "), false);
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
            val += 128; // Revert the recentering step by adding 128
        }
    }
}

// Encoding function that takes a channel matrix and returns encoded data
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

// Decoding function that takes encoded data and returns reconstructed channel
Mat decodeChannel(const string &encoded_data, int height, int width, const vector<vector<int>> &quantization_table, HuffmanNode *huffman_tree)
{
    // Huffman decode
    vector<int> decoded_values = huffman_decode(encoded_data, huffman_tree);

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

// Convert a string of '0' and '1' characters (Huffman encoded data) to raw bitstream
vector<unsigned char> stringToBitstream(const string &huffman_data)
{
    vector<unsigned char> bitstream;
    unsigned char current_byte = 0;
    int bit_pos = 0;

    for (char bit : huffman_data)
    {
        if (bit == '0')
        {
            current_byte = current_byte & ~(1 << (7 - bit_pos)); // Set bit to 0
        }
        else if (bit == '1')
        {
            current_byte = current_byte | (1 << (7 - bit_pos)); // Set bit to 1
        }
        else
        {
            throw invalid_argument("Huffman data contains invalid characters. Only '0' and '1' are allowed.");
        }

        bit_pos++;
        if (bit_pos == 8)
        {
            // Once 8 bits are filled, add the byte to the bitstream
            bitstream.push_back(current_byte);
            current_byte = 0;
            bit_pos = 0;
        }
    }

    // If there are any leftover bits, add them as a byte
    if (bit_pos > 0)
    {
        bitstream.push_back(current_byte);
    }

    return bitstream;
}

// Function to save the encoded data to a file (now using bitstream)
void saveEncodedData(const string &filename,
                     const string &y_data, const string &cb_data, const string &cr_data,
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

    // Convert Huffman strings to raw bitstreams
    vector<unsigned char> y_bitstream = stringToBitstream(y_data);
    vector<unsigned char> cb_bitstream = stringToBitstream(cb_data);
    vector<unsigned char> cr_bitstream = stringToBitstream(cr_data);

    // Write the lengths of the bitstreams
    int y_len = y_bitstream.size();
    int cb_len = cb_bitstream.size();
    int cr_len = cr_bitstream.size();

    file.write(reinterpret_cast<const char *>(&y_len), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cb_len), sizeof(int));
    file.write(reinterpret_cast<const char *>(&cr_len), sizeof(int));

    // Write the raw bitstreams
    file.write(reinterpret_cast<const char *>(y_bitstream.data()), y_len);
    file.write(reinterpret_cast<const char *>(cb_bitstream.data()), cb_len);
    file.write(reinterpret_cast<const char *>(cr_bitstream.data()), cr_len);

    file.close();
}

// Function to convert bitstream (byte data) back to Huffman-encoded string
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

// Function to load encoded data from file
void loadEncodedData(const string &filename,
                     string &y_data, string &cb_data, string &cr_data,
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

    // Read the lengths of the encoded strings
    int y_len, cb_len, cr_len;
    file.read(reinterpret_cast<char *>(&y_len), sizeof(int));
    file.read(reinterpret_cast<char *>(&cb_len), sizeof(int));
    file.read(reinterpret_cast<char *>(&cr_len), sizeof(int));

    // Read the raw bitstream data
    vector<unsigned char> y_bitstream(y_len), cb_bitstream(cb_len), cr_bitstream(cr_len);
    file.read(reinterpret_cast<char *>(y_bitstream.data()), y_len);
    file.read(reinterpret_cast<char *>(cb_bitstream.data()), cb_len);
    file.read(reinterpret_cast<char *>(cr_bitstream.data()), cr_len);

    file.close();

    // Reconstruct the Huffman-encoded strings from bitstream
    y_data = bitstreamToString(y_bitstream, y_len * 8); // 8 bits per byte
    cb_data = bitstreamToString(cb_bitstream, cb_len * 8);
    cr_data = bitstreamToString(cr_bitstream, cr_len * 8);
}

int main()
{
    string folder_path = "img/";
    string image_name = "test2";
    // Load and convert image
    Mat image = readImage(folder_path, image_name);
    if (image.empty())
    {
        throw runtime_error("Failed to load image");
    }

    Mat ycbcr_image = RGB2YCbCr(image);

    // Perform chroma subsampling
    Mat Y, Cb, Cr;
    chromaSubsampling(ycbcr_image, Y, Cb, Cr);

    // Show the original and subsampled images
    showImages(image, Y, Cb, Cr);

    // Encode each channel
    EncodedData y_encoded = encodeChannel(Y, quantization_table_Y);
    EncodedData cb_encoded = encodeChannel(Cb, quantization_table_CbCr);
    EncodedData cr_encoded = encodeChannel(Cr, quantization_table_CbCr);

    string y_huffman_str = y_encoded.huffman_encoded_str;
    HuffmanNode *y_huffman_tree = y_encoded.huffman_tree;
    string cb_huffman_str = cb_encoded.huffman_encoded_str;
    HuffmanNode *cb_huffman_tree = cb_encoded.huffman_tree;
    string cr_huffman_str = cr_encoded.huffman_encoded_str;
    HuffmanNode *cr_huffman_tree = cr_encoded.huffman_tree;

    // Save three encoded data (EncodedData) and ows and cols for each channel to one bin file
    saveEncodedData("compressed_images.bin",
                    y_huffman_str, cb_huffman_str, cr_huffman_str,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols);

    // Calculate compression statistics
    CompressionStats stats = calculateCompressionStats(
        image,
        y_encoded.huffman_encoded_str,
        cb_encoded.huffman_encoded_str,
        cr_encoded.huffman_encoded_str);

    // Print compression information
    cout << "Original size: " << stats.original_size << " bytes" << endl;
    cout << "Compressed size: " << stats.compressed_size << " bytes" << endl;
    cout << "Compression ratio: " << stats.compression_ratio << ":1" << endl;
    cout << "Space saving: " << (1.0 - 1.0 / stats.compression_ratio) * 100 << "%" << endl;

    // Load encoded data (As EncodedData) from file and Read the rows and cols for each channel
    string y_loaded, cb_loaded, cr_loaded;
    int y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols;
    loadEncodedData("compressed_images.bin",
                    y_loaded, cb_loaded, cr_loaded,
                    y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols);

    // Decode each channel
    Mat Y_reconstructed = decodeChannel(y_loaded, y_rows, y_cols, quantization_table_Y, y_huffman_tree);
    Mat Cb_reconstructed = decodeChannel(cb_loaded, cb_rows, cr_rows, quantization_table_CbCr, cb_huffman_tree);
    Mat Cr_reconstructed = decodeChannel(cr_loaded, cr_rows, cr_cols, quantization_table_CbCr, cr_huffman_tree);

    // Show the reconstructed images
    imshow("Reconstructed Y Channel", Y_reconstructed);
    imshow("Reconstructed Cb Channel", Cb_reconstructed);
    imshow("Reconstructed Cr Channel", Cr_reconstructed);
    waitKey(0);

    // Merge the Y, Cb, Cr channels and convert back to BGR
    // Resize Cb and Cr channels to match the size of Y channel
    resize(Cb_reconstructed, Cb_reconstructed, Y_reconstructed.size(), 0, 0, INTER_LINEAR);
    resize(Cr_reconstructed, Cr_reconstructed, Y_reconstructed.size(), 0, 0, INTER_LINEAR);

    Mat reconstructed_image;
    vector<Mat> channels = {Y_reconstructed, Cb_reconstructed, Cr_reconstructed};

    // Check sizes and depths of the channels
    for (int i = 0; i < channels.size(); i++)
    {
        cout << "Channel " << i << " size: " << channels[i].size() << ", depth: " << channels[i].depth() << endl;
    }

    // Ensure all channels have the same size and depth
    if (channels[0].size() == channels[1].size() && channels[0].size() == channels[2].size() &&
        channels[0].depth() == channels[1].depth() && channels[0].depth() == channels[2].depth())
    {
        merge(channels, reconstructed_image);
        Mat final_image;
        cvtColor(reconstructed_image, final_image, COLOR_YCrCb2BGR);

        // Display the final image
        imshow("Final Image", final_image);
        waitKey(0);
    }
    else
    {
        cerr << "Error: Channel sizes or depths do not match." << endl;
    }

    // Clean up Huffman trees
    delete y_encoded.huffman_tree;
    delete cb_encoded.huffman_tree;
    delete cr_encoded.huffman_tree;

    return 0;
}
