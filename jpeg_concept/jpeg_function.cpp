#include <iostream>
#include <vector>
#include <cmath> // For cosine and sqrt functions
#include <unordered_map>
#include <queue>
#include <string>
#include <algorithm>

using namespace std;

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
            dct_block[u][v] = round(0.25 * cu * cv * sum_val);
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
vector<int> run_length_encode(const vector<int> &ac_coefficients)
{
    vector<int> rle_encoded;
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
            rle_encoded.push_back(prev_chr);
            rle_encoded.push_back(run_length);
            prev_chr = ac_coefficients[i];
            run_length = 1;
        }
    }
    // Add the last entry
    rle_encoded.push_back(prev_chr);
    rle_encoded.push_back(run_length);
    return rle_encoded;
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
    // Convert unordered_map to vector of pairs
    vector<pair<int, int>> freq_vector(freq_dict.begin(), freq_dict.end());

    // Sort by frequency, breaking ties by key (ascending order)
    sort(freq_vector.begin(), freq_vector.end(), [](const pair<int, int> &a, const pair<int, int> &b)
         {
             if (a.second == b.second)
                 return a.first < b.first; // Break ties by key
             return a.second < b.second;   // Sort by frequency
         });

    // Create a priority queue (min-heap) and populate it with sorted elements
    priority_queue<HuffmanNode *, vector<HuffmanNode *>, Compare> min_heap;
    for (const auto &pair : freq_vector)
    {
        min_heap.push(new HuffmanNode(pair.first, pair.second));
    }

    // Build the Huffman tree
    while (min_heap.size() > 1)
    {
        HuffmanNode *left = min_heap.top();
        min_heap.pop();
        HuffmanNode *right = min_heap.top();
        min_heap.pop();

        // Create an internal node with combined frequency
        HuffmanNode *internal = new HuffmanNode(99999, left->frequency + right->frequency);
        internal->left = left;
        internal->right = right;

        // Push the internal node back into the priority queue
        min_heap.push(internal);
    }

    // Return the root of the Huffman tree
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

vector<vector<int>> quantization_table = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}};

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

int main()
{
    vector<vector<vector<int>>> image_blocks = {
        {{52, 55, 61, 66, 70, 61, 64, 73},
         {63, 59, 55, 90, 109, 85, 69, 72},
         {62, 59, 68, 113, 144, 104, 66, 73},
         {63, 58, 71, 122, 154, 106, 70, 69},
         {67, 61, 68, 104, 126, 88, 68, 70},
         {79, 65, 60, 70, 77, 68, 58, 75},
         {85, 71, 64, 59, 55, 61, 65, 83},
         {87, 79, 69, 68, 65, 76, 78, 94}},
    };

    // Vector to store results
    vector<vector<int>> encoded_values_per_block; // Store encoded values (AC)

    for (auto &image_block : image_blocks)
    {
        cout << "Processing a new block." << endl;

        cout << "Image: " << endl;
        print_with_tab(image_block);

        // Recenter
        recenterAroundZero(image_block);

        cout << "Image after recentering: " << endl;
        print_with_tab(image_block);

        // Perform DCT
        vector<vector<float>> dct_block(8, vector<float>(8, 0.0));
        dct(image_block, dct_block);

        cout << "DCT block: " << endl;
        print_with_tab(dct_block);

        // Quantization
        quantize(dct_block, quantization_table);

        cout << "Quantized DCT block: " << endl;
        print_with_tab(dct_block);

        // Zigzag Scan
        vector<int> zigzag;
        zigzag_scan(dct_block, zigzag);

        // Print the zigzag scanned vector
        cout << "Zigzag scanned vector: " << endl;
        for (const auto &val : zigzag)
        {
            cout << val << " ";
        }
        cout << endl;

        // Perform RLE encoding on AC coefficients
        vector<int> rle_encoded = run_length_encode(zigzag);

        // Store RLE encoded AC values for this block
        encoded_values_per_block.push_back(rle_encoded); // Collect the first coefficient (DC)
    }

    // Combine all encoded values into a single vector
    vector<int> combined_encoded_values;
    for (const auto &block : encoded_values_per_block)
    {
        for (int val : block)
        {
            combined_encoded_values.push_back(val);
        }
    }

    // Print the final combined encoded values
    cout << "Combined Encoded Values: " << endl;
    for (int val : combined_encoded_values)
    {
        cout << val << " ";
    }
    cout << endl;

    // Build frequency dictionary and Huffman tree
    unordered_map<int, int> freq_dict = build_frequency_dict(combined_encoded_values);
    HuffmanNode *huffman_tree = build_huffman_tree(freq_dict);

    // Generate Huffman codes
    unordered_map<int, string> huffman_codes;
    generate_huffman_codes(huffman_tree, "", huffman_codes);

    // Print the Huffman tree structure
    cout << "Huffman Tree: " << endl;
    // print_huffman_tree(huffman_tree);

    // Encode all data using Huffman codes
    string huffman_encoded_str = huffman_encode(combined_encoded_values, huffman_codes);

    cout << "Huffman Encoded String: " << endl;
    cout << huffman_encoded_str << endl;

    vector<int> decoded_values = huffman_decode(huffman_encoded_str, huffman_tree);
    cout << "Decoded Values: ";
    for (int val : decoded_values)
    {
        cout << val << " ";
    }
    cout << endl;

    vector<vector<int>> decoded_blocks = decodeRLE(decoded_values);

    // Print the decoded blocks
    for (size_t i = 0; i < decoded_blocks.size(); ++i)
    {
        cout << "Block " << i + 1 << ":\n";
        for (size_t j = 0; j < decoded_blocks[i].size(); ++j)
        {
            cout << decoded_blocks[i][j] << " ";
        }
        cout << endl;
    }

    vector<vector<float>> restored_block(8, vector<float>(8, 0.0));

    // Assuming `decoded_blocks[i]` is the Zigzag 1D array for block `i`
    for (size_t i = 0; i < decoded_blocks.size(); ++i)
    {
        inverse_zigzag(decoded_blocks[i], restored_block);

        cout << "Restored Block " << i + 1 << ":" << endl;
        print_with_tab(restored_block);

        // Step 2: Inverse Quantization
        inverse_quantize(restored_block, quantization_table);

        cout << "Restored Block after Inverse Quantization " << i + 1 << ":" << endl;
        print_with_tab(restored_block);

        // Step 3: Inverse DCT
        vector<vector<int>> reconstructed_image_block(8, vector<int>(8, 0));
        idct(restored_block, reconstructed_image_block);

        cout << "Reconstructed Image Block " << i + 1 << ":" << endl;
        print_with_tab(reconstructed_image_block);

        // Step 4: Add 128 back to restore the original pixel values
        add_back_128(reconstructed_image_block);

        cout << "Restored Image Block " << i + 1 << " with 128 added back:" << endl;
        print_with_tab(reconstructed_image_block);
    }

    return 0;
}
