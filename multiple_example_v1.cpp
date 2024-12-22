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

// DPCM encoding for DC components
vector<int> dpcm_encode_dc(const vector<int> &dc_coefficients)
{
    vector<int> dpcm_encoded_dc;
    dpcm_encoded_dc.push_back(dc_coefficients[0]);
    for (size_t i = 1; i < dc_coefficients.size(); ++i)
    {
        int diff = dc_coefficients[i] - dc_coefficients[i - 1];
        dpcm_encoded_dc.push_back(diff);
    }
    return dpcm_encoded_dc;
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

// Function to build Huffman tree
HuffmanNode *build_huffman_tree(const unordered_map<int, int> &frequency_dict)
{
    priority_queue<HuffmanNode *, vector<HuffmanNode *>, Compare> pq;

    for (const auto &pair : frequency_dict)
    {
        cout << "Pair: ";
        cout << pair.first << " " << pair.second << endl;
        pq.push(new HuffmanNode(pair.first, pair.second));
    }

    while (pq.size() > 1)
    {
        HuffmanNode *left = pq.top();
        pq.pop();
        HuffmanNode *right = pq.top();
        pq.pop();

        HuffmanNode *merged = new HuffmanNode(-1, left->frequency + right->frequency);
        merged->left = left;
        merged->right = right;
        pq.push(merged);
    }

    return pq.top();
}

// Function to generate Huffman codes
void generate_huffman_codes(HuffmanNode *node, const string &code, unordered_map<int, string> &huffman_codes)
{
    if (!node)
        return;

    huffman_codes[node->value] = code;

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
        {{65, 70, 72, 74, 76, 73, 70, 65},
         {68, 65, 60, 58, 56, 55, 60, 70},
         {72, 75, 80, 85, 87, 90, 88, 80},
         {60, 58, 55, 52, 50, 51, 55, 60},
         {70, 75, 80, 85, 90, 95, 100, 105},
         {110, 115, 120, 125, 130, 128, 125, 120},
         {100, 95, 90, 85, 80, 75, 70, 65},
         {60, 55, 50, 45, 40, 38, 35, 30}}};

    // Vector to store results
    vector<int> all_dc_coefficients;              // To collect all DC coefficients
    vector<vector<int>> encoded_values_per_block; // Store encoded values (DC + AC)

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

        // Extract DC and AC coefficients
        int dc_coefficient = zigzag[0];                                // First value is DC
        vector<int> ac_coefficients(zigzag.begin() + 1, zigzag.end()); // Rest are AC

        // Collect DC coefficient for DPCM encoding
        all_dc_coefficients.push_back(dc_coefficient);

        // Perform RLE encoding on AC coefficients
        vector<int> rle_encoded_ac = run_length_encode_ac(ac_coefficients);

        // Store RLE encoded AC values for this block
        encoded_values_per_block.push_back(rle_encoded_ac); // Collect the first coefficient (DC)
    }

    // Perform DPCM Encoding on DC coefficients
    vector<int> dpcm_encoded_dc = dpcm_encode_dc(all_dc_coefficients);

    // Combine DC and AC encoded values for each block
    vector<int> combined_encoded_values; // Final combined encoded values
    for (size_t i = 0; i < encoded_values_per_block.size(); ++i)
    {
        // Prepend DPCM-encoded DC to RLE-encoded AC
        vector<int> block_encoded_values = {dpcm_encoded_dc[i]};
        block_encoded_values.insert(block_encoded_values.end(), encoded_values_per_block[i].begin(), encoded_values_per_block[i].end());

        // Append to the final combined sequence
        combined_encoded_values.insert(combined_encoded_values.end(), block_encoded_values.begin(), block_encoded_values.end());
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

    return 0;
}
