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
    for (int val : data)
    {
        encoded_str += huffman_codes.at(val);
    }
    return encoded_str;
}

int main()
{
    vector<vector<int>> image_block = {
        {52, 55, 61, 66, 70, 61, 64, 73},
        {63, 59, 55, 90, 109, 85, 69, 72},
        {62, 59, 68, 113, 144, 104, 66, 73},
        {63, 58, 71, 122, 154, 106, 70, 69},
        {67, 61, 68, 104, 126, 88, 68, 70},
        {79, 65, 60, 70, 77, 68, 58, 75},
        {85, 71, 64, 59, 55, 61, 65, 83},
        {87, 79, 69, 68, 65, 76, 78, 94}};

    cout << "Image: " << endl;
    print_with_tab(image_block);

    // Value Shifting to make the image block values centered around 0
    recenterAroundZero(image_block);

    cout << "Image after recentering: " << endl;
    print_with_tab(image_block);

    //(8, vector<float>(8, 0.0))
    // The outer vector will contain 8 elements because we want an 8x8 matrix for the DCT block.
    // The inner vector, vector<float>(8, 0.0), creates a vector of float values with 8 elements, all initialized to 0.0. This represents a row of the matrix, and each element in that row is of type float.
    // So, the inner vector<float>(8, 0.0) creates a row with 8 columns, all initialized to 0.0.

    vector<vector<float>> dct_block(8, vector<float>(8, 0.0));

    // Perform the DCT transformation
    dct(image_block, dct_block);

    // Print the resulting DCT block
    cout << "DCT block: " << endl;
    print_with_tab(dct_block);

    // Quantization table for luminance for quality factor 50
    vector<vector<int>> quantization_table = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}};

    // Perform quantization
    quantize(dct_block, quantization_table);

    // Print the quantized DCT block
    cout << "Quantized DCT block: " << endl;
    print_with_tab(dct_block);

    // Zigzag scan
    vector<int> zigzag;
    zigzag_scan(dct_block, zigzag);

    // Print the zigzag scanned vector
    cout << "Zigzag scanned vector: " << endl;
    for (const auto &val : zigzag)
    {
        cout << val << " ";
    }
    cout << endl;

    // DPCM encode the DC component
    vector<int> dpcm_encoded_dc = dpcm_encode_dc({zigzag[0]});

    // RLE encode the AC components
    vector<int> rle_encoded_ac = run_length_encode_ac(vector<int>(zigzag.begin() + 1, zigzag.end()));

    // Print the encoded values
    cout << "DPCM Encoded values: " << endl;
    for (int val : dpcm_encoded_dc)
    {
        cout << val << " ";
    }
    cout << endl;

    cout << "RLE Encoded values: " << endl;
    for (int val : rle_encoded_ac)
    {
        cout << val << " ";
    }
    cout << endl;

    // Combine the encoded values
    vector<int> encoded_val = dpcm_encoded_dc;
    encoded_val.insert(encoded_val.end(), rle_encoded_ac.begin(), rle_encoded_ac.end());

    cout << "Encoded values: " << endl;
    for (int val : encoded_val)
    {
        cout << val << " ";
    }
    cout << endl;

    // Build frequency dictionary
    unordered_map<int, int> freq_dict = build_frequency_dict(encoded_val);
    cout << "Frequency dictionary: " << endl;
    for (const auto &pair : freq_dict)
    {
        cout << pair.first << ": " << pair.second << endl;
    }

    // Build Huffman tree
    HuffmanNode *huffman_tree = build_huffman_tree(freq_dict);
    cout << "Huffman Tree built." << endl;

    // Generate Huffman codes
    unordered_map<int, string> huffman_codes;
    generate_huffman_codes(huffman_tree, "", huffman_codes);
    cout << "Huffman Codes: " << endl;
    for (const auto &pair : huffman_codes)
    {
        cout << pair.first << ": " << pair.second << endl;
    }

    // Huffman encode the data
    string huffman_encoded_str = huffman_encode(encoded_val, huffman_codes);
    cout << "Huffman Encoded String: " << endl;
    cout << huffman_encoded_str << endl;

    return 0;
}
