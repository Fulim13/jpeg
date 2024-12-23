#pragma once
#include <vector>

using namespace std;

void encodeGPU(const vector<vector<vector<int>>> &input_blocks, const vector<vector<int>> &quantization_table,
               vector<vector<int>> &encoded_blocks);

void decodeGPU(const vector<vector<int>> &encoded_blocks, const vector<vector<int>> &quantization_table,
               vector<vector<vector<int>>> &decoded_blocks);
