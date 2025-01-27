#pragma once
#include <vector>

using namespace std;

void dctQuantizationParallel(const vector<vector<vector<int>>> &input_blocks, const vector<vector<int>> &quantization_table,
                             vector<vector<float>> &quantize_coefficients);

void dequantizeInverseDCTParallel(const vector<vector<vector<float>>> &quantize_coefficients, const vector<vector<int>> &quantization_table,
                                  vector<vector<vector<int>>> &output_blocks);
