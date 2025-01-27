#include "encode_decode.cpp"
vector<vector<int>> baseline_quantization_table_luminance = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}};

vector<vector<int>> baseline_quantization_table_chrominance = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}};

int main(int argc, char *argv[])
{
    // Default quality percentage
    int quality = 50;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "-q" && i + 1 < argc)
        {
            try
            {
                quality = stoi(argv[++i]);
                if (quality < 1 || quality > 100)
                {
                    throw out_of_range("Quality must be between 1 and 100");
                }
            }
            catch (exception &e)
            {
                cerr << "Invalid quality argument: " << e.what() << endl;
                return 1;
            }
        }
    }

    cout << "Using quality: " << quality << "%" << endl;

    int scaling_factor = 0;
    if (quality < 50)
    {
        // Increase the quality for lower quality values
        scaling_factor = 5000 / quality;
    }
    else
    {
        // Decrease the quality for higher quality values
        scaling_factor = 200 - quality * 2;
    }

    // Scale quantization tables based on quality
    vector<vector<int>> quantization_table_Y = baseline_quantization_table_luminance;
    for (auto &row : quantization_table_Y)
    {
        for (int &val : row)
        {
            val = max(1, min(255, (val * scaling_factor + 50) / 100));
        }
    }

    vector<vector<int>> quantization_table_CbCr = baseline_quantization_table_chrominance;
    for (auto &row : quantization_table_CbCr)
    {
        for (int &val : row)
        {
            val = max(1, min(255, (val * scaling_factor + 50) / 100));
        }
    }

    string image_name = "";

    cout << "Enter the image name: ";
    cin >> image_name;

    // Load the image
    Mat image = readImage(image_name);

    ensureMultipleOf16(image);

    // Print the Image Dimensions
    cout << "Image Dimensions: " << image.rows << " x " << image.cols << endl;
    cout << "====================================================\n";

    // Convert the image From RGB to YCbCr
    Mat ycbcr_image;
    // cvtColor(image, ycbcr_image, COLOR_BGR2YCrCb);
    ycbcr_image = RGB2YCbCr(image);

    // Perform chroma subsampling
    Mat Y, Cb, Cr;
    chromaSubsampling(ycbcr_image, Y, Cb, Cr);
    ycbcr_image.release();

    // Encode and Calculate the time for encoding
    EncodedData y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu;
    EncodedData y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu;
    EncodedData y_encoded_omp, cb_encoded_omp, cr_encoded_omp;
    cout << "Compressed with CPU" << endl;
    cout << "======================================\n";
    double originalTimeForEncode = mainEncode(Y, Cb, Cr,
                                              quantization_table_Y, quantization_table_CbCr,
                                              y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu, "CPU");

    string y_huffman_str_cpu = y_encoded_cpu.huffman_encoded_str;
    const unordered_map<int, int> y_freq_dict = y_encoded_cpu.freq_dict;
    string cb_huffman_str_cpu = cb_encoded_cpu.huffman_encoded_str;
    const unordered_map<int, int> cb_freq_dict = cb_encoded_cpu.freq_dict;
    string cr_huffman_str_cpu = cr_encoded_cpu.huffman_encoded_str;
    const unordered_map<int, int> cr_freq_dict = cr_encoded_cpu.freq_dict;

    cout << "Compressed with GPU" << endl;
    cout << "======================================\n";
    double modifiedTimeGPUForEncode = mainEncode(Y, Cb, Cr,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu, "GPU");

    string y_huffman_str_gpu = y_encoded_gpu.huffman_encoded_str;
    const unordered_map<int, int> y_freq_dict_gpu = y_encoded_gpu.freq_dict;
    string cb_huffman_str_gpu = cb_encoded_gpu.huffman_encoded_str;
    const unordered_map<int, int> cb_freq_dict_gpu = cb_encoded_gpu.freq_dict;
    string cr_huffman_str_gpu = cr_encoded_gpu.huffman_encoded_str;
    const unordered_map<int, int> cr_freq_dict_gpu = cr_encoded_gpu.freq_dict;

    cout << "Compressed with OMP" << endl;
    cout << "======================================\n";
    double modifiedTimeOMPForEncode = mainEncode(Y, Cb, Cr,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 y_encoded_omp, cb_encoded_omp, cr_encoded_omp, "OMP");

    string y_huffman_str_omp = y_encoded_omp.huffman_encoded_str;
    const unordered_map<int, int> y_freq_dict_omp = y_encoded_omp.freq_dict;
    string cb_huffman_str_omp = cb_encoded_omp.huffman_encoded_str;
    const unordered_map<int, int> cb_freq_dict_omp = cb_encoded_omp.freq_dict;
    string cr_huffman_str_omp = cr_encoded_omp.huffman_encoded_str;
    const unordered_map<int, int> cr_freq_dict_omp = cr_encoded_omp.freq_dict;

    cout << "Encoding time (CPU): " << originalTimeForEncode << " ms" << endl;
    cout << "Encoding time (GPU): " << modifiedTimeGPUForEncode << " ms" << endl;
    cout << "Encoding time (OMP): " << modifiedTimeOMPForEncode << " ms" << endl;
    cout << "Encoding Performance Improvement (CPU/GPU): " << originalTimeForEncode / modifiedTimeGPUForEncode << "x" << endl;
    cout << "Encoding Performance Improvement (CPU/OMP): " << originalTimeForEncode / modifiedTimeOMPForEncode << "x" << endl;
    cout << "======================================\n\n";

    // CPU - Save three encoded data (EncodedData) and rows and cols for each channel to one bin file
    string compressed_filename_cpu = "output/compressed_image_cpu_" + to_string(quality) + ".bin";
    saveEncodedData(compressed_filename_cpu,
                    y_huffman_str_cpu, cb_huffman_str_cpu, cr_huffman_str_cpu,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols,
                    y_freq_dict, cb_freq_dict, cr_freq_dict);

    cout << "Compressed file (CPU) saved in: " << compressed_filename_cpu << endl;

    // GPU - Save three encoded data (EncodedData) and rows and cols for each channel to one bin file
    string compressed_filename_gpu = "output/compressed_image_gpu_" + to_string(quality) + ".bin";
    saveEncodedData(compressed_filename_gpu,
                    y_huffman_str_gpu, cb_huffman_str_gpu, cr_huffman_str_gpu,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols,
                    y_freq_dict_gpu, cb_freq_dict_gpu, cr_freq_dict_gpu);
    cout << "Compressed file (GPU) saved in: " << compressed_filename_cpu << endl;

    // OMP - Save three encoded data (EncodedData) and rows and cols for each channel to one bin file
    string compressed_filename_omp = "output/compressed_image_omp" + to_string(quality) + ".bin";
    saveEncodedData(compressed_filename_omp,
                    y_huffman_str_omp, cb_huffman_str_omp, cr_huffman_str_omp,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols,
                    y_freq_dict_omp, cb_freq_dict_omp, cr_freq_dict_omp);
    cout << "Compressed file (OMP) saved in: " << compressed_filename_omp << endl;

    // Stat For CPU
    CompressionStats statsCPU = calculateCompressionStats(image_name, compressed_filename_cpu);
    cout << "Original size (CPU): " << statsCPU.original_size << " bytes (" << statsCPU.original_size_kb << " KB)" << endl;
    cout << "Compressed size (CPU): " << statsCPU.compressed_size << " bytes (" << statsCPU.compressed_size_kb << " KB)" << endl;
    cout << "Compression ratio (CPU): " << statsCPU.compression_ratio << endl;
    cout << "Percentage reduction (CPU): " << statsCPU.percentage_reduction << "%" << endl;
    cout << "======================================\n\n";

    // Stat For GPU
    CompressionStats statsGPU = calculateCompressionStats(image_name, compressed_filename_gpu);
    cout << "Original size (GPU): " << statsGPU.original_size << " bytes (" << statsGPU.original_size_kb << " KB)" << endl;
    cout << "Compressed size (GPU): " << statsGPU.compressed_size << " bytes (" << statsGPU.compressed_size_kb << " KB)" << endl;
    cout << "Compression ratio (GPU): " << statsGPU.compression_ratio << endl;
    cout << "Percentage reduction (GPU): " << statsGPU.percentage_reduction << "%" << endl;
    cout << "======================================\n\n";

    // Stat For OMP
    CompressionStats statsOMP = calculateCompressionStats(image_name, compressed_filename_omp);
    cout << "Original size (OMP): " << statsOMP.original_size << " bytes (" << statsOMP.original_size_kb << " KB)" << endl;
    cout << "Compressed size (OMP): " << statsOMP.compressed_size << " bytes (" << statsOMP.compressed_size_kb << " KB)" << endl;
    cout << "Compression ratio (OMP): " << statsOMP.compression_ratio << endl;
    cout << "Percentage reduction (OMP): " << statsOMP.percentage_reduction << "%" << endl;
    cout << "======================================\n\n";

    // CPU - Load encoded data (As EncodedData) from file and Read the rows and cols for each channel
    string y_loaded, cb_loaded, cr_loaded;
    int y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols;
    unordered_map<int, int> y_loaded_freq_dict, cb_loaded_freq_dict, cr_loaded_freq_dict;
    loadEncodedData(compressed_filename_cpu,
                    y_loaded, cb_loaded, cr_loaded,
                    y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols,
                    y_loaded_freq_dict, cb_loaded_freq_dict, cr_loaded_freq_dict);

    // GPU - Load encoded data (As EncodedData) from file and Read the rows and cols for each channel
    string y_loaded_gpu, cb_loaded_gpu, cr_loaded_gpu;
    int y_rows_gpu, y_cols_gpu, cb_rows_gpu, cb_cols_gpu, cr_rows_gpu, cr_cols_gpu;
    unordered_map<int, int> y_loaded_freq_dict_gpu, cb_loaded_freq_dict_gpu, cr_loaded_freq_dict_gpu;
    loadEncodedData(compressed_filename_gpu,
                    y_loaded_gpu, cb_loaded_gpu, cr_loaded_gpu,
                    y_rows_gpu, y_cols_gpu, cb_rows_gpu, cb_cols_gpu, cr_rows_gpu, cr_cols_gpu,
                    y_loaded_freq_dict_gpu, cb_loaded_freq_dict_gpu, cr_loaded_freq_dict_gpu);

    // OMP - Load encoded data (As EncodedData) from file and Read the rows and cols for each channel
    string y_loaded_omp, cb_loaded_omp, cr_loaded_omp;
    int y_rows_omp, y_cols_omp, cb_rows_omp, cb_cols_omp, cr_rows_omp, cr_cols_omp;
    unordered_map<int, int> y_loaded_freq_dict_omp, cb_loaded_freq_dict_omp, cr_loaded_freq_dict_omp;
    loadEncodedData(compressed_filename_omp,
                    y_loaded_omp, cb_loaded_omp, cr_loaded_omp,
                    y_rows_omp, y_cols_omp, cb_rows_omp, cb_cols_omp, cr_rows_omp, cr_cols_omp,
                    y_loaded_freq_dict_omp, cb_loaded_freq_dict_omp, cr_loaded_freq_dict_omp);

    // Decode and Calculate the time for decoding
    Mat Y_reconstructed, Cb_reconstructed, Cr_reconstructed;
    cout << "Decompressed with CPU" << endl;
    cout << "======================================\n";
    double originalTimeForDecode = mainDecode(y_loaded, cb_loaded, cr_loaded,
                                              y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols,
                                              quantization_table_Y, quantization_table_CbCr,
                                              Y_reconstructed, Cb_reconstructed, Cr_reconstructed,
                                              y_loaded_freq_dict, cb_loaded_freq_dict, cr_loaded_freq_dict,
                                              "CPU");

    Mat Y_reconstructed_gpu, Cb_reconstructed_gpu, Cr_reconstructed_gpu;
    cout << "Decompressed with GPU" << endl;
    cout << "======================================\n";
    double modifiedTimeGPUForDecode = mainDecode(y_loaded_gpu, cb_loaded_gpu, cr_loaded_gpu,
                                                 y_rows_gpu, y_cols_gpu, cb_rows_gpu, cb_cols_gpu, cr_rows_gpu, cr_cols_gpu,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 Y_reconstructed_gpu, Cb_reconstructed_gpu, Cr_reconstructed_gpu,
                                                 y_loaded_freq_dict_gpu, cb_loaded_freq_dict_gpu, cr_loaded_freq_dict_gpu,
                                                 "GPU");

    Mat Y_reconstructed_omp, Cb_reconstructed_omp, Cr_reconstructed_omp;
    cout << "Decompressed with OMP" << endl;
    cout << "======================================\n";
    double modifiedTimeOMPForDecode = mainDecode(y_loaded_omp, cb_loaded_omp, cr_loaded_omp,
                                                 y_rows_omp, y_cols_omp, cb_rows_omp, cb_cols_omp, cr_rows_omp, cr_cols_omp,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 Y_reconstructed_omp, Cb_reconstructed_omp, Cr_reconstructed_omp,
                                                 y_loaded_freq_dict_omp, cb_loaded_freq_dict_omp, cr_loaded_freq_dict_omp,
                                                 "OMP");

    cout << "Decoding time (CPU): " << originalTimeForDecode << " ms" << endl;
    cout << "Decoding time (GPU): " << modifiedTimeGPUForDecode << " ms" << endl;
    cout << "Decoding time (OMP): " << modifiedTimeOMPForDecode << " ms" << endl;
    cout << "Decoding Performance Improvement (CPU/GPU): " << originalTimeForDecode / modifiedTimeGPUForDecode << "x" << endl;
    cout << "Decoding Performance Improvement (CPU/OMP): " << originalTimeForDecode / modifiedTimeOMPForDecode << "x" << endl;
    cout << "======================================\n\n";

    // Resize Cb and Cr channels to match the size of Y channel
    // resize(Cb_reconstructed, Cb_reconstructed, Y_reconstructed.size(), 0, 0, INTER_LINEAR);
    // resize(Cr_reconstructed, Cr_reconstructed, Y_reconstructed.size(), 0, 0, INTER_LINEAR);

    // resize(Cb_reconstructed_gpu, Cb_reconstructed_gpu, Y_reconstructed_gpu.size(), 0, 0, INTER_LINEAR);
    // resize(Cr_reconstructed_gpu, Cr_reconstructed_gpu, Y_reconstructed_gpu.size(), 0, 0, INTER_LINEAR);
    Mat Cb_reconstructed_output, Cr_reconstructed_output;
    Mat Cb_reconstructed_output_gpu, Cr_reconstructed_output_gpu;
    Mat Cb_reconstructed_output_omp, Cr_reconstructed_output_omp;
    upsampleChroma(Cb_reconstructed, Cr_reconstructed, Cb_reconstructed_output, Cr_reconstructed_output, image.cols, image.rows);
    upsampleChroma(Cb_reconstructed_gpu, Cr_reconstructed_gpu, Cb_reconstructed_output_gpu, Cr_reconstructed_output_gpu, image.cols, image.rows);
    upsampleChroma(Cb_reconstructed_omp, Cr_reconstructed_omp, Cb_reconstructed_output_omp, Cr_reconstructed_output_omp, image.cols, image.rows);

    vector<Mat> channels_cpu = {Y_reconstructed, Cb_reconstructed_output, Cr_reconstructed_output};
    vector<Mat> channels_gpu = {Y_reconstructed_gpu, Cb_reconstructed_output_gpu, Cr_reconstructed_output_gpu};
    vector<Mat> channels_omp = {Y_reconstructed_omp, Cb_reconstructed_output_omp, Cr_reconstructed_output_omp};
    Mat reconstructed_image_cpu, reconstructed_image_gpu, reconstructed_image_omp;
    merge(channels_cpu, reconstructed_image_cpu);
    merge(channels_gpu, reconstructed_image_gpu);
    merge(channels_omp, reconstructed_image_omp);

    Mat final_image_cpu, final_image_gpu, final_image_omp;
    // cvtColor(reconstructed_image_cpu, final_image_cpu, COLOR_YCrCb2BGR);
    // cvtColor(reconstructed_image_gpu, final_image_gpu, COLOR_YCrCb2BGR);
    final_image_cpu = YCbCr2RGB(reconstructed_image_cpu);
    final_image_gpu = YCbCr2RGB(reconstructed_image_gpu);
    final_image_omp = YCbCr2RGB(reconstructed_image_omp);

    // Save the final image
    string final_image_name_cpu = "output/decompress_image_cpu" + to_string(quality) + ".png";
    string final_image_name_gpu = "output/decompress_image_gpu" + to_string(quality) + ".png";
    string final_image_name_omp = "output/decompress_image_omp" + to_string(quality) + ".png";

    imwrite(final_image_name_cpu, final_image_cpu);
    imwrite(final_image_name_gpu, final_image_gpu);
    imwrite(final_image_name_omp, final_image_omp);

    // Metrics for CPU
    ImageMetric metrics_cpu;
    metrics_cpu.MSE = calculateMSE(image, final_image_cpu);
    metrics_cpu.PSNR = calculatePSNR(image, final_image_cpu);

    // Metrics for GPU
    ImageMetric metrics_gpu;
    metrics_gpu.MSE = calculateMSE(image, final_image_gpu);
    metrics_gpu.PSNR = calculatePSNR(image, final_image_gpu);

    // Metrics for OMP
    ImageMetric metrics_omp;
    metrics_omp.MSE = calculateMSE(image, final_image_omp);
    metrics_omp.PSNR = calculatePSNR(image, final_image_omp);

    // Output the results
    cout << "Image Metrics (CPU):" << endl;
    cout << "  MSE: " << metrics_cpu.MSE << endl;
    cout << "  PSNR: " << metrics_cpu.PSNR << " dB" << endl;
    cout << "======================================\n\n";

    cout << "Image Metrics (GPU):" << endl;
    cout << "  MSE: " << metrics_gpu.MSE << endl;
    cout << "  PSNR: " << metrics_gpu.PSNR << " dB" << endl;
    cout << "======================================\n\n";

    cout << "Image Metrics (OMP):" << endl;
    cout << "  MSE: " << metrics_omp.MSE << endl;
    cout << "  PSNR: " << metrics_omp.PSNR << " dB" << endl;
    cout << "======================================\n\n";

    // vector<double> executionTimesForEncode = {originalTimeForEncode, modifiedTimeGPUForEncode, modifiedTimeOMPForEncode};
    // vector<double> executionTimesForDecode = {originalTimeForDecode, modifiedTimeGPUForDecode, modifiedTimeOMPForDecode};
    // vector<string> labels = {"CPU", "CUDA GPU", "OMP"};
    // drawBarChart(executionTimesForEncode, labels, "Encoding Time Comparison");
    // drawBarChart(executionTimesForDecode, labels, "Decoding Time Comparison");

    return 0;
}
