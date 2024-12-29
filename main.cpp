#include "encode_decode.cpp"

int main()
{
    string folder_name = "img/";
    string image_name = "img/Boy_1024.png";

    cout << "Enter the image name: ";
    cin >> image_name;

    image_name = folder_name + image_name;

    // Load the image
    Mat image = readImage(image_name);
    if (image.empty())
    {
        throw runtime_error("Failed to load image");
    }

    ensureMultipleOf16(image);

    // Print the Image Dimensions
    cout << "Image Dimensions: " << image.rows << " x " << image.cols << endl;
    cout << "====================================================\n";

    // Convert the image From RGB to YCbCr
    Mat ycbcr_image;
    cvtColor(image, ycbcr_image, COLOR_BGR2YCrCb);

    // Perform chroma subsampling
    Mat Y, Cb, Cr;
    chromaSubsampling(ycbcr_image, Y, Cb, Cr);

    // Show the original
    // imshow("Original Image", image);

    // // Show the Y, Cb, and Cr channels
    // imshow("Y Channel", Y);
    // imshow("Cb Channel (Subsampled)", Cb);
    // imshow("Cr Channel (Subsampled)", Cr);
    // waitKey(0);

    // Encode and Calculate the time for encoding
    EncodedData y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu;
    EncodedData y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu;
    cout << "Compressed with CPU" << endl;
    cout << "======================================\n";
    double originalTimeForEncode = mainEncode(Y, Cb, Cr,
                                              quantization_table_Y, quantization_table_CbCr,
                                              y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu, "CPU");

    cout << "Compressed with GPU" << endl;
    cout << "======================================\n";
    double modifiedTimeGPUForEncode = mainEncode(Y, Cb, Cr,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu, "GPU");

    cout << "Encoding time (CPU): " << originalTimeForEncode << " ms" << endl;
    cout << "Encoding time (GPU): " << modifiedTimeGPUForEncode << " ms" << endl;
    cout << "Encoding Performance Improvement: " << originalTimeForEncode / modifiedTimeGPUForEncode << "x" << endl;
    cout << "======================================\n\n";

    // CPU - Save three encoded data (EncodedData) and rows and cols for each channel to one bin file
    string compressed_filename_cpu = "output/compressed_image_cpu.bin";
    saveEncodedData(compressed_filename_cpu,
                    y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols);

    cout << "Compressed file (CPU) saved in: " << compressed_filename_cpu << endl;

    // GPU - Save three encoded data (EncodedData) and rows and cols for each channel to one bin file
    string compressed_filename_gpu = "output/compressed_image_gpu.bin";
    saveEncodedData(compressed_filename_gpu,
                    y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols);
    cout << "Compressed file (GPU) saved in: " << compressed_filename_cpu << endl;

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

    // CPU - Load encoded data (As EncodedData) from file and Read the rows and cols for each channel
    EncodedData y_loaded, cb_loaded, cr_loaded;
    int y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols;
    loadEncodedData(compressed_filename_cpu,
                    y_loaded, cb_loaded, cr_loaded,
                    y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols);

    // GPU - Load encoded data (As EncodedData) from file and Read the rows and cols for each channel
    EncodedData y_loaded_gpu, cb_loaded_gpu, cr_loaded_gpu;
    int y_rows_gpu, y_cols_gpu, cb_rows_gpu, cb_cols_gpu, cr_rows_gpu, cr_cols_gpu;
    loadEncodedData(compressed_filename_gpu,
                    y_loaded_gpu, cb_loaded_gpu, cr_loaded_gpu,
                    y_rows_gpu, y_cols_gpu, cb_rows_gpu, cb_cols_gpu, cr_rows_gpu, cr_cols_gpu);

    // Decode and Calculate the time for decoding
    Mat Y_reconstructed, Cb_reconstructed, Cr_reconstructed;
    cout << "Decompressed with CPU" << endl;
    cout << "======================================\n";
    double originalTimeForDecode = mainDecode(y_loaded, cb_loaded, cr_loaded,
                                              y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols,
                                              quantization_table_Y, quantization_table_CbCr,
                                              Y_reconstructed, Cb_reconstructed, Cr_reconstructed, "CPU");

    Mat Y_reconstructed_gpu, Cb_reconstructed_gpu, Cr_reconstructed_gpu;
    cout << "Decompressed with GPU" << endl;
    cout << "======================================\n";
    double modifiedTimeGPUForDecode = mainDecode(y_loaded_gpu, cb_loaded_gpu, cr_loaded_gpu,
                                                 y_rows_gpu, y_cols_gpu, cb_rows_gpu, cb_cols_gpu, cr_rows_gpu, cr_cols_gpu,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 Y_reconstructed_gpu, Cb_reconstructed_gpu, Cr_reconstructed_gpu, "GPU");

    cout << "Decoding time (CPU): " << originalTimeForDecode << " ms" << endl;
    cout << "Decoding time (GPU): " << modifiedTimeGPUForDecode << " ms" << endl;
    cout << "Decoding Performance Improvement (GPU): " << originalTimeForDecode / modifiedTimeGPUForDecode << "x" << endl;
    cout << "======================================\n\n";

    // // Show the reconstructed images for CPU
    // imshow("Reconstructed Y Channel (CPU)", Y_reconstructed);
    // imshow("Reconstructed Cb Channel (CPU)", Cb_reconstructed);
    // imshow("Reconstructed Cr Channel (CPU)", Cr_reconstructed);

    // // Show the reconstructed images for GPU
    // imshow("Reconstructed Y Channel (GPU)", Y_reconstructed_gpu);
    // imshow("Reconstructed Cb Channel (GPU)", Cb_reconstructed_gpu);
    // imshow("Reconstructed Cr Channel (GPU)", Cr_reconstructed_gpu);
    // waitKey(0);

    // Merge the Y, Cb, Cr channels and convert back to BGR
    // Resize Cb and Cr channels to match the size of Y channel
    resize(Cb_reconstructed, Cb_reconstructed, Y_reconstructed.size(), 0, 0, INTER_LINEAR);
    resize(Cr_reconstructed, Cr_reconstructed, Y_reconstructed.size(), 0, 0, INTER_LINEAR);

    resize(Cb_reconstructed_gpu, Cb_reconstructed_gpu, Y_reconstructed_gpu.size(), 0, 0, INTER_LINEAR);
    resize(Cr_reconstructed_gpu, Cr_reconstructed_gpu, Y_reconstructed_gpu.size(), 0, 0, INTER_LINEAR);
    Mat reconstructed_image_cpu, reconstructed_image_gpu;

    vector<Mat> channels_cpu = {Y_reconstructed, Cb_reconstructed, Cr_reconstructed};
    vector<Mat> channels_gpu = {Y_reconstructed_gpu, Cb_reconstructed_gpu, Cr_reconstructed_gpu};

    merge(channels_cpu, reconstructed_image_cpu);
    merge(channels_gpu, reconstructed_image_gpu);

    Mat final_image_cpu, final_image_gpu;
    cvtColor(reconstructed_image_cpu, final_image_cpu, COLOR_YCrCb2BGR);
    cvtColor(reconstructed_image_gpu, final_image_gpu, COLOR_YCrCb2BGR);

    // Save the final image
    string final_image_name_cpu = "output/final_image_cpu.png";
    string final_image_name_gpu = "output/final_image_gpu.png";

    imwrite(final_image_name_cpu, final_image_cpu);
    imwrite(final_image_name_gpu, final_image_gpu);

    // Display the original and final image
    imshow("Original Image", image);
    imshow("Final Image (CPU)", final_image_cpu);
    imshow("Final Image (GPU)", final_image_gpu);
    waitKey(0);

    // Metrics for CPU
    ImageMetric metrics_cpu;
    metrics_cpu.MSE = calculateMSE(image, final_image_cpu);
    metrics_cpu.PSNR = calculatePSNR(image, final_image_cpu);

    // Metrics for GPU
    ImageMetric metrics_gpu;
    metrics_gpu.MSE = calculateMSE(image, final_image_gpu);
    metrics_gpu.PSNR = calculatePSNR(image, final_image_gpu);

    // Output the results
    cout << "Image Metrics (CPU):" << endl;
    cout << "  MSE: " << metrics_cpu.MSE << endl;
    cout << "  PSNR: " << metrics_cpu.PSNR << " dB" << endl;
    cout << "======================================\n\n";

    cout << "Image Metrics (GPU):" << endl;
    cout << "  MSE: " << metrics_gpu.MSE << endl;
    cout << "  PSNR: " << metrics_gpu.PSNR << " dB" << endl;
    cout << "======================================\n\n";

    vector<double> executionTimesForEncode = {originalTimeForEncode, modifiedTimeGPUForEncode};
    vector<double> executionTimesForDecode = {originalTimeForDecode, modifiedTimeGPUForDecode};
    vector<string> labels = {"CPU", "CUDA GPU"};
    drawBarChart(executionTimesForEncode, labels, "Encoding Time Comparison");
    drawBarChart(executionTimesForDecode, labels, "Decoding Time Comparison");

    return 0;
}
