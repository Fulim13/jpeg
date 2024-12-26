#include "encode_decode.cpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    string image_name = argv[1];

    // Load the image
    Mat image = readImage(image_name);
    if (image.empty())
    {
        throw runtime_error("Failed to load image");
    }

    ensureMultipleOf16(image);

    // Convert the image From RGB to YCbCr
    Mat ycbcr_image;
    cvtColor(image, ycbcr_image, COLOR_BGR2YCrCb);

    // Perform chroma subsampling
    Mat Y, Cb, Cr;
    chromaSubsampling(ycbcr_image, Y, Cb, Cr);

    // Encode and Calculate the time for encoding
    EncodedData y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu;
    EncodedData y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu;
    double originalTimeForEncode = mainEncode(Y, Cb, Cr,
                                              quantization_table_Y, quantization_table_CbCr,
                                              y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu, "CPU");

    double modifiedTimeGPUForEncode = mainEncode(Y, Cb, Cr,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu, "GPU");

    // CPU - Save three encoded data (EncodedData) and rows and cols for each channel to one bin file
    string compressed_filename_cpu = "output/compressed_image_cpu.bin";
    saveEncodedData(compressed_filename_cpu,
                    y_encoded_cpu, cb_encoded_cpu, cr_encoded_cpu,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols);

    // GPU - Save three encoded data (EncodedData) and rows and cols for each channel to one bin file
    string compressed_filename_gpu = "output/compressed_image_gpu.bin";
    saveEncodedData(compressed_filename_gpu,
                    y_encoded_gpu, cb_encoded_gpu, cr_encoded_gpu,
                    Y.rows, Y.cols, Cb.rows, Cb.cols, Cr.rows, Cr.cols);

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
    double originalTimeForDecode = mainDecode(y_loaded, cb_loaded, cr_loaded,
                                              y_rows, y_cols, cb_rows, cb_cols, cr_rows, cr_cols,
                                              quantization_table_Y, quantization_table_CbCr,
                                              Y_reconstructed, Cb_reconstructed, Cr_reconstructed, "CPU");

    Mat Y_reconstructed_gpu, Cb_reconstructed_gpu, Cr_reconstructed_gpu;
    double modifiedTimeGPUForDecode = mainDecode(y_loaded_gpu, cb_loaded_gpu, cr_loaded_gpu,
                                                 y_rows_gpu, y_cols_gpu, cb_rows_gpu, cb_cols_gpu, cr_rows_gpu, cr_cols_gpu,
                                                 quantization_table_Y, quantization_table_CbCr,
                                                 Y_reconstructed_gpu, Cb_reconstructed_gpu, Cr_reconstructed_gpu, "GPU");

    cout << originalTimeForEncode / modifiedTimeGPUForEncode << endl;
    cout << originalTimeForDecode / modifiedTimeGPUForDecode << endl;

    // Stat For CPU
    CompressionStats statsCPU = calculateCompressionStats(image_name, compressed_filename_cpu);
    cout << statsCPU.compression_ratio << endl;

    // Stat For GPU
    CompressionStats statsGPU = calculateCompressionStats(image_name, compressed_filename_gpu);
    cout << statsGPU.compression_ratio << endl;

    return 0;
}
