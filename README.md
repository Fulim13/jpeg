# Things havent completed yet

1. Study the Compression Metrics: PSNR, MSE, Compression Ratio
2. Write Report about the result

# How to Run the Program

```bash
nvcc main_with_store_huffman_tree.cpp kernel.cu -o main `pkg-config --cflags --libs opencv4` -diag-suppress=611
./main
```

# Install OpenCV in C++

```bash
sudo apt-get install libopencv-dev
```

# Setting up OpenCV in VS Code

Check the version of OpenCV installed in your system using the following command:

```bash
pkg-config --modversion opencv4
```

Locate the path of the OpenCV library using the following command:

```bash
pkg-config --cflags opencv4
```

Ctrl+Shfit+P -> C/C++: Edit Configurations (JSON) to add the open cv folder path in the includePath.

```
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/include/opencv4"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```

# Run the Test Code for OpenCV in C++

```bash
g++ opencv_test.cpp -o opencv_test `pkg-config --cflags --libs opencv4`
./opencv_test
```

# Run cu

```bash
nvcc main.cu -o main `pkg-config --cflags --libs opencv4` -diag-suppress=611
./main
```

# Run main.cpp

```
nvcc main.cpp -o main `pkg-config --cflags --libs opencv4` -diag-suppress=611
./main
```
