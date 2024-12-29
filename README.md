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

Ctrl+Shfit+P -> C/C++: Edit Configurations (JSON) to add the open cv and cuda folder path in the includePath.

# How to Run the Program

```bash
nvcc main.cpp kernel.cu -o main `pkg-config --cflags --libs opencv4`
./main
```

# How to Run the Analysis

```bash
nvcc analysis.cpp kernel.cu -o analysis `pkg-config --cflags --libs opencv4`
```

For Auto Generation Image
```
python3 metric_auto_generate_image.py
```

For Own Image (But need to include you image file path at the execute function in metric_image_filee.py
```
python3 metric_image_filee.py
```
