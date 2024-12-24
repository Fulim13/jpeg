# How to Run opencv_test.cpp

```
g++ opencv_test.cpp -o opencv_test `pkg-config --cflags --libs opencv4`
./opencv_test
```

# How to Run single_ImageBlock.cpp

```
g++ single_ImageBlock.cpp -o single_ImageBlock
./single_ImageBlock
```

# How to Run multiple_ImageBlock.cpp

```
g++ multiple_ImageBlock.cpp -o multiple_ImageBlock
./multiple_ImageBlock
```

# How to Run multiple_ImageBlock_Parallel.cu

```
nvcc multiple_ImageBlock_Parallel.cu -o multiple_ImageBlock_Parallel

```
