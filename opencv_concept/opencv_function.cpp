#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // Load the image and store it in a matrix
    cv::Mat img = cv::imread("../img/Lenna_512.png");
    if (img.empty())
    {
        std::cout << "Error: Could not read the image!" << std::endl;
        return -1;
    }
    std::cout << "Image loaded successfully. Size: " << img.rows << "x" << img.cols << std::endl;

    // Display the original image
    cv::imshow("Original Image", img);
    cv::waitKey(0);

    // Convert the image to grayscale
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    // Display the grayscale image
    cv::imshow("Grayscale Image", grayImg);
    cv::waitKey(0);

    // Blur the image
    cv::Mat blurredImg;
    cv::GaussianBlur(img, blurredImg, cv::Size(15, 15), 0);

    // Display the blurred image
    cv::imshow("Blurred Image", blurredImg);
    cv::waitKey(0);

    // Detect edges using Canny
    cv::Mat edges;
    cv::Canny(grayImg, edges, 100, 200);

    // Display the edges
    cv::imshow("Edges", edges);
    cv::waitKey(0);

    // Draw a circle on the image
    cv::Mat imgWithCircle = img.clone();
    // circle(imgWithCircle, center , radius, color, thickness)
    // center: center of the circle is set to the center of the image
    // radius: radius of the circle
    // color: color of the circle (BGR) G - 255
    // thickness: 3-pixel thick outline.
    cv::circle(imgWithCircle, cv::Point(img.cols / 2, img.rows / 2), 50, cv::Scalar(0, 255, 0), 3);

    // Display the image with the circle
    cv::imshow("Image with Circle", imgWithCircle);
    cv::waitKey(0);

    return 0;
}
