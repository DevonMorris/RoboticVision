#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>


cv::Mat frame_in;
cv::Mat frame_out;

int kernel_size = 1;
int max_kernel_size = 3;
int filter_low = 1;
int max_filter_low = 100;
int filter_high = 50;
int max_filter_high = 300;

const char* window_name = "Filtered Image";

enum Filter
{
  NO_FILTER,
  CANNY_EDGE,
  GAUSSIAN_BLUR,
  GRADIENT
};

int main(int argc, char *argv[])
{
  // Open default device (Hopefully webcam)
  cv::VideoCapture cap(0);
  Filter mode = Filter::NO_FILTER;

  // Check to see if successfully opend webcam
  if(!cap.isOpened()){
    std::cout << "Error opening video stream" << std::endl;
    return -1;
  }

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::createTrackbar("Kernel Size:", window_name, &kernel_size, max_kernel_size);
  cv::createTrackbar("Filter Low:", window_name, &filter_low, max_filter_low);
  cv::createTrackbar("Filter High:", window_name, &filter_high, max_filter_high);

  // Display images until quit
  while(1){
    cap >> frame_in;

    if (frame_in.empty()){
      std::cout << "Could not grab frame from stream" << std::endl;
    }

    switch(mode) {
      case NO_FILTER: frame_out = frame_in; break;
      case CANNY_EDGE: cv::Canny(frame_in, frame_out, filter_low, filter_high, 2*kernel_size+1); break;
      case GAUSSIAN_BLUR: cv::GaussianBlur(frame_in, frame_out, cv::Size(2*kernel_size + 1, 2*kernel_size+1), filter_high); break; 
      case GRADIENT: cv::Laplacian(frame_in, frame_out, CV_16S, 2*kernel_size+1, 1, 0); break;
    }

    cv::imshow( window_name , frame_out );

    // Press ESC to quit
    char c = (char)cv::waitKey(25);
    if(c==27){
      break;
    }
    switch(c) {
      case 101: mode = Filter::CANNY_EDGE; break;
      case 98: mode = Filter::GAUSSIAN_BLUR; break;
      case 110: mode = Filter::NO_FILTER; break;
      case 103: mode = Filter::GRADIENT; break;
    }

  }

  // Release the default device
  cap.release();

  // Close the windows
  cv::destroyAllWindows();
  return 0;
}
