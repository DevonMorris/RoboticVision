#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>


cv::Mat frame_in;
cv::Mat frame_in_gray;
cv::Mat frame_out;
cv::Mat frame_out_gray;

int kernel_size = 1;
int max_kernel_size = 3;
int filter_low = 1;
int max_filter_low = 100;
int filter_high = 50;
int max_filter_high = 300;

const std::string window_name = "Filtered Image";

enum Filter
{
  NO_FILTER,
  CANNY_EDGE,
  SOBEL,
  SOBELX,
  SOBELY,
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
    cv::cvtColor(frame_in, frame_in_gray, cv::COLOR_BGR2GRAY); 

    if (frame_in.empty()){
      std::cout << "Could not grab frame from stream" << std::endl;
    }

    switch(mode) {
      case NO_FILTER: 
        frame_out = frame_in; 
        frame_out_gray = frame_in_gray;
        break;
      case CANNY_EDGE: 
        cv::Canny(frame_in, frame_out, filter_low, filter_high, 2*kernel_size+1); 
        cv::Canny(frame_in_gray, frame_out_gray, filter_low, filter_high, 2*kernel_size+1); 
        break;
      case GAUSSIAN_BLUR: 
       cv::GaussianBlur(frame_in, frame_out, cv::Size(2*kernel_size + 1, 2*kernel_size+1), filter_high); 
       cv::GaussianBlur(frame_in_gray, frame_out_gray, cv::Size(2*kernel_size + 1, 2*kernel_size+1), filter_high);
       break; 
      case GRADIENT: 
       cv::Laplacian(frame_in, frame_out, CV_32F, 2*kernel_size+1, 1, 0); 
       cv::Laplacian(frame_in_gray, frame_out_gray, CV_32F, 2*kernel_size+1, 1, 0); 
       break;
      case SOBEL: 
       cv::Sobel(frame_in, frame_out, CV_32F, 1, 1, 2*kernel_size+1); 
       cv::Sobel(frame_in_gray, frame_out_gray, CV_32F, 1, 1, 2*kernel_size+1); 
       break;
      case SOBELX: 
       cv::Sobel(frame_in, frame_out, CV_32F, 1, 0, 2*kernel_size+1); 
       cv::Sobel(frame_in_gray, frame_out_gray, CV_32F, 1, 0, 2*kernel_size+1); 
       break;
      case SOBELY: 
       cv::Sobel(frame_in, frame_out, CV_32F, 0, 1, 2*kernel_size+1); 
       cv::Sobel(frame_in_gray, frame_out_gray, CV_32F, 0, 1, 2*kernel_size+1); 
       break;
    }

    cv::imshow( window_name , frame_out );
    cv::imshow( window_name+" gray" , frame_out_gray );

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
      case 115: mode = Filter::SOBEL; break;
      case 104: mode = Filter::SOBELX; break;
      case 108: mode = Filter::SOBELY; break;
    }

  }

  // Release the default device
  cap.release();

  // Close the windows
  cv::destroyAllWindows();
  return 0;
}
