#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
  cv::VideoCapture cap(0);

  if(!cap.isOpened()){
    std::cout << "Error opening video stream" << std::endl;
  }

  while(1){
    cv::Mat frame;
    cap >> frame;
    
    if (frame.empty()){
      std::cout << "Could not grab frame from stream" << std::endl;
    }
    
    cv::imshow( "Frame", frame );

    // Press ESC to quit
    char c = (char)cv::waitKey(25);
    if(c==27){
      break;
    }

  }

  cap.release();

  cv::destroyAllWindows();
  return 0;
}
