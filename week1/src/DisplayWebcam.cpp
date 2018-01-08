#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
  // Open default device (Hopefully webcam)
  cv::VideoCapture cap(0);

  // Check to see if successfully opend webcam
  if(!cap.isOpened()){
    std::cout << "Error opening video stream" << std::endl;
  }

  // Display images until quit
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

  // Release the default device
  cap.release();

  // Close the windows
  cv::destroyAllWindows();
  return 0;
}
