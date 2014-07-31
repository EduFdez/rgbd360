/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the holder(s) nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *  Author: Eduardo Fernandez-Moral
 */

#include <RGBDGrabber_OpenNI2.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <signal.h>

using namespace std;

// This program grabs images from a PrimeSense RGB-D camera (like ASUS Xtion Pro Live) using OpenNI2.

RGBDGrabber_OpenNI2 *grabber;

// Interruption handler to catch ctrl-C.
void INThandler(int sig)
{
  char c;

  signal(sig, SIG_IGN);
  printf("\n  Do you really want to quit? [y/n] ");
//  c = getchar();
  cin >> c;
  if (c == 'y' || c == 'Y')
  {
    delete grabber;
    exit(0);
  }
}

void print_help(char ** argv)
{
  cout << "\nThis program access the available OpenNI2 sensors connected to the computer, and asks ";
  cout << " the user to choose one of them and the resolution before displaying its image stream.";
  cout << "  usage: " <<  argv[0] << "\n";
}

int main (int argc, char ** argv)
{
    if(argc != 1)
    {
      print_help(argv);
      return 0;
    }

    int rc = openni::OpenNI::initialize();
    printf("After initialization:\n %s\n", openni::OpenNI::getExtendedError());

    // Show devices list
    openni::Array<openni::DeviceInfo> deviceList;
    openni::OpenNI::enumerateDevices(&deviceList);
    printf("Get device list. %d devices connected\n", deviceList.getSize() );
    for (unsigned i=0; i < deviceList.getSize(); i++)
    {
      printf("Device %u: name=%s uri=%s vendor=%s \n", i+1 , deviceList[i].getName(), deviceList[i].getUri(), deviceList[i].getVendor());
    }
    if(deviceList.getSize() == 0)
    {
      cout << "No devices connected -> EXIT\n";
      return 0;
    }

//    grabber->showDeviceList();
    cout << "Choose a device: ";
    int device_id; // = static_cast<int>(getchar());
    cin >> device_id;
    cout << "Choose the resolution: 0 (640x480) or 1 (320x240): ";
    int resolution; // = static_cast<int>(getchar());
    cin >> resolution;
    cout << "device_id " << device_id << " resolution " << resolution << endl;

    grabber = new RGBDGrabber_OpenNI2(deviceList[device_id-1].getUri(), resolution);
//    grabber->setSensor(device_id);
//    grabber->setResolution(resolution);

    grabber->init();
    cout << "Grabber initialized\n";

//    sleep(2);
//    grabber->setShutter(15);

    //Create and initialize an object to grab RGBD frames
    FrameRGBD *frame;
//    RGBDGrabber_OpenNI2 *grabber;

    try
    {
      signal(SIGINT, INThandler);
      cv::namedWindow( "rgb", CV_WINDOW_AUTOSIZE );
      cv::namedWindow( "depth", CV_WINDOW_AUTOSIZE );

//      int shutter = 0;
      unsigned frame_count = 0;
      char keypressed = 'a';
      while(keypressed!='\n')
      {
        keypressed = cv::waitKey(10);
//      cout << "Get frame " << frame_count << "\n"; cout << "keypressed " << keypressed << "\n";
//        ++frame_count;
//        if(frame_count%10 == 0 && shutter < 30)
//          grabber->setShutter(++shutter);

        frame = new FrameRGBD();
        grabber->grab(frame);

//        // Shutter control. To change by PID control
//        int newExposure;
//        float exposureFactor = 1.0 - frame->getAverageIntensity()/128.0;
//        if(exposureFactor > 0) // Increase shutter time
//          newExposure = min(30, static_cast<int>(floor((1+exposureFactor)*grabber->getShutter())+0.5));
//        else // Decrease shutter time
//          newExposure = max(1, static_cast<int>(ceil((1+0.1*grabber->getShutter()*exposureFactor)*grabber->getShutter())));
//        grabber->setShutter(newExposure);
//      cout << "frame AverageIntensity " << frame->getAverageIntensity() << " shutter " << grabber->getShutter() << "\n";
//      cout << "newExposure " << newExposure << " exposureFactor " << exposureFactor << "\n\n";
////      cout << "grab frame " << frame->getRGBImage().rows << "x" << frame->getRGBImage().cols << "\n";

        // Visualize image
        cv::imshow( "rgb", frame->getRGBImage() );

        cv::Mat depthVisualize = cv::Mat(frame->getRGBImage().rows,frame->getRGBImage().cols,CV_8UC1);
        for(unsigned i=0; i < frame->getRGBImage().rows; i++)
          for(unsigned j=0; j < frame->getRGBImage().cols; j++)
            depthVisualize.at<u_int8_t>(i,j) = static_cast<u_int8_t>(255.0-255.0*frame->getDepthImage().at<unsigned short>(i,j)/10000);
        cv::imshow( "depth", depthVisualize );

//        // Save images
//        cv::imwrite("RGB.ppm", &frame->getRGBImage());
//        cv::imwrite("range.ppm", &frame->getDepthImage());

        delete frame;
      }

      cv::destroyAllWindows();
//      cv::destroyWindow("rgb");
//      cv::destroyWindow("depth");

    }catch(std::exception &e){ //try-catch to avoid crashing when there are no more frames in the dataset
			cerr << e.what() << endl;
    }

    //Free the allocated objects
    delete grabber; //Stop the RGBD grabber

  return (0);
}
