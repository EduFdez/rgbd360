/*
 *  Copyright (c) 2012, Universidad de Málaga - Grupo MAPIR
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

#ifndef CLOUD_UNDISTORT
#define CLOUD_UNDISTORT

#include <CloudRGBD_Ext.h>
#include <CloudVisualizer.h>
#include <RGBDGrabber_OpenNI2.h>
//#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <signal.h>

using namespace std;

RGBDGrabber_OpenNI2 *grabber; // This variable is global to be able to catch interruptions lke CTRL-C

// Interruption handler to catch ctrl-C.
void INThandler(int sig)
{
  char c;

  signal(sig, SIG_IGN);
  printf("\n  Do you really want to quit? [y/n] ");
  c = getchar();
  if (c == 'y' || c == 'Y')
  {
    delete grabber;
    exit(0);
  }

  signal(SIGINT, INThandler);
  getchar(); // Get new line character
}

void print_help(char **argv)
{
  cout << "usage: ./" << argv[0] << "<pathToIntrinsicModel>" << "\n";
}

// This program grabs images from a PrimeSense RGB-D camera (like ASUS Xtion Pro Live) using OpenNI2.
// The data stream can be saved to binary files Using MRPT or Boost. This program also allows to read
// the images from such binary files.
int main (int argc, char ** argv)
{
    //Create and initialize an object to grab RGBD frames
    CloudRGBD *frameCloud = new CloudRGBD();
//    RGBDGrabber_OpenNI2 *grabber;

    string intrisic_model_path;
    if(argc == 1)
      intrisic_model_path();
    else
      intrisic_model_path = static_cast<string>(argv[1]);

    grabber = new RGBDGrabber_OpenNI2(openni::ANY_DEVICE, 1);
    grabber->init();
    cout << "Grabber initialized\n";

    try
    {
      signal(SIGINT, INThandler);

      CloudVisualization visualizer(frameCloud);

      clams::DiscreteDepthDistortionModel intrinsic_model;
      intrinsic_model.load(intrisic_model_path);
      intrinsic_model.downsampleParams(2);

      unsigned frame_count = 0;
      while(!visualizer.cloudViewer.wasStopped() )
      {
        ++frame_count;
//        cout << "grab new frame \n";

//        frameCloud = new FrameRGBD_undistort(intrinsic_model);
        frameCloud = new CloudRGBD();
        grabber->grab(frameCloud);
        frameCloud->getPointCloud();
//      cout << "Cloud size " << frameCloud->getPointCloud()->size() << endl;

        { boost::mutex::scoped_lock lock (visualizer.mtx_visualization);
//          if(visualizer.cloudRGBD != NULL)
            delete visualizer.cloudRGBD;
          visualizer.cloudRGBD = frameCloud;
        }
//        sleep(1);
//        cout << "grab frame " << frameCloud->getPointCloud()->size() << " " << visualizer.cloudRGBD->getPointCloud()->size() << "\n";
      }
      cout << "EXIT program \n";

    }catch(std::exception &e){ //try-catch to avoid crashing when there are no more frames in the dataset
			cerr << e.what() << endl;
    }

    //Free the allocated objects
    delete grabber; //Stop the RGBD grabber
//    delete frameCloud;

  return (0);
}
#endif
