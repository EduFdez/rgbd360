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
 *  Author: efernand Fernandez-Moral
 */

#define SHOW_IMAGES 0
#define SAVE_IMAGES 0
#define VISUALIZE_POINT_CLOUD 0
#define PBMAP_ODOMETRY 0

#include <Frame360.h>

#if VISUALIZE_POINT_CLOUD
  #include <Frame360_Visualizer.h>
#endif

#if PBMAP_ODOMETRY
  #include <RegisterRGBD360.h>
#endif

#include <pcl/console/parse.h>

#include <RGBDGrabber_OpenNI2.h>

#include <SerializeFrameRGBD.h> // For time-stamp conversion

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <signal.h>

using namespace std;

RGBDGrabber_OpenNI2 *grabber[8]; // This object is declared as global to be able to use it after capturing Ctrl-C interruptions

/*! Catch interruptions like Ctrl-C */
void INThandler(int sig)
{
  char c;

  signal(sig, SIG_IGN);
  printf("\n  Do you really want to quit? [y/n] ");
  c = getchar();
  if (c == 'y' || c == 'Y')
  {
    for(unsigned sensor_id = 0; sensor_id < 8; sensor_id++)
    {
      delete grabber[sensor_id];
    }
    exit(0);
  }
}


/*! This class provides the functionality to read the omnidirectional RGB-D image stream from the RGBD360 sensor
 *  developed at INRIA Sophia Antipolis, and to save the data to disk. */
class RGBD360_Grabber
{
  public:
    RGBD360_Grabber()
    {
    }

    void run(int exposure, string path)
    {
      // Handle interruptions
      signal(SIGINT, INThandler);

      unsigned frame = 0, keyframe_id = 0;
      cv::Mat timeStampMatrix;

      Calib360 calib;
      calib.loadIntrinsicCalibration();
      calib.loadExtrinsicCalibration();

//      int rc = openni::OpenNI::initialize();
//      printf("After initialization:\n %s\n", openni::OpenNI::getExtendedError());
//
//      // Show devices list
//      openni::Array<openni::DeviceInfo> deviceList;
//      openni::OpenNI::enumerateDevices(&deviceList);
//      printf("Get device list. %d devices connected\n", deviceList.getSize() );
//      for (unsigned i=0; i < deviceList.getSize(); i++)
//      {
//        printf("Device %u: name=%s uri=%s vendor=%s \n", i+1 , deviceList[i].getName(), deviceList[i].getUri(), deviceList[i].getVendor());
//      }
//      if(deviceList.getSize() == 0)
//      {
//        cout << "No devices connected -> EXIT\n";
//        return;
//      }

//      RGBDGrabber *grabber[8];
      // Create one instance of RGBDGrabber_OpenNI2 for each sensor
      grabber[0] = new RGBDGrabber_OpenNI2("1d27/0601@5/2", 1, exposure);
      grabber[1] = new RGBDGrabber_OpenNI2("1d27/0601@4/2", 1, exposure);
      grabber[2] = new RGBDGrabber_OpenNI2("1d27/0601@3/2", 1, exposure);
      grabber[3] = new RGBDGrabber_OpenNI2("1d27/0601@6/2", 1, exposure);
      grabber[4] = new RGBDGrabber_OpenNI2("1d27/0601@9/2", 1, exposure);
      grabber[5] = new RGBDGrabber_OpenNI2("1d27/0601@8/2", 1, exposure);
      grabber[6] = new RGBDGrabber_OpenNI2("1d27/0601@7/2", 1, exposure);
      grabber[7] = new RGBDGrabber_OpenNI2("1d27/0601@10/2", 1, exposure);

      // Initialize the sensor
//      int rc = openni::OpenNI::initialize();
//      printf("After initialization:\n %s\n", openni::OpenNI::getExtendedError());
      for(int sensor_id = 0; sensor_id < 8; sensor_id++)
        grabber[sensor_id]->init();
      cout << "Grabber initialized\n";

      // Get the first frame
      Frame360 *frame360_1 = new Frame360(&calib);

      // Skip the first frames
      unsigned skipNFrames = 2;
      for(unsigned skipFrames = 0; skipFrames < skipNFrames; skipFrames++ )
        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
          grabber[sensor_id]->grab(&frame360_1->frameRGBD_[sensor_id]);

//      // Save the first frame
//      string fileName = path + mrpt::format("/sphere_images_%d.bin",keyframe_id++);
//      std::ofstream ofs_images(fileName.c_str(), std::ios::out | std::ios::binary);
//      boost::archive::binary_oarchive oa_images(ofs_images);
//
//      for(int sensor_id = 0; sensor_id < 8; sensor_id++)
//        oa_images << frame360_1->frameRGBD_[sensor_id].getRGBImage() << frame360_1->frameRGBD_[sensor_id].getDepthImage();
//      getMatrixNumberRepresentationOf_uint64_t(mrpt::system::getCurrentTime(),timeStampMatrix);
////      frame360_1->setTimeStamp(mrpt::system::getCurrentTime());
//      oa_images << timeStampMatrix;
//      ofs_images.close();

      // Initialize visualizer
      #if VISUALIZE_POINT_CLOUD
        Frame360_Visualizer viewer(frame360_1);
      #endif

      #if PBMAP_ODOMETRY
        RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));

        frame360_1->undistort();
        frame360_1->buildSphereCloud_rgbd360();
        frame360_1->getPlanes();
      #endif

      #if SHOW_IMAGES
        cv::namedWindow("sphereRGB");
      #endif

    #if SHOW_IMAGES
      char pressedKey = 'a';
      while (pressedKey!='\n') {
        pressedKey = cv::waitKey(1);
//          cout << "Pressed key " << pressedKey << endl;
    #else
      while(true) {
//        while (!viewer.wasStopped() )
    #endif
        double frame_start = pcl::getTime ();

        Frame360 *frame360_2 = new Frame360(&calib);

        #pragma omp parallel num_threads(8)
        {
          int sensor_id = omp_get_thread_num();
          grabber[sensor_id]->grab(&(frame360_2->frameRGBD_[sensor_id]));
        }
//        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
//          grabber[sensor_id]->grab(&(frame360_2->frameRGBD_[sensor_id]));

        getMatrixNumberRepresentationOf_uint64_t(mrpt::system::getCurrentTime(),timeStampMatrix);
//      cout << "grab new frame\n";

        #if SHOW_IMAGES
//          frame360_2->stitchSphericalImage();
          frame360_2->fastStitchImage360();
          cv::imshow( "sphereRGB", frame360_2->sphereRGB );
//            cv::imshow( "sphereDepth", sphereDepth );
//        imwrite(mrpt::format("/home/efernand/Datasets_RGBD360/Data_RGBD360/PNG/sphereRGB_%d.png", frame), sphereRGB);
//        imwrite(mrpt::format("/home/efernand/Datasets_RGBD360/Data_RGBD360/PNG/sphereDepth_%d.png", frame), sphereDepth);
        #endif

        if(VISUALIZE_POINT_CLOUD || PBMAP_ODOMETRY)
        {
          frame360_2->undistort();
          frame360_2->buildSphereCloud_rgbd360();
        }

        #if VISUALIZE_POINT_CLOUD
        { boost::mutex::scoped_lock updateLock(viewer.visualizationMutex);
          Frame360_Visualizer viewer(frame360_2);
        }
        #endif
//
        #if PBMAP_ODOMETRY
          cout << " Register frame\n";

          frame360_2->getPlanes();

          bool bGoodRegistration = registerer.Register(&frame360_1, &frame360_2, 25, RegisterRGBD360::PLANAR_3DoF);

          // Save as a keyframe
          if(!bGoodRegistration || registerer.getMatchedPlanes().size() < 8 || registerer.getPose().block(0,3,3,1).norm() > 0.1)
          {
            cout << "Save new keyframe\n";
            delete frame360_1;
            frame360_1 = frame360_2;

            string fileName = path + mrpt::format("/sphere_images_%d.bin",keyframe_id++);
            std::ofstream ofs_images(fileName.c_str(), std::ios::out | std::ios::binary);
            boost::archive::binary_oarchive oa_images(ofs_images);

            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
              oa_images << frame360_1->frameRGBD_[sensor_id].getRGBImage() << frame360_1->frameRGBD_[sensor_id].getDepthImage();
            oa_images << timeStampMatrix;

            ofs_images.close();
            cout << " binary file saved... " << std::endl;
          }
          else
          {
            delete frame360_2;
          }
        #else
          if(frame++ % 3 == 0) // Reduce the sensor framerate
          {
            string fileName = path + mrpt::format("/sphere_images_%d.bin",keyframe_id++);
            std::ofstream ofs_images(fileName.c_str(), std::ios::out | std::ios::binary);
            boost::archive::binary_oarchive oa_images(ofs_images);

            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
              oa_images << frame360_2->frameRGBD_[sensor_id].getRGBImage() << frame360_2->frameRGBD_[sensor_id].getDepthImage();
            oa_images << timeStampMatrix;

            ofs_images.close();

            cout << " binary file saved... " << std::endl;
          }
          delete frame360_2;
        #endif

        double frame_end = pcl::getTime ();
        cout << " Grabbing in " << (frame_end - frame_start)*1e3 << " ms\n";
      }

      delete frame360_1;

      #if SHOW_IMAGES
        cv::destroyWindow("sphereRGB");
  //        cv::destroyWindow("sphereDepth");
      #endif

      for(unsigned sensor_id = 0; sensor_id < 8; sensor_id++)
        delete grabber[sensor_id];

      openni::OpenNI::shutdown();
    }

};


void print_help(char ** argv)
{
  cout << "\nThis program accesses the omnidirectional RGB-D sensor, and reads the image streaming it captures."
      << " The image stream is recorded in the path specified by the user.\n";
  cout << "usage: " << argv[0] << " [options] \n";
  cout << argv[0] << " -h | --help : shows this help" << endl;
  cout << argv[0] << " <intExposure> <pathToSaveToDisk>" << endl;
}

int main (int argc, char ** argv)
{
  if(argc != 3 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  int exposure = atoi(argv[1]);
  string path_results = static_cast<string>(argv[2]);

  cout << "Create RGBD360_Grabber object\n";
  RGBD360_Grabber rgbd360;
  rgbd360.run(exposure, path_results);

  cout << "EXIT\n";

  return (0);
}
