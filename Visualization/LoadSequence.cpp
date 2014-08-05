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

#define RECORD_VIDEO 0

#include <Frame360.h>
#include <Frame360_Visualizer.h>

#include <pcl/common/time.h>
#include <pcl/console/parse.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include <dirent.h> // To read files from disk

using namespace std;

/*! This class' main function 'run' loads a sequence of omnidirectional RGB-D images (RGBD360.bin)
 *  and visualizes and/or saves the spherical images (.png) and/or the PointCloud (.pcd) + PbMap (.pbmap)
 */
class LoadSequence
{
  public:
//    LoadSequence()
//    {
//    }

    void run(int &mode, string &path_dataset, string &path_results)
    {
      // Make sure that a valid mode is used
      assert(mode >= 1 && mode <= 5);
//      cout << " mode = 1 -> Show the reconstructed spherical images \n";
//      cout << " mode = 2 -> Show and save the reconstructed spherical images \n";
//      cout << " mode = 3 -> Show the reconstructed PointCloud and the PbMap \n";
//      cout << " mode = 4 -> Show the reconstructed PointCloud and the PbMap \n";
//      cout << " mode = 5 -> Show a video streaming of the reconstructed PointCloud \n";

      // Load calibration
      Calib360 calib;
      calib.loadIntrinsicCalibration();
      calib.loadExtrinsicCalibration();

      // For PointCloud visualization
      Frame360 *frame360 = new Frame360(&calib);
      Frame360_Visualizer *SphereViewer;

      unsigned frame = 0;

      if(mode == 1 || mode == 2)
      {
        cv::namedWindow( "sphereRGB" );
//        cv::namedWindow( "sphereRGB", CV_WINDOW_AUTOSIZE );
        cv::namedWindow( "sphereDepth", CV_WINDOW_AUTOSIZE );
      }
      else if(mode == 3 || mode == 5)
      {
        SphereViewer = new Frame360_Visualizer(frame360);
        SphereViewer->frameIdx = frame;
        #if RECORD_VIDEO
          sleep(2); // Give me some time to play with the view
        #endif
      }

      string fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
      cout << "fileName " << fileName << " " << fexists(fileName.c_str()) << endl;

      while( fexists(fileName.c_str()) )
      {
      cout << "Create Sphere from " << fileName << endl;
      double time_start = pcl::getTime();

        frame360 = new Frame360(&calib);
        frame360->loadFrame(fileName);

//        // Save as a keyframe
//        {
//          string fileName_serialize = path_results + mrpt::format("/sphere_images_%d.bin",frame);
//          cout << "write " << fileName_serialize << endl;
//            frame360->serialize(fileName_serialize);
//        }
//        fileName = path_dataset + mrpt::format("/sphere_images_%d.bin", ++frame);
//        continue;

        frame360->undistort();

        if(mode == 1 || mode == 2)
        {
          frame360->stitchSphericalImage();
          cv::imshow( "sphereRGB", frame360->sphereRGB );

          cv::Mat sphDepthVis;
          frame360->sphereDepth.convertTo( sphDepthVis, CV_8U, 25 ); //CV_16UC1
          cv::imshow( "sphereDepth", sphDepthVis );
          cv::waitKey(1);

          if(mode == 2)
          {
            frame360->sphereDepth.convertTo( frame360->sphereDepth, CV_8U, 25 ); //CV_16UC1
//            frame360->sphereDepth.convertTo( frame360->sphereDepth, CV_16U, 1000 ); //CV_16UC1
            cv::imwrite(path_results + mrpt::format("/rgb_%04d.png",frame), frame360->sphereRGB);
            cv::imwrite(path_results + mrpt::format("/depth_%04d.png",frame), frame360->sphereDepth);
            for(unsigned sensor_id=0; sensor_id < 8; sensor_id++)
            {
                cv::imwrite(path_results + mrpt::format("/rgb_%04d_%d.png", frame, sensor_id), frame360->getFrameRGBD_id(sensor_id).getRGBImage());

//                cv::Mat auxDepthVis;
//                frame360->getFrameRGBD_id(sensor_id).getDepthImage().convertTo( auxDepthVis, CV_8U, 0.025 ); //CV_16UC1
                cv::imwrite(path_results + mrpt::format("/depth__%04d_%d.png", frame, sensor_id), frame360->getFrameRGBD_id(sensor_id).getDepthImage());
            }
            sleep(1000000000);
          }
        }
        else if(mode == 3 || mode == 4)
        {
//          frame360->undistort();
          frame360->buildSphereCloud();
          frame360->getPlanes();

          if(mode == 3)
          {
            boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
            delete SphereViewer->frame360;
            SphereViewer->frame360 = frame360;
          }
          else
          {
            frame360->save(path_results, frame);
          }

        }
        else if(mode == 5)
        {
          frame360->buildSphereCloud_fast();
          boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
          delete SphereViewer->frame360;
          SphereViewer->frame360 = frame360;
        }

        if(mode != 3 && mode != 5)
          delete frame360;

        if(frame == SphereViewer->frameIdx)
        {
          frame+=1;
          SphereViewer->frameIdx = frame;
        }
        else
          frame = SphereViewer->frameIdx;

        fileName = path_dataset + mrpt::format("/sphere_images_%d.bin", frame);

        double time_end = pcl::getTime();
        std::cout << "Sphere processed in " << double (time_end - time_start)*1000 << " ms.\n";
      }

      // Free memory
      if(mode == 3)
      {
        boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
        delete SphereViewer;
        delete frame360;
      }

      if(mode == 1 || mode == 2)
      {
        cv::destroyWindow("sphereRGB");
        cv::destroyWindow("sphereDepth");
      }
    }

};


void print_help(char ** argv)
{
  cout << "\nThis program loads a sequence of observations RGBD360.bin and visualizes";
  cout << " and/or saves the spherical images, the pointCloud or the PbMap extracted from it";
  cout << " according to the command options. The keys 'a' and 's' are used to interactively";
  cout << " jump 50 frames backwards and forwards respectively\n\n";
  cout << "  usage: " << argv[0] << " [options] \n";
  cout << "    " << argv[0] << " -h | --help : shows this help" << endl;
  cout << "    " << argv[0] << " <mode> <pathToData> <pathToResultsFolder> \n";
  cout << "          mode = 1 -> Show the reconstructed spherical images \n";
  cout << "          mode = 2 -> Show and save the reconstructed spherical images \n";
  cout << "          mode = 3 -> Show the reconstructed PointCloud and the PbMap \n";
  cout << "          mode = 4 -> Save the reconstructed PointCloud and the PbMap \n";
  cout << "          mode = 5 -> Show a video streaming of the reconstructed PointCloud \n";
}

int main (int argc, char ** argv)
{
  if(argc < 2 || argc > 4 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  int mode = atoi(argv[1]);
  if(mode < 1 || mode > 5)
  {
    print_help(argv);
    return 0;
  }

  string path_dataset = static_cast<string>(argv[2]);
  string path_results;
  if(mode == 2 || mode == 4)
    path_results = static_cast<string>(argv[3]);

  cout << "Create LoadSequence object\n";
  LoadSequence rgbd360_seq;
  rgbd360_seq.run(mode, path_dataset, path_results);

  cout << "EXIT\n";

  return (0);
}
