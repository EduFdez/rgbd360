/*
 *  Copyright (c) 2013, Universidad de Málaga  - Grupo MAPIR
 *                      INRIA Sophia Antipolis - LAGADIC Team
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
 * Author: Eduardo Fernandez-Moral
 */

#include <Calibrator.h>
#include <Frame360.h>
#include <Frame360_Visualizer.h>

#include <pcl/console/parse.h>

#define VISUALIZE_SENSOR_DATA 0
#define SHOW_IMAGES 0

using namespace std;


void print_help(char ** argv)
{
  cout << "\nThis program calibrates the extrinsic parameters of the omnidirectional RGB-D device (RGBD360)"
        << " by segmenting and matching non-overlapping planar patches from different sensors.\n\n";

  cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> \n";
//  cout << "    <pathToRawRGBDImagesDir> is the directory containing the data stream as a set of '.bin' files" << endl << endl;
  cout << "    <pathPlaneCorrespondences> is the directory containing the data stream as a set of '.bin' files" << endl << endl;
  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}

int main (int argc, char ** argv)
{
  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
    print_help(argv);

  string path_dataset;
  if(argc > 1)
    path_dataset = static_cast<string>(argv[1]);

  cout << "Calibrate RGBD360 multisensor\n";
  Calibrator calibrator;

  // Get the plane correspondences
  calibrator.matchedPlanes.loadPlaneCorrespondences(path_dataset);
//  calibrator.matchedPlanes.loadPlaneCorrespondences(mrpt::format("%s/Calibration/ControlPlanes", PROJECT_SOURCE_PATH));

  // Calibrate the omnidirectional RGBD multisensor
  calibrator.loadConstructionSpecs(); // Get the inital extrinsic matrices for the different sensors
//      calibrator.Calibrate();
  calibrator.CalibrateRotation(0);
//  calibrator.CalibrateTranslation(0);

  #if VISUALIZE_SENSOR_DATA
//    bool showNewCalibration = true;

    Calib360 calib_prev;
    calib_prev.loadIntrinsicCalibration();
    calib_prev.loadExtrinsicCalibration();
//    calib_prev.loadExtrinsicCalibration(mrpt::format("%s/Calibration/Extrinsics", PROJECT_SOURCE_PATH));

    Calib360 calib_new;
    calib_new.loadIntrinsicCalibration();
    for(unsigned sensor_id=0; sensor_id<NUM_ASUS_SENSORS; sensor_id++)
      calib_new.setRt_id(sensor_id, calibrator.Rt_estimated[sensor_id]);

//      RGBDGrabber *grabber[8];

    // Create one instance of RGBDGrabber_OpenNI2 for each sensor
    grabber[0] = new RGBDGrabber_OpenNI2("1d27/0601@5/2", 1);
    grabber[1] = new RGBDGrabber_OpenNI2("1d27/0601@4/2", 1);
    grabber[2] = new RGBDGrabber_OpenNI2("1d27/0601@3/2", 1);
    grabber[3] = new RGBDGrabber_OpenNI2("1d27/0601@6/2", 1);
    grabber[4] = new RGBDGrabber_OpenNI2("1d27/0601@9/2", 1);
    grabber[5] = new RGBDGrabber_OpenNI2("1d27/0601@8/2", 1);
    grabber[6] = new RGBDGrabber_OpenNI2("1d27/0601@7/2", 1);
    grabber[7] = new RGBDGrabber_OpenNI2("1d27/0601@10/2", 1);

    // Initialize the sensor
    for(int sensor_id = 0; sensor_id < 8; sensor_id++)
//      #pragma omp parallel num_threads(8)
    {
//        int sensor_id = omp_get_thread_num();
      grabber[sensor_id]->init();
    }
    cout << "Grabber initialized\n";

    Calib360 *currentCalib = &calib_new;

    // Get the first frame
    Frame360 *frame360 = new Frame360(currentCalib);

    // Skip the first frames
    unsigned skipNFrames = 2;
    for(unsigned skipFrames = 0; skipFrames < skipNFrames; skipFrames++ )
      for(int sensor_id = 0; sensor_id < 8; sensor_id++)
        grabber[sensor_id]->grab(&frame360->frameRGBD_[sensor_id]);

//        Frame360_Visualizer sphereViewer(&frame360);

    char keyEvent = 'a';
    while (keyEvent!='\n')
//    while (!viewer.wasStopped() )
    {
      double frame_start = pcl::getTime ();
      keyEvent = cv::waitKey(1);

      if(keyEvent == 'p')
      {
        cout << "Use previous calibration\n";
        currentCalib = &calib_prev;
      }
      else if(keyEvent == 'p')
      {
        cout << "Use NEW calibration\n";
        currentCalib = &calib_new;
      }

      Frame360 *frame360 = new Frame360(currentCalib);

      for(int sensor_id = 0; sensor_id < 8; sensor_id++)
        grabber[sensor_id]->grab(&(frame360->frameRGBD_[sensor_id]));

//      cout << "grab new frame\n";

    #if SHOW_IMAGES
      frame360->stitchSphericalImage();
      cv::imshow( "sphereRGB", frame360->sphereRGB );
//            cv::imshow( "sphereDepth", sphereDepth );
//        imwrite(mrpt::format("/home/efernand/Datasets_RGBD360/Data_RGBD360/PNG/sphereRGB_%d.png", frame), sphereRGB);
//        imwrite(mrpt::format("/home/efernand/Datasets_RGBD360/Data_RGBD360/PNG/sphereDepth_%d.png", frame), sphereDepth);
    #endif

      double frame_end = pcl::getTime ();
      cout << "Grabbing in " << (frame_end - frame_start)*1e3 << " ms\n";

//          boost::this_thread::sleep (boost::posix_time::microseconds(5000));
    }

    cv::destroyWindow("sphereRGB");
//        cv::destroyWindow("sphereDepth");

    for(unsigned sensor_id = 0; sensor_id < 8; sensor_id++)
      grabber[sensor_id]->stop();
  #endif

  // Ask the user if he wants to save the calibration matrices
  string input;
  cout << "Do you want to save the calibrated extrinsic matrices? (y/n)" << endl;
  getline(cin, input);
  if(input == "y" || input == "Y")
  {
    ofstream calibFile;
    for(unsigned sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
    {
      string calibFileName = mrpt::format("%s/Calibration/Rt_0%u.txt", PROJECT_SOURCE_PATH, sensor_id+1);
      calibFile.open(calibFileName.c_str());
      if (calibFile.is_open())
      {
        calibFile << calibrator.Rt_estimated[sensor_id];
        calibFile.close();
      }
      else
        cout << "Unable to open file " << calibFileName << endl;
    }
  }

  cout << "   CalibrateTranslation \n";
  calibrator.CalibrateTranslation();

  cout << "EXIT\n";

  return (0);
}
