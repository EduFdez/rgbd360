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

#include <Calibrator.h>
#include <Frame360.h>
#include <Frame360_Visualizer.h>

#include <pcl/console/parse.h>

#define VISUALIZE_SENSOR_DATA 0
#define SHOW_IMAGES 0

using namespace std;


void print_help(char ** argv)
{
  cout << "\nThis program calibrates the extrinsic parameters of a pair of non-overlapping depth cameras"
        << " by segmenting and matching planar patches from different sensors. The matched planar patches"
        << " are provided with a file from a previous segmentation.\n\n";

  cout << "  usage: " << argv[0] << " <matchedPlanesFile> <pathToRawRGBDImagesDir>\n";
  cout << "    <matchedPlanesFile> is the directory containing the plane correspondences" << endl << endl;
  cout << "    <pathToRawRGBDImagesDir> is the directory containing the data stream as a set of '.bin' files" << endl << endl;
  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}

int main (int argc, char ** argv)
{
  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
    print_help(argv);

  string matchedPlanesFile;
  if(argc > 1)
    matchedPlanesFile = static_cast<string>(argv[1]);

  cout << "Calibrate Pair\n";
  PairCalibrator calibrator;

  // Get the plane correspondences
  calibrator.loadPlaneCorrespondences(matchedPlanesFile);

  Eigen::Matrix4f initOffset = Eigen::Matrix4f::Identity();
//  initOffset(1,1) = initOffset(2,2) = -1;
//  initOffset(1,1) = initOffset(2,2) = cos(45*PI/180);
//  initOffset(1,2) = -sin(45*PI/180);
//  initOffset(2,1) = -initOffset(1,2);
  calibrator.setInitRt(initOffset);
  calibrator.CalibratePair();
  cout <<"Rt_estimated \n" << calibrator.Rt_estimated << endl;

  ofstream calibFile;
  string calibFileName = mrpt::format("%s/Calibration/Rt_Pair1.txt", PROJECT_SOURCE_PATH);
  calibFile.open(calibFileName.c_str());
  if (calibFile.is_open())
  {
    calibFile << calibrator.Rt_estimated;
    calibFile.close();
  }

//  calibrator.CalibrateRotationD();

  cout << "CalibrateRotationManifold \n";
  calibrator.CalibrateRotationManifold();
  cout <<"Rt_estimated \n" << calibrator.Rt_estimated << endl;

  calibFileName = mrpt::format("%s/Calibration/Rt_Pair2.txt", PROJECT_SOURCE_PATH);
  calibFile.open(calibFileName.c_str());
  if (calibFile.is_open())
  {
    calibFile << calibrator.Rt_estimated;
    calibFile.close();
  }

  calibrator.setInitRt(initOffset);
  calibrator.CalibrateRotationManifold();
  calibrator.Rt_estimated.block(0,3,3,1) = calibrator.CalibrateTranslation();
  cout <<"Rt_estimated \n" << calibrator.Rt_estimated << endl;

  calibFileName = mrpt::format("%s/Calibration/Rt_Pair3.txt", PROJECT_SOURCE_PATH);
  calibFile.open(calibFileName.c_str());
  if (calibFile.is_open())
  {
    calibFile << calibrator.Rt_estimated;
    calibFile.close();
  }

//  // Calculate average error with the initial approaximated Pose
//  Eigen::Matrix4f initOffset = Eigen::Matrix4f::Identity();
//  initOffset(1,1) = initOffset(2,2) = -1;
//  float av_rot_error = 0;
////  float av_trans_error = 0;
//  for(unsigned i=0; i < calibrator.correspondences.getRowCount(); i++)
//  {
//    Eigen::Vector3f n_obs_i; n_obs_i << calibrator.correspondences(i,0), calibrator.correspondences(i,1), calibrator.correspondences(i,2);
//    Eigen::Vector3f n_obs_ii; n_obs_ii << calibrator.correspondences(i,4), calibrator.correspondences(i,5), calibrator.correspondences(i,6);
//    av_rot_error += fabs(acos(n_obs_i .dot( initOffset.block(0,0,3,3) * n_obs_ii ) ));
////    av_trans_error += fabs(calibrator.correspondences(i,3) - calibrator.correspondences(i,7) - n_obs_i .dot(translation));
////        params_error += plane_calibrator.correspondences1[i] .dot( plane_calibrator.correspondences2[i] );
//  }
//  av_rot_error /= calibrator.correspondences.getRowCount();
////  av_trans_error /= calibrator.correspondences.getRowCount();
//  std::cout << "Errors " << av_rot_error << std::endl;// << " " << av_trans_error << std::endl;


  // Visualize

//  // Ask the user if he wants to save the calibration matrices
//  string input;
//  cout << "Do you want to save the calibrated extrinsic matricx? (y/n)" << endl;
//  getline(cin, input);
//  if(input == "y" || input == "Y")
//  {
//    ofstream calibFile;
//
//    string calibFileName = mrpt::format("%s/Calibration/Rt_Pair.txt", PROJECT_SOURCE_PATH);
//    calibFile.open(calibFileName.c_str());
//    if (calibFile.is_open())
//    {
//      calibFile << calibrator.Rt_estimated;
//      calibFile.close();
//    }
//    else
//      cout << "Unable to open file " << calibFileName << endl;
//  }

  cout << "EXIT\n";

  return (0);
}
