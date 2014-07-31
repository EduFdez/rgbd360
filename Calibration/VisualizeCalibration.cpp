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

#include <Frame360.h>

#include <pcl/console/parse.h>

#define VISUALIZE_POINT_CLOUD 1
#define MAX_MATCH_PLANES 25

using namespace std;

void print_help(char ** argv)
{
  cout << "\nVisualize the calibrated images from the RGBD360 multisensor "
       << "(so that the non-overlapping content is ngligible)\n";

  cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> <path_extrinsicCalib1> <path_extrinsicCalib2> <path_extrinsicCalib3> " << endl;
  cout << "    <pathToRawRGBDImagesDir> is the directory containing the data stream as a set of '.bin' files" << endl;
  cout << "    <path_extrinsicCalib1> is the directory containing the extrinsic model 1" << endl;
  cout << "    <path_extrinsicCalib2> is the directory containing the extrinsic model 2" << endl;
  cout << "    <path_extrinsicCalib3> is the directory containing the extrinsic model 3" << endl;

  cout << "\n         " << argv[0] << " -h | --help : shows this help" << endl;
}

int main (int argc, char ** argv)
{
  if(argc != 5 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  string path_dataset = static_cast<string>(argv[1]);
  string path_extrinsicCalib1 = static_cast<string>(argv[2]);
  string path_extrinsicCalib2 = static_cast<string>(argv[3]);
  string path_extrinsicCalib3 = static_cast<string>(argv[4]);

  int sampleDataset = 50;

  cout << "Visualize the calibrated images from the RGBD360 multisensor\n";


  Calib360 calib1, calib2, calib3;
  calib1.loadExtrinsicCalibration(path_extrinsicCalib1);
  calib2.loadExtrinsicCalibration(path_extrinsicCalib2);
  calib3.loadExtrinsicCalibration(path_extrinsicCalib3);

  unsigned frame = 1;
  string fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
//  cout << "fileName " << fileName << " " << fexists(fileName.c_str()) << endl;

  while( fexists(fileName.c_str()) )
  {
  cout << "Create Sphere from " << fileName << endl;
  double time_start = pcl::getTime();

    Frame360 frame360_calib1(&calib1);
    Frame360 frame360_calib2(&calib2);
    Frame360 frame360_calib3(&calib3);

    frame360_calib1.loadFrame(fileName);
    frame360_calib2.loadFrame(fileName);
    frame360_calib3.loadFrame(fileName);

    frame360_calib1.stitchSphericalImage();
    frame360_calib2.stitchSphericalImage();
    frame360_calib3.stitchSphericalImage();

    cv::Mat sphereCalibs(3*frame360_calib1.sphereRGB.rows, frame360_calib1.sphereRGB.cols, CV_8UC3);
    cv::Mat tmp1 = sphereCalibs(cv::Rect(0, 0, frame360_calib1.sphereRGB.cols, frame360_calib1.sphereRGB.rows));
    cv::Mat tmp2 = sphereCalibs(cv::Rect(0, frame360_calib1.sphereRGB.rows, frame360_calib1.sphereRGB.cols, frame360_calib1.sphereRGB.rows));
    cv::Mat tmp3 = sphereCalibs(cv::Rect(0, 2*frame360_calib1.sphereRGB.rows, frame360_calib1.sphereRGB.cols, frame360_calib1.sphereRGB.rows));
    frame360_calib1.sphereRGB.copyTo(tmp1);
    frame360_calib2.sphereRGB.copyTo(tmp2);
    frame360_calib3.sphereRGB.copyTo(tmp3);

//    cv::imshow( "sphereRGB", frame360_calib1.sphereRGB );
    cv::imshow( "sphereRGB", sphereCalibs );
    cv::waitKey(0);

    frame += sampleDataset;
    fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
  }

  cout << "EXIT\n";

  return (0);
}
