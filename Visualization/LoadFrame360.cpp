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

#include <Frame360_Visualizer.h>

#include <pcl/console/parse.h>

using namespace std;

void print_help(char ** argv)
{
  cout << "\nThis program loads a Frame360.bin (an omnidirectional RGB-D image in raw binary format).";
  cout << " It builds the pointCloud and creates a PbMap from it. The spherical frame is shown: the";
  cout << " keys 'k' and 'l' are used to switch between visualization modes.\n\n";
  cout << "  usage: " <<  argv[0] << " <pathToFrame360.bin> \n\n";
}

/*! This program loads a Frame360 from an omnidirectional RGB-D image (in raw binary format), creates a PbMap from it,
 *  and displays both. The keys 'k' and 'l' are used to switch visualization between PointCloud representation or PointCloud+PbMap.
 */
int main (int argc, char ** argv)
{
  if(argc != 2 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  string fileName = static_cast<string>(argv[1]);

  Calib360 calib;
  calib.loadIntrinsicCalibration();
  calib.loadExtrinsicCalibration();
//std::cout << "LoadFrame360: " << fileName << std::endl;

  Frame360 frame360(&calib);
  frame360.loadFrame(fileName);
  frame360.undistort();
  frame360.buildSphereCloud_rgbd360();
  frame360.getPlanes();

//  frame360.stitchSphericalImage();
//  cv::imwrite("rgb_test.png", frame360.sphereRGB);
//  cv::imwrite("depth_test.png", frame360.sphereDepth);

//  // Visualize spherical image
//  frame360.fastStitchImage360();
//  cv::imshow( "sphereRGB", frame360.sphereRGB );
////  cv::imshow( "sphereRGB", frame360.frameRGBD_[0].getRGBImage() );
//  while (cv::waitKey(1)!='\n')
//    boost::this_thread::sleep (boost::posix_time::milliseconds (10));

  // Visualize point cloud
  Frame360_Visualizer sphereViewer(&frame360);
  cout << "\n  Press 'q' to close the program\n";

  while (!sphereViewer.viewer.wasStopped() )
    boost::this_thread::sleep (boost::posix_time::milliseconds (10));

  cout << "EXIT\n\n";

  return (0);
}

