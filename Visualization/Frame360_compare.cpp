/*
 *  Copyright (c) 2015,   INRIA Sophia Antipolis - LAGADIC Team
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

#include <Frame360CompareVisualizer.h>
#include <Miscellaneous.h>

#include <pcl/filters/passthrough.h>
#include <pcl/console/parse.h>

using namespace std;

void print_help(char ** argv)
{
  cout << "\nThis program loads a SphericalStereo image, builds the pointCloud and creates a PbMap from it, showing the result in a 3D viewer. \n\n";
  cout << " A raw depth image file and a RGB files should be given as input. In the case of a single input of a depth image,";
  cout << " the RGB image which is expected to be named 'topXXX.png' will be searched in the same directory, where XXX is the image index as in the Depth file name.";
  cout << " The user can play with the view using the keys 'k' and 'l', which are used to switch between visualization modes (i.e. show/hide PbMap).\n\n";
  cout << "  usage: " <<  argv[0] << " <DepthImg.raw> <optionalRGBImg.png> \n\n";
}

/*! This program loads a Frame360 from an omnidirectional RGB-D image (in raw binary format), creates a PbMap from it,
 *  and displays both. The keys 'k' and 'l' are used to switch visualization between PointCloud representation or PointCloud+PbMap.
 */
int main (int argc, char ** argv)
{
  if((argc != 2 && argc != 3) || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  string fileDepth = static_cast<string>(argv[1]);
  string fileDepth_fused;
  string fileRGB;
  string fileType = ".raw";
  if( !fileType.compare( fileDepth.substr(fileDepth.length()-4) ) == 0 ) // Check that the depth image fileName is correct and that the file exists
  {
      std::cerr << "\n... INVALID IMAGE FILE!!! \n";
      return 0;
  }

  if(argc == 3) // Take the input RGB image file
  {
    fileRGB = static_cast<string>(argv[2]);
    string RGB_type = ".png";
    if( !RGB_type.compare( fileRGB.substr(fileRGB.length()-4) ) == 0 ) // Check that the depth image fileName is correct and that the file exists
    {
        std::cerr << "\n... INVALID IMAGE FILE!!! \n";
        std::cerr << "\t " << fileRGB << " \n";
        return 0;
    }
  }
  else // Search the corresponding RGB image file in the same directory
  {
      string extRenato = "pT";
      string depthType;
      if( extRenato.compare( fileDepth.substr(fileDepth.length()-6, 2) ) == 0 )
      {
          depthType = "pT.raw";
          fileRGB = fileDepth.substr(0, fileDepth.length()-18) + "top" + fileDepth.substr(fileDepth.length()-13, 7) + ".png";
      }
      else
      {
          depthType = ".raw";
          fileRGB = fileDepth.substr(0, fileDepth.length()-16) + "top" + fileDepth.substr(fileDepth.length()-11, 7) + ".png";
      }
      fileDepth_fused = fileDepth.substr(0, fileDepth.length()-16) + "gapFillingPlusFusion/" + fileDepth.substr(fileDepth.length()-16);
  }


  std::cout << "  fileDepth: " << fileDepth << "\n  fileRGB: " << fileRGB << std::endl;
  if ( !fexists(fileDepth.c_str()) || !fexists(fileDepth_fused.c_str()) || !fexists(fileRGB.c_str()) )
  {
      std::cerr << "\n... Input images do not exist!!! \n Check that these files are correct: " << fileDepth << "\n " << fileDepth_fused << "\n " << fileRGB << "\n ";
      return 0;
  }

  std::cout << "Load frames raw and fused "<< std::endl;

  cv::Mat maskCar = cv::imread("/Data/Shared_Lagadic/useful_code/maskCar_.png",0);

  Frame360 frame360;
  frame360.loadDepth(fileDepth, &maskCar);
  //cv::namedWindow( "sphereDepth", WINDOW_AUTOSIZE );// Create a window for display.
  cv::Mat sphDepthVis;
  frame360.sphereDepth.convertTo( sphDepthVis, CV_8U, 10 ); //CV_16UC1
  std::cout << "  Show depthImage " << fileRGB << std::endl;
  cv::imshow( "sphereDepth", sphDepthVis );
  cv::waitKey(0);
  cv::destroyWindow("sphereDepth");

  frame360.loadRGB(fileRGB);
  //cv::namedWindow( "sphereRGB", WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "sphereRGB", frame360.sphereRGB );
  cv::waitKey(0);
  cv::destroyWindow("sphereRGB");

  frame360.buildSphereCloud();
  frame360.filterCloudBilateral_stereo();
  frame360.segmentPlanesStereo();
//  frame360.segmentPlanesStereoRANSAC();
  std::cout << "Planes " << frame360.planes.vPlanes.size () << std::endl;

  size_t plane_inliers = 0;
  for(size_t i=0; i < frame360.planes.vPlanes.size (); i++)
  {
      plane_inliers += frame360.planes.vPlanes[i].inliers.size();
      //std::cout << plane_inliers << " Plane inliers " << frame360.planes.vPlanes[i].inliers.size() << std::endl;
  }
  if (frame360.planes.vPlanes.size () > 0)
    std::cout << "Plane inliers " << plane_inliers << " average plane size " << plane_inliers/frame360.planes.vPlanes.size () << std::endl;

  Frame360 frame360_fusion;
  frame360_fusion.loadDepth(fileDepth_fused, &maskCar);
  frame360_fusion.loadRGB(fileRGB);
  frame360_fusion.buildSphereCloud();
  frame360_fusion.filterCloudBilateral_stereo();
  frame360_fusion.segmentPlanesStereo();
//  frame360_fusion.segmentPlanesStereoRANSAC();
  std::cout << "FUSION Planes " << frame360_fusion.planes.vPlanes.size () << std::endl;

  plane_inliers = 0;
  for(size_t i=0; i < frame360_fusion.planes.vPlanes.size (); i++)
  {
      plane_inliers += frame360_fusion.planes.vPlanes[i].inliers.size();
      //std::cout << plane_inliers << " Plane inliers " << frame360_fusion.planes.vPlanes[i].inliers.size() << std::endl;
  }
  if (frame360_fusion.planes.vPlanes.size () > 0)
    std::cout << "FUSION Plane inliers " << plane_inliers << " average plane size " << plane_inliers/frame360_fusion.planes.vPlanes.size () << std::endl;


  /*! Filter points in the direction x */
  pcl::PassThrough<PointT> filter_pass_x;
  filter_pass_x.setFilterFieldName ("x"); // Vertical axis of the spherical point clouds in RGBD360 (it points towards the ceiling)
  filter_pass_x.setFilterLimits (-0.0, 20.0);
  //typename pcl::PointCloud<PointT>::Ptr filteredCloud(new pcl::PointCloud<PointT>);
  filter_pass_x.setInputCloud ( frame360.getSphereCloud() );
  filter_pass_x.filter ( *frame360.getSphereCloud() );
  filter_pass_x.setInputCloud ( frame360_fusion.getSphereCloud() );
  filter_pass_x.filter ( *frame360_fusion.getSphereCloud() );


  // Visualize point cloud
  Frame360CompareVisualizer sphereCompareViewer(&frame360, &frame360_fusion);
  cout << "\n  Press 'q' to close the program\n";
//   pcl::io::savePCDFile ("/home/efernand/cloud_test.pcd", *frame360.sphereCloud);

  while (!sphereCompareViewer.viewer.wasStopped() )
    boost::this_thread::sleep (boost::posix_time::milliseconds (10));

  // Simple visualizer with viewport (only point clouds)
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  viewer->addCoordinateSystem (1,0);
//  viewer->initCameraParameters ();

//  int v1(0);
//  viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
//  viewer->setBackgroundColor (0, 0, 0, v1);
//  viewer->addText ("VO_Ent", 15, 15, "v1 text", v1);
//  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb (frame360.sphereCloud);
//  viewer->addPointCloud<pcl::PointXYZRGBA> (frame360.sphereCloud, rgb, "sample cloud1", v1);

//  int v2(0);
//  viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
//  viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
//  viewer->addText ("Unc_fusion", 10, 10, "v2 text", v2);
//   pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb2 (frame360_fusion..sphereCloud);
//  viewer->addPointCloud<pcl::PointXYZRGBA> (frame360_fusion..sphereCloud, rgb2, "sample cloud2", v2);

//  while (!viewer->wasStopped ())
//  {
//     viewer->spinOnce (100);
//     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//  }


  cout << "EXIT\n\n";

  return (0);
}

