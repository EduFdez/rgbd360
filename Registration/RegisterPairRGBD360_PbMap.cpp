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

#include <RegisterRGBD360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP
#include <pcl/registration/warp_point_rigid.h>

#define VISUALIZE_POINT_CLOUD 1
#ifndef _DEBUG_MSG
    #define _DEBUG_MSG 1
#endif

using namespace std;

Frame360 *frame1, *frame2;
int vizMode; // 0 -> show the pair of frames and their planes // 1-> show the frame 1 and its planes // 2-> show the frame 2 and its planes

/*! Visualization callback */
void viz_cb(pcl::visualization::PCLVisualizer& viz)
{

  {
    // Render the data
    viz.removeAllPointClouds();
    viz.removeAllShapes();
    viz.setSize(800,600); // Set the window size

    char name[1024];

    if(vizMode == 0 || vizMode == 1)
    {
      if (!viz.updatePointCloud (frame1->sphereCloud, "sphereCloud"))
        viz.addPointCloud (frame1->sphereCloud, "sphereCloud");

      // Draw planes
      for(size_t i=0; i < frame1->planes.vPlanes.size(); i++)
      {
        mrpt::pbmap::Plane &plane_i = planes.vPlanes[i];
        sprintf (name, "normal_%u", static_cast<unsigned>(i));
        pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
        pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
        pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
                            plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
                            plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
        viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

        {
          sprintf (name, "n%u %s", static_cast<unsigned>(i), plane_i.label.c_str());
//            sprintf (name, "n%u %.1f %.2f", static_cast<unsigned>(i), plane_i.curvature*1000, plane_i.areaHull);
          viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
        }

        sprintf (name, "approx_plane_%02d", int (i));
        viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);

        if(true)
        {
          sprintf (name, "plane_%02u", static_cast<unsigned>(i));
          pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
          viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);
        }

      }
    }

    if(vizMode == 0 || vizMode == 1)
    {
      if (!viz.updatePointCloud (frame2->sphereCloud, "sphereCloud"))
        viz.addPointCloud (frame2->sphereCloud, "sphereCloud");

      // Draw planes
      for(size_t i=0; i < frame12>planes.vPlanes.size(); i++)
      {
        mrpt::pbmap::Plane &plane_i = planes.vPlanes[i];
        sprintf (name, "normal_%u", static_cast<unsigned>(i));
        pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
        pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
        pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
                            plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
                            plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
        viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

        {
          sprintf (name, "n%u %s", static_cast<unsigned>(i), plane_i.label.c_str());
//            sprintf (name, "n%u %.1f %.2f", static_cast<unsigned>(i), plane_i.curvature*1000, plane_i.areaHull);
          viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
        }

        sprintf (name, "approx_plane_%02d", int (i));
        viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);

        if(true)
        {
          sprintf (name, "plane_%02u", static_cast<unsigned>(i));
          pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
          viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);
        }

      }
    }

  }
}

/*! Get events from the keyboard */
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
  if ( event.keyDown() )
  {
//      cout << "Key pressed " << event.getKeySym () << endl;
    if(event.getKeySym() == "k" || event.getKeySym() == "K"){
      vizMode = (vizMode+1) % 3;
    }
//    else if(event.getKeySym() == "l" || event.getKeySym() == "L"){cout << " Press L: fill/unfill plane colors\n";
//      bColoredPlanes = !bColoredPlanes;
//      b_update_vis_ = true;}
//    else if(event.getKeySym () == "a" || event.getKeySym () == "A"){
//      if(frameIdx <= 50)
//        frameIdx = 0;
//      else
//        frameIdx -= 50;
//      }
}

void print_help(char ** argv)
{
  cout << "\nThis program loads two raw omnidireactional RGB-D images and aligns them using PbMap-based registration.\n";
  cout << "usage: " << argv[0] << " [options] \n";
  cout << argv[0] << " -h | --help : shows this help" << endl;
  cout << argv[0] << " <frame360_1_1.bin> <frame360_1_2.bin>" << endl;
}

int main (int argc, char ** argv)
{
  if(argc != 3)
    print_help(argv);

  string file360_1 = static_cast<string>(argv[1]);
  string file360_2 = static_cast<string>(argv[2]);

//  string path = static_cast<string>(argv[1]);
//  unsigned id_frame1 = atoi(argv[2]);
//  unsigned id_frame2 = atoi(argv[3]);

  Calib360 calib;
  calib.loadExtrinsicCalibration();
  calib.loadIntrinsicCalibration();

cout << "Create sphere 1\n";
  Frame360 frame360_1(&calib);
  frame360_1.loadFrame(file360_1);
  frame360_1.undistort();
  frame360_1.stitchSphericalImage();
  frame360_1.buildSphereCloud_rgbd360();
  frame360_1.getPlanes();

//  frame360_1.load_PbMap_Cloud(path, id_frame1);

cout << "Create sphere 2\n";

  Frame360 frame360_2(&calib);
  frame360_2.loadFrame(file360_2);
  frame360_2.undistort();
  frame360_2.stitchSphericalImage();
  frame360_2.buildSphereCloud_rgbd360();
  frame360_2.getPlanes();

//  frame360_2.load_PbMap_Cloud(path, id_frame2);

//  double time_start = pcl::getTime();
  RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));
  registerer.RegisterPbMap(&frame360_1, &frame360_2, 25, RegisterRGBD360::PLANAR_3DoF);
//  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::DEFAULT_6DoF);
//  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::ODOMETRY_6DoF);

//#if _DEBUG_MSG
  std::map<unsigned, unsigned> bestMatch = registerer.getMatchedPlanes();
//  std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << " areaMatched " << registerer.getAreaMatched() << std::endl;
  for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
    std::cout << it->first << " " << it->second << std::endl;

  Eigen::Matrix4f rigidTransf_pbmap = registerer.getPose();
  cout << "Distance " << rigidTransf_pbmap.block(0,3,3,1).norm() << endl;
  cout << "Pose \n" << rigidTransf_pbmap << endl;
//#endif


  frame1 = &frame360_1;
  frame2 = &frame360_2;
  vizMode = 0;
  pcl::visualization::CloudViewer viewer("viz");
  viewer.runOnVisualizationThread(viz_cb, "viz_cb");
  viewer.registerKeyboardCallback ( keyboardEventOccurred );
  mrpt::system::pause();

  cout << "EXIT\n";

  return (0);
}

