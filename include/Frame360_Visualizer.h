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

#ifndef FRAME360_VISUALIZER_H
#define FRAME360_VISUALIZER_H

#include "Frame360.h"
#include "colours.h"

#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>

/*! This class creates a visualizer to display a Frame360 object. The visualizer runs in a separate thread,
 *  and can be sychronized using the visualizationMutex.
 */
class Frame360_Visualizer
{
 public:
  /*! Omnidirectional RGB-D image to be displayed */
  Frame360 *frame360;

  /*! Frame Index */
  unsigned frameIdx;

  /*! Visualizer object */
  pcl::visualization::CloudViewer viewer;

  /*! Constructor. It starts the visualization in a separate thread */
  Frame360_Visualizer(Frame360 *frame = NULL) :
    frame360(frame),
    viewer("PbMap"),
    bShowPlanes(false),
    bColoredPlanes(false)
//    numScreenshot(0)
  {
    b_update_vis_ = true;
    b_init_viewer_ = true;
    viewer.runOnVisualizationThread (boost::bind(&Frame360_Visualizer::viz_cb, this, _1), "viz_cb");
    viewer.registerKeyboardCallback (&Frame360_Visualizer::keyboardEventOccurred, *this);

    std::cout << "Initialize Frame360_Visualizer \n\n";
    std::cout << "\n  Press 'q' to close the program\n";
  }

  /*! Mutex to syncrhronize eventual changes in frame360 */
  boost::mutex visualizationMutex;

  #if RECORD_VIDEO
    /*! Index of visualizer screenshot. This is used to create videos with the results */
    int numScreenshot;
  #endif

 private:
  /*! Show the PbMap's planes. It is control through keyboard event. */
  bool bShowPlanes;

  /*! Show the PbMap's planes filled with different colors. It is control through keyboard event. */
  bool bColoredPlanes;

  bool b_update_vis_;

  bool b_init_viewer_;

  /*! Visualization callback */
  void viz_cb(pcl::visualization::PCLVisualizer& viz)
  {
//    {
//      boost::mutex::scoped_lock updateLock(visualizationMutex);
//
//      std::cout << "viz_cb\n";
//    if (frame360 != NULL)
//      std::cout << frame360->sphereCloud->empty() << "\n";

    if (frame360 == NULL || frame360->sphereCloud->empty())
    {
      boost::this_thread::sleep (boost::posix_time::milliseconds (10));
      return;
    }

    if (b_init_viewer_)
    {
      b_init_viewer_ = false;
      viz.setCameraPosition (
        0,0,-9,		// Position
        0,0,1,		// Viewpoint
        0,-1,0);	// Up

      viz.setSize(800,600); // Set the window size
      //viz.setBackgroundColor (1.0, 1.0, 1.0);
      viz.setBackgroundColor (0.5, 0.5, 0.5);
      viz.addCoordinateSystem (0.3, "global");   
    }

//    if (!viz.updatePointCloud (frame360->sphereCloud, "sphereCloud"))
//      viz.addPointCloud (frame360->sphereCloud, "sphereCloud");
//      viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sphereCloud");
//      viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0,0.0,0.0, "sphereCloud");

    // Render the data
    if (b_update_vis_)
    {
      boost::mutex::scoped_lock updateLock(visualizationMutex);

      viz.removeAllPointClouds();
      viz.removeAllShapes();

      if (!viz.updatePointCloud (frame360->sphereCloud, "sphereCloud"))
        viz.addPointCloud (frame360->sphereCloud, "sphereCloud");

      char name[1024];

//      sprintf (name, "Frame %u", frameIdx);
      sprintf (name, "Frame %lu", frame360->sphereCloud->size () );
      viz.addText (name, 20, 20, "info");

      if(bShowPlanes)
      {
        // Draw planes
        for(size_t i=0; i < frame360->planes.vPlanes.size(); i++)
        {
          mrpt::pbmap::Plane &plane_i = frame360->planes.vPlanes[i];
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

          if(bColoredPlanes)
          {
            sprintf (name, "plane_%02u", static_cast<unsigned>(i));
            pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
            if (!viz.updatePointCloud (frame360->sphereCloud, name))
                viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
            viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);
          }

        }
        b_update_vis_ = false;
      }

    }

    viz.spinOnce();
    boost::this_thread::sleep (boost::posix_time::milliseconds (10));

    #if RECORD_VIDEO
      std::string screenshotFile = mrpt::format("im_%04u.png", ++numScreenshot);
      viz.saveScreenshot (screenshotFile);
    #endif
  }

  /*! Get events from the keyboard */
  void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
  {
    if ( event.keyDown() )
    {
//      std::cout << "Key pressed " << event.getKeySym () << endl;
      if(event.getKeySym() == "k" || event.getKeySym() == "K"){std::cout << " Press K: Show/hide planes\n";
        bShowPlanes = !bShowPlanes;
        b_update_vis_ = true;}
      else if(event.getKeySym() == "l" || event.getKeySym() == "L"){std::cout << " Press L: fill/unfill plane colors\n";
        bColoredPlanes = !bColoredPlanes;
        b_update_vis_ = true;}
//      else if(event.getKeySym () == "a" || event.getKeySym () == "A"){
//        if(frameIdx <= 50)
//          frameIdx = 0;
//        else
//          frameIdx -= 50;
//        }
//      else if(event.getKeySym () == "s" || event.getKeySym () == "S"){
//          frameIdx += 50;}
//      else
//      {
//        std::cout << "\n\tASCII key: " << unsigned(event.getKeyCode()) << endl;
////        string key_input = event.getKeySym();
//      }
    }
  }

};

#endif
