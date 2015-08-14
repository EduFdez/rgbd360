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

#ifndef MAP360_VISUALIZER_H
#define MAP360_VISUALIZER_H

#define RECORD_VIDEO 0

#include "Map360.h"
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>

/*! This class creates a visualizer to display a Map360 object. The visualization runs in a separate thread.
 */
class Map360_Visualizer
{
public:

    /*! Reference to the map containing a pose-graph of spherical keyframes */
    Map360 &Map;

//    /*! Draw the optimized pose-graph */
//    bool bGraphSLAM;

    /*! Freeze the viewer */
    int opt_graph;

    /*! Freeze the viewer */
    bool bFreezeFrame;

    /*! Set some properties for the viewer in the first run */
    bool bFirstRun;

    /*! Draw the camera's current location */
    bool bDrawCurrentLocation;

    /*! Switch between one of the 4 different visualization modes:
     * 0 (Voxelized point cloud),
     * 1 (Overlapping point clouds),
     * 2 (Main keyframe),
     * 3 (Keyframe topology) */
    int nVizMode;

    /*! The index of the pseudo-nearest SphereKF to the camera's current location */
    int currentSphere;

    /*! The pose of the camera's current location */
    Eigen::Affine3f currentLocation;

    /*! Vector of the Keyframe's centers globally referenced */
    std::vector<pcl::PointXYZ> sphere_centers;

    /*! Global point cloud of the map (each keyframe's point cloud is filtered and merged here) */
    pcl::PointCloud<PointT>::Ptr globalMap;

    //    bool bDrawPlanes;

    /*! Global PbMap integrating all the observed planes */ // TODO: probabilistic matching
    mrpt::pbmap::PbMap gobalPbMap;

    /*! Visualizer object */
    pcl::visualization::CloudViewer viewer;
//    pcl::visualization::PCLVisualizer viewer;

public:

    /*! Index of visualizer screenshot. This is used to create videos with the results */
    int numScreenshot;

    /*! Mutex to syncrhronize eventual changes in the map */
    boost::mutex visualizationMutex;

    /*! Constructor. It starts the visualization in a separate thread */
    Map360_Visualizer(Map360 &map, int viz_mode = 0) :
        Map(map),
        //bGraphSLAM(false),
        opt_graph(0),
        bFreezeFrame(false),
        bFirstRun(true),
        bDrawCurrentLocation(false),
        nVizMode(viz_mode),
        currentSphere(-1),
        globalMap(new pcl::PointCloud<PointT>),
        viewer("Map360"),
        numScreenshot(0)
    {
//        viewer.setFullScreen(true); // ERROR. This only works with PCLVisualizer
        viewer.runOnVisualizationThread (boost::bind(&Map360_Visualizer::viz_cb, this, _1), "viz_cb");
        viewer.registerKeyboardCallback (&Map360_Visualizer::keyboardEventOccurred, *this);
    }

    /*! Visualization callback */
    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
        if(bFirstRun)
        {
            viz.setBackgroundColor (0.3, 0.3, 0.3);
            //viz.setBackgroundColor (1.0, 1.0, 1.0);
            viz.setCameraPosition (
//                5,0,-9,		// Position
//                0,0,1,		// Viewpoint
//                1,0,0 );	// Up
                5,0,-3,		// Position
                0,1,1,		// Viewpoint
                0,-1,0 );	// Up
//            viz.setFullScreen(true);
            bFirstRun = false;
        }

//        std::cout << "Map360_Visualizer::viz_cb(...)\n";
        if (Map.vpSpheres.size() == 0 || bFreezeFrame)
        {
            boost::this_thread::sleep (boost::posix_time::milliseconds (10));
            return;
        }

        //    viz.setFullscreen(true);
        viz.removeAllShapes();
        viz.removeAllCoordinateSystems();
//        viz.removeAllPointClouds();

        {
            boost::mutex::scoped_lock updateLockVisualizer(visualizationMutex);
            boost::mutex::scoped_lock updateLock(Map.mapMutex);

            char name[1024];

            sprintf (name, "Frames %lu. Graph-SLAM %d/%d", Map.vpSpheres.size(), opt_graph, Map.vOptimizedPoses.size());
            viz.addText (name, 20, 20, "params");

            // Draw the current camera
            //viewer.addCoordinateSystem (1.0);
            if(bDrawCurrentLocation)
                viz.addCoordinateSystem(0.2, currentLocation, "camera");

            if(nVizMode == 0)
            {
//            std::cout << "globalMap " << globalMap->size() << std::endl;
              if(globalMap->size() > 0)
              {
                if (!viz.updatePointCloud (globalMap, "globalMap")){
                  viz.addPointCloud (globalMap, "globalMap"); //std::cout << "Add globalMap \n";
                  //viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_1");
                  //viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.4f, 0.0f, "cloud_1");
                }
              }
            }
            else if(nVizMode == 1)
            {
//            std::cout << "Draw pointClouds " << Map.vTrajectoryPoses.size() << std::endl;
                // Draw the overlapping keyframe's point clouds
                for(unsigned i=0; i < Map.vOptimizedPoses[opt_graph].size(); i++)
                {
                    sprintf (name, "cloud%u", i);
                    if (!viz.updatePointCloud (Map.vpSpheres[i]->sphereCloud, name))
                        viz.addPointCloud (Map.vpSpheres[i]->sphereCloud, name);
                    Eigen::Affine3f Rt;
                    Rt.matrix() = Map.vOptimizedPoses[opt_graph][i];//.cast<float>();
                    viz.updatePointCloudPose(name, Rt);
                }
            }
            else if(nVizMode == 2) // Draw the keyframes
            {
                for(unsigned i=0; i< Map.vSelectedKFs.size(); i++)
                {
                    sprintf (name, "KF%u", Map.vSelectedKFs[i]);
                    if (!viz.updatePointCloud (Map.vpSpheres[Map.vSelectedKFs[i]]->sphereCloud, name))
                        viz.addPointCloud (Map.vpSpheres[Map.vSelectedKFs[i]]->sphereCloud, name);
                    Eigen::Affine3f Rt;
//                    Rt.matrix() = Map.vTrajectoryPoses[Map.vSelectedKFs[i]];
                    Rt.matrix() = Map.vOptimizedPoses[opt_graph][Map.vSelectedKFs[i]];
                    viz.updatePointCloudPose(name, Rt);
                }
//                boost::this_thread::sleep (boost::posix_time::milliseconds (10));
            }

            // Draw sphere locations
            assert( Map.vOptimizedPoses[opt_graph].size() == Map.vpSpheres.size() );
            pcl::PointXYZ pt_center;//, pt_center_prev;

//            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > *vPoses;
//            if(bGraphSLAM)
//                (*vPoses) = Map.vOptimizedPoses[opt_graph];
//            assert( (*vPoses).size() == (*vPoses).size() );

//            //if(Map.vsAreas.size() > 1) std::cout << "Draw spheres " << (*vPoses).size() << std::endl;
//            sphere_centers.resize((*vPoses).size());
//            //        for(unsigned i=0; i < Map.vpSpheres.size(); i++)
//            for(unsigned i=0; i < (*vPoses).size(); i++)
//            {
//                pt_center = pcl::PointXYZ((*vPoses)[i](0,3), (*vPoses)[i](1,3), (*vPoses)[i](2,3));
//                sphere_centers[i] = pt_center;
//                sprintf (name, "pose%u", i);
//                viz.addSphere (pt_center, 0.04, ared[Map.vpSpheres[i]->node%10], agrn[Map.vpSpheres[i]->node%10], ablu[Map.vpSpheres[i]->node%10], name);

//                sprintf (name, "%u", i);
//                pt_center.x += 0.05;
//                viz.addText3D (name, pt_center, 0.05, ared[Map.vpSpheres[i]->node%10], agrn[Map.vpSpheres[i]->node%10], ablu[Map.vpSpheres[i]->node%10], name);
//            }

//            // Draw the locations of the selected keyframes
//            for(unsigned i=0; i< Map.vSelectedKFs.size(); i++)
//            {
//                //if(Map.vsAreas.size() > 1) std::cout << " Draw sphere " << i << " " << Map.vSelectedKFs[i] << std::endl;
//                pt_center = pcl::PointXYZ((*vPoses)[Map.vSelectedKFs[i]](0,3), (*vPoses)[Map.vSelectedKFs[i]](1,3), (*vPoses)[Map.vSelectedKFs[i]](2,3));
//                sprintf (name, "poseKF%u", i);
//                viz.addSphere (pt_center, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
//            }

            // Draw trajectory and reference systems
            {
                sphere_centers.resize(Map.vOptimizedPoses[opt_graph].size());
                for(unsigned i=0; i < Map.vOptimizedPoses[opt_graph].size(); i++)
                {
                    pt_center = pcl::PointXYZ(Map.vOptimizedPoses[opt_graph][i](0,3), Map.vOptimizedPoses[opt_graph][i](1,3), Map.vOptimizedPoses[opt_graph][i](2,3));
                    sphere_centers[i] = pt_center;
//                    sprintf (name, "pose%u", i);
//                    if(i != currentSphere)
//                        viz.addSphere (pt_center, 0.04, ared[Map.vpSpheres[i]->node%10], agrn[Map.vpSpheres[i]->node%10], ablu[Map.vpSpheres[i]->node%10], name);
//                    else
//                        viz.addSphere (pt_center, 0.04, ared[(Map.vpSpheres[i]->node+5)%10], agrn[(Map.vpSpheres[i]->node+5)%10], ablu[(Map.vpSpheres[i]->node+5)%10], name);

//                    sprintf (name, "%u", i);
//                    pt_center.x += 0.05;
//                    viz.addText3D (name, pt_center, 0.05, ared[Map.vpSpheres[i]->node%10], agrn[Map.vpSpheres[i]->node%10], ablu[Map.vpSpheres[i]->node%10], name);

                    Eigen::Affine3f pose;
                    pose.matrix() = Map.vOptimizedPoses[opt_graph][i];
                    sprintf (name, "cam%u", i);
                    viz.addCoordinateSystem(0.2, pose, name);

                    // Draw edges Odometry
                    if(i>0)
                    {
                        sprintf (name, "link%u", i);
                        viz.addLine (sphere_centers[i-1], sphere_centers[i], name);
                    }

                }

//                // Draw the locations of the selected keyframes
//                for(unsigned i=0; i< Map.vSelectedKFs.size(); i++)
//                {
//                    pt_center = pcl::PointXYZ(Map.vOptimizedPoses[opt_graph][Map.vSelectedKFs[i]](0,3), Map.vOptimizedPoses[opt_graph][Map.vSelectedKFs[i]](1,3), Map.vOptimizedPoses[opt_graph][Map.vSelectedKFs[i]](2,3));
//                    sprintf (name, "poseKF%u", i);
//                    if(Map.vSelectedKFs[i] != currentSphere)
//                        viz.addSphere (pt_center, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
//                    else
//                        viz.addSphere (pt_center, 0.1, ared[(i+5)%10], agrn[(i+5)%10], ablu[(i+5)%10], name);
//                }
            }

            // Draw edges
            for(std::map<unsigned, std::map<unsigned, std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > > >::iterator it1=Map.mmConnectionKFs.begin();
                it1 != Map.mmConnectionKFs.end(); it1++)
                for(std::map<unsigned, std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > >::iterator it2=it1->second.begin(); it2 != it1->second.end(); it2++)
                {
                    //          std::cout << " Draw links " << it1->first << " " << it2->first << "\n";
                    sprintf (name, "link%u_%u", it1->first, it2->first);
                    viz.addLine (sphere_centers[it1->first], sphere_centers[it2->first], name);
                }

//            bFreezeFrame = true;

#if RECORD_VIDEO
            std::string screenshotFile = mrpt::format("im_%04u.png", ++numScreenshot);
            viz.saveScreenshot(screenshotFile);
#endif

            updateLock.unlock();
            viz.spinOnce ();
//            boost::this_thread::sleep (boost::posix_time::milliseconds (1000));
        }
    }

    /*! Get events from the keyboard */
    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
    {
        if ( event.keyDown () )
        {
            //      std::cout << "Key pressed " << event.getKeySym () << std::endl;
            if(event.getKeySym () == "k" || event.getKeySym () == "K")
                bFreezeFrame = !bFreezeFrame;
            else if(event.getKeySym () == "l" || event.getKeySym () == "L")
            {
                //bGraphSLAM = !bGraphSLAM;
                opt_graph = (opt_graph+1) % Map.vOptimizedPoses.size();
            }
            else if(event.getKeySym () == "n" || event.getKeySym () == "N")
            {
                nVizMode = (nVizMode+1) % 4;
                std::cout << "Visualizatio swap to mode " << nVizMode << std::endl;
            }
        }
    }

};

#endif
