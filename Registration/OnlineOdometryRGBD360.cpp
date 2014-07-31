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
#include <CloudGrabber.h>

#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/visualization/image_viewer.h>
#include <pcl/io/openni_camera/openni_driver.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cvmat_serialization.h>

#include <FrameRGBD.h>
#include <RGBDGrabber.h>
#include <RGBDGrabberOpenNI_PCL.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Eigen/Core>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define ENABLE_OPENMP_MULTITHREADING 0
#define SAVE_IMAGES 0
#define SHOW_IMAGES 1
#define ILUMINATION_FACTOR 0
#define VISUALIZE_POINT_CLOUD 1

using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class OnlineRegistrationRGBD360
{
  public:
    OnlineRegistrationRGBD360() //:
    {
      bTakeKeyframe = false;
    }

    void run()
    {
      openni_wrapper::OpenNIDriver& driver = openni_wrapper::OpenNIDriver::getInstance();
//      if (driver.getNumberDevices () >= 2)
      {
        // Show a list of connected devices
        cout << "Num of devices connected: " << driver.getNumberDevices() << endl;
        for (unsigned deviceIdx = 0; deviceIdx < driver.getNumberDevices(); ++deviceIdx)
        {
          cout << "Device: " << deviceIdx + 1 << ", vendor: " << driver.getVendorName (deviceIdx) << ", product: " << driver.getProductName (deviceIdx) << endl;
    //        << ", connected: " << driver.getBus (deviceIdx) << " @ " << driver.getAddress (deviceIdx) << ", serial number: \'" << driver.getSerialNumber (deviceIdx) << "\'" << endl;
        }

        Calib360 calib;
        calib.loadIntrinsicCalibration();
        calib.loadExtrinsicCalibration();

        // Acces devices
        pcl::OpenNIGrabber::Mode image_mode = pcl::OpenNIGrabber::OpenNI_QVGA_30Hz;  //pcl::OpenNIGrabber::OpenNI_Default_Mode;

        RGBDGrabber *grabber[8];
//        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
//        {
//          string sensorID = mrpt::format("#%i", sensor_id+1);
//          grabber[sensor_id] = new RGBDGrabberOpenNI_PCL(sensorID, image_mode);
//          grabber[sensor_id]->init();
//        }
        grabber[0] = new RGBDGrabberOpenNI_PCL("#3", image_mode);
        grabber[1] = new RGBDGrabberOpenNI_PCL("#4", image_mode);
        grabber[2] = new RGBDGrabberOpenNI_PCL("#2", image_mode);
        grabber[3] = new RGBDGrabberOpenNI_PCL("#6", image_mode);
        grabber[4] = new RGBDGrabberOpenNI_PCL("#7", image_mode);
        grabber[5] = new RGBDGrabberOpenNI_PCL("#5", image_mode);
        grabber[6] = new RGBDGrabberOpenNI_PCL("#8", image_mode);
        grabber[7] = new RGBDGrabberOpenNI_PCL("#1", image_mode);
        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
          grabber[sensor_id]->init();
        cout << "Grabber initialized\n";

////        // Test saving in pcd
//        Eigen::Matrix3f cameraMatrix;
//        cameraMatrix << 262.5, 0., 1.5950000000000000e+02,
//                        0., 262.5, 1.1950000000000000e+02,
//                        0., 0., 1.;
//        Frame360 frame360_(&calib);
//        grabber1->grab(&frame360_.frameRGBD_[sensor_id]);
//        pcl::io::savePCDFile("/home/eduardo/cloud1.pcd", *frame360_.frameRGBD_[sensor_id].getPointCloud(cameraMatrix) );

        Frame360 frame360(&calib);

        // Initialize visualizer
      #if VISUALIZE_POINT_CLOUD
        frame_360 = &frame360;
        pcl::visualization::CloudViewer viewer("RGBD360");
//      string mouseMsg2D ("Mouse coordinates in image viewer");
//      string keyMsg2D ("Key event for image viewer");
        viewer.runOnVisualizationThread (boost::bind(&OnlineRegistrationRGBD360::viz_cb, this, _1), "viz_cb");
        viewer.registerKeyboardCallback(&OnlineRegistrationRGBD360::keyboardEventOccurred, *this);
      #endif

//        cout << "control1\n";
        mrpt::pbmap::PbMap pbmap1, pbmap2;

        unsigned frame = 0;
        while (cv::waitKey(1)!='\n')
//        while (!viewer.wasStopped() )
        {
          double frame_start = pcl::getTime ();
//        cout << "control2\n";

          // Grab frame
//          Frame360 frame360(&calib);
          for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            grabber[sensor_id]->grab(&frame360.frameRGBD_[sensor_id]);
          frame360.setTimeStamp(mrpt::system::getCurrentTime());
//        cout << "control3\n";

          frame360.stitchSphericalImage();
          cv::imshow( "sphereRGB", frame360.sphereRGB );

//        cout << "control4\n";

//          if(bTakeKeyframe)
          {
            ++frame;
            bTakeKeyframe = false;

            {
            boost::mutex::scoped_lock updateLock(visualizationMutex);

              frame360.buildSphereCloud();

              frame360.planes.vPlanes.clear();
              frame360.getPlanes();

              frame360.mergePlanes();

              cout << frame360.planes.vPlanes.size() << " planes segmented in the sphere\n";

//              if(frame > 0) // Register Frame360
//              {
//                mrpt::pbmap::Subgraph refGraph;
//                refGraph.pPBM = &pbmap1;
//                for(unsigned i=0; i < pbmap1.vPlanes.size(); i++)
//                  refGraph.subgraphPlanesIdx.insert(pbmap1.vPlanes[i].id);
//
//                mrpt::pbmap::Subgraph trgGraph;
//                trgGraph.pPBM = &pbmap2;
//                for(unsigned i=0; i < pbmap2.vPlanes.size(); i++)
//                  trgGraph.subgraphPlanesIdx.insert(pbmap2.vPlanes[i].id);
//
//                cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << endl;
//
//                configLocaliser.load_params("../../../Dropbox/Doctorado/Projects/mrpt/share/mrpt/config_files/pbmap/configLocaliser.ini");
//              cout << "color_threshold " << configLocaliser.color_threshold << endl;
//
//                mrpt::pbmap::SubgraphMatcher matcher;
//                map<unsigned, unsigned> bestMatch = matcher.compareSubgraphs(refGraph, trgGraph);
//
//                cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << endl;
//                for(map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
//                  cout << it->first << " " << it->second << endl;
//
//                // Superimpose model
//                Eigen::Matrix4f rigidTransf;    // Pose of map as from current model
//                Eigen::Matrix4f rigidTransfInv; // Pose of model as from current map
//                mrpt::pbmap::ConsistencyTest fitModel(pbmap1, pbmap2);
//
//                rigidTransf = fitModel.initPose(manualMatch_12_13);
//                rigidTransfInv = mrpt::pbmap::inverse(rigidTransf);
//                cout << "Rigid transformation:\n" << rigidTransf << endl << "Inverse\n" << rigidTransfInv << endl;
//
//                pbmap1.MergeWith(pbmap2,rigidTransfInv);
//              }

              bFreezeFrame = false;
//              mrpt::system::pause();

            updateLock.unlock();
            } // CS_visualize

          #if SAVE_IMAGES
          cout << "   Saving images\n";
            cv::Mat timeStampMatrix;
//          SerializeFrameRGBD frameSerializer;
            {
            std::ofstream ofs(mrpt::format("sphereRGBD_%d.bin",frame).c_str(), std::ios::out | std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
//            uint64_t timeStamp = mrpt::system::getCurrentTime();
            getMatrixNumberRepresentationOf_uint64_t(mrpt::system::getCurrentTime(),timeStampMatrix);
            oa << sphereRGB << sphereDepth << timeStampMatrix;
            ofs.close();
            }
            {
            std::ofstream ofs_images(mrpt::format("sphere_images_%d.bin",frame).c_str(), std::ios::out | std::ios::binary);
            boost::archive::binary_oarchive oa_images(ofs_images);
//            oa_images << frameRGBD1->getRGBImage() << frameRGBD2->getRGBImage();
//            oa_images << frameRGBD1->getDepthImage() << frameRGBD2->getDepthImage();
//            oa_images << frameRGBD3->getDepthImage() << frameRGBD4->getDepthImage();

            for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
              oa_images << frameRGBD_[sensor_id].getRGBImage() << frameRGBD_[sensor_id].getDepthImage();
            oa_images << timeStampMatrix;
//            oa_images << frameRGBD1->getRGBImage() << frameRGBD1->getDepthImage() << frameRGBD2->getRGBImage() << frameRGBD2->getDepthImage()
//                      << frameRGBD3->getRGBImage() << frameRGBD3->getDepthImage() << frameRGBD4->getRGBImage() << frameRGBD4->getDepthImage()
//                      << frameRGBD5->getRGBImage() << frameRGBD5->getDepthImage() << frameRGBD6->getRGBImage() << frameRGBD6->getDepthImage()
//                      << frameRGBD7->getRGBImage() << frameRGBD7->getDepthImage() << frameRGBD8->getRGBImage() << frameRGBD8->getDepthImage() << timeStampMatrix;
            ofs_images.close();
            }
//          sleep(2);
//          cout << " binary file saved... " << std::endl;
//            {
//            std::ifstream ifs(mrpt::format("/home/eduardo/Datasets_RGBD360/Data_RGBD360/raw/sphere_images_%d.bin",frame).c_str(), std::ios::in | std::ios::binary);
//              boost::archive::binary_iarchive binary360(ifs);
//              binary360 >> frameRGBD1->getRGBImage() >> frameRGBD1->getDepthImage() >> frameRGBD2->getRGBImage() >> frameRGBD2->getDepthImage()
//                        >> frameRGBD3->getRGBImage() >> frameRGBD3->getDepthImage() >> frameRGBD4->getRGBImage() >> frameRGBD4->getDepthImage()
//                        >> frameRGBD5->getRGBImage() >> frameRGBD5->getDepthImage() >> frameRGBD6->getRGBImage() >> frameRGBD6->getDepthImage()
//                        >> frameRGBD7->getRGBImage() >> frameRGBD7->getDepthImage() >> frameRGBD8->getRGBImage() >> frameRGBD8->getDepthImage() >> timeStampMatrix;
//            cout << " binary file loaded... " << std::endl;
//            //Close the binary bile
//            ifs.close();
//            }

            imwrite(mrpt::format("/home/eduardo/Datasets_RGBD360/Data_RGBD360/PNG/sphereRGB_%d.png", frame), sphereRGB);
            imwrite(mrpt::format("/home/eduardo/Datasets_RGBD360/Data_RGBD360/PNG/sphereDepth_%d.png", frame), sphereDepth);

          #endif
          }

//          #if VISUALIZE_POINT_CLOUD
//            frame_360 = &frame360;
//            bFreezeFrame = false;
//
//            // Initialize visualizer
//            pcl::visualization::CloudViewer viewer("RGBD360");
//            viewer.runOnVisualizationThread (boost::bind(&OnlineRegistrationRGBD360::viz_cb, this, _1), "viz_cb");
//
//            mrpt::system::pause();
//          #endif

          double frame_end = pcl::getTime ();
          cout << "Grabbing in " << (frame_end - frame_start)*1e3 << " ms\n";

//          boost::this_thread::sleep (boost::posix_time::microseconds(1));
//          boost::this_thread::sleep (boost::posix_time::microseconds(20000));
        }

        cv::destroyWindow("sphereRGB");
//        cv::destroyWindow("sphereDepth");

        // Stop grabbing
        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
          grabber[sensor_id]->stop();
      }
//      else
//        cout << "Less than two devices connected: at least two RGB-D sensors are required to perform extrinsic calibration.\n";

    }

  private:

    boost::mutex visualizationMutex;

    Frame360 *frame_360;

    bool bFreezeFrame;

    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
//    cout << "SphericalSequence::viz_cb(...)\n";
      if (frame_360->sphereCloud->empty() )//|| bFreezeFrame)
      {
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
        return;
      }
//    cout << "   ::viz_cb(...)\n";

      viz.removeAllShapes();
      viz.removeAllPointClouds();

      { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
        boost::mutex::scoped_lock updateLock(visualizationMutex);

        if (!viz.updatePointCloud (frame_360->sphereCloud, "sphereCloud"))
          viz.addPointCloud (frame_360->sphereCloud, "sphereCloud");


        // Draw planes
        char name[1024];

        for(size_t i=0; i < frame_360->planes.vPlanes.size(); i++)
        {
          mrpt::pbmap::Plane &plane_i = frame_360->planes.vPlanes[i];
          sprintf (name, "normal_%u", static_cast<unsigned>(i));
          pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
          pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
          pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
                              plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
                              plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
          viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

          {
            sprintf (name, "n%u", static_cast<unsigned>(i));
//            sprintf (name, "n%u_%u", static_cast<unsigned>(i), static_cast<unsigned>(plane_i.semanticGroup));
            viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
          }

          sprintf (name, "plane_%02u", static_cast<unsigned>(i));
          pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
          viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);

          sprintf (name, "approx_plane_%02d", int (i));
          viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);
        }

        bFreezeFrame = true;

      updateLock.unlock();
      }
    }

    bool bTakeKeyframe;

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
    {
      if ( event.keyDown () )
      {
        if(event.getKeySym () == "k" || event.getKeySym () == "K")
          bTakeKeyframe = true;
      }
    }

};


//void print_help(char ** argv)
//{
//  cout << "\nThis program calibrates the Rt between two RGB-D sensors mounted rigidly (wrt the RGB optical centers)"
//        << "by acquiring observations of a dominant plane from different views\n";
//  cout << "usage: " << argv[0] << " [options] \n";
//  cout << argv[0] << " -h | --help : shows this help" << endl;
//  cout << argv[0] << " -s | --save <pathToCalibrationFile>" << endl;
//}


int main (int argc, char ** argv)
{
//  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
//    print_help(argv);

cout << "Create OnlineRegistrationRGBD360 object\n";
  OnlineRegistrationRGBD360 rgbd360;
  rgbd360.run();

cout << "EXIT\n";
  return (0);
}

