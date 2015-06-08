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

#define NUM_ASUS_SENSORS 4

#include <mrpt/poses/CPose3D.h>
//#include <mrpt/utils/CFileGZInputStream.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/pbmap/PbMap.h>

//#include <opencv/cv.h>
//#include <opencv2/features2d.hpp>
//#include <opencv2/line_descriptor.hpp>
//using namespace cv;
//using namespace cv::line_descriptor;

//#include <eigen3/Eigen/Dense>
//#include <eigen3/Eigen/Eigenvalues>
//using namespace Eigen;

#include <Calibrator.h>
#include <Frame360.h>
#include <Frame360_Visualizer.h>

#include <pcl/console/parse.h>

#define VISUALIZE_SENSOR_DATA 0

using namespace std;
using namespace mrpt::obs;
using namespace mrpt::utils;

//void print_help(char ** argv)
//{
//  cout << "\nThis program calibrates the extrinsic parameters of the omnidirectional RGB-D device (RGBD360)"
//        << " by segmenting and matching non-overlapping planar patches from different sensors.\n\n";

//  cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> \n";
//  cout << "    <pathPlaneCorrespondences> is the directory containing the data stream as a set of '.bin' files" << endl << endl;
//  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
//}

//class Calib360_Visualizer
//{
// public:
//  /*! Omnidirectional RGB-D image to be displayed */
//  Frame360 *frame360;

//  /*! Frame Index */
//  unsigned frameIdx;

//  size_t match_l, match_r;

//  /*! Visualizer object */
//  pcl::visualization::CloudViewer viewer;

//  /*! Constructor. It starts the visualization in a separate thread */
//  Calib360_Visualizer(Frame360 *frame, size_t match_l_, size_t match_r_) :
//    frame360(frame),
//    match_l(match_l_),
//    match_r(match_r_),
//    viewer("PbMap"),
//    bColoredPlanes(false)
////    numScreenshot(0)
//  {
//    viewer.runOnVisualizationThread (boost::bind(&Calib360_Visualizer::viz_cb, this, _1), "viz_cb");
//    viewer.registerKeyboardCallback (&Calib360_Visualizer::keyboardEventOccurred, *this);
//    b_update_vis_ = true;
//    b_init_viewer_ = true;
//  }

//  /*! Mutex to syncrhronize eventual changes in frame360 */
//  boost::mutex visualizationMutex;

//  #if RECORD_VIDEO
//    /*! Index of visualizer screenshot. This is used to create videos with the results */
//    int numScreenshot;
//  #endif

// private:
//  /*! Show the PbMap's planes. It is control through keyboard event. */
//  bool bShowPlanes;

//  /*! Show the PbMap's planes filled with different colors. It is control through keyboard event. */
//  bool bColoredPlanes;

//  bool b_update_vis_;

//  bool b_init_viewer_;

//  /*! Visualization callback */
//  void viz_cb(pcl::visualization::PCLVisualizer& viz)
//  {
////    {
////      boost::mutex::scoped_lock updateLock(visualizationMutex);
////
////      std::cout << "viz_cb\n";
////    if (frame360 != NULL)
////      std::cout << frame360->sphereCloud->empty() << "\n";

//    if (frame360 == NULL || frame360->sphereCloud->empty())
//    {
//      boost::this_thread::sleep (boost::posix_time::milliseconds (10));
//      return;
//    }

//    if (b_init_viewer_)
//    {
//      b_init_viewer_ = false;
//      viz.setCameraPosition (
//        0,0,-9,		// Position
//        0,0,1,		// Viewpoint
//        0,-1,0);	// Up

//      viz.setSize(800,600); // Set the window size
//      //viz.setBackgroundColor (1.0, 1.0, 1.0);
//      viz.setBackgroundColor (0.5, 0.5, 0.5);
//      viz.addCoordinateSystem (0.3, "global");

//      std::cout << "b_init_viewer_ \n";
//    }

////    if (!viz.updatePointCloud (frame360->sphereCloud, "sphereCloud"))
////      viz.addPointCloud (frame360->sphereCloud, "sphereCloud");
////      viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sphereCloud");
////      viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0,0.0,0.0, "sphereCloud");

//    // Render the data
//    if (b_update_vis_)
//    {
//      boost::mutex::scoped_lock updateLock(visualizationMutex);

//      viz.removeAllPointClouds();
//      viz.removeAllShapes();

//      if (!viz.updatePointCloud (frame360->sphereCloud, "sphereCloud"))
//        viz.addPointCloud (frame360->sphereCloud, "sphereCloud");

//      char name[1024];

////      sprintf (name, "Frame %u", frameIdx);
//      sprintf (name, "Frame %lu", frame360->sphereCloud->size () );
//      viz.addText (name, 20, 20, "info");

//      {
//        // Draw planes
//        vector<int> showPair(2);
//        showPair[0] = match_l;
//        showPair[1] = match_r;
//        for(size_t j=0; j < 2; j++)
//        {
//          size_t i = showPair[j];
//          mrpt::pbmap::Plane &plane_i = frame360->planes.vPlanes[showPair[j]];
//          sprintf (name, "normal_%u", static_cast<unsigned>(i));
//          pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
//          pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
//          pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
//                              plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
//                              plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
//          viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

//          {
//            sprintf (name, "n%u %s", static_cast<unsigned>(i), plane_i.label.c_str());
////            sprintf (name, "n%u %.1f %.2f", static_cast<unsigned>(i), plane_i.curvature*1000, plane_i.areaHull);
//            viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
//          }

//          sprintf (name, "approx_plane_%02d", int (i));
//          viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);

//          if(bColoredPlanes)
//          {
//            sprintf (name, "plane_%02u", static_cast<unsigned>(i));
//            pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
//            if (!viz.updatePointCloud (frame360->sphereCloud, name))
//                viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
//            viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);
//          }

//        }
//        b_update_vis_ = false;
//      }

//    }

//    viz.spinOnce();
//    boost::this_thread::sleep (boost::posix_time::milliseconds (10));

//    #if RECORD_VIDEO
//      std::string screenshotFile = mrpt::format("im_%04u.png", ++numScreenshot);
//      viz.saveScreenshot (screenshotFile);
//    #endif
//  }

//  /*! Get events from the keyboard */
//  void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
//  {
//    if ( event.keyDown() )
//    {
////      cout << "Key pressed " << event.getKeySym () << endl;
//      if(event.getKeySym() == "k" || event.getKeySym() == "K"){cout << " Press K: Show/hide planes\n";
//        bShowPlanes = !bShowPlanes;
//        b_update_vis_ = true;}
//      else if(event.getKeySym() == "l" || event.getKeySym() == "L"){cout << " Press L: fill/unfill plane colors\n";
//        bColoredPlanes = !bColoredPlanes;
//        b_update_vis_ = true;}
////      else if(event.getKeySym () == "a" || event.getKeySym () == "A"){
////        if(frameIdx <= 50)
////          frameIdx = 0;
////        else
////          frameIdx -= 50;
////        }
////      else if(event.getKeySym () == "s" || event.getKeySym () == "S"){
////          frameIdx += 50;}
////      else
////      {
////        cout << "\n\tASCII key: " << unsigned(event.getKeyCode()) << endl;
//////        string key_input = event.getKeySym();
////      }
//    }
//  }

//};

int main (int argc, char ** argv)
{
    ControlPlanes matches;
}

//int main (int argc, char ** argv)
//{
//    if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
//        print_help(argv);

//    string filename;
//    if(argc > 1)
//        filename = static_cast<string>(argv[1]);

//    const size_t decimation = 1;
//    const size_t skip_frames = 3;

//    //mrpt::poses::CPose3D a(1.0,2.0,3.0,DEG2RAD(10),DEG2RAD(50),DEG2RAD(-30));

//    cout << "Calibrate RGBD360 multisensor\n";
////    Calib360 calib;
////    // Load initial calibration
////    cout << "Load initial calibration\n";
////    mrpt::poses::CPose3D pose[NUM_ASUS_SENSORS];
////    //  pose = mrpt::poses::CPose3D(const double x,const double  y,const double  z,const double  yaw=0, const double  pitch=0, const double roll=0);
////    //pose[0] = mrpt::poses::CPose3D(0.285, 0, 1.015, DEG2RAD(0), DEG2RAD(1.3), DEG2RAD(-90));
////    //mrpt::poses::CPose3D pose_ = mrpt::poses::CPose3D(0,0,0);//, DEG2RAD(0), DEG2RAD(1.3), DEG2RAD(-90));
////    cout << "Load initial calibration AA\n";
////    //setYawPitchRoll
////    cout << "Load initial calibration_\n";
//////    pose[1] = mrpt::poses::CPose3D(0.271, -0.031, 1.015, DEG2RAD(-45), DEG2RAD(0), DEG2RAD(-90));
//////    pose[2] = mrpt::poses::CPose3D(0.271, 0.031, 1.125, DEG2RAD(45), DEG2RAD(2), DEG2RAD(-89));
//////    pose[3] = mrpt::poses::CPose3D(0.24, -0.045, 0.975, DEG2RAD(-90), DEG2RAD(1.5), DEG2RAD(-90));
////    cout << "Load initial calibration ___\n";
////    Eigen::Matrix4f pose_mat;
////    for(size_t sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
////    {
//////        cout << "Get matrix\n";
//////        pose_mat = getPoseEigenMatrix( pose[sensor_id] );
//////        cout << sensor_id << " sensor pose\n" << pose_mat << endl;
//////        calib.setRt_id( sensor_id, pose_mat);
////        pose_mat = Eigen::Matrix4f::Identity();
////        calib.setRt_id( sensor_id, pose_mat );
////    }

//    cout << "Init ControlPlanes \n";
//    ControlPlanes matches;
////    for(unsigned sensor_id1=0; sensor_id1 < NUM_ASUS_SENSORS; sensor_id1++)
////    {
////        matches.mmCorrespondences[sensor_id1] = std::map<unsigned, mrpt::math::CMatrixDouble>();
////        for(unsigned sensor_id2=sensor_id1+1; sensor_id2 < NUM_ASUS_SENSORS; sensor_id2++)
////        {
////            matches.mmCorrespondences[sensor_id1][sensor_id2] = mrpt::math::CMatrixDouble(0, 10);
////        }
////    }

////    cout << "Open CRawlog \n";
////    mrpt::obs::CRawlog dataset;
////    //						Open Rawlog File
////    //==================================================================
////    if (!dataset.loadFromRawLogFile(filename))
////        throw std::runtime_error("\nCouldn't open dataset dataset file for input...");

////    cout << "dataset size " << dataset.size() << "\n";
////    //dataset_count = 0;

////    // Set external images directory:
////    const string imgsPath = CRawlog::detectImagesDirectory(filename);
////    CImage::IMAGES_PATH_BASE = imgsPath;
////    mrpt::obs::CObservationPtr observation;

////    mrpt::obs::CObservation3DRangeScanPtr obsRGBD[NUM_ASUS_SENSORS];  // The RGBD observation
////    bool obs_sensor[NUM_ASUS_SENSORS];
////    obs_sensor[0] = false, obs_sensor[1] = false, obs_sensor[2] = false, obs_sensor[3] = false;
////    //CObservation2DRangeScanPtr laserObs;    // Pointer to the laser observation
////    size_t n_obs = 0, frame = 0;

////    while ( n_obs < dataset.size() )
////    {
////        observation = dataset.getAsObservation(n_obs);
////        ++n_obs;
////        if(!IS_CLASS(observation, CObservation3DRangeScan))
////        {
////            continue;
////        }
////        cout << n_obs << " observation: " << observation->sensorLabel << ". Timestamp " << observation->timestamp << endl;

////        size_t sensor_id;
////        if(observation->sensorLabel == "RGBD_1")
////        {
////            sensor_id = 0;
////        }
////        else if(observation->sensorLabel == "RGBD_2")
////        {
////            sensor_id = 1;
////        }
////        else if(observation->sensorLabel == "RGBD_3")
////        {
////            sensor_id = 2;
////        }
////        else if(observation->sensorLabel == "RGBD_4")
////        {
////            sensor_id = 3;
////        }
////        obs_sensor[sensor_id] = true;

////        obsRGBD[sensor_id] = mrpt::obs::CObservation3DRangeScanPtr(observation);
////        obsRGBD[sensor_id]->load();

////        if(obs_sensor[0] && obs_sensor[1] && obs_sensor[2] && obs_sensor[3])
////        {
////            ++frame;
////            obs_sensor[0] = false, obs_sensor[1] = false, obs_sensor[2] = false, obs_sensor[3] = false;

////            // Apply decimation
////            if( frame < skip_frames )
////                continue;
////            if( frame % decimation != 0)
////                continue;

//            //          CloudRGBD_Ext cloud[NUM_ASUS_SENSORS];
//            //          cloud[sensor_id].setRGBImage( cv::Mat(obsRGBD[sensor_id]->intensityImage.getAs<IplImage>()) );
//            //          cv::Mat depth_mat;
//            //          convertRange_mrpt2cvMat(obsRGBD[sensor_id]->rangeImage, depth_mat);
//            //          cloud[sensor_id].setDepthImage(depth_mat);
//            //          cloud[sensor_id].getPointCloud();


////            // Declarations
////            Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
////            Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
////            //Ptr<ORB>                orb = ORB::create(nFeatures,scaleFactor,nLevels);

////            // Feature detection and description
////            //if(lines)
////            {
////                lsd->detect(imgLeft,linesThird,scale,nOctaves );
////                lbd->compute(imgLeft,linesThird,ldescThird);
////            }

////            Frame360 frame360(&calib);
////            frame360.getLocalPlanes();

////            Frame360_Visualizer sphereViewer(&frame360);
////            while (!sphereViewer.viewer.wasStopped() )
////              boost::this_thread::sleep (boost::posix_time::milliseconds (10));


////            cout << "Merge planes \n";
////            mrpt::pbmap::PbMap &planes = frame360.planes;
////            //planes.vPlanes.clear();
////            vector<unsigned> planesSourceIdx(5, 0);
////            for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
////            {
////                planes.MergeWith(frame360.local_planes_[sensor_id], calib.Rt_[sensor_id]);
////                planesSourceIdx[sensor_id+1] = planesSourceIdx[sensor_id] + frame360.local_planes_[sensor_id].vPlanes.size();
////                //          cout << planesSourceIdx[sensor_id+1] << " ";
////            }
////            //        cout << endl;

////            for(unsigned sensor_id1=0; sensor_id1 < NUM_ASUS_SENSORS; sensor_id1++)
////            {
////                //          matches.mmCorrespondences[sensor_id1] = std::map<unsigned, mrpt::math::CMatrixDouble>();
////                for(unsigned sensor_id2=sensor_id1+1; sensor_id2 < NUM_ASUS_SENSORS; sensor_id2++)
////                    //            if( sensor_id2 - sensor_id1 == 1 || sensor_id2 - sensor_id1 == 7)
////                {
////                    //            cout << " sensor_id1 " << sensor_id1 << " " << frame360.local_planes_[sensor_id1].vPlanes.size() << " sensor_id2 " << sensor_id2 << " " << frame360.local_planes_[sensor_id2].vPlanes.size() << endl;
////                    //              matches.mmCorrespondences[sensor_id1][sensor_id2] = mrpt::math::CMatrixDouble(0, 10);

////                    for(unsigned i=0; i < frame360.local_planes_[sensor_id1].vPlanes.size(); i++)
////                    {
////                        unsigned planesIdx_i = planesSourceIdx[sensor_id1];
////                        for(unsigned j=0; j < frame360.local_planes_[sensor_id2].vPlanes.size(); j++)
////                        {
////                            unsigned planesIdx_j = planesSourceIdx[sensor_id2];

////                            //                if(sensor_id1 == 0 && sensor_id2 == 2 && i == 0 && j == 0)
////                            //                {
////                            //                  cout << "Inliers " << planes.vPlanes[planesIdx_i+i].inliers.size() << " and " << planes.vPlanes[planesIdx_j+j].inliers.size() << endl;
////                            //                  cout << "elongation " << planes.vPlanes[planesIdx_i+i].elongation << " and " << planes.vPlanes[planesIdx_j+j].elongation << endl;
////                            //                  cout << "normal " << planes.vPlanes[planesIdx_i+i].v3normal.transpose() << " and " << planes.vPlanes[planesIdx_j+j].v3normal.transpose() << endl;
////                            //                  cout << "d " << planes.vPlanes[planesIdx_i+i].d << " and " << planes.vPlanes[planesIdx_j+j].d << endl;
////                            //                  cout << "color " << planes.vPlanes[planesIdx_i+i].hasSimilarDominantColor(planes.vPlanes[planesIdx_j+j],0.06) << endl;
////                            //                  cout << "nearby " << planes.vPlanes[planesIdx_i+i].isPlaneNearby(planes.vPlanes[planesIdx_j+j], 0.5) << endl;
////                            //                }

////                            //                cout << "  Check planes " << planesIdx_i+i << " and " << planesIdx_j+j << endl;

////                            if( planes.vPlanes[planesIdx_i+i].inliers.size() > 1000 && planes.vPlanes[planesIdx_j+j].inliers.size() > 1000 &&
////                                    planes.vPlanes[planesIdx_i+i].elongation < 5 && planes.vPlanes[planesIdx_j+j].elongation < 5 &&
////                                    planes.vPlanes[planesIdx_i+i].v3normal .dot (planes.vPlanes[planesIdx_j+j].v3normal) > 0.99 &&
////                                    fabs(planes.vPlanes[planesIdx_i+i].d - planes.vPlanes[planesIdx_j+j].d) < 0.2 )//&&
////                                //                    planes.vPlanes[planesIdx_i+i].hasSimilarDominantColor(planes.vPlanes[planesIdx_j+j],0.06) &&
////                                //                    planes.vPlanes[planesIdx_i+i].isPlaneNearby(planes.vPlanes[planesIdx_j+j], 0.5) )
////                                //                      matches.inliersUpperFringe(planes.vPlanes[planesIdx_i+i], 0.2) > 0.2 &&
////                                //                      matches.inliersLowerFringe(planes.vPlanes[planesIdx_j+j], 0.2) > 0.2 ) // Assign correspondence
////                            {

////                                unsigned prevSize = matches.mmCorrespondences[sensor_id1][sensor_id2].getRowCount();
////                                matches.mmCorrespondences[sensor_id1][sensor_id2].setSize(prevSize+1, matches.mmCorrespondences[sensor_id1][sensor_id2].getColCount());
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 0) = frame360.local_planes_[sensor_id1].vPlanes[i].v3normal[0];
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 1) = frame360.local_planes_[sensor_id1].vPlanes[i].v3normal[1];
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 2) = frame360.local_planes_[sensor_id1].vPlanes[i].v3normal[2];
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 3) = frame360.local_planes_[sensor_id1].vPlanes[i].d;
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 4) = frame360.local_planes_[sensor_id2].vPlanes[j].v3normal[0];
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 5) = frame360.local_planes_[sensor_id2].vPlanes[j].v3normal[1];
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 6) = frame360.local_planes_[sensor_id2].vPlanes[j].v3normal[2];
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 7) = frame360.local_planes_[sensor_id2].vPlanes[j].d;
////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 8) = std::min(frame360.local_planes_[sensor_id1].vPlanes[i].inliers.size(), frame360.local_planes_[sensor_id2].vPlanes[j].inliers.size());

////                                float dist_center1 = 0, dist_center2 = 0;
////                                for(unsigned k=0; k < frame360.local_planes_[sensor_id1].vPlanes[i].inliers.size(); k++)
////                                    dist_center1 += frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] / frame360.frameRGBD_[sensor_id1].getPointCloud()->width + frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] % frame360.frameRGBD_[sensor_id1].getPointCloud()->width;
////                                //                      dist_center1 += (frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] / frame360.sphereCloud->width)*(frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] / frame360.sphereCloud->width) + (frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] % frame360.sphereCloud->width)+(frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] % frame360.sphereCloud->width);
////                                dist_center1 /= frame360.local_planes_[sensor_id1].vPlanes[i].inliers.size();

////                                for(unsigned k=0; k < frame360.local_planes_[sensor_id2].vPlanes[j].inliers.size(); k++)
////                                    dist_center2 += frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] / frame360.frameRGBD_[sensor_id2].getPointCloud()->width + frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] % frame360.frameRGBD_[sensor_id2].getPointCloud()->width;
////                                //                      dist_center2 += (frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] / frame360.sphereCloud->width)*(frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] / frame360.sphereCloud->width) + (frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] % frame360.sphereCloud->width)+(frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] % frame360.sphereCloud->width);
////                                dist_center2 /= frame360.local_planes_[sensor_id2].vPlanes[j].inliers.size();

////                                matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 9) = std::max(dist_center1, dist_center2);
////                                //                  cout << "\t Size " << matches.mmCorrespondences[sensor_id1][sensor_id2].getRowCount() << " x " << matches.mmCorrespondences[sensor_id1][sensor_id2].getColCount() << endl;

////                                if( sensor_id2 - sensor_id1 == 1 ) // Calculate conditioning
////                                {
////                                    //                      updateConditioning(couple_id, correspondences[couple_id].back());
////                                    matches.covariances[sensor_id1] += planes.vPlanes[planesIdx_i+i].v3normal * planes.vPlanes[planesIdx_j+j].v3normal.transpose();
////                                    matches.calcAdjacentConditioning(sensor_id1);
////                                    //                    cout << "Update " << sensor_id1 << endl;

////                                    //                    // For visualization
////                                    //                    plane_corresp[couple_id].push_back(pair<mrpt::pbmap::Plane*, mrpt::pbmap::Plane*>(&planes.vPlanes[planesIdx_i+i], &planes.vPlanes[planes_counter_j+j]));
////                                }
////                                else if(sensor_id2 - sensor_id1 == 3)
////                                {
////                                    //                      updateConditioning(couple_id, correspondences[couple_id].back());
////                                    matches.covariances[sensor_id2] += planes.vPlanes[planesIdx_i+i].v3normal * planes.vPlanes[planesIdx_j+j].v3normal.transpose();
////                                    matches.calcAdjacentConditioning(sensor_id2);
////                                    //                    cout << "Update " << sensor_id2 << endl;
////                                }

////                                Calib360_Visualizer calibViewer(&frame360, planesIdx_i+i, planesIdx_j+j);
////                                while (!calibViewer.viewer.wasStopped() )
////                                  boost::this_thread::sleep (boost::posix_time::milliseconds (10));
////                            }
////                        }
////                    }
////                }
////            }
////        }
////    }

////  // Calibrate the omnidirectional RGBD multisensor
////    Calibrator calibrator;
////    for(size_t sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
////        calibrator.Rt_specs[sensor_id] = calib.getRt_id( sensor_id );

////  //calibrator.loadConstructionSpecs(); // Get the inital extrinsic matrices for the different sensors
////    // Get the plane correspondences
////    //calibrator.matchedPlanes.loadPlaneCorrespondences(filename);
////    //  calibrator.matchedPlanes.loadPlaneCorrespondences(mrpt::format("%s/Calibration/ControlPlanes", PROJECT_SOURCE_PATH));
////    calibrator.matchedPlanes = matches;
//////      calibrator.Calibrate();
////  calibrator.CalibrateRotation(0);
//////  calibrator.CalibrateTranslation(0);

////  #if VISUALIZE_SENSOR_DATA

////  #endif

////  // Ask the user if he wants to save the calibration matrices
////  string input;
////  cout << "Do you want to save the calibrated extrinsic matrices? (y/n)" << endl;
////  getline(cin, input);
////  if(input == "y" || input == "Y")
////  {
////    ofstream calibFile;
////    for(unsigned sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
////    {
////      string calibFileName = mrpt::format("%s/Calibration/Rt_0%u.txt", PROJECT_SOURCE_PATH, sensor_id+1);
////      calibFile.open(calibFileName.c_str());
////      if (calibFile.is_open())
////      {
////        calibFile << calibrator.Rt_estimated[sensor_id];
////        calibFile.close();
////      }
////      else
////        cout << "Unable to open file " << calibFileName << endl;
////    }
////  }

////  cout << "   CalibrateTranslation \n";
////  calibrator.CalibrateTranslation();

//  cout << "EXIT\n";

//  return (0);
//}
