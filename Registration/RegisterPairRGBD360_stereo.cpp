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
 *  Author: efernand Fernandez-Moral
 */

#include <Frame360.h>
#include <Frame360_Visualizer.h>
#include <pcl/console/parse.h>

#include <RegisterRGBD360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP
#include <pcl/registration/warp_point_rigid.h>

#include <iostream>
#include <string>
#include <assert.h>

using namespace std;

void print_help(char ** argv)
{
  cout << "\nThis program loads two raw omnidireactional RGB-D images and aligns them using PbMap-based registration.\n";
  cout << "usage: " << argv[0] << " [options] \n";
  cout << argv[0] << " -h | --help : shows this help" << endl;
  cout << argv[0] << " <frame360_1.png> <frame360_2.png>" << endl;
}


/*! This program loads a Frame360 from an omnidirectional RGB-D image (in raw binary format), creates a PbMap from it,
 *  and displays both. The keys 'k' and 'l' are used to switch visualization between PointCloud representation or PointCloud+PbMap.
 */
int main (int argc, char ** argv)
{
  if(argc != 3 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  double time_start, time_end; // Measure timings of the different processes

  cv::Mat maskCar = cv::imread("/Data/Shared_Lagadic/useful_code/maskCar_.png",0);
//  cv::imshow( "maskCar", maskCar );
//  cv::waitKey(0);

  string rgb1 = static_cast<string>(argv[1]);
  string rgb2 = static_cast<string>(argv[2]);
  std::cout << "  rgb1: " << rgb1 << "\n  rgb2: " << rgb2 << std::endl;

  string fileType = ".png";
  string depth1, depth2;
  //std::cout << "  end: " << rgb1.substr(rgb1.length()-4) << std::endl;

  if( fileType.compare( rgb1.substr(rgb1.length()-4) ) == 0 && fileType.compare( rgb2.substr(rgb2.length()-4) ) == 0 ) // If the first string correspond to a pointCloud path
  {
//    depth1 = rgb1.substr(0, rgb1.length()-14) + "depth" + rgb1.substr(rgb1.length()-11, 7) + ".raw";
//    depth2 = rgb2.substr(0, rgb2.length()-14) + "depth" + rgb2.substr(rgb2.length()-11, 7) + ".raw";
//    depth1 = rgb1.substr(0, rgb1.length()-14) + "depth" + rgb1.substr(rgb1.length()-11, 7) + "pT.raw";
//    depth2 = rgb2.substr(0, rgb2.length()-14) + "depth" + rgb2.substr(rgb2.length()-11, 7) + "pT.raw";
    depth1 = rgb1.substr(0, rgb1.length()-14) + "gapFillingPlusFusion/depth" + rgb1.substr(rgb1.length()-11, 7) + ".raw";
    depth2 = rgb2.substr(0, rgb2.length()-14) + "gapFillingPlusFusion/depth" + rgb2.substr(rgb2.length()-11, 7) + ".raw";

    std::cout << "  depth1: " << depth1 << "\n  depth2: " << depth2 << std::endl;
  }
  else
  {
      std::cerr << "\n... INVALID IMAGE FILE!!! \n";
      return 0;
  }

  Frame360 frame360_1, frame360_2;
  frame360_1.loadDepth(depth1, &maskCar);
//  frame360_1.loadDepth(depth1);
//  //cv::namedWindow( "sphereDepth", WINDOW_AUTOSIZE );// Create a window for display.
//  cv::Mat sphDepthVis;
//  frame360_1.sphereDepth.convertTo( sphDepthVis, CV_8U, 10 ); //CV_16UC1
//  std::cout << "  Show depthImage " << fileRGB << std::endl;

//  cv::imshow( "sphereDepth", sphDepthVis );
//  //cv::waitKey(1);
//  cv::waitKey(0);
  frame360_1.loadRGB(rgb1);
//  //cv::namedWindow( "sphereRGB", WINDOW_AUTOSIZE );// Create a window for display.
//  cv::imshow( "sphereRGB", frame360_1.sphereRGB );
//  cv::waitKey(0);

  frame360_1.buildSphereCloud();
//  frame360_1.filterCloudBilateral_stereo();
//  frame360_1.segmentPlanesStereo();

cout << "frame360_1 " << frame360_1.sphereCloud->width << " " << frame360_1.sphereCloud->height << " " << frame360_1.sphereCloud->is_dense << " " << endl;
//cout << "frame360_1 filtered " << frame360_1.filteredCloud->width << " " << frame360_1.filteredCloud->height << " " << frame360_1.filteredCloud->is_dense << " " << endl;


//  size_t plane_inliers = 0;
//  for(size_t i=0; i < frame360_1.planes.vPlanes.size (); i++)
//  {
//      plane_inliers += frame360_1.planes.vPlanes[i].inliers.size();
//      //std::cout << plane_inliers << " Plane inliers " << frame360_1.planes.vPlanes[i].inliers.size() << std::endl;
//  }
//  std::cout << "Plane inliers " << plane_inliers << " average plane size " << plane_inliers/frame360_1.planes.vPlanes.size () << std::endl;

//  frame360_2.loadDepth(depth2);
  frame360_2.loadDepth(depth2, &maskCar);
  frame360_2.loadRGB(rgb2);
  frame360_2.buildSphereCloud();
//  frame360_2.filterCloudBilateral_stereo();
//  frame360_2.segmentPlanesStereo();


//  RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));
//  registerer.RegisterPbMap(&frame360_1, &frame360_2, 25, RegisterRGBD360::PLANAR_3DoF);
////  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::DEFAULT_6DoF);
////  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::ODOMETRY_6DoF);
//  Eigen::Matrix4f poseRegPbMap = registerer.getPose();

////#if _DEBUG_MSG
//  std::map<unsigned, unsigned> bestMatch = registerer.getMatchedPlanes();
////  std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << " areaMatched " << registerer.getAreaMatched() << std::endl;
//  for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
//    std::cout << it->first << " " << it->second << std::endl;

////  std::cout << "Distance " << registerer.getPose().block(0,3,3,1).norm() << std::endl;
////  std::cout << "Pose \n" << registerer.getPose() << std::endl;
////#endif

  // Dense registration
//  float angleOffset = 157.5;
//  Eigen::Matrix4f rotOffset = Eigen::Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);
  RegisterDense align360; // Dense RGB-D alignment
  align360.setSensorType( RegisterDense::STEREO_OUTDOOR); // This is use to adapt some features/hacks for each type of image (see the implementation of RegisterDense::alignFrames360 for more details)
  align360.setNumPyr(6);
  align360.setMinDepth(1.f);
  align360.setMaxDepth(15.f);
  align360.useSaliency(false);
  //align360.setVisualization(true);
  align360.setGrayVariance(8.f/255);
  align360.setTargetFrame(frame360_1.sphereRGB, frame360_1.sphereDepth);
  align360.setSourceFrame(frame360_2.sphereRGB, frame360_2.sphereDepth);
  cout << "RegisterDense \n";
  time_start = pcl::getTime();
//  for(size_t i=0; i < 20; i++)
  align360.alignFrames360(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  time_end = pcl::getTime();
  std::cout << "alignFrames360 took " << double (time_end - time_start) << std::endl;
//  Eigen::Matrix4f initTransf_dense = rotOffset * poseRegPbMap * rotOffset.inverse();
//  align360.alignFrames360(initTransf_dense, RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  Eigen::Matrix4f rigidTransf_dense_ref = align360.getOptimalPose();
//  Eigen::Matrix4f rigidTransf_dense = rotOffset.inverse() * rigidTransf_dense_ref * rotOffset;
  cout << "Pose Dense \n" << rigidTransf_dense_ref << endl;
//  cout << "Pose Dense2 \n" << rigidTransf_dense << endl;

  time_start = pcl::getTime();
//  for(size_t i=0; i < 20; i++)
  align360.alignFrames360_inv(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  time_end = pcl::getTime();
  std::cout << "alignFrames360 took " << double (time_end - time_start) << std::endl;
  cout << "Pose Dense Inv \n" << align360.getOptimalPose() << endl;

  time_start = pcl::getTime();
//  for(size_t i=0; i < 20; i++)
  align360.alignFrames360_bidirectional(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  time_end = pcl::getTime();
  std::cout << "alignFrames360 took " << double (time_end - time_start) << std::endl;
  cout << "Pose Dense Bidirectional \n" <<  align360.getOptimalPose() << endl;


////  time_start = pcl::getTime();
//  align360.setSourceFrame(frame360_2.sphereRGB, frame360_2.sphereDepth);
//  cout << "RegisterDense UNITY \n";
//  align360.alignFrames360_unity(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
////  time_end = pcl::getTime();
////  std::cout << "alignFrames360_unity took " << double (time_end - time_start) << std::endl;
//  std::cout << "Pose Dense unity \n" << align360.getOptimalPose() << std::endl;

  align360.alignFrames360(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  std::cout << "Pose PHOTO_CONSISTENCY \n" << align360.getOptimalPose() << std::endl;

  align360.alignFrames360(Eigen::Matrix4f::Identity(), RegisterDense::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  std::cout << "Pose DEPTH_CONSISTENCY \n" << align360.getOptimalPose() << std::endl;

  // ICP alignement
  time_start = pcl::getTime();
//  pcl::IterativeClosestPoint<PointT,PointT> icp;
  pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
//  pcl::IterativeClosestPointNonLinear<PointT,PointT> icp;

  icp.setMaxCorrespondenceDistance (0.4);
  icp.setMaximumIterations (10);
  icp.setTransformationEpsilon (1e-9);
//  icp.setEuclideanFitnessEpsilon (1);
  icp.setRANSACOutlierRejectionThreshold (0.1);

//  // Transformation function
//  boost::shared_ptr<pcl::registration::WarpPointRigid3D<PointT, PointT> > warp_fcn(new pcl::registration::WarpPointRigid3D<PointT, PointT>);
//
//  // Create a TransformationEstimationLM object, and set the warp to it
//  boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointT, PointT> > te(new pcl::registration::TransformationEstimationLM<PointT, PointT>);
//  te->setWarpFunction(warp_fcn);
//  pcl::registration::TransformationEstimation<PointT,PointT>::Ptr te(new pcl::registration::TransformationEstimationPointToPlaneLLS<PointT,PointT>);

//  // Pass the TransformationEstimation objec to the ICP algorithm
//  icp.setTransformationEstimation (te);

//  // Filter the point clouds and remove nan points (needed for ICP)
//  filterCloudBilateral_stereo<PointT> filter(0.1);
//  filter.filterVoxel(frame360_1.sphereCloud);
//  filter.filterVoxel(frame360_2.sphereCloud);
//  icp.setInputSource(frame360_2.sphereCloud);
//  icp.setInputTarget(frame360_1.sphereCloud);
  pcl::VoxelGrid<PointT> filter_voxel;
  filter_voxel.setLeafSize(0.1,0.1,0.1);
  pcl::PointCloud<PointT>::Ptr sphereCloud_dense1(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr sphereCloud_dense2(new pcl::PointCloud<PointT>);
//  pcl::copyPointCloud()
  filter_voxel.setInputCloud (frame360_1.sphereCloud);
  filter_voxel.filter (*sphereCloud_dense1);
  filter_voxel.setInputCloud (frame360_2.sphereCloud);
  filter_voxel.filter (*sphereCloud_dense2);
  cout << "frame360_1 " << frame360_1.sphereCloud->width << " " << frame360_1.sphereCloud->height << " " << frame360_1.sphereCloud->is_dense << " " << endl;
  cout << "voxel filtered frame360_1 " << sphereCloud_dense1->width << " " << sphereCloud_dense1->height << " " << sphereCloud_dense1->is_dense << " " << endl;

  icp.setInputSource(sphereCloud_dense2);
  icp.setInputTarget(sphereCloud_dense1);
  pcl::PointCloud<PointT>::Ptr alignedICP(new pcl::PointCloud<PointT>);
//  Eigen::Matrix4d initRigidTransf = registerer.getPose();
  Eigen::Matrix4f initRigidTransf = Eigen::Matrix4f::Identity();
  icp.align(*alignedICP, initRigidTransf);

  time_end = pcl::getTime();
  std::cout << "ICP took " << double (time_end - time_start) << std::endl;

  //std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
  Eigen::Matrix4f icpTransformation = icp.getFinalTransformation(); //.cast<double>();
  std::cout << "ICP transformation:\n" << icpTransformation << endl << std::endl;


//  // ICP point-to-plane
//  time_start = pcl::getTime();
//  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
//  ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);   //      ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);  //      ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);  //      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
//  ne.setMaxDepthChangeFactor (0.1); // For VGA: 0.02f, 10.01
//  ne.setNormalSmoothingSize (10.0f);
//  ne.setDepthDependentSmoothing (true);
//  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_source (new pcl::PointCloud<pcl::Normal>);
//  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_target (new pcl::PointCloud<pcl::Normal>);
//  ne.setInputCloud ( frame360_1.sphereCloud );
//  ne.compute (*normal_cloud_target);
//  ne.setInputCloud ( frame360_2.sphereCloud );
//  ne.compute (*normal_cloud_source);

//  pcl::GeneralizedIterativeClosestPoint<pcl::PointNormal,pcl::PointNormal> icp_p2P;
////    pcl::IterativeClosestPointWithNormals<pcl::PointNormal,pcl::PointNormal> icp_p2P;
//    icp_p2P.setMaxCorrespondenceDistance (0.4);
//    icp_p2P.setMaximumIterations (10);
//    icp_p2P.setTransformationEpsilon (1e-9);
//    icp_p2P.setRANSACOutlierRejectionThreshold (0.1);
//    pcl::registration::TransformationEstimation<pcl::PointNormal,pcl::PointNormal>::Ptr te(new pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal,pcl::PointNormal>);
//    icp_p2P.setTransformationEstimation (te); // Pass the TransformationEstimation objec to the ICP algorithm
//    pcl::PointCloud<pcl::PointNormal>::Ptr alignedICP_p2P(new pcl::PointCloud<pcl::PointNormal>);
//    icp_p2P.align(*alignedICP_p2P, initRigidTransf);
//    time_end = pcl::getTime();
//    std::cout << "ICP Point-to-plane took " << double (time_end - time_start) << std::endl;
//    std::cout << "ICP transformation Point-to-plane:\n" << icp_p2P.getFinalTransformation() << std::endl;


  // Visualize
//  #if VISUALIZE_POINT_CLOUD
  // It runs PCL viewer in a different thread.
//    cout << "Superimpose cloud\n";
  Map360 Map;
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  Map.addKeyframe(&frame360_1, pose );
//  pose = registerer.getPose();//.cast<double>();
//  pose = align360.getOptimalPose();
  pose = rigidTransf_dense_ref;
  Map.addKeyframe(&frame360_2, pose );
  Map.vOptimizedPoses = Map.vTrajectoryPoses;
  Map.vOptimizedPoses[1] = icpTransformation;
  Map360_Visualizer Viewer(Map, 1);
  *Viewer.globalMap += *frame360_1.sphereCloud;
  *Viewer.globalMap += *frame360_2.sphereCloud;
//  Viewer.viz_cb(Viewer.viewer);

  while (!Viewer.viewer.wasStopped() )
  {
//    Viewer.viewer.spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::milliseconds (100) );
  }
//  #endif


  cout << "EXIT\n";

  return (0);
}


//#include <iostream>

//#include <pcl/point_types.h>
//#include <pcl/visualization/pcl_visualizer.h>

//typedef pcl::PointXYZRGBA PointT;
//typedef pcl::PointCloud<PointT> PointCloudT;

//int
//main (int argc, char* argv[])
//{
//        PointCloudT::Ptr cloud_1 (new PointCloudT),
//                                                cloud_2	(new PointCloudT);

//        cloud_1->points.resize (300);
//        cloud_2->points.resize (300);

//        // First random point cloud
//        for (size_t i = 0; i < cloud_1->points.size (); ++i) {
//                cloud_1->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
//                cloud_1->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
//                cloud_1->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);

//                cloud_1->points[i].r = 0.0;
//                cloud_1->points[i].g = 255 *(1024 * rand () / (RAND_MAX + 1.0f));
//                cloud_1->points[i].b = 255 *(1024 * rand () / (RAND_MAX + 1.0f));
//        }

//        // Second random point cloud
//        for (size_t i = 0; i < cloud_2->points.size (); ++i) {
//                cloud_2->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
//                cloud_2->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
//                cloud_2->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);

//                cloud_2->points[i].r = 255 *(1024 * rand () / (RAND_MAX + 1.0f));
//                cloud_2->points[i].g = 0.0;
//                cloud_2->points[i].b = 255 *(1024 * rand () / (RAND_MAX + 1.0f));
//        }

//        // Viewer with 2 vertical viewports
//        pcl::visualization::PCLVisualizer viewer ("Visualizer");
////        int v1(0); int v2(1);
////        viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
////        viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);

//        // Add point clouds
//        viewer.addPointCloud (cloud_1, "cloud_1");
////        viewer.addPointCloud (cloud_1, "cloud_1", v1);
////        viewer.addPointCloud (cloud_2, "cloud_2", v2);

////        viewer.resetCamera ();

//        // Display the visualiser
//        while (!viewer.wasStopped ()) {
//                viewer.spinOnce ();
//        }

//        return (0);
//}
