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

#define _DEBUG_MSG 1

#include <RegisterRGBD360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP
#include <pcl/registration/warp_point_rigid.h>

#define VISUALIZE_POINT_CLOUD 1

using namespace std;

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
  cout << "frame360_1.getPlanes()\n";

//  frame360_1.load_PbMap_Cloud(path, id_frame1);

cout << "Create sphere 2\n";

  Frame360 frame360_2(&calib);
  frame360_2.loadFrame(file360_2);
  frame360_2.undistort();
  frame360_2.stitchSphericalImage();
  frame360_2.buildSphereCloud_rgbd360();
  frame360_2.getPlanes();
  cout << "frame360_2.getPlanes()\n";

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

  // Dense registration
  float angle_offset = 90;
  Eigen::Matrix4f rot_offset = Eigen::Matrix4f::Identity(); rot_offset(0,0) = rot_offset(1,1) = cos(angle_offset*PI/180); rot_offset(0,1) = -sin(angle_offset*PI/180); rot_offset(1,0) = -rot_offset(0,1);
  RegisterDense align360; // Dense RGB-D alignment
  align360.setSensorType( RegisterDense::RGBD360_INDOOR); // This is use to adapt some features/hacks for each type of image (see the implementation of RegisterDense::register360 for more details)
  align360.setNumPyr(6);
  align360.useSaliency(false);
// align360.setVisualization(true);
  align360.setGrayVariance(9.f/255);
  align360.setTargetFrame(frame360_1.sphereRGB, frame360_1.sphereDepth);
  align360.setSourceFrame(frame360_2.sphereRGB, frame360_2.sphereDepth);
  align360.register360(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//  Eigen::Matrix4f initTransf_dense = rot_offset * poseRegPbMap * rot_offset.inverse();
//  align360.register360(initTransf_dense, RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  Eigen::Matrix4f rigidTransf_dense_ref = align360.getOptimalPose();
  Eigen::Matrix4f rigidTransf_dense = rot_offset.inverse() * rigidTransf_dense_ref * rot_offset;
//  cout << "Pose Dense X Upwards \n" << rigidTransf_dense_ref << endl;
  cout << "Pose Dense \n" << rigidTransf_dense << endl;

  align360.useSaliency(true);
  align360.setSaliencyThreshodIntensity(0.04f);
  align360.setSaliencyThreshodDepth(0.04f);
  align360.register360(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  cout << "Pose Dense Saliency \n" << rot_offset.inverse() * align360.getOptimalPose() * rot_offset << endl;

  align360.register360(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  cout << "Pose Dense Saliency PHOTO_CONSISTENCY \n" << rot_offset.inverse() * align360.getOptimalPose() * rot_offset << endl;

  align360.register360(Eigen::Matrix4f::Identity(), RegisterDense::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  cout << "Pose Dense Saliency DEPTH_CONSISTENCY \n" << rot_offset.inverse() * align360.getOptimalPose() * rot_offset << endl;

  align360.register360_inv(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  cout << "Pose Dense Inv \n" << rot_offset.inverse() * align360.getOptimalPose() * rot_offset << endl;

//  align360.register360(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//  Eigen::Matrix4f rigidTransf_dense2 = rot_offset.inverse() * align360.getOptimalPose() * rot_offset;
//  cout << "Pose Dense PHOTO_CONSISTENCY \n" << rigidTransf_dense2 << endl;

//  align360.register360(Eigen::Matrix4f::Identity(), RegisterDense::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//  Eigen::Matrix4f rigidTransf_dense3 = rot_offset.inverse() * align360.getOptimalPose() * rot_offset;
//  cout << "Pose Dense DEPTH_CONSISTENCY \n" << rigidTransf_dense3 << endl;

//  align360.register360_depthPyr(Eigen::Matrix4f::Identity(), RegisterDense::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//  Eigen::Matrix4f rigidTransf_depthPyr = rot_offset.inverse() * align360.getOptimalPose() * rot_offset;
//  cout << "Pose Dense depthPyr \n" << rigidTransf_depthPyr << endl;

  align360.setSaliencyThreshodIntensity(0.08f);
  align360.setSaliencyThreshodDepth(0.06f);
  align360.register360_bidirectional(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
  cout << "Pose Dense Bidirectional \n" << rot_offset.inverse() * align360.getOptimalPose() * rot_offset << endl;

//  align360.register360(align360.getOptimalPose(), RegisterDense::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//  Eigen::Matrix4f rigidTransf_dense4 = rot_offset.inverse() * align360.getOptimalPose() * rot_offset;
//  cout << "Pose Dense PHOTO_CONSISTENCY INIT \n" << rigidTransf_dense4 << endl;

//  align360.setBilinearInterp(true);
//  align360.register360(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//  Eigen::Matrix4f rigidTransf_dense_BI = rot_offset.inverse() * align360.getOptimalPose() * rot_offset;
//  cout << "Pose Dense BILINEAR \n" << rigidTransf_dense_BI << endl;

//  mrpt::system::pause();
//  align360.register360_unity(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//  Eigen::Matrix4f rigidTransf_unity = rot_offset.inverse() * align360.getOptimalPose() * rot_offset;
//  std::cout << "Pose Dense unity \n" << rigidTransf_unity << std::endl;

//  // ICP alignement
//  double time_start = pcl::getTime();
////  pcl::IterativeClosestPoint<PointT,PointT> icp;
//  pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
////  pcl::IterativeClosestPointNonLinear<PointT,PointT> icp;

//  icp.setMaxCorrespondenceDistance (0.4);
//  icp.setMaximumIterations (10);
//  icp.setTransformationEpsilon (1e-9);
////  icp.setEuclideanFitnessEpsilon (1);
//  icp.setRANSACOutlierRejectionThreshold (0.1);

////  // Transformation function
////  boost::shared_ptr<pcl::registration::WarpPointRigid3D<PointT, PointT> > warp_fcn(new pcl::registration::WarpPointRigid3D<PointT, PointT>);
////
////  // Create a TransformationEstimationLM object, and set the warp to it
////  boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointT, PointT> > te(new pcl::registration::TransformationEstimationLM<PointT, PointT>);
////  te->setWarpFunction(warp_fcn);
////
////  // Pass the TransformationEstimation objec to the ICP algorithm
////  icp.setTransformationEstimation (te);

//  // ICP
//  // Filter the point clouds and remove nan points
//  FilterPointCloud<PointT> filter(0.1);
//  filter.filterVoxel(frame360_1.sphereCloud);
//  filter.filterVoxel(frame360_2.sphereCloud);

//  icp.setInputSource(frame360_2.sphereCloud);
//  icp.setInputTarget(frame360_1.sphereCloud);
//  pcl::PointCloud<PointT>::Ptr alignedICP(new pcl::PointCloud<PointT>);
////  Eigen::Matrix4d initRigidTransf = registerer.getPose();
//  Eigen::Matrix4f initRigidTransf = Eigen::Matrix4f::Identity();
//  icp.align(*alignedICP, initRigidTransf);

//  double time_end = pcl::getTime();
//  std::cout << "ICP took " << double (time_end - time_start) << std::endl;

//  //std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
//  Eigen::Matrix4f icpTransformation = icp.getFinalTransformation(); //.cast<double>();
//  cout << "ICP transformation:\n" << icpTransformation << endl << "PbMap-Registration\n" << registerer.getPose() << endl;

  // Visualize
  #if VISUALIZE_POINT_CLOUD
  // It runs PCL viewer in a different thread.
//    cout << "Superimpose cloud\n";
  Map360 Map;
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  Map.addKeyframe(&frame360_1, pose );
  pose = rigidTransf_pbmap;//.cast<double>();
  //pose = rigidTransf_dense;
  Map.addKeyframe(&frame360_2, pose );
  Map.vOptimizedPoses = Map.vTrajectoryPoses;
  Map.vOptimizedPoses[1] = rigidTransf_dense;
  Map360_Visualizer Viewer(Map,1);

  while (!Viewer.viewer.wasStopped() )
    boost::this_thread::sleep (boost::posix_time::milliseconds (10));
  #endif

  cout << "EXIT\n";

  return (0);
}

