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

#include <mrpt/system/os.h>

#define VISUALIZE_POINT_CLOUD 1
//#ifndef _DEBUG_MSG
    #define _DEBUG_MSG 1
//#endif

using namespace std;
using namespace Eigen;

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

    double time_start, time_end; // Measure performance of the different methods


    Calib360 calib;
    calib.loadExtrinsicCalibration();
    calib.loadIntrinsicCalibration();

    cout << "Create sphere 1\n";
    Frame360 frame360_1(&calib);
    frame360_1.loadFrame(file360_1);
    frame360_1.undistort();
    frame360_1.buildSphereCloud();
    //  frame360_1.getPlanes();
    //  frame360_1.stitchSphericalImage();

    cout << "Create sphere 2\n";
    Frame360 frame360_2(&calib);
    frame360_2.loadFrame(file360_2);
    frame360_2.undistort();
    frame360_2.buildSphereCloud();
    //  frame360_2.getPlanes();

    time_start = pcl::getTime();
    RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));
    frame360_1.getPlanes();
    frame360_2.getPlanes();
    registerer.RegisterPbMap(&frame360_1, &frame360_2, 25, RegisterRGBD360::PLANAR_3DoF);
    //  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::DEFAULT_6DoF);
    //  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::ODOMETRY_6DoF);
    Eigen::Matrix4f relPosePbMap = registerer.getPose();
    time_end = pcl::getTime();
    std::cout << "PbMap registration took " << double (time_end - time_start) << std::endl;

#if _DEBUG_MSG
    std::map<unsigned, unsigned> bestMatch = registerer.getMatchedPlanes();
    //  std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << " areaMatched " << registerer.getAreaMatched() << std::endl;
    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
        std::cout << it->first << " " << it->second << std::endl;

    cout << "Distance " << relPosePbMap.block(0,3,3,1).norm() << endl;
    cout << "PosePbMap \n" << relPosePbMap << endl;
#endif


//      // ICP alignement
//      double time_start_icp = pcl::getTime();
//      pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;

//      icp.setMaxCorrespondenceDistance (0.3);
//      icp.setMaximumIterations (10);
//      icp.setTransformationEpsilon (1e-6);
//    //  icp.setEuclideanFitnessEpsilon (1);
//      icp.setRANSACOutlierRejectionThreshold (0.1);

//      // ICP
//      // Filter the point clouds and remove nan points
//      FilterPointCloud filter(0.1);
//      filter.filterVoxel(frame360_1.sphereCloud);
//      filter.filterVoxel(frame360_2.sphereCloud);

//      icp.setInputSource(frame360_2.sphereCloud);
//      icp.setInputTarget(frame360_1.sphereCloud);
//      pcl::PointCloud<PointT>::Ptr alignedICP(new pcl::PointCloud<PointT>);
//      Matrix4f initRigidTransf = relPosePbMap;
////      Matrix4f initRigidTransf = Matrix4f::Identity();
//      icp.align(*alignedICP, initRigidTransf);

//      double time_end_icp = pcl::getTime();
//      std::cout << "ICP took " << double (time_end_icp - time_start_icp) << std::endl;

//      std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
//      Matrix4f icpTransformation = icp.getFinalTransformation();
//      cout << "ICP transformation:\n" << icpTransformation << endl << "PbMap-Registration\n" << registerer.getPose() << endl;


    // Test: align one image
    int sensorID = 0;
    const float img_width = frame360_1.frameRGBD_[sensorID].getRGBImage().cols;
    const float img_height = frame360_1.frameRGBD_[sensorID].getRGBImage().rows;
    const float res_factor_VGA = img_width / 640.0;
    const float focal_length = 525 * res_factor_VGA;
    const float inv_fx = 1.f/focal_length;
    const float inv_fy = 1.f/focal_length;
    const float ox = img_width/2 - 0.5;
    const float oy = img_height/2 - 0.5;
    Eigen::Matrix3f camIntrinsicMat; camIntrinsicMat << focal_length, 0, ox, 0, focal_length, oy, 0, 0, 1;
    RegisterPhotoICP alignSingleFrames;
    alignSingleFrames.setCameraMatrix(camIntrinsicMat);

    std::cout << "\n\n SENSOR " << sensorID << std::endl;
    Eigen::Matrix4f sensorID_relPosePbMap = frame360_1.calib->Rt_[sensorID].inverse()*relPosePbMap*frame360_1.calib->Rt_[sensorID];
//    cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << endl;

    time_start = pcl::getTime();
    alignSingleFrames.setTargetFrame(frame360_1.frameRGBD_[sensorID].getRGBImage(), frame360_1.frameRGBD_[sensorID].getDepthImage());
    alignSingleFrames.setSourceFrame(frame360_2.frameRGBD_[sensorID].getRGBImage(), frame360_2.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.setVisualization(true);
    double time_start1 = pcl::getTime();
//    alignSingleFrames.alignFrames(Eigen::Matrix4f::Identity(), RegisterPhotoICP::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / PHOTO_DEPTH / DEPTH_CONSISTENCY
    alignSingleFrames.alignFrames(sensorID_relPosePbMap, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / PHOTO_DEPTH / DEPTH_CONSISTENCY

    Eigen::Matrix4f sensorID_dense_relPose = alignSingleFrames.getOptimalPose();
    time_end = pcl::getTime();
    std::cout << "Dense alignment Single Frame took " << double (time_end - time_start) << std::endl;
//    std::cout << "alignment took " << double (time_end - time_start1) << std::endl;
//    std::cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << "\nsensorID_dense_relPose:\n" << sensorID_dense_relPose << std::endl;

//    // Use saliency
//    time_start = pcl::getTime();
//    alignSingleFrames.useSaliency(true);
//    alignSingleFrames.setTargetFrame(frame360_1.frameRGBD_[sensorID].getRGBImage(), frame360_1.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.setSourceFrame(frame360_2.frameRGBD_[sensorID].getRGBImage(), frame360_2.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.alignFrames(sensorID_relPosePbMap, RegisterPhotoICP::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY
//    time_end = pcl::getTime();
//    std::cout << "Dense alignment Single Frame took " << double (time_end - time_start) << std::endl;
//    std::cout << "saliency:\n" << alignSingleFrames.getOptimalPose() << "\nsensorID_dense_relPose:\n" << sensorID_dense_relPose << std::endl;

    Eigen::Matrix4f dense_relPose = frame360_1.calib->Rt_[sensorID]*sensorID_dense_relPose*frame360_1.calib->Rt_[sensorID].inverse();
    std::cout << "dense_relPose:\n" << dense_relPose << std::endl;

//    // ICP preparation time
//    double time_start_icp = pcl::getTime();
//    pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>);
//    pcl::transformPointCloud(*frame360_2.getFrameRGBD_id(sensorID).getPointCloud(), *transformedCloud, sensorID_dense_relPose);
//    FilterPointCloud filter(0.05);
//    filter.filterVoxel(transformedCloud);
//    double time_end_icp = pcl::getTime();
//    std::cout << "ICP preparation " << (time_end_icp - time_start_icp) << std::endl;

//    std::cout << "\n\n SENSOR " << sensorID << std::endl;
//    sensorID = 7;
//    sensorID_relPosePbMap = frame360_1.calib->Rt_[sensorID].inverse()*relPosePbMap*frame360_1.calib->Rt_[sensorID];
////    Eigen::Matrix4f sensorID_relPosePbMap = frame360_1.calib->Rt_[sensorID]*relPosePbMap.inverse()*frame360_1.calib->Rt_[sensorID].inverse();
//    cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << endl;

//    time_start = pcl::getTime();
//    alignSingleFrames.setTargetFrame(frame360_1.frameRGBD_[sensorID].getRGBImage(), frame360_1.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.setSourceFrame(frame360_2.frameRGBD_[sensorID].getRGBImage(), frame360_2.frameRGBD_[sensorID].getDepthImage());
////    alignSingleFrames.setVisualization(true);
//    alignSingleFrames.alignFrames(sensorID_relPosePbMap, RegisterPhotoICP::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / PHOTO_DEPTH / DEPTH_CONSISTENCY

//    sensorID_dense_relPose = alignSingleFrames.getOptimalPose();
//    time_end = pcl::getTime();
//    std::cout << "Dense alignment Single Frame took " << double (time_end - time_start) << std::endl;
//    std::cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << "\nsensorID_dense_relPose:\n" << sensorID_dense_relPose << std::endl;

//    Eigen::Matrix4f dense_relPose2 = frame360_1.calib->Rt_[sensorID]*sensorID_dense_relPose*frame360_1.calib->Rt_[sensorID].inverse();
//    std::cout << "dense_relPose:\n" << dense_relPose2 << std::endl;



//    // Dense Photo/Depth consistency
//    time_start = pcl::getTime();
//    registerer.RegisterDensePhotoICP(&frame360_1, &frame360_2, relPosePbMap, RegisterPhotoICP::PHOTO_CONSISTENCY, RegisterRGBD360::DEFAULT_6DoF);
////    registerer.RegisterDensePhotoICP_2(&frame360_1, &frame360_2, relPosePbMap, RegisterPhotoICP::PHOTO_CONSISTENCY, RegisterRGBD360::DEFAULT_6DoF);
//    Matrix4f relPoseDense = registerer.getPose();
//    time_end = pcl::getTime();
//    std::cout << "Odometry dense alignment 360 took " << double (time_end - time_start) << std::endl;
//    cout << "DensePhotoICP transformation:\n" << relPoseDense << endl;
////mrpt::system::pause();


    // Dense Photo/Depth consistency spherical
    time_start = pcl::getTime();
    RegisterPhotoICP align360;
    align360.setNumPyr(5);
    frame360_1.stitchSphericalImage();
    frame360_2.stitchSphericalImage();
    align360.useSaliency(false);
    align360.setTargetFrame(frame360_1.sphereRGB, frame360_1.sphereDepth);
    align360.setSourceFrame(frame360_2.sphereRGB, frame360_2.sphereDepth);

//    cv::Mat rgb_transposed, rgb_rotated, depth_transposed, depth_rotated;
//    cv::transpose(frame360_1.sphereRGB, rgb_transposed);
//    cv::flip(rgb_transposed, rgb_rotated, 0);
//    cv::transpose(frame360_1.sphereDepth, depth_transposed);
//    cv::flip(depth_transposed, depth_rotated, 0);

//    cv::imshow("rgb_rotated", rgb_rotated);
//    cv::waitKey(0);

//    align360.setTargetFrame(depth_rotated, depth_transposed);
//    cv::transpose(frame360_2.sphereRGB, rgb_transposed);
//    cv::flip(rgb_transposed, rgb_rotated, 0);
//    cv::transpose(frame360_2.sphereDepth, depth_transposed);
//    cv::flip(depth_transposed, depth_rotated, 0);
//    align360.setSourceFrame(depth_rotated, depth_transposed);

    // The reference of the spherical image and the point Clouds are not the same! I should always use the same coordinate system (TODO)
    float angleOffset = 157.5;
    Matrix4f rotOffset = Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);
    Matrix4f initDenseMatching = rotOffset * dense_relPose * rotOffset.inverse();
//    cout << "initDenseMatching \n" << initDenseMatching << endl;

//    align360.setVisualization(true);
    align360.setGrayVariance(2.f/255);
//    align360.alignFrames360(initDenseMatching, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//    align360.alignFrames360(relPosePbMap, RegisterPhotoICP::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    align360.alignFrames360(Matrix4f::Identity(), RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    Matrix4f relPoseDenseSphere = align360.getOptimalPose();
    Matrix4f relPoseDenseSphere_ref = rotOffset.inverse() * relPoseDenseSphere * rotOffset;
    time_end = pcl::getTime();
    std::cout << "Spherical dense alignment took " << double (time_end - time_start) << std::endl;
//    cout << "relPoseDenseSphere: " << align360.avResidual << "\n" << relPoseDenseSphere << endl;

    angleOffset = 157.5;
    rotOffset = Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);
    cout << "relPoseDenseSphere_ref: " << align360.avResidual << "\n" << relPoseDenseSphere_ref << endl;

    //    cv::imwrite( mrpt::format("/home/edu/rgb.png"), frame360_1.sphereRGB);
    //    cv::imwrite( mrpt::format("/home/edu/depth.png"), frame360_1.sphereDepth);
    //    mrpt::system::pause();


    // Visualize
#if VISUALIZE_POINT_CLOUD
    // It runs PCL viewer in a different thread.
    //    cout << "Superimpose cloud\n";
    Map360 Map;
    Matrix4f pose = Matrix4f::Identity();
//    frame360_1.buildSphereCloud_fromImage();
//    frame360_2.buildSphereCloud_fromImage();
    Map.addKeyframe(&frame360_1, pose );
    Map.addKeyframe(&frame360_2, relPosePbMap );
    Map.vOptimizedPoses = Map.vTrajectoryPoses;
    //  Map.vOptimizedPoses[1] = icpTransformation;
//    Map.vOptimizedPoses[1] = dense_relPose;
//    Map.vOptimizedPoses[1] = relPoseDense;
    Map.vOptimizedPoses[1] = relPoseDenseSphere_ref;

    Map360_Visualizer Viewer(Map, 1);
    while (!Viewer.viewer.wasStopped() )
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
#endif

    cout << "EXIT\n";

    return (0);
}

