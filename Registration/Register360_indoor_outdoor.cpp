/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR
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

#include <mrpt/system/os.h> // To use pause()

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

    string fileTypeIndoor360 = ".bin";
    string fileTypeOutdoor360 = ".png";

    ASSERT_( fileTypeIndoor360.compare( file360_1.substr(file360_1.length()-4) ) == 0 || fileTypeOutdoor360.compare( file360_1.substr(file360_1.length()-4) ) == 0 );
    ASSERT_( fileTypeIndoor360.compare( file360_2.substr(file360_2.length()-4) ) == 0 || fileTypeOutdoor360.compare( file360_2.substr(file360_2.length()-4) ) == 0 );
    ASSERT_(fexists(file360_1.c_str()));
    ASSERT_(fexists(file360_2.c_str()));

    // Direct registration
    cout << "Direct registration \n";
    cout << "Dense Registration\n" << endl;
    DirectRegistration dir_reg(DirectRegistration::RGBD360_INDOOR); // Dense RGB-D alignment


    // For the indoor sensor
    Calib360 calib;
    calib.loadExtrinsicCalibration();
    calib.loadIntrinsicCalibration();

    // Load frames
    Frame360 *frame360_1, *frame360_2;
    cout << "Load sphere 1\n";
    if( fileTypeIndoor360.compare( file360_1.substr(file360_1.length()-4) ) == 0 )
    {
        frame360_1 = new Frame360(&calib);
        frame360_1->loadFrame(file360_1);
        frame360_1->undistort();
        frame360_1->stitchSphericalImage();
        //frame360_1->buildPointCloud_rgbd360();
        //frame360_1->segmentPlanes();
    }
    else
    {       
        frame360_1 = new Frame360;
        frame360_1->loadRGB(file360_1);
        string depth_img = file360_1.substr(0, file360_1.length()-14) + "depth" + file360_1.substr(file360_1.length()-11, 7) + ".raw";
        //cout << "depth_img " << depth_img << endl;
        ASSERT_(fexists(depth_img.c_str()));
        frame360_1->loadDepth(depth_img);
        //frame360_1->buildPointCloud();
    }
    cout << "frame360_1->sphereDepth " << frame360_1->sphereRGB.rows << "x" << frame360_1->sphereRGB.cols << " "
                                       << frame360_1->sphereDepth.rows << "x" << frame360_1->sphereDepth.cols << endl;

    cout << "Load sphere 2\n";
    if( fileTypeIndoor360.compare( file360_2.substr(file360_2.length()-4) ) == 0 )
    {
        dir_reg.sensor_type_ref = DirectRegistration::RGBD360_INDOOR;
        frame360_2 = new Frame360(&calib);
        frame360_2->loadFrame(file360_1);
        frame360_2->undistort();
        frame360_2->stitchSphericalImage();
        //frame360_2->buildPointCloud_rgbd360();
        //frame360_2->segmentPlanes();
    }
    else
    {
        dir_reg.sensor_type_ref = DirectRegistration::RGBD360_INDOOR;
        frame360_2 = new Frame360;
        frame360_2->loadRGB(file360_2);
        string depth_img = file360_2.substr(0, file360_2.length()-14) + "depth" + file360_2.substr(file360_2.length()-11, 7) + ".raw";
        //cout << "depth_img " << depth_img << endl;
        ASSERT_(fexists(depth_img.c_str()));
        frame360_2->loadDepth(depth_img);
        //frame360_2->buildPointCloud();
    }
    cout << "frame360_2->sphereDepth " << frame360_2->sphereRGB.rows << "x" << frame360_2->sphereRGB.cols << " " << frame360_2->sphereDepth.rows << "x" << frame360_2->sphereDepth.cols << endl;

    // Visualize images
    {
        cv::Mat sphDepthVis1;
        frame360_1->sphereDepth.convertTo( sphDepthVis1, CV_8U, 10 ); //CV_16UC1
        cv::imshow( "sphereDepth1", sphDepthVis1 );
        cv::imshow( "frame360_1", frame360_1->sphereRGB );

        cv::Mat sphDepthVis2;
        frame360_2->sphereDepth.convertTo( sphDepthVis2, CV_8U, 10 ); //CV_16UC1
        cv::imshow( "sphereDepth2", sphDepthVis2 );
        cv::imshow( "frame360_2", frame360_2->sphereRGB );
        cv::waitKey(0);

        cv::destroyWindow("sphereDepth1");
        cv::destroyWindow("frame360_1");
        cv::destroyWindow("sphereDepth2");
        cv::destroyWindow("frame360_2");
    }

    
//    //  double time_start = pcl::getTime();
//    RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));
//    registerer.RegisterPbMap(&frame360_1, &frame360_2, 25, RegisterRGBD360::PLANAR_3DoF);
//    //  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::DEFAULT_6DoF);
//    //  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::ODOMETRY_6DoF);
    
//    std::map<unsigned, unsigned> bestMatch = registerer.getMatchedPlanes();
//    //  std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << " areaMatched " << registerer.getAreaMatched() << std::endl;
//    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
//        std::cout << it->first << " " << it->second << std::endl;
    
//    Eigen::Matrix4f rigidTransf_pbmap = registerer.getPose();
//    cout << "Distance " << rigidTransf_pbmap.block(0,3,3,1).norm() << endl;
//    cout << "Pose \n" << rigidTransf_pbmap << endl;
    

    //dir_reg.setSensorType(DirectRegistration::RGBD360_INDOOR); // This is use to adapt some features/hacks for each type of image (see the implementation of DirectRegistration::regist for more details)
    dir_reg.setNumPyr(5);
    dir_reg.useSaliency(false);
    // dir_reg.setVisualization(true);
    dir_reg.setGrayVariance(4.f/255);
    dir_reg.setSaliencyThreshodIntensity(0.04f);
    dir_reg.setSaliencyThreshodDepth(0.05f);
    SphericalModel pm_ref, pm_trg;
    dir_reg.setProjectionModel_ref(&pm_ref);
    dir_reg.setProjectionModel_trg(&pm_trg);
    dir_reg.setTargetFrame(frame360_1->sphereRGB, frame360_1->sphereDepth);
    dir_reg.setSourceFrame(frame360_2->sphereRGB, frame360_2->sphereDepth);
    dir_reg.doRegistration(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    //  Eigen::Matrix4f initTransf_dense = rot_offset * poseRegPbMap * rot_offset.inverse();
    //  dir_reg.doRegistration(initTransf_dense, DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    Eigen::Matrix4f rigidTransf_dense = dir_reg.getOptimalPose();
    //Eigen::Matrix4f rigidTransf_dense_ref = rot_offset.inverse() * rigidTransf_dense_ref * rot_offset;
    //cout << "Pose Dense Y Downwards \n" << rigidTransf_dense_ref << endl;
    cout << "Pose Dense \n" << rigidTransf_dense << endl;
    //mrpt::system::pause();
    
//    dir_reg.regist_IC(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//    cout << "Pose Dense IC \n" << dir_reg.getOptimalPose() << endl;

    dir_reg.doRegistration(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    cout << "Pose Dense Photo \n" << dir_reg.getOptimalPose() << endl;
    Eigen::Matrix4f rigidTransf_dense2 = dir_reg.getOptimalPose();

//    dir_reg.regist_IC(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//    cout << "Pose Dense Photo IC \n" << dir_reg.getOptimalPose() << endl;

//    dir_reg.doRegistration(rigidTransf_dense, DirectRegistration::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//    cout << "Pose Dense Depth Init \n" << dir_reg.getOptimalPose() << endl;

    dir_reg.doRegistration(Eigen::Matrix4f::Identity(), DirectRegistration::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    cout << "Pose Dense Depth \n" << dir_reg.getOptimalPose() << endl;

    dir_reg.doRegistration(Eigen::Matrix4f::Identity(), DirectRegistration::DIRECT_ICP); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    cout << "Pose Dense ICP \n" << dir_reg.getOptimalPose() << endl;

    
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
    //  filter.filterVoxel(frame360_1->sphereCloud);
    //  filter.filterVoxel(frame360_2->sphereCloud);
    
    //  icp.setInputSource(frame360_2->sphereCloud);
    //  icp.setInputTarget(frame360_1->sphereCloud);
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
    Map.addKeyframe(frame360_1, pose );
    pose = rigidTransf_dense;//.cast<double>();
    //pose = rigidTransf_dense;
    Map.addKeyframe(frame360_2, pose );
    Map.vOptimizedPoses = Map.vTrajectoryPoses;
    Map.vOptimizedPoses[1] = rigidTransf_dense2;
    Map360_Visualizer Viewer(Map,1);
    
    while (!Viewer.viewer.wasStopped() )
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
#endif
    
    cout << "EXIT\n";
    
    return (0);
}

