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

#define RECORD_VIDEO 0

#include <Map360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>
#include <RegisterRGBD360.h>

//#include <pcl/registration/icp.h>
//#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP

#include <pcl/console/parse.h>

#define MAX_MATCH_PLANES 25
#define SAVE_TRAJECTORY 0
#define SAVE_IMAGES 0
#define VISUALIZE_POINT_CLOUD 1

using namespace std;

/*! This class' main function 'run' performs PbMap-based odometry with the input data of a stream of
 *  omnidirectional RGB-D images (from a recorded sequence).
 */
class Odometry360
{
private:
    Map360 Map;

    RegisterRGBD360 registerer;

    Calib360 calib;

    Frame360 *frame360_1, *frame360_2;

    RegisterPhotoICP align360; // Dense RGB-D alignment

    pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;

public:
    Odometry360() :
        registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH))
    {
        // Get the calibration matrices for the different sensors
        calib.loadExtrinsicCalibration();
        calib.loadIntrinsicCalibration();

        align360.setNumPyr(6);
        align360.useSaliency(false);
      // align360.setVisualization(true);
        align360.setGrayVariance(3.f/255);

        icp.setMaxCorrespondenceDistance (0.4);
        icp.setMaximumIterations (10);
        icp.setTransformationEpsilon (1e-9);
        icp.setRANSACOutlierRejectionThreshold (0.1);
    }

    /*! Run the odometry from the image specified by "imgRGB_1st".
     */
    void run(string &imgRGB_1st, string &path_results, const int &selectSample)
    {
        unsigned frame = 275;
        unsigned frameOrder = frame+1;

        cv::Mat maskCar = cv::imread("/Data/Shared_Lagadic/useful_code/maskCar_.png",0);
      //  cv::imshow( "maskCar", maskCar );
      //  cv::waitKey(0);

        string rgb1 = static_cast<string>(argv[1]);
        string rgb2 = static_cast<string>(argv[2]);
        std::cout << "  rgb1: " << rgb1 << "\n  rgb2: " << rgb2 << std::endl;

        string fileType = ".png";
        string depth1, depth2;
        std::cout << "  end: " << rgb1.substr(rgb1.length()-4) << std::endl;

        if( fileType.compare( rgb1.substr(rgb1.length()-4) ) == 0 && fileType.compare( rgb2.substr(rgb2.length()-4) ) == 0 ) // If the first string correspond to a pointCloud path
        {
          depth1 = rgb1.substr(0, rgb1.length()-14) + "depth" + rgb1.substr(rgb1.length()-11, 7) + "pT.raw";
          depth2 = rgb2.substr(0, rgb2.length()-14) + "depth" + rgb2.substr(rgb2.length()-11, 7) + "pT.raw";
          std::cout << "  depth1: " << depth1 << "\n  depth2: " << depth2 << std::endl;
        }
        else
            assert(0);



        string fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
        cout << "Frame " << fileName << endl;

        frame360_2 = new Frame360(&calib);
        frame360_2->loadFrame(fileName);
        frame360_2->undistort();
        frame360_2->stitchSphericalImage();
        frame360_2->buildSphereCloud();
        frame360_2->getPlanes();
        //    cout << "regsitrationCloud has " << registrationClouds[1]->size() << " Pts\n";

//        RegisterPhotoICP align360; // Dense RGB-D alignment
//        align360.setNumPyr(5);
//        align360.useSaliency(false);
////        align360.setVisualization(true);
//        align360.setGrayVariance(3.f/255);

//        // ICP alignement
//        pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
//        icp.setMaxCorrespondenceDistance (0.3);
//        icp.setMaximumIterations (10);
//        icp.setTransformationEpsilon (1e-6);
//      //  icp.setEuclideanFitnessEpsilon (1);
//        icp.setRANSACOutlierRejectionThreshold (0.1);


        bool bGoodRegistration = true;
        Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();


        // Filter the spherical point cloud
        // ICP -> Filter the point clouds and remove nan points
        FilterPointCloud<PointT> filter(0.05); // Initialize filters (for visualization)
        filter.filterEuclidean(frame360_2->sphereCloud);
        filter.filterVoxel(frame360_2->sphereCloud);

        // Visualize
#if VISUALIZE_POINT_CLOUD
        Map360_Visualizer viewer(Map);
        viewer.bDrawCurrentLocation = true;

        // Add the first observation to the map
        {boost::mutex::scoped_lock updateLock(Map.mapMutex);

            Map.addKeyframe(frame360_2, currentPose);
            *viewer.globalMap += *frame360_2->sphereCloud;
        }

#endif

        frame += selectSample;
        fileName = mrpt::format("%s/sphere_images_%d.bin", path_dataset.c_str(), frame);
        //      fileName = mrpt::format("%s/sphere_images_%d.bin", path_dataset.c_str(), ++frame);
        Eigen::Matrix4f rigidTransf = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f rigidTransf_pbmap = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f rigidTransf_dense = Eigen::Matrix4f::Identity();

        // The reference of the spherical image and the point Clouds are not the same! I should always use the same coordinate system (TODO)
        float angleOffset = 157.5;
        Eigen::Matrix4f rotOffset = Eigen::Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);

        while( fexists(fileName.c_str()) )
        {
            cout << "Frame " << fileName << endl;

            if(bGoodRegistration)
            {
                frame360_1 = frame360_2;
            }

            // Load pointCLoud
            frame360_2 = new Frame360(&calib);
            frame360_2->loadFrame(fileName);
            frame360_2->undistort();
            frame360_2->stitchSphericalImage();
            frame360_2->buildSphereCloud();
            frame360_2->getPlanes();

#if SAVE_IMAGES
            frame360_2->stitchSphericalImage();
            cv::imwrite(path_results + mrpt::format("/rgb_%03d.png",frameOrder), frame360_2->sphereRGB);
            frame360_2->sphereDepth.convertTo( frame360_2->sphereDepth, CV_16U, 1000 ); //CV_16UC1
            cv::imwrite(path_results + mrpt::format("/depth_%03d.png",frameOrder), frame360_2->sphereDepth);
#endif

            // Align the two frames
            bGoodRegistration = registerer.RegisterPbMap(frame360_1, frame360_2, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
            //        bGoodRegistration = registerer.RegisterPbMap(frame360_1, frame360_2, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_ODOMETRY_3DoF);


            align360.setTargetFrame(frame1.sphereRGB, frame1.sphereDepth);
            align360.setSourceFrame(frame2.sphereRGB, frame2.sphereDepth);
            cout << "RegisterPhotoICP \n";
            align360.alignFrames360(Eigen::Matrix4f::Identity(), RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
          //  Eigen::Matrix4f initTransf_dense = rotOffset * poseRegPbMap * rotOffset.inverse();
          //  align360.alignFrames360(initTransf_dense, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            Eigen::Matrix4f rigidTransf_dense_ref = align360.getOptimalPose();

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

            double time_end = pcl::getTime();
            std::cout << "ICP took " << double (time_end - time_start) << std::endl;

            //std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
            Eigen::Matrix4f icpTransformation = icp.getFinalTransformation(); //.cast<double>();


            if(!bGoodRegistration)
            {
                cout << "\tBad registration\n\n";
                cout << "\nPbMap regist \n" << registerer.getPose() << endl;
//                fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",++frame);
//                bGoodRegistration = false;
//                delete frame360_2;
//                continue;
            }
            else
            {
                rigidTransf_pbmap = registerer.getPose();
                cout << "\nPbMap regist \n" << rigidTransf_pbmap << endl;
            }

//            double turn_x = -asin(rigidTransf(2,0));
//            if(turn_x > 25*PI/180)
            {
                cout << "Align Sphere " << endl;
                double time_start_dense = pcl::getTime();
                align360.setTargetFrame(frame360_1->sphereRGB, frame360_1->sphereDepth);
                align360.setSourceFrame(frame360_2->sphereRGB, frame360_2->sphereDepth);
//                align360.alignFrames360(Eigen::Matrix4f::Identity(), RegisterPhotoICP::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
                align360.alignFrames360(rigidTransf_dense, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
                rigidTransf_dense = rotOffset.inverse() * align360.getOptimalPose() * rotOffset;
                double time_end_dense = pcl::getTime();
                std::cout << "Dense " << (time_end_dense - time_start_dense) << std::endl;
                cout << "Dense regist \n" << rigidTransf_dense << endl;
            }
//            else
//            {
//                cout << "Align imgs " << endl;
//                registerer.RegisterDensePhotoICP(Map.vpSpheres[*compareSphereId], frame360, Eigen::Matrix4f::Identity(), RegisterPhotoICP::PHOTO_CONSISTENCY, RegisterRGBD360::DEFAULT_6DoF);
////                        registerer.RegisterDensePhotoICP(Map.vpSpheres[*compareSphereId], frame360, rigidTransf, RegisterPhotoICP::PHOTO_CONSISTENCY, RegisterRGBD360::DEFAULT_6DoF);
//                rigidTransf = registerer.getPose();
//            }

//            cout << "entropy-Pbmap " << registerer.calcEntropy() << endl;
//            cout << "entropy " << align360.calcEntropy() << endl;

//            double time_start_icp = pcl::getTime();
//            filter.filterVoxel(frame360_1->sphereCloud);
//            filter.filterVoxel(frame360_2->sphereCloud);
//            icp.setInputSource(frame360_2->sphereCloud);
//            icp.setInputTarget(frame360_1->sphereCloud);
//            pcl::PointCloud<PointT>::Ptr alignedICP(new pcl::PointCloud<PointT>);
//            Eigen::Matrix4f initRigidTransf = rigidTransf;
//      //      Eigen::Matrix4f initRigidTransf = Eigen::Matrix4f::Identity();
//            icp.align(*alignedICP, initRigidTransf);

//            double time_end_icp = pcl::getTime();
//            std::cout << "ICP took " << double (time_end_icp - time_start_icp) << std::endl;
//            std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
//            Eigen::Matrix4f icpTransformation = icp.getFinalTransformation();
//            cout << "ICP transformation:\n" << icpTransformation << endl << "PbMap-Registration\n" << rigidTransf << endl;

            rigidTransf = rigidTransf_dense;


            float dist = rigidTransf.block(0,3,3,1).norm();
            cout << "dist " << dist << endl;
            if(dist < 0.4)
            {
                bGoodRegistration = false;
                delete frame360_2;
                // Prepare the next frame selection\n";
                frame += selectSample;
                fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
                continue;
            }


            // Filter the spherical point cloud
            filter.filterEuclidean(frame360_2->sphereCloud);
            //        filter.filterVoxel(frame360_2->sphereCloud);

            //        // ICP
            //        icp.setInputSource(frame360_1->sphereCloud);
            //        icp.setInputTarget(frame360_2->sphereCloud);
            //        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr alignedICP(new pcl::PointCloud<pcl::PointXYZRGBA>);
            ////        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
            ////        Eigen::Matrix4f transformation = rigidTransf;
            //        icp.align (*alignedICP, rigidTransf);
            //      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
            ////        transformation = icp.getFinalTransformation ();
            //      cout << "ICP transformation:\n" << icp.getFinalTransformation() << endl << "PbMap-Registration\n" << rigidTransf << endl;

            // Update currentPose
            currentPose = currentPose * rigidTransf;

#if VISUALIZE_POINT_CLOUD
            {boost::mutex::scoped_lock updateLock(Map.mapMutex);

                Map.addKeyframe(frame360_2, currentPose); // Add keyframe

                // Update the global cloud
                pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>);
                pcl::transformPointCloud(*frame360_2->sphereCloud, *transformedCloud, currentPose);
                *viewer.globalMap += *transformedCloud;
                filter.filterVoxel(viewer.globalMap);
                viewer.currentLocation.matrix() = currentPose;

                rigidTransf_dense = Eigen::Matrix4f::Identity();
            cout << "Add frame " << endl;
            }
            //        #elif
            //          delete frame360_1;
#endif

#if SAVE_TRAJECTORY
            rigidTransf.saveToTextFile(mrpt::format("%s/Rt_%d_%d.txt", path_results.c_str(), frameOrder-1, frameOrder).c_str());
            //          ofstream saveRt;
            //          saveRt.open(mrpt::format("%s/poses/Rt_%d_%d.txt", path_dataset.c_str(), frame-1, frame).c_str());
            //          saveRt << rigidTransf;
            //          saveRt.close();
#endif

            ++frameOrder;
            frame += selectSample;
            //        fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",++frame);
            fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
            cout << "fileName " << fileName << endl;

            //        boost::this_thread::sleep (boost::posix_time::milliseconds (500));
            //    mrpt::system::pause();
        }

        //      delete frame360_2;
    }

};


void print_help(char ** argv)
{
    cout << "\nThis program performs Odometry with different kind of methods (PbMap, Direct registration, ICP) from the data stream recorded by an omnidirectional stereo .\n\n";

    cout << "  usage: " << argv[0] << " <imgRGB> <pathToResults> <sampleStream> " << endl;
    cout << "    <imgRGB> is the first reference image of the sequence, such sequence must contain the RGB and raw Depth images which must be in the same directory as imgRGB" << endl;
    cout << "    <pathToResults> is the directory where results (PbMap, Spherical PNG images, Poses files, etc.) should be saved" << endl;
    cout << "    <sampleStream> is the sampling step used in the dataset (e.g. 1: all the frames are used " << endl;
    cout << "                                                                  2: each other frame is used " << endl;
    cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}

int main (int argc, char ** argv)
{
    if(argc != 4 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
    {
        print_help(argv);
        return 0;
    }

    string imgRGB_1st = static_cast<string>(argv[1]);
    string path_results = static_cast<string>(argv[2]);
    int selectSample = atoi(argv[3]);

    cout << "Create Odometry360 object\n";
    Odometry360 odometry360;
    odometry360.run(imgRGB_1st, path_results, selectSample);

    cout << " EXIT\n";

    return (0);
}
