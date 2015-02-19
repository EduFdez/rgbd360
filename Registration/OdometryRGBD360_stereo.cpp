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
#include <filterCloudBilateral_stereo.h>
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

    //Calib360 calib;

    Frame360 *frame_trg, *frame_src;
    Frame360 *frame_trg_fused, *frame_src_fused;

public:
    Odometry360() :
        registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH))
    {
        // Get the calibration matrices for the different sensors
//        calib.loadExtrinsicCalibration();
//        calib.loadIntrinsicCalibration();

        frame_trg = new Frame360;
        frame_src = new Frame360;
        frame_trg_fused = new Frame360;
        frame_src_fused = new Frame360;

    }

    /*! Run the odometry from the image specified by "imgRGB_1st".
     */
    void run(const string &rgb1, const string &path_results, const int &selectSample)
    {
        cv::Mat maskCar = cv::imread("/Data/Shared_Lagadic/useful_code/maskCar_.png",0);
      //  cv::imshow( "maskCar", maskCar );
      //  cv::waitKey(0);

        assert (fexists(rgb1.c_str()) );
        std::cout << "  rgb1: " << rgb1 << std::endl;
        string rgb = rgb1;

        string fileExtRGB = ".png";
        string fileExtRawDepth = ".raw";
        string fileExtFusedDepth = "pT.raw";

        string depth_raw;
        string depth_fused;
        std::cout << "  end: " << rgb.substr(rgb.length()-4) << std::endl;
        string path_sequence = rgb.substr(0, rgb.length()-14);
        string frame_num_s = rgb.substr(rgb.length()-11, 7);
        int frame_num = atoi( frame_num_s.c_str() );

        if( fileExtRGB.compare( rgb.substr(rgb.length()-4) ) == 0 ) // If the first string correspond to a pointCloud path
        {
          depth_raw = path_sequence + "depth" + frame_num_s + fileExtRawDepth;
          depth_fused = path_sequence + "depth" + frame_num_s + fileExtFusedDepth;
          std::cout << "  depth1: " << depth1 << "\n  depth2: " << depth2 << std::endl;
        }
        else
        {
            std::cerr << "\n... INVALID IMAGE FILE!!! \n";
            return;
        }

        // Open reference RGB and Depth images
        if ( fexists(depth_raw.c_str()) && fexists(depth_fused.c_str()) )
        {
            frame_src.loadDepth(depth_raw, &maskCar);
            frame_src.loadRGB(rgb);
            frame_src.buildSphereCloud();
            frame_src.filterCloudBilateral_stereo();
            frame_src.segmentPlanesStereo();
            //  //cv::namedWindow( "sphereRGB", WINDOW_AUTOSIZE );// Create a window for display.
            //  cv::imshow( "sphereRGB", frame_src.sphereRGB );
            //  cv::waitKey(0);

            frame_src_fused.loadDepth(depth_fused, &maskCar);
            frame_src_fused.loadRGB(rgb);
            frame_src_fused.buildSphereCloud();
            frame_src_fused.filterCloudBilateral_stereo();
            frame_src_fused.segmentPlanesStereo();
        }
        else
        {
            std::cerr << "\n... Some of the initial image files do not exist!!! \n";
            return;
        }

        // Initialize Dense aligner
        RegisterPhotoICP align360; // Dense RGB-D alignment
        align360.setNumPyr(6);
        align360.useSaliency(false);
      // align360.setVisualization(true);
        align360.setGrayVariance(3.f/255);

        // Initialize ICP
        pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
        icp.setMaxCorrespondenceDistance (0.4);
        icp.setMaximumIterations (10);
        icp.setTransformationEpsilon (1e-9);
        icp.setRANSACOutlierRejectionThreshold (0.1);
        pcl::PointCloud<PointT>::Ptr cloud_dense_trg(new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr cloud_dense_src(new pcl::PointCloud<PointT>);

        bool bGoodRegistration = true;
        Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();


        // Filter the spherical point cloud
        // ICP -> Filter the point clouds and remove nan points
        FilterPointCloud<PointT> filter(0.1); // Initialize filters (for visualization)
//        filter.filterEuclidean(frame_src_fused->getSphereCloud());
        filter.filterVoxel(frame_src_fused->getSphereCloud(), cloud_dense_src);

        // Visualize
#if VISUALIZE_POINT_CLOUD
        Map360_Visualizer viewer(Map,1);
        viewer.bDrawCurrentLocation = true;

        // Add the first observation to the map
        {boost::mutex::scoped_lock updateLock(Map.mapMutex);

            Map.addKeyframe(frame_src_fused, currentPose);
            *viewer.globalMap += *frame_src_fused->getSphereCloud();
        }

#endif

        Eigen::Matrix4f rigidTransf = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f rigidTransf_pbmap = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f rigidTransf_dense = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f rigidTransf_icp = Eigen::Matrix4f::Identity();

        // The reference of the spherical image and the point Clouds are not the same! I should always use the same coordinate system (TODO)
        float angleOffset = 157.5;
        Eigen::Matrix4f rotOffset = Eigen::Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);

        while( fexists(fileName.c_str()) )
        {
            if(bGoodRegistration)
            {
                frame_trg = frame_src;
                frame_trg_fused = frame_src_fused;
                cloud_dense_trg = cloud_dense_src;
            }

            frame_num += selectSample;
            cout << "Frame " << frame_num << endl;

            depth_raw = mrpt::format("%sdepth%07%s", path_sequence, frame_num, fileExtRawDepth);
            depth_fused = mrpt::format("%sdepth%07%s", path_sequence, frame_num, fileExtFusedDepth);
            rgb = mrpt::format("%stop%07%s", path_sequence, frame_num, fileExtRGB);
            std::cout << "  rgb: " << rgb << std::endl;

            // Open reference RGB and Depth images
            if ( fexists(rgb.c_str()) && fexists(depth_raw.c_str()) && fexists(depth_fused.c_str()) )
            {
                frame_src.loadDepth(depth_raw, &maskCar);
                frame_src.loadRGB(rgb);
                frame_src.buildSphereCloud();
                frame_src.filterCloudBilateral_stereo();
                frame_src.segmentPlanesStereo();
                //  //cv::namedWindow( "sphereRGB", WINDOW_AUTOSIZE );// Create a window for display.
                //  cv::imshow( "sphereRGB", frame_src.sphereRGB );
                //  cv::waitKey(0);

                frame_src_fused.loadDepth(depth_fused, &maskCar);
                frame_src_fused.loadRGB(rgb);
                frame_src_fused.buildSphereCloud();
                frame_src_fused.filterCloudBilateral_stereo();
                frame_src_fused.segmentPlanesStereo();
            }
            else
            {
                std::cout << "\n... END OF SEQUENCE -> EXIT PROGRAM \n";

                delete frame_trg, frame_src;
                delete frame_trg_fused, frame_src_fused;
                return;
            }

            // Align the two frames
            bGoodRegistration = registerer.RegisterPbMap(frame_trg_fused, frame_src_fused, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
            //        bGoodRegistration = registerer.RegisterPbMap(frame_trg_fused, frame_src_fused, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_ODOMETRY_3DoF);
            //            cout << "entropy-Pbmap " << registerer.calcEntropy() << endl;
            //            cout << "entropy " << align360.calcEntropy() << endl;

            //cout << "RegisterPhotoICP \n";
            align360.setTargetFrame(frame_trg_fused.sphereRGB, frame_trg_fused.sphereDepth);
            align360.setSourceFrame(frame_src_fused.sphereRGB, frame_src_fused.sphereDepth);
            align360.alignFrames360(Eigen::Matrix4f::Identity(), RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            rigidTransf_dense = align360.getOptimalPose();


            cout << "frame_trg_fused " << frame_trg_fused.sphereCloud->width << " " << frame_trg_fused.sphereCloud->height << " " << frame_trg_fused.sphereCloud->is_dense << " " << endl;
            cout << "voxel filtered frame_trg_fused " << sphereCloud_dense1->width << " " << sphereCloud_dense1->height << " " << sphereCloud_dense1->is_dense << " " << endl;

            icp.setInputTarget(cloud_dense_trg);
            filter.filterVoxel(frame_src_fused->getSphereCloud(), cloud_dense_src);
            icp.setInputSource(frame_src_fused);
            pcl::PointCloud<PointT>::Ptr alignedICP(new pcl::PointCloud<PointT>);
          //  Eigen::Matrix4d initRigidTransf = registerer.getPose();
            Eigen::Matrix4f initRigidTransf = Eigen::Matrix4f::Identity();
            icp.align(*alignedICP, initRigidTransf);

            double time_end = pcl::getTime();
            std::cout << "ICP took " << double (time_end - time_start) << std::endl;

            //std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
            rigidTransf_icp = icp.getFinalTransformation(); //.cast<double>();


//            if(!bGoodRegistration)
//            {
//                cout << "\tBad registration\n\n";
//                cout << "\nPbMap regist \n" << registerer.getPose() << endl;
////                fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",++frame);
////                bGoodRegistration = false;
////                delete frame_src_fused;
////                continue;
//            }
//            else
//            {
//                rigidTransf_pbmap = registerer.getPose();
//                cout << "\nPbMap regist \n" << rigidTransf_pbmap << endl;
//            }



//            float dist = rigidTransf.block(0,3,3,1).norm();
//            cout << "dist " << dist << endl;
//            if(dist < 0.4)
//            {
//                bGoodRegistration = false;
//                delete frame_src_fused;
//                // Prepare the next frame selection\n";
//                frame += selectSample;
//                fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
//                continue;
//            }


            // Filter the spherical point cloud
            filter.filterEuclidean(frame_src_fused->getSphereCloud());

            // Update currentPose
            currentPose = currentPose * rigidTransf_dense.inverse();

#if VISUALIZE_POINT_CLOUD
            {boost::mutex::scoped_lock updateLock(Map.mapMutex);

                Map.addKeyframe(frame_src_fused, currentPose); // Add keyframe

                // Update the global cloud
                pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>);
                pcl::transformPointCloud(*frame_src_fused->getSphereCloud(), *transformedCloud, currentPose);
                *viewer.globalMap += *transformedCloud;
                filter.filterVoxel(viewer.globalMap);
                viewer.currentLocation.matrix() = currentPose;

                rigidTransf_dense = Eigen::Matrix4f::Identity();
                rigidTransf_icp = Eigen::Matrix4f::Identity();
            cout << "Add keyframe " << endl;
            }
            //        #elif
            //          delete frame_trg_fused;
#endif

#if SAVE_TRAJECTORY
            rigidTransf.saveToTextFile(mrpt::format("%s/Rt_%d_%d.txt", path_results.c_str(), frameOrder-1, frameOrder).c_str());
            //          ofstream saveRt;
            //          saveRt.open(mrpt::format("%s/poses/Rt_%d_%d.txt", path_dataset.c_str(), frame-1, frame).c_str());
            //          saveRt << rigidTransf;
            //          saveRt.close();
#endif

            //        boost::this_thread::sleep (boost::posix_time::milliseconds (500));
            //    mrpt::system::pause();
        }

        delete frame_trg, frame_src;
        delete frame_trg_fused, frame_src_fused;
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
