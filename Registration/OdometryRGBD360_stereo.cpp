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

typedef pcl::PointXYZRGBA PointT;
using namespace std;

/*! This class' main function 'run' performs PbMap-based odometry with the input data of a stream of
 *  omnidirectional RGB-D images (from a recorded sequence).
 */
class Odometry360
{
private:

    RegisterRGBD360 registerer;

    //Calib360 calib;

    Frame360 *frame_trg, *frame_src;
    Frame360 *frame_trg_fused, *frame_src_fused;

    Map360 map;

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

        assert ( fexists(rgb1.c_str()) );
        //std::cout << "  rgb1:\t\t" << rgb1 << std::endl;
        string rgb = rgb1;

        string fileExtRGB = ".png"; // std::cout << "  end: " << rgb.substr(rgb.length()-4) << std::endl;
        string fileExtRawDepth = ".raw";
        string fileExtFusedDepth = "pT.raw";

        string depth_raw;
        string depth_fused;

        string path_sequence = rgb.substr(0, rgb.length()-14);
        string frame_num_s = rgb.substr(rgb.length()-11, 7);
        int frame_num = atoi( frame_num_s.c_str() );

        if( fileExtRGB.compare( rgb.substr(rgb.length()-4) ) == 0 ) // If the first string correspond to a pointCloud path
        {
          depth_raw = path_sequence + "depth" + frame_num_s + fileExtRawDepth;
          depth_fused = path_sequence + "gapFillingPlusFusion/depth" + frame_num_s + fileExtRawDepth;
//          depth_fused = path_sequence + "depth" + frame_num_s + fileExtFusedDepth;
          std::cout << "  rgb:\t\t" << rgb << "\n  depth_raw:\t" << depth_raw << "\n  depth_fused:\t" << depth_fused << std::endl;
        }
        else
        {
            std::cerr << "\n... INVALID IMAGE FILE!!! \n";
            return;
        }

        // Open reference RGB and Depth images
        if ( fexists(depth_raw.c_str()) && fexists(depth_fused.c_str()) )
        {
            frame_src->loadDepth(depth_raw, &maskCar);
            frame_src->loadRGB(rgb);
//            frame_src->buildSphereCloud();
//            frame_src->filterCloudBilateral_stereo();
//            frame_src->segmentPlanesStereo();
            //  //cv::namedWindow( "sphereRGB", WINDOW_AUTOSIZE );// Create a window for display.
            //  cv::imshow( "sphereRGB", frame_src->sphereRGB );
            //  cv::waitKey(0);

            frame_src_fused->loadDepth(depth_fused, &maskCar);
            frame_src_fused->loadRGB(rgb);
            frame_src_fused->buildSphereCloud();
//            frame_src_fused->filterCloudBilateral_stereo();
//            frame_src_fused->segmentPlanesStereo();
        }
        else
        {
            std::cerr << "\n... Some of the initial image files do not exist!!! \n";
            return;
        }
        //cout << "Odometry360 1" << endl;


        bool bGoodRegistration = true;
        Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f currentPose_fused = Eigen::Matrix4f::Identity();
        //Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f Rt_dense = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f Rt_dense_fused = Eigen::Matrix4f::Identity();
//        Eigen::Matrix4f Rt_dense_photo = Eigen::Matrix4f::Identity();
//        Eigen::Matrix4f Rt_dense_depth = Eigen::Matrix4f::Identity();
//        Eigen::Matrix4f Rt_icp = Eigen::Matrix4f::Identity();
//        Eigen::Matrix4f Rt_pbmap = Eigen::Matrix4f::Identity();
//        // Non-fused data
//        Eigen::Matrix4f Rt_dense_NF = Eigen::Matrix4f::Identity();
//        Eigen::Matrix4f Rt_dense_photo_NF = Eigen::Matrix4f::Identity();


        // Initialize Dense aligner
        DirectRegistration align360; // Dense RGB-D alignment
        align360.setSensorType(ProjectionModel::STEREO_OUTDOOR); // This is use to adapt some features/hacks for each type of image (see the implementation of DirectRegistration::register360 for more details)
        align360.setNumPyr(6);
        align360.setMaxDepth(15.f);
        align360.useSaliency(false);
//        align360.thresSaliencyIntensity(0.f);
//        align360.thresSaliencyIntensity(0.f);
      // align360.setVisualization(true);
        align360.setGrayVariance(8.f/255);

//        // Initialize ICP
//        pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
//        icp.setMaxCorrespondenceDistance (0.4);
//        icp.setMaximumIterations (10);
//        icp.setTransformationEpsilon (1e-9);
//        icp.setRANSACOutlierRejectionThreshold (0.1);
//        pcl::PointCloud<PointT>::Ptr cloud_dense_trg(new pcl::PointCloud<PointT>);
//        pcl::PointCloud<PointT>::Ptr cloud_dense_src(new pcl::PointCloud<PointT>);


//        // Filter the spherical point cloud
//        // ICP -> Filter the point clouds and remove nan points
//        FilterPointCloud<PointT> filter(0.1, 10.f); // Initialize filters (for visualization)
////        filter.filterEuclidean(frame_src_fused->getSphereCloud());
//        filter.filterVoxel(frame_src_fused->getSphereCloud(), cloud_dense_src);

        // Visualize
#if VISUALIZE_POINT_CLOUD
        Map360 map;
        Map360_Visualizer viewer(map,2);
        //viewer.bDrawCurrentLocation = true;

        // Add the first observation to the map
        {boost::mutex::scoped_lock updateLock(map.mapMutex);
            map.addKeyframe(frame_src_fused, currentPose);
//            *viewer.globalMap += *frame_src_fused->getSphereCloud();
//            filter.filterEuclidean (frame_src_fused->getSphereCloud(), viewer.globalMap);
            map.vSelectedKFs.push_back(0);
        }
        std::cout << "VISUALIZE_POINT_CLOUD " << std::endl;
#endif

        // Results verifications (IROS 2015)
        //Eigen::Matrix<float,2,12> convergence_005 = Eigen::Matrix<float,2,12>::Zeros();

//        size_t num_frames = 1;
//        size_t num_planes = frame_src->planes.vPlanes.size ();
//        size_t plane_inliers = 0;
//        for(size_t i=0; i < frame_src->planes.vPlanes.size (); i++)
//            plane_inliers += frame_src->planes.vPlanes[i].inliers.size();
//        size_t num_planes_fused = frame_src_fused->planes.vPlanes.size ();
//        size_t plane_inliers_fused = 0;
//        for(size_t i=0; i < frame_src_fused->planes.vPlanes.size (); i++)
//            plane_inliers_fused += frame_src_fused->planes.vPlanes[i].inliers.size();

        ofstream poses_dense, poses_dense_fused, poses_dense_photo, poses_dense_depth, poses_icp, poses_pbmap;
        //string trajectoryFile = mrpt::format("%s/poses_dense.txt", path_results);
        string trajectoryFile(path_results + "/poses_dense.txt");
        poses_dense.open ( trajectoryFile.c_str() );
        trajectoryFile = string(path_results + "/poses_dense_fused.txt");
        poses_dense_fused.open ( trajectoryFile.c_str() );
//        trajectoryFile = string(path_results + "/poses_dense_photo.txt");
//        poses_dense_photo.open ( trajectoryFile.c_str() );
//        trajectoryFile = string(path_results + "/poses_dense_depth.txt");
//        poses_dense_depth.open ( trajectoryFile.c_str() );
//        trajectoryFile = string(path_results + "/poses_icp.txt");
//        poses_icp.open ( trajectoryFile.c_str() );
//        trajectoryFile = string(path_results + "/poses_pbmap.txt");
//        poses_pbmap.open ( trajectoryFile.c_str() );
//        if(!poses_dense.is_open() || !poses_dense_photo.is_open() || !poses_dense_depth.is_open() || !poses_icp.is_open() || !poses_pbmap.is_open() )
//            { std::cerr << "\n ...Cannot open output file: " << poses_dense << "\n"; return; }

        ofstream trajectory_raw, trajectory_fused_GT;
        trajectoryFile = string(path_results + "/trajectory_fused_GT.txt");
        trajectory_fused_GT.open ( trajectoryFile.c_str() );
        trajectoryFile = string(path_results + "/trajectory_raw.txt");
        trajectory_raw.open ( trajectoryFile.c_str() );


        // The reference of the spherical image and the point Clouds are not the same! I should always use the same coordinate system (TODO)
//        float angleOffset = 157.5;
//        Eigen::Matrix4f rotOffset = Eigen::Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);

        frame_num += selectSample;
        depth_raw = mrpt::format("%sdepth%07d%s", path_sequence.c_str(), frame_num, fileExtRawDepth.c_str());
        depth_fused = mrpt::format("%sgapFillingPlusFusion/depth%07d%s", path_sequence.c_str(), frame_num, fileExtRawDepth.c_str());
//        depth_fused = mrpt::format("%sdepth%07d%s", path_sequence.c_str(), frame_num, fileExtFusedDepth.c_str());
        rgb = mrpt::format("%stop%07d%s", path_sequence.c_str(), frame_num, fileExtRGB.c_str());

        while( fexists(depth_raw.c_str()) && fexists(depth_fused.c_str()) && fexists(rgb.c_str()) )
        {
//            std::cout << "  rgb:\t\t" << rgb << "\n  depth_raw:\t" << depth_raw << "\n  depth_fused:\t" << depth_fused << std::endl;
            std::cout << "  rgb: " << rgb << std::endl;

            if(bGoodRegistration)
            {
//                if( map.vpSpheres.size() % 50 != 0 )
//                {
//                delete frame_trg;
//                delete frame_trg_fused;
//                }

                frame_trg = frame_src;
                frame_trg_fused = frame_src_fused;
//                cloud_dense_trg = cloud_dense_src;

                frame_src = new Frame360;
                frame_src_fused = new Frame360;
//                cloud_dense_src.reset(new pcl::PointCloud<PointT>);
            }

            // Open reference RGB and Depth images
            if ( fexists(rgb.c_str()) && fexists(depth_raw.c_str()) && fexists(depth_fused.c_str()) )
            {
                frame_src->loadDepth(depth_raw, &maskCar);
                frame_src->loadRGB(rgb);
//                frame_src->buildSphereCloud();
//                frame_src->filterCloudBilateral_stereo();
//                frame_src->segmentPlanesStereo();
                //  //cv::namedWindow( "sphereRGB", WINDOW_AUTOSIZE );// Create a window for display.
                //  cv::imshow( "sphereRGB", frame_src->sphereRGB );
                //  cv::waitKey(0);

                frame_src_fused->loadDepth(depth_fused, &maskCar);
                frame_src_fused->loadRGB(rgb);
                frame_src_fused->buildSphereCloud();
//                frame_src_fused->filterCloudBilateral_stereo();
//                frame_src_fused->segmentPlanesStereo();
                //cout << "frame_trg_fused " << frame_trg_fused->sphereCloud->width << " " << frame_trg_fused->sphereCloud->height << " " << frame_trg_fused->sphereCloud->is_dense << " " << endl;
            }
            else
            {
                std::cout << "\n... END OF SEQUENCE -> EXIT PROGRAM \n";

                delete frame_trg, frame_src;
                delete frame_trg_fused, frame_src_fused;
                return;
            }


//            // Align the two frames
//            //bGoodRegistration =
//            registerer.RegisterPbMap(frame_trg_fused, frame_src_fused, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
//            //        bGoodRegistration = registerer.RegisterPbMap(frame_trg_fused, frame_src_fused, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_ODOMETRY_3DoF);
//            //            cout << "entropy-Pbmap " << registerer.calcEntropy() << endl;
//            //            cout << "entropy " << align360.calcEntropy() << endl;

            //cout << "DirectRegistration \n";
            align360.setTargetFrame(frame_trg_fused->sphereRGB, frame_trg_fused->sphereDepth);
            align360.setSourceFrame(frame_src_fused->sphereRGB, frame_src_fused->sphereDepth);
            align360.register360(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            Rt_dense_fused = align360.getOptimalPose();
//            align360.register360(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            Rt_dense_photo = align360.getOptimalPose();
//            align360.register360(Eigen::Matrix4f::Identity(), DirectRegistration::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            Rt_dense_depth = align360.getOptimalPose();
//            align360.register360(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            Rt_dense = align360.getOptimalPose();

            align360.setTargetFrame(frame_trg->sphereRGB, frame_trg->sphereDepth);
            align360.setSourceFrame(frame_src->sphereRGB, frame_src->sphereDepth);
            align360.register360(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            Rt_dense = align360.getOptimalPose();
//            align360.register360(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            Rt_dense_photo_NF = align360.getOptimalPose();


//            double time_start = pcl::getTime();

//            icp.setInputTarget(cloud_dense_trg);
//            filter.filterVoxel(frame_src_fused->getSphereCloud(), cloud_dense_src);
//            icp.setInputSource(cloud_dense_src);
//            pcl::PointCloud<PointT>::Ptr alignedICP(new pcl::PointCloud<PointT>);
//          //  Eigen::Matrix4d initRt = registerer.getPose();
//            Eigen::Matrix4f initRt = Eigen::Matrix4f::Identity();
//            icp.align(*alignedICP, initRt);

//            double time_end = pcl::getTime();
//            std::cout << "ICP took " << double (time_end - time_start) << std::endl;

//            //std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
//            Rt_icp = icp.getFinalTransformation(); //.cast<double>();


            // Update currentPose
            currentPose = currentPose * Rt_dense;
//            currentPose2 = currentPose2 * Rt_dense_photo;
            currentPose_fused = currentPose_fused * Rt_dense_fused;



            // Update verifications (IROS 2015)
//            ++num_frames;
//            num_planes += frame_src->planes.vPlanes.size ();
//            for(size_t i=0; i < frame_src->planes.vPlanes.size (); i++)
//                plane_inliers += frame_src->planes.vPlanes[i].inliers.size();
//            num_planes_fused += frame_src_fused->planes.vPlanes.size ();
//            for(size_t i=0; i < frame_src_fused->planes.vPlanes.size (); i++)
//                plane_inliers_fused += frame_src_fused->planes.vPlanes[i].inliers.size();

            for(int i=0;i<4;i++)
              for(int j=0;j<4;j++)
              {
                poses_dense_fused << Rt_dense_fused(j,i) << "\t";
                poses_dense << Rt_dense(j,i) << "\t";
//                poses_dense_photo << Rt_dense_photo(j,i) << "\t";
//                poses_dense_depth << Rt_dense_depth(j,i) << "\t";
//                poses_icp << Rt_icp(j,i) << "\t";
//                poses_pbmap << Rt_pbmap(j,i) << "\t";

                trajectory_raw << currentPose(j,i) << "\t";
                trajectory_fused_GT << currentPose_fused(j,i) << "\t";
              }
            poses_dense_fused << std::endl;
            poses_dense << std::endl;
//            poses_dense_photo << std::endl;
//            poses_dense_depth << std::endl;
//            poses_icp << std::endl;
//            poses_pbmap << std::endl;

            trajectory_raw << std::endl;
            trajectory_fused_GT << std::endl;

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
//                Rt_pbmap = registerer.getPose();
//                cout << "\nPbMap regist \n" << Rt_pbmap << endl;
//            }



//            float dist = Rt.block(0,3,3,1).norm();
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


//            if(select_keyframe)
//            {
//                cout << "Add keyframe " << endl;
//                Rt_dense = Eigen::Matrix4f::Identity();
//                Rt_icp = Eigen::Matrix4f::Identity();
//            }

            // Next frame
            frame_num += selectSample;
            depth_raw = mrpt::format("%sdepth%07d%s", path_sequence.c_str(), frame_num, fileExtRawDepth.c_str());
            depth_fused = mrpt::format("%sgapFillingPlusFusion/depth%07d%s", path_sequence.c_str(), frame_num, fileExtRawDepth.c_str());
//            depth_fused = mrpt::format("%sdepth%07d%s", path_sequence.c_str(), frame_num, fileExtFusedDepth.c_str());
            rgb = mrpt::format("%stop%07d%s", path_sequence.c_str(), frame_num, fileExtRGB.c_str());
            std::cout << "Frame " << frame_num << std::endl;

#if VISUALIZE_POINT_CLOUD
            // Filter the spherical point cloud
//            filter.filterEuclidean(frame_src_fused->getSphereCloud());

            {boost::mutex::scoped_lock updateLock(map.mapMutex);
                map.addKeyframe(frame_src_fused, currentPose);// Add keyframe
                map.vOptimizedPoses.back() = currentPose_fused;

                if( map.vpSpheres.size() % 50 == 0 )
                    map.vSelectedKFs.push_back(map.vpSpheres.size());


//                // Update the global cloud
//                pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>);
//                pcl::transformPointCloud(*frame_src_fused->getSphereCloud(), *transformedCloud, currentPose);
//                *viewer.globalMap += *transformedCloud;
//                filter.filterVoxel(viewer.globalMap);
//                viewer.currentLocation.matrix() = currentPose;
            }
#endif

#if SAVE_TRAJECTORY
            // Rt.saveToTextFile(mrpt::format("%s/Rt_%d_%d.txt", path_results.c_str(), frameOrder-1, frameOrder).c_str());
            //          ofstream saveRt;
            //          saveRt.open(mrpt::format("%s/poses/Rt_%d_%d.txt", path_dataset.c_str(), frame-1, frame).c_str());
            //          saveRt << Rt;
            //          saveRt.close();
#endif

            //        boost::this_thread::sleep (boost::posix_time::milliseconds (500));
            //    mrpt::system::pause();
        }

        if( !fexists(depth_raw.c_str()) )
            std::cout << "FILE DOES NOT EXIST " << depth_raw << std::endl;

        if( !fexists(depth_fused.c_str()) )
            std::cout << "FILE DOES NOT EXIST " << depth_fused << std::endl;

        if( !fexists(rgb.c_str()) )
            std::cout << "FILE DOES NOT EXIST " << rgb << std::endl;


//        // Show verifications (IROS 2015)
//        std::cout << "Average num of planes " << float(num_planes)/num_frames << " average inliers " << float(plane_inliers)/num_frames << std::endl;
//        std::cout << "FUSED: Average num of planes " << float(num_planes_fused)/num_frames << " average inliers " << float(plane_inliers_fused)/num_frames << std::endl;

        poses_dense.close();
        poses_dense_fused.close();
//        poses_dense_photo.close();
//        poses_dense_depth.close();
//        poses_icp.close();
//        poses_pbmap.close();

        trajectory_raw.close();
        trajectory_fused_GT.close();

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
