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

#include <mrpt/gui.h>
#include <mrpt/opengl.h>
#include <mrpt/utils/CFileGZInputStream.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
//#include <mrpt/slam/CObservationRGBD360.h>
//#include <mrpt/slam/CObservationIMU.h>
#include <mrpt/obs/CRawlog.h>
//#include <mrpt/slam/CActionCollection.h>
//#include <mrpt/slam/CSensoryFrame.h>

#include "CloudRGBD.h"
//#include <SerializeFrameRGBD.h>
#include <FilterPointCloud.h>
#include <RegisterDense.h>

//#include <pcl/registration/icp.h>
//#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP

#include <pcl/console/parse.h>

#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>

#include <CloudRGBD_Ext.h>
#include <Frame360_Visualizer.h>

#define SAVE_TRAJECTORY 0
#define SAVE_IMAGES 0
#define VISUALIZE_POINT_CLOUD 1

typedef pcl::PointXYZRGBA PointT;
using namespace std;
using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace cv;

/*! This class' main function 'run' performs PbMap-based odometry with the input data of a stream of
 *  omnidirectional RGB-D images (from a recorded sequence).
 */
class OdometryRGBD
{
private:

    RegisterDense registerer;

public:

    bool bVisualize;
    size_t skip_frames;

    OdometryRGBD()
    {
        registerer.useSaliency(true);
        skip_frames = 0;
        bVisualize = false;
    };

    void run(const string &filename, const int &selectSample = 1) //const string &path_results,
    {
        cout << "OdometryRGBD::run() " << selectSample << "\n";
        mrpt::obs::CRawlog dataset;
        //						Open Rawlog File
        //==================================================================
        if (!dataset.loadFromRawLogFile(filename))
            throw std::runtime_error("\nCouldn't open dataset dataset file for input...");

        cout << "dataset size " << dataset.size() << "\n";
        //dataset_count = 0;

        // Set external images directory:
        const string imgsPath = CRawlog::detectImagesDirectory(filename);
        CImage::IMAGES_PATH_BASE = imgsPath;

//        mrpt::utils::CFileGZInputStream dataset(filename);
//        mrpt::obs::CActionCollectionPtr action;
//        mrpt::obs::CSensoryFramePtr observations;
        mrpt::obs::CObservationPtr observation;
        size_t rawlogEntry=0;
        //bool end = false;

        mrpt::obs::CObservation3DRangeScanPtr obsRGBD;  // The RGBD observation
        //CObservation2DRangeScanPtr laserObs;    // Pointer to the laser observation
        size_t n_RGBD = 0, n_obs = 0;

        bool bGoodRegistration = true;
        Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f currentPoseIC = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f relativePose = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f relativePose_photo = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f relativePose_depth = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f relativePoseIC = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f relativePoseIC_photo = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f relativePoseIC_depth = Eigen::Matrix4f::Identity();

        //mrpt::math::CMatrix rangeImage;
        //mrpt::utils::CImage intensityImage;
        cv::Mat intensity_src, intensity_trg;
        cv::Mat depth_src, depth_trg;
        CloudRGBD frameRGBD_src, frameRGBD_trg;
        bool bFirstFrame = true;

        // Initialize Dense registration
        RegisterDense registerRGBD; // Dense RGB-D alignment
        registerRGBD.setSensorType( RegisterDense::KINECT ); // This is use to adapt some features/hacks for each type of image (see the implementation of RegisterDense::register360 for more details)
        registerRGBD.setNumPyr(5);
        registerRGBD.setMaxDepth(8.f);
//        registerRGBD.useSaliency(true);
//        registerRGBD.thresSaliencyIntensity(0.f);
//        registerRGBD.thresSaliencyIntensity(0.f);
//        registerRGBD.setBilinearInterp(true);
//        registerRGBD.setVisualization(true);
        registerRGBD.setGrayVariance(8.f/255);

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


//        while ( mrpt::obs::CRawlog::getActionObservationPairOrObservation(
//                    dataset,      // Input file
//                    action,            // Possible out var: action of a pair action/obs
//                    observations,  // Possible out var: obs's of a pair action/obs
//                    observation,    // Possible out var: a single obs.
//                    rawlogEntry    // Just an I/O counter
//                    ) )

        //n_obs = dataset.size() - 100; 1014;
        while ( n_obs < dataset.size() )
        {
            observation = dataset.getAsObservation(n_obs);
            ++n_obs;
            if(!IS_CLASS(observation, CObservation3DRangeScan))
            {
                continue;
            }
            cout << n_obs << " observation: " << observation->sensorLabel << ". Timestamp " << observation->timestamp << endl;

            // Apply selectSample
            ++n_RGBD;
            if( n_RGBD < skip_frames )
                continue;

            if(n_RGBD % selectSample != 0)
                continue;

            obsRGBD = mrpt::obs::CObservation3DRangeScanPtr(observation);
            obsRGBD->load();

//            // Draw frame
//            CloudRGBD_Ext cloud;
//            cloud.setRGBImage( cv::Mat(obsRGBD->intensityImage.getAs<IplImage>()) );
//            cv::Mat depth_mat;
//            convertRange_mrpt2cvMat(obsRGBD->rangeImage, depth_mat);
//            cloud.setDepthImage(depth_mat);
//            cloud.getPointCloud();

//            RGBD_Visualizer cloud_viewer(cloud.getPointCloud());
//            while (!cloud_viewer.viewer.wasStopped ())
//                boost::this_thread::sleep (boost::posix_time::milliseconds (10));

//            cv::imshow( "img_", cloud.getRGBImage( ) );
//            cv::imshow( "depth_", cloud.getDepthImage( ) );
//            while (cv::waitKey(1)!='\n')
//                boost::this_thread::sleep (boost::posix_time::milliseconds (10));

            if( bFirstFrame )
            {
                assert( obsRGBD->hasIntensityImage && obsRGBD->hasRangeImage ); // Check that the images are really RGBD

                intensity_src = cv::Mat(obsRGBD->intensityImage.getAs<IplImage>());
                //intensity_src = cv::cvarrToMat(obsRGBD->intensityImage.getAs<IplImage>());
                convertRange_mrpt2cvMat(obsRGBD->rangeImage, depth_src);
                //registerRGBD.setSourceFrame(intensity_src, depth_src);

                bFirstFrame = false;
                continue;
            }

            if(bGoodRegistration)
            {
                cout << "bGoodRegistration " << n_RGBD << " timestamp " << observation->timestamp << endl;
                intensity_trg = intensity_src;
                depth_trg = depth_src;
                //depth_trg = depth_src.clone();
                intensity_src = cv::Mat(obsRGBD->intensityImage.getAs<IplImage>());
                depth_src = cv::Mat(depth_trg.rows, depth_trg.cols, CV_32FC1);
                convertRange_mrpt2cvMat(obsRGBD->rangeImage, depth_src);
//                cv::Mat new_depth;
//                convertRange_mrpt2cvMat(obsRGBD->rangeImage, new_depth);
//                depth_src = new_depth;

//                cv::imwrite(mrpt::format("/home/efernand/rgb_trg.png"), intensity_trg);
//                cv::imwrite(mrpt::format("/home/efernand/rgb_src.png"), intensity_src);
            }
            else
                cout << "badRegistration " << endl;

            if(bVisualize)
            {
                cv::imshow( "intensity_src", intensity_src );
                cv::Mat sphDepthVis_src;
                depth_src.convertTo( sphDepthVis_src, CV_8U, 25 ); //CV_16UC1
                cv::imshow( "depth_src", sphDepthVis_src );

                cv::imshow( "intensity_trg", intensity_trg );
                cv::Mat sphDepthVis_trg;
                depth_trg.convertTo( sphDepthVis_trg, CV_8U, 25 ); //CV_16UC1
                cv::imshow( "depth_trg", sphDepthVis_trg );

                cv::Mat diff;
                cv::absdiff(intensity_src, intensity_trg, diff);
                cv::imshow( "diff", diff );

                cv::Mat diff_depth;
                cv::absdiff(sphDepthVis_src, sphDepthVis_trg, diff_depth);
                cv::imshow( "diff_depth", diff_depth );

                cv::waitKey(0);

                //                    if(mode == 2)
                //                    {
                //                        depth_src.convertTo( depth_src, CV_8U, 25 ); //CV_16UC1
                //                        //            depth_src.convertTo( depth_src, CV_16U, 1000 ); //CV_16UC1
                //                        cv::imwrite(path_results + mrpt::format("/rgb_%04d.png",frame), intensity_src);
                //                        cv::imwrite(path_results + mrpt::format("/depth_%04d.png",frame), depth_src);
                //                    }

            }

//            // Visualize the data
//            if(bVisualize)
//            {
//                // It IS a new observation:
//                mrpt::system::TTimeStamp last_obs_tim = observation->timestamp;

//                // Update visualization ---------------------------------------

//                win3D.get3DSceneAndLock();

//                // Estimated grabbing rate:
//                win3D.addTextMessage(-350,-13, format("Timestamp: %s", mrpt::system::dateTimeLocalToString(last_obs_tim).c_str()), TColorf(0.6,0.6,0.6),"mono",10,mrpt::opengl::FILL, 100);

//                //        // Show intensity image:
//                //        if (obsRGBD[0]->hasIntensityImage )
//                //        {
//                //          viewInt->setImageView(obsRGBD[0]->intensityImage); // This is not "_fast" since the intensity image may be needed later on.
//                //        }
//                win3D.unlockAccess3DScene();

//                // -------------------------------------------------------
//                //           Create 3D points from RGB+D data
//                //
//                // There are several methods to do this.
//                //  Switch the #if's to select among the options:
//                // See also: http://www.mrpt.org/Generating_3D_point_clouds_from_RGB_D_observations
//                // -------------------------------------------------------
//                {
//                    win3D.get3DSceneAndLock();
//                    //            obsRGBD[0]->project3DPointsFromDepthImageInto(*gl_points, false /* without obs.sensorPose */);
//                    for(unsigned i=0; i < num_sensors; i++)
//                        obs360.rgbd[i].project3DPointsFromDepthImageInto(*gl_points[i], true);
//                    //              obsRGBD[i]->project3DPointsFromDepthImageInto(*gl_points[i], true, &sensorPoses[i]);

//                    win3D.unlockAccess3DScene();
//                }

//                win3D.repaint();

//                mrpt::system::pause();
//            }

            cout << "RegisterDense \n";

            // Forward compositional
            //registerRGBD.swapSourceTarget();
            registerRGBD.setTargetFrame(intensity_trg, depth_trg);
            registerRGBD.setSourceFrame(intensity_src, depth_src);

            registerRGBD.registerRGBD(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePose = registerRGBD.getOptimalPose();
            cout << "registerRGBD \n" << relativePose << endl;

            registerRGBD.registerRGBD(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePose_photo = registerRGBD.getOptimalPose();
            cout << "registerRGBD Photo \n" << relativePose_photo << endl;

            registerRGBD.registerRGBD(Eigen::Matrix4f::Identity(), RegisterDense::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePose_depth = registerRGBD.getOptimalPose();
            cout << "registerRGBD Depth \n" << relativePose_depth << endl;

            cout << "\n\n\n";
            //mrpt::system::pause();

            // Inverse compositional
            registerRGBD.registerRGBD_IC(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePoseIC = registerRGBD.getOptimalPose();
            cout << "registerRGBD IC \n" << relativePoseIC << endl;

            registerRGBD.registerRGBD_IC(Eigen::Matrix4f::Identity(), RegisterDense::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePoseIC_photo = registerRGBD.getOptimalPose();
            cout << "registerRGBD IC Photo \n" << relativePoseIC_photo << endl;

            registerRGBD.registerRGBD_IC(Eigen::Matrix4f::Identity(), RegisterDense::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePoseIC_depth = registerRGBD.getOptimalPose();
            cout << "registerRGBD IC Depth \n" << relativePoseIC_depth << endl;

            mrpt::system::pause();

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


            // Update relativePose
            currentPose = relativePose * currentPose;
            currentPoseIC = currentPoseIC * relativePoseIC;
            //            relativePose2 = relativePose2 * Rt_dense_photo;
        };



//#if SAVE_TRAJECTORY
            // Rt.saveToTextFile(mrpt::format("%s/Rt_%d_%d.txt", path_results.c_str(), frameOrder-1, frameOrder).c_str());
            //          ofstream saveRt;
            //          saveRt.open(mrpt::format("%s/poses/Rt_%d_%d.txt", path_dataset.c_str(), frame-1, frame).c_str());
            //          saveRt << Rt;
            //          saveRt.close();
//#endif

        std::cout << "RAWLOG END " << std::endl;

//        trajectory_raw.close();
//        trajectory_fused_GT.close();

//        delete frame_trg, frame_src;
//        delete frame_trg_fused, frame_src_fused;
    };

};


void print_help(char ** argv)
{
    cout << "\nThis program performs Visual-Range Odometry by direct RGBD registration using a rawlong sequence recorded by Kinect or Asus XPL.\n\n";

    cout << "Usage: " << argv[0] << " <path_to_filename_dataset> <sampling>\n";
    cout << "    <path_to_filename_dataset> the RGBD video sequence" << endl;
    //cout << "    <pathToResults> is the directory where results should be saved" << endl;
    cout << "    <sampling> is the sampling step used in the dataset (e.g. 1: all the frames are used " << endl;
    cout << "                                                              2: each other frame is used " << endl;
    cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}

int main (int argc, char ** argv)
{
    try
    {
        if(argc < 2 || argc > 3 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
        {
            print_help(argv);
            return 0;
        }

        const string RAWLOG_FILENAME = string( argv[1] );
        //string path_results = static_cast<string>(argv[2]);

        int sampling = 1;
        if(argc > 2)
            sampling = atoi(argv[2]);

        cout << "Create OdometryRGBD object\n";
        OdometryRGBD OdometryRGBD;
        OdometryRGBD.run(RAWLOG_FILENAME, sampling);

        cout << " EXIT\n";

        return (0);
    }
    catch (std::exception &e)
    {
        std::cout << "RGBD360 exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        printf("Unspecified exception!!");
        return -1;
    }
}
