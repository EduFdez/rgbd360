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

#include <pcl/registration/gicp.h> //GICP
#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/gicp.h> //GICP

#include <opencv2/opencv.hpp>

#include <FilterPointCloud.h>
#include <DirectRegistration.h>
#include <CloudRGBD_Ext.h>
#include <Frame360_Visualizer.h>

#include <iostream>

#define SAVE_TRAJECTORY 0
#define SAVE_IMAGES 0
#define VISUALIZE_POINT_CLOUD 1

typedef pcl::PointXYZRGBA PointT;
using namespace std;
using namespace mrpt;
using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace mrpt::poses;
using namespace cv;

/*! This class' main function 'run' performs PbMap-based odometry with the input data of a stream of
 *  omnidirectional RGB-D images (from a recorded sequence).
 */
class OdometryRGBD
{
private:

    DirectRegistration registerer;

    // Used to interpolate grountruth poses
    bool groundtruth_ok;
    bool last_groundtruth_ok;

    double last_groundtruth;	//!< Timestamp of the last groundtruth read
    double timestamp_obs;		//!< Timestamp of the last observation
    double last_gt_data[7];		//!< Last ground truth read (x y z qx qy qz w)

    /** Camera poses */
    mrpt::poses::CPose3D cam_pose;		//!< Last camera pose
    mrpt::poses::CPose3D cam_oldpose;	//!< Previous camera pose
    mrpt::poses::CPose3D gt_pose;       //!< Groundtruth camera pose
    mrpt::poses::CPose3D gt_oldpose;	//!< Groundtruth camera previous pose
    mrpt::poses::CPose3D transf;        //!< Transformation between reference systems (TUM RGBD -> edu)

    std::ifstream		f_gt;
    std::ofstream		f_res;

    bool first_pose;
    bool dataset_finished;

public:

    bool bVisualize;
    size_t skip_frames;

    OdometryRGBD() :
        //registerer(DirectRegistration(DirectRegistration::KINECT)),
        first_pose(false),
        dataset_finished(false),
        bVisualize(false),
        skip_frames(0)
    {
        registerer.useSaliency(true);
    };

    bool getGroundtruthPose(const double timestamp_obs)
    {
        //mrpt::poses::CPose3D gt_pose;

        //Exit if there is no groundtruth at this time
        if (last_groundtruth > timestamp_obs)
        {
            cout << "no groundtruth at this time \n";
            groundtruth_ok = false;
            return false;
        }

        //Search the corresponding groundtruth data and interpolate
        bool new_data = 0;
        last_groundtruth_ok = groundtruth_ok;
        double timestamp_gt;
        while (last_groundtruth < timestamp_obs - 0.01)
        {
            f_gt.ignore(100,'\n');
            f_gt >> timestamp_gt;
            last_groundtruth = timestamp_gt;
            new_data = 1;

            if (f_gt.eof())
            {
                dataset_finished = true;
                return false;
            }
        }

        //Read the inmediatly previous groundtruth
        double x0,y0,z0,qx0,qy0,qz0,w0,t0;
        if (new_data == 1)
        {
            f_gt >> x0; f_gt >> y0; f_gt >> z0;
            f_gt >> qx0; f_gt >> qy0; f_gt >> qz0; f_gt >> w0;
        }
        else
        {
            x0 = last_gt_data[0]; y0 = last_gt_data[1]; z0 = last_gt_data[2];
            qx0 = last_gt_data[3]; qy0 = last_gt_data[4]; qz0 = last_gt_data[5]; w0 = last_gt_data[6];
        }

        t0 = last_groundtruth;

        //Read the inmediatly posterior groundtruth
        f_gt.ignore(10,'\n');
        f_gt >> timestamp_gt;
        if (f_gt.eof())
        {
            dataset_finished = true;
            return false;
        }
        last_groundtruth = timestamp_gt;

        //last_gt_data = [x y z qx qy qz w]
        f_gt >> last_gt_data[0]; f_gt >> last_gt_data[1]; f_gt >> last_gt_data[2];
        f_gt >> last_gt_data[3]; f_gt >> last_gt_data[4]; f_gt >> last_gt_data[5]; f_gt >> last_gt_data[6];

        if (last_groundtruth - timestamp_obs > 0.01)
            groundtruth_ok = false;
        else
        {
            gt_oldpose = gt_pose;

            //							Update pose
            //-----------------------------------------------------------------
            const float incr_t0 = timestamp_obs - t0;
            const float incr_t1 = last_groundtruth - timestamp_obs;
            const float incr_t = incr_t0 + incr_t1;

            if (incr_t == 0.f) //Deal with defects in the groundtruth files
            {
                groundtruth_ok = false;
                return false;
            }

            //Sometimes the quaternion sign changes in the groundtruth
            if (abs(qx0 + last_gt_data[3]) + abs(qy0 + last_gt_data[4]) + abs(qz0 + last_gt_data[5]) + abs(w0 + last_gt_data[6]) < 0.05)
            {
                qx0 = -qx0; qy0 = -qy0; qz0 = -qz0; w0 = -w0;
            }

            double x,y,z,qx,qy,qz,w;
            x = (incr_t0*last_gt_data[0] + incr_t1*x0)/(incr_t);
            y = (incr_t0*last_gt_data[1] + incr_t1*y0)/(incr_t);
            z = (incr_t0*last_gt_data[2] + incr_t1*z0)/(incr_t);
            qx = (incr_t0*last_gt_data[3] + incr_t1*qx0)/(incr_t);
            qy = (incr_t0*last_gt_data[4] + incr_t1*qy0)/(incr_t);
            qz = (incr_t0*last_gt_data[5] + incr_t1*qz0)/(incr_t);
            w = (incr_t0*last_gt_data[6] + incr_t1*w0)/(incr_t);

            mrpt::math::CMatrixDouble33 mat;
            mat(0,0) = 1- 2*qy*qy - 2*qz*qz;
            mat(0,1) = 2*(qx*qy - w*qz);
            mat(0,2) = 2*(qx*qz + w*qy);
            mat(1,0) = 2*(qx*qy + w*qz);
            mat(1,1) = 1 - 2*qx*qx - 2*qz*qz;
            mat(1,2) = 2*(qy*qz - w*qx);
            mat(2,0) = 2*(qx*qz - w*qy);
            mat(2,1) = 2*(qy*qz + w*qx);
            mat(2,2) = 1 - 2*qx*qx - 2*qy*qy;

            mrpt::poses::CPose3D gt;
            gt.setFromValues(x,y,z,0,0,0);
            gt.setRotationMatrix(mat);


            //Alternative - directly quaternions
            //vector<float> quat;
            //quat[0] = x, quat[1] = y; quat[2] = z;
            //quat[3] = w, quat[4] = qx; quat[5] = qy; quat[6] = qz;
            //gt.setFromXYZQ(quat);

            //Set the initial pose (if appropiate)
            if (first_pose == false)
            {
                transf.setFromValues(0,0,0,0.5*M_PI, -0.5*M_PI, 0);
                //cout << "transf pose \n" << transf.getHomogeneousMatrixVal() << endl;
                mrpt::poses::CPose3D ref_mrpt_edu;
//                ref_mrpt_edu.setFromValues(0,0,0,M_PI,0,0);
                ref_mrpt_edu.setFromValues(0,0,0,0,-0.5*M_PI,0.5*M_PI);
                cout << "ref_mrpt_edu \n" << ref_mrpt_edu.getHomogeneousMatrixVal() << endl;
                transf = transf - ref_mrpt_edu;

                cam_pose = gt + transf;
                first_pose = true;
            }

            gt_pose = gt + transf;
            groundtruth_ok = 1;
        }

        return true;
    }

    void CreateResultsFile()
    {
        try
        {
            // Open file, find the first free file-name.
            char	aux[100];
            int     nFile = 0;
            bool    free_name = false;

            system::createDirectory("./difodo.results");

            while (!free_name)
            {
                nFile++;
                sprintf(aux, "./difodo.results/experiment_%03u.txt", nFile );
                free_name = !system::fileExists(aux);
            }

            // Open log file:
            f_res.open(aux);
            printf(" Saving results to file: %s \n", aux);
        }
        catch (...)
        {
            printf("Exception found trying to create the 'results file' !!\n");
        }

    }

    void writeTrajectoryFile()
    {
//        //Don't take into account those iterations with consecutive equal depth images
//        if (abs(dt.sumAll()) > 0)
//        {
            mrpt::math::CQuaternionDouble quat;
            CPose3D auxpose, transf;
            transf.setFromValues(0,0,0,0.5*M_PI, -0.5*M_PI, 0);

            auxpose = cam_pose - transf;
            auxpose.getAsQuaternion(quat);

            char aux[24];
            sprintf(aux,"%.04f", timestamp_obs);
            f_res << aux << " ";
            f_res << cam_pose[0] << " ";
            f_res << cam_pose[1] << " ";
            f_res << cam_pose[2] << " ";
            f_res << quat(2) << " ";
            f_res << quat(3) << " ";
            f_res << -quat(1) << " ";
            f_res << -quat(0) << endl;
//        }
    }

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


        //					Load ground-truth
        //=========================================================
        string filename_gt = mrpt::system::extractFileDirectory(filename);
        filename_gt.append("/groundtruth.txt");
        f_gt.open(filename_gt.c_str());

        if (f_gt.fail())
            throw std::runtime_error("\nError finding the groundtruth file: it should be contained in the same folder than the rawlog file");

        char aux[100];
        f_gt.getline(aux, 100);
        f_gt.getline(aux, 100);
        f_gt.getline(aux, 100);
        f_gt >> last_groundtruth;
        f_gt >> last_gt_data[0]; f_gt >> last_gt_data[1]; f_gt >> last_gt_data[2];
        f_gt >> last_gt_data[3]; f_gt >> last_gt_data[4]; f_gt >> last_gt_data[5]; f_gt >> last_gt_data[6];
        last_groundtruth_ok = 1;
        cout << "last_gt_data " << last_gt_data[0] << " " << last_gt_data[1] << " " << last_gt_data[2] << " " << last_gt_data[3] << " "
                                 << last_gt_data[4] << " " << last_gt_data[5] << " " << last_gt_data[6] << endl;

        mrpt::poses::CPose3D gt_firstPose;	//!< Groundtruth camera first pose
        mrpt::poses::CPose3D gt_prevPose;	//!< Groundtruth camera previous pose

//        mrpt::utils::CFileGZInputStream dataset(filename);
//        mrpt::obs::CActionCollectionPtr action;
//        mrpt::obs::CSensoryFramePtr observations;
        mrpt::obs::CObservationPtr observation;
        //bool end = false;

        mrpt::obs::CObservation3DRangeScanPtr obsRGBD;  // The RGBD observation
        //CObservation2DRangeScanPtr laserObs;    // Pointer to the laser observation
        size_t n_RGBD = 0, n_obs = 0;

        bool bGoodRegistration = true;
        Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f currentPose_photo = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f currentPose_depth = Eigen::Matrix4f::Identity();
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
        DirectRegistration registerRGBD; // Dense RGB-D alignment
        registerRGBD.setSensorType( DirectRegistration::KINECT ); // This is use to adapt some features/hacks for each type of image (see the implementation of DirectRegistration::register360 for more details)
        //registerRGBD.setNumPyr(0);
        registerRGBD.setNumPyr(5);
        registerRGBD.setMaxIterations(5);
        registerRGBD.setMaxDepth(8.f);
//        registerRGBD.useSaliency(true);
//        registerRGBD.thresSaliencyIntensity(0.f);
//        registerRGBD.thresSaliencyIntensity(0.f);
//        registerRGBD.setBilinearInterp(true);
        registerRGBD.setVisualization(true);
        registerRGBD.setGrayVariance(8.f/255);

        // Initialize ICP
        pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
        icp.setMaxCorrespondenceDistance (0.4);
        icp.setMaximumIterations (10);
        icp.setTransformationEpsilon (1e-9);
        icp.setRANSACOutlierRejectionThreshold (0.1);
//        pcl::PointCloud<PointT>::Ptr cloud_dense_trg(new pcl::PointCloud<PointT>);
//        pcl::PointCloud<PointT>::Ptr cloud_dense_src(new pcl::PointCloud<PointT>);
        CloudRGBD src, trg;

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

            // Get ground-truth pose
            gt_prevPose = gt_pose;
            timestamp_obs = mrpt::system::timestampTotime_t(obsRGBD->timestamp);
            if( !getGroundtruthPose(timestamp_obs) )
            {
                cout << " SKIP FRAME: no groundtruth pose \n";
                mrpt::system::pause();
                obsRGBD->unload();
                continue;
            }

            cout << "GT pose \n" << gt_pose.getHomogeneousMatrixVal() << endl;

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
                ASSERT_( obsRGBD->hasIntensityImage && obsRGBD->hasRangeImage ); // Check that the images are really RGBD

                intensity_src = cv::Mat(obsRGBD->intensityImage.getAs<IplImage>());
                //intensity_src = cv::cvarrToMat(obsRGBD->intensityImage.getAs<IplImage>());
                convertRange_mrpt2cvMat(obsRGBD->rangeImage, depth_src);
                //registerRGBD.setSourceFrame(intensity_src, depth_src);

                src.setRGBImage(intensity_src);
                src.setDepthImage(depth_src);
                src.getPointCloud();

                gt_firstPose = gt_pose;
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

                trg = src;
                src.setRGBImage(intensity_src);
                src.setDepthImage(depth_src);
                src.getPointCloud();

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

            cout << "DirectRegistration \n";

            // Forward compositional
            //registerRGBD.swapSourceTarget();
            registerRGBD.setTargetFrame(intensity_trg, depth_trg);
            registerRGBD.setSourceFrame(intensity_src, depth_src);

            registerRGBD.regist(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePose = registerRGBD.getOptimalPose();
            cout << "registerRGBD \n" << relativePose << endl;

            registerRGBD.regist(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePose_photo = registerRGBD.getOptimalPose();
            cout << "registerRGBD Photo \n" << relativePose_photo << endl;

            registerRGBD.regist(Eigen::Matrix4f::Identity(), DirectRegistration::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            relativePose_depth = registerRGBD.getOptimalPose();
            cout << "registerRGBD Depth \n" << relativePose_depth << endl;

            cout << "\n\n\n";
            //mrpt::system::pause();

//            // Inverse compositional
//            registerRGBD.registerRGBD_IC(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            relativePoseIC = registerRGBD.getOptimalPose();
//            cout << "registerRGBD IC \n" << relativePoseIC << endl;

//            registerRGBD.registerRGBD_IC(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            relativePoseIC_photo = registerRGBD.getOptimalPose();
//            cout << "registerRGBD IC Photo \n" << relativePoseIC_photo << endl;

//            registerRGBD.registerRGBD_IC(Eigen::Matrix4f::Identity(), DirectRegistration::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            relativePoseIC_depth = registerRGBD.getOptimalPose();
//            cout << "registerRGBD IC Depth \n" << relativePoseIC_depth << endl;


//            registerRGBD.useSaliency(true);
//            registerRGBD.registerRGBD_IC(Eigen::Matrix4f::Identity(), DirectRegistration::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//            cout << "registerRGBD IC Depth Saliency \n" << registerRGBD.getOptimalPose() << endl;

            //mrpt::system::pause();

//            // Visualize point cloud
//            pcl::visualization::CloudViewer cloud_viewer("cloud");
//            cloud_viewer.showCloud (trg.getPointCloud());
//            while (!cloud_viewer.wasStopped ())
//                boost::this_thread::sleep (boost::posix_time::milliseconds (10));

//            cout << "ICP \n";
//            double time_start = pcl::getTime();
////            icp.setInputTarget(trg.getPointCloud());
////            icp.setInputSource(src.getPointCloud());
//            pcl::PointCloud<PointT>::Ptr cloud_dense_trg = trg.getPointCloud();
//            pcl::PointCloud<PointT>::Ptr cloud_dense_src = src.getPointCloud();
//            icp.setInputTarget(cloud_dense_trg);
//            icp.setInputSource(cloud_dense_src);
//            //filter.filterVoxel(frame_src_fused->getSphereCloud(), cloud_dense_src);
//            pcl::PointCloud<PointT>::Ptr cloudAlignedICP(new pcl::PointCloud<PointT>);
//          //  Eigen::Matrix4d initRt = registerer.getPose();
//            Eigen::Matrix4f initRt = Eigen::Matrix4f::Identity();
//            icp.align(*cloudAlignedICP, initRt);

//            double time_end = pcl::getTime();
//            std::cout << "ICP took " << double (time_end - time_start) << std::endl;
            //std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
            //Rt_icp = icp.getFinalTransformation(); //.cast<double>();


            // Update relativePose
            currentPose = currentPose * relativePose;
            currentPose_photo = currentPose_photo * relativePose_photo;
            currentPose_depth = currentPose_depth * relativePose_depth;

            //currentPoseIC = currentPoseIC * relativePoseIC;
            //            relativePose2 = relativePose2 * Rt_dense_photo;

            std::cout << "currentPose \n " << currentPose << std::endl;
            std::cout << "currentPose_photo \n " << currentPose_photo << std::endl;
            std::cout << "currentPose_depth \n " << currentPose_depth << std::endl;

            mrpt::poses::CPose3D path = -gt_firstPose + gt_pose;
            cout << "GT path \n" << path.getHomogeneousMatrixVal() << endl;
            //cout << "GT path " << path << endl;
//            mrpt::poses::CPose3D path_ref0 = path + ref_mrpt_edu;
//            cout << "GT path_ref0 \n" << path_ref0.getHomogeneousMatrixVal() << endl;

            std::cout << "relativePose \n " << relativePose_photo << std::endl;

            mrpt::poses::CPose3D pose_diff = -gt_prevPose + gt_pose ;
            cout << "GT pose increment \n" << pose_diff.getHomogeneousMatrixVal() << endl;
            //cout << "NUmericalErrors pose_diff 2 \n" << gt_oldpose.getHomogeneousMatrixVal().inverse() * gt_pose.getHomogeneousMatrixVal() << endl;

            mrpt::system::pause();

//            writeTrajectoryFile();

        };


//        CreateResultsFile();
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
