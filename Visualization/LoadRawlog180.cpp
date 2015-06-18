/*
 *  Copyright (c) 2015,   INRIA Sophia Antipolis - LAGADIC Team
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
 * Author: Eduardo Fernandez-Moral
 */

//#include <definitions.h>

#include <mrpt/poses/CPose3D.h>
//#include <mrpt/utils/CFileGZInputStream.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/pbmap/PbMap.h>
#include <mrpt/system/os.h>

//#include <opencv/cv.h>
//#include <opencv2/features2d.hpp>
//#include <opencv2/line_descriptor.hpp>
//using namespace cv;
//using namespace cv::line_descriptor;

//#include <eigen3/Eigen/Dense>
//#include <eigen3/Eigen/Eigenvalues>
//using namespace Eigen;

#include <pcl/console/parse.h>

#include <Calibrator.h>
#include <Frame360_Visualizer.h>

#define VISUALIZE_SENSOR_DATA 1

using namespace std;
using namespace mrpt::obs;
using namespace mrpt::utils;

void print_help(char ** argv)
{
  cout << "\nThis program shows the pointclouds constructed from a RGBD180 rawlog dataset";

  cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> \n";
  cout << "    <pathPlaneCorrespondences> is the directory containing the data stream as a set of '.bin' files" << endl << endl;
  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}


int main (int argc, char ** argv)
{
    if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
        print_help(argv);

    string filename;
    if(argc > 1)
        filename = static_cast<string>(argv[1]);

    const size_t decimation = 2;
    const size_t skip_frames = 10;

    //mrpt::poses::CPose3D a(1.0,2.0,3.0,DEG2RAD(10),DEG2RAD(50),DEG2RAD(-30));

    cout << "Calibrate RGBD360 multisensor\n";
    Calib360 calib;
    float angle_offset = 22.5; //45
    Eigen::Matrix4f rot_offset = Eigen::Matrix4f::Identity(); rot_offset(1,1) = rot_offset(2,2) = cos(angle_offset*PI/180); rot_offset(1,2) = -sin(angle_offset*PI/180); rot_offset(2,1) = -rot_offset(1,2);
    // Load initial calibration
    cout << "Load initial calibration\n";
//    mrpt::poses::CPose3D pose[NUM_ASUS_SENSORS];
//    pose[0] = mrpt::poses::CPose3D(0.285, 0, 1.015, DEG2RAD(0), DEG2RAD(1.3), DEG2RAD(-90));
//    pose[1] = mrpt::poses::CPose3D(0.271, -0.031, 1.015, DEG2RAD(-45), DEG2RAD(0), DEG2RAD(-90));
//    pose[2] = mrpt::poses::CPose3D(0.271, 0.031, 1.125, DEG2RAD(45), DEG2RAD(2), DEG2RAD(-89));
//    pose[3] = mrpt::poses::CPose3D(0.24, -0.045, 0.975, DEG2RAD(-90), DEG2RAD(1.5), DEG2RAD(-90));
    int rgbd180_arrangement[4] = {1,8,2,7};
    for(size_t sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
    {
//        Eigen::Matrix4f pose_mat = getPoseEigenMatrix( pose[sensor_id] ); //.inverse();
//        calib.setRt_id( sensor_id, pose_mat);
        calib.Rt_[sensor_id].loadFromTextFile( mrpt::format("/home/efernand/Libraries/rgbd360/Calibration/test/Rt_0%i.txt",sensor_id+1) );
        //cout << sensor_id << " sensor pose\n" << calib.Rt_[sensor_id] << endl;
        //calib.Rt_[sensor_id] = rot_offset * calib.Rt_[sensor_id];
    }

    cout << "Init ControlPlanes \n";
    ControlPlanes matches;
    for(unsigned sensor_id1=0; sensor_id1 < NUM_ASUS_SENSORS; sensor_id1++)
    {
        matches.mmCorrespondences[sensor_id1] = std::map<unsigned, mrpt::math::CMatrixDouble>();
        for(unsigned sensor_id2=sensor_id1+1; sensor_id2 < NUM_ASUS_SENSORS; sensor_id2++)
        {
            matches.mmCorrespondences[sensor_id1][sensor_id2] = mrpt::math::CMatrixDouble(0, 10);
        }
    }

    cout << "Open CRawlog \n";
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
    mrpt::obs::CObservationPtr observation;

    mrpt::obs::CObservation3DRangeScanPtr obsRGBD[NUM_ASUS_SENSORS];  // The RGBD observation
    bool obs_sensor[NUM_ASUS_SENSORS];
    obs_sensor[0] = false, obs_sensor[1] = false, obs_sensor[2] = false, obs_sensor[3] = false;
    //CObservation2DRangeScanPtr laserObs;    // Pointer to the laser observation
    size_t n_obs = 0, frame = 0;

    while ( n_obs < dataset.size() )
    {
        observation = dataset.getAsObservation(n_obs);
        //cout << n_obs << " observation: " << observation->sensorLabel << ". Timestamp " << observation->timestamp << endl;
        ++n_obs;
        if(!IS_CLASS(observation, CObservation3DRangeScan))
        {
            continue;
        }

        size_t sensor_id = 0;
        if(observation->sensorLabel == "RGBD_1")
        {
            sensor_id = 0;
        }
        else
        if(observation->sensorLabel == "RGBD_2")
        {
            sensor_id = 1;
        }
        else if(observation->sensorLabel == "RGBD_3")
        {
            sensor_id = 2;
        }
        else if(observation->sensorLabel == "RGBD_4")
        {
            sensor_id = 3;
        }
        else // This RGBD sensor is not taken into account for calibration
            continue;

        obs_sensor[sensor_id] = true;

        obsRGBD[sensor_id] = mrpt::obs::CObservation3DRangeScanPtr(observation);
        obsRGBD[sensor_id]->load();

        if(obs_sensor[0] && obs_sensor[1] && obs_sensor[2] && obs_sensor[3])
        {
            ++frame;
            obs_sensor[0] = false, obs_sensor[1] = false, obs_sensor[2] = false, obs_sensor[3] = false;

            // Apply decimation
            if( frame < skip_frames )
                continue;
            if( frame % decimation != 0)
                continue;

            cout << " Process frame: " << frame << endl;

//          CloudRGBD_Ext cloud[NUM_ASUS_SENSORS];
//          cloud[sensor_id].setRGBImage( cv::Mat(obsRGBD[sensor_id]->intensityImage.getAs<IplImage>()) );
//          cv::Mat depth_mat;
//          convertRange_mrpt2cvMat(obsRGBD[sensor_id]->rangeImage, depth_mat);
//          cloud[sensor_id].setDepthImage(depth_mat);
//          cloud[sensor_id].getPointCloud();

            Frame360 frame360(&calib);
            frame360.setTimeStamp(obsRGBD[0]->timestamp);

            #if ENABLE_OPENMP
                #pragma omp parallel num_threads(NUM_ASUS_SENSORS)
                {
                    int sensor_id = omp_get_thread_num();
            #else
                for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
                {
            #endif
                    frame360.frameRGBD_[sensor_id].setRGBImage( cv::Mat(obsRGBD[sensor_id]->intensityImage.getAs<IplImage>()) );
                    cv::Mat depth_mat;
                    convertRange_mrpt2cvMat(obsRGBD[sensor_id]->rangeImage, depth_mat);
                    frame360.frameRGBD_[sensor_id].setDepthImage(depth_mat);
                    frame360.frameRGBD_[sensor_id].loadDepthEigen();
                    frame360.frameRGBD_[sensor_id].getPointCloud();
                    //cout << "width " << frame360.frameRGBD_[sensor_id].getPointCloud()->width << " height " << frame360.frameRGBD_[sensor_id].getPointCloud()->height << endl;

//                    string img_ = mrpt::format("img_%d",sensor_id);
//                    string depth_ = mrpt::format("depth_%d",sensor_id);
//                    cv::imshow( img_, frame360.frameRGBD_[sensor_id].getRGBImage( ) );
//                    cv::imshow( depth_, frame360.frameRGBD_[sensor_id].getDepthImage( ) );
                }

//                for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
//                {
//                    string cloud_ = mrpt::format("cloud_%d",sensor_id);
//                    pcl::visualization::CloudViewer cloud_viewer(cloud_.c_str());
//                    cloud_viewer.showCloud (frame360.frameRGBD_[sensor_id].getPointCloud());
//                    while (!cloud_viewer.wasStopped ())
//                        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
//                }
//              pcl::visualization::CloudViewer cloud_viewer("cloud");
//              cloud_viewer.showCloud (frame360.frameRGBD_[0].getPointCloud());

//              frame360.stitchSphericalImage();
//              // Visualize spherical image
//              cv::imshow( "sphereRGB", frame360.sphereRGB );

//              while (cv::waitKey(1)!='\n')
//                boost::this_thread::sleep (boost::posix_time::milliseconds (10));


            // Visualize cloud
            frame360.buildSphereCloud_rgbd360();
            frame360.getPlanes();
            Frame360_Visualizer sphereViewer(&frame360);
            while (!sphereViewer.viewer.wasStopped() )
              boost::this_thread::sleep (boost::posix_time::milliseconds (10));

        }
    }

  cout << "EXIT\n";

  return (0);
}
