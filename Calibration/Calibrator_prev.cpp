/*
 *  Copyright (c) 2012, Universidad de Málaga - Grupo MAPIR
 *
 *  http://code.google.com/p/photoconsistency-visual-odometry/
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

#include <mrpt/base.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h> //Save global map as PCD file
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>

#include <FrameRGBD.h>
#include <RGBDGrabber.h>
#include <RGBDGrabberOpenNI_PCL.h>

#include <Eigen/Core>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define PI 3.14159
#define VISUALIZE_POINT_CLOUD 0

using namespace std;

typedef pcl::PointXYZRGBA PointT;

Eigen::Matrix3d skew(Eigen::Vector3d vec)
{
  Eigen::Matrix3d skew_matrix = Eigen::Matrix3d::Zero();
  skew_matrix(0,1) = -vec(2);
  skew_matrix(1,2) = vec(2);
  skew_matrix(0,2) = vec(1);
  skew_matrix(2,0) = -vec(1);
  skew_matrix(1,2) = -vec(0);
  skew_matrix(2,1) = vec(0);
  return skew_matrix;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Calibrator
{
  public:
    Calibrator()
    // :
//      successful(false)
    {
//      string mouseMsg2D ("Mouse coordinates in image viewer");
//      string keyMsg2D ("Key event for image viewer");
//      viewer.registerKeyboardCallback(&RGBD360_Visualizer::keyboard_callback, *this, static_cast<void*> (&keyMsg2D));
    }

    void loadCalibrationMatrices()
    {
      Rt_estimated.resize(8);
      Rt_01 = Eigen::Matrix4f::Identity();
      Rt_01(2,3) = 0.05; // Set the origin of the sphere to the approximate mass center of optical centers
      Rt_10 = Rt_01.inverse();
      Rt_estimated[0] = Rt_01;

//        string calibFile12 = "/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_12.txt";
      mrpt::math::CMatrixDouble44 calibMatrix;

      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_12.txt");
      Rt_12 <<  calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;
      Rt_02 = Rt_01 * Rt_12;
      Rt_20 = Rt_02.inverse();
      Rt_estimated[1] = Rt_02;

//        cout << "calib:\n" << Rt_10<< endl;
      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_23.txt");
      Rt_23 << calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;
      Rt_03 = Rt_02 * Rt_23;
      Rt_30 = Rt_03.inverse();
      Rt_estimated[2] = Rt_03;

      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_34.txt");
      Rt_34 << calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;
      Rt_04 = Rt_03 * Rt_34;
      Rt_40 = Rt_04.inverse();
      Rt_estimated[3] = Rt_04;

      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_45.txt");
      Rt_45 << calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;
      Rt_05 = Rt_04 * Rt_45;
      Rt_50 = Rt_05.inverse();
      Rt_estimated[4] = Rt_05;

      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_56.txt");
      Rt_56 << calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;
      Rt_06 = Rt_05 * Rt_56;
      Rt_60 = Rt_06.inverse();
      Rt_estimated[5] = Rt_06;

      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_67.txt");
      Rt_67 << calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;
      Rt_07 = Rt_06 * Rt_67;
      Rt_70 = Rt_07.inverse();
      Rt_estimated[6] = Rt_07;

      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_78.txt");
      Rt_78 << calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;
      Rt_08 = Rt_07 * Rt_78;
      Rt_80 = Rt_08.inverse();
      Rt_estimated[7] = Rt_08;

      calibMatrix.loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/calibratedRt_81.txt");
      Rt_81 << calibMatrix(0,0), calibMatrix(0,1), calibMatrix(0,2), calibMatrix(0,3),
                calibMatrix(1,0), calibMatrix(1,1), calibMatrix(1,2), calibMatrix(1,3),
                calibMatrix(2,0), calibMatrix(2,1), calibMatrix(2,2), calibMatrix(2,3),
                0,0,0,1;

      cout << "Rt loop\n" << Rt_08 * Rt_81 << endl;

      // Load correspondences for calibration with loop closure
      correspondences.resize(8);
      correspondences[0].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_12.txt");
      correspondences[1].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_23.txt");
      correspondences[2].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_34.txt");
      correspondences[3].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_45.txt");
      correspondences[4].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_56.txt");
      correspondences[5].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_67.txt");
      correspondences[6].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_78.txt");
      correspondences[7].loadFromTextFile("/home/eduardo/Dropbox/Doctorado/Projects/RGBD360/Calibrator/correspondences_81.txt");
//      cout << "correspondences_12\n" << correspondences_12 << endl;
    }

    // Find the Rt of each sensor in the multisensor RGBD360 setup
    void Calibrate()
    {
      Eigen::Matrix<double,21,21> hessian = Eigen::Matrix<double,21,21>::Zero();
      Eigen::Matrix<double,21,1> gradient = Eigen::Matrix<double,21,1>::Zero();
      Eigen::Matrix<double,21,1> solution;
      Eigen::Matrix3d jacobian_rot;
      Eigen::Vector3d rot_error;
      double accum_rot_error = 0;
      Rt_estimated_temp.resize(8);
    cout << "Calibrate()...\n";
//      // For the fixed camera:
//      unsigned sensor_id = 0;
//      for(unsigned i=0; i < correspondences[sensor_id].getRowCount(); i++)
//      {
//        Eigen::Vector3d n_i = Rt_estimated[sensor_id].block(0,0,3,3) * correspondences[sensor_id].block(i,0,1,3).transpose();
//        Eigen::Vector3d n_ii = Rt_estimated[sensor_id+1].block(0,0,3,3) * correspondences[sensor_id].block(i,4,1,3).transpose();
//        jacobian_rot_ii = skew(n_ii);
//        rot_error = n_i - n_ii;
//        accum_rot_error += rot_error.dot(rot_error);
//        hessian.block(0,0,3,3) += jacobian_rot_ii.transpose() * jacobian_rot_ii;
//        gradient.block(0,0,3,1) += jacobian_rot_ii.transpose() * rot_error;
//      }
//
//      for(sensor_id = 1; sensor_id < 7; sensor_id++)
//      {
//        for(unsigned i=0; i < correspondences[sensor_id].getRowCount(); i++)
//        {
//          Eigen::Vector3d n_i = Rt_estimated[sensor_id].block(0,0,3,3) * correspondences[sensor_id].block(i,0,1,3).transpose();
//          Eigen::Vector3d n_ii = Rt_estimated[sensor_id+1].block(0,0,3,3) * correspondences[sensor_id].block(i,4,1,3).transpose();
//          jacobian_rot_i = skew(-n_i);
//          jacobian_rot_ii = skew(n_ii);
//          rot_error = n_i - n_ii;
//          accum_rot_error += rot_error.dot(rot_error);
//          hessian.block(3*(sensor_id-1), 3*(sensor_id-1), 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_i;
//          gradient.block(3*(sensor_id-1),0,3,1) += jacobian_rot_i.transpose() * rot_error;
//          hessian.block(3*sensor_id, 3*sensor_id, 3, 3) += jacobian_rot_ii.transpose() * jacobian_rot_ii;
//          gradient.block(3*sensor_id,0,3,1) += jacobian_rot_ii.transpose() * rot_error;
//          // Cross term
//          hessian.block(3*(sensor_id-1), 3*sensor_id, 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_ii;
//        }
//      }
//      // Fill the lower left triangle with the corresponding cross terms
//      for(unsigned sensor_id = 1; sensor_id < 7; sensor_id++)
//        hessian.block(3*sensor_id, 3*(sensor_id-1), 3, 3) = hessian.block(3*(sensor_id-1), 3*sensor_id, 3, 3).transpose();
//
//      // For the loop closure constraint
//      for(unsigned i=0; i < correspondences[sensor_id].getRowCount(); i++)
//      {
//        Eigen::Vector3d n_i = Rt_estimated[sensor_id].block(0,0,3,3) * correspondences[sensor_id].block(i,0,1,3).transpose();
//        Eigen::Vector3d n_ii = Rt_estimated[sensor_id+1].block(0,0,3,3) * correspondences[sensor_id].block(i,4,1,3).transpose();
//        jacobian_rot_i = skew(-n_i);
//        rot_error = n_i - n_ii;
//        accum_rot_error += rot_error.dot(rot_error);
//        hessian.block(18,18,3,3) += jacobian_rot_i.transpose() * jacobian_rot_i;
//        gradient.block(18,0,3,1) += jacobian_rot_i.transpose() * rot_error;
//      }
//      cout << "Hessian\n" << hessian << endl;
//      cout << "Error accumulated " << accum_rot_error;
//
//      // Solve for rotation
//      solution = hessian.inverse() * gradient;
//      cout << "solution " << solution.transpose() << endl;
//
//      // Update rotation of the poses
//      mrpt::poses::CPose3D pose;
//      mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
//      rot_manifold[0] = solution(0,0);
//      rot_manifold[1] = solution(0,1);
//      rot_manifold[2] = solution(0,2);
//      mrpt::math::CMatrixDouble33 rotation_temp = pose.exp_rotation(rot_manifold);
//      Rt_estimated_temp[1] << rotation_temp(0,0), rotation_temp(0,1), rotation_temp(0,2), 0,
//                              rotation_temp(1,0), rotation_temp(1,1), rotation_temp(1,2), 0,
//                              rotation_temp(2,0), rotation_temp(2,1), rotation_temp(2,2), 0,
//                              0, 0, 0, 1;
//      cout << "old rotation\n" << Rt_estimated[1].block(0,0,3,3) << endl;
//      cout << "new rotation\n" << Rt_estimated_temp[1].block(0,0,3,3) << endl;

    }


    #if VISUALIZE_POINT_CLOUD
    void buildSphere2()
    {
      {
      boost::mutex::scoped_lock updateLock(visualizationMutex);

        cloud1.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD1->getPointCloud(cameraMatrix),*cloud1,Rt_01);

        cloud2.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD2->getPointCloud(cameraMatrix),*cloud2,Rt_02);

        cloud3.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD3->getPointCloud(cameraMatrix),*cloud3,Rt_03);

        cloud4.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD4->getPointCloud(cameraMatrix),*cloud4,Rt_04);

        cloud5.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD5->getPointCloud(cameraMatrix),*cloud5,Rt_05);

        cloud6.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD6->getPointCloud(cameraMatrix),*cloud6,Rt_06);

        cloud7.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD7->getPointCloud(cameraMatrix),*cloud7,Rt_07);

        cloud8.reset(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*frameRGBD8->getPointCloud(cameraMatrix),*cloud8,Rt_08);
      }
    }
    #endif

    void run()
    {
      // Get the calibration matrices for the different sensors
      loadCalibrationMatrices();

      // Calibrate the omnidirectional RGBD multisensor
      Calibrate();

      #if VISUALIZE_POINT_CLOUD
      openni_wrapper::OpenNIDriver& driver = openni_wrapper::OpenNIDriver::getInstance();
//      if (driver.getNumberDevices () >= 2)
      {
        // Show a list of connected devices
        cout << "Num of devices connected: " << driver.getNumberDevices() << endl;
        for (unsigned deviceIdx = 0; deviceIdx < driver.getNumberDevices(); ++deviceIdx)
        {
          cout << "Device: " << deviceIdx + 1 << ", vendor: " << driver.getVendorName (deviceIdx) << ", product: " << driver.getProductName (deviceIdx) << endl;
    //        << ", connected: " << driver.getBus (deviceIdx) << " @ " << driver.getAddress (deviceIdx) << ", serial number: \'" << driver.getSerialNumber (deviceIdx) << "\'" << endl;
        }

        // Acces devices
        pcl::OpenNIGrabber::Mode image_mode = pcl::OpenNIGrabber::OpenNI_QVGA_30Hz;  //pcl::OpenNIGrabber::OpenNI_Default_Mode;

        RGBDGrabber *grabber1(new RGBDGrabberOpenNI_PCL("#3", image_mode));
        RGBDGrabber *grabber2(new RGBDGrabberOpenNI_PCL("#2", image_mode));
        RGBDGrabber *grabber3(new RGBDGrabberOpenNI_PCL("#1", image_mode));
        RGBDGrabber *grabber4(new RGBDGrabberOpenNI_PCL("#4", image_mode));
        RGBDGrabber *grabber5(new RGBDGrabberOpenNI_PCL("#4", image_mode));
        RGBDGrabber *grabber6(new RGBDGrabberOpenNI_PCL("#4", image_mode));
        RGBDGrabber *grabber7(new RGBDGrabberOpenNI_PCL("#4", image_mode));
        RGBDGrabber *grabber8(new RGBDGrabberOpenNI_PCL("#4", image_mode));
//        grabber = new RGBDGrabberOpenNI_PCL("#1", image_mode);
        grabber1->init();
        grabber2->init();
        grabber3->init();
        grabber4->init();
        grabber5->init();
        grabber6->init();
        grabber7->init();
        grabber8->init();
        cout << "Grabber initialized\n";

        frameRGBD1 = new FrameRGBD();
        frameRGBD2 = new FrameRGBD();
        frameRGBD3 = new FrameRGBD();
        frameRGBD4 = new FrameRGBD();
        frameRGBD5 = new FrameRGBD();
        frameRGBD6 = new FrameRGBD();
        frameRGBD7 = new FrameRGBD();
        frameRGBD8 = new FrameRGBD();
//        FrameRGBD *frameRGBD1(new FrameRGBD());
//        FrameRGBD *frameRGBD2(new FrameRGBD());
//        FrameRGBD *frameRGBD3(new FrameRGBD());
//        FrameRGBD *frameRGBD4(new FrameRGBD());

//        cameraMatrix << 525., 0., 3.1950000000000000e+02,
//                        0., 525., 2.3950000000000000e+02,
//                        0., 0., 1.;
        cameraMatrix << 262.5, 0., 1.5950000000000000e+02,
                        0., 262.5, 1.1950000000000000e+02,
                        0., 0., 1.;

//        // Test saving in pcd
        grabber1->grab(frameRGBD1);
//        pcl::io::savePCDFile("/home/eduardo/cloud1.pcd", *frameRGBD1->getPointCloud(cameraMatrix) );

        // Initialize visualizer
        pcl::visualization::CloudViewer viewer("RGBD360");
        viewer.runOnVisualizationThread (boost::bind(&RGBD360_Visualizer::viz_cb, this, _1), "viz_cb");

        int width = frameRGBD1->getRGBImage().rows*8;
        int height = width/2;
        int height_useful = height * 61.0/180; // Store only the part of the sphere which contains information
        angle_pixel = width/(2*PI);

//        sphereCloud->points.resize(width*height_useful);
//        sphereCloud->is_dense = false;
//        sphereCloud->width = height_useful;
//        sphereCloud->height = width;

      cout << "Sphere size " << height << "x" << width << "\n";
        unsigned frame = 0;
        while (cv::waitKey(1)!='\n')
//        while (!viewer.wasStopped() )
        {
          double frame_start = pcl::getTime ();

          grabber1->grab(frameRGBD1);
          grabber2->grab(frameRGBD2);
          grabber3->grab(frameRGBD3);
          grabber4->grab(frameRGBD4);
          grabber5->grab(frameRGBD5);
          grabber6->grab(frameRGBD6);
          grabber7->grab(frameRGBD7);
          grabber8->grab(frameRGBD8);

          buildSphere2();

          double frame_end = pcl::getTime ();
          cout << "Grabbing in " << (frame_end - frame_start)*1e3 << " ms\n";

          boost::this_thread::sleep (boost::posix_time::microseconds(5000));
        }

        grabber1->stop();
        grabber2->stop();
        grabber3->stop();
        grabber4->stop();
        grabber5->stop();
        grabber6->stop();
        grabber7->stop();
        grabber8->stop();
      }
//      else
//        cout << "Less than two devices connected: at least two RGB-D sensors are required to perform extrinsic calibration.\n";

      #endif
    }

  private:

    Eigen::Matrix3f cameraMatrix;
    Eigen::Matrix4f Rt_12, Rt_23, Rt_34, Rt_45, Rt_56, Rt_67, Rt_78, Rt_81, Rt_calib, turn45deg;
    Eigen::Matrix4f Rt_01, Rt_02, Rt_03, Rt_04, Rt_05, Rt_06, Rt_07, Rt_08;
    Eigen::Matrix4f Rt_10, Rt_20, Rt_30, Rt_40, Rt_50, Rt_60, Rt_70, Rt_80;

    vector<Eigen::Matrix4f> Rt_estimated, Rt_estimated_temp;
    vector<mrpt::math::CMatrixDouble> correspondences;

  #if VISUALIZE_POINT_CLOUD
    boost::mutex visualizationMutex;

//    /*!Thread handles for parallel image stitching*/
    mrpt::system::TThreadHandle stitching1_hd, stitching2_hd, stitching3_hd, stitching4_hd, stitching5_hd, stitching6_hd, stitching7_hd, stitching8_hd;
    mrpt::system::TThreadHandle buildCloud1_hd, buildCloud2_hd, buildCloud3_hd, buildCloud4_hd, buildCloud5_hd, buildCloud6_hd, buildCloud7_hd, buildCloud8_hd;
    bool stitch_im1, stitch_im2, stitch_im3, stitch_im4, stitch_im5, stitch_im6, stitch_im7, stitch_im8;
    bool build_cloud1, build_cloud2, build_cloud3, build_cloud4, build_cloud5, build_cloud6, build_cloud7, build_cloud8;

    FrameRGBD *frameRGBD1, *frameRGBD2, *frameRGBD3, *frameRGBD4, *frameRGBD5, *frameRGBD6, *frameRGBD7, *frameRGBD8;

    pcl::PointCloud<PointT>::Ptr sphereCloud;
    pcl::PointCloud<PointT>::Ptr cloud1, cloud2, cloud3, cloud4, cloud5, cloud6, cloud7, cloud8;

    cv::Mat sphereRGB;
    cv::Mat sphereDepth;
    float angle_pixel;

    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
//    cout << "PbMapMaker::viz_cb(...)\n";
      if (cloud1->empty())
      {
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
        return;
      }

      {
//        boost::mutex::scoped_lock lock (viz_mutex);

//        viz.removeAllShapes();
        viz.removeAllPointClouds();

        { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
          boost::mutex::scoped_lock updateLock(visualizationMutex);

//          if (!viz.updatePointCloud (cloud, "sphereCloud"))
//            viz.addPointCloud (sphereCloud, "sphereCloud");

          if (!viz.updatePointCloud (cloud1, "cloud1"))
            viz.addPointCloud (cloud1, "cloud1");

          if (!viz.updatePointCloud (cloud2, "cloud2"))
            viz.addPointCloud (cloud2, "cloud2");

          if (!viz.updatePointCloud (cloud3, "cloud3"))
            viz.addPointCloud (cloud3, "cloud3");

          if (!viz.updatePointCloud (cloud4, "cloud4"))
            viz.addPointCloud (cloud4, "cloud4");

          if (!viz.updatePointCloud (cloud5, "cloud5"))
            viz.addPointCloud (cloud5, "cloud5");

          if (!viz.updatePointCloud (cloud6, "cloud6"))
            viz.addPointCloud (cloud6, "cloud6");

          if (!viz.updatePointCloud (cloud7, "cloud7"))
            viz.addPointCloud (cloud7, "cloud7");

          if (!viz.updatePointCloud (cloud8, "cloud8"))
            viz.addPointCloud (cloud8, "cloud8");
        updateLock.unlock();
        }
      }
    }
    #endif

//  void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
//  {
//    if ( event.keyDown () )
//    {
//      if(event.getKeySym () == "s" || event.getKeySym () == "S")
//        bDoCalibration = true;
//      else
//        bTakeSample = true;
//    }
//  }

};


//void print_help(char ** argv)
//{
//  cout << "\nThis program calibrates the Rt between two RGB-D sensors mounted rigidly (wrt the RGB optical centers)"
//        << "by acquiring observations of a dominant plane from different views\n";
//  cout << "usage: " << argv[0] << " [options] \n";
//  cout << argv[0] << " -h | --help : shows this help" << endl;
//  cout << argv[0] << " -s | --save <pathToCalibrationFile>" << endl;
//}


int main (int argc, char ** argv)
{
//  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
//    print_help(argv);

cout << "Calibrate RGBD360 multisensor\n";
  Calibrator calibrator;
  calibrator.run();

cout << "EXIT\n";
  return (0);
}
