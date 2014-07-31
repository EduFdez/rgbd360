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

#include <Frame360.h>
#include <Frame360_Visualizer.h>
#include <RegisterRGBD360.h>

#include <RGBDGrabber_OpenNI2.h>
#include <SerializeFrameRGBD.h> // For time-stamp conversion

#include <pcl/console/parse.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <signal.h>

using namespace std;

RGBDGrabber_OpenNI2 *grabber[8]; // This object is declared as global to be able to use it after capturing Ctrl-C interruptions

/*! Catch interruptions like Ctrl-C */
void INThandler(int sig)
{
  char c;

  signal(sig, SIG_IGN);
  printf("\n  Do you really want to quit? [y/n] ");
  c = getchar();
  if (c == 'y' || c == 'Y')
  {
    for(unsigned sensor_id = 0; sensor_id < 8; sensor_id++)
    {
      delete grabber[sensor_id]; // Turn off each Asus XPL sensor before exiting the program
    }
    exit(0);
  }
}


/*! This class calibrates the extrinsic parameters of the omnidirectional RGB-D sensor. For that, the sensor is accessed
 *  and big planes are segmented and matched between different single sensors.
*/
class OnlineCalibration
{
  public:
    OnlineCalibration() :
              bTakeKeyframe(false),
              bFreezeFrame(false),
              sphere_cloud(new pcl::PointCloud<PointT>)
    {
      correspondences.resize(8); // 8 pairs of RGBD sensors
      correspondences_2.resize(8); // 8 pairs of RGBD sensors
      std::fill(conditioning, conditioning+8, 9999.9);
      std::fill(weight_pair, weight_pair+8, 0.0);
      std::fill(conditioning_measCount, conditioning_measCount+8, 0);
      std::fill(covariances, covariances+8, Eigen::Matrix3f::Zero());
      calib.loadIntrinsicCalibration();
//      calib.loadExtrinsicCalibration();
    }

    /*! Upd */
    void updateConditioning(unsigned couple_id, pair< Eigen::Vector4f, Eigen::Vector4f> &match)
    {
      ++conditioning_measCount[couple_id];
      covariances[couple_id] += match.second.head(3) * match.first.head(3).transpose();
//      Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariances[couple_id], Eigen::ComputeFullU | Eigen::ComputeFullV);
//      conditioning[couple_id] = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
    }

    void calcConditioning(unsigned couple_id)
    {
//      if(conditioning_measCount[couple_id] > 3)
//      {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariances[couple_id], Eigen::ComputeFullU | Eigen::ComputeFullV);
        conditioning[couple_id] = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//      }
    }

   // Find the Pose of each sensor in the multisensor RGBD360 setup. This calibration is performed for the rotation of
   // each Asus sensor with respect to its theoretical pose in the omnidirectional setup. The translation given by the
   // construction specifications is assumed to be good enough and it is not optimized here
    void Calibrate()
    {
    cout << "OnlineCalibration::Calibrate...\n";
      Eigen::Matrix<float,21,21> hessian;
      Eigen::Matrix<float,21,1> gradient;
      Eigen::Matrix<float,21,1> update_vector;
      Eigen::Matrix<float,21,1> update_translation;
      Eigen::Matrix3f jacobian_rot_i, jacobian_rot_ii;
      Eigen::Matrix<float,1,3> jacobian_trans_i, jacobian_trans_ii;
      Eigen::Vector3f rot_error;
      float trans_error;
      float accum_error2;
      Eigen::Matrix4f Rt_estimated_temp[8];
      float increment = 1000, diff_error = 1000;
      int it = 0;

      unsigned sensor_id = 0;
      for(sensor_id = 0; sensor_id < 8; sensor_id++)
        Rt_estimated[sensor_id] = calib.getRt_id(sensor_id);
      Rt_estimated_temp[0] = Rt_estimated[0];
//        cout << "Rt_estimated_temp[8] " << Rt_estimated_temp[0] << endl;

      while(it < 50 && increment > 0.0001 && diff_error > 0.000001)
      {
        hessian = Eigen::Matrix<float,21,21>::Zero();
        gradient = Eigen::Matrix<float,21,1>::Zero();
        accum_error2 = 0.0;
        for(sensor_id = 0; sensor_id < 8; sensor_id++)
        {
//        cout << "sensor_id " << sensor_id << endl;
//        cout << "Rt+1 " << sensor_id << endl;
//        cout << correspondences[sensor_id] << endl;
          for(unsigned i=0; i < correspondences[sensor_id].size(); i++)
          {
            Eigen::Vector3f n_i = (Rt_estimated[sensor_id].block(0,0,3,3) * correspondences[sensor_id][i].first.head(3)) / weight_pair[sensor_id];// / correspondences[sensor_id].size();
            Eigen::Vector3f n_ii = (Rt_estimated[(sensor_id+1)%8].block(0,0,3,3) * correspondences[sensor_id][i].second.head(3)) / weight_pair[sensor_id];// / correspondences[sensor_id].size();
            jacobian_rot_i = skew(-n_i);
            jacobian_rot_ii = skew(n_ii);
            rot_error = (n_i - n_ii);
//          cout << "rotation error_i " << rot_error.transpose() << endl;
            accum_error2 += rot_error.dot(rot_error);

            if(sensor_id != 0)
            {
              hessian.block(3*(sensor_id-1), 3*(sensor_id-1), 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_i;
              gradient.block(3*(sensor_id-1),0,3,1) += jacobian_rot_i.transpose() * rot_error;
            }

            if(sensor_id != 7)
            {
              hessian.block(3*sensor_id, 3*sensor_id, 3, 3) += jacobian_rot_ii.transpose() * jacobian_rot_ii;
              gradient.block(3*sensor_id,0,3,1) += jacobian_rot_ii.transpose() * rot_error;

              // Cross term
              if(sensor_id != 0)
                hessian.block(3*(sensor_id-1), 3*sensor_id, 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_ii;
            }
          }
//          // Next to adjacent sensor
//          for(unsigned i=0; i < correspondences_2[sensor_id].size(); i++)
//          {
//            Eigen::Vector3f n_i = (Rt_estimated[sensor_id].block(0,0,3,3) * correspondences_2[sensor_id][i].first.head(3)) / weight_pair_2[sensor_id];// / correspondences_2[sensor_id].size();
//            Eigen::Vector3f n_ii = (Rt_estimated[(sensor_id+1)%8].block(0,0,3,3) * correspondences_2[sensor_id][i].second.head(3)) / weight_pair_2[sensor_id];// / correspondences_2[sensor_id].size();
//            jacobian_rot_i = skew(-n_i);
//            jacobian_rot_ii = skew(n_ii);
//            rot_error = (n_i - n_ii);
////          cout << "rotation error_i " << rot_error.transpose() << endl;
//
//            if(sensor_id != 0)
//            {
//              hessian.block(3*(sensor_id-1), 3*(sensor_id-1), 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_i;
//              gradient.block(3*(sensor_id-1),0,3,1) += jacobian_rot_i.transpose() * rot_error;
//            }
//
//            if(sensor_id != 6)
//            {
//              hessian.block(3*((sensor_id+1)%8), 3*((sensor_id+1)%8), 3, 3) += jacobian_rot_ii.transpose() * jacobian_rot_ii;
//              gradient.block(3*((sensor_id+1)%8),0,3,1) += jacobian_rot_ii.transpose() * rot_error;
//
//              // Cross term
//              if(sensor_id != 0)
//                hessian.block(3*(sensor_id-1), 3*((sensor_id+1)%8), 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_ii;
//            }
//          }
        }
        // Fill the lower left triangle with the corresponding cross terms
        for(sensor_id = 1; sensor_id < 7; sensor_id++)
          hessian.block(3*sensor_id, 3*(sensor_id-1), 3, 3) = hessian.block(3*(sensor_id-1), 3*sensor_id, 3, 3).transpose();

  //      cout << "Hessian\n" << hessian << endl;
        cout << "Error accumulated " << accum_error2 << endl;

        // Solve for rotation
        update_vector = -hessian.inverse() * gradient;
      cout << "update_vector " << update_vector.transpose() << endl;

        // Update rotation of the poses
        for(sensor_id = 1; sensor_id < 8; sensor_id++)
        {
          mrpt::poses::CPose3D pose;
          mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
          rot_manifold[0] = update_vector(3*sensor_id-3,0) / 10;
//          rot_manifold[0] = update_vector(3*sensor_id-3,0);
          rot_manifold[1] = update_vector(3*sensor_id-2,0);
          rot_manifold[2] = update_vector(3*sensor_id-1,0);
          mrpt::math::CMatrixDouble33 update_rot = pose.exp_rotation(rot_manifold);
  //      cout << "update_rot\n" << update_rot << endl;
          Eigen::Matrix3f update_rot_eig;
          update_rot_eig <<  update_rot(0,0), update_rot(0,1), update_rot(0,2),
                              update_rot(1,0), update_rot(1,1), update_rot(1,2),
                              update_rot(2,0), update_rot(2,1), update_rot(2,2);
          Rt_estimated_temp[sensor_id] = Rt_estimated[sensor_id];
          Rt_estimated_temp[sensor_id].block(0,0,3,3) = update_rot_eig * Rt_estimated[sensor_id].block(0,0,3,3);
  //      cout << "old rotation" << sensor_id << "\n" << Rt_estimated[sensor_id].block(0,0,3,3) << endl;
  //      cout << "new rotation\n" << Rt_estimated_temp[sensor_id].block(0,0,3,3) << endl;
        }

        // Calculate new error
        double new_accum_error2 = 0;
        for(sensor_id = 0; sensor_id < 8; sensor_id++)
        {
          for(unsigned i=0; i < correspondences[sensor_id].size(); i++)
          {
            Eigen::Vector3f n_i = Rt_estimated_temp[sensor_id].block(0,0,3,3) * correspondences[sensor_id][i].first.head(3);// / correspondences[sensor_id].size();
            Eigen::Vector3f n_ii = Rt_estimated_temp[(sensor_id+1)%8].block(0,0,3,3) * correspondences[sensor_id][i].second.head(3);// / correspondences[sensor_id].size();

            rot_error = (n_i - n_ii) / weight_pair[sensor_id];// / correspondences[sensor_id].size();
//        if(sensor_id==7)
//          cout << "rotation error_i LC " << rot_error.transpose() << endl;
            new_accum_error2 += rot_error.dot(rot_error);
          }
        }
//        cout << "New rotation error " << new_accum_error2 << endl;
  //    cout << "Closing loop? \n" << Rt_estimated[0].inverse() * Rt_estimated[7] * Rt_78;

        // Assign new rotations
        if(new_accum_error2 < accum_error2)
          for(sensor_id = 0; sensor_id < 8; sensor_id++)
            Rt_estimated[sensor_id] = Rt_estimated_temp[sensor_id];
//            Rt_estimated[sensor_id].block(0,0,3,3) = Rt_estimated_temp[sensor_id].block(0,0,3,3);

        increment = update_vector .dot (update_vector);
        diff_error = accum_error2 - new_accum_error2;
        ++it;
      cout << "Iteration " << it << " increment " << increment << " diff_error " << diff_error << endl;
      }

//      // Calculate average rotation of the X axis
//      Eigen::Matrix3f hessian_rot = Eigen::Matrix3f::Zero();
//      Eigen::Vector3f gradient_rot = Eigen::Vector3f::Zero();
//      Eigen::Vector3f X_axis; X_axis << 1, 0, 0;
//      float rotation_error2 = 0;
//      for(unsigned i=0; i<8; i++)
//      {
//        Eigen::Vector3f X_pose = Rt_estimated[i].block(0,0,3,1);
//        Eigen::Vector3f error = X_axis .cross(X_pose);
//        Eigen::Matrix3f jacobian = -skew(X_axis) * skew(X_pose);
//        hessian_rot += jacobian.transpose() * jacobian;
//        gradient_rot += jacobian.transpose() * error;
//        rotation_error2 += error .dot(error);
//      }
//      Eigen::Vector3f rotation_manifold = - hessian_rot.inverse() * gradient_rot;
//      mrpt::poses::CPose3D pose;
//      mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
//      rot_manifold[0] = 0;
////      rot_manifold[0] = rotation_manifold(0);
//      rot_manifold[1] = rotation_manifold(1);
//      rot_manifold[2] = rotation_manifold(2);
//      mrpt::math::CMatrixDouble33 update_rot = pose.exp_rotation(rot_manifold);
////    cout << "manifold update " << rot_manifold << "\nupdate_rot\n" << update_rot << endl;
//      Eigen::Matrix3f rotation;
//      rotation << update_rot(0,0), update_rot(0,1), update_rot(0,2),
//                  update_rot(1,0), update_rot(1,1), update_rot(1,2),
//                  update_rot(2,0), update_rot(2,1), update_rot(2,2);
//
//
//      // Rotate the camera system to make the vertical direction correspond to the X axis
//      for(unsigned sensor_id=0; sensor_id<8; sensor_id++)
//        Rt_estimated[sensor_id].block(0,0,3,3) = rotation * Rt_estimated[sensor_id].block(0,0,3,3);

//      // Calculate translations
////    cout << "\n Get translation\n";
//      hessian = Eigen::Matrix<float,21,21>::Zero();
//      gradient = Eigen::Matrix<float,21,1>::Zero();
//      accum_error2 = 0.0;
//      for(sensor_id = 0; sensor_id < 8; sensor_id++)
//      {
//        for(unsigned i=0; i < correspondences[sensor_id].size(); i++)
//        {
//          Eigen::Vector3f n_i = Rt_estimated[sensor_id].block(0,0,3,3) * correspondences[sensor_id][i].first.head(3);// / correspondences[sensor_id].size();
//          Eigen::Vector3f n_ii = Rt_estimated[(sensor_id+1)%8].block(0,0,3,3) * correspondences[sensor_id][i].second.head(3);// / correspondences[sensor_id].size();
////            trans_error = correspondences[sensor_id][i].first(3) + n_i.transpose()*Rt_estimated[sensor_id].block(0,3,3,1) - (correspondences[sensor_id][i].second(3) + n_ii.transpose()*Rt_estimated[sensor_id+1].block(0,3,3,1));
//          trans_error = (correspondences[sensor_id][i].first[3] - correspondences[sensor_id][i].second[3]);// / correspondences[sensor_id].size();
////          trans_error = correspondences[sensor_id][i].first(3) - (n_i(0)*Rt_estimated[sensor_id](0,3) + n_i(1)*Rt_estimated[sensor_id](1,3) + n_i(2)*Rt_estimated[sensor_id](2,3))
////                      - (correspondences[sensor_id][i].second(3) - (n_ii(0)*Rt_estimated[sensor_id+1](0,3) + n_ii(1)*Rt_estimated[sensor_id+1](1,3) + n_ii(2)*Rt_estimated[sensor_id+1](2,3)));
////  //      cout << "translation error1 " << trans_error << endl;
//          accum_error2 += trans_error * trans_error;
//
//            if(sensor_id != 0)
//            {
//              hessian.block(3*(sensor_id-1), 3*(sensor_id-1), 3, 3) += n_i * n_i.transpose();
//              gradient.block(3*(sensor_id-1),0,3,1) += -n_i * trans_error;
//            }
//
//            if(sensor_id != 7)
//            {
//              hessian.block(3*sensor_id, 3*sensor_id, 3, 3) += n_ii * n_ii.transpose();
//              gradient.block(3*sensor_id,0,3,1) += n_ii * trans_error;
//
//              // Cross term
//              if(sensor_id != 0)
//                hessian.block(3*(sensor_id-1), 3*sensor_id, 3, 3) += -n_i * n_ii.transpose();
//            }
//        }
////        // Next to adjacent sensor
////        for(unsigned i=0; i < correspondences_2[sensor_id].size(); i++)
////        {
//write properly
////          Eigen::Vector3f n_i = Rt_estimated[sensor_id].block(0,0,3,3) * correspondences_2[sensor_id][i].first.head(3) / correspondences_2[sensor_id].size();
////          Eigen::Vector3f n_ii = Rt_estimated[(sensor_id+1)%8].block(0,0,3,3) * correspondences_2[sensor_id][i].second.head(3) / correspondences_2[sensor_id].size();
////          jacobian_rot_i = skew(-n_i);
////          jacobian_rot_ii = skew(n_ii);
////          rot_error = (n_i - n_ii);
//////          cout << "rotation error_i " << rot_error.transpose() << endl;
////
////          if(sensor_id != 0)
////          {
////            hessian.block(3*(sensor_id-1), 3*(sensor_id-1), 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_i;
////            gradient.block(3*(sensor_id-1),0,3,1) += jacobian_rot_i.transpose() * rot_error;
////          }
////
////          if(sensor_id != 6)
////          {
////            hessian.block(3*((sensor_id+1)%8), 3*((sensor_id+1)%8), 3, 3) += jacobian_rot_ii.transpose() * jacobian_rot_ii;
////            gradient.block(3*((sensor_id+1)%8),0,3,1) += jacobian_rot_ii.transpose() * rot_error;
////
////            // Cross term
////            if(sensor_id != 0)
////              hessian.block(3*(sensor_id-1), 3*((sensor_id+1)%8), 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_ii;
////          }
////        }
//      }
//      // Fill the lower left triangle with the corresponding cross terms
//      for(unsigned sensor_id = 1; sensor_id < 7; sensor_id++)
//        hessian.block(3*sensor_id, 3*(sensor_id-1), 3, 3) = hessian.block(3*(sensor_id-1), 3*sensor_id, 3, 3).transpose();
//
//      // Solve for translation
//      update_vector = -hessian.inverse() * gradient;
//    cout << "update_vector translations " << update_vector.transpose() << endl;
//
//      // Update translation of the poses
//      for(sensor_id = 1; sensor_id < 8; sensor_id++)
//        Rt_estimated_temp[sensor_id].block(0,3,3,1) = Rt_estimated[sensor_id].block(0,3,3,1) + update_vector.block(3*sensor_id-3,0,3,1);
////        Rt_estimated_temp[sensor_id].block(0,3,3,1) = update_vector.block(3*sensor_id-3,0,3,1) + Rt_estimated[sensor_id].block(0,3,3,1);
//
//      // Get center of the multicamera system
//      Eigen::Vector3f centerDevice = Eigen::Vector3f::Zero();
//      for(sensor_id = 0; sensor_id < 8; sensor_id++)
//        centerDevice += Rt_estimated_temp[sensor_id].block(0,3,3,1);
//      centerDevice = centerDevice / 8;
//    cout << "Center device " << centerDevice.transpose();
//      for(sensor_id = 0; sensor_id < 8; sensor_id++)
//        Rt_estimated_temp[sensor_id].block(0,3,3,1) -= centerDevice;
//
//      // Calculate new error
//      double new_accum_error2 = 0;
//      for(sensor_id = 0; sensor_id < 8; sensor_id++)
//      {
//        for(unsigned i=0; i < correspondences[sensor_id].size(); i++)
//        {
//          Eigen::Vector3f n_i = Rt_estimated_temp[sensor_id].block(0,0,3,3) * correspondences[sensor_id][i].first.head(3);
//          Eigen::Vector3f n_ii = Rt_estimated_temp[(sensor_id+1)%8].block(0,0,3,3) * correspondences[sensor_id][i].second.head(3);
////            trans_error = correspondences[sensor_id][i].first(3) + n_i.transpose()*Rt_estimated_temp[sensor_id].block(0,3,3,1) - (correspondences[sensor_id][i].second(3) + n_ii.transpose()*Rt_estimated_temp[sensor_id+1].block(0,3,3,1));
//          trans_error = (correspondences[sensor_id][i].first[3] - (n_i(0)*Rt_estimated_temp[sensor_id](0,3) + n_i(1)*Rt_estimated_temp[sensor_id](1,3) + n_i(2)*Rt_estimated_temp[sensor_id](2,3))
//                      - (correspondences[sensor_id][i].second[3] - (n_ii(0)*Rt_estimated_temp[(sensor_id+1)%8](0,3) + n_ii(1)*Rt_estimated_temp[(sensor_id+1)%8](1,3) + n_ii(2)*Rt_estimated_temp[(sensor_id+1)%8](2,3))))
//                        ;// / correspondences[sensor_id].size();
////      if(sensor_id==7)
////        cout << "translation error_i LC " << trans_error << endl;
//          new_accum_error2 += trans_error * trans_error;
//        }
//      }
//      cout << "New translation error " << new_accum_error2 << " old " << accum_error2 << endl;
//      // Assign new translations
//      if(new_accum_error2 < accum_error2)
//        for(sensor_id = 0; sensor_id < 8; sensor_id++)
//          Rt_estimated[sensor_id].block(0,3,3,1) = Rt_estimated_temp[sensor_id].block(0,3,3,1);
////          Rt_estimated[sensor_id].block(0,3,3,1) = Eigen::Matrix<float,3,1>::Zero();

    cout << "\tCalibration finished\n";
    }

    void run()
    {
      #if USE_DEBUG_SEQUENCE
        string obsName;
        unsigned frame = 0;
      #else
//        RGBDGrabber *grabber[8];

        // Create one instance of RGBDGrabber_OpenNI2 for each sensor
        grabber[0] = new RGBDGrabber_OpenNI2("1d27/0601@5/2", 1);
        grabber[1] = new RGBDGrabber_OpenNI2("1d27/0601@4/2", 1);
        grabber[2] = new RGBDGrabber_OpenNI2("1d27/0601@3/2", 1);
        grabber[3] = new RGBDGrabber_OpenNI2("1d27/0601@6/2", 1);
        grabber[4] = new RGBDGrabber_OpenNI2("1d27/0601@9/2", 1);
        grabber[5] = new RGBDGrabber_OpenNI2("1d27/0601@8/2", 1);
        grabber[6] = new RGBDGrabber_OpenNI2("1d27/0601@7/2", 1);
        grabber[7] = new RGBDGrabber_OpenNI2("1d27/0601@10/2", 1);

        // Initialize the sensor
        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
  //      #pragma omp parallel num_threads(8)
        {
  //        int sensor_id = omp_get_thread_num();
          grabber[sensor_id]->init();
        }
        cout << "Grabber initialized\n";

        // Get the first frame
        Frame360 *frame360_1 = new Frame360(&calib);

        // Skip the first frames
        for(unsigned skipFrames = 0; skipFrames < 5; skipFrames++ )
          for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            grabber[sensor_id]->grab(&frame360_1->frameRGBD_[sensor_id]);
        #endif

        // Initialize visualizer
        Frame360_Visualizer visualizer;

        // Receive frame
        while (!visualizer.viewer.wasStopped() )
        {
          // Grab frame
          Frame360 frame360(&calib);

        cout << "Get new frame\n";
          #if USE_DEBUG_SEQUENCE
          obsName = mrpt::format("/home/eduardo/Data_RGBD360_Maison1/raw/sphere_images_%d.bin",frame);
//          cout << "obsName " << obsName << endl;
          frame360.loadFrame(obsName);
          frame++;
          #else
          for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            grabber[sensor_id]->grab(&frame360.frameRGBD_[sensor_id]);
          frame360.setTimeStamp(mrpt::system::getCurrentTime());
          #endif

        cout << "Build point cloud\n";
          // Get spherical point cloud for visualization
          frame360.buildSphereCloud();

          { boost::mutex::scoped_lock updateLock(visualizationMutex);

            pcl::copyPointCloud(*frame360.sphereCloud, *sphere_cloud);
    //        bFreezeFrame = true;

          updateLock.unlock();
          }

//          if(bTakeKeyframe)
//          {
            // Intrinsic Calibration and Smoothing

        cout << "Segment planes\n";
            // Segment planes
            frame360.getLocalPlanes();

            { boost::mutex::scoped_lock updateLock(visualizationMutex);

        cout << "Merge planes for visualization\n";
            planes.vPlanes.clear();
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
              planes.MergeWith(frame360.local_planes_[sensor_id], calib.Rt_[sensor_id]);

            updateLock.unlock();
            }

        cout << "Data association\n";
            // Data association
            plane_corresp.clear(); // For visualization
            map<unsigned, pair< Eigen::Vector4f, Eigen::Vector4f> > matches;
            int planes_counter_i = 0, planes_counter_j = 0, planes_counter_k = 0;
            float distThreshold = 0.1, angleThreshold = 0.99, proxThreshold = 0.2;
            for(unsigned couple_id=0; couple_id < 8; ++couple_id)
            {
              mrpt::pbmap::PbMap &planes_i = frame360.local_planes_[couple_id];
              mrpt::pbmap::PbMap &planes_j = frame360.local_planes_[(couple_id+1)%8];
              mrpt::pbmap::PbMap &planes_k = frame360.local_planes_[(couple_id+2)%8];

              set<unsigned> alreadyMatched_j, alreadyMatched_k;
              for(unsigned i=0; i < planes_i.vPlanes.size(); i++)
              {
                // Set thresholds depending on distance to the sensor
                distThreshold = 0.1*planes.vPlanes[planes_counter_i+i].v3center.norm();
                proxThreshold = 0.15*planes.vPlanes[planes_counter_i+i].v3center.norm();

                // Adjacent sensor
                if(couple_id != 7) // To compare sensor 7 with 0 we have to reset the counter in planes_counter_j
                  planes_counter_j = planes_counter_i+planes_i.vPlanes.size();
                else
                  planes_counter_j = 0;

                for(unsigned j=0; j < planes_j.vPlanes.size(); j++)
                {
                  if(alreadyMatched_j.count(j))
                    continue;

//float colorCond = (planes.vPlanes[planes_counter_i+i].bDominantColor && planes.vPlanes[planes_counter_j+j].bDominantColor) ? max(fabs(planes.vPlanes[planes_counter_i+i].v3colorNrgb[0] - planes.vPlanes[planes_counter_j+j].v3colorNrgb[0]), fabs(planes.vPlanes[planes_counter_i+i].v3colorNrgb[1] - planes.vPlanes[planes_counter_j+j].v3colorNrgb[1])) : 0.0;
//                  cout << "comparing couple " << couple_id << " planes " << planes_counter_i+i << " and " << planes_counter_j+j << endl;
////                  cout << "n1 " << planes_i.vPlanes[i].v3normal.transpose() << " n2 " << planes_j.vPlanes[j].v3normal.transpose() << " dot " << planes.vPlanes[planes_counter_i+i].v3normal .dot (planes.vPlanes[planes_counter_i+planes_i.vPlanes.size()+j].v3normal) << endl;
////                  cout << "n1 " << planes.vPlanes[planes_counter_i+i].v3normal.transpose() << " n2 " << planes.vPlanes[planes_counter_j+j].v3normal.transpose() << endl;
////                  cout << "d1 " << planes.vPlanes[planes_counter_i+i].d << " d2 " << planes.vPlanes[planes_counter_i+planes_i.vPlanes.size()+j].d << " diff " << fabs(planes.vPlanes[planes_counter_i+i].d - planes.vPlanes[planes_counter_j+j].d) << " nearHull " << planes.vPlanes[planes_counter_i+i].isPlaneNearby(planes.vPlanes[planes_counter_j+j], 0.4) << endl;
//                  cout << "cos(n1*n2) " << planes.vPlanes[planes_counter_i+i].v3normal .dot (planes.vPlanes[planes_counter_j+j].v3normal)
//                       << " d1-d2 " << fabs(planes.vPlanes[planes_counter_i+i].d - planes.vPlanes[planes_counter_j+j].d)
//                       << " nearHull " << planes.vPlanes[planes_counter_i+i].isPlaneNearby(planes.vPlanes[planes_counter_j+j], 0.4)
//                       << " color diff " << colorCond << endl;
////                  cout << "inliers1 " << inliersUpperFringe(planes_i.vPlanes[i], 0.2) << " inliers2 " << inliersLowerFringe(planes_j.vPlanes[j], 0.2) << endl;
                  if( planes.vPlanes[planes_counter_i+i].inliers.size() > 500 && planes.vPlanes[planes_counter_j+j].inliers.size() > 500 &&
                      planes.vPlanes[planes_counter_i+i].elongation < 5 && planes.vPlanes[planes_counter_j+j].elongation < 5 &&
                      planes.vPlanes[planes_counter_i+i].v3normal .dot (planes.vPlanes[planes_counter_j+j].v3normal) > 0.99 &&
                      fabs(planes.vPlanes[planes_counter_i+i].d - planes.vPlanes[planes_counter_j+j].d) < 0.2 &&
                      planes_i.vPlanes[i].hasSimilarDominantColor(planes_j.vPlanes[j],0.06) &&
                      planes.vPlanes[planes_counter_i+i].isPlaneNearby(planes.vPlanes[planes_counter_j+j], 0.5) )
//                      inliersUpperFringe(planes_i.vPlanes[i], 0.2) > 0.2 &&
//                      inliersLowerFringe(planes_j.vPlanes[j], 0.2) > 0.2 ) // Assign correspondence
                      {
//                      cout << "\tAssociate planes " << planes_counter_i+i << " and " << planes_counter_j+j << endl;
                        Eigen::Vector4f pl1, pl2;
                        pl1.head(3) = planes_i.vPlanes[i].v3normal; pl1[3] = planes_i.vPlanes[i].d;
                        pl2.head(3) = planes_j.vPlanes[j].v3normal; pl2[3] = planes_j.vPlanes[j].d;

////                        float factorDistInliers = std::min(planes_i.vPlanes[i].inliers.size(), planes_j.vPlanes[j].inliers.size()) / std::max(planes_i.vPlanes[i].v3center.norm(), planes_j.vPlanes[j].v3center.norm());
//                        float factorDistInliers = (planes_i.vPlanes[i].inliers.size() + planes_j.vPlanes[j].inliers.size()) / (planes_i.vPlanes[i].v3center.norm() * planes_j.vPlanes[j].v3center.norm());
//                        weight_pair[couple_id] += factorDistInliers;
//                        pl1 *= factorDistInliers;
//                        pl2 *= factorDistInliers;
                        ++weight_pair[couple_id];

                        //Add constraints
                        correspondences[couple_id].push_back(pair<Eigen::Vector4f, Eigen::Vector4f>(pl1, pl2));
//                        correspondences[couple_id].push_back(pair<Eigen::Vector4f, Eigen::Vector4f>(pl1/planes_i.vPlanes[i].v3center.norm(), pl2/planes_j.vPlanes[j].v3center.norm());

                        // Calculate conditioning
                        updateConditioning(couple_id, correspondences[couple_id].back());

                        // For visualization
                        plane_corresp[couple_id].push_back(pair<mrpt::pbmap::Plane*, mrpt::pbmap::Plane*>(&planes.vPlanes[planes_counter_i+i], &planes.vPlanes[planes_counter_j+j]));

//                        i = planes_i.vPlanes.size(); // Exit loops
                        alreadyMatched_j.insert(j);
                        break;
                      }
                }

                // Next adjacent sensor (0-2)
                if(couple_id != 6) // To compare sensor 7 with 0 we have to reset the counter in planes_counter_k
                  planes_counter_k = planes_counter_j+planes_j.vPlanes.size();
                else
                  planes_counter_k = 0;
                for(unsigned k=0; k < planes_k.vPlanes.size(); k++)
                {
                  if(alreadyMatched_k.count(k))
                    continue;

                  if( planes.vPlanes[planes_counter_i+i].inliers.size() > 500 && planes.vPlanes[planes_counter_k+k].inliers.size() > 500 &&
                      planes.vPlanes[planes_counter_i+i].elongation < 5 && planes.vPlanes[planes_counter_k+k].elongation < 5 &&
                      planes.vPlanes[planes_counter_i+i].v3normal .dot (planes.vPlanes[planes_counter_k+k].v3normal) > 0.99 &&
                      fabs(planes.vPlanes[planes_counter_i+i].d - planes.vPlanes[planes_counter_k+k].d) < 0.2 &&
                      planes_i.vPlanes[i].hasSimilarDominantColor(planes_k.vPlanes[k],0.06) &&
                      planes.vPlanes[planes_counter_i+i].isPlaneNearby(planes.vPlanes[planes_counter_k+k], 0.5) )
//                      inliersUpperFringe(planes_i.vPlanes[i], 0.2) > 0.2 &&
//                      inliersLowerFringe(planes_k.vPlanes[k], 0.2) > 0.2 ) // Assign correspondence
                      {
                      cout << "\t   Associate planes " << planes_counter_i+i << " and " << planes_counter_k+k << endl;
                        Eigen::Vector4f pl1, pl2;
                        pl1.head(3) = planes_i.vPlanes[i].v3normal; pl1[3] = planes_i.vPlanes[i].d;
                        pl2.head(3) = planes_k.vPlanes[k].v3normal; pl2[3] = planes_k.vPlanes[k].d;

//                        float factorDistInliers = std::min(planes_i.vPlanes[i].inliers.size(), planes_j.vPlanes[j].inliers.size()) / std::max(planes_i.vPlanes[i].v3center.norm(), planes_j.vPlanes[j].v3center.norm());
                        float factorDistInliers = (planes_i.vPlanes[i].inliers.size() + planes_k.vPlanes[k].inliers.size()) / (planes_i.vPlanes[i].v3center.norm() * planes_k.vPlanes[k].v3center.norm());
//                        weight_pair_2[couple_id] += factorDistInliers;
                        pl1 *= factorDistInliers;
                        pl2 *= factorDistInliers;

                        //Add constraints
                        correspondences_2[couple_id].push_back(pair<Eigen::Vector4f, Eigen::Vector4f>(pl1, pl2));

//                        // For visualization
//                        plane_corresp[couple_id].push_back(pair<mrpt::pbmap::Plane*, mrpt::pbmap::Plane*>(&planes.vPlanes[planes_counter_i+i], &planes.vPlanes[planes_counter_k+k]));

//                        i = planes_i.vPlanes.size(); // Exit loops
                        alreadyMatched_k.insert(k);
                        break;
                      }
                }
              }

              if(correspondences[couple_id].size() > 3)
                calcConditioning(couple_id);

              planes_counter_i += planes_i.vPlanes.size();
            }
//            cout << plane_corresp[couple_id].size() << " asociations in this frame\n";
//            printConditioning();

//            #if SAVE_IMAGES
//            if(matches.size() > 3)
//            {
//            cout << "   Saving images\n";
//              cv::Mat timeStampMatrix;
//              getMatrixNumberRepresentationOf_uint64_t(mrpt::system::getCurrentTime(),timeStampMatrix);
//              {
//              std::ofstream ofs_images(mrpt::format("/home/eduardo/Calibration_RGBD360/sphere_images_%d.bin",frame).c_str(), std::ios::out | std::ios::binary);
//              boost::archive::binary_oarchive oa_images(ofs_images);
//  //            oa_images << frameRGBD1->getRGBImage() << frameRGBD2->getRGBImage();
//  //            oa_images << frameRGBD1->getDepthImage() << frameRGBD2->getDepthImage();
//  //            oa_images << frameRGBD3->getDepthImage() << frameRGBD4->getDepthImage();
//
//              oa_images << frameRGBD1->getRGBImage() << frameRGBD1->getDepthImage() << frameRGBD2->getRGBImage() << frameRGBD2->getDepthImage()
//                        << frameRGBD3->getRGBImage() << frameRGBD3->getDepthImage() << frameRGBD4->getRGBImage() << frameRGBD4->getDepthImage()
//                        << frameRGBD5->getRGBImage() << frameRGBD5->getDepthImage() << frameRGBD6->getRGBImage() << frameRGBD6->getDepthImage()
//                        << frameRGBD7->getRGBImage() << frameRGBD7->getDepthImage() << frameRGBD8->getRGBImage() << frameRGBD8->getDepthImage() << timeStampMatrix;
//              ofs_images.close();
//              }
//            }
//            #endif
//
//          }

//          mrpt::system::pause();
//      boost::this_thread::sleep (boost::posix_time::milliseconds (200));
        }

        #if USE_DEBUG_SEQUENCE==0
        // Stop grabbing
        for(unsigned sensor_id = 0; sensor_id < 8; sensor_id++)
          delete grabber[sensor_id];
        #endif

        float threshold_conditioning = 8000.0;
        cout << "Conditioning " << *std::max_element(conditioning,conditioning+8) << " threshold " << threshold_conditioning << endl;
        if(*std::max_element(conditioning,conditioning+8) < threshold_conditioning)
        {
          printConditioning();
        cout << "\tSave CorrespMat\n";
          // Save correspondence matrices
          for(unsigned sensor_id=0; sensor_id < 8; sensor_id++)
          {
          cout << "Save CorrespMat " << sensor_id << "\n";
            ofstream corresp(mrpt::format("%s/OnlineCalibration/correspMat_%i", PROJECT_SOURCE_PATH, sensor_id+1).c_str());
            for(unsigned i=0; i < correspondences[sensor_id].size(); i++)
              corresp << correspondences[sensor_id][i].first.transpose() << "\t" << correspondences[sensor_id][i].second.transpose() << endl;
            corresp.close();
          }

//        cout << "Load CorrespMat\n";
//          // Load Plane correspondences from file
//          vector<mrpt::math::CMatrixDouble> correspMat(8);
//          for(unsigned sensor_id=0; sensor_id < 8; sensor_id++)
//          {
//            unsigned sensor_id=0;
//            correspMat[sensor_id].loadFromTextFile(mrpt::format("../../Dropbox/Doctorado/Projects/RGBD360/Calibration/correspondences/correspMat_%i.txt",sensor_id).c_str());
//          }

          Calibrate();

          // Save calibration matrices
          #if SAVE_CALIBRATION
          cout << "   SAVE_CALIBRATION \n";
          ofstream calibFile;
          for(unsigned sensor_id=0; sensor_id < 8; sensor_id++)
          {
            string calibFileName = mrpt::format("%s/OnlineCalibration/Rt_%i", PROJECT_SOURCE_PATH, sensor_id+1);
            calibFile.open(calibFileName.c_str());
            if (calibFile.is_open())
            {
              calibFile << Rt_estimated[sensor_id];
              calibFile.close();
            }
            else
              cout << "Unable to open file " << calibFileName << endl;
          }
          #endif

          // Visualize Calibration
          planes.vPlanes.clear();
          for(unsigned sensor_id=0; sensor_id < 8; sensor_id++)
            calib.setRt_id(sensor_id, Rt_estimated[sensor_id]);
          pcl::visualization::CloudViewer viewer2("RGBD360_calib");
          viewer2.runOnVisualizationThread (boost::bind(&OnlineCalibration::viz_cb, this, _1), "viz_cb");
  //        viewer2.registerKeyboardCallback(&OnlineCalibration::keyboardEventOccurred, *this);

          // Receive frame
          int frame = 0;
          while (!viewer2.wasStopped() )
          {
            // Grab frame
            Frame360 frame360(&calib);

            #if USE_DEBUG_SEQUENCE
  //          obsName = mrpt::format("/home/eduardo/Data_RGBD360_0/raw/sphere_images_%d.bin",frame);
  //          obsName = mrpt::format("/home/eduardo/Data_RGBD360_Bureaux k001-k010/raw/sphere_images_%d.bin",frame);
  //          obsName = mrpt::format("../../Dropbox/Doctorado/Compartidas/Romain-Eduardo/Data_RGBD360_0/raw/sphere_images_%d.bin",frame);
            obsName = mrpt::format("/home/eduardo/Data_RGBD360_Maison1/raw/sphere_images_%d.bin",frame);
            cout << "obsName " << obsName << endl;
            frame360.loadFrame(obsName);
            frame++;
            #else
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
              grabber[sensor_id]->grab(&frame360.frameRGBD_[sensor_id]);
            frame360.setTimeStamp(mrpt::system::getCurrentTime());
            #endif

            // Get spherical point cloud for visualization
            frame360.buildSphereCloud();

            { boost::mutex::scoped_lock updateLock(visualizationMutex);

              pcl::copyPointCloud(*frame360.sphereCloud, *sphere_cloud);
      //        bFreezeFrame = true;

            updateLock.unlock();
            }

//              // Segment planes
//              frame360.getLocalPlanes();
//
//              { boost::mutex::scoped_lock updateLock(visualizationMutex);
//
//              planes.vPlanes.clear();
//              for(int sensor_id = 0; sensor_id < 8; sensor_id++)
//                planes.MergeWith(frame360.local_planes_[sensor_id], calib.Rt_[sensor_id]);
//
//              updateLock.unlock();
//              }

            boost::this_thread::sleep (boost::posix_time::milliseconds (1000));
          }
        } // End perform calibration

//      }
//      else
//        cout << "Less than 8 devices connected: at least two RGB-D sensors are required to perform extrinsic calibration.\n";

    }

  private:

    boost::mutex visualizationMutex;

//    Frame360 *frame_360;
    mrpt::pbmap::PbMap planes;

    Calib360 calib;
    Eigen::Matrix4f Rt_estimated[8];//, Rt_estimated_temp;
//    vector<mrpt::math::CMatrixDouble> correspondences;
    float weight_pair[8];
    vector< vector< pair< Eigen::Vector4f, Eigen::Vector4f> > > correspondences;
    vector< vector< pair< Eigen::Vector4f, Eigen::Vector4f> > > correspondences_2;
    Eigen::Matrix3f covariances[8];
    float conditioning[8];
    unsigned conditioning_measCount[8];

    pcl::PointCloud<PointT>::Ptr sphere_cloud;
    map<unsigned, vector<pair<mrpt::pbmap::Plane*, mrpt::pbmap::Plane*> > > plane_corresp;

    bool bFreezeFrame;

    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
//    cout << "SphericalSequence::viz_cb(...)\n";
      if (sphere_cloud->empty() || bFreezeFrame)
      {
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
        return;
      }
//    cout << "   ::viz_cb(...)\n";

      viz.removeAllShapes();
      viz.removeAllPointClouds();

      { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
        boost::mutex::scoped_lock updateLock(visualizationMutex);

        if (!viz.updatePointCloud (sphere_cloud, "sphereCloud"))
          viz.addPointCloud (sphere_cloud, "sphereCloud");


        // Draw camera system
        viz.removeCoordinateSystem();
        for(unsigned sensor_id=0; sensor_id < 8; sensor_id++)
        {
          Eigen::Affine3f Rt;
          Rt.matrix() = calib.Rt_[sensor_id];
          viz.addCoordinateSystem(0.1, Rt);
          pcl::ModelCoefficients coeffCylinder;
//          coeffCylinder.values[0] = calib.Rt_[sensor_id](0,3);
//          coeffCylinder.values[1] = calib.Rt_[sensor_id](1,3);
//          coeffCylinder.values[2] = calib.Rt_[sensor_id](2,3);
//          coeffCylinder.values[3] = calib.Rt_[sensor_id](0,0);
//          coeffCylinder.values[4] = calib.Rt_[sensor_id](1,0);
//          coeffCylinder.values[5] = calib.Rt_[sensor_id](2,0);
//          coeffCylinder.values[6] = 0.5;
//          viz.addCylinder(coeffCylinder, mrpt::format("X%u", sensor_id));
        }

        // Draw planes
        char name[1024];
//
        for(size_t i=0; i < planes.vPlanes.size(); i++)
        {
          mrpt::pbmap::Plane &plane_i = planes.vPlanes[i];
          sprintf (name, "normal_%u", static_cast<unsigned>(i));
          pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
          pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
          pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
                              plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
                              plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
          viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

          {
            sprintf (name, "n%u", static_cast<unsigned>(i));
//            sprintf (name, "n%u_%u", static_cast<unsigned>(i), static_cast<unsigned>(plane_i.semanticGroup));
            viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
          }

          sprintf (name, "plane_%02u", static_cast<unsigned>(i));
          pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[3], grn[3], blu[3]);
          pcl::visualization::PointCloudColorHandlerCustom <PointT> color_match (plane_i.planePointCloudPtr, red[0], grn[0], blu[0]);
          bool bMatched = false;
          for(map<unsigned, vector<pair<mrpt::pbmap::Plane*, mrpt::pbmap::Plane*> > >::iterator it=plane_corresp.begin(); it != plane_corresp.end(); it++)
            for(vector<pair<mrpt::pbmap::Plane*, mrpt::pbmap::Plane*> >::iterator it2=it->second.begin(); it2 != it->second.end(); it2++)
              if(it2->first == &plane_i || it2->second == &plane_i)
              {
                bMatched = true;
                break;
              }
          if(bMatched)
          viz.addPointCloud (plane_i.planePointCloudPtr, color_match, name);
          else
          viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, name);

          sprintf (name, "approx_plane_%02d", int (i));
          viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);
        }

        sprintf (name, "%zu pts. Params ...", sphere_cloud->size());
        viz.addText (name, 20, 20, "params");
//        bFreezeFrame = true;

      updateLock.unlock();
      }
    }

    bool bTakeKeyframe;

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
    {
      if ( event.keyDown () )
      {
        if(event.getKeySym () == "k" || event.getKeySym () == "K")
          bTakeKeyframe = true;
        else if(event.getKeySym () == "l" || event.getKeySym () == "L")
          bFreezeFrame = !bFreezeFrame;
      }
    }

};


void print_help(char ** argv)
{
  cout << "\nThis program calibrates the extrinsic parameters of the omnidirectional RGB-D device."
       << " This program opens the sensor, which has to be moved to take different"
       << " plane observations at different angles and distances. When enough information has been collected, a "
       << " Gauss-Newtown optimization is launched to obtain the extrinsic calibration. The user can decide whether to"
       << " save or not the new calibration after visualizing the calibrated image streaming from the sensor.\n";
  cout << "usage: " << argv[0] << " [options] \n";
  cout << argv[0] << " -h | --help : shows this help" << endl;
  cout << argv[0] << " -s | --save <pathToCalibrationFile>" << endl;
}


int main (int argc, char ** argv)
{
  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
    print_help(argv);

  cout << "Create OnlineCalibration object\n";
  OnlineCalibration calib_rgbd360;
  calib_rgbd360.run();

  cout << "EXIT\n";

  return (0);
}

