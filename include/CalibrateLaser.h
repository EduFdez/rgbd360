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

#ifndef CALIBRATELASER_H
#define CALIBRATELASER_H

#include <mrpt/base.h>
#include <pcl/filters/extract_indices.h>

#include <Frame360.h>

struct line
{
  Eigen::Matrix<float,6,1> params;
  pcl::PointCloud<PointT>::Ptr point_cloud;

  line() :
    point_cloud(new pcl::PointCloud<PointT>())
  {}
};


/*! This class contains the functionality to calibrate the extrinsic parameters of a pair Laser-Range camera (e.g. Kinect, ToF, etc).
 *  This extrinsic calibration is obtained by matching planes and lines that are observed by both types of sensors at the same time.
 */
class CalibPairLaserKinect
{
  public:

    /*! The extrinsic matrix estimated by this calibration method */
    Eigen::Matrix4f Rt_estimated;

    Eigen::Matrix3f rotation;

    Eigen::Vector3f translation;

    Eigen::Matrix3f FIM_rot;
    Eigen::Matrix3f FIM_trans;

    /*! The plane-line correspondences */
    mrpt::math::CMatrixDouble correspondences;

    /*! Load an initial estimation of Rt between the pair of Asus sensors from file */
    void setInitRt(const std::string Rt_file)
    {
      assert( fexists(Rt_file.c_str()) );

      Rt_estimated.loadFromTextFile(Rt_file);
    }

    /*! Load an initial estimation of Rt between the pair of Asus sensors from file */
    void setInitRt(Eigen::Matrix4f initRt)
    {
      Rt_estimated = initRt;
//      Rt_estimated = Eigen::Matrix4f::Identity();
//      Rt_estimated(1,1) = Rt_estimated(2,2) = cos(45*PI/180);
//      Rt_estimated(1,2) = -sin(45*PI/180);
//      Rt_estimated(2,1) = -Rt_estimated(1,2);
    }

    /*! Get the sum of squared rotational errors for the input extrinsic matrices. TODO: the input argument of this function is unsafe -> fix it */
    float calcCorrespRotError(Eigen::Matrix4f &Rt_)
    {
      Eigen::Matrix3f R = Rt_.block(0,0,3,3);
      return calcCorrespRotError(R);
    }

    float calcCorrespRotError()
    {
      Eigen::Matrix3f R = Rt_estimated.block(0,0,3,3);
      return calcCorrespRotError(R);
    }

    float calcCorrespRotError(Eigen::Matrix3f &Rot_)
    {
//      cout << "calcCorrespRotError \n" << Rot_ << endl << correspondences.getRowCount() << "x" << correspondences.getColCount() << endl;
      float accum_error2 = 0.0;
//      float accum_error_deg = 0.0;
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
//        float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));
//        float weight = 1.0;
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3f l_obs_ii; l_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
//        Eigen::Vector3f n_ii = Rot_ * n_obs_ii;
//        Eigen::Vector3f rot_error = (n_obs_i - n_ii);
        accum_error2 += pow(n_obs_i.transpose() * Rot_ * l_obs_ii, 2);
//        accum_error2 += weight * fabs(rot_error.dot(rot_error));
//        accum_error_deg += acos(fabs(rot_error.dot(rot_error)));
      }

//      std::cout << "AvError deg " << accum_error_deg/correspondences.getRowCount() << std::endl;
      return accum_error2/correspondences.getRowCount();
    }

    float calcCorrespRotErrorCross(Eigen::Matrix3f Rot_)
    {
//      cout << "calcCorrespRotError \n" << Rot_ << endl << correspondences.getRowCount() << "x" << correspondences.getColCount() << endl;
      float accum_error2 = 0.0;
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3f l_obs_ii; l_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        Eigen::Vector3f rot_error = n_obs_i.cross(Rot_ * l_obs_ii);
        accum_error2 += rot_error.transpose() * rot_error;
      }
      return accum_error2/correspondences.getRowCount();
    }

    float calcCorrespTransError(Eigen::Matrix4f &Rt_)
    {
      Eigen::Matrix3f R = Rt_.block(0,0,3,3);
      Eigen::Vector3f t = Rt_.block(0,3,3,1);

      float accum_error2 = 0.0;
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
//        float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));
        float weight = 1.0;
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        float d_obs_i = correspondences(i,3);
        Eigen::Vector3f c_obs_ii; c_obs_ii << correspondences(i,7), correspondences(i,8), correspondences(i,9);

        float trans_error = (d_obs_i + n_obs_i.dot(t - R * c_obs_ii));
        accum_error2 += weight * trans_error * trans_error;
      }
      return accum_error2/correspondences.getRowCount();
    }

    /*! Get the rotation of each sensor in the multisensor RGBD360 setup */
//    Eigen::Matrix3f CalibrateRotation(const mrpt::math::CMatrixDouble &corresp, int weightedLS = 0)
    Eigen::Matrix3f CalibrateRotationCross(int weightedLS = 0)
    {
    cout << "CalibrateRotationCross Plane-Line...\n";
      Eigen::Matrix<float,3,3> hessian;
      Eigen::Matrix<float,3,1> gradient;
      Eigen::Matrix<float,3,1> update_vector;
      Eigen::Matrix<float,3,3> jacobian_rot_ii; // Jacobians of the rotation
      float accum_error2;
      float av_angle_error;
      unsigned numControlPlanes;

//      Rt_estimated = Eigen::Matrix4f::Identity();
      Eigen::Matrix4f Rt_estimatedTemp;
      Rt_estimatedTemp = Rt_estimated;

      // Parameters of the Least-Squares optimization
      unsigned _max_iterations = 10;
      float _epsilon_transf = 0.00001;
      float _convergence_error = 0.000001;

      float increment = 1000, diff_error = 1000;
      int it = 0;
      while(it < _max_iterations && increment > _epsilon_transf && diff_error > _convergence_error)
      {
        // Calculate the hessian and the gradient
        hessian = Eigen::Matrix<float,3,3>::Zero(); // Hessian of the rotation of the decoupled system
        gradient = Eigen::Matrix<float,3,1>::Zero(); // Gradient of the rotation of the decoupled system
        accum_error2 = 0.0;
        av_angle_error = 0.0;
        numControlPlanes = 0;

//        for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
        {
//          assert( correspondences.getRowCount() >= 3 );

          for(unsigned i=0; i < correspondences.getRowCount(); i++)
          {
//          float weight = (inliers / correspondences(i,3)) / correspondences.getRowCount()
            Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
//            float d_obs_i = correspondences(i,3);
            Eigen::Vector3f l_obs_ii; l_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
//            Eigen::Vector3f c_obs_ii; c_obs_ii << correspondences(i,7), correspondences(i,8), correspondences(i,9);
//            jacobian_rot_i = skew(-n_i);
            Eigen::Vector3f l_i = Rt_estimated.block(0,0,3,3) * l_obs_ii;
            Eigen::Matrix<float,3,3> jacobian_R_l = skew(l_i);
            for(unsigned j=0; j < 3; j++)
            {
              Eigen::Vector3f jac_j = jacobian_R_l.block(0,j,3,1);
              jacobian_rot_ii.block(0,j,3,1) = n_obs_i.cross(jac_j);
            }

            Eigen::Vector3f rot_error = n_obs_i.cross(l_i);
            accum_error2 += rot_error.dot(rot_error);
            numControlPlanes++;
            {
              hessian += jacobian_rot_ii.transpose() * jacobian_rot_ii;
              gradient += jacobian_rot_ii.transpose() * rot_error;
            }
          }
          accum_error2 /= numControlPlanes;
        }

        // Solve the rotation
        update_vector = hessian.inverse() * gradient;
//      cout << "update_vector " << update_vector.transpose() << endl;

        // Update rotation of the poses
//        for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
        {
          mrpt::poses::CPose3D pose;
          mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
          rot_manifold[0] = update_vector(0);
          rot_manifold[1] = update_vector(1,0);
          rot_manifold[2] = update_vector(2,0);
//          rot_manifold[2] = update_vector(3*sensor_id-3,0) / 4; // Limit the turn around the Z (depth) axis
//          rot_manifold[2] = 0; // Limit the turn around the Z (depth) axis
          mrpt::math::CMatrixDouble33 update_rot = pose.exp_rotation(rot_manifold);
  //      cout << "update_rot\n" << update_rot << endl;
          Eigen::Matrix3f update_rot_eig;
          update_rot_eig << update_rot(0,0), update_rot(0,1), update_rot(0,2),
                            update_rot(1,0), update_rot(1,1), update_rot(1,2),
                            update_rot(2,0), update_rot(2,1), update_rot(2,2);
          Rt_estimatedTemp = Rt_estimated;
          Rt_estimatedTemp.block(0,0,3,3) = update_rot_eig * Rt_estimated.block(0,0,3,3);
  //      cout << "old rotation" << sensor_id << "\n" << Rt_estimated.block(0,0,3,3) << endl;
  //      cout << "new rotation\n" << Rt_estimatedTemp.block(0,0,3,3) << endl;
        }
//      cout << "It " << it << " Rt_estimated \n" << Rt_estimated << "\n Rt_estimatedTemp \n" << Rt_estimatedTemp << endl;

        accum_error2 = calcCorrespRotErrorCross( Rt_estimated.block(0,0,3,3) );
//        float new_accum_error2 = calcCorrespRotErrorWeight(Rt_estimatedTemp);
        float new_accum_error2 = calcCorrespRotErrorCross( Rt_estimatedTemp.block(0,0,3,3) );

        // Assign new rotations
        if(new_accum_error2 < accum_error2)
//          for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
            Rt_estimated = Rt_estimatedTemp;
//            Rt_estimated.block(0,0,3,3) = Rt_estimatedTemp.block(0,0,3,3);

        increment = update_vector .dot (update_vector);
        diff_error = accum_error2 - new_accum_error2;
        ++it;
      cout << "Iteration " << it << " increment " << increment << " diff_error " << diff_error << endl;
      }

////      std::cout << "ErrorCalibRotation " << accum_error2 << std::endl;
      std::cout << "Rotation it " << it << " \n"<< Rt_estimated.block(0,0,3,3) << std::endl;

      return Rt_estimated.block(0,0,3,3);
    }

    /*! Get the rotation of each sensor in the multisensor RGBD360 setup */
//    Eigen::Matrix3f CalibrateRotation(const mrpt::math::CMatrixDouble &corresp, int weightedLS = 0)
    Eigen::Matrix3f CalibrateRotation(int weightedLS = 0)
    {
//    cout << "CalibrateRotation Plane-Line...\n";
      Eigen::Matrix<float,3,3> hessian;
      Eigen::Matrix<float,3,1> gradient;
      Eigen::Matrix<float,3,1> update_vector;
      Eigen::Matrix<float,1,3> jacobian_rot_ii; // Jacobians of the rotation
      float accum_error2;
      float av_angle_error;
      unsigned numControlPlanes;

//      Rt_estimated = Eigen::Matrix4f::Identity();
      Eigen::Matrix4f Rt_estimatedTemp;
      Rt_estimatedTemp = Rt_estimated;
//    cout << "Initial Rt \n" << Rt_estimated << endl;

      // Parameters of the Least-Squares optimization
      unsigned _max_iterations = 10;
      float _epsilon_transf = 0.00001;
      float _convergence_error = 0.000001;

      float increment = 1000, diff_error = 1000;
      int it = 0;
      while(it < _max_iterations && increment > _epsilon_transf && diff_error > _convergence_error)
      {
        // Calculate the hessian and the gradient
        hessian = Eigen::Matrix<float,3,3>::Zero(); // Hessian of the rotation of the decoupled system
        gradient = Eigen::Matrix<float,3,1>::Zero(); // Gradient of the rotation of the decoupled system
        accum_error2 = 0.0;
        av_angle_error = 0.0;
        numControlPlanes = 0;

//        for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
        {
//          assert( correspondences.getRowCount() >= 3 );

          for(unsigned i=0; i < correspondences.getRowCount(); i++)
          {
//          float weight = (inliers / correspondences(i,3)) / correspondences.getRowCount()
            Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
//            float d_obs_i = correspondences(i,3);
            Eigen::Vector3f l_obs_ii; l_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
//            Eigen::Vector3f c_obs_ii; c_obs_ii << correspondences(i,7), correspondences(i,8), correspondences(i,9);
//            jacobian_rot_i = skew(-n_i);
            jacobian_rot_ii = (n_obs_i.transpose() * skew(Rt_estimated.block(0,0,3,3) * l_obs_ii));
            float rot_error = n_obs_i.transpose() * Rt_estimated.block(0,0,3,3) * l_obs_ii;
            accum_error2 += pow(rot_error,2);
            av_angle_error += PI/2 - fabs(acos(rot_error));
            numControlPlanes++;
//          cout << "rotation error_i " << rot_error << endl;
//          cout << "jacobian_rot_ii " << jacobian_rot_ii << endl;
//          cout << "n_obs_i " << n_obs_i.transpose() << " l_obs_ii " << l_obs_ii.transpose() << endl;
//          cout << "Rt_estimated.block(0,0,3,3) \n" << skew(Rt_estimated.block(0,0,3,3) * l_obs_ii) << endl;
//          mrpt::system::pause();
//            if(weightedLS == 1 && correspondences.getColCount() == 10)
//            {
//              // The weight takes into account the number of inliers of the patch, the distance of the patch's center to the image center and the distance of the plane to the sensor
////              float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));// / correspondences.getRowCount();
////              hessian += weight * (jacobian_rot_ii.transpose() * jacobian_rot_ii);
////              gradient += weight * (jacobian_rot_ii.transpose() * rot_error);
//              Eigen::Matrix3f information;
//              information << correspondences(i,8), correspondences(i,9), correspondences(i,10), correspondences(i,11),
//                            correspondences(i,9), correspondences(i,12), correspondences(i,13), correspondences(i,14),
//                            correspondences(i,10), correspondences(i,13), correspondences(i,15), correspondences(i,16),
//                            correspondences(i,11), correspondences(i,14), correspondences(i,16), correspondences(i,17);
//              hessian += jacobian_rot_ii.transpose() * information.block(0,0,3,3) * jacobian_rot_ii;
//              gradient += jacobian_rot_ii.transpose() * information.block(0,0,3,3) * rot_error;
//            }
//            else
            {
              hessian += jacobian_rot_ii.transpose() * jacobian_rot_ii;
              gradient += jacobian_rot_ii.transpose() * rot_error;
            }

//            Eigen::JacobiSVD<Eigen::Matrix3f> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
//            float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//            Eigen::Matrix3f cov = hessian.inverse();
//            Eigen::JacobiSVD<Eigen::Matrix3f> svd2(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

//            float minFIM_rot = std::min(hessian(0,0), std::min(hessian(1,1), hessian(2,2)));
//            std::cout << " det " << hessian.determinant() << " minFIM_rot " << minFIM_rot << " conditioningX " << conditioning << std::endl;
////            std::cout << hessian(0,0) << " " << hessian(1,1) << " " << hessian(2,2) << endl;
////            std::cout << "COV " << svd2.singularValues().transpose() << endl;
//            std::cout << "FIM rotation " << svd.singularValues().transpose() << endl;
          }
          accum_error2 /= numControlPlanes;
          av_angle_error /= numControlPlanes;
        }

        FIM_rot = hessian;

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
        float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//        Eigen::Matrix3f cov;
//        svd.pinv(cov);
//        std::cout << "hessian \n" << hessian << "inv\n" << hessian.inverse() << "\ncov \n" << cov << std::endl;

//        std::cout << "conditioning " << conditioning << std::endl;
//        if(conditioning > 100)
//          return Eigen::Matrix3f::Identity();

        // Solve the rotation
        update_vector = hessian.inverse() * gradient;
//      cout << "hessian\n" << hessian << endl << "gradient " << gradient.transpose() << endl;
//      cout << "update_vectorPL " << update_vector.transpose() << endl;

        // Update rotation of the poses
//        for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
        {
          mrpt::poses::CPose3D pose;
          mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
          rot_manifold[0] = update_vector(0);
          rot_manifold[1] = update_vector(1,0);
          rot_manifold[2] = update_vector(2,0);
//          rot_manifold[2] = update_vector(3*sensor_id-3,0) / 4; // Limit the turn around the Z (depth) axis
//          rot_manifold[2] = 0; // Limit the turn around the Z (depth) axis
          mrpt::math::CMatrixDouble33 update_rot = pose.exp_rotation(rot_manifold);
//        cout << "update_rot\n" << update_rot << endl;
          Eigen::Matrix3f update_rot_eig;
          update_rot_eig << update_rot(0,0), update_rot(0,1), update_rot(0,2),
                            update_rot(1,0), update_rot(1,1), update_rot(1,2),
                            update_rot(2,0), update_rot(2,1), update_rot(2,2);
          Rt_estimatedTemp.block(0,0,3,3) = update_rot_eig * Rt_estimated.block(0,0,3,3);
//        cout << "old rotation\n" << Rt_estimated.block(0,0,3,3) << endl;
//        cout << "new rotation\n" << Rt_estimatedTemp.block(0,0,3,3) << endl;
        }
//      cout << "It " << it << " Rt_estimated \n" << Rt_estimated << "\n Rt_estimatedTemp \n" << Rt_estimatedTemp << endl;

        accum_error2 = calcCorrespRotError(Rt_estimated);
//        float new_accum_error2 = calcCorrespRotErrorWeight(Rt_estimatedTemp);
        float new_accum_error2 = calcCorrespRotError(Rt_estimatedTemp);

//        cout << "New rotation error " << new_accum_error2 << " previous " << accum_error2 << endl;
  //    cout << "Closing loop? \n" << Rt_estimated[0].inverse() * Rt_estimated[7] * Rt_78;

        // Assign new rotations
        if(new_accum_error2 < accum_error2)
//          for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
            Rt_estimated = Rt_estimatedTemp;
//            Rt_estimated.block(0,0,3,3) = Rt_estimatedTemp.block(0,0,3,3);

        increment = update_vector .dot (update_vector);
        diff_error = accum_error2 - new_accum_error2;
        ++it;
//      cout << "Iteration " << it << " increment " << increment << " diff_error " << diff_error << endl;
      }

////      std::cout << "ErrorCalibRotation " << accum_error2 << " " << av_angle_error << std::endl;
//      std::cout << "Rotation \n"<< Rt_estimated << std::endl;

      return Rt_estimated.block(0,0,3,3);
    }

    float calcCorrespPlanePointError(Eigen::Matrix4f &Rt_)
    {
      float accum_error2 = 0.0;
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        float d_obs_i = correspondences(i,3);
        Eigen::Vector3f pt_obs_ii; pt_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        Eigen::Vector3f pt_i = Rt_.block(0,0,3,3) * pt_obs_ii + Rt_.block(0,3,3,1);
        float error = n_obs_i.transpose() * pt_i + d_obs_i;
        accum_error2 += pow(error,2);
      }
      return accum_error2/correspondences.getRowCount();
    }

    Eigen::Matrix4f CalibrateRt_plane_point(int weightedLS = 0)
    {
      double calib_start = pcl::getTime();

//    cout << "CalibrateRotation Plane-Line...\n" << Rt_estimated << endl;
      Eigen::Matrix<float,6,6> hessian;
      Eigen::Matrix<float,6,1> gradient;
      Eigen::Matrix<float,6,1> update_vector;
      Eigen::Matrix<float,1,6> jacobian_rot_ii; // Jacobians of the rotation
      float accum_error2;
      float av_angle_error;
      unsigned numControlPlanes;

//      Rt_estimated = Eigen::Matrix4f::Identity();
      Eigen::Matrix4f Rt_estimatedTemp;
      Rt_estimatedTemp = Rt_estimated;

      // Parameters of the Least-Squares optimization
      unsigned _max_iterations = 10;
      float _epsilon_transf = 0.00001;
      float _convergence_error = 0.000001;

      float increment = 1000, diff_error = 1000;
      int it = 0;
      while(it < _max_iterations && increment > _epsilon_transf && diff_error > _convergence_error)
      {
        // Calculate the hessian and the gradient
        hessian = Eigen::Matrix<float,6,6>::Zero(); // Hessian of the rotation of the decoupled system
        gradient = Eigen::Matrix<float,6,1>::Zero(); // Gradient of the rotation of the decoupled system
        accum_error2 = 0.0;
        numControlPlanes = 0;

        for(unsigned i=0; i < correspondences.getRowCount(); i++)
        {
          Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
          float d_obs_i = correspondences(i,3);
          Eigen::Vector3f pt_obs_ii; pt_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
          Eigen::Vector3f pt_i = Rt_estimated.block(0,0,3,3) * pt_obs_ii + Rt_estimated.block(0,3,3,1);
          jacobian_rot_ii.block(0,0,1,3) = n_obs_i.transpose();
          jacobian_rot_ii.block(0,3,1,3) = -(n_obs_i.transpose() * skew(pt_i));
          float error = -(n_obs_i.transpose() * pt_i + d_obs_i);
          accum_error2 += pow(error,2);
          numControlPlanes++;
//        cout << "Plane-point error_i " << error << " " << n_obs_i.transpose() * pt_i << " " << d_obs_i << endl;
//        cout << " Plane-point error_i " << jacobian_rot_ii << "  " << jacobian_rot_ii << endl;
//        mrpt::system::pause();
          {
            hessian += jacobian_rot_ii.transpose() * jacobian_rot_ii;
            gradient += jacobian_rot_ii.transpose() * error;
          }
        }
        accum_error2 /= numControlPlanes;

        // Solve the rotation
        update_vector = hessian.inverse() * gradient;
//      cout << "update_vectorPP " << update_vector.transpose() << endl;

        // Update rotation of the poses
//        for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
        {
          mrpt::poses::CPose3D pose;
          mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
          rot_manifold[0] = update_vector(3);
          rot_manifold[1] = update_vector(4,0);
          rot_manifold[2] = update_vector(5,0);
//          rot_manifold[2] = update_vector(3*sensor_id-3,0) / 4; // Limit the turn around the Z (depth) axis
//          rot_manifold[2] = 0; // Limit the turn around the Z (depth) axis
          mrpt::math::CMatrixDouble33 update_rot = pose.exp_rotation(rot_manifold);
  //      cout << "update_rot\n" << update_rot << endl;
          Eigen::Matrix3f update_rot_eig;
          update_rot_eig << update_rot(0,0), update_rot(0,1), update_rot(0,2),
                            update_rot(1,0), update_rot(1,1), update_rot(1,2),
                            update_rot(2,0), update_rot(2,1), update_rot(2,2);
          Rt_estimatedTemp = Rt_estimated;
          Rt_estimatedTemp.block(0,0,3,3) = update_rot_eig * Rt_estimated.block(0,0,3,3);
          Rt_estimatedTemp.block(0,3,3,1) = Eigen::Vector3f(update_vector(0), update_vector(1), update_vector(2));
  //      cout << "old rotation" << sensor_id << "\n" << Rt_estimated.block(0,0,3,3) << endl;
  //      cout << "new rotation\n" << Rt_estimatedTemp.block(0,0,3,3) << endl;
        }
//      cout << "It " << it << " Rt_estimated \n" << Rt_estimated << "\n Rt_estimatedTemp \n" << Rt_estimatedTemp << endl;

        accum_error2 = calcCorrespPlanePointError(Rt_estimated);
//        float new_accum_error2 = calcCorrespRotErrorWeight(Rt_estimatedTemp);
        float new_accum_error2 = calcCorrespPlanePointError(Rt_estimatedTemp);

        cout << "New Plane-point error " << new_accum_error2 << " previous " << accum_error2 << endl;
  //    cout << "Closing loop? \n" << Rt_estimated[0].inverse() * Rt_estimated[7] * Rt_78;

        // Assign new rotations
        if(new_accum_error2 < accum_error2)
//          for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
            Rt_estimated = Rt_estimatedTemp;
//            Rt_estimated.block(0,0,3,3) = Rt_estimatedTemp.block(0,0,3,3);

        increment = update_vector .dot (update_vector);
        diff_error = accum_error2 - new_accum_error2;
        ++it;
//      cout << "Iteration " << it << " increment " << increment << " diff_error " << diff_error << endl;
      }

      double calib_end = pcl::getTime();
      cout << "CalibrateRt_plane_point " << calib_end - calib_start << endl;

      return Rt_estimated;
    }

//    Eigen::Vector3f CalibrateTranslation(const mrpt::math::CMatrixDouble &correspondences, const int weightedLS = 0)
    Eigen::Vector3f CalibrateTranslation(const int weightedLS = 0)
    {
//    cout << "CalibrateTranslation Laser-Kinect\n";
//      cout << "Rt_estimated.block(0,0,3,3) \n" << Rt_estimated.block(0,0,3,3) << endl;
      // Calibration system
      Eigen::Matrix3f translationHessian = Eigen::Matrix3f::Zero();
      Eigen::Vector3f translationGradient = Eigen::Vector3f::Zero();

//      Eigen::Vector3f translation2 = Eigen::Vector3f::Zero();

//              translationHessian += v3normal1 * v3normal1.transpose();
//  //            double error = d2 - d1;
//              translationGradient += v3normal1 * (d2 - d1);
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        float d_obs_i = correspondences(i,3);
//        Eigen::Vector3f l_obs_ii; l_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        Eigen::Vector3f c_obs_ii; c_obs_ii << correspondences(i,7), correspondences(i,8), correspondences(i,9);
        float trans_error = -(d_obs_i + n_obs_i.dot(Rt_estimated.block(0,0,3,3) * c_obs_ii));
//      cout << "trans_error " << trans_error << endl;
//      cout << "d_obs_i " << d_obs_i << endl;
//      cout << "c_obs_ii " << c_obs_ii.transpose() << endl;
//      cout << "n_obs_i " << n_obs_i.transpose() << endl;
//      cout << "Rt_estimated.block(0,0,3,3) \n" << Rt_estimated.block(0,0,3,3) << endl;
//        if(weightedLS == 1 && correspondences.getColCount() == 18)
//        {
//          // The weight takes into account the number of inliers of the patch, the distance of the patch's center to the image center and the distance of the plane to the sensor
////          float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));// / correspondences.getRowCount();
//          float weight = correspondences(i,17);
//          translationHessian += weight * (n_obs_i * n_obs_i.transpose() );
//          translationGradient += weight * (n_obs_i * trans_error);
//        }
//        else
        {
          translationHessian += (n_obs_i * n_obs_i.transpose() );
          translationGradient += (n_obs_i * trans_error);
        }
      }

      Eigen::JacobiSVD<Eigen::Matrix3f> svd(translationHessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
//      std::cout << "FIM translation " << svd.singularValues().transpose() << endl;

//      cout << "translationHessian \n" << translationHessian << "\n HessianInv \n" << translationHessian.inverse() << endl;
//      calcFisherInfMat();

      FIM_trans = translationHessian;
      translation = translationHessian.inverse() * translationGradient;
//    std::cout << "translation " << translation.transpose() << std::endl;

      return translation;
    }

    void CalibratePair()
    {
      double calib_start = pcl::getTime();

//      calibrated_Rt = Eigen::Matrix4f::Identity();
      Rt_estimated.block(0,0,3,3) = CalibrateRotation();
      Rt_estimated.block(0,3,3,1) = CalibrateTranslation();
//      std::cout << "Rt_estimated\n" << Rt_estimated << std::endl;

      double calib_end = pcl::getTime();
      cout << "CalibratePair in " << calib_end - calib_start << endl;

      std::cout << "Errors av rot " << calcCorrespRotError(Rt_estimated) << " av trans " << calcCorrespTransError(Rt_estimated) << std::endl;
    }

    /*---------------------------------------------------------------
        Aux. functions needed by ransac_detect_2D_lines
     ---------------------------------------------------------------*/
    void static ransac2Dline_fit_(
      const mrpt::math::CMatrixTemplateNumeric<float> &allData,
      const mrpt::vector_size_t &useIndices,
      std::vector< mrpt::math::CMatrixTemplateNumeric<float> > &fitModels )
    {
      ASSERT_(useIndices.size()==2);

      mrpt::math::TPoint2D p1( allData(0,useIndices[0]),allData(1,useIndices[0]) );
      mrpt::math::TPoint2D p2( allData(0,useIndices[1]),allData(1,useIndices[1]) );

//      try
      {
        mrpt::math::TLine2D line(p1,p2);
        fitModels.resize(1);
        mrpt::math::CMatrixTemplateNumeric<float> &M = fitModels[0];

        M.setSize(1,3);
        for (size_t i=0;i<3;i++)
          M(0,i)=line.coefs[i];
    //  cout << "Line model " << allData(0,useIndices[0]) << " " << allData(1,useIndices[0]) << " " << allData(0,useIndices[1]) << " " << allData(1,useIndices[1]) << " M " << M << endl;
      }
//      catch(exception &)
//      {
//        fitModels.clear();
//        return;
//      }
    }

    void static ransac2Dline_distance_(
      const mrpt::math::CMatrixTemplateNumeric<float> &allData,
      const std::vector< mrpt::math::CMatrixTemplateNumeric<float> > & testModels,
      const float distanceThreshold,
      unsigned int & out_bestModelIndex,
      mrpt::vector_size_t & out_inlierIndices )
    {
      out_inlierIndices.clear();
      out_bestModelIndex = 0;

      if (testModels.empty()) return; // No model, no inliers.

      ASSERTMSG_( testModels.size()==1, mrpt::format("Expected testModels.size()=1, but it's = %u",static_cast<unsigned int>(testModels.size()) ) )
      const mrpt::math::CMatrixTemplateNumeric<float> &M = testModels[0];

      ASSERT_( size(M,1)==1 && size(M,2)==3 )

      mrpt::math::TLine2D line;
      line.coefs[0] = M(0,0);
      line.coefs[1] = M(0,1);
      line.coefs[2] = M(0,2);

      const size_t N = size(allData,2);
      out_inlierIndices.reserve(100);
      for (size_t i=0;i<N;i++)
      {
        const double d = line.distance( mrpt::math::TPoint2D( allData.get_unsafe(0,i),allData.get_unsafe(1,i) ) );
    //  cout << "distance " << d << " " << allData.get_unsafe(0,i) << " " << allData.get_unsafe(1,i) << endl;
        if (d<distanceThreshold)
          out_inlierIndices.push_back(i);
      }
    }

    /** Return "true" if the selected points are a degenerate (invalid) case.
      */
    bool static ransac2Dline_degenerate_(
      const mrpt::math::CMatrixTemplateNumeric<float> &allData,
      const mrpt::vector_size_t &useIndices )
    {
      ASSERT_( useIndices.size()==2 )

      const Eigen::Vector2d origin = Eigen::Vector2d(allData(0,useIndices[0]), allData(1,useIndices[0]));
      const Eigen::Vector2d end = Eigen::Vector2d(allData(0,useIndices[1]), allData(1,useIndices[1]));

      if( (end-origin).norm() < 0.1 )
        return true;

      return false;
    }

    /*---------------------------------------------------------------
            ransac_detect_3D_lines
     ---------------------------------------------------------------*/
//    template<class pointPCL>
    void ransac_detect_3D_lines(
      const pcl::PointCloud<PointT>::Ptr &scan,
      std::vector<line> &lines,
    //	Eigen::Matrix<float,Eigen::Dynamic,6> &lines,
    //	mrpt::vector_size_t	&inliers,
    //	mrpt::math::CMatrixTemplateNumeric<float,Eigen::Dynamic,6> &lines,
      const double           threshold,
      const size_t           min_inliers_for_valid_line
      )
    {
//    	ASSERT_( !scan->empty() )
    //cout << "ransac_detect_2D_lines \n";
      lines.clear();

      if(scan->size() < min_inliers_for_valid_line)
        return;

    //  std::vector<int> indices;
    //  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scanWithNAN(new pcl::PointCloud<pcl::PointXYZRGBA>);
    //  *scanWithNAN = *scan;
    //  pcl::removeNaNFromPointCloud(*scanWithNAN, *scan, indices);

      // The running lists of remaining points after each plane, as a matrix:
      mrpt::math::CMatrixTemplateNumeric<float> remainingPoints( 2, scan->size() );
    //  cout << "scan->size() "	<< scan->size() << endl;
      for(unsigned i=0; i < scan->size(); i++)
      {
//        remainingPoints(0,i) = scan->points[i].x;
        remainingPoints(0,i) = scan->points[i].y;
        remainingPoints(1,i) = scan->points[i].z;
    //  cout << " Pt " << scan->points[i].x << " " << scan->points[i].y << " " << scan->points[i].z;
      }

    //cout << "Size remaining pts " << size(remainingPoints,1) << " " << size(remainingPoints,2) << endl;

      // ---------------------------------------------
      // For each line:
      // ---------------------------------------------
      std::vector<std::pair<size_t, mrpt::math::TLine2D> > out_detected_lines;
    //	while (size(remainingPoints,2)>=2)
      {
        mrpt::vector_size_t	inliers;
        mrpt::math::CMatrixTemplateNumeric<float> this_best_model;

        mrpt::math::RANSAC_Template<float>::execute(
          remainingPoints,
          ransac2Dline_fit_,
          ransac2Dline_distance_,
          ransac2Dline_degenerate_,
          threshold,
          2,  // Minimum set of points
          inliers,
          this_best_model,
          false, // Verbose
          0.99  // Prob. of good result
          );
    //cout << "Size inliers " << inliers.size() << endl;

        // Is this plane good enough?
        if (inliers.size()>=min_inliers_for_valid_line)
        {
          // Add this plane to the output list:
          out_detected_lines.push_back(
            std::make_pair<size_t,mrpt::math::TLine2D>(
              inliers.size(),
              mrpt::math::TLine2D(this_best_model(0,0), this_best_model(0,1),this_best_model(0,2) )
              ) );

          out_detected_lines.rbegin()->second.unitarize();

          line segmentedLine;
    //			int prev_size = size(lines,1);
    ////    cout << "prevSize lines " << prev_size << endl;
    //			lines.setSize(prev_size+1,6);
          float mod_dir = sqrt(1+pow(this_best_model(0,0)/this_best_model(0,1),2));
    //			lines(prev_size,0) = 0; // The reference system for the laser is aligned in the horizontal axis
    //			lines(prev_size,1) = 1/mod_dir;
    //			lines(prev_size,2) = -(this_best_model(0,0)/this_best_model(0,1))/mod_dir;
    //			lines(prev_size,3) = 0;
    //			lines(prev_size,4) = scan->points[inliers[0]].y;
    ////			lines(prev_size,4) = scan->points[inliers[0]].x;
    //			lines(prev_size,5) = scan->points[inliers[0]].z;
    //    cout << "\nLine " << lines(prev_size,0) << " " << lines(prev_size,1) << " " << lines(prev_size,2) << " " << lines(prev_size,3) << " " << lines(prev_size,4) << " " << lines(prev_size,5);
          float center_x = 0.0, center_y = 0.0, center_z = 0.0;
          std::vector<int> inliers_(inliers.size());
          Eigen::Matrix2f M = Eigen::Matrix2f::Zero();
          for(unsigned i=0; i < inliers.size(); i++)
          {
//            center_x += scan->points[inliers[0]].x;
            center_y += scan->points[inliers[i]].y;
            center_z += scan->points[inliers[i]].z;
            inliers_[i] = inliers[i];
            Eigen::Vector2f pt_; pt_ << scan->points[inliers[i]].y, scan->points[inliers[i]].z;
            M += pt_ * pt_.transpose();
          }
//          center_x /= inliers.size();
          center_y /= inliers.size();
          center_z /= inliers.size();
          Eigen::JacobiSVD<Eigen::Matrix2f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
//        cout << "normalV " << svd.singularValues().transpose() << "\n U \n" << svd.matrixU() << "\n V \n" << svd.matrixV() << endl;
//        cout << "n__" << 1/mod_dir << " " << -(this_best_model(0,0)/this_best_model(0,1))/mod_dir << endl;
//          segmentedLine.params << 0, svd.matrixU()(1,0), svd.matrixU()(0,0), 0, center_y, center_z;
          segmentedLine.params << 0, 1/mod_dir, -(this_best_model(0,0)/this_best_model(0,1))/mod_dir, 0, center_y, center_z;
//          segmentedLine.params << 0, 1/mod_dir, -(this_best_model(0,0)/this_best_model(0,1))/mod_dir, 0, center_y, center_z;
          pcl::ExtractIndices<PointT> eifilter(true); // Initializing with true will allow us to extract the removed indices
          eifilter.setInputCloud(scan);
          pcl::IndicesPtr inliersPtr(new std::vector<int>(inliers_));
          eifilter.setIndices(inliersPtr);
          eifilter.filter(*segmentedLine.point_cloud);

          lines.push_back(segmentedLine);

          // Discard the selected points so they are not used again for finding subsequent planes:
          remainingPoints.removeColumns(inliers);
        }
    //		else
    //		{
    //			break; // Do not search for more planes.
    //		}
      }
    }
};

/*! This class is used to gather a set of control planes-lines ...
*/
class ControlPlaneLines
{
  public:

    /*! The plane correspondences between the different Asus sensors */
//    std::map<unsigned, std::map<unsigned, mrpt::math::CMatrixDouble> > mmCorrespondences;

//    /*! The plane-line correspondences */
//    mrpt::math::CMatrixDouble correspondences;

//    ControlPlaneLines()
//    {
//    }

    // Obtain the rigid transformation from 3 matched planes
    static mrpt::math::CMatrixDouble getAlignment( const mrpt::math::CMatrixDouble &correspPL )
    {
      assert(size(correspPL,1) == 10 && size(correspPL,2) == 3);

      CalibPairLaserKinect calibPair;
      Eigen::Matrix4f initOffset = Eigen::Matrix4f::Identity();
      float angle_offset = -90;
      initOffset(0,0) = initOffset(1,1) = cos(angle_offset*3.14159/180);
      initOffset(0,1) = -sin(angle_offset*3.14159/180);
      initOffset(1,0) = -initOffset(0,1);
      calibPair.setInitRt(initOffset);
      calibPair.correspondences = correspPL.transpose();

      //Calculate rotation
      Eigen::Matrix3f Rotation = calibPair.CalibrateRotation();//(correspPL);

      // Calculate translation
      Eigen::Vector3f translation = calibPair.CalibrateTranslation();//(correspPL);

      mrpt::math::CMatrixDouble rigidTransf(4,4);
      rigidTransf(0,0) = Rotation(0,0);
      rigidTransf(0,1) = Rotation(0,1);
      rigidTransf(0,2) = Rotation(0,2);
      rigidTransf(1,0) = Rotation(1,0);
      rigidTransf(1,1) = Rotation(1,1);
      rigidTransf(1,2) = Rotation(1,2);
      rigidTransf(2,0) = Rotation(2,0);
      rigidTransf(2,1) = Rotation(2,1);
      rigidTransf(2,2) = Rotation(2,2);
      rigidTransf(0,3) = translation(0);
      rigidTransf(1,3) = translation(1);
      rigidTransf(2,3) = translation(2);
      rigidTransf(3,0) = 0;
      rigidTransf(3,1) = 0;
      rigidTransf(3,2) = 0;
      rigidTransf(3,3) = 1;

      return rigidTransf;
    }

    // Ransac functions to detect outliers in the plane matching
    static void ransacPlaneAlignment_fit(
            const mrpt::math::CMatrixDouble &correspPL,
            const mrpt::vector_size_t  &useIndices,
            std::vector< mrpt::math::CMatrixDouble > &fitModels )
    //        std::vector< Eigen::Matrix4f > &fitModels )
    {
      ASSERT_(useIndices.size()==3);

//      try
      {
        mrpt::math::CMatrixDouble corresp(10,3);

    //  cout << "Size correspPL: " << endl;
    //  cout << "useIndices " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << endl;
        for(unsigned i=0; i<3; i++)
          corresp.col(i) = correspPL.col(useIndices[i]);

        fitModels.resize(1);
    //    Eigen::Matrix4f &M = fitModels[0];
        mrpt::math::CMatrixDouble &M = fitModels[0];
        M = getAlignment(corresp);
      }
//      catch(exception &)
//      {
//        fitModels.clear();
//        return;
//      }
    }

    static void ransac3Dplane_distance(
            const mrpt::math::CMatrixDouble &corresp,
            const std::vector< mrpt::math::CMatrixDouble > & testModels,
            const double distanceThreshold,
            unsigned int & out_bestModelIndex,
            mrpt::vector_size_t & out_inlierIndices )
    {
      ASSERT_( testModels.size()==1 )
      out_bestModelIndex = 0;
      const mrpt::math::CMatrixDouble &M = testModels[0];

      Eigen::Matrix3f Rotation; Rotation << M(0,0), M(0,1), M(0,2), M(1,0), M(1,1), M(1,2), M(2,0), M(2,1), M(2,2);
      Eigen::Vector3f translation; translation << M(0,3), M(1,3), M(2,3);
//    cout << "RANSAC rot \n" << Rotation << "\n translation " << translation.transpose() << endl;
      ASSERT_( mrpt::math::size(M,1)==4 && mrpt::math::size(M,2)==4 )

      const double angleThreshold = distanceThreshold / 2;

      const size_t N = size(corresp,2);
      out_inlierIndices.clear();
      out_inlierIndices.reserve(100);
      for (size_t i=0;i<N;i++)
      {
        const Eigen::Vector3f n_i = Eigen::Vector3f(corresp(0,i), corresp(1,i), corresp(2,i));
        const Eigen::Vector3f l_i = Rotation * Eigen::Vector3f(corresp(4,i), corresp(5,i), corresp(6,i));
        const Eigen::Vector3f c_i = Rotation * Eigen::Vector3f(corresp(7,i), corresp(8,i), corresp(9,i));
        const float d_error = fabs(corresp(3,i) + n_i.dot(c_i + translation));
        const float angle_error = fabs(n_i .dot (l_i ));
//      cout << "d_error " << d_error << " angle_error " << angle_error << endl;
        if (d_error < distanceThreshold)
         if (angle_error < angleThreshold)
          out_inlierIndices.push_back(i);
      }
    }

    /** Return "true" if the selected points are a degenerate (invalid) case.
      */
    static bool ransacPlaneLine_degenerate(
            const mrpt::math::CMatrixDouble &corresp,
            const mrpt::vector_size_t &useIndices )
    {
      ASSERT_( useIndices.size()==3 )

      const Eigen::Vector3f n_1 = Eigen::Vector3f(corresp(0,useIndices[0]), corresp(1,useIndices[0]), corresp(2,useIndices[0]));
      const Eigen::Vector3f n_2 = Eigen::Vector3f(corresp(0,useIndices[1]), corresp(1,useIndices[1]), corresp(2,useIndices[1]));
      const Eigen::Vector3f n_3 = Eigen::Vector3f(corresp(0,useIndices[2]), corresp(1,useIndices[2]), corresp(2,useIndices[2]));
    //cout << "degenerate " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << " - " << fabs(n_1. dot( n_2. cross(n_3) ) ) << endl;

      if( fabs(n_1. dot( n_2. cross(n_3) ) ) < 0.3 )
        return true;

      return false;
    }

    void trimOutliersPlaneLineRANSAC(mrpt::math::CMatrixDouble &correspPL)
    {
      cout << "trimOutliersPlaneLineRANSAC... " << endl;

    //  assert(correspPL.size() >= 3);
    //  CTicTac tictac;

      if(correspPL.getRowCount() <= 3)
      {
        cout << "Insuficient matched planes " << correspPL.getRowCount() << endl;
//        return Eigen::Matrix4f::Identity();
        return;
      }

      mrpt::math::CMatrixDouble corresp(10, correspPL.getRowCount());
      corresp = correspPL.block(0,0,correspPL.getRowCount(),10).transpose();

      mrpt::vector_size_t inliers;
    //  Eigen::Matrix4f best_model;
      mrpt::math::CMatrixDouble best_model;

      mrpt::math::RANSAC::execute(corresp,
                            ransacPlaneAlignment_fit,
                            ransac3Dplane_distance,
                            ransacPlaneLine_degenerate,
                            0.08, // distance threshold
                            3,  // Minimum set of correspondences
                            inliers,
                            best_model,
                            false,   // Verbose
                            0.5, // probGoodSample
                            5000 // maxIter
                            );

    //  cout << "Computation time: " << tictac.Tac()*1000.0/TIMES << " ms" << endl;

//      cout << "Size corresp: " << size(corresp,2) << endl;
      cout << "RANSAC finished: " << inliers.size() << " from " << correspPL.getRowCount() << ". \nBest model: \n" << best_model << endl;
    //        cout << "Best inliers: " << best_inliers << endl;

      mrpt::math::CMatrixDouble trimMatchedPlanes(inliers.size(), 10);
      std::vector<double> row;
      for(unsigned i=0; i < inliers.size(); i++)
        trimMatchedPlanes.row(i) = correspPL.row(inliers[i]);

      correspPL = trimMatchedPlanes;
    }

//    /*! Load the plane correspondences between the different Asus sensors from file */
//    void savePlaneCorrespondences(const std::string &planeCorrespDirectory)
//    {
//      correspondences.saveToTextFile( mrpt::format("%s/correspondences_PlaneLine.txt", planeCorrespDirectory.c_str() ) );
//
////      for(std::map<unsigned, std::map<unsigned, mrpt::math::CMatrixDouble> >::iterator it_pair1=mmCorrespondences.begin();
////          it_pair1 != mmCorrespondences.end(); it_pair1++)
////      {
////        for(std::map<unsigned, mrpt::math::CMatrixDouble>::iterator it_pair2=it_pair1->second.begin();
////            it_pair2 != it_pair1->second.end(); it_pair2++)
////        {
////          if(it_pair2->second.getRowCount() > 0)
////            it_pair2->second.saveToTextFile( mrpt::format("%s/correspondences_%u_%u.txt", planeCorrespDirectory.c_str(), it_pair1->first, it_pair2->first) );
////        }
////      }
//    }

    /*! Load the plane correspondences between a pair of Asus sensors from file */
    mrpt::math::CMatrixDouble getPlaneCorrespondences(const std::string matchedPlanesFile)
    {
      assert( fexists(matchedPlanesFile.c_str()) );

      mrpt::math::CMatrixDouble correspMat;
      correspMat.loadFromTextFile(matchedPlanesFile);
//      std::cout << "Load ControlPlaneLines " << sensor_id << " and " << sensor_corresp << std::endl;
      std::cout << correspMat.getRowCount() << " correspondences " << std::endl;

      return correspMat;
    }

//    /*! Load the plane correspondences between the different Asus sensors from file */
//    void loadPlaneCorrespondences(const std::string planeCorrespDirectory)
//    {
//      mmCorrespondences.clear();
//      for(unsigned sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
//      {
//        mmCorrespondences[sensor_id] = std::map<unsigned, mrpt::math::CMatrixDouble>();
//        for(unsigned sensor_corresp = sensor_id+1; sensor_corresp < NUM_ASUS_SENSORS; sensor_corresp++)
//        {
//          std::string fileCorresp = mrpt::format("%s/correspondences_%u_%u.txt", planeCorrespDirectory.c_str(), sensor_id, sensor_corresp);
//          if( fexists(fileCorresp.c_str()) )
//          {
//            mrpt::math::CMatrixDouble correspMat;
//            correspMat.loadFromTextFile(fileCorresp);
//            mmCorrespondences[sensor_id][sensor_corresp] = correspMat;
////          std::cout << "Load ControlPlaneLines " << sensor_id << " and " << sensor_corresp << std::endl;
////          std::cout << correspMat.getRowCount() << " correspondences " << std::endl;
//          }
//        }
//      }
//    }
//
//
//    /*! Print the number of correspondences and the conditioning number to the standard output */
//    void printConditioning()
//    {
//      cout << "Conditioning\n";
//      for(unsigned sensor_id = 0; sensor_id < 7; sensor_id++)
//        cout << mmCorrespondences[sensor_id][sensor_id+1].getRowCount() << "\t";
//      cout << mmCorrespondences[0][7].getRowCount() << "\t";
//      cout << endl;
//      for(unsigned sensor_id = 0; sensor_id < 8; sensor_id++)
//        cout << conditioning[sensor_id] << "\t";
//      cout << endl;
//    }

};

#endif
