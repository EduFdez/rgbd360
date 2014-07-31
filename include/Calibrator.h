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

#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <Calib360.h>
#include <Frame360.h>

/*! This class is used to gather a set of control planes (they are analogous to the control points used to create panoramic images with a regular
    camera) from a sequence of RGBD360 observations. They permit to find the rigid transformations (extrinsic calibration) between the different
    Asus XPL sensors of the RGBD360 device.
*/
class ControlPlanes
{
  public:

    /*! The plane correspondences between the different Asus sensors */
    std::map<unsigned, std::map<unsigned, mrpt::math::CMatrixDouble> > mmCorrespondences;

    /*! Conditioning numbers used to indicate if there is enough reliable information to calculate the extrinsic calibration */
    float conditioning[8];

    /*! Rotation covariance matrices from adjacent sensors */
    Eigen::Matrix3f covariances[8];

    ControlPlanes()
    {
      std::fill(conditioning, conditioning+8, 9999.9);
//      conditioning[0] = 1; // The first sensor is fixed
//      std::fill(weight_pair, weight_pair+8, 0.0);
//      std::fill(conditioning_measCount, conditioning_measCount+8, 0);
      std::fill(covariances, covariances+8, Eigen::Matrix3f::Zero());
    }

    /*! Load the plane correspondences between the different Asus sensors from file */
    void savePlaneCorrespondences(const std::string &planeCorrespDirectory)
    {
      for(std::map<unsigned, std::map<unsigned, mrpt::math::CMatrixDouble> >::iterator it_pair1=mmCorrespondences.begin();
          it_pair1 != mmCorrespondences.end(); it_pair1++)
      {
        for(std::map<unsigned, mrpt::math::CMatrixDouble>::iterator it_pair2=it_pair1->second.begin();
            it_pair2 != it_pair1->second.end(); it_pair2++)
        {
          if(it_pair2->second.getRowCount() > 0)
            it_pair2->second.saveToTextFile( mrpt::format("%s/correspondences_%u_%u.txt", planeCorrespDirectory.c_str(), it_pair1->first, it_pair2->first) );
        }
      }
    }

    /*! Load the plane correspondences between a pair of Asus sensors from file */
    mrpt::math::CMatrixDouble getPlaneCorrespondences(const std::string matchedPlanesFile)
    {
      assert( fexists(matchedPlanesFile.c_str()) );

      mrpt::math::CMatrixDouble correspMat;
      correspMat.loadFromTextFile(matchedPlanesFile);
//      std::cout << "Load ControlPlanes " << sensor_id << " and " << sensor_corresp << std::endl;
      std::cout << correspMat.getRowCount() << " correspondences " << std::endl;

      return correspMat;
    }

    /*! Load the plane correspondences between the different Asus sensors from file */
    void loadPlaneCorrespondences(const std::string planeCorrespDirectory)
    {
      mmCorrespondences.clear();
      for(unsigned sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
      {
        mmCorrespondences[sensor_id] = std::map<unsigned, mrpt::math::CMatrixDouble>();
        for(unsigned sensor_corresp = sensor_id+1; sensor_corresp < NUM_ASUS_SENSORS; sensor_corresp++)
        {
          std::string fileCorresp = mrpt::format("%s/correspondences_%u_%u.txt", planeCorrespDirectory.c_str(), sensor_id, sensor_corresp);
          if( fexists(fileCorresp.c_str()) )
          {
            mrpt::math::CMatrixDouble correspMat;
            correspMat.loadFromTextFile(fileCorresp);
            mmCorrespondences[sensor_id][sensor_corresp] = correspMat;
//          std::cout << "Load ControlPlanes " << sensor_id << " and " << sensor_corresp << std::endl;
//          std::cout << correspMat.getRowCount() << " correspondences " << std::endl;
          }
        }
      }
    }

    /*! Get the rate of inliers near the border of the sensor (the border nearer to the next lower index Asus sensor) */
    float inliersUpperFringe(mrpt::pbmap::Plane &plane, float fringeWidth) // This only works for QVGA resolution
    {
      unsigned count = 0;
      unsigned im_size = 320 * 240;
      unsigned limit = fringeWidth * im_size;
      for(unsigned i=0; i < plane.inliers.size(); ++i)
        if(plane.inliers[i] < limit)
          ++count;

      return float(count) / (im_size*fringeWidth);
    }

    /*! Get the rate of inliers near the border of the sensor (the border nearer to the next upper index Asus sensor) */
    float inliersLowerFringe(mrpt::pbmap::Plane &plane, float fringeWidth)
    {
      unsigned count = 0;
      unsigned im_size = 320 * 240;
      unsigned limit = (1 - fringeWidth) * im_size;
      for(unsigned i=0; i < plane.inliers.size(); ++i)
        if(plane.inliers[i] > limit)
          ++count;

      return float(count) / (im_size*fringeWidth);
    }

    /*! Print the number of correspondences and the conditioning number to the standard output */
    void printConditioning()
    {
      cout << "Conditioning\n";
      for(unsigned sensor_id = 0; sensor_id < 7; sensor_id++)
        cout << mmCorrespondences[sensor_id][sensor_id+1].getRowCount() << "\t";
      cout << mmCorrespondences[0][7].getRowCount() << "\t";
      cout << endl;
      for(unsigned sensor_id = 0; sensor_id < 8; sensor_id++)
        cout << conditioning[sensor_id] << "\t";
      cout << endl;
    }

//    /*! Update adjacent conditioning (information between a pair of adjacent sensors) */
//    void updateAdjacentConditioning(unsigned couple_id, pair< Eigen::Vector4f, Eigen::Vector4f> &match)
//    {
//      ++conditioning_measCount[couple_id];
//      covariances[couple_id] += match.second.head(3) * match.first.head(3).transpose();
////      Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariances[couple_id], Eigen::ComputeFullU | Eigen::ComputeFullV);
////      conditioning[couple_id] = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//    }

    /*! Calculate adjacent conditioning (information between a pair of adjacent sensors) */
    void calcAdjacentConditioning(unsigned couple_id)
    {
//      if(conditioning_measCount[couple_id] > 3)
//      {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariances[couple_id], Eigen::ComputeFullU | Eigen::ComputeFullV);
        conditioning[couple_id] = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//      }
    }
};

/*! This class contains the functionality to calibrate the extrinsic parameters of a pair of non-overlapping depth cameras.
 *  This extrinsic calibration is obtained by matching planes that are observed by both Asus XPL sensors at the same time.
 */
class PairCalibrator
{
  public:

    /*! The extrinsic matrix estimated by this calibration method */
    Eigen::Matrix4f Rt_estimated;

    Eigen::Matrix3f rotation;

    Eigen::Vector3f translation;

    /*! The plane correspondences between the pair of Asus sensors */
    mrpt::math::CMatrixDouble correspondences;
    mrpt::math::CMatrixDouble correspondences_cov;

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
      float accum_error2 = 0.0;
//      float accum_error_deg = 0.0;
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
//        float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));
        float weight = 1.0;
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        Eigen::Vector3f n_ii = Rot_ * n_obs_ii;
        Eigen::Vector3f rot_error = (n_obs_i - n_ii);
        accum_error2 += weight * fabs(rot_error.dot(rot_error));
//        accum_error_deg += acos(fabs(rot_error.dot(rot_error)));
      }

//      std::cout << "AvError deg " << accum_error_deg/correspondences.getRowCount() << std::endl;
      return accum_error2/correspondences.getRowCount();
    }

//    float calcCorrespTransError(Eigen::Matrix3f &Rot_)
//    {
//      float accum_error2 = 0.0;
//      float accum_error_m = 0.0;
//      for(unsigned i=0; i < correspondences.getRowCount(); i++)
//      {
////        float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));
//        float weight = 1.0;
//        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
//        float trans_error = (correspondences(i,3) - correspondences(i,7) + n_obs_i.dot());
//        accum_error2 += weight * fabs(rot_error.dot(rot_error));
//        accum_error_deg += acos(fabs(rot_error.dot(rot_error)));
//      }
//
//      std::cout << "AvError deg " << accum_error_deg/correspondences.getRowCount() << std::endl;
//      return accum_error2/correspondences.getRowCount();
//    }

    Eigen::Vector3f calcScoreRotation(Eigen::Vector3f &n1, Eigen::Vector3f &n2)
    {
      Eigen::Vector3f score = - skew(Rt_estimated.block(0,0,3,3)*n2) * n1;

      return score;
    }

    Eigen::Matrix3f calcFIMRotation()
    {
      Eigen::Matrix3f FIM = Eigen::Matrix3f::Zero();

      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
//          float weight = (inliers / correspondences(i,3)) / correspondences.getRowCount()
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);

        Eigen::Vector3f score = calcScoreRotation(n_obs_i, n_obs_ii);

        FIM += score * score.transpose();
      }

      return FIM;
    }

    Eigen::Vector3f calcScoreTranslation(Eigen::Vector3f &n1, float &d1, float &d2)
    {
      Eigen::Vector3f score = (d1 - d2) * n1;

//      score[0] = fabs(score[0]);
//      score[1] = fabs(score[1]);
//      score[2] = fabs(score[2]);

      return score;
    }

    Eigen::Matrix3f calcFIMTranslation()
    {
      Eigen::Matrix3f FIM = Eigen::Matrix3f::Zero();

      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
//          float weight = (inliers / correspondences(i,3)) / correspondences.getRowCount()
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
//        Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        float d_obs_i = correspondences(i,3);
        float d_obs_ii = correspondences(i,7);

        Eigen::Vector3f score = calcScoreTranslation(n_obs_i, d_obs_i, d_obs_ii);

        FIM += score * score.transpose();
      }

      return FIM;
    }

    /*! Load the plane correspondences between the pair of Asus sensors from file */
    void loadPlaneCorrespondences(const std::string matchedPlanesFile)
    {
      assert( fexists(matchedPlanesFile.c_str()) );

      correspondences.loadFromTextFile(matchedPlanesFile);
//      std::cout << "Load ControlPlanes " << sensor_id << " and " << sensor_corresp << std::endl;
      std::cout << correspondences.getRowCount() << " correspondences " << std::endl;
    }

//    Eigen::Matrix3f calcFisherInfMat(const int weightedLS = 0)
//    {
//      // Calibration system
//      Eigen::Matrix3f rotationCov = Eigen::Matrix3f::Zero();
//
//      Eigen::Matrix3f FIM_rot = Eigen::Matrix3f::Zero();
//      Eigen::Matrix3f FIM_trans = Eigen::Matrix3f::Zero();
//      Eigen::Vector3f score;
//
//      float accum_error2 = 0;
////      rotationCov += v3normal2 * v3normal1.transpose();
//      for(unsigned i=0; i < correspondences.getRowCount(); i++)
//      {
////          float weight = (inliers / correspondences(i,3)) / correspondences.getRowCount()
//        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
//        Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
//        Eigen::Vector3f n_i = n_obs_i;
//        Eigen::Vector3f n_ii = Rt_estimated.block(0,0,3,3) * n_obs_ii;
//        Eigen::Vector3f rot_error = (n_i - n_ii);
//        accum_error2 += fabs(rot_error.dot(rot_error));
//
//        if(weightedLS == 1 && correspondences.getColCount() == 10)
//        {
//          float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));// / correspondences.getRowCount();
//          rotationCov += weight * n_obs_ii * n_obs_i.transpose();
//        }
//        else
//          rotationCov += n_obs_ii * n_obs_i.transpose();
//
//        float d_obs_i = correspondences(i,3);
//        float d_obs_ii = correspondences(i,7);
//        score = calcScoreRotation(n_obs_i, n_obs_ii);
//        FIM_rot += score * score.transpose();
//        score = calcScoreTranslation(n_obs_i, d_obs_i, d_obs_ii);
//        FIM_trans += score * score.transpose();
////      std::cout << "\nFIM_rot \n" << FIM_rot << "\nrotationCov \n" << rotationCov << "\nFIM_trans \n" << FIM_trans << "\n det " << FIM_rot.determinant() << "\n det2 " << FIM_trans.determinant() << std::endl;
//      }
//      Eigen::JacobiSVD<Eigen::Matrix3f> svd(rotationCov, Eigen::ComputeFullU | Eigen::ComputeFullV);
//      float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//
//      float minFIM_rot = std::min(FIM_rot(0,0), std::min(FIM_rot(1,1), FIM_rot(2,2)));
//      float minFIM_trans = std::min(FIM_trans(0,0), std::min(FIM_trans(1,1), FIM_trans(2,2)));
////      std::cout << "minFIM_rot " << minFIM_rot << " " << minFIM_trans << " conditioning " << conditioning << " numCorresp " << correspondences.getRowCount() << std::endl;
//      std::cout << "\nFIM_rot \n" << FIM_rot << std::endl;
//      std::cout << "\nFIM_trans \n" << FIM_trans << std::endl;
//    }

    Eigen::Matrix3f CalibrateRotation(const int weightedLS = 0)
    {
      // Calibration system
      Eigen::Matrix3f rotationCov = Eigen::Matrix3f::Zero();

//      Eigen::Matrix3f FIM_rot = Eigen::Matrix3f::Zero();
//      Eigen::Matrix3f FIM_trans = Eigen::Matrix3f::Zero();
//      Eigen::Vector3f score;

      float accum_error2 = 0;
//      rotationCov += v3normal2 * v3normal1.transpose();
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
//          float weight = (inliers / correspondences(i,3)) / correspondences.getRowCount()
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        Eigen::Vector3f n_i = n_obs_i;
        Eigen::Vector3f n_ii = Rt_estimated.block(0,0,3,3) * n_obs_ii;
        Eigen::Vector3f rot_error = (n_i - n_ii);
        accum_error2 += fabs(rot_error.dot(rot_error));

        if(weightedLS == 1 && correspondences.getColCount() == 10)
        {
          float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));// / correspondences.getRowCount();
          rotationCov += weight * n_obs_ii * n_obs_i.transpose();
        }
        else
          rotationCov += n_obs_ii * n_obs_i.transpose();

//        float d_obs_i = correspondences(i,3);
//        float d_obs_ii = correspondences(i,7);
//        score = calcScoreRotation(n_obs_i, n_obs_ii);
//        FIM_rot += score * score.transpose();
//        score = calcScoreTranslation(n_obs_i, d_obs_i, d_obs_ii);
//        FIM_trans += score * score.transpose();
////      std::cout << "\nFIM_rot \n" << FIM_rot << "\nrotationCov \n" << rotationCov << "\nFIM_trans \n" << FIM_trans << "\n det " << FIM_rot.determinant() << "\n det2 " << FIM_trans.determinant() << std::endl;
      }
//      float minFIM_rot = std::min(FIM_rot(0,0), std::min(FIM_rot(1,1), FIM_rot(2,2)));
//      std::cout << "minFIM_rot " << minFIM_rot << std::endl;// << " " << calcCorrespRotError(Rt_estimated) << std::endl;
//      std::cout << "accum_rot_error2 av_deg " << acos(accum_error2/correspondences.getRowCount()) << std::endl;// << " " << calcCorrespRotError(Rt_estimated) << std::endl;
//      std::cout << "Rt_estimated\n" << Rt_estimated << std::endl;

      // Calculate calibration Rt
//      cout << "Solve system\n";
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(rotationCov, Eigen::ComputeFullU | Eigen::ComputeFullV);
      float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
      std::cout << "conditioning " << conditioning << std::endl;
//      if(conditioning > 20000)
      if(conditioning > 100)
      {
        std::cout << "Bad conditioning " << conditioning << std::endl;
        return Eigen::Matrix3f::Identity();
      }

      rotation = svd.matrixV() * svd.matrixU().transpose();
      double det = rotation.determinant();
      if(det != 1)
      {
        Eigen::Matrix3f aux;
        aux << 1, 0, 0, 0, 1, 0, 0, 0, det;
        rotation = svd.matrixV() * aux * svd.matrixU().transpose();
      }
      std::cout << "accum_rot_error2 av_deg" << acos(calcCorrespRotError(rotation)) << std::endl;

      return rotation;
    }


    Eigen::Matrix3f CalibrateRotationD(const int weightedLS = 0)
    {
      // Calibration system
      Eigen::Matrix3d rotationCov = Eigen::Matrix3d::Zero();
      Eigen::Matrix3d rotation_estim = Rt_estimated.block(0,0,3,3).cast<double>();

      double accum_error2 = 0;
//      rotationCov += v3normal2 * v3normal1.transpose();
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
//          double weight = (inliers / correspondences(i,3)) / correspondences.getRowCount()
        Eigen::Vector3d n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3d n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        Eigen::Vector3d n_i = n_obs_i;
        Eigen::Vector3d n_ii = rotation_estim * n_obs_ii;
        Eigen::Vector3d rot_error = (n_i - n_ii);
        accum_error2 += fabs(rot_error.dot(rot_error));

        if(weightedLS == 1 && correspondences.getColCount() == 10)
        {
          double weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));// / correspondences.getRowCount();
          rotationCov += weight * n_obs_ii * n_obs_i.transpose();
        }
        else
          rotationCov += n_obs_ii * n_obs_i.transpose();
      }
      std::cout << "accum_rot_error2 av deg" << acos(accum_error2/correspondences.getRowCount()) << std::endl;// << " " << calcCorrespRotError(Rt_estimated) << std::endl;
      std::cout << "Rt_estimated\n" << Rt_estimated << std::endl;

      // Calculate calibration Rt
      cout << "Solve system\n";
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotationCov, Eigen::ComputeFullU | Eigen::ComputeFullV);
      double conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
      std::cout << "conditioning " << conditioning << std::endl;
//      if(conditioning > 20000)
      if(conditioning > 100)
      {
        std::cout << "Bad conditioning " << conditioning << std::endl;
        return Eigen::Matrix3f::Identity();
      }

      Eigen::Matrix3d rotation = svd.matrixV() * svd.matrixU().transpose();
      double det = rotation.determinant();
      if(det != 1)
      {
        Eigen::Matrix3d aux;
        aux << 1, 0, 0, 0, 1, 0, 0, 0, det;
        rotation = svd.matrixV() * aux * svd.matrixU().transpose();
      }
      std::cout << "accum_rot_error2 optim av deg" << acos(calcCorrespRotError(Rt_estimated)) << std::endl;
      cout << "Rotation (double)\n" << rotation << endl;

      return rotation.cast<float>();
//      return Eigen::Matrix3f::Identity();
    }

    /*! Get the rotation of each sensor in the multisensor RGBD360 setup */
    Eigen::Matrix3f CalibrateRotationManifold(int weightedLS = 0)
    {
    cout << "CalibrateRotationManifold...\n";
      Eigen::Matrix<float,3,3> hessian;
      Eigen::Matrix<float,3,1> gradient;
      Eigen::Matrix<float,3,1> update_vector;
      Eigen::Matrix3f jacobian_rot_i, jacobian_rot_ii; // Jacobians of the rotation
      float accum_error2;
      float av_angle_error;
      unsigned numControlPlanes;

      // Load the extrinsic calibration from the device' specifications
//        Rt_estimated[sensor_id] = Rt_specs[sensor_id];

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
            Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
//            Eigen::Vector3f n_i = n_obs_i;
            Eigen::Vector3f n_ii = Rt_estimated.block(0,0,3,3) * n_obs_ii;
//            jacobian_rot_i = skew(-n_i);
            jacobian_rot_ii = skew(n_ii);
            Eigen::Vector3f rot_error = (n_obs_i - n_ii);
            accum_error2 += fabs(rot_error.dot(rot_error));
            av_angle_error += acos(n_obs_i.dot(n_ii));
            numControlPlanes++;
//          cout << "rotation error_i " << rot_error.transpose() << endl;
            if(weightedLS == 1 && correspondences.getColCount() == 10)
            {
              // The weight takes into account the number of inliers of the patch, the distance of the patch's center to the image center and the distance of the plane to the sensor
//              float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));// / correspondences.getRowCount();
//              hessian += weight * (jacobian_rot_ii.transpose() * jacobian_rot_ii);
//              gradient += weight * (jacobian_rot_ii.transpose() * rot_error);
              Eigen::Matrix3f information;
              information << correspondences(i,8), correspondences(i,9), correspondences(i,10), correspondences(i,11),
                            correspondences(i,9), correspondences(i,12), correspondences(i,13), correspondences(i,14),
                            correspondences(i,10), correspondences(i,13), correspondences(i,15), correspondences(i,16),
                            correspondences(i,11), correspondences(i,14), correspondences(i,16), correspondences(i,17);
              hessian += jacobian_rot_ii.transpose() * information.block(0,0,3,3) * jacobian_rot_ii;
              gradient += jacobian_rot_ii.transpose() * information.block(0,0,3,3) * rot_error;
            }
            else
            {
              hessian += jacobian_rot_ii.transpose() * jacobian_rot_ii;
              gradient += jacobian_rot_ii.transpose() * rot_error;
            }

            Eigen::JacobiSVD<Eigen::Matrix3f> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
            float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
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

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
        float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//        Eigen::Matrix3f cov;
//        svd.pinv(cov);
//        std::cout << "hessian \n" << hessian << "inv\n" << hessian.inverse() << "\ncov \n" << cov << std::endl;

//        std::cout << "conditioning " << conditioning << std::endl;
//        if(conditioning > 100)
//          return Eigen::Matrix3f::Identity();

        // Solve the rotation
        update_vector = -hessian.inverse() * gradient;
//      cout << "update_vector " << update_vector.transpose() << endl;

        // Update rotation of the poses
//        for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
        {
          mrpt::poses::CPose3D pose;
          mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
          rot_manifold[0] = update_vector(0,0);
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

      std::cout << "ErrorCalibRotation " << accum_error2 << " " << av_angle_error << std::endl;
      std::cout << "Rotation \n"<< Rt_estimated.block(0,0,3,3) << std::endl;
    }

    Eigen::Vector3f CalibrateTranslation(const int weightedLS = 0)
    {
      // Calibration system
      Eigen::Matrix3f translationHessian = Eigen::Matrix3f::Zero();
      Eigen::Vector3f translationGradient = Eigen::Vector3f::Zero();

//      Eigen::Vector3f translation2 = Eigen::Vector3f::Zero();
//      Eigen::Vector3f sumNormals = Eigen::Vector3f::Zero();

//              translationHessian += v3normal1 * v3normal1.transpose();
//  //            double error = d2 - d1;
//              translationGradient += v3normal1 * (d2 - d1);
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
//        Eigen::Vector3f n_i = Rt_estimated[sensor_id].block(0,0,3,3) * n_obs_i;
//        Eigen::Vector3f n_ii = Rt_estimated.block(0,0,3,3) * n_obs_ii;
        float trans_error = (correspondences(i,7) - correspondences(i,3));
//        accum_error2 += trans_error * trans_error;

//        translation2[0] += n_obs_i[0] * trans_error;
//        translation2[1] += n_obs_i[1] * trans_error;
//        translation2[2] += n_obs_i[2] * trans_error;
//        sumNormals += n_obs_i;

        if(weightedLS == 1 && correspondences.getColCount() == 18)
        {
          // The weight takes into account the number of inliers of the patch, the distance of the patch's center to the image center and the distance of the plane to the sensor
//          float weight = (correspondences(i,8) / (correspondences(i,3) * correspondences(i,9)));// / correspondences.getRowCount();
          float weight = correspondences(i,17);
          translationHessian += weight * (n_obs_i * n_obs_i.transpose() );
          translationGradient += weight * (n_obs_i * trans_error);
        }
        else
        {
          translationHessian += (n_obs_i * n_obs_i.transpose() );
          translationGradient += (n_obs_i * trans_error);
        }
      }

      Eigen::JacobiSVD<Eigen::Matrix3f> svd(translationHessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
      std::cout << "FIM translation " << svd.singularValues().transpose() << endl;

//      cout << "translationHessian \n" << translationHessian << "\n HessianInv \n" << translationHessian.inverse() << endl;
//      calcFisherInfMat();

      translation = translationHessian.inverse() * translationGradient;

//      translation2[0] /= sumNormals[0];
//      translation2[1] /= sumNormals[1];
//      translation2[2] /= sumNormals[2];
//      std::cout << "translation " << translation.transpose() << " translation2 " << translation2.transpose() << std::endl;

      return translation;
    }

    void CalibratePair()
    {
//      calibrated_Rt = Eigen::Matrix4f::Identity();
      Rt_estimated.block(0,0,3,3) = CalibrateRotation();
      Rt_estimated.block(0,3,3,1) = CalibrateTranslation();
//      std::cout << "Rt_estimated\n" << Rt_estimated << std::endl;

      // Calculate average error
      float av_rot_error = 0;
      float av_trans_error = 0;
      for(unsigned i=0; i < correspondences.getRowCount(); i++)
      {
        Eigen::Vector3f n_obs_i; n_obs_i << correspondences(i,0), correspondences(i,1), correspondences(i,2);
        Eigen::Vector3f n_obs_ii; n_obs_ii << correspondences(i,4), correspondences(i,5), correspondences(i,6);
        av_rot_error += fabs(acos(n_obs_i .dot( rotation * n_obs_ii ) ));
        av_trans_error += fabs(correspondences(i,3) - correspondences(i,7) - n_obs_i .dot(translation));
//        params_error += plane_correspondences1[i] .dot( plane_correspondences2[i] );
      }
      av_rot_error /= correspondences.getRowCount();
      av_trans_error /= correspondences.getRowCount();
      std::cout << "Errors n.n " << calcCorrespRotError(Rt_estimated) << " av deg " << av_rot_error*180/PI << " av trans " << av_trans_error << std::endl;
    }
};

/*! This class contains the functionality to calibrate the extrinsic parameters of the omnidirectional RGB-D device (RGBD360).
 *  This extrinsic calibration is obtained by matching planes that are observed by several Asus XPL sensors at the same time.
 */
class Calibrator : public Calib360
{
  private:

    /*! Conditioning of the system of equations used to indicate if there is enough reliable information to calculate the extrinsic calibration */
    float conditioning;

    /*! Hessian of the of the least-squares problem. This container is used indifferently for both rotation and translation as both systems are decoupled and have the same dimensions */
    Eigen::Matrix<float,21,21> hessian;

    /*! Gradient of the of the least-squares problem. This container is used indifferently for both rotation and translation as both systems are decoupled and have the same dimensions */
    Eigen::Matrix<float,21,1> gradient;

  public:

    /*! The plane correspondences between the different Asus sensors */
//    std::map<unsigned, std::map<unsigned, mrpt::math::CMatrixDouble> > mmCorrespondences;
    ControlPlanes matchedPlanes;

    /*! The extrinsic parameters given by the construction specifications of the omnidirectional sensor */
    Eigen::Matrix4f Rt_specs[NUM_ASUS_SENSORS];

    /*! The extrinsic parameter matrices estimated by this calibration method */
    Eigen::Matrix4f Rt_estimated[NUM_ASUS_SENSORS];

//    Calibrator()
    // :
//      successful(false)
//    {
//      string mouseMsg2D ("Mouse coordinates in image viewer");
//      string keyMsg2D ("Key event for image viewer");
//      viewer.registerKeyboardCallback(&RGBD360_Visualizer::keyboard_callback, *this, static_cast<void*> (&keyMsg2D));
//    }

    /*! Load the extrinsic parameters given by the construction specifications of the omnidirectional sensor */
    void loadConstructionSpecs()
    {
      Rt_specs[0] = Eigen::Matrix4f::Identity();
      Rt_specs[0](2,3) = 0.055; // This is the theoretical distance from the first Asus sensor to the center

      // The pose of each sensor is given by a turn of 45deg around the vertical axis which passes through the device' center
      Eigen::Matrix4f turn45deg = Eigen::Matrix4f::Identity();
      turn45deg(1,1) = turn45deg(2,2) = cos(45*PI/180);
      turn45deg(1,2) = -sin(45*PI/180);
      turn45deg(2,1) = -turn45deg(1,2);
      for(unsigned sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
        Rt_specs[sensor_id] = turn45deg * Rt_specs[sensor_id-1];
    }

    /*! Get the sum of squared rotational errors for the input extrinsic matrices. TODO: the input argument of this function is unsafe -> fix it */
    float calcCorrespRotError(Eigen::Matrix4f *Rt_)
    {
      float accum_error2 = 0; // Accumulated square errors
      for(unsigned sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
      {
        for(std::map<unsigned, mrpt::math::CMatrixDouble>::iterator it_pair=matchedPlanes.mmCorrespondences[sensor_id].begin();
            it_pair != matchedPlanes.mmCorrespondences[sensor_id].end(); it_pair++)
        {
          for(unsigned i=0; i < it_pair->second.getRowCount(); i++)
          {
//          float weight = (inliers / it_pair->second(i,3)) / it_pair->second.getRowCount()
            Eigen::Vector3f n_obs_i; n_obs_i << it_pair->second(i,0), it_pair->second(i,1), it_pair->second(i,2);
            Eigen::Vector3f n_obs_ii; n_obs_ii << it_pair->second(i,4), it_pair->second(i,5), it_pair->second(i,6);
            Eigen::Vector3f n_i = Rt_[sensor_id].block(0,0,3,3) * n_obs_i;
            Eigen::Vector3f n_ii = Rt_[it_pair->first].block(0,0,3,3) * n_obs_ii;
            Eigen::Vector3f rot_error = (n_i - n_ii);
            accum_error2 += rot_error.dot(rot_error);
          }
        }
      }

      return accum_error2;
    }

    /*! Get the sum of weighted squared rotational errors for the input extrinsic matrices. TODO: the input argument of this function is unsafe -> fix it */
    float calcCorrespRotErrorWeight(Eigen::Matrix4f *Rt_)
    {
      float accum_error2 = 0; // Accumulated square errors
      for(unsigned sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
      {
        for(std::map<unsigned, mrpt::math::CMatrixDouble>::iterator it_pair=matchedPlanes.mmCorrespondences[sensor_id].begin();
            it_pair != matchedPlanes.mmCorrespondences[sensor_id].end(); it_pair++)
        {
          for(unsigned i=0; i < it_pair->second.getRowCount(); i++)
          {
            float weight = (it_pair->second(i,8) / (it_pair->second(i,3) * it_pair->second(i,9)));
            Eigen::Vector3f n_obs_i; n_obs_i << it_pair->second(i,0), it_pair->second(i,1), it_pair->second(i,2);
            Eigen::Vector3f n_obs_ii; n_obs_ii << it_pair->second(i,4), it_pair->second(i,5), it_pair->second(i,6);
            Eigen::Vector3f n_i = Rt_[sensor_id].block(0,0,3,3) * n_obs_i;
            Eigen::Vector3f n_ii = Rt_[it_pair->first].block(0,0,3,3) * n_obs_ii;
            Eigen::Vector3f rot_error = (n_i - n_ii);
            accum_error2 += weight * rot_error.dot(rot_error);
          }
        }
      }

      return accum_error2;
    }

    /*! Get the sum of squared translational errors for the input extrinsic matrices. TODO: the input argument of this function is unsafe -> fix it */
    float calcCorrespTransError(Eigen::Matrix4f *Rt_)
    {
      float accum_error2 = 0; // Accumulated square errors
//      for(unsigned sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
//      {
//        for(std::map<unsigned, mrpt::math::CMatrixDouble>::iterator it_pair=matchedPlanes.mmCorrespondences[sensor_id].begin();
//            it_pair != matchedPlanes.mmCorrespondences[sensor_id].end(); it_pair++)
//        {
//          for(unsigned i=0; i < it_pair->second.getRowCount(); i++)
//          {
////          float weight = (inliers / it_pair->second(i,3)) / it_pair->second.getRowCount()
//            Eigen::Vector3f n_obs_i; n_obs_i << it_pair->second(i,0), it_pair->second(i,1), it_pair->second(i,2);
//            Eigen::Vector3f n_obs_ii; n_obs_ii << it_pair->second(i,4), it_pair->second(i,5), it_pair->second(i,6);
//            Eigen::Vector3f n_i = Rt_[sensor_id].block(0,0,3,3) * n_obs_i;
//            Eigen::Vector3f n_ii = Rt_[it_pair->first].block(0,0,3,3) * n_obs_ii;
//            rot_error = (n_i - n_ii);
//            accum_error2 += rot_error.dot(rot_error);
//          }
//        }
//      }

//      // Calculate new error
//      double new_accum_error2 = 0;
//      for(sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
//      {
//        for(unsigned i=0; i < it_pair->second.getRowCount(); i++)
//        {
//          Eigen::Vector3f n_obs_i; n_obs_i << it_pair->second(i,0), it_pair->second(i,1), it_pair->second(i,2);
//          Eigen::Vector3f n_obs_ii; n_obs_ii << it_pair->second(i,4), it_pair->second(i,5), it_pair->second(i,6);
//          Eigen::Vector3f n_i = Rt_estimatedTemp[sensor_id].block(0,0,3,3) * n_obs_i;
//          Eigen::Vector3f n_ii = Rt_estimatedTemp[(sensor_id+1)%8].block(0,0,3,3) * n_obs_ii;
////            trans_error = it_pair->second(i,3) + n_i.transpose()*Rt_estimatedTemp[sensor_id].block(0,3,3,1) - (it_pair->second(i,7) + n_ii.transpose()*Rt_estimatedTemp[sensor_id+1].block(0,3,3,1));
//          trans_error = it_pair->second(i,3) - (n_i(0)*Rt_estimatedTemp[sensor_id](0,3) + n_i(1)*Rt_estimatedTemp[sensor_id](1,3) + n_i(2)*Rt_estimatedTemp[sensor_id](2,3))
//                      - (it_pair->second(i,7) - (n_ii(0)*Rt_estimatedTemp[(sensor_id+1)%8](0,3) + n_ii(1)*Rt_estimatedTemp[(sensor_id+1)%8](1,3) + n_ii(2)*Rt_estimatedTemp[(sensor_id+1)%8](2,3)));
////      if(sensor_id==7)
////        cout << "translation error_i LC " << trans_error << endl;
//          new_accum_error2 += trans_error * trans_error;
//        }
//      }
      return accum_error2;
    }

    /*! Get the rotation of each sensor in the multisensor RGBD360 setup */
    void CalibrateRotation(int weightedLS = 0)
    {
    cout << "Calibrate...\n";
      Eigen::Matrix<float,21,1> update_vector;
      Eigen::Matrix3f jacobian_rot_i, jacobian_rot_ii; // Jacobians of the rotation
      float accum_error2;
      float av_angle_error;
      unsigned numControlPlanes;

      // Load the extrinsic calibration from the device' specifications
      for(unsigned sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
        Rt_estimated[sensor_id] = Rt_specs[sensor_id];

      Eigen::Matrix4f Rt_estimatedTemp[NUM_ASUS_SENSORS];
      Rt_estimatedTemp[0] = Rt_estimated[0];

      // Parameters of the Least-Squares optimization
      unsigned _max_iterations = 10;
      float _epsilon_transf = 0.00001;
      float _convergence_error = 0.000001;

      float increment = 1000, diff_error = 1000;
      int it = 0;
      while(it < _max_iterations && increment > _epsilon_transf && diff_error > _convergence_error)
      {
        // Calculate the hessian and the gradient
        hessian = Eigen::Matrix<float,21,21>::Zero(); // Hessian of the rotation of the decoupled system
        gradient = Eigen::Matrix<float,21,1>::Zero(); // Gradient of the rotation of the decoupled system
        accum_error2 = 0.0;
        av_angle_error = 0.0;
        numControlPlanes = 0;

        for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
        {
          assert( matchedPlanes.mmCorrespondences[sensor_id].count(sensor_id+1) && matchedPlanes.mmCorrespondences[sensor_id][sensor_id+1].getRowCount() >= 3 );
//        cout << "sensor_id " << sensor_id << endl;
//        cout << "Rt+1 " << sensor_id << endl;
//        cout << it_pair->second << endl;
          for(std::map<unsigned, mrpt::math::CMatrixDouble>::iterator it_pair=matchedPlanes.mmCorrespondences[sensor_id].begin();
              it_pair != matchedPlanes.mmCorrespondences[sensor_id].end(); it_pair++)
//            if( it_pair->first - sensor_id == 1 || it_pair->first - sensor_id == 7)
          {
            std::cout << "Add pair " << sensor_id << " " << it_pair->first << std::endl;
            int id_corresp1 = 3*(sensor_id-1);
            int id_corresp2 = 3*(it_pair->first - 1);

            for(unsigned i=0; i < it_pair->second.getRowCount(); i++)
            {
//          float weight = (inliers / it_pair->second(i,3)) / it_pair->second.getRowCount()
              Eigen::Vector3f n_obs_i; n_obs_i << it_pair->second(i,0), it_pair->second(i,1), it_pair->second(i,2);
              Eigen::Vector3f n_obs_ii; n_obs_ii << it_pair->second(i,4), it_pair->second(i,5), it_pair->second(i,6);
              Eigen::Vector3f n_i = Rt_estimated[sensor_id].block(0,0,3,3) * n_obs_i;
              Eigen::Vector3f n_ii = Rt_estimated[it_pair->first].block(0,0,3,3) * n_obs_ii;
              jacobian_rot_i = skew(-n_i);
              jacobian_rot_ii = skew(n_ii);
              Eigen::Vector3f rot_error = (n_i - n_ii);
              accum_error2 += rot_error.dot(rot_error);
              av_angle_error += acos(n_i.dot(n_ii));
              numControlPlanes++;
  //          cout << "rotation error_i " << rot_error.transpose() << endl;
              if(weightedLS == 1 && it_pair->second.getColCount() == 18)
              {
                // The weight takes into account the number of inliers of the patch, the distance of the patch's center to the image center and the distance of the plane to the sensor
                float weight = (it_pair->second(i,8) / (it_pair->second(i,3) * it_pair->second(i,9)));// / it_pair->second.getRowCount();
                if(sensor_id != 0) // The pose of the first camera is fixed
                {
                  hessian.block(id_corresp1, id_corresp1, 3, 3) += weight * (jacobian_rot_i.transpose() * jacobian_rot_i);
                  gradient.block(id_corresp1,0,3,1) += weight * (jacobian_rot_i.transpose() * rot_error);

                  // Cross term
                  hessian.block(id_corresp1, id_corresp2, 3, 3) += weight * (jacobian_rot_i.transpose() * jacobian_rot_ii);
                }

                hessian.block(id_corresp2, id_corresp2, 3, 3) += weight * (jacobian_rot_ii.transpose() * jacobian_rot_ii);
                gradient.block(id_corresp2,0,3,1) += weight * (jacobian_rot_ii.transpose() * rot_error);
              }
              else
              {
                if(sensor_id != 0) // The pose of the first camera is fixed
                {
                  hessian.block(id_corresp1, id_corresp1, 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_i;
                  gradient.block(id_corresp1,0,3,1) += jacobian_rot_i.transpose() * rot_error;

                  // Cross term
                  hessian.block(id_corresp1, id_corresp2, 3, 3) += jacobian_rot_i.transpose() * jacobian_rot_ii;
                }
                hessian.block(id_corresp2, id_corresp2, 3, 3) += jacobian_rot_ii.transpose() * jacobian_rot_ii;
                gradient.block(id_corresp2,0,3,1) += jacobian_rot_ii.transpose() * rot_error;
              }

            }

            if(sensor_id != 0) // Fill the lower left triangle with the corresponding cross terms
              hessian.block(id_corresp2, id_corresp1, 3, 3) = hessian.block(id_corresp1, id_corresp2, 3, 3).transpose();
          }
        }

        cout << "Error accumulated " << accum_error2 << endl;

        if( calcConditioning() > threshold_conditioning )
        {
          cout << "\tRotation system is bad conditioned " << conditioning << " threshold " << threshold_conditioning << "\n";
          break;
        }

        // Solve the rotation
        update_vector = -hessian.inverse() * gradient;
      cout << "update_vector " << update_vector.transpose() << endl;

        // Update rotation of the poses
        for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
        {
          mrpt::poses::CPose3D pose;
          mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
          rot_manifold[0] = update_vector(3*sensor_id-3,0);
          rot_manifold[1] = update_vector(3*sensor_id-2,0);
          rot_manifold[2] = update_vector(3*sensor_id-1,0);
//          rot_manifold[2] = update_vector(3*sensor_id-3,0) / 4; // Limit the turn around the Z (depth) axis
//          rot_manifold[2] = 0; // Limit the turn around the Z (depth) axis
          mrpt::math::CMatrixDouble33 update_rot = pose.exp_rotation(rot_manifold);
  //      cout << "update_rot\n" << update_rot << endl;
          Eigen::Matrix3f update_rot_eig;
          update_rot_eig << update_rot(0,0), update_rot(0,1), update_rot(0,2),
                            update_rot(1,0), update_rot(1,1), update_rot(1,2),
                            update_rot(2,0), update_rot(2,1), update_rot(2,2);
          Rt_estimatedTemp[sensor_id] = Rt_estimated[sensor_id];
          Rt_estimatedTemp[sensor_id].block(0,0,3,3) = update_rot_eig * Rt_estimated[sensor_id].block(0,0,3,3);
  //      cout << "old rotation" << sensor_id << "\n" << Rt_estimated[sensor_id].block(0,0,3,3) << endl;
  //      cout << "new rotation\n" << Rt_estimatedTemp[sensor_id].block(0,0,3,3) << endl;
        }

        accum_error2 = calcCorrespRotErrorWeight(Rt_estimated);
        float new_accum_error2 = calcCorrespRotErrorWeight(Rt_estimatedTemp);
//        float new_accum_error2 = calcCorrespRotError(Rt_estimatedTemp);

//        cout << "New rotation error " << new_accum_error2 << endl;
  //    cout << "Closing loop? \n" << Rt_estimated[0].inverse() * Rt_estimated[7] * Rt_78;

        // Assign new rotations
        if(new_accum_error2 < accum_error2)
          for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
            Rt_estimated[sensor_id] = Rt_estimatedTemp[sensor_id];
//            Rt_estimated[sensor_id].block(0,0,3,3) = Rt_estimatedTemp[sensor_id].block(0,0,3,3);

        increment = update_vector .dot (update_vector);
        diff_error = accum_error2 - new_accum_error2;
        ++it;
      cout << "Iteration " << it << " increment " << increment << " diff_error " << diff_error << endl;
      }

      // Make the X axis of the device (the average X axis of all Asus' rotation matrices) coincide with the vertical axis
      Eigen::Matrix3f hessian_rot = Eigen::Matrix3f::Zero();
      Eigen::Vector3f gradient_rot = Eigen::Vector3f::Zero();
      Eigen::Vector3f X_axis; X_axis << 1, 0, 0;
      float rotation_error2 = 0;
      for(unsigned i=0; i < NUM_ASUS_SENSORS; i++)
      {
        Eigen::Vector3f X_pose = Rt_estimated[i].block(0,0,3,1);
        Eigen::Vector3f error = X_axis .cross(X_pose);
        Eigen::Matrix3f jacobian = -skew(X_axis) * skew(X_pose);
        hessian_rot += jacobian.transpose() * jacobian;
        gradient_rot += jacobian.transpose() * error;
        rotation_error2 += error .dot(error);
      }
      Eigen::Vector3f rotation_manifold = - hessian_rot.inverse() * gradient_rot;
      mrpt::poses::CPose3D pose;
      mrpt::math::CArrayNumeric< double, 3 > rot_manifold;
      rot_manifold[0] = 0;
//      rot_manifold[0] = rotation_manifold(0);
      rot_manifold[1] = rotation_manifold(1);
      rot_manifold[2] = rotation_manifold(2);
      mrpt::math::CMatrixDouble33 update_rot = pose.exp_rotation(rot_manifold);
    cout << "manifold update " << rot_manifold.transpose() << "\nupdate_rot\n" << update_rot << endl;
      Eigen::Matrix3f rotation;
      rotation << update_rot(0,0), update_rot(0,1), update_rot(0,2),
                  update_rot(1,0), update_rot(1,1), update_rot(1,2),
                  update_rot(2,0), update_rot(2,1), update_rot(2,2);

//      float new_rotation_error2 = 0;
//      for(unsigned i=0; i<8; i++)
//      {
//        Eigen::Vector3f X_pose = rotation * Rt_estimated[i].block(0,0,3,1);
//        Eigen::Vector3f error = X_axis .cross(X_pose);
//        new_rotation_error2 += error .dot(error);
//      }
//    cout << "Previous error " << rotation_error2 << " new error " << new_rotation_error2 << endl;

      // Rotate the camera system to make the vertical direction correspond to the X axis
      for(unsigned sensor_id=0; sensor_id<8; sensor_id++)
        Rt_estimated[sensor_id].block(0,0,3,3) = rotation * Rt_estimated[sensor_id].block(0,0,3,3);

      std::cout << "ErrorCalibRotation " << accum_error2/numControlPlanes << " " << av_angle_error/numControlPlanes << std::endl;
    }


    /*! Get the translation of each sensor in the multisensor RGBD360 setup. Warning: this method has being implemented to be applied always after rotation calibration */
    void CalibrateTranslation(int weightedLS = 0)
    {
//    cout << "\n Get translation\n";
      hessian = Eigen::Matrix<float,21,21>::Zero(); // Hessian of the translation of the decoupled system
      gradient = Eigen::Matrix<float,21,1>::Zero(); // Gradient of the translation of the decoupled system
      Eigen::Matrix<float,21,1> update_vector;
      Eigen::Matrix<float,21,1> update_translation;
      Eigen::Matrix<float,1,3> jacobian_trans_i, jacobian_trans_ii; // Jacobians of the translation
      float trans_error;
      float accum_error2 = 0.0;
      unsigned numControlPlanes = 0;

      Eigen::Matrix4f Rt_estimatedTemp[NUM_ASUS_SENSORS];
      Rt_estimatedTemp[0] = Rt_estimated[0];

      for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS-1; sensor_id++)
      {
        assert( matchedPlanes.mmCorrespondences[sensor_id].count(sensor_id+1) && matchedPlanes.mmCorrespondences[sensor_id][sensor_id+1].getRowCount() >= 3 );
//        cout << "sensor_id " << sensor_id << endl;
//        cout << "Rt_estimated \n" << Rt_estimated[sensor_id] << endl;
//        cout << "Rt+1 " << sensor_id << endl;
//        cout << it_pair->second << endl;
        for(std::map<unsigned, mrpt::math::CMatrixDouble>::iterator it_pair=matchedPlanes.mmCorrespondences[sensor_id].begin();
            it_pair != matchedPlanes.mmCorrespondences[sensor_id].end(); it_pair++)
//          if( it_pair->first - sensor_id == 1 || it_pair->first - sensor_id == 7)
        {
          int id_corresp1 = 3*(sensor_id-1);
          int id_corresp2 = 3*(it_pair->first - 1);
//        cout << "id_corresp1 " << id_corresp1 << "id_corresp2 " << id_corresp2 << endl;

          for(unsigned i=0; i < it_pair->second.getRowCount(); i++)
          {
//          float weight = (inliers / it_pair->second(i,3)) / it_pair->second.getRowCount()
            Eigen::Vector3f n_obs_i; n_obs_i << it_pair->second(i,0), it_pair->second(i,1), it_pair->second(i,2);
            Eigen::Vector3f n_obs_ii; n_obs_ii << it_pair->second(i,4), it_pair->second(i,5), it_pair->second(i,6);
            Eigen::Vector3f n_i = Rt_estimated[sensor_id].block(0,0,3,3) * n_obs_i;
            Eigen::Vector3f n_ii = Rt_estimated[it_pair->first].block(0,0,3,3) * n_obs_ii;
            trans_error = (it_pair->second(i,3) - it_pair->second(i,7));
            accum_error2 += trans_error * trans_error;
            numControlPlanes++;
//          cout << "Rt_estimated \n" << Rt_estimated[sensor_id] << " n_i " << n_i.transpose() << " n_ii " << n_ii.transpose() << endl;
            if(weightedLS == 1 && it_pair->second.getColCount() == 18)
            {
              // The weight takes into account the number of inliers of the patch, the distance of the patch's center to the image center and the distance of the plane to the sensor
              float weight = (it_pair->second(i,8) / (it_pair->second(i,3) * it_pair->second(i,9)));// / it_pair->second.getRowCount();

              if(sensor_id != 0) // The pose of the first camera is fixed
              {
                hessian.block(id_corresp1, id_corresp1, 3, 3) += weight * (n_i * n_i.transpose() );
                gradient.block(id_corresp1,0,3,1) += weight * (-n_i * trans_error);

                // Cross term
                hessian.block(id_corresp1, id_corresp2, 3, 3) += weight * (-n_i * n_ii.transpose() );
              }

              hessian.block(id_corresp2, id_corresp2, 3, 3) += weight * (n_ii * n_ii.transpose() );
              gradient.block(id_corresp2,0,3,1) += weight * (n_ii * trans_error);
            }
            else
            {
              if(sensor_id != 0) // The pose of the first camera is fixed
              {
                hessian.block(id_corresp1, id_corresp1, 3, 3) += n_i * n_i.transpose();
                gradient.block(id_corresp1,0,3,1) += -n_i * trans_error;

                // Cross term
                hessian.block(id_corresp1, id_corresp2, 3, 3) += -n_i * n_ii.transpose();
              }

              hessian.block(id_corresp2, id_corresp2, 3, 3) += n_ii * n_ii.transpose();
              gradient.block(id_corresp2,0,3,1) += n_ii * trans_error;
            }
          }

          // Fill the lower left triangle with the corresponding cross terms
          if(sensor_id != 0)
            hessian.block(id_corresp2, id_corresp1, 3, 3) = hessian.block(id_corresp1, id_corresp2, 3, 3).transpose();
        }
      }

//      cout << "hessian\n" << hessian << endl;
//      cout << "calcConditioning " << calcConditioning() << endl;

      if( calcConditioning() < threshold_conditioning )
      {
        // Solve translation
        update_vector = -hessian.inverse() * gradient;
      cout << "update_vector translations " << update_vector.transpose() << endl;

        // Get center of the multicamera system
        Eigen::Vector3f centerDevice = Eigen::Vector3f::Zero();
        for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
          centerDevice += update_vector.block(3*sensor_id-3,0,3,1);
        centerDevice = centerDevice / 8;

        // Update translation of the poses
        Rt_estimatedTemp[0].block(0,3,3,1) = -centerDevice;
        for(int sensor_id = 1; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
          Rt_estimatedTemp[sensor_id].block(0,3,3,1) = update_vector.block(3*sensor_id-3,0,3,1) - centerDevice;

//      // Assign new translations
//      float new_accum_error2 = calcCorrespTransError(Rt_estimatedTemp);
//      if(new_accum_error2 < accum_error2)
        for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
          Rt_estimated[sensor_id].block(0,3,3,1) = Rt_estimatedTemp[sensor_id].block(0,3,3,1);
//          Rt_estimated[sensor_id].block(0,3,3,1) = Eigen::Matrix<float,3,1>::Zero();

      std::cout << "ErrorCalibTranslation " << accum_error2/numControlPlanes << std::endl;
//      cout << "\tCalibration finished\n";
      }
      else
        cout << "\tTranslation system is bad conditioned " << conditioning << " threshold " << threshold_conditioning << "\n";
    }

    /*! Get the Rt of each sensor in the multisensor RGBD360 setup */
    void Calibrate()
    {
      CalibrateRotation();
      CalibrateTranslation();
    }

  private:

    /*! Calculate system's conditioning */
    float calcConditioning()
    {
      Eigen::JacobiSVD<Eigen::Matrix<float,21,21> > svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
      conditioning = svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();

      return conditioning;
    }
};
#endif
