/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR
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

#include <mrpt/base.h>
#include <mrpt/gui.h>
#include <mrpt/opengl.h>
#include <mrpt/slam.h>
#include <mrpt/utils.h>
#include <mrpt/obs.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Frame360.h>
#include <Frame360_Visualizer.h>
#include <RegisterRGBD360.h>
#include <Calibrator.h>

#include <RGBDGrabber_OpenNI2.h>
#include <SerializeFrameRGBD.h> // For time-stamp conversion

#include <pcl/console/parse.h>

//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <signal.h>

#define USE_DEBUG_SEQUENCE 0
#define NUM_SENSORS 2
//const int NUM_SENSORS = 2;

using namespace std;
using namespace Eigen;
using namespace mrpt;
using namespace mrpt::gui;
using namespace mrpt::opengl;
using namespace mrpt::math;
using namespace mrpt::utils;
using namespace mrpt::slam;
using namespace mrpt::poses;

#define SIGMA2 0.01
#define RECORD_VIDEO 0
int numScreenshot = 0;

RGBDGrabber_OpenNI2 *grabber[NUM_SENSORS]; // This object is declared as global to be able to use it after capturing Ctrl-C interruptions

const int min_inliers = 1000;

/*! Catch interruptions like Ctrl-C */
void INThandler(int sig)
{
  char c;

  signal(sig, SIG_IGN);
  printf("\n  Do you really want to quit? [y/n] ");
  c = getchar();
  if (c == 'y' || c == 'Y')
  {
    for(unsigned sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
    {
      delete grabber[sensor_id]; // Turn off each Asus XPL sensor before exiting the program
    }
    exit(0);
  }
}

/*---------------------------------------------------------------
		Aux. functions needed by ransac_detect_2D_lines
 ---------------------------------------------------------------*/
void  ransac2Dline_fit_(
  const CMatrixTemplateNumeric<float> &allData,
  const vector_size_t &useIndices,
  vector< CMatrixTemplateNumeric<float> > &fitModels )
{
  ASSERT_(useIndices.size()==2);

  TPoint2D p1( allData(0,useIndices[0]),allData(1,useIndices[0]) );
  TPoint2D p2( allData(0,useIndices[1]),allData(1,useIndices[1]) );

  try
  {
    TLine2D  line(p1,p2);
    fitModels.resize(1);
    CMatrixTemplateNumeric<float> &M = fitModels[0];

    M.setSize(1,3);
    for (size_t i=0;i<3;i++)
      M(0,i)=line.coefs[i];
//  cout << "Line model " << allData(0,useIndices[0]) << " " << allData(1,useIndices[0]) << " " << allData(0,useIndices[1]) << " " << allData(1,useIndices[1]) << " M " << M << endl;
  }
  catch(exception &)
  {
    fitModels.clear();
    return;
  }
}


void ransac2Dline_distance_(
  const CMatrixTemplateNumeric<float> &allData,
  const vector< CMatrixTemplateNumeric<float> > & testModels,
  const float distanceThreshold,
  unsigned int & out_bestModelIndex,
  vector_size_t & out_inlierIndices )
{
  out_inlierIndices.clear();
  out_bestModelIndex = 0;

  if (testModels.empty()) return; // No model, no inliers.

  ASSERTMSG_( testModels.size()==1, format("Expected testModels.size()=1, but it's = %u",static_cast<unsigned int>(testModels.size()) ) )
  const CMatrixTemplateNumeric<float> &M = testModels[0];

  ASSERT_( size(M,1)==1 && size(M,2)==3 )

  TLine2D  line;
  line.coefs[0] = M(0,0);
  line.coefs[1] = M(0,1);
  line.coefs[2] = M(0,2);

  const size_t N = size(allData,2);
  out_inlierIndices.reserve(100);
  for (size_t i=0;i<N;i++)
  {
    const double d = line.distance( TPoint2D( allData.get_unsafe(0,i),allData.get_unsafe(1,i) ) );
//  cout << "distance " << d << " " << allData.get_unsafe(0,i) << " " << allData.get_unsafe(1,i) << endl;
    if (d<distanceThreshold)
      out_inlierIndices.push_back(i);
  }
}

/** Return "true" if the selected points are a degenerate (invalid) case.
  */
bool ransac2Dline_degenerate_(
  const CMatrixTemplateNumeric<float> &allData,
  const mrpt::vector_size_t &useIndices )
{
//  ASSERT_( useIndices.size()==2 )
//
//  const Eigen::Vector2d origin = Eigen::Vector2d(allData(0,useIndices[0]), allData(1,useIndices[0]));
//  const Eigen::Vector2d end = Eigen::Vector2d(allData(0,useIndices[1]), allData(1,useIndices[1]));
//
//  if( (end-origin).norm() < 0.01 )
//    return true;
  return false;
}

/*---------------------------------------------------------------
				ransac_detect_3D_lines
 ---------------------------------------------------------------*/
void ransac_detect_3D_lines(
	const pcl::PointCloud<PointT>::Ptr &scan,
	Eigen::Matrix<float,Eigen::Dynamic,6> &lines,
//	CMatrixTemplateNumeric<float,Eigen::Dynamic,6> &lines,
	const double           threshold,
	const size_t           min_inliers_for_valid_line
	)
{
	ASSERT_(scan->size() )
//cout << "ransac_detect_2D_lines \n";

	if(scan->empty())
		return;

	// The running lists of remaining points after each plane, as a matrix:
	CMatrixTemplateNumeric<float> remainingPoints( 2, scan->size() );
	for(unsigned i=0; i < scan->size(); i++)
	{
    remainingPoints(0,i) = scan->points[i].y;
    remainingPoints(1,i) = scan->points[i].z;
	}

//cout << "Size remaining pts " << size(remainingPoints,1) << " " << size(remainingPoints,2) << endl;

	// ---------------------------------------------
	// For each line:
	// ---------------------------------------------
	std::vector<std::pair<size_t,TLine2D> > out_detected_lines;
//	while (size(remainingPoints,2)>=2)
	{
		mrpt::vector_size_t				this_best_inliers;
		CMatrixTemplateNumeric<float> this_best_model;

		math::RANSAC_Template<float>::execute(
			remainingPoints,
			ransac2Dline_fit_,
			ransac2Dline_distance_,
			ransac2Dline_degenerate_,
			threshold,
			2,  // Minimum set of points
			this_best_inliers,
			this_best_model,
			false, // Verbose
			0.99  // Prob. of good result
			);
//cout << "Size this_best_inliers " << this_best_inliers.size() << endl;

		// Is this plane good enough?
		if (this_best_inliers.size()>=min_inliers_for_valid_line)
		{
			// Add this plane to the output list:
			out_detected_lines.push_back(
				std::make_pair<size_t,TLine2D>(
					this_best_inliers.size(),
					TLine2D(this_best_model(0,0), this_best_model(0,1),this_best_model(0,2) )
					) );

			out_detected_lines.rbegin()->second.unitarize();

			int prev_size = size(lines,1);
//    cout << "prevSize lines " << prev_size << endl;
			lines.setSize(prev_size+1,6);
			float mod_dir = sqrt(1+pow(this_best_model(0,0)/this_best_model(0,1),2));
			lines(prev_size,0) = 0; // The reference system for the laser is aligned in the horizontal axis
			lines(prev_size,1) = 1/mod_dir;
			lines(prev_size,2) = -(this_best_model(0,0)/this_best_model(0,1))/mod_dir;
			lines(prev_size,3) = 0;
			lines(prev_size,4) = scan->points[this_best_inliers[0]].y;
//			lines(prev_size,4) = scan->points[this_best_inliers[0]].x;
			lines(prev_size,5) = scan->points[this_best_inliers[0]].z;
			// Discard the selected points so they are not used again for finding subsequent planes:
			remainingPoints.removeColumns(this_best_inliers);
		}
//		else
//		{
//			break; // Do not search for more planes.
//		}
	}
}

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
    Eigen::Matrix3f CalibrateRotation(int weightedLS = 0)
    {
    cout << "CalibrateRotation Plane-Line...\n";
      Eigen::Matrix<float,3,3> hessian;
      Eigen::Matrix<float,3,1> gradient;
      Eigen::Matrix<float,3,1> update_vector;
      Eigen::Matrix<float,1,3> jacobian_rot_ii; // Jacobians of the rotation
      float accum_error2;
      float av_angle_error;
      unsigned numControlPlanes;

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
            jacobian_rot_ii = -(n_obs_i.transpose() * skew(Rt_estimated.block(0,0,3,3) * l_obs_ii));
            float rot_error = n_obs_i.transpose() * Rt_estimated.block(0,0,3,3) * l_obs_ii;
            accum_error2 += pow(rot_error,2);
            av_angle_error += PI/2 - fabs(acos(rot_error));
            numControlPlanes++;
//          cout << "rotation error_i " << rot_error.transpose() << endl;
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

        cout << "New rotation error " << new_accum_error2 << " previous " << accum_error2 << endl;
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

    Eigen::Vector3f CalibrateTranslation(int weightedLS = 0)
    {
    cout << "CalibrateTranslation Laser-Kinect\n";
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
        float trans_error = (d_obs_i - n_obs_i.dot(Rt_estimated.block(0,0,3,3) * c_obs_ii));

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
      std::cout << "FIM translation " << svd.singularValues().transpose() << endl;

//      cout << "translationHessian \n" << translationHessian << "\n HessianInv \n" << translationHessian.inverse() << endl;
//      calcFisherInfMat();

      translation = translationHessian.inverse() * translationGradient;
    std::cout << "translation " << translation.transpose() << std::endl;

      return translation;
    }

    void CalibratePair()
    {
//      calibrated_Rt = Eigen::Matrix4f::Identity();
      Rt_estimated.block(0,0,3,3) = CalibrateRotation();
      Rt_estimated.block(0,3,3,1) = CalibrateTranslation();
//      std::cout << "Rt_estimated\n" << Rt_estimated << std::endl;

      std::cout << "Errors av rot " << calcCorrespRotError(Rt_estimated) << " av trans " << calcCorrespTransError(Rt_estimated) << std::endl;
    }
};


//// Obtain a line from two 3D points
//CMatrixDouble ransac3Dline_fit( const CMatrixDouble &pts )
//{
//  assert(size(pts,1) == 3 && size(pts,2) == 2);
//
////  TPoint3D p1(pts(0,0),pts(1,0),pts(2,0));
////  TPoint3D p2(pts(0,1),pts(1,1),pts(2,1));
////  TLine3D line(p1,p2);
//
//  Eigen::Vector3d dir;
//  double dist = sqrt((pts(0,0)-pts(0,1))*(pts(0,0)-pts(0,1)) + (pts(1,0)-pts(1,1))*(pts(1,0)-pts(1,1)) + (pts(2,0)-pts(2,1))*(pts(2,0)-pts(2,1)));
//  dir[0] = (pts(0,0)-pts(0,1)) / dist;
//  dir[1] = (pts(1,0)-pts(1,1)) / dist;
//  dir[2] = (pts(2,0)-pts(2,1)) / dist;
//
//  CMatrixDouble line(1,6);
//  line(0,0) = pts(0,0);
//  line(0,1) = pts(1,0);
//  line(0,2) = pts(2,0);
//  line(0,3) = dir[0];
//  line(0,4) = dir[1];
//  line(0,5) = dir[2];
//
//  return line;
//}
//
//// Ransac functions to detect outliers in the line matching
//void ransacLineAlignment_fit(
//        const CMatrixDouble &lineCorresp,
//        const vector_size_t  &useIndices,
//        vector< CMatrixDouble > &fitModels )
////        vector< Eigen::Matrix4f > &fitModels )
//{
//  ASSERT_(useIndices.size()==3);
//
//  try
//  {
//    CMatrixDouble corresp(3,2);
//
////  cout << "Size lineCorresp: " << endl;
////  cout << "useIndices " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << endl;
//    for(unsigned i=0; i<3; i++)
//      corresp.col(i) = lineCorresp.col(useIndices[i]);
//
//    fitModels.resize(1);
////    Eigen::Matrix4f &M = fitModels[0];
//    CMatrixDouble &M = fitModels[0];
//    M = ransac3Dline_fit(corresp);
////  cout << "Ransac M\n" << M << endl;
//  }
//  catch(exception &)
//  {
//    fitModels.clear();
//    return;
//  }
//}
//
//void ransac3Dline_distance(
//        const CMatrixDouble &lineCorresp,
//        const vector< CMatrixDouble > & testModels,
//        const double distanceThreshold,
//        unsigned int & out_bestModelIndex,
//        vector_size_t & out_inlierIndices )
//{
//  ASSERT_( testModels.size()==1 )
//  out_bestModelIndex = 0;
//  const CMatrixDouble &M = testModels[0];
//
//  Eigen::Vector3d origin; origin << M(0,0), M(0,1), M(0,2);
//  Eigen::Vector3d orientation; orientation << M(0,3), M(0,4), M(0,5);
//
//	ASSERT_( size(M,1)==1 && size(M,2)==6 )
//
//  const size_t N = size(lineCorresp,2);
//  out_inlierIndices.clear();
//  out_inlierIndices.reserve(100);
//  for (size_t i=0;i<N;i++)
//  {
//    const Eigen::Vector3d pt = Eigen::Vector3d(lineCorresp(0,i), lineCorresp(1,i), lineCorresp(2,i));
//    const float error = (pt-origin).dot(pt-origin) - pow((pt-origin).dot(orientation),2);
//
//    if (error < distanceThreshold){//cout << "new inlier\n";
//      out_inlierIndices.push_back(i);}
//  }
//}
//
///** Return "true" if the selected points are a degenerate (invalid) case.
//  */
//bool ransac3Dline_degenerate(
//        const CMatrixDouble &lineCorresp,
//        const mrpt::vector_size_t &useIndices )
//{
//  ASSERT_( useIndices.size()==2 )
//
//  const Eigen::Vector3d origin = Eigen::Vector3d(lineCorresp(0,useIndices[0]), lineCorresp(1,useIndices[0]), lineCorresp(2,useIndices[0]));
//  const Eigen::Vector3d end = Eigen::Vector3d(lineCorresp(0,useIndices[1]), lineCorresp(1,useIndices[1]), lineCorresp(2,useIndices[1]));
//
//  if( (end-origin).norm() < 0.01 )
//    return true;
//
//  return false;
//}
//
///*---------------------------------------------------------------
//				ransac_detect_3D_lines
// ---------------------------------------------------------------*/
//template <typename float>
//void ransac_detect_3D_lines(
//	const Eigen::Matrix<float,3,Eigen::Dynamic> &pts,
//	std::vector<std::pair<size_t,TLine3D> > &out_detected_lines,
//	const double           threshold,
//	const size_t           min_inliers_for_valid_line
//	)
//{
//	out_detected_lines.clear();
//
//	if(pts.getColCount() < 2)
//		return;
//
//	// The running lists of remaining points after each line, as a matrix:
//	CMatrixTemplateNumeric<float> remainingPoints = pts;
//
//	// ---------------------------------------------
//	// For each line:
//	// ---------------------------------------------
//	while (size(remainingPoints,2)>=2)
//	{
//		mrpt::vector_size_t				this_best_inliers;
//		CMatrixTemplateNumeric<float> this_best_model;
//
//		math::RANSAC_Template<float>::execute(
//			remainingPoints,
//			ransac3Dline_fit,
//			ransac3Dline_distance,
//			ransac3Dline_degenerate,
//			threshold,
//			2,  // Minimum set of points
//			this_best_inliers,
//			this_best_model,
//			false, // Verbose
//			0.999  // Prob. of good result
//			);
//
//		// Is this line good enough?
//		if (this_best_inliers.size()>=min_inliers_for_valid_line)
//		{
//			// Add this line to the output list:
//			out_detected_lines.push_back(
//				std::make_pair<size_t,TLine3D>(
//					this_best_inliers.size(),
//					TLine3D(this_best_model(0,0), this_best_model(0,1),this_best_model(0,2) )
//					) );
//
//			out_detected_lines.rbegin()->second.unitarize();
//
//			// Discard the selected points so they are not used again for finding subsequent planes:
//			remainingPoints.removeColumns(this_best_inliers);
//		}
//		else
//		{
//			break; // Do not search for more planes.
//		}
//	}
//
//}

// Class to parse rawlogs
class DatasetParser{
private:
	mrpt::slam::CRawlog m_dataset;
	ifstream m_fgt;
	bool m_last_groundtruth_ok;
	bool m_groundtruth_ok;
	//bool m_first_frame = true;

	double m_last_groundtruth;
	CPose3D m_pose;
	CColouredPointsMap m_pntsMap;
	CImage  m_depthimg, m_colorimg;
	pcl::PointCloud<pcl::PointXYZRGB> m_pclCloud;


public:
	int m_count;
	DatasetParser(string path_rawlog){//,string path_groundtruth){

//		m_last_groundtruth_ok = true;
//		m_groundtruth_ok = true;
		m_count = 0;
//		m_last_groundtruth = 0;

		//TODO error checking
		m_dataset.loadFromRawLogFile(path_rawlog);
//		m_fgt.open(path_groundtruth.c_str());

//		path_rawlog.replace(path_rawlog.find(".rawlog"),7,"_Images/");
//		// Set external images directory:
//		CImage::IMAGES_PATH_BASE = path_rawlog;

//		char aux[100];
//		m_fgt.getline(aux, 100);
//		m_fgt.getline(aux, 100);
//		m_fgt.getline(aux, 100);
//		m_fgt >> m_last_groundtruth;
//		m_last_groundtruth_ok = true;
	}

	//LoadNextFrame returns true if the next frame has correct ground truth.
	bool LoadNextFrame(void){
		CObservationPtr obs;
//#include <mrpt/slam/CSensoryFrame.h>
//		mrpt::slam::CSensoryFrame obs;

		do
		{
			if (m_dataset.size() <= m_count)
				return false;
			obs = m_dataset.getAsObservation(m_count);

//  cout << "Dataset size " << m_dataset.size() << endl;
//  if (IS_CLASS(obs, CObservation3DRangeScan))
//    cout << "LoadNextFrame CObservation3DRangeScan" << endl;
//  //if (IS_CLASS(obs, mrpt::slam::CSensoryFrame))
//  else
//    cout << "LoadNextFrame CSensoryFrame" << endl;

			m_count++;
		}
		while (!IS_CLASS(obs, CObservation3DRangeScan));

		CObservation3DRangeScanPtr obsRGBD = CObservation3DRangeScanPtr(obs);
		obsRGBD->load();


		//Calculate the 3D cloud from RGBD and include in the observation.
		//obsRGBD->project3DPointsFromDepthImage();
		obsRGBD->project3DPointsFromDepthImageInto(*obsRGBD,false,NULL,true);
		//obsRGBD->project3DPointsFromDepthImageInto(m_pclCloud,false,NULL,true);

		cout << "obsRGBD->points3D_x.size(): " << obsRGBD->points3D_x.size() << endl;


//		//Copy to the CColouredPointsMap
//		m_pntsMap.clear();
//		m_pntsMap.colorScheme.scheme = CColouredPointsMap::cmFromIntensityImage;
//		m_pntsMap.insertionOptions.minDistBetweenLaserPoints = 0; // don't drop any point
//		m_pntsMap.insertionOptions.disableDeletion = true;
//		m_pntsMap.insertionOptions.fuseWithExisting = false;
//		m_pntsMap.insertionOptions.insertInvalidPoints = true;
//		m_pntsMap.insertObservation(obs.pointer());
//
//		cout << "m_pntsMap.size():" << m_pntsMap.size() << endl;

		//Copy rgb and depth images.
		m_depthimg.setFromMatrix(obsRGBD->rangeImage);
		m_colorimg = obsRGBD->intensityImage;

		cout << "depth img width: " << m_depthimg.getWidth()<<endl;
		cout << "depth img height: " << m_depthimg.getHeight()<<endl;


		double timestamp_gt;
		double timestamp_obs = timestampTotime_t(obsRGBD->timestamp);

		obsRGBD->unload();

//		return m_groundtruth_ok;
		return true;
	}


	CPose3D getGroundTruthPose(){
		return m_pose;
	}

	CColouredPointsMap getPointCloud(){
		return m_pntsMap;
	}

	pcl::PointCloud<pcl::PointXYZRGB> getpclCloud(){
		return m_pclCloud;
	}

	CImage getRGB(){
		return m_colorimg;
	}

	CImage getD(){
		return m_depthimg;
	}

};


//int main ()
//{
//	CTicTac  tictac;
//
//  CObservation3DRangeScanPtr obsKinect;
////  CObservation3DRangeScanPtr obsToF;
//  CObservation2DRangeScanPtr obsLaser;
//  const size_t fieldOfView = 241; // Limit the field of view of the laser to 60 deg
//  const size_t offset60deg = (1081-241)/2; // Limit the field of view of the laser to 60 deg
//	bool bPrevLaserScan = false;
//  mrpt::math::CMatrixDouble correspPlaneLine(0,10);
//
//  CFileGZInputStream   rawlogFile("/media/Data/Datasets360/Laser+Kinect/dataset_2014-02-12_17h06m40s.rawlog");   // "file.rawlog"
//  CActionCollectionPtr action;
//  CSensoryFramePtr     observations;
//  CObservationPtr         observation;
//  size_t               rawlogEntry=0;
//  bool        end = false;
//
//  while ( CRawlog::getActionObservationPairOrObservation(
//         rawlogFile,      // Input file
//         action,            // Possible out var: action of a pair action/obs
//         observations,  // Possible out var: obs's of a pair action/obs
//         observation,    // Possible out var: a single obs.
//         rawlogEntry    // Just an I/O counter
//         ) )
//  {
//    // Process observations
//    if (observation)
//    {
////      cout << "Read observation\n";
//      if(IS_CLASS(observation, CObservation2DRangeScan))
//      {
//        assert(observation->sensorLabel == "HOKUYO_UTM");
//
//        obsLaser = CObservation2DRangeScanPtr(observation);
//        bPrevLaserScan = true;
////        cout << "Laser timestamp " << obsLaser->timestamp << endl;
////        cout << "Scan width " << obsLaser->scan.size() << endl;
//      }
//      else if(IS_CLASS(observation, CObservation3DRangeScan))
//      {
//        assert(observation->sensorLabel == "KINECT");
//
//        obsKinect = CObservation3DRangeScanPtr(observation);
////        cout << "Kinect timestamp " << obsKinect->timestamp << endl;
//
//        if(bPrevLaserScan && (obsKinect->timestamp - obsLaser->timestamp) < 250000)
//        {
//          mrpt::slam::CSimplePointsMap m_cache_points;
//          m_cache_points.clear();
//          m_cache_points.insertionOptions.minDistBetweenLaserPoints = 0;
//          m_cache_points.insertionOptions.isPlanarMap=false;
//          m_cache_points.insertObservation( &(*obsLaser) );
//          size_t n;
//          const float	*x,*y,*z;
//          m_cache_points.getPointsBuffer(n,x,y,z);
////          for(size_t i=0; i < obsLaser->scan.size(); i++)
////            cout << i << " scan " << obsLaser->scan[i] << " x " << x[i] << " y " << y[i] << " z " << z[i] << endl;
//
////          Eigen::Matrix<float,Eigen::Dynamic,1> x_(241);
//          vector_float x_(1081), y_(1081);
//          for(size_t i=0; i < 1081; i++)
////          vector_float x_(fieldOfView), y_(fieldOfView);
////          for(size_t i=offset60deg; i < fieldOfView; i++)
//          {
//            x_[i] = x[i+offset60deg];
//            y_[i] = y[i+offset60deg];
//          }
//
//          // Run RANSAC
//          // ------------------------------------
//          vector<pair<size_t,TLine2D > > detectedLines;
//          const double DIST_THRESHOLD = 0.1;
//          ransac_detect_2D_lines(x_, y_, detectedLines, DIST_THRESHOLD, 20);
//        cout << detectedLines.size() << " detected lines " << endl;
//
//          //Copy to the CColouredPointsMap
//          CColouredPointsMap m_pntsMap;
//    //      CPointsMap m_pntsMap;
//
//    //  		m_pntsMap.clear();
//          m_pntsMap.colorScheme.scheme = CColouredPointsMap::cmFromIntensityImage;
//          m_pntsMap.insertionOptions.minDistBetweenLaserPoints = 0; // don't drop any point
//          m_pntsMap.insertionOptions.disableDeletion = true;
//          m_pntsMap.insertionOptions.fuseWithExisting = false;
//          m_pntsMap.insertionOptions.insertInvalidPoints = true;
//          m_pntsMap.insertObservation(obsKinect.pointer());
//          pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudKinect(new pcl::PointCloud<pcl::PointXYZRGBA>);
//          m_pntsMap.getPCLPointCloud(*cloudKinect);
//
//          cout << "Cloud-Kinect pts " << cloudKinect->points.size() << endl;
//
//          CloudRGBD_Ext cloud_;
//          DownsampleRGBD downsampler(2);
//          pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampledCloud = downsampler.downsamplePointCloud(cloudKinect);
//          cloud_.setPointCloud(downsampledCloud);
//          mrpt::pbmap::PbMap planes_;
////          segmentPlanesInFrame(cloud_, planes_);
//
////          //Extract a plane with RANSAC
////          Eigen::VectorXf modelcoeff_Plane(4);
////          vector<int> inliers;
////          pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloudKinect));
////          pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_p);
////          ransac.setDistanceThreshold (.03);
////          tictac.Tic();
////          ransac.computeModel();
////          ransac.getModelCoefficients(modelcoeff_Plane);
////          if(modelcoeff_Plane[3] < 0) modelcoeff_Plane *= -1;
////    //      modelcoeff_Plane *= (modelcoeff_Plane[3]/fabs(modelcoeff_Plane[3]));
////        cout << "RANSAC (pcl) computation time: " << tictac.Tac()*1000.0 << " ms " << modelcoeff_Plane.transpose() << endl;
////          ransac.getInliers(inliers);
//
////          // Stablish the correspondences
////          for(unsigned i=0; i < planes_.vPlanes.size(); i++)
////          {
////            for(unsigned j=0; j < detectedLines.size(); j++)
////            {
////              if( planes_i.vPlanes[i].inliers.size() > min_inliers && planes_j.vPlanes[j].inliers.size() > min_inliers &&
////                  planes_i.vPlanes[i].elongation < 5 && planes_j.vPlanes[j].elongation < 5 &&
////                  planes_i.vPlanes[i].v3normal .dot (planes_j.vPlanes[j].v3normal) > 0.99 &&
////                  fabs(planes_i.vPlanes[i].d - planes_j.vPlanes[j].d) < 0.1 //&&
//////                    planes_i.vPlanes[i].hasSimilarDominantColor(planes_j.vPlanes[j],0.06) &&
//////                    planes_i.vPlanes[planes_counter_i+i].isPlaneNearby(planes_j.vPlanes[planes_counter_j+j], 0.5)
////                )
////              {
////                unsigned prevSize = correspPlaneLine.getRowCount();
////                correspPlaneLine.setSize(prevSize+1, correspPlaneLine.getColCount());
////                correspPlaneLine(prevSize, 0) = modelcoeff_Plane[0];
////                correspPlaneLine(prevSize, 1) = modelcoeff_Plane[1];
////                correspPlaneLine(prevSize, 2) = modelcoeff_Plane[2];
////                correspPlaneLine(prevSize, 3) = modelcoeff_Plane[3];
////                correspPlaneLine(prevSize, 4) = modelcoeff_Line[0];
////                correspPlaneLine(prevSize, 5) = modelcoeff_Line[1];
////                correspPlaneLine(prevSize, 6) = modelcoeff_Line[2];
////                correspPlaneLine(prevSize, 7) = modelcoeff_Line[3];
////                correspPlaneLine(prevSize, 8) = modelcoeff_Line[4];
////                correspPlaneLine(prevSize, 9) = modelcoeff_Line[5];
////              }
////            }
////          }
//        }
//      }
//    }
//  }
//
//  cout << "\tSave correspPlaneLine\n";
//  correspPlaneLine.saveToTextFile( mrpt::format("%s/correspPlaneLine.txt", PROJECT_SOURCE_PATH) );
//
//	return (0);
//}



//// Obtain the rigid transformation from 3 matched planes
//CMatrixDouble getAlignment( const CMatrixDouble &matched_planes )
//{
//  assert(size(matched_planes,1) == 8 && size(matched_planes,2) == 3);
//
//  //Calculate rotation
//  Matrix3f normalCovariances = Matrix3f::Zero();
//  normalCovariances(0,0) = 1;
//  for(unsigned i=0; i<3; i++)
//  {
//    Vector3f n_i = Vector3f(matched_planes(0,i), matched_planes(1,i), matched_planes(2,i));
//    Vector3f n_ii = Vector3f(matched_planes(4,i), matched_planes(5,i), matched_planes(6,i));
//    normalCovariances += n_ii * n_i.transpose();
////    normalCovariances += matched_planes.block(i,0,1,3) * matched_planes.block(i,4,1,3).transpose();
//  }
//
//  JacobiSVD<MatrixXf> svd(normalCovariances, ComputeThinU | ComputeThinV);
//  Matrix3f Rotation = svd.matrixV() * svd.matrixU().transpose();
//
////  float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
////  if(conditioning > 100)
////  {
////    cout << " ConsistencyTest::initPose -> Bad conditioning: " << conditioning << " -> Returning the identity\n";
////    return Eigen::Matrix4f::Identity();
////  }
//
//  double det = Rotation.determinant();
//  if(det != 1)
//  {
//    Eigen::Matrix3f aux;
//    aux << 1, 0, 0, 0, 1, 0, 0, 0, det;
//    Rotation = svd.matrixV() * aux * svd.matrixU().transpose();
//  }
//
//
//  // Calculate translation
//  Vector3f translation;
//  Matrix3f hessian = Matrix3f::Zero();
//  Vector3f gradient = Vector3f::Zero();
//  hessian(0,0) = 1;
//  for(unsigned i=0; i<3; i++)
//  {
//    float trans_error = (matched_planes(3,i) - matched_planes(7,i)); //+n*t
////    hessian += matched_planes.block(i,0,1,3) * matched_planes.block(i,0,1,3).transpose();
////    gradient += matched_planes.block(i,0,1,3) * trans_error;
//    Vector3f n_i = Vector3f(matched_planes(0,i), matched_planes(1,i), matched_planes(2,i));
//    hessian += n_i * n_i.transpose();
//    gradient += n_i * trans_error;
//  }
//  translation = -hessian.inverse() * gradient;
////cout << "Previous average translation error " << sumError / matched_planes.size() << endl;
//
////  // Form SE3 transformation matrix. This matrix maps the model into the current data reference frame
////  Eigen::Matrix4f rigidTransf;
////  rigidTransf.block(0,0,3,3) = Rotation;
////  rigidTransf.block(0,3,3,1) = translation;
////  rigidTransf.row(3) << 0,0,0,1;
//
//  CMatrixDouble rigidTransf(4,4);
//  rigidTransf(0,0) = Rotation(0,0);
//  rigidTransf(0,1) = Rotation(0,1);
//  rigidTransf(0,2) = Rotation(0,2);
//  rigidTransf(1,0) = Rotation(1,0);
//  rigidTransf(1,1) = Rotation(1,1);
//  rigidTransf(1,2) = Rotation(1,2);
//  rigidTransf(2,0) = Rotation(2,0);
//  rigidTransf(2,1) = Rotation(2,1);
//  rigidTransf(2,2) = Rotation(2,2);
//  rigidTransf(0,3) = translation(0);
//  rigidTransf(1,3) = translation(1);
//  rigidTransf(2,3) = translation(2);
//  rigidTransf(3,0) = 0;
//  rigidTransf(3,1) = 0;
//  rigidTransf(3,2) = 0;
//  rigidTransf(3,3) = 1;
//
//  return rigidTransf;
//}
//
//// Ransac functions to detect outliers in the plane matching
//void ransacPlaneAlignment_fit(
//        const CMatrixDouble &planeCorresp,
//        const vector_size_t  &useIndices,
//        vector< CMatrixDouble > &fitModels )
////        vector< Eigen::Matrix4f > &fitModels )
//{
//  ASSERT_(useIndices.size()==3);
//
//  try
//  {
//    CMatrixDouble corresp(8,3);
//
////  cout << "Size planeCorresp: " << endl;
////  cout << "useIndices " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << endl;
//    for(unsigned i=0; i<3; i++)
//      corresp.col(i) = planeCorresp.col(useIndices[i]);
//
//    fitModels.resize(1);
////    Eigen::Matrix4f &M = fitModels[0];
//    CMatrixDouble &M = fitModels[0];
//    M = getAlignment(corresp);
////  cout << "Ransac M\n" << M << endl;
//  }
//  catch(exception &)
//  {
//    fitModels.clear();
//    return;
//  }
//}
//
//void ransac3Dplane_distance(
//        const CMatrixDouble &planeCorresp,
//        const vector< CMatrixDouble > & testModels,
//        const double distanceThreshold,
//        unsigned int & out_bestModelIndex,
//        vector_size_t & out_inlierIndices )
//{
//  ASSERT_( testModels.size()==1 )
//  out_bestModelIndex = 0;
//  const CMatrixDouble &M = testModels[0];
//
//  Eigen::Matrix3f Rotation; Rotation << M(0,0), M(0,1), M(0,2), M(1,0), M(1,1), M(1,2), M(2,0), M(2,1), M(2,2);
//  Eigen::Vector3f translation; translation << M(0,3), M(1,3), M(2,3);
//
//	ASSERT_( size(M,1)==4 && size(M,2)==4 )
//
//  const double angleThreshold = distanceThreshold / 3;
//
//  const size_t N = size(planeCorresp,2);
//  out_inlierIndices.clear();
//  out_inlierIndices.reserve(100);
//  for (size_t i=0;i<N;i++)
//  {
//    const Eigen::Vector3f n_i = Eigen::Vector3f(planeCorresp(0,i), planeCorresp(1,i), planeCorresp(2,i));
//    const Eigen::Vector3f n_ii = Rotation * Eigen::Vector3f(planeCorresp(4,i), planeCorresp(5,i), planeCorresp(6,i));
//    const float d_error = fabs((planeCorresp(7,i) - translation.dot(n_i)) - planeCorresp(3,i));
//    const float angle_error = acos(n_i .dot (n_ii ));// * (180/PI);
////    const float angle_error = (n_i .cross (n_ii )).norm();
////    cout << "n_i " << n_i.transpose() << " n_ii " << n_ii.transpose() << "\n";
////    cout << "angle_error " << angle_error << " " << n_i .dot (n_ii ) << " d_error " << d_error << "\n";
//
//    if (d_error < distanceThreshold)
//     if (angle_error < 0.035){//cout << "new inlier\n";
//      out_inlierIndices.push_back(i);}
//  }
//}
//
///** Return "true" if the selected points are a degenerate (invalid) case.
//  */
//bool ransac3Dplane_degenerate(
//        const CMatrixDouble &planeCorresp,
//        const mrpt::vector_size_t &useIndices )
//{
//  ASSERT_( useIndices.size()==3 )
//
//  const Eigen::Vector3f n_1 = Eigen::Vector3f(planeCorresp(0,useIndices[0]), planeCorresp(1,useIndices[0]), planeCorresp(2,useIndices[0]));
//  const Eigen::Vector3f n_2 = Eigen::Vector3f(planeCorresp(0,useIndices[1]), planeCorresp(1,useIndices[1]), planeCorresp(2,useIndices[1]));
//  const Eigen::Vector3f n_3 = Eigen::Vector3f(planeCorresp(0,useIndices[2]), planeCorresp(1,useIndices[2]), planeCorresp(2,useIndices[2]));
////cout << "degenerate " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << " - " << n_1.transpose() << " v " << n_2.transpose() << " v " << n_3.transpose() << endl;
//
//  Eigen::Matrix3f condTranslacion = n_1*n_1.transpose() + n_2*n_2.transpose() + n_3*n_3.transpose();
////cout << "degenerate " << condTranslacion.determinant() << endl;
//  if( condTranslacion.determinant() < 0.01 )
////  if( fabs(n_1. dot( n_2. cross(n_3) ) ) < 0.9 )
//    return true;
//
//  return false;
//}

/*! This class calibrates the extrinsic parameters of the omnidirectional RGB-D sensor. For that, the sensor is accessed
 *  and big planes are segmented and matched between different single sensors.
*/
class OnlinePairCalibration_
{
  private:

    boost::mutex visualizationMutex;

    mrpt::pbmap::PbMap planes_i, planes_j;
    pcl::PointCloud<PointT>::Ptr cloud_i, cloud_j;
    pcl::PointCloud<PointT>::Ptr scan;
    Eigen::Matrix4f initOffset;

    bool bTakeSample;
    bool bDoCalibration;
    bool bFreezeFrame;
//    bool bTakeKeyframe;

    Calib360 calib;
//    Frame360 *frame_360;
//    pcl::PointCloud<PointT>::Ptr calibrated_cloud;
//    mrpt::pbmap::PbMap planes;

    Eigen::Matrix4f Rt_estimated[NUM_SENSORS];//, Rt_estimated_temp;
//    vector<mrpt::math::CMatrixDouble> correspondences;
    float weight_pair[NUM_SENSORS];
//    vector< vector< pair< Eigen::Vector4f, Eigen::Vector4f> > > correspondences;
    vector< pair< Eigen::Vector4f, Eigen::Vector4f> > correspondences;
//    vector< vector< pair< Eigen::Vector4f, Eigen::Vector4f> > > correspondences_2;
//    Eigen::Matrix3f covariances[NUM_SENSORS];
//    float conditioning[NUM_SENSORS];
//    unsigned conditioning_measCount[NUM_SENSORS];
    Eigen::Matrix3f covariance;
    float conditioning;
    unsigned conditioning_measCount;

    map<unsigned, unsigned> plane_corresp;

  public:
    OnlinePairCalibration_() :
              cloud_i(new pcl::PointCloud<PointT>()),
              cloud_j(new pcl::PointCloud<PointT>()),
              scan(new pcl::PointCloud<PointT>()),
              bDoCalibration(false),
              bTakeSample(false),
              bFreezeFrame(false)
    {
//      correspondences.resize(8); // 8 pairs of RGBD sensors
//      correspondences_2.resize(8); // 8 pairs of RGBD sensors
//      std::fill(conditioning, conditioning+8, 9999.9);
//      std::fill(weight_pair, weight_pair+8, 0.0);
//      std::fill(conditioning_measCount, conditioning_measCount+8, 0);
//      std::fill(covariances, covariances+8, Eigen::Matrix3f::Zero());
//      calib.loadIntrinsicCalibration();
//      calib.loadExtrinsicCalibration();
      conditioning = 9999.9;
      conditioning_measCount = 0;
      covariance = Eigen::Matrix3f::Zero();
    }


  /*! This function segments planes from the point cloud */
    void segmentPlanesInFrame(CloudRGBD_Ext &cloudImg, mrpt::pbmap::PbMap &planes)
    {
      // Downsample and filter point cloud
//      DownsampleRGBD downsampler(2);
//      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampledCloud = downsampler.downsamplePointCloud(cloudImg.getPointCloud());
//      cloudImg.setPointCloud(downsampler.downsamplePointCloud(cloudImg.getPointCloud()));
      pcl::FastBilateralFilter<pcl::PointXYZRGBA> filter;
      filter.setSigmaS (10.0);
      filter.setSigmaR (0.05);
//      filter.setInputCloud(cloudImg.getDownsampledPointCloud(2));
      filter.setInputCloud(cloudImg.getPointCloud());
      filter.filter(*cloudImg.getPointCloud());

      // Segment planes
      std::cout << "extractPlaneFeatures, size " << cloudImg.getPointCloud()->size() << "\n";
      double extractPlanes_start = pcl::getTime();
    assert(cloudImg.getPointCloud()->height > 1 && cloudImg.getPointCloud()->width > 1);

      planes.vPlanes.clear();

      pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  //      ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);
  //      ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);
      ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
  //      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
      ne.setMaxDepthChangeFactor (0.02); // For VGA: 0.02f, 10.01
      ne.setNormalSmoothingSize (8.0f);
      ne.setDepthDependentSmoothing (true);

      pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
  //      mps.setMinInliers (std::max(uint32_t(40),cloudImg.getPointCloud()->height*2));
      mps.setMinInliers (min_inliers);
      mps.setAngularThreshold (0.039812); // (0.017453 * 2.0) // 3 degrees
      mps.setDistanceThreshold (0.02); //2cm
  //    cout << "PointCloud size " << cloudImg.getPointCloud()->size() << endl;

      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
      ne.setInputCloud ( cloudImg.getPointCloud() );
      ne.compute (*normal_cloud);

      mps.setInputNormals (normal_cloud);
      mps.setInputCloud ( cloudImg.getPointCloud() );
      std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
      std::vector<pcl::ModelCoefficients> model_coefficients;
      std::vector<pcl::PointIndices> inlier_indices;
      pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
      std::vector<pcl::PointIndices> label_indices;
      std::vector<pcl::PointIndices> boundary_indices;
      mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

      // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
      unsigned single_cloud_size = cloudImg.getPointCloud()->size();
      for (size_t i = 0; i < regions.size (); i++)
      {
        mrpt::pbmap::Plane plane;

        plane.v3center = regions[i].getCentroid();
        plane.v3normal = Eigen::Vector3f(model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);
        if( plane.v3normal.dot(plane.v3center) > 0)
        {
          plane.v3normal = -plane.v3normal;
  //          plane.d = -plane.d;
        }
        plane.curvature = regions[i].getCurvature ();
  //    cout << i << " getCurvature\n";

  //        if(plane.curvature > max_curvature_plane)
  //          continue;

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
        extract.setInputCloud ( cloudImg.getPointCloud() );
        extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices[i]) );
        extract.setNegative (false);
        extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
        plane.inliers = inlier_indices[i].indices;

        Eigen::Matrix3f cov_normal = Eigen::Matrix3f::Zero();
        Eigen::Vector3f cov_nd = Eigen::Vector3f::Zero();
        Eigen::Vector3f gravity_center = Eigen::Vector3f::Zero();
        for(size_t j=0; j < inlier_indices[i].indices.size(); j++)
        {
          Eigen::Vector3f pt; pt << plane.planePointCloudPtr->points[j].x, plane.planePointCloudPtr->points[j].y, plane.planePointCloudPtr->points[j].z;
          gravity_center += pt;
        }
        cov_nd = gravity_center;
        gravity_center /= plane.planePointCloudPtr->size();
//      cout << "gravity_center " << gravity_center.transpose() << "   " << plane.v3center.transpose() << endl;
//        Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
        for(size_t j=0; j < inlier_indices[i].indices.size(); j++)
        {
          Eigen::Vector3f pt; pt << plane.planePointCloudPtr->points[j].x, plane.planePointCloudPtr->points[j].y, plane.planePointCloudPtr->points[j].z;
          cov_normal += -pt*pt.transpose();// + (plane.v3normal.dot(pt-gravity_center))*(plane.v3normal.dot(pt))*Eigen::Matrix3f::Identity();
//          Eigen::Vector3f pt_rg = (pt-gravity_center);
//          M += pt_rg * pt_rg.transpose();
        }
//        Eigen::JacobiSVD<Eigen::Matrix3f> svdM(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
//      cout << "normalV " << plane.v3normal.transpose() << " covM \n" << svdM.matrixU() << endl;

        Eigen::Matrix4f fim;//, covariance;
        fim.block(0,0,3,3) = cov_normal;
        fim.block(0,3,3,1) = cov_nd;
        fim.block(3,0,1,3) = cov_nd.transpose();
        fim(3,3) = -plane.planePointCloudPtr->size();
        fim *= 1 / SIGMA2;
        Eigen::JacobiSVD<Eigen::Matrix4f> svd(fim, Eigen::ComputeFullU | Eigen::ComputeFullV);
        svd.pinv(plane.information);
//        std::cout << "covariance \n" << plane.information << std::endl;
        plane.information = -fim;
//
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr contourPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);
        contourPtr->points = regions[i].getContour();
//        std::vector<size_t> indices_hull;

//      cout << "Extract contour\n";
        if(contourPtr->size() != 0)
        {
//      cout << "Extract contour2 " << contourPtr->size() << "\n";
          plane.calcConvexHull(contourPtr);
        }
        else
        {
  //        assert(false);
        std::cout << "HULL 000\n" << plane.planePointCloudPtr->size() << std::endl;
          static pcl::VoxelGrid<pcl::PointXYZRGBA> plane_grid;
          plane_grid.setLeafSize(0.05,0.05,0.05);
          plane_grid.setInputCloud (plane.planePointCloudPtr);
          plane_grid.filter (*contourPtr);
          plane.calcConvexHull(contourPtr);
        }

  //        assert(contourPtr->size() > 0);
  //        plane.calcConvexHull(contourPtr);
//      cout << "calcConvexHull\n";
        plane.computeMassCenterAndArea();
  //    cout << "Extract convexHull\n";
        // Discard small planes
//        if(plane.areaHull < min_area_plane)
//          continue;

        plane.d = -plane.v3normal .dot( plane.v3center );

        plane.calcElongationAndPpalDir();
        // Discard narrow planes
//        if(plane.elongation > max_elongation_plane)
//          continue;

        double color_start = pcl::getTime();
        plane.calcPlaneHistH();
        plane.calcMainColor2();
        double color_end = pcl::getTime();
//      std::cout << "color in " << (color_end - color_start)*1000 << " ms\n";

  //      color_start = pcl::getTime();
//        plane.transform(initOffset);
  //      color_end = pcl::getTime();
  //    std::cout << "transform in " << (color_end - color_start)*1000 << " ms\n";

        bool isSamePlane = false;
        if(plane.curvature < max_curvature_plane)
          for (size_t j = 0; j < planes.vPlanes.size(); j++)
            if( planes.vPlanes[j].curvature < max_curvature_plane && planes.vPlanes[j].isSamePlane(plane, 0.99, 0.05, 0.2) ) // The planes are merged if they are the same
            {
//            cout << "Merge local region\n";
              isSamePlane = true;
  //            double time_start = pcl::getTime();
              planes.vPlanes[j].mergePlane2(plane);
  //            double time_end = pcl::getTime();
  //          std::cout << " mergePlane2 took " << double (time_start - time_end) << std::endl;

              break;
            }
        if(!isSamePlane)
        {
//          cout << "New plane\n";
  //          plane.calcMainColor();
          plane.id = planes.vPlanes.size();
          planes.vPlanes.push_back(plane);
        }
      }
        double extractPlanes_end = pcl::getTime();
      std::cout << "segmentPlanesInFrame in " << (extractPlanes_end - extractPlanes_start)*1000 << " ms\n";
    }


//    void trimOutliersRANSAC(mrpt::math::CMatrixDouble &matched_planes, mrpt::math::CMatrixDouble &FIM_values)
//    {
//      cout << "trimOutliersRANSAC... " << endl;
//
//    //  assert(matched_planes.size() >= 3);
//    //  CTicTac tictac;
//
//      if(matched_planes.getRowCount() <= 3)
//      {
//        cout << "Insuficient matched planes " << matched_planes.getRowCount() << endl;
////        return Eigen::Matrix4f::Identity();
//        return;
//      }
//
//      CMatrixDouble planeCorresp(8, matched_planes.getRowCount());
//      planeCorresp = matched_planes.block(0,0,matched_planes.getRowCount(),8).transpose();
//
//      mrpt::vector_size_t inliers;
//    //  Eigen::Matrix4f best_model;
//      CMatrixDouble best_model;
//
//      math::RANSAC::execute(planeCorresp,
//                            ransacPlaneAlignment_fit,
//                            ransac3Dplane_distance,
//                            ransac3Dplane_degenerate,
//                            0.05, // threshold
//                            3,  // Minimum set of points
//                            inliers,
//                            best_model,
//                            false,   // Verbose
//                            0.51, // probGoodSample
//                            1000 // maxIter
//                            );
//
//    //  cout << "Computation time: " << tictac.Tac()*1000.0/TIMES << " ms" << endl;
//
//      cout << "Size planeCorresp: " << size(planeCorresp,2) << endl;
//      cout << "RANSAC finished: " << inliers.size() << " from " << matched_planes.getRowCount() << ". \nBest model: \n" << best_model << endl;
//    //        cout << "Best inliers: " << best_inliers << endl;
//
//      mrpt::math::CMatrixDouble trimMatchedPlanes(inliers.size(), matched_planes.getColCount());
//      mrpt::math::CMatrixDouble trimFIM_values(inliers.size(), FIM_values.getColCount());
//      std::vector<double> row;
//      for(unsigned i=0; i < inliers.size(); i++)
//      {
//        trimMatchedPlanes.row(i) = matched_planes.row(inliers[i]);
//        trimFIM_values.row(i) = FIM_values.row(inliers[i]);
//      }
//
//      matched_planes = trimMatchedPlanes;
//      FIM_values = trimFIM_values;
//    }

    void run()
    {
        int rc = openni::OpenNI::initialize();
        printf("After initialization:\n %s\n", openni::OpenNI::getExtendedError());

        // Show devices list
        openni::Array<openni::DeviceInfo> deviceList;
        openni::OpenNI::enumerateDevices(&deviceList);
        printf("Get device list. %d devices connected\n", deviceList.getSize() );
        for (unsigned i=0; i < deviceList.getSize(); i++)
        {
          printf("Device %u: name=%s uri=%s vendor=%s \n", i+1 , deviceList[i].getName(), deviceList[i].getUri(), deviceList[i].getVendor());
        }
        if(deviceList.getSize() == 0)
        {
          cout << "No devices connected -> EXIT\n";
          return;
        }

    //    grabber->showDeviceList();
        int device_ref, device_trg; // = static_cast<int>(getchar());
        cout << "Choose the reference camera: ";
        cin >> device_ref;
        cout << "Choose the target camera: ";
        cin >> device_trg;

        cout << "Choose the resolution: 0 (640x480) or 1 (320x240): ";
        int resolution; // = static_cast<int>(getchar());
        cin >> resolution;
//        cout << "device_ref " << device_ref << " resolution " << resolution << endl;

        grabber[0] = new RGBDGrabber_OpenNI2(deviceList[device_ref-1].getUri(), resolution);
        grabber[1] = new RGBDGrabber_OpenNI2(deviceList[device_trg-1].getUri(), resolution);

        // Initialize the sensor
        for(int sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
  //      #pragma omp parallel num_threads(8)
        {
  //        int sensor_id = omp_get_thread_num();
          grabber[sensor_id]->init();
        }
        cout << "Grabber initialized\n";

        unsigned countMeas = 0;
        unsigned frame = 0;

        // Initialize visualizer
        pcl::visualization::CloudViewer viewer("PairCalibrator");
        viewer.runOnVisualizationThread (boost::bind(&OnlinePairCalibration_::viz_cb, this, _1), "viz_cb");
        viewer.registerKeyboardCallback ( &OnlinePairCalibration_::keyboardEventOccurred, *this );

        float angle_offset = 20;
//        float angle_offset = 180;
        initOffset = Eigen::Matrix4f::Identity();
        initOffset(1,1) = initOffset(2,2) = cos(angle_offset*3.14159/180);
        initOffset(1,2) = -sin(angle_offset*3.14159/180);
        initOffset(2,1) = -initOffset(1,2);

        PairCalibrator calibrator;
        // Get the plane correspondences
//        calibrator.correspondences = mrpt::math::CMatrixDouble(0,10);
        calibrator.correspondences = mrpt::math::CMatrixDouble(0,18);
        calibrator.setInitRt(initOffset);

        CalibPairLaserKinect calibLaserKinect;
        calibLaserKinect.correspondences = mrpt::math::CMatrixDouble(0,10);

        CMatrixDouble conditioningFIM(0,6);
        Eigen::Matrix3f FIMrot = Eigen::Matrix3f::Zero();
        Eigen::Matrix3f FIMtrans = Eigen::Matrix3f::Zero();

        while (!viewer.wasStopped() && !bDoCalibration)
//        while (true && calibLaserKinect.correspondences.getRowCount() < 20)
        {
          frame++;

          CloudRGBD_Ext frameRGBD_[NUM_SENSORS];

//          for(int sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
//            grabber[sensor_id]->grab(&frameRGBD_[sensor_id]);
          #pragma omp parallel num_threads(NUM_SENSORS)
          {
            int sensor_id = omp_get_thread_num();
            grabber[sensor_id]->grab(&frameRGBD_[sensor_id]);
          }
//cout << "run3\n";

          if(frame < 2)
            continue;

//          if(bTakeSample == true)
//            continue;

          { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
//          boost::mutex::scoped_lock updateLock(visualizationMutex);

//            cloud_i.reset(new pcl::PointCloud<PointT>());
              cloud_i = frameRGBD_[0].getDownsampledPointCloud(2);
//            *cloud_i = *frameRGBD_[0].getPointCloud();
//            pcl::transformPointCloud(*frameRGBD_[1].getPointCloud(), *cloud_j, calibrator.Rt_estimated);
//            *cloud_j = *frameRGBD_[1].getPointCloud();
//            frameRGBD_[1].getDownsampledPointCloud(2);
//            // Vertical row scan wrt RGBD360 ref
//            cloud_j->points.resize(frameRGBD_[1].getPointCloud()->width);
//            for(unsigned i=0; i < frameRGBD_[1].getPointCloud()->width)
//              cloud_j->points[i] = frameRGBD_[1].getPointCloud()->points[i];
            // Horizontal row scan wrt RGBD360 ref
            scan->points.resize(frameRGBD_[1].getPointCloud()->height);
            for(unsigned i=0; i < frameRGBD_[1].getPointCloud()->height; i++)
              scan->points[i] = frameRGBD_[1].getPointCloud()->points[i*frameRGBD_[1].getPointCloud()->width];
            pcl::transformPointCloud(*scan, *cloud_j, calibrator.Rt_estimated);

//          updateLock.unlock();
//          } // CS_visualize

            segmentPlanesInFrame(frameRGBD_[0], planes_i);
          cout << "segmentPlanesInFrame " << planes_i.vPlanes.size() << endl;

            Eigen::Matrix<float,Eigen::Dynamic,6> lines;
            ransac_detect_3D_lines(cloud_j, lines, 0.1, 140);
          cout << "ransac_detect_3D_lines " << lines.getRowCount() << endl;

//            break;

            if(planes_i.vPlanes.size() == 1 && lines.getRowCount() == 1)
            {
              unsigned prevSize = calibLaserKinect.correspondences.getRowCount();
              calibLaserKinect.correspondences.setSize(prevSize+1, calibLaserKinect.correspondences.getColCount());
              calibLaserKinect.correspondences(prevSize, 0) = planes_i.vPlanes[0].v3normal[0];
              calibLaserKinect.correspondences(prevSize, 1) = planes_i.vPlanes[0].v3normal[1];
              calibLaserKinect.correspondences(prevSize, 2) = planes_i.vPlanes[0].v3normal[2];
              calibLaserKinect.correspondences(prevSize, 3) = planes_i.vPlanes[0].d;
              calibLaserKinect.correspondences(prevSize, 4) = lines(0,0);
              calibLaserKinect.correspondences(prevSize, 5) = lines(0,1);
              calibLaserKinect.correspondences(prevSize, 6) = lines(0,2);
              calibLaserKinect.correspondences(prevSize, 7) = lines(0,3);
              calibLaserKinect.correspondences(prevSize, 8) = lines(0,4);
              calibLaserKinect.correspondences(prevSize, 9) = lines(0,5);

              cout << "\tget Planes And Lines In Frame " << calibLaserKinect.correspondences.getRowCount() << endl;
              FIMtrans += planes_i.vPlanes[0].v3normal * planes_i.vPlanes[0].v3normal.transpose();
              Eigen::JacobiSVD<Eigen::Matrix3f> svd(FIMtrans, Eigen::ComputeFullU | Eigen::ComputeFullV);
              conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
            cout << "conditioning " << conditioning << endl;
            }

//          updateLock.unlock();
          } // CS_visualize

//          plane_corresp.clear();
//          for(unsigned i=0; i < planes_i.vPlanes.size(); i++)
//          {
//            for(unsigned j=0; j < planes_j.vPlanes.size(); j++)
//            {
//              if( planes_i.vPlanes[i].inliers.size() > min_inliers && planes_j.vPlanes[j].inliers.size() > min_inliers &&
//                  planes_i.vPlanes[i].elongation < 5 && planes_j.vPlanes[j].elongation < 5 &&
//                  planes_i.vPlanes[i].v3normal .dot (planes_j.vPlanes[j].v3normal) > 0.99 &&
//                  fabs(planes_i.vPlanes[i].d - planes_j.vPlanes[j].d) < 0.1 //&&
////                    planes_i.vPlanes[i].hasSimilarDominantColor(planes_j.vPlanes[j],0.06) &&
////                    planes_i.vPlanes[planes_counter_i+i].isPlaneNearby(planes_j.vPlanes[planes_counter_j+j], 0.5)
//                )
//              {
//                bTakeSample = true;
//
//              cout << "\tAssociate planes " << endl;
//                Eigen::Vector4f pl1, pl2;
//                pl1.head(3) = planes_i.vPlanes[i].v3normal; pl1[3] = planes_i.vPlanes[i].d;
//                pl2.head(3) = planes_j_orig.vPlanes[j].v3normal; pl2[3] = planes_j_orig.vPlanes[j].d;
////              cout << "Corresp " << planes_i.vPlanes[i].v3normal.transpose() << " vs " << planes_j_orig.vPlanes[j].v3normal.transpose() << " = " << planes_j.vPlanes[j].v3normal.transpose() << endl;
//////                        float factorDistInliers = std::min(planes_i.vPlanes[i].inliers.size(), planes_j.vPlanes[j].inliers.size()) / std::max(planes_i.vPlanes[i].v3center.norm(), planes_j.vPlanes[j].v3center.norm());
////                        float factorDistInliers = (planes_i.vPlanes[i].inliers.size() + planes_j.vPlanes[j].inliers.size()) / (planes_i.vPlanes[i].v3center.norm() * planes_j.vPlanes[j].v3center.norm());
////                        weight_pair[couple_id] += factorDistInliers;
////                        pl1 *= factorDistInliers;
////                        pl2 *= factorDistInliers;
////                ++weight_pair[couple_id];
//
//                //Add constraints
////                  correspondences.push_back(pair<Eigen::Vector4f, Eigen::Vector4f>(pl1, pl2));
////                        correspondences[couple_id].push_back(pair<Eigen::Vector4f, Eigen::Vector4f>(pl1/planes_i.vPlanes[i].v3center.norm(), pl2/planes_j.vPlanes[j].v3center.norm());
//
//                // Calculate conditioning
//                ++conditioning_measCount;
//                covariance += pl2.head(3) * pl1.head(3).transpose();
//                Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
//                conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//              cout << "conditioning " << conditioning << endl;
//
////          if(bTakeSample)
//
//                unsigned prevSize = calibrator.correspondences.getRowCount();
//                calibrator.correspondences.setSize(prevSize+1, calibrator.correspondences.getColCount());
//                calibrator.correspondences(prevSize, 0) = pl1[0];
//                calibrator.correspondences(prevSize, 1) = pl1[1];
//                calibrator.correspondences(prevSize, 2) = pl1[2];
//                calibrator.correspondences(prevSize, 3) = pl1[3];
//                calibrator.correspondences(prevSize, 4) = pl2[0];
//                calibrator.correspondences(prevSize, 5) = pl2[1];
//                calibrator.correspondences(prevSize, 6) = pl2[2];
//                calibrator.correspondences(prevSize, 7) = pl2[3];
//
//                Eigen::Matrix4f informationFusion;
//                Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
//                tf.block(0,0,3,3) = calibrator.Rt_estimated.block(0,0,3,3);
//                tf.block(3,0,1,3) = -calibrator.Rt_estimated.block(0,3,3,1).transpose();
//                informationFusion = planes_i.vPlanes[i].information;
//                informationFusion += tf * planes_j_orig.vPlanes[j].information * tf.inverse();
//                Eigen::JacobiSVD<Eigen::Matrix4f> svd_cov(informationFusion, Eigen::ComputeFullU | Eigen::ComputeFullV);
//                Eigen::Vector4f minEigenVector = svd_cov.matrixU().block(0,3,4,1);
//              cout << "minEigenVector " << minEigenVector.transpose() << endl;
//                informationFusion -= svd.singularValues().minCoeff() * minEigenVector * minEigenVector.transpose();
//              cout << "informationFusion \n" << informationFusion << "\n minSV " << svd.singularValues().minCoeff() << endl;
//
//                calibrator.correspondences(prevSize, 8) = informationFusion(0,0);
//                calibrator.correspondences(prevSize, 9) = informationFusion(0,1);
//                calibrator.correspondences(prevSize, 10) = informationFusion(0,2);
//                calibrator.correspondences(prevSize, 11) = informationFusion(0,3);
//                calibrator.correspondences(prevSize, 12) = informationFusion(1,1);
//                calibrator.correspondences(prevSize, 13) = informationFusion(1,2);
//                calibrator.correspondences(prevSize, 14) = informationFusion(1,3);
//                calibrator.correspondences(prevSize, 15) = informationFusion(2,2);
//                calibrator.correspondences(prevSize, 16) = informationFusion(2,3);
//                calibrator.correspondences(prevSize, 17) = informationFusion(3,3);
//
//
//              FIMrot += -skew(planes_j_orig.vPlanes[j].v3normal) * informationFusion.block(0,0,3,3) * skew(planes_j_orig.vPlanes[j].v3normal);
//              FIMtrans += planes_i.vPlanes[i].v3normal * planes_i.vPlanes[i].v3normal.transpose() * informationFusion(3,3);
//
//              Eigen::JacobiSVD<Eigen::Matrix3f> svd_rot(FIMrot, Eigen::ComputeFullU | Eigen::ComputeFullV);
//              Eigen::JacobiSVD<Eigen::Matrix3f> svd_trans(FIMtrans, Eigen::ComputeFullU | Eigen::ComputeFullV);
////              float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//              conditioningFIM.setSize(prevSize+1, conditioningFIM.getColCount());
//              conditioningFIM(prevSize, 0) = svd_rot.singularValues()[0];
//              conditioningFIM(prevSize, 1) = svd_rot.singularValues()[1];
//              conditioningFIM(prevSize, 2) = svd_rot.singularValues()[2];
//              conditioningFIM(prevSize, 3) = svd_trans.singularValues()[0];
//              conditioningFIM(prevSize, 4) = svd_trans.singularValues()[1];
//              conditioningFIM(prevSize, 5) = svd_trans.singularValues()[2];

//                { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
//                boost::mutex::scoped_lock updateLock(visualizationMutex);
//
//                  // For visualization
//                  plane_corresp[i] = j;
//
//                updateLock.unlock();
//                } // CS_visualize

//                break;
//              }
//            }
//          }

//          calibrator.CalibrateRotationManifold(0);
//          calibrator.CalibrateTranslation(0);

//          if(conditioning < 100 & conditioning_measCount > 3)
//            calibrator.CalibratePair();

//          if(conditioning < 50 & conditioning_measCount > 30)
//            bDoCalibration = true;

//cout << "run9\n";

//          while(bTakeSample == true)
//            boost::this_thread::sleep (boost::posix_time::milliseconds(10));
        }

        // Stop grabbing
        for(unsigned sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
          delete grabber[sensor_id];

        openni::OpenNI::shutdown();

//        // Trim outliers
//        trimOutliersRANSAC(calibrator.correspondences, conditioningFIM);


        float threshold_conditioning = 800.0;
        if(conditioning < threshold_conditioning)
        {
        cout << "\tSave CorrespMat\n";
//          calibrator.correspondences.saveToTextFile( mrpt::format("%s/correspondences.txt", PROJECT_SOURCE_PATH) );
//          conditioningFIM.saveToTextFile( mrpt::format("%s/conditioningFIM.txt", PROJECT_SOURCE_PATH) );

          calibLaserKinect.CalibratePair();
          cout << "Rt_estimated \n" << calibLaserKinect.Rt_estimated << endl;
        }

    }

//    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
//    {
//      if ( event.keyDown () )
//      {
//        if(event.getKeySym () == "s" || event.getKeySym () == "S")
//          bDoCalibration = true;
//        else
//          bTakeSample = true;
//      }
//    }


    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
//    cout << "SphericalSequence::viz_cb(...)\n";
      if (cloud_i->empty() || bFreezeFrame)
      {
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
        return;
      }
//    cout << "   ::viz_cb(...)\n";

      viz.removeAllShapes();
      viz.removeAllPointClouds();

//      viz.setCameraPosition(0,0,-3,-1,0,0);
//      viz.setSize(640,480); // Set the window size
      viz.setSize(1280,960); // Set the window size
//      viz.setSize(800,800); // Set the window size
//      viz.setCameraPosition(0,0,-5,0,-0.707107,0.707107,1,0,0);

      { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
        boost::mutex::scoped_lock updateLock(visualizationMutex);

        if (!viz.updatePointCloud (cloud_i, "cloud_i"))
          viz.addPointCloud (cloud_i, "cloud_i");

        if (!viz.updatePointCloud (cloud_j, "sphereCloud"))
          viz.addPointCloud (cloud_j, "sphereCloud");

        // Draw camera system
        viz.removeCoordinateSystem();
        Eigen::Affine3f Rt;
        Rt.matrix() = initOffset;
        viz.addCoordinateSystem(0.05, Rt);
        viz.addCoordinateSystem(0.1, Eigen::Affine3f::Identity());

        char name[1024];

        sprintf (name, "%zu pts. Params ...", cloud_j->size());
        viz.addText (name, 20, 20, "params");

        // Draw planes
//        if(plane_corresp.size() > 0)
        {
//          bFreezeFrame = true;

          for(size_t i=0; i < planes_i.vPlanes.size(); i++)
          {
//            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
//              if(it->first == i)
//              {
            mrpt::pbmap::Plane &plane_i = planes_i.vPlanes[i];
            sprintf (name, "normal_%u", static_cast<unsigned>(i));
            pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
            pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
            pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
                                plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
                                plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
//            viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
            viz.addArrow (pt2, pt1, ared[0], agrn[0], ablu[0], false, name);

            sprintf (name, "approx_plane_%02d", int (i));
//            viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[5], 0.5 * grn[5], 0.5 * blu[5], name);
            viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[0], grn[0], blu[0], name);

//            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
//              if(it->first == i)
//              {
//                mrpt::pbmap::Plane &plane_i = planes_i.vPlanes[i];
//                sprintf (name, "normal_%u", static_cast<unsigned>(i));
//                pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
//                pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
//                pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
//                                    plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
//                                    plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
//                viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
//
//                sprintf (name, "approx_plane_%02d", int (i));
//                viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[0], grn[0], blu[0], name);
//
////                sprintf (name, "inliers_%02d", int (i));
////                pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[0], grn[0], blu[0]);
////                viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
//              }
          }
//          for(size_t i=0; i < planes_j.vPlanes.size(); i++)
//          {
////            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
////              if(it->second == i)
////              {
////    cout << "   planes_j " << i << "\n";
//
//            mrpt::pbmap::Plane &plane_i = planes_j.vPlanes[i];
//            sprintf (name, "normal_j_%u", static_cast<unsigned>(i));
//            pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
//            pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
//            pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
//                                plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
//                                plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
////            viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
//            viz.addArrow (pt2, pt1, ared[3], agrn[3], ablu[3], false, name);
//
//            sprintf (name, "approx_plane_j_%02d", int (i));
////            viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[5], 0.5 * grn[5], 0.5 * blu[5], name);
//            viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[3], grn[3], blu[3], name);
//
////            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
////              if(it->second == i)
////              {
////                mrpt::pbmap::Plane &plane_i = planes_j.vPlanes[i];
////                sprintf (name, "normal_j_%u", static_cast<unsigned>(i));
////                pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
////                pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
////                pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
////                                    plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
////                                    plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
////                viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
////
////                sprintf (name, "approx_plane_j_%02d", int (i));
////                viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[3], grn[3], blu[3], name);
////
//////                sprintf (name, "inliers_j_%02d", int (i));
//////                pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[3], grn[3], blu[3]);
//////                viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
////              }
//          }

          #if RECORD_VIDEO
            std::string screenshotFile = mrpt::format("im_%04u.png", ++numScreenshot);
            viz.saveScreenshot(screenshotFile);
          #endif
        }

      updateLock.unlock();
      }
    }

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
    {
      if ( event.keyDown () )
      {
        if(event.getKeySym () == "k" || event.getKeySym () == "K")
//          bTakeKeyframe = true;
          bDoCalibration = true;
        else if(event.getKeySym () == "l" || event.getKeySym () == "L"){
          bFreezeFrame = !bFreezeFrame;
          bTakeSample = !bTakeSample;
        }
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

//  // Handle interruptions
//  signal(SIGINT, INThandler);

  cout << "Create OnlinePairCalibration_ object\n";
  OnlinePairCalibration_ calib_rgbd360;
  calib_rgbd360.run();

  cout << "EXIT\n";

  return (0);
}
