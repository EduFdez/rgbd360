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
 * Author: Eduardo Fernandez-Moral
 */

#ifndef FRAME_RGBD_UNDISTORT
#define FRAME_RGBD_UNDISTORT

//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//#include <pcl/filters/fast_bilateral.h>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include "FrameRGBD.h"
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <clams/discrete_depth_distortion_model.h>

//#include <boost/lexical_cast.hpp>
//
////#include "../third_party/cvmat_serialization.h"
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>
//#include <fstream>

/*!
The class FrameRGBD_undistort inherites from FrameRGBD, and provides the functions to undistort its depth image.
*/
class FrameRGBD_undistort : public FrameRGBD
{
 private:

  /*! Intrinsic model to undistort the depth image*/
  clams::DiscreteDepthDistortionModel &intrinsic_model;

 protected:

  /*!Depth image used for intrinsic calibration*/
  Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> m_depthEigUndistort;

  /*!True if the depth image is already undistorted*/
  bool undistortedDepthAvailabe;

 public:

  FrameRGBD_undistort(clams::DiscreteDepthDistortionModel &clams_intrinsic_model)
    : intrinsic_model(clams_intrinsic_model),
      undistortedDepthAvailabe(false)
  {
//    intrinsic_model.load(mrpt::format("../../../Dropbox/Doctorado/Projects/RGBD360/Calibration/Intrinsics/distortion_model%i",sensor_id+1));
//    intrinsic_model.downsampleParams(2);
  };

  ~FrameRGBD_undistort(){};

//  /*!Set distortion model (it must be previously generated with CLAMS).*/
//  inline void setDistortionModel(clams::DiscreteDepthDistortionModel &clams_intrinsic_model)
//  {
//    intrinsic_model = &clams_intrinsic_model;
//  }

  /*!Get undistorted depth image.*/
  inline Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> & getUndistortedDepth()
  {
    if(!undistortedDepthAvailabe)
    {
      loadDepthEigen();
      intrinsic_model.undistort(&m_depthEigUndistort);

      undistortedDepthAvailabe = true;
    }
    return m_depthEigUndistort;
  }

  /*!Return the depth image of the RGBD frame.*/
  inline void loadDepthEigen()
  {
    cv::Mat depthInMeters;// = cv::Mat(m_depthImage.rows, m_depthImage.cols, CV_32FC1);
    getDepthImgMeters(depthInMeters);
    cv::cv2eigen(depthInMeters, m_depthEigUndistort);
//    std::cout << " u16 " << m_depthImage.at<unsigned short>(100,100) << std::endl;
//    m_depthImage.convertTo( m_depthImage, CV_32FC1, 0.001 ); //CV_16UC1
//  std::cout << " float " << m_depthImage.at<unsigned short>(100,100) << std::endl;
//    cv::cv2eigen(m_depthImage,m_depthEigUndistort);
  }

};
#endif
