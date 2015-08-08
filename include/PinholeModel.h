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

#pragma once

#include "ProjectionModel.h"
//#include <Saliency.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
//#include "/usr/local/include/eigen3/Eigen/Core"

/*! This class encapsulates different projection models including both perspective and spherical.
 *  It implements the functionality to project and reproject from the image domain to 3D and viceversa.
 *
 * TODO: This class could be split using polymorphism, now it uses function pointers and it is compatible with C.
 * Polymorphism has the advantage that the code is easier to read and tu reuse, but it may have a lower performance.
 */
class PinholeModel : public ProjectionModel
{
  protected:

    /*! Camera matrix (intrinsic parameters). This is only required for pinhole perspective sensors */
    Eigen::Matrix3f cameraMatrix;

    /*! Camera intrinsic parameters */
    float fx, fy, ox, oy;
    float inv_fx, inv_fy;

  public:

    PinholeModel();

    /*! Return the depth value of the 3D point projected on the image.*/
    inline float getDepth(const Eigen::Vector3f &xyz)
    {
        return xyz(2);
    }

    /*! Set the 3x3 matrix of (pinhole) camera intrinsic parameters used to obtain the 3D colored point cloud from the RGB and depth images.*/
    inline void setCameraMatrix(const Eigen::Matrix3f & camMat)
    {
        cameraMatrix = camMat;
    };

    /*! Scale the intrinsic calibration parameters according to the image resolution (i.e. the reduced resolution being used). */
    void scaleCameraParams(std::vector<cv::Mat> & depthPyr, const int pyrLevel);

//    /*! Check if a pixel is within the image limits. */
//    template<typename T>
//    inline bool isInImage(const T x, const T y)
//    {
//        return ( y >= 0 && y < nRows && x >= 0 && x < nCols );
//    }

    /*! Project 3D points XYZ. */
    inline cv::Point2f project2Image(Eigen::Vector3f & xyz)
    {
        //Project the 3D point to the 2D plane
        float inv_transf_z = 1.f/xyz(2);
        // 2D coordinates of the transformed pixel(r,c) of frame 1
        float transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
        float transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
        cv::Point2f pixel(transformed_c, transformed_r);
        return pixel;
    };

    /*! Re-project the warping image into the reference one. The input points 'xyz' represent the reference point cloud as seen by the target image. */
    void reproject(const Eigen::MatrixXf & xyz, const cv::Mat & gray, cv::Mat & warped_gray, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible);

    /*! Project 3D points XYZ according to the pinhole camera model (3D -> 2D). */
    void project(const Eigen::MatrixXf & xyz, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible);

    /*! Project 3D points XYZ according to the pinhole camera model (3D -> 1D nearest neighbor). */
    void projectNN(const Eigen::MatrixXf & xyz, Eigen::VectorXi & valid_pixels, Eigen::VectorXi & warped_pixels);

    /*! Compute the 3D points XYZ according to the pinhole camera model. */
    void reconstruct3D(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels);

    /*! Compute the 3D points XYZ according to the pinhole camera model. */
    void reconstruct3D_saliency ( const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels, const int method,
                                  const cv::Mat & depth_gradX, const cv::Mat & depth_gradY, const float max_depth_grad, const float thres_saliency_depth,
                                  const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY, const float thres_saliency_gray
                                );

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT(const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        float inv_transf_z = 1.0/xyz_transf(2);

        //Derivative with respect to x
        jacobianWarpRt(0,0)=fx*inv_transf_z;
        jacobianWarpRt(1,0)=0.f;

        //Derivative with respect to y
        jacobianWarpRt(0,1)=0.f;
        jacobianWarpRt(1,1)=fy*inv_transf_z;

        //Derivative with respect to z
        float inv_transf_z_2 = inv_transf_z*inv_transf_z;
        jacobianWarpRt(0,2)=-fx*xyz_transf(0)*inv_transf_z_2;
        jacobianWarpRt(1,2)=-fy*xyz_transf(1)*inv_transf_z_2;

        //Derivative with respect to \w_x
        jacobianWarpRt(0,3)=-fx*xyz_transf(1)*xyz_transf(0)*inv_transf_z_2;
        jacobianWarpRt(1,3)=-fy*(1+xyz_transf(1)*xyz_transf(1)*inv_transf_z_2);

        //Derivative with respect to \w_y
        jacobianWarpRt(0,4)= fx*(1+xyz_transf(0)*xyz_transf(0)*inv_transf_z_2);
        jacobianWarpRt(1,4)= fy*xyz_transf(0)*xyz_transf(1)*inv_transf_z_2;

        //Derivative with respect to \w_z
        jacobianWarpRt(0,5)=-fx*xyz_transf(1)*inv_transf_z;
        jacobianWarpRt(1,5)= fy*xyz_transf(0)*inv_transf_z;

        //return jacobianWarpRt;
    }

    /*! Compute the Jacobian of the warp */
    inline void
    computeJacobian23_warp(const Eigen::Vector3f & xyz, Eigen::Matrix<float,2,3> &jacobianWarp)
    {
        float inv_transf_z = 1.0/xyz(2);

        //Derivative with respect to x
        jacobianWarp(0,0)=fx*inv_transf_z;
        jacobianWarp(1,0)=0.f;

        //Derivative with respect to y
        jacobianWarp(0,1)=0.f;
        jacobianWarp(1,1)=fy*inv_transf_z;

        //Derivative with respect to z
        float inv_transf_z_2 = inv_transf_z*inv_transf_z;
        jacobianWarp(0,2)=-fx*xyz(0)*inv_transf_z_2;
        jacobianWarp(1,2)=-fy*xyz(1)*inv_transf_z_2;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wTTx(const Eigen::Matrix4f & Rt, const Eigen::Vector3f & xyz, const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        Eigen::Matrix<float,2,3> jacobianWarp;
        Eigen::Matrix<float,3,6> jacobianRt;

        computeJacobian23_warp(xyz_transf, jacobianWarp);
        computeJacobian36_Tx_p(Rt.block(0,0,3,3), xyz, jacobianRt);

        jacobianWarpRt = jacobianWarp * jacobianRt;
    }

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
    void computeJacobiansPhoto(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians, float *_grayGradX, float *_grayGradY);

    void computeJacobiansPhoto2(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXi & warped_pixels, const float stdDevPhoto_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_photo, float *_grayGradX, float *_grayGradY);

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
    void computeJacobiansDepth(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians, float *_depthGradX, float *_depthGradY);

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
    void computeJacobiansPhotoDepth(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights,
                                    Eigen::MatrixXf & jacobians_photo, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY);
};
