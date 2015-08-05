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
//#include "Miscellaneous.h"
//#include <Saliency.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
//#include "/usr/local/include/eigen3/Eigen/Core"

#include "MEstimator.h"

/*! This class encapsulates different projection models including both perspective and spherical.
 *  It implements the functionality to project and reproject from the image domain to 3D and viceversa.
 *
 * TODO: This class could be split using polymorphism, now it uses function pointers and it is compatible with C.
 * Polymorphism has the advantage that the code is easier to read and tu reuse, but it may have a lower performance.
 */
class SphericalModel : public ProjectionModel , MEstimator
{
  protected:

    /*! Spherical model parameters */
    float pixel_angle;
    float pixel_angle_inv;
    float half_width;
    float row_phi_start;
    size_t start_row;

  public:

    SphericalModel();

    /*! Scale the intrinsic calibration parameters according to the image resolution (i.e. the reduced resolution being used). */
    void scaleCameraParams(std::vector<cv::Mat> & depthPyr, const int pyrLevel);

    /*! Return the depth value of the 3D point projected on the image.*/
    inline float getDepth(const Eigen::Vector3f &xyz)
    {
        return xyz.norm();
    }

//    /*! Return the depth value of the 3D point projected on the image.*/
//    inline float reProjDepth(const Eigen::Vector3f &xyz_src, const Eigen::Vector3f &xyz_trg)
//    {
//        return xyz.norm();
//    }

//    Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
//    Vector3f xyz_trg = LUT_xyz_target.block(warp_pixels_src(i),0,1,3).transpose();
//    float residual = (pose_guess_inv(2,0)*xyz_trg(0)+pose_guess_inv(2,1)*xyz_trg(1)+pose_guess_inv(2,2)*xyz_trg(2) + pose_guess_inv(2,3) - xyz_src(2)) * stdDevError_inv_src(i);

//    float residual = (((xyz_src .dot (xyz_trg - pose_guess.block(0,3,3,1))) / dist_src) - dist_src) * stdDevError_inv;


//    /*! Check if a pixel is within the image limits. */
//    template<typename T>
//    inline bool isInImage(const T x, const T y)
//    {
//        return ( y >= 0 && y < nRows );
//    }

    /*! Check if a pixel is within the image limits. */
    template<typename T>
    inline bool isInImage(const T y)
    {
        //cout << "isInImage " << y << " " << nRows << endl;
        return ( y >= 0 && y < nRows );
    }

    /*! Project 3D points XYZ. */
    inline cv::Point2f project2Image(Eigen::Vector3f & xyz)
    {
        //Project the 3D point to the 2D plane
        float dist = xyz.norm();
        float dist_inv = 1.f / dist;
        float phi = asin(xyz(1)*dist_inv);
        float theta = atan2(xyz(0),xyz(2));
        float transformed_r = phi*pixel_angle_inv + row_phi_start;
        //float transformed_r = phi*pixel_angle_inv + start_row;
        float transformed_c = theta*pixel_angle_inv + half_width; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
        assert(transformed_c < nCols);
        assert(transformed_c >= 0);
        //cout << "project2Image " << transformed_c << " " << transformed_r << " phi " << phi << " start_row " << start_row << " theta " << theta << " half_width " << half_width << endl;

        cv::Point2f pixel(transformed_c, transformed_r);
        return pixel;
    };

    /*! Project pixel spherical */
    inline void warp_pixel ( const int nCols, const int nRows, const float phi_FoV, const int c, const int r, const float depth,
                              const Eigen::Matrix4f & pose_guess,
                              int transformed_r_int, int transformed_c_int) // output parameters
    {
        float phi = r*(2*phi_FoV/nRows);
        float theta = c*(2*PI/nCols);

        //Compute the 3D coordinates of the pij of the source frame
        Eigen::Vector4f point3D;
        point3D(0) = depth*sin(phi);
        point3D(1) = -depth*cos(phi)*sin(theta);
        point3D(2) = -depth*cos(phi)*cos(theta);
        point3D(3) = 1;

        //Transform the 3D point using the transformation matrix Rt
        Eigen::Vector4f xyz = pose_guess*point3D;
        float depth_trg = sqrt(xyz(0)*xyz(0) + xyz(1)*xyz(1) + xyz(2)*xyz(2));

        //Project the 3D point to the S2 sphere
        float phi_trg = asin(xyz(2)/depth_trg);
        float theta_trg = atan2(-xyz(1),-xyz(2));
        transformed_r_int = round(phi_trg*nRows/phi_FoV + nRows/2);
        transformed_c_int = round(theta_trg*nCols/(2*PI));
    };

    /*! Warp the input images (gray and/or depth) according to a given geometric transformation Rt. */
    void warpImage( const cv::Mat gray,               // The original image
                    cv::Mat  & warped_gray,        // The warped image
                    const cv::Mat depth,               // The original image
                    cv::Mat & warped_depth,        // The warped image
                    const Eigen::Matrix4f & Rt,        // The relative pose of the robot between the two frames
                    const bool bilinear,
                    const int method );

    /*! Re-project the warping image into the reference one. The input points 'xyz' represent the reference point cloud as seen by the target image. */
    void reproject(const Eigen::MatrixXf & xyz, const cv::Mat & gray, cv::Mat & warped_gray, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible);

    /*! Project 3D points XYZ according to the spherical camera model (3D -> 2D). */
    void project(const Eigen::MatrixXf & xyz, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible);

    /*! Project 3D points XYZ according to the spherical camera model (3D -> 1D nearest neighbor). */
    void projectNN(const Eigen::MatrixXf & xyz, Eigen::VectorXi & valid_pixels, Eigen::VectorXi & warped_pixels);

    /*! Compute the 3D points XYZ by multiplying the unit sphere by the spherical depth image. */
    void reconstruct3D(const cv::Mat & depth_img, Eigen::MatrixXf & sphere_xyz, Eigen::VectorXi & validPixels);

    /*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
    void reconstruct3D_saliency ( const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels, const int method,
                                  const cv::Mat & depth_gradX, const cv::Mat & depth_gradY, const float max_depth_grad, const float thres_saliency_depth,
                                  const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY, const float thres_saliency_gray
                                ); // TODO extend this function to employ only depth

    /*! Compute the unit sphere for the given spherical image dimmensions. This serves as a LUT to speed-up calculations. */
    void reconstruct3D_unitSphere();

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT(const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        float dist2 = xyz_transf.squaredNorm();
        float x2_z2 = dist2 - xyz_transf(1)*xyz_transf(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );

        jacobianWarpRt(0,0) = commonDer_c * xyz_transf(2);
        jacobianWarpRt(0,1) = 0.f;
        jacobianWarpRt(0,2) = -commonDer_c * xyz_transf(0);
//        jacobianWarpRt(1,0) = commonDer_r * xyz_transf(0) * xyz_transf(1);
        jacobianWarpRt(1,1) =-commonDer_r * x2_z2;
//        jacobianWarpRt(1,2) = commonDer_r * xyz_transf(2) * xyz_transf(1);
        float commonDer_r_y = commonDer_r * xyz_transf(1);
        jacobianWarpRt(1,0) = commonDer_r_y * xyz_transf(0);
        jacobianWarpRt(1,2) = commonDer_r_y * xyz_transf(2);

        jacobianWarpRt(0,3) = jacobianWarpRt(0,2) * xyz_transf(1);
        jacobianWarpRt(0,4) = jacobianWarpRt(0,0) * xyz_transf(2) - jacobianWarpRt(0,2) * xyz_transf(0);
        jacobianWarpRt(0,5) =-jacobianWarpRt(0,0) * xyz_transf(1);
        jacobianWarpRt(1,3) =-jacobianWarpRt(1,1) * xyz_transf(2) + jacobianWarpRt(1,2) * xyz_transf(1);
        jacobianWarpRt(1,4) = jacobianWarpRt(1,0) * xyz_transf(2) - jacobianWarpRt(1,2) * xyz_transf(0);
        jacobianWarpRt(1,5) =-jacobianWarpRt(1,0) * xyz_transf(1) + jacobianWarpRt(1,1) * xyz_transf(0);

        //return jacobianWarpRt;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF of the inverse transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_inv(const Eigen::Vector3f & xyz_transf, const Eigen::Vector3f & xyz_orig, const Eigen::Matrix3f & rotation, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        // The Jacobian of the spherical projection
        Eigen::Matrix<float,2,3> jacobianProj23;
        float dist2 = xyz_transf.squaredNorm();
        float x2_z2 = dist2 - xyz_transf(1)*xyz_transf(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
        jacobianProj23(0,0) = commonDer_c * xyz_transf(2);
        jacobianProj23(0,1) = 0.f;
        jacobianProj23(0,2) =-commonDer_c * xyz_transf(0);
//        jacobianProj23(1,0) = commonDer_r * xyz_transf(0) * xyz_transf(1);
        jacobianProj23(1,1) =-commonDer_r * x2_z2;
//        jacobianProj23(1,2) = commonDer_r * xyz_transf(2) * xyz_transf(1);
        float commonDer_r_y = commonDer_r * xyz_transf(1);
        jacobianProj23(1,0) = commonDer_r_y * xyz_transf(0);
        jacobianProj23(1,2) = commonDer_r_y * xyz_transf(2);

        // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
        Eigen::Matrix<float,3,6> jacobianT36_inv;
        jacobianT36_inv.block(0,0,3,3) = -rotation.transpose();
        jacobianT36_inv.block(0,3,3,1) = xyz_orig(2)*rotation.block(0,1,3,1) - xyz_orig(1)*rotation.block(0,2,3,1);
        jacobianT36_inv.block(0,4,3,1) = xyz_orig(0)*rotation.block(0,2,3,1) - xyz_orig(2)*rotation.block(0,0,3,1);
        jacobianT36_inv.block(0,5,3,1) = xyz_orig(1)*rotation.block(0,0,3,1) - xyz_orig(0)*rotation.block(0,1,3,1);

        std::cout << "jacobianProj23 \n" << jacobianProj23 << "\n jacobianT36_inv \n" << jacobianT36_inv << std::endl;

        jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

        //return jacobianWarpRt;
    }

    /*! Compute the Jacobian of the warp */
    inline void
    computeJacobian23_warp(const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,3> &jacobianWarp)
    {
        // The Jacobian of the spherical projection
        float dist2 = xyz_transf.squaredNorm();
        float x2_z2 = dist2 - xyz_transf(1)*xyz_transf(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
        jacobianWarp(0,0) = commonDer_c * xyz_transf(2);
        jacobianWarp(0,1) = 0.f;
        jacobianWarp(0,2) =-commonDer_c * xyz_transf(0);
//        jacobianWarp(1,0) = commonDer_r * xyz_transf(0) * xyz_transf(1);
        jacobianWarp(1,1) =-commonDer_r * x2_z2;
//        jacobianWarp(1,2) = commonDer_r * xyz_transf(2) * xyz_transf(1);
        float commonDer_r_y = commonDer_r * xyz_transf(1);
        jacobianWarp(1,0) = commonDer_r_y * xyz_transf(0);
        jacobianWarp(1,2) = commonDer_r_y * xyz_transf(2);
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

    /*! Warp the image according to a given geometric transformation. */
    //void warpImage ( const int pyrLevel, const Eigen::Matrix4f &pose_guess, costFuncType method );

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
    void computeJacobiansPhoto(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_photo, float *_grayGradX, float *_grayGradY);
    void computeJacobiansPhoto2(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXi & warped_pixels, const float stdDevPhoto_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_photo, float *_grayGradX, float *_grayGradY);

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
    void computeJacobiansDepth(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY);

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
    void computeJacobiansPhotoDepth(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights,
                                    Eigen::MatrixXf & jacobians_photo, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY);

};
