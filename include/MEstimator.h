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

//#include <Miscellaneous.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#ifndef M_ESTIMATOR_H
#define M_ESTIMATOR_H

/*! Robust estimation.
 */
class MEstimator
//namespace mestimator
{
    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightHuber(const T & error)//, const T &scale)
    {
        //        assert(!std::isnan(error) && !std::isnan(scale))
        T weight = (T)1;
        const T scale = 1.345;
        T error_abs = fabs(error);
        if(error_abs < scale){//std::cout << "weight One\n";
            return weight;}

        weight = scale / error_abs;
        //std::cout << "weight " << weight << "\n";
        return weight;
    }

    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightHuber_sqrt(const T & error)//, const T &scale)
    {
        //        assert(!std::isnan(error) && !std::isnan(scale))
        T weight = (T)1;
        const T scale = 1.345;
        T error_abs = fabs(error);
        if(error_abs < scale){//std::cout << "weight One\n";
            return weight;}

        weight = sqrt(scale / error_abs);
        //std::cout << "weight " << weight << "\n";
        return weight;
    }

    /*! Tukey weight for robust estimation. */
    template<typename T>
    inline T weightTukey(const T & error)//, const T &scale)
    {
        T weight = (T)0.;
        const T scale = 4.685;
        T error_abs = fabs(error);
        if(error_abs > scale)
            return weight;

        T error_scale = error_abs/scale;
        T w_aux = 1 - error_scale*error_scale;
        weight = w_aux*w_aux;
        return weight;
    }

    /*! T-distribution weight for robust estimation. This is computed following the paper: "Robust Odometry Estimation for RGB-D Cameras" Kerl et al. ICRA 2013 */
    template<typename T>
    inline T weightTDist(const T & error, const T & stdDev, const T & nu)//, const T &scale)
    {
        T err_std = error / stdDev;
        T weight = nu+1 / (nu + err_std*err_std);
        return weight;
    }

    /*! Compute the standard deviation of the T-distribution following the paper: "Robust Odometry Estimation for RGB-D Cameras" Kerl et al. ICRA 2013 */
    template<typename T>
    inline T stdDev_TDist(const std::vector<T> & v_error, const T & stdDev, const T & nu)//, const T &scale)
    {
        std::vector<T> &v_error2( v_error.size() );
        for(size_t i=0; i < v_error.size(); ++i)
            v_error2[i] = v_error[i]*v_error[i];

        int it = 0;
        int max_iterations = 5;
        T diff_convergence = 1e-3;
        T diff_var = 100;
        T variance_prev = 10000;
        while (diff_var > diff_convergence && it < max_iterations)
        {
            T variance = 0.f;
            for(size_t i=0; i < v_error.size(); ++i)
                variance += v_error2[i]*weightTDist(v_error[i]);
            variance /= v_error.size();
            diff_var = fabs(variance_prev - variance);
            variance_prev = variance;
            ++it;
        }
        return sqrt(variance_prev);
    }

    template<typename T>
    inline T weightMEstimator( T error_scaled)
    {
        //std::cout << " error_scaled " << error_scaled << "weightHuber(error_scaled) " << weightHuber(error_scaled) << "\n";
        return weightHuber(error_scaled);
        //return weightTukey(error_scaled);
        //return weightTDist(error_scaled,dev,5);
    }


    /*! Return the value of the bilinear interpolation on the image 'img' given by the floating point indices 'x' and 'y' */
//    inline cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2f pt)
    //template <typename T>
    inline float bilinearInterp(const cv::Mat & img, cv::Point2f pt)
    {
        assert( img.type() == CV_32FC1 && !img.empty() );
        cv::Mat patch;
        cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
        return patch.at<float>(0,0);
    }

    /*! Return the value of the bilinear interpolation on the image 'img' given by the floating point indices 'x' and 'y'.
     * It takes into account NaN pixels and (<= 0 && > max_depth_) values to rule them out of the interpolation
     */
    inline float bilinearInterp_depth(const cv::Mat& img, const cv::Point2f &pt)
    {
        assert( img.type() == CV_32FC1 && !img.empty() );

        float *_img = reinterpret_cast<float*>(img.data);

        int x = (int)pt.x;
        int y = (int)pt.y;

        size_t x0_y0 = y * img.cols + x;
        size_t x1_y0 = x0_y0 + 1;
        size_t x0_y1 = x0_y0 + img.cols;
        size_t x1_y1 = x0_y1 + 1;

        float a = pt.x - (float)x;
        float b = 1.f - a;
        float c = pt.y - (float)y;
        float d = 1.f - c;

        float pt_y0;
        if( _img[x0_y0] < max_depth_ && _img[x1_y0] < max_depth_ && _img[x0_y0] >= 0 && _img[x1_y0] >= 0 )
            pt_y0 = _img[x0_y0] * b + _img[x1_y0] * a;
        else if (_img[x0_y0] < max_depth_ && _img[x0_y0] >= 0 )
            pt_y0 = _img[x0_y0];
        else //if(_img[x0_y0] < max_depth_)
            pt_y0 = _img[x1_y0];
        // The NaN/OutOfDepth case (_img[x0_y0] > max_depth_ && _img[x1_y0] > max_depth_) is automatically assumed

        float pt_y1;
        if( _img[x0_y1] < max_depth_ && _img[x1_y1] < max_depth_ && _img[x0_y1] >= 0 && _img[x1_y1] >= 0 )
            pt_y1 = _img[x0_y1] * b + _img[x1_y1] * a;
        else if (_img[x0_y1] < max_depth_ && _img[x0_y1] >= 0)
            pt_y1 = _img[x0_y1];
        else //if(_img[x0_y1] < max_depth_)
            pt_y1 = _img[x1_y1];

        float interpDepth;
        if( pt_y0 < max_depth_ && _img[x1_y1] < max_depth_ )
            interpDepth = pt_y0 * d + pt_y1 * c;
        else if (_img[x0_y1] < max_depth_)
            interpDepth = pt_y0;
        else
            interpDepth = pt_y1;

//        int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
//        int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
//        int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
//        int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

//        float pt_y0;
//        if( img.at<float>(y0, x0) < max_depth_ && img.at<float>(y0, x1) < max_depth_ && img.at<float>(y0, x0) >= 0 && img.at<float>(y0, x1) >= 0 )
//            pt_y0 = img.at<float>(y0, x0) * b + img.at<float>(y0, x1) * a;
//        else if (img.at<float>(y0, x0) < max_depth_ && img.at<float>(y0, x0) >= 0 )
//            pt_y0 = img.at<float>(y0, x0);
//        else //if(img.at<float>(y0, x0) < max_depth_)
//            pt_y0 = img.at<float>(y0, x1);
//        // The NaN/OutOfDepth case (img.at<float>(y0, x0) > max_depth_ && img.at<float>(y0, x1) > max_depth_) is automatically assumed

//        float pt_y1;
//        if( img.at<float>(y1, x0) < max_depth_ && img.at<float>(y1, x1) < max_depth_ && img.at<float>(y1, x0) >= 0 && img.at<float>(y1, x1) >= 0 )
//            pt_y1 = img.at<float>(y1, x0) * b + img.at<float>(y1, x1) * a;
//        else if (img.at<float>(y1, x0) < max_depth_ && img.at<float>(y1, x0) >= 0)
//            pt_y1 = img.at<float>(y1, x0);
//        else //if(img.at<float>(y1, x0) < max_depth_)
//            pt_y1 = img.at<float>(y1, x1);

//        float interpDepth;
//        if( pt_y0 < max_depth_ && img.at<float>(y1, x1) < max_depth_ )
//            interpDepth = pt_y0 * d + pt_y1 * c;
//        else if (img.at<float>(y1, x0) < max_depth_)
//            interpDepth = pt_y0;
//        else
//            interpDepth = pt_y1;

        return interpDepth;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_sphere(const Eigen::Vector3f & xyz, const float & dist, const float & pixel_angle_inv, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz(1)*xyz(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );

        jacobianWarpRt(0,0) = commonDer_c * xyz(2);
        jacobianWarpRt(0,1) = 0.f;
        jacobianWarpRt(0,2) = -commonDer_c * xyz(0);
//        jacobianWarpRt(1,0) = commonDer_r * xyz(0) * xyz(1);
        jacobianWarpRt(1,1) =-commonDer_r * x2_z2;
//        jacobianWarpRt(1,2) = commonDer_r * xyz(2) * xyz(1);
        float commonDer_r_y = commonDer_r * xyz(1);
        jacobianWarpRt(1,0) = commonDer_r_y * xyz(0);
        jacobianWarpRt(1,2) = commonDer_r_y * xyz(2);

        jacobianWarpRt(0,3) = jacobianWarpRt(0,2) * xyz(1);
        jacobianWarpRt(0,4) = jacobianWarpRt(0,0) * xyz(2) - jacobianWarpRt(0,2) * xyz(0);
        jacobianWarpRt(0,5) =-jacobianWarpRt(0,0) * xyz(1);
        jacobianWarpRt(1,3) =-jacobianWarpRt(1,1) * xyz(2) + jacobianWarpRt(1,2) * xyz(1);
        jacobianWarpRt(1,4) = jacobianWarpRt(1,0) * xyz(2) - jacobianWarpRt(1,2) * xyz(0);
        jacobianWarpRt(1,5) =-jacobianWarpRt(1,0) * xyz(1) + jacobianWarpRt(1,1) * xyz(0);

        //return jacobianWarpRt;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF of the inverse transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_sphere_inv(const Eigen::Vector3f & xyz, const Eigen::Vector3f & xyz_orig, const Eigen::Matrix3f & rotation, const float & dist, const float & pixel_angle_inv, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        // The Jacobian of the spherical projection
        Eigen::Matrix<float,2,3> jacobianProj23;
        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz(1)*xyz(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
        jacobianProj23(0,0) = commonDer_c * xyz(2);
        jacobianProj23(0,1) = 0;
        jacobianProj23(0,2) =-commonDer_c * xyz(0);
//        jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
        jacobianProj23(1,1) =-commonDer_r * x2_z2;
//        jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);
        float commonDer_r_y = commonDer_r * xyz(1);
        jacobianProj23(1,0) = commonDer_r_y * xyz(0);
        jacobianProj23(1,2) = commonDer_r_y * xyz(2);

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

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_pinhole(const Eigen::Vector3f & xyz, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        float inv_transf_z = 1.0/xyz(2);

        //Derivative with respect to x
        jacobianWarpRt(0,0)=fx*inv_transf_z;
        jacobianWarpRt(1,0)=0.f;

        //Derivative with respect to y
        jacobianWarpRt(0,1)=0.f;
        jacobianWarpRt(1,1)=fy*inv_transf_z;

        //Derivative with respect to z
        float inv_transf_z_2 = inv_transf_z*inv_transf_z;
        jacobianWarpRt(0,2)=-fx*xyz(0)*inv_transf_z_2;
        jacobianWarpRt(1,2)=-fy*xyz(1)*inv_transf_z_2;

        //Derivative with respect to \w_x
        jacobianWarpRt(0,3)=-fx*xyz(1)*xyz(0)*inv_transf_z_2;
        jacobianWarpRt(1,3)=-fy*(1+xyz(1)*xyz(1)*inv_transf_z_2);

        //Derivative with respect to \w_y
        jacobianWarpRt(0,4)= fx*(1+xyz(0)*xyz(0)*inv_transf_z_2);
        jacobianWarpRt(1,4)= fy*xyz(0)*xyz(1)*inv_transf_z_2;

        //Derivative with respect to \w_z
        jacobianWarpRt(0,5)=-fx*xyz(1)*inv_transf_z;
        jacobianWarpRt(1,5)= fy*xyz(0)*inv_transf_z;

        //return jacobianWarpRt;
    }

    /*! Compute the Jacobian of the warp */
    inline void
    computeJacobian23_warp_pinhole(const Eigen::Vector3f & xyz, Eigen::Matrix<float,2,3> &jacobianWarp)
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

    /*! Compute the Jacobian of the warp */
    inline void
    computeJacobian23_warp_sphere(const Eigen::Vector3f & xyz, const float & dist, const float & pixel_angle_inv, Eigen::Matrix<float,2,3> &jacobianWarp)
    {
        // The Jacobian of the spherical projection
        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz(1)*xyz(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
        jacobianWarp(0,0) = commonDer_c * xyz(2);
        jacobianWarp(0,1) = 0;
        jacobianWarp(0,2) =-commonDer_c * xyz(0);
//        jacobianWarp(1,0) = commonDer_r * xyz(0) * xyz(1);
        jacobianWarp(1,1) =-commonDer_r * x2_z2;
//        jacobianWarp(1,2) = commonDer_r * xyz(2) * xyz(1);
        float commonDer_r_y = commonDer_r * xyz(1);
        jacobianWarp(1,0) = commonDer_r_y * xyz(0);
        jacobianWarp(1,2) = commonDer_r_y * xyz(2);
    }

    /*! Compute the Jacobian composition of the transformed point: T(x)Tp */
    inline void
    //Eigen::Matrix<float,3,6>
    computeJacobian36_TxT_p(const Eigen::Vector3f & xyz, Eigen::Matrix<float,3,6> &jacobianRt)
    {
        //Eigen::Matrix<float,3,6> jacobianWarpRt;

        jacobianRt.block(0,0,3,3) = Eigen::Matrix3f::Identity();
        jacobianRt.block(0,3,3,3) = -skew(xyz);
    }

    /*! Compute the Jacobian composition of the transformed point: TT(x)p */
    inline void
    //Eigen::Matrix<float,3,6>
    computeJacobian36_TTx_p(const Eigen::Matrix3f & rot, const Eigen::Vector3f & xyz, Eigen::Matrix<float,3,6> &jacobianRt)
    {
        //Eigen::Matrix<float,3,6> jacobianWarpRt;

        jacobianRt.block(0,0,3,3) = Eigen::Matrix3f::Identity();
        jacobianRt.block(0,3,3,3) = -skew(xyz);

        jacobianRt = rot * jacobianRt;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wTTx_pinhole(const Eigen::Matrix4f & Rt, const Eigen::Vector3f & xyz, const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        Eigen::Matrix<float,2,3> jacobianWarp;
        Eigen::Matrix<float,3,6> jacobianRt;

        computeJacobian23_warp_pinhole(xyz_transf, jacobianWarp);
        computeJacobian36_TTx_p(Rt.block(0,0,3,3), xyz, jacobianRt);

        jacobianWarpRt = jacobianWarp * jacobianRt;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wTTx_sphere(const Eigen::Matrix4f & Rt, const Eigen::Vector3f & xyz, const float & dist, const float & pixel_angle_inv, const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        Eigen::Matrix<float,2,3> jacobianWarp;
        Eigen::Matrix<float,3,6> jacobianRt;

        computeJacobian23_warp_sphere(xyz_transf, dist, pixel_angle_inv, jacobianWarp);
        computeJacobian36_TTx_p(Rt.block(0,0,3,3), xyz, jacobianRt);

        jacobianWarpRt = jacobianWarp * jacobianRt;
    }
};

#endif
