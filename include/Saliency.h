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

#ifndef SALIENCY_H
#define SALIENCY_H

/*! This class encapsulates different projection models including both perspective and spherical.
 *  It implements the functionality to project and reproject from the image domain to 3D and viceversa.
 */
class Saliency
{
//public:

    /*! A threshold to select salient pixels on the intensity and depth images.*/
    float thresSaliency;
    float thres_saliency_gray_;
    float thres_saliency_depth_;
    float _max_depth_grad;

    std::vector<size_t> salient_pixels_;
    std::vector<size_t> salient_pixels_photo;
    std::vector<size_t> salient_pixels_depth;

public:

    Saliency();

    /*! Set the saliency threshold of the intensity. */
    void setSaliencyThreshodIntensity(const float thres)
    {
        thres_saliency_gray_ = thres;
    };

    /*! Set the saliency threshold of the depth. */
    void setSaliencyThreshodDepth(const float thres)
    {
        thres_saliency_depth_ = thres;
    };

    /*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
    void getSalientPoints_sphere(Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                 const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                 const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY
                                ); // TODO extend this function to employ only depth

    /*! Get a list of salient points from a list of Jacobians corresponding to a set of 3D points */
    void getSalientPts(const Eigen::MatrixXf & jacobians, std::vector<size_t> & salient_pixels, const float r_salient = 0.05f );

    void trimValidPoints(Eigen::MatrixXf & LUT_xyz, Eigen::VectorXi & validPixels, Eigen::MatrixXf & xyz_transf,
                         Eigen::VectorXi & validPixelsPhoto, Eigen::VectorXi & validPixelsDepth,
                         costFuncType method,
                         std::vector<size_t> &salient_pixels, std::vector<size_t> &salient_pixels_photo, std::vector<size_t> &salient_pixels_depth);
};

#endif
