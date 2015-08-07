/*
 *  Copyright (c) 2015,   INRIA Sophia Antipolis - LAGADIC Team
 *
 *  All rights reserved.
 *
 *  Redistribution and use in ref and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of ref code must retain the above copyright
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

#include <DirectRegistration.h>
#include <transformPts3D.h>
#include <config.h>
#include <definitions.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/common/time.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP
#include <pcl/registration/warp_point_rigid.h>

#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/obs/CObservation2DRangeScan.h>
#include <mrpt/slam/CICP.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/poses/CPosePDF.h>
#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/math/utils.h>
#include <mrpt/system/os.h>

// For SIMD performance
//#include "cvalarray.hpp"
//#include "veclib.h"

#if _SSE2
    #include <emmintrin.h>
    #include <mmintrin.h>
#endif
#if _SSE3
    #include <immintrin.h>
    #include <pmmintrin.h>
#endif
#if _SSE4_1
#  include <smmintrin.h>
#endif

#define ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS 1
#define INVALID_POINT -10000

using namespace std;
using namespace Eigen;

DirectRegistration::DirectRegistration(sensorType sensor) :
    bDifSensors(false),
    use_salient_pixels(false),
    compute_MAD_stdDev(false),
    use_bilinear(false),
    visualize(false),
    nPyrLevels(0),
    sensor_type(sensor),        //RGBD360_INDOOR, STEREO_OUTDOOR
    sensor_type_ref(sensor),
    sensor_type_trg(sensor),
    method(PHOTO_DEPTH),
    compositional(FC)
{
    if(sensor_type == KINECT)
        ProjModel = new PinholeModel;
    else //RGBD360_INDOOR, STEREO_OUTDOOR
        ProjModel = new SphericalModel;
    ProjModel_ref = ProjModel;
    ProjModel_trg = ProjModel;

    stdDevPhoto = 4./255;
    varPhoto = stdDevPhoto*stdDevPhoto;

    stdDevDepth = 0.01;
    varDepth = stdDevDepth*stdDevDepth;

    min_depth_Outliers = 2*stdDevDepth; // in meters
    max_depth_Outliers = 1; // in meters

    // Set the Saliency parameters
    thresSaliency = 0.04f;
//    thres_saliency_gray_ = 0.04f;
//    thres_saliency_depth_ = 0.04f;
    thres_saliency_gray_ = 0.001f;
    thres_saliency_depth_ = 0.001f;
    _max_depth_grad = 0.3f;

    // Set the optimization parameters
    max_iters_ = 10;
    tol_update_ = 1e-3;
    tol_update_rot_ = 1e-4;
    tol_update_trans_ = 1e-3;
    tol_residual_ = 1e-3;

    //        double lambda = 0.01; // Levenberg-Marquardt (LM) lambda
    //        double step = 10; // Update step
    //        unsigned LM_max_iters_ = 1;

    registered_pose = Matrix4f::Identity();
}

DirectRegistration::~DirectRegistration()
{
//    if(ProjModel_ref != ProjModel)
//        delete ProjModel_ref;
//    if(ProjModel_trg != ProjModel && ProjModel_trg != ProjModel_ref)
//        delete ProjModel_trg;

    delete ProjModel;
}

/*! Sets the ref (Intensity+Depth) frame.*/
void DirectRegistration::setSourceFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth)
{
    #if PRINT_PROFILING
    double time_start = pcl::getTime();
    #endif

    //Create a float auxialiary image from the imput image
    cv::Mat gray;
//    cv::cvtColor(imgRGB, gray, CV_RGB2GRAY);
    cv::cvtColor(imgRGB, gray, cv::COLOR_RGB2GRAY);
    gray.convertTo(gray, CV_32FC1, 1./255 );

    //Compute image pyramids for the grayscale and depth images
    buildPyramid(gray, ref.grayPyr, nPyrLevels);
    buildPyramidRange(imgDepth, ref.depthPyr, nPyrLevels);

    //Compute image pyramids for the gradients images
    buildGradientPyramids( ref.grayPyr, ref.grayPyr_GradX, ref.grayPyr_GradY,
                           ref.depthPyr, ref.depthPyr_GradX, ref.depthPyr_GradY,
                           nPyrLevels );

//    // This is intended to show occlussions
//    rgbRef = imgRGB;
//    buildPyramid(rgbRef, colorRef, nPyrLevels);

    #if PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "DirectRegistration::setSourceFrame construction " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Sets the ref (Intensity+Depth) frame. Depth image is ignored*/
void DirectRegistration::setTargetFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth)
{
    //cout << "DirectRegistration::setTargetFrame() \n";
    #if PRINT_PROFILING
    double time_start = pcl::getTime();
    #endif

    assert(imgRGB.rows == imgDepth.rows && imgRGB.cols == imgDepth.cols);
    assert(imgRGB.cols % (int)(pow(2,nPyrLevels)) == 0);

    //Create a float auxialiary image from the imput image
    cv::Mat gray;
//    cv::cvtColor(imgRGB, gray, CV_RGB2GRAY);
    cv::cvtColor(imgRGB, gray, cv::COLOR_RGB2GRAY);
    gray.convertTo(gray, CV_32FC1, 1./255 );

    //Compute image pyramids for the grayscale and depth images
    buildPyramid(gray, trg.grayPyr, nPyrLevels);
    buildPyramidRange(imgDepth, trg.depthPyr, nPyrLevels);

    //Compute image pyramids for the gradients images
    buildGradientPyramids( trg.grayPyr, trg.grayPyr_GradX, trg.grayPyr_GradY,
                           trg.depthPyr, trg.depthPyr_GradX, trg.depthPyr_GradY,
                           nPyrLevels );

    //        cv::imwrite("/home/efernand/test.png", grayTrgGrad[nPyrLevels]);
    //        cv::imshow("GradX_pyr ", grayTrgGrad[nPyrLevels]);
    //        cv::imshow("GradY_pyr ", grayTrgGrad[nPyrLevels]);
    //        cv::imshow("GradX ", grayTrgGrad[0]);
    //        cv::imshow("GradY ", grayTrgGrad[0]);
    //        cv::imshow("GradX_d ", depthTrgGrad[0]);
    //        cv::waitKey(0);

#if PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "DirectRegistration::setTargetFrame construction " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Swap the ref and target images */
void DirectRegistration::swapSourceTarget()
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    ASSERT_(!bDifSensors);
    trg = ref;

//    grayTrg = ref.grayPyr;
//    grayTrgGrad = grayRefGrad;
//    grayTrgGrad = grayRefGrad;
//    depthTrg = ref.depthPyr;
//    depthTrgGrad = depthRefGrad;
//    depthTrgGrad = depthRefGrad;

//    cv::imshow( "sphereGray", grayTrg[1] );
//    cv::imshow( "sphereGray1", grayTrg[2] );
//    cv::imshow( "sphereDepth2", depthTrg[3] );
//    cv::waitKey(0);

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << "DirectRegistration::swapSourceTarget took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
double DirectRegistration::computeError(const Matrix4f & poseGuess)
{
    //cout << " DirectRegistration::computeError \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t n_pts = itRef.xyz.rows();
    transformPts3D(itRef.xyz, poseGuess, itRef.xyz_tf);
    MatrixXf poseGuess_inv = poseGuess.inverse();

    float stdDevPhoto_inv = 1./stdDevPhoto;
    itRef.residualsPhoto = VectorXf::Zero(n_pts);
    if(method == DIRECT_ICP)
        itRef.residualsDepth = VectorXf::Zero(3*n_pts);
    else
        itRef.residualsDepth = VectorXf::Zero(n_pts);
    itRef.stdDevError_inv = VectorXf::Zero(n_pts);
    itRef.wEstimPhoto = VectorXf::Zero(n_pts);
    itRef.wEstimDepth = VectorXf::Zero(n_pts);
    itRef.validPixelsPhoto = VectorXi::Zero(n_pts);
    itRef.validPixelsDepth = VectorXi::Zero(n_pts);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    //std::vector<float> v_AD_intensity(imgSize);

    const float *_depthRef = reinterpret_cast<float*>(ref.depthPyr[currPyrLevel].data);
    const float *_depthTrg = reinterpret_cast<float*>(trg.depthPyr[currPyrLevel].data);
    float *_grayRef = reinterpret_cast<float*>(ref.grayPyr[currPyrLevel].data);
    float *_grayTrg = reinterpret_cast<float*>(trg.grayPyr[currPyrLevel].data);

    //cout << " use_salient_pixels " << use_salient_pixels << " use_bilinear " << use_bilinear << " pts " << itRef.xyz.rows()  << endl;

    // Set the correct formulation for the depth error according to the compositional formulation
    float (DirectRegistration::* calcDepthError)(const Eigen::Matrix4f &, const size_t, const float*, const float*);
    if(compositional == FC || compositional == ESM)
        calcDepthError = &DirectRegistration::calcDepthErrorFC;
    else if(compositional == IC)
        calcDepthError = &DirectRegistration::calcDepthErrorIC;

    //Asign the intensity/depth value to the warped image and compute the difference between the transformed
    //pixel of the ref frame and the corresponding pixel of target frame. Compute the error function
    if( !use_bilinear || currPyrLevel !=0 || method == DEPTH_CONSISTENCY || method == DIRECT_ICP ) // Only range registration is always performed with nearest-neighbour
    {
        // Warp the image
        //ProjModel->projectNN(itRef.xyz_tf, itRef.validPixels, itRef.warp_pixels);
        ProjModel_trg->projectNN(itRef.xyz_tf, itRef.validPixels, itRef.warp_pixels);
//        if( method == DIRECT_ICP && currPyrLevel ==4 )
//            cout << "itRef.warp_pixels: " << itRef.warp_pixels.transpose() << endl;

        if(method == PHOTO_DEPTH)
        {
            //cout << " method == PHOTO_DEPTH " << endl;

//            Eigen::VectorXf diff_photo(n_pts); //= Eigen::VectorXf::Zero(n_pts);
//            Eigen::VectorXf diff_depth(n_pts);
    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //cout << i << " itRef.validPixels " << itRef.validPixels(i) << " warp_pixel " << itRef.warp_pixels(i) << endl;
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    //ASSERT_(itRef.validPixels(i) != -1);

                    //cout << i << " itRef.validPixels " << itRef.validPixels(i) << " warp_pixel " << itRef.warp_pixels(i) << endl;
                    ++numVisiblePts;
                    // cout << thres_saliency_gray_ << " Grad " << fabs(trg.grayPyr_GradX[currPyrLevel].at<float>(r_transf,c_transf)) << " " << fabs(trg.grayPyr_GradY[currPyrLevel].at<float>(r_transf,c_transf)) << endl;
                    //if( fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_ || fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_)
                    //if( fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_ || fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_)
                    {
                        itRef.validPixelsPhoto(i) = 1;
                        float diff = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        //diff_photo(i) = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        float residual = diff * stdDevPhoto_inv;
                        itRef.wEstimPhoto(i) = sqrt(weightMEstimator(residual)); // The weight computed by an M-estimator
                        itRef.residualsPhoto(i) = itRef.wEstimPhoto(i) * residual;
                        error2_photo += itRef.residualsPhoto(i) * itRef.residualsPhoto(i);
                        //v_AD_intensity[i] = fabs(diff);
                        //cout << i << " warp_pixel " << itRef.warp_pixels(i) << " weight " << itRef.wEstimPhoto(i) << " error2_photo " << error2_photo << " diff " << diff << endl;
                    }

                    float depth = _depthTrg[itRef.warp_pixels(i)];
                    if(depth > ProjModel_trg->min_depth_) // if(depth > ProjModel->min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        //if( fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_ || fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_)
                        {
                            itRef.validPixelsDepth(i) = 1;
                            itRef.stdDevError_inv(i) = 1 / std::max (stdDevDepth*(depth*depth), stdDevDepth);
                            float diff = (this->*calcDepthError)(poseGuess, i, _depthRef, _depthTrg);
                            //float diff = (this->*calcDepthError)(poseGuess_inv, i, _depthRef, _depthTrg);
                            float residual = diff * itRef.stdDevError_inv(i);
                            itRef.wEstimDepth(i) = sqrt(weightMEstimator(residual));
                            itRef.residualsDepth(i) = itRef.wEstimDepth(i) * residual;
                            error2_depth += itRef.residualsDepth(i) * itRef.residualsDepth(i);
                            // cout << i << " error2_depth " << error2_depth << " weight " << itRef.wEstimDepth(i) << " residual " << residual << " stdDevInv " << itRef.stdDevError_inv(i) << endl;
                        }
                    }
                    //mrpt::system::pause();
                }
                //mrpt::system::pause();
            }
        }
        else if(method == PHOTO_CONSISTENCY) // || method == PHOTO_DEPTH
//        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        {
            //cout << " method == PHOTO_CONSISTENCY " << endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    ++numVisiblePts;
                    // cout << thres_saliency_gray_ << " Grad " << fabs(trg.grayPyr_GradX[currPyrLevel].at<float>(r_transf,c_transf)) << " " << fabs(trg.grayPyr_GradY[currPyrLevel].at<float>(r_transf,c_transf)) << endl;
                    //if( fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_ || fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_)
                    //if( fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_ || fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_)
                    {
                        itRef.validPixelsPhoto(i) = 1;
                        float diff = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        //diff_photo(i) = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        float residual = diff * stdDevPhoto_inv;
                        itRef.wEstimPhoto(i) = sqrt(weightMEstimator(residual)); // The weight computed by an M-estimator
                        itRef.residualsPhoto(i) = itRef.wEstimPhoto(i) * residual;
                        error2_photo += itRef.residualsPhoto(i) * itRef.residualsPhoto(i);
                        //v_AD_intensity[i] = fabs(diff);
                        //cout << i << " warp_pixel " << itRef.warp_pixels(i) << " weight " << itRef.wEstimPhoto(i) << " error2_photo " << error2_photo << " diff " << diff << endl;
                    }
                }
            }
        }
        else if(method == DEPTH_CONSISTENCY) // || method == PHOTO_DEPTH
//        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        {
            //cout << " method == DEPTH_CONSISTENCY " << endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    float depth = _depthTrg[itRef.warp_pixels(i)];
                    if(depth > ProjModel_trg->min_depth_) // if(depth > ProjModel->min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        //if( fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_ || fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_)
                        {
                            ++numVisiblePts;
                            itRef.validPixelsDepth(i) = 1;
                            itRef.stdDevError_inv(i) = 1 / std::max (stdDevDepth*(depth*depth), stdDevDepth);
                            //diff_depth(i) = _depthTrg[itRef.warp_pixels(i)] - ProjModel->getDepth(xyz);
                            float diff = (this->*calcDepthError)(poseGuess, i, _depthRef, _depthTrg);
                            //float diff = (this->*calcDepthError)(poseGuess_inv, i, _depthRef, _depthTrg);
                            float residual = diff * itRef.stdDevError_inv(i);
                            itRef.wEstimDepth(i) = sqrt(weightMEstimator(residual));
                            itRef.residualsDepth(i) = itRef.wEstimDepth(i) * residual;
                            error2_depth += itRef.residualsDepth(i) * itRef.residualsDepth(i);
                        }
                    }
                }
            }
        }
        else if(method == DIRECT_ICP) // Fast ICP implementation: the data association is given by the image warping (still to optimize)
        {
            cout << " computeError DIRECT_ICP \n";
            float thres_max_dist = 0.5f;
    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    float depth = _depthTrg[itRef.warp_pixels(i)];
                    if(depth > ProjModel_trg->min_depth_) // if(depth > ProjModel->min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        //if( fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_ || fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_)
                        {
                            //itRef.stdDevError_inv(i) = 1;
                            itRef.stdDevError_inv(i) = 1 / std::max (stdDevDepth*(depth*depth), stdDevDepth);
//                            Vector3f residual3D = (itTrg.xyz.block(itRef.warp_pixels(i),0,1,3).transpose() - ref.xyz_tf.block(i,0,1,3).transpose());
                            Vector3f residual3D = (itRef.xyz_tf.block(i,0,1,3) - itTrg.xyz.block(itRef.warp_pixels(i),0,1,3)).transpose();
                            float res_norm = residual3D.norm();
                            if(res_norm < thres_max_dist)
                            {
                                ++numVisiblePts;
                                itRef.validPixelsDepth(i) = 1;
                                //float weight2 = weightMEstimator(res_norm);
                                //itRef.wEstimDepth(i) = sqrt(weight2);
                                itRef.wEstimDepth(i) = 1;
                                itRef.residualsDepth.block(3*i,0,3,1) = itRef.wEstimDepth(i) * itRef.stdDevError_inv(i) * residual3D;
                                error2_depth += residual3D .dot (residual3D);
                                // cout << i << " error2_depth " << error2_depth << " weight " << itRef.wEstimDepth(i) << " residual " << residual3D.transpose() << " stdDevInv " << itRef.stdDevError_inv(i) << endl;
                            }
                        }
                    }
                }
            }
        }
    }
    else // Bilinear
    {
        cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION " << endl;
        cout << "poseGuess \n" << poseGuess << endl;

        // Warp the image
        ProjModel_trg->project(itRef.xyz, itRef.warp_img, itRef.warp_pixels);

        if(method == PHOTO_DEPTH)
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                {
                    //ASSERT_(itRef.validPixels(i) != -1);

                    ++numVisiblePts;
                    cv::Point2f warped_pixel(itRef.warp_img(i,0), itRef.warp_img(i,1));
                    // cout << thres_saliency_gray_ << " Grad " << fabs(trg.grayPyr_GradX[currPyrLevel].at<float>(r_transf,c_transf)) << " " << fabs(trg.grayPyr_GradY[currPyrLevel].at<float>(r_transf,c_transf)) << endl;
                    //if( fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_ || fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_)
                    //if( fabs(_grayRefGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_ || fabs(_grayRefGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_)
                    {
                        itRef.validPixelsPhoto(i) = 1;
                        float intensity = ProjModel->bilinearInterp( trg.grayPyr[currPyrLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        float diff = intensity - _grayRef[itRef.validPixels(i)];
                        float residual = diff * stdDevPhoto_inv;
                        //diff_photo(i) = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        itRef.wEstimPhoto(i) = sqrt(weightMEstimator(residual)); // The weight computed by an M-estimator
                        itRef.residualsPhoto(i) = itRef.wEstimPhoto(i) * residual;
                        error2_photo += itRef.residualsPhoto(i) * itRef.residualsPhoto(i);
                        //v_AD_intensity[i] = fabs(diff);
                        //cout << i << " warp_pixel " << itRef.warp_pixels(i) << " weight " << itRef.wEstimPhoto(i) << " error2_photo " << error2_photo << " diff " << diff << endl;
                    }

                    float depth = _depthTrg[(int)(warped_pixel.y) * ProjModel->nCols + (int)(warped_pixel.x)];
//                    float depth = ProjModel->bilinearInterp_depth( trg.grayPyr[currPyrLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                    if(depth > ProjModel_trg->min_depth_) // if(depth > ProjModel->min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        //if( fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_ || fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_)
                        {
                            itRef.validPixelsDepth(i) = 1;
                            itRef.stdDevError_inv(i) = 1 / std::max (stdDevDepth*(depth*depth), stdDevDepth);
                            //diff_depth(i) = _depthTrg[itRef.validPixels(i)] - ProjModel->getDepth(xyz);
                            float diff = (this->*calcDepthError)(poseGuess, i, _depthRef, _depthTrg);
                            //Vector3f xyz_transf = itRef.xyz_tf.block(i,0,1,3).transpose();
                            //cout << "diff " << diff << " " << (depth - ProjModel->getDepth(xyz_transf)) << endl;
                            float residual = diff * itRef.stdDevError_inv(i);
                            itRef.wEstimDepth(i) = sqrt(weightMEstimator(residual));
                            itRef.residualsDepth(i) = itRef.wEstimDepth(i) * residual;
                            error2_depth += itRef.residualsDepth(i) * itRef.residualsDepth(i);
                            // cout << i << " error2_depth " << error2_depth << " weight " << itRef.wEstimDepth(i) << " residual " << residual << " stdDevInv " << itRef.stdDevError_inv(i) << endl;
                        }
                    }
                }
            }
        }
        else if(method == PHOTO_CONSISTENCY)
        {
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                {
                    ++numVisiblePts;
                    cv::Point2f warped_pixel(itRef.warp_img(i,0), itRef.warp_img(i,1));
                    // cout << thres_saliency_gray_ << " Grad " << fabs(trg.grayPyr_GradX[currPyrLevel].at<float>(r_transf,c_transf)) << " " << fabs(trg.grayPyr_GradY[currPyrLevel].at<float>(r_transf,c_transf)) << endl;
                    //if( fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_ || fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_)
                    //if( fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_ || fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_)
                    {
                        itRef.validPixelsPhoto(i) = 1;
                        float intensity = ProjModel->bilinearInterp( trg.grayPyr[currPyrLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        float diff = intensity - _grayRef[itRef.validPixels(i)];
                        float residual = diff * stdDevPhoto_inv;
                        //diff_photo(i) = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        itRef.wEstimPhoto(i) = sqrt(weightMEstimator(residual)); // The weight computed by an M-estimator
                        itRef.residualsPhoto(i) = itRef.wEstimPhoto(i) * residual;
                        error2_photo += itRef.residualsPhoto(i) * itRef.residualsPhoto(i);
                        //v_AD_intensity[i] = fabs(diff);
                        //cout << i << " warp_pixel " << itRef.warp_pixels(i) << " weight " << itRef.wEstimPhoto(i) << " error2_photo " << error2_photo << " diff " << diff << endl;
                    }
                }
            }
        }
    }

    SSO = (float)numVisiblePts / n_pts;
    //        cout << "numVisiblePixels " << numVisiblePixels << " imgSize " << imgSize << " sso " << SSO << endl;

    // Compute the median absulute deviation of the projection of reference image onto the target one to update the value of the standard deviation of the intesity error
//    if(error2_photo > 0 && compute_MAD_stdDev)
//    {
//        cout << " stdDevPhoto PREV " << stdDevPhoto << endl;
//        size_t count_valid_pix = 0;
//        std::vector<float> v_AD_intensity(n_ptsPhoto);
//        for(size_t i=0; i < imgSize; i++)
//            if( itRef.validPixelsPhoto(i) ) //Compute the jacobian only for the valid points
//            {
//                v_AD_intensity[count_valid_pix] = v_AD_intensity_[i];
//                ++count_valid_pix;
//            }
//        //v_AD_intensity.conservativeResize(n_pts);
//        v_AD_intensity.conservativeResize(n_ptsPhoto);
//        float stdDevPhoto_updated = 1.4826 * median(v_AD_intensity);
//        error2_photo *= stdDevPhoto*stdDevPhoto / (stdDevPhoto_updated*stdDevPhoto_updated);
//        stdDevPhoto = stdDevPhoto_updated;
//        cout << " stdDevPhoto_updated    " << stdDevPhoto_updated << endl;
//    }

    error2 = (error2_photo + error2_depth) / numVisiblePts;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << "Level " << currPyrLevel << " computeError took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << endl;
#endif

    return error2;
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
void DirectRegistration::calcHessGrad() //( const costFuncType method )
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t n_pts = itRef.xyz.rows();

    float stdDevPhoto_inv = 1./stdDevPhoto;

    float *_depthRefGradX = reinterpret_cast<float*>(ref.depthPyr_GradX[currPyrLevel].data);
    float *_depthRefGradY = reinterpret_cast<float*>(ref.depthPyr_GradY[currPyrLevel].data);
    float *_grayRefGradX = reinterpret_cast<float*>(ref.grayPyr_GradX[currPyrLevel].data);
    float *_grayRefGradY = reinterpret_cast<float*>(ref.grayPyr_GradY[currPyrLevel].data);

//    float *_depthTrgGradX = reinterpret_cast<float*>(trg.depthPyr_GradX[currPyrLevel].data);
//    float *_depthTrgGradY = reinterpret_cast<float*>(trg.depthPyr_GradY[currPyrLevel].data);
//    float *_grayTrgGradX = reinterpret_cast<float*>(trg.grayPyr_GradX[currPyrLevel].data);
//    float *_grayTrgGradY = reinterpret_cast<float*>(trg.grayPyr_GradY[currPyrLevel].data);

    // Build the aligned versions of the image derivatives, so that we can benefit from SIMD performance
    if(use_salient_pixels)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        {
            _grayRefGradX = &grayGradX_sal(0);
            _grayRefGradY = &grayGradY_sal(0);
        }
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        {
            _depthRefGradX = &depthGradX_sal(0);
            _depthRefGradY = &depthGradY_sal(0);
        }
    }

    // Compute the Jacobians
    if(method == PHOTO_DEPTH)
    {
        ProjModel_trg->computeJacobiansPhotoDepth(itRef.xyz_tf, stdDevPhoto_inv, itRef.stdDevError_inv, itRef.wEstimDepth, itRef.jacobiansPhoto, itRef.jacobiansDepth,
                                                    _depthRefGradX, _depthRefGradY, _grayRefGradX, _grayRefGradY);
    }
    else if(method == PHOTO_CONSISTENCY)
    {
        ProjModel_trg->computeJacobiansPhoto(itRef.xyz_tf, stdDevPhoto_inv, itRef.wEstimPhoto, itRef.jacobiansPhoto, _grayRefGradX, _grayRefGradY);
    }
    else if(method == DEPTH_CONSISTENCY)
    {
        ProjModel_trg->computeJacobiansDepth(itRef.xyz_tf, itRef.stdDevError_inv, itRef.wEstimDepth, itRef.jacobiansDepth, _depthRefGradX, _depthRefGradY);
    }
    else if(method == DIRECT_ICP)
    {
        ProjModel_trg->computeJacobiansICP(itRef.xyz_tf, itRef.stdDevError_inv, itRef.wEstimDepth, itRef.jacobiansDepth); //, _depthRefGradX, _depthRefGradY);
    }

    if(visualize)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        {
            warped_gray = cv::Mat::zeros(ref.grayPyr[currPyrLevel].rows, ref.grayPyr[currPyrLevel].cols, ref.grayPyr[currPyrLevel].type());
            if(!use_bilinear)
            {
                for(size_t i=0; i < n_pts; i++)
                    if(itRef.warp_pixels(i) != -1)
                    {
                        //cout << i << " itRef.warp_pixels(i) " << itRef.warp_pixels(i) << endl;
                        warped_gray.at<float>(itRef.validPixels(i)) = trg.grayPyr[currPyrLevel].at<float>(itRef.warp_pixels(i));
                    }
            }
            else
            {
                for(size_t i=0; i < n_pts; i++)
                    if(itRef.warp_pixels(i) != -1 && itRef.validPixels(i) != -1)
                    {
                        cv::Point2f warped_pixel(itRef.warp_img(i,0), itRef.warp_img(i,1));
                        warped_gray.at<float>(itRef.validPixels(i)) = ProjModel->bilinearInterp( trg.grayPyr[currPyrLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                    }
            }
        }
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        {
            warped_depth = cv::Mat::zeros(ref.depthPyr[currPyrLevel].rows, ref.depthPyr[currPyrLevel].cols, ref.depthPyr[currPyrLevel].type());
            for(size_t i=0; i < n_pts; i++)
                if(itRef.warp_pixels(i) != -1)
                {
                    Vector3f xyz_trans = itRef.xyz_tf.block(i,0,1,3).transpose();
                    warped_depth.at<float>(itRef.warp_pixels(i)) = ProjModel->getDepth(xyz_trans);
                }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(itRef.jacobiansPhoto, itRef.residualsPhoto, itRef.validPixelsPhoto);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(itRef.jacobiansDepth, itRef.residualsDepth, itRef.validPixelsDepth);
    if(method == DIRECT_ICP)
        updateHessianAndGradient3D(itRef.jacobiansDepth, itRef.residualsDepth, itRef.validPixelsDepth);
    //cout << "hessian \n" << hessian << endl;
    //cout << "gradient \n" << gradient.transpose() << endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << currPyrLevel << " pyr calcHessGrad took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
double DirectRegistration::calcHessGradError_IC(const Eigen::Matrix4f & poseGuess) //( const costFuncType method )
{
    //cout << " DirectRegistration::calcHessGradError_IC \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t n_pts = itRef.xyz.rows();
    transformPts3D(itRef.xyz, poseGuess, itRef.xyz_tf);
    MatrixXf poseGuess_inv = poseGuess.inverse();

    float stdDevPhoto_inv = 1./stdDevPhoto;
    itRef.residualsPhoto = VectorXf::Zero(n_pts);
    itRef.jacobiansPhoto = MatrixXf::Zero(n_pts,6);
    if(method == DIRECT_ICP)
    {
        itRef.residualsDepth = VectorXf::Zero(3*n_pts);
        itRef.jacobiansDepth = MatrixXf::Zero(3*n_pts,6);
    }
    else
    {
        itRef.residualsDepth = VectorXf::Zero(n_pts);
        itRef.jacobiansDepth = MatrixXf::Zero(n_pts,6);
    }
    itRef.stdDevError_inv = VectorXf::Zero(n_pts);
    itRef.wEstimPhoto = VectorXf::Zero(n_pts);
    itRef.wEstimDepth = VectorXf::Zero(n_pts);
    itRef.validPixelsPhoto = VectorXi::Zero(n_pts);
    itRef.validPixelsDepth = VectorXi::Zero(n_pts);

//    _itRef.residualsPhoto = VectorXf::Zero(imgSize);
//    _itRef.residualsDepth = VectorXf::Zero(imgSize);
//    _itRef.stdDevError_inv = VectorXf::Zero(imgSize);
//    _itRef.wEstimPhoto = VectorXf::Zero(imgSize);
//    _itRef.wEstimDepth = VectorXf::Zero(imgSize);
//    _itRef.validPixelsPhoto = VectorXi::Zero(imgSize);
//    _itRef.validPixelsDepth = VectorXi::Zero(imgSize);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    //std::vector<float> v_AD_intensity(imgSize);

    const float *_depthRef = reinterpret_cast<float*>(ref.depthPyr[currPyrLevel].data);
    const float *_depthTrg = reinterpret_cast<float*>(trg.depthPyr[currPyrLevel].data);
    float *_grayRef = reinterpret_cast<float*>(ref.grayPyr[currPyrLevel].data);
    float *_grayTrg = reinterpret_cast<float*>(trg.grayPyr[currPyrLevel].data);

    float *_depthRefGradX = reinterpret_cast<float*>(ref.depthPyr_GradX[currPyrLevel].data);
    float *_depthRefGradY = reinterpret_cast<float*>(ref.depthPyr_GradY[currPyrLevel].data);
    float *_grayRefGradX = reinterpret_cast<float*>(ref.grayPyr_GradX[currPyrLevel].data);
    float *_grayRefGradY = reinterpret_cast<float*>(ref.grayPyr_GradY[currPyrLevel].data);

    //cout << " use_salient_pixels " << use_salient_pixels << " use_bilinear " << use_bilinear << " pts " << itRef.xyz.rows()  << endl;

    //Asign the intensity/depth value to the warped image and compute the difference between the transformed
    //pixel of the ref frame and the corresponding pixel of target frame. Compute the error function
    if( !use_bilinear || currPyrLevel !=0 || method == DEPTH_CONSISTENCY || method == DIRECT_ICP ) // Only range registration is always performed with nearest-neighbour
    {
        // Warp the image
        ProjModel_trg->projectNN(itRef.xyz_tf, itRef.validPixels, itRef.warp_pixels);
//        if( method == DIRECT_ICP && currPyrLevel ==4 )
//            cout << "itRef.warp_pixels: " << itRef.warp_pixels.transpose() << endl;

        if(method == PHOTO_DEPTH)
        {
            //cout << " method == PHOTO_DEPTH " << endl;

//            Eigen::VectorXf diff_photo(n_pts); //= Eigen::VectorXf::Zero(n_pts);
//            Eigen::VectorXf diff_depth(n_pts);
    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //cout << i << " itRef.validPixels " << itRef.validPixels(i) << " warp_pixel " << itRef.warp_pixels(i) << endl;
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    //ASSERT_(itRef.validPixels(i) != -1);

                    //cout << i << " itRef.validPixels " << itRef.validPixels(i) << " warp_pixel " << itRef.warp_pixels(i) << endl;
                    ++numVisiblePts;

                    Matrix<float,1,3> xyz_ref_ = itRef.xyz.block(i,0,1,3);
                    Vector3f xyz_ref = xyz_ref_.transpose();
                    Matrix<float,2,6> jacobianWarpRt;
                    ProjModel_trg->computeJacobian26_wT(xyz_ref, jacobianWarpRt);

                    // cout << thres_saliency_gray_ << " Grad " << fabs(trg.grayPyr_GradX[currPyrLevel].at<float>(r_transf,c_transf)) << " " << fabs(trg.grayPyr_GradY[currPyrLevel].at<float>(r_transf,c_transf)) << endl;
                    //if( fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_ || fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_)
                    //if( fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_ || fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_)
                    {
                        itRef.validPixelsPhoto(i) = 1;
                        float diff = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        //diff_photo(i) = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        float residual = diff * stdDevPhoto_inv;
                        itRef.wEstimPhoto(i) = sqrt(weightMEstimator(residual)); // The weight computed by an M-estimator
                        itRef.residualsPhoto(i) = itRef.wEstimPhoto(i) * residual;
                        error2_photo += itRef.residualsPhoto(i) * itRef.residualsPhoto(i);

                        Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        img_gradient(0,0) = _grayRefGradX[itRef.validPixels(i)];
                        img_gradient(0,1) = _grayRefGradY[itRef.validPixels(i)];
                        // (NOTICE that that the sign of this jacobian has been changed)
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        itRef.jacobiansPhoto.block(i,0,1,6) = ((itRef.wEstimPhoto(i)*stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        //cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << itRef.jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << itRef.residualsPhoto(i) << endl;
                        //mrpt::system::pause();

                        //v_AD_intensity[i] = fabs(diff);
                        //cout << i << " warp_pixel " << itRef.warp_pixels(i) << " weight " << itRef.wEstimPhoto(i) << " error2_photo " << error2_photo << " diff " << diff << endl;
                    }

                    float depth = _depthTrg[itRef.warp_pixels(i)];
                    if(depth > ProjModel_trg->min_depth_) // if(depth > ProjModel->min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        //if( fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_ || fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_)
                        {
                            itRef.validPixelsDepth(i) = 1;
                            itRef.stdDevError_inv(i) = 1 / std::max (stdDevDepth*(depth*depth), stdDevDepth);
                            //diff_depth(i) = _depthTrg[itRef.warp_pixels(i)] - ProjModel->getDepth(xyz_ref);
                            float diff = calcDepthErrorIC(poseGuess, i, _depthRef, _depthTrg);
                            //float diff = calcDepthErrorIC(poseGuess_inv, i, _depthRef, _depthTrg);
                            float residual = diff * itRef.stdDevError_inv(i);
                            itRef.wEstimDepth(i) = sqrt(weightMEstimator(residual));
                            itRef.residualsDepth(i) = itRef.wEstimDepth(i) * residual;
                            error2_depth += itRef.residualsDepth(i) * itRef.residualsDepth(i);

                            Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            depth_gradient(0,0) = _depthRefGradX[itRef.validPixels(i)];
                            depth_gradient(0,1) = _depthRefGradY[itRef.validPixels(i)];
                            // cout << "depth_gradient \n " << depth_gradient << endl;
                            //itRef.jacobiansDepth.block(i,0,1,6) = ((itRef.wEstimDepth(i)*itRef.stdDevError_inv(i)) * depth_gradient) * jacobianWarpRt;

                            Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
//                            Eigen::Matrix<float,1,3> xyz_trg2ref_rot = itTrg.xyz.block(itRef.warp_pixels(i),0,1,3) - poseGuess.block(0,3,3,1).transpose();
//                            jacobian16_depthT.block(0,0,1,3) = xyz_trg2ref_rot / xyz_trg2ref_rot.norm();
//                            Vector3f jacDepthProj_trans = jacobian16_depthT.block(0,0,1,3).transpose();
//                            Vector3f translation = poseGuess.block(0,3,3,1);
//                            jacobian16_depthT.block(0,3,1,3) = translation. cross (jacDepthProj_trans);

                            jacobian16_depthT.block(0,0,1,3) = (1 / _depthRef[itRef.validPixels(i)]) * xyz_ref_;
                            Vector3f xyz_trg2ref_rot = itTrg.xyz.block(itRef.warp_pixels(i),0,1,3).transpose() - poseGuess.block(0,3,3,1);
                            //Vector3f jacDepthProj_rot = xyz_trg2ref_rot. cross (jacDepthProj_trans);
                            //jacobian16_depthT.block(0,3,1,3) = jacDepthProj_rot.transpose();
                            Vector3f jacDepthProj_trans = jacobian16_depthT.block(0,0,1,3).transpose();
                            jacobian16_depthT.block(0,3,1,3) = xyz_trg2ref_rot.transpose() * skew(jacDepthProj_trans);
                            //itRef.jacobiansDepth.block(i,0,1,6) = itRef.stdDevError_inv(i)*(depth_gradient * jacobianWarpRt - jacobian16_depthT);
                            itRef.jacobiansDepth.block(i,0,1,6) = (itRef.wEstimDepth(i)*itRef.stdDevError_inv(i)) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);

//                             cout << i << " error2_depth " << error2_depth << " weight " << itRef.wEstimDepth(i) << " residual " << residual << " stdDevInv " << itRef.stdDevError_inv(i) << endl;
//                              cout << "itRef.jacobiansDepth: " << itRef.jacobiansDepth.block(i,0,1,6) << endl;
//                             mrpt::system::pause();
                        }
                    }
                    //mrpt::system::pause();
                }
                //mrpt::system::pause();
            }
        }
        else if(method == PHOTO_CONSISTENCY) // || method == PHOTO_DEPTH
//        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        {
            //cout << " method == PHOTO_CONSISTENCY " << endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    ++numVisiblePts;
                    // cout << thres_saliency_gray_ << " Grad " << fabs(trg.grayPyr_GradX[currPyrLevel].at<float>(r_transf,c_transf)) << " " << fabs(trg.grayPyr_GradY[currPyrLevel].at<float>(r_transf,c_transf)) << endl;
                    //if( fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_ || fabs(_grayTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_gray_)
                    //if( fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_ || fabs(_grayRefGrad[itRef.validPixels(i)]) > thres_saliency_gray_)
                    {
                        itRef.validPixelsPhoto(i) = 1;
                        float diff = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        //diff_photo(i) = _grayTrg[itRef.warp_pixels(i)] - _grayRef[itRef.validPixels(i)];
                        float residual = diff * stdDevPhoto_inv;
                        itRef.wEstimPhoto(i) = sqrt(weightMEstimator(residual)); // The weight computed by an M-estimator
                        itRef.residualsPhoto(i) = itRef.wEstimPhoto(i) * residual;
                        error2_photo += itRef.residualsPhoto(i) * itRef.residualsPhoto(i);

                        //Vector3f xyz = ref.xyz_tf.block(i,0,1,3).transpose();
                        Vector3f xyz_ref = itRef.xyz.block(i,0,1,3).transpose();
                        Matrix<float,2,6> jacobianWarpRt;
                        ProjModel_trg->computeJacobian26_wT(xyz_ref, jacobianWarpRt);
                        Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        img_gradient(0,0) = _grayRefGradX[itRef.validPixels(i)];
                        img_gradient(0,1) = _grayRefGradY[itRef.validPixels(i)];
                        // (NOTICE that that the sign of this jacobian has been changed)
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        itRef.jacobiansPhoto.block(i,0,1,6) = ((itRef.wEstimPhoto(i)*stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        //cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << itRef.jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << itRef.residualsPhoto(i) << endl;
                        //mrpt::system::pause();

                        //v_AD_intensity[i] = fabs(diff);
                        //cout << i << " warp_pixel " << itRef.warp_pixels(i) << " weight " << itRef.wEstimPhoto(i) << " error2_photo " << error2_photo << " diff " << diff << endl;
                    }
                }
            }
        }
        else if(method == DEPTH_CONSISTENCY) // || method == PHOTO_DEPTH
//        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        {
            //cout << " method == DEPTH_CONSISTENCY " << endl;
//            for(size_t i=0; i < n_pts; i++)
//                cout << itRef.validPixels(i) << " depth_gradient " << _depthRefGradX[itRef.validPixels(i)] << " " << _depthRefGradY[itRef.validPixels(i)] << endl;

#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    float depth_ref = _depthRef[itRef.validPixels(i)];
                    float depth = _depthTrg[itRef.warp_pixels(i)];
                    if(depth > ProjModel->min_depth_) // if(depth > ProjModel->min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        //if( fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_ || fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_)
                        {
                            ++numVisiblePts;
                            itRef.validPixelsDepth(i) = 1;
                            itRef.stdDevError_inv(i) = 1 / std::max (stdDevDepth*(depth*depth), stdDevDepth);
                            //diff_depth(i) = _depthTrg[itRef.warp_pixels(i)] - ProjModel->getDepth(xyz);
                            float diff = calcDepthErrorIC(poseGuess, i, _depthRef, _depthTrg);
                            //float diff = calcDepthErrorIC(poseGuess_inv, i, _depthRef, _depthTrg);
                            float residual = diff * itRef.stdDevError_inv(i);
                            itRef.wEstimDepth(i) = sqrt(weightMEstimator(residual));
                            itRef.residualsDepth(i) = itRef.wEstimDepth(i) * residual;
                            error2_depth += itRef.residualsDepth(i) * itRef.residualsDepth(i);

                            Matrix<float,1,3> xyz_ref_ = itRef.xyz.block(i,0,1,3);
                            Vector3f xyz_ref = xyz_ref_.transpose();
                            Matrix<float,2,6> jacobianWarpRt;
                            ProjModel_trg->computeJacobian26_wT(xyz_ref, jacobianWarpRt);
                            Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            depth_gradient(0,0) = _depthRefGradX[itRef.validPixels(i)];
                            depth_gradient(0,1) = _depthRefGradY[itRef.validPixels(i)];
                            //itRef.jacobiansDepth.block(i,0,1,6) = ((itRef.wEstimDepth(i)*itRef.stdDevError_inv(i)) * depth_gradient) * jacobianWarpRt;

                            Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
//                            Eigen::Matrix<float,1,3> xyz_trg2ref_rot = itTrg.xyz.block(itRef.warp_pixels(i),0,1,3) - poseGuess.block(0,3,3,1).transpose();
//                            jacobian16_depthT.block(0,0,1,3) = xyz_trg2ref_rot / xyz_trg2ref_rot.norm();
//                            Vector3f jacDepthProj_trans = jacobian16_depthT.block(0,0,1,3).transpose();
//                            Vector3f translation = poseGuess.block(0,3,3,1);
//                            jacobian16_depthT.block(0,3,1,3) = translation. cross (jacDepthProj_trans);

                            jacobian16_depthT.block(0,0,1,3) = (1 / _depthRef[itRef.validPixels(i)]) * xyz_ref_;
                            Vector3f xyz_trg2ref_rot = itTrg.xyz.block(itRef.warp_pixels(i),0,1,3).transpose() - poseGuess.block(0,3,3,1);
                            //Vector3f jacDepthProj_rot = xyz_trg2ref_rot. cross (jacDepthProj_trans);
                            //jacobian16_depthT.block(0,3,1,3) = jacDepthProj_rot.transpose();
                            Vector3f jacDepthProj_trans = jacobian16_depthT.block(0,0,1,3).transpose();
                            jacobian16_depthT.block(0,3,1,3) = xyz_trg2ref_rot.transpose() * skew(jacDepthProj_trans);
                            //itRef.jacobiansDepth.block(i,0,1,6) = itRef.stdDevError_inv(i)*(depth_gradient * jacobianWarpRt - jacobian16_depthT);
                            itRef.jacobiansDepth.block(i,0,1,6) = (itRef.wEstimDepth(i)*itRef.stdDevError_inv(i)) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);

//                            cout << i << " error2_depth " << error2_depth << " diff " << diff << " weight " << itRef.wEstimDepth(i) << " residual " << residual << " stdDevInv " << itRef.stdDevError_inv(i) << endl;
//                            cout << "itRef.validPixels(i) " << itRef.validPixels(i) << endl;
//                            cout << "itRef.jacobiansDepth: " << itRef.jacobiansDepth.block(i,0,1,6) << endl;
//                            cout << "depth_gradient \n " << depth_gradient << endl;
//                            cout << "jacobianWarpRt \n " << jacobianWarpRt << endl;
//                            cout << "jacobian16_depthT \n " << jacobian16_depthT << endl;
//                            mrpt::system::pause();
                        }
                    }
                    else
                        itRef.warp_pixels(i) = -1;
                }
            }
        }
        else if(method == DIRECT_ICP) // Fast ICP implementation: the data association is given by the image warping (still to optimize)
        {
            cout << " computeError DIRECT_ICP \n";
            float thres_max_dist = 0.5f;
    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( itRef.validPixels(i) != -1 && itRef.warp_pixels(i) != -1 )
                if( itRef.warp_pixels(i) != -1 )
                {
                    float depth = _depthTrg[itRef.warp_pixels(i)];
                    if(depth > ProjModel_trg->min_depth_) // if(depth > ProjModel->min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        //if( fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_ || fabs(_depthTrgGrad[itRef.warp_pixels(i)]) > thres_saliency_depth_)
                        {
                            //itRef.stdDevError_inv(i) = 1;
                            itRef.stdDevError_inv(i) = 1 / std::max (stdDevDepth*(depth*depth), stdDevDepth);
                            Vector3f xyz_ref = itRef.xyz.block(i,0,1,3).transpose();
                            Vector3f xyz_trg = itTrg.xyz.block(itRef.warp_pixels(i),0,1,3).transpose();
                            Eigen::Vector3f xyz_trg2ref = poseGuess_inv.block(0,0,3,3) * xyz_trg + poseGuess_inv.block(0,3,3,1);
                            Vector3f residual3D = xyz_ref - xyz_trg2ref;
                            float res_norm = residual3D.norm();
                            if(res_norm < thres_max_dist)
                            {
                                ++numVisiblePts;
                                itRef.validPixelsDepth(i) = 1;
                                float weight2 = 1;
                                //float weight2 = weightMEstimator(res_norm);
                                itRef.wEstimDepth(i) = sqrt(weight2);
                                residual3D = itRef.wEstimDepth(i) * itRef.stdDevError_inv(i) * residual3D;
                                itRef.residualsDepth.block(3*i,0,3,1) = residual3D;
                                error2_depth += residual3D .dot (residual3D);

                                Matrix<float,3,6> jacobianRt, jacobianRt_inv;
                                ProjModel->computeJacobian36_xT_p(xyz_ref, jacobianRt);
                                ProjModel->computeJacobian36_xT_p(xyz_trg2ref, jacobianRt_inv);
                                itRef.jacobiansDepth.block(3*i,0,1,6) = (itRef.wEstimDepth(i) * itRef.stdDevError_inv(i)) * jacobianRt + jacobianRt_inv;
                                // cout << i << " error2_depth " << error2_depth << " weight " << itRef.wEstimDepth(i) << " residual " << residual3D.transpose() << " stdDevInv " << itRef.stdDevError_inv(i) << endl;
                            }
                        }
                    }
                }
            }
        }
    }
    else // Bilinear
    {
        ASSERT_(0);

    }

    hessian.setZero();
    gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(itRef.jacobiansPhoto, itRef.residualsPhoto, itRef.wEstimPhoto, itRef.validPixelsPhoto);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(itRef.jacobiansDepth, itRef.residualsDepth, itRef.wEstimDepth, itRef.validPixelsDepth);

    SSO = (float)numVisiblePts / n_pts;
    //        cout << "numVisiblePixels " << numVisiblePixels << " imgSize " << imgSize << " sso " << SSO << endl;

    // Compute the median absulute deviation of the projection of reference image onto the target one to update the value of the standard deviation of the intesity error
//    if(error2_photo > 0 && compute_MAD_stdDev)
//    {
//        cout << " stdDevPhoto PREV " << stdDevPhoto << endl;
//        size_t count_valid_pix = 0;
//        std::vector<float> v_AD_intensity(n_ptsPhoto);
//        for(size_t i=0; i < imgSize; i++)
//            if( itRef.validPixelsPhoto(i) ) //Compute the jacobian only for the valid points
//            {
//                v_AD_intensity[count_valid_pix] = v_AD_intensity_[i];
//                ++count_valid_pix;
//            }
//        //v_AD_intensity.conservativeResize(n_pts);
//        v_AD_intensity.conservativeResize(n_ptsPhoto);
//        float stdDevPhoto_updated = 1.4826 * median(v_AD_intensity);
//        error2_photo *= stdDevPhoto*stdDevPhoto / (stdDevPhoto_updated*stdDevPhoto_updated);
//        stdDevPhoto = stdDevPhoto_updated;
//        cout << " stdDevPhoto_updated    " << stdDevPhoto_updated << endl;
//    }

    error2 = (error2_photo + error2_depth) / numVisiblePts;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << "Level " << currPyrLevel << " calcHessGradError_IC took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << endl;
#endif

    return error2;
}

///*! Compute the median absulute deviation of the projection of reference image onto the target one */
//float computeMAD(currPyrLevel)
//{
//}

void DirectRegistration::initWindows()
{
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
    {
        win_name_photo_diff = "Intensity Diff";
        cv::namedWindow(win_name_photo_diff, cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::moveWindow(win_name_photo_diff, 10, 50);
    }
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
    {
        win_name_depth_diff = "Depth Diff";
        cv::namedWindow(win_name_depth_diff, cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::moveWindow(win_name_depth_diff, 610, 50);
    }
}

void DirectRegistration::showImgDiff()
{
    //cout << "DirectRegistration::showImgDiff ...\n";
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
    {
        cv::Mat imgDiff = cv::Mat::zeros(trg.grayPyr[currPyrLevel].rows, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].type());
        //cv::absdiff(trg.grayPyr[currPyrLevel], warped_gray, imgDiff);
        cv::absdiff(ref.grayPyr[currPyrLevel], warped_gray, imgDiff);
        // cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << trg.grayPyr[currPyrLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << endl;
        // cout << "type " << trg.grayPyr[currPyrLevel].type() << " " << warped_gray.type() << endl;

        //cv::imshow("orig", trg.grayPyr[currPyrLevel]);
        //cv::imshow("ref", ref.grayPyr[currPyrLevel]);
        cv::imshow(win_name_photo_diff, imgDiff);
        //cv::imshow("warp", warped_gray);

//                    // Save Abs Diff image
//                    cv::Mat imgDiff_show, img_warped;
//                    imgDiff.convertTo(imgDiff_show, CV_8UC1, 255);
//                    warped_gray.convertTo(img_warped, CV_8UC1, 255);
//                    cv::imwrite(mrpt::format("/home/efernand/tmp/pyr_intensity_AD_%d_%d.png", currPyrLevel, num_iters[currPyrLevel]), imgDiff_show);
//                    cv::imwrite(mrpt::format("/home/efernand/tmp/warped_intensity_%d_%d.png", currPyrLevel, num_iters[currPyrLevel]), img_warped);

//                    cv::Mat DispImage = cv::Mat(2*trg.grayPyr[currPyrLevel].rows+4, 2*trg.grayPyr[currPyrLevel].cols+4, trg.grayPyr[currPyrLevel].type(), cv::Scalar(255)); // cv::Scalar(100, 100, 200)
//                    trg.grayPyr[currPyrLevel].copyTo(DispImage(cv::Rect(0, 0, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    ref.grayPyr[currPyrLevel].copyTo(DispImage(cv::Rect(trg.grayPyr[currPyrLevel].cols+4, 0, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    warped_gray.copyTo(DispImage(cv::Rect(0, trg.grayPyr[currPyrLevel].rows+4, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    imgDiff.copyTo(DispImage(cv::Rect(trg.grayPyr[currPyrLevel].cols+4, trg.grayPyr[currPyrLevel].rows+4, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    //cv::namedWindow("Photoconsistency", cv::WINDOW_AUTOSIZE );// Create a window for display.
//                    cv::imshow("Photoconsistency", DispImage);
    }
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
    {
        //cout << "sizes " << nRows << " " << nCols << " " << "sizes " << trg.depthPyr[currPyrLevel].rows << " " << trg.depthPyr[currPyrLevel].cols << " " << "sizes " << warped_depth.rows << " " << warped_depth.cols << " " << trg.grayPyr[currPyrLevel].type() << endl;
        cv::Mat depthDiff = cv::Mat::zeros(trg.depthPyr[currPyrLevel].rows, trg.depthPyr[currPyrLevel].cols, trg.depthPyr[currPyrLevel].type());
        cv::absdiff(trg.depthPyr[currPyrLevel], warped_depth, depthDiff);
        cv::imshow(win_name_depth_diff, depthDiff);

//                    // Save Abs Diff image
//                    cv::Mat img_show;
//                    const float viz_factor_meters = 82.5;
//                    depthDiff.convertTo(img_show, CV_8U, viz_factor_meters);
//                    cv::imwrite(mrpt::format("/home/efernand/tmp/pyr_depth_AD_%d_%d.png", currPyrLevel, num_iters[currPyrLevel]), img_show);

//                    cv::Mat DispImage = cv::Mat(2*trg.grayPyr[currPyrLevel].rows+4, 2*trg.grayPyr[currPyrLevel].cols+4, trg.grayPyr[currPyrLevel].type(), cv::Scalar(255)); // cv::Scalar(100, 100, 200)
//                    trg.depthPyr[currPyrLevel].copyTo(DispImage(cv::Rect(0, 0, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    ref.depthPyr[currPyrLevel].copyTo(DispImage(cv::Rect(trg.grayPyr[currPyrLevel].cols+4, 0, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    warped_depth.copyTo(DispImage(cv::Rect(0, trg.grayPyr[currPyrLevel].rows+4, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    weightedError.copyTo(DispImage(cv::Rect(trg.grayPyr[currPyrLevel].cols+4, trg.grayPyr[currPyrLevel].rows+4, trg.grayPyr[currPyrLevel].cols, trg.grayPyr[currPyrLevel].rows)));
//                    DispImage.convertTo(DispImage, CV_8U, 22.5);

//                    //cv::namedWindow("Depth-consistency", cv::WINDOW_AUTOSIZE );// Create a window for display.
//                    cv::imshow("Depth-consistency", DispImage);
    }

    cout << "visualize\n";
    cv::waitKey(0);
}

void DirectRegistration::closeWindows()
{
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        cv::destroyWindow(win_name_photo_diff);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        cv::destroyWindow(win_name_depth_diff);
}

void DirectRegistration::extractGradSalient(Eigen::VectorXi & valid_pixels)
{
    const size_t n_pts = valid_pixels.rows();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
    {
        float *_grayRefGradX = reinterpret_cast<float*>(ref.grayPyr_GradX[currPyrLevel].data);
        float *_grayRefGradY = reinterpret_cast<float*>(ref.grayPyr_GradY[currPyrLevel].data);
        grayGradX_sal.resize(n_pts);
        grayGradY_sal.resize(n_pts);
        for(size_t i=0; i < n_pts; i++)
        {
            grayGradX_sal(i) = _grayRefGradX[valid_pixels(i)];
            grayGradY_sal(i) = _grayRefGradY[valid_pixels(i)];
        }
    }
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
    {
        float *_depthRefGradX = reinterpret_cast<float*>(ref.depthPyr_GradX[currPyrLevel].data);
        float *_depthRefGradY = reinterpret_cast<float*>(ref.depthPyr_GradY[currPyrLevel].data);
        depthGradX_sal.resize(n_pts);
        depthGradY_sal.resize(n_pts);
        for(size_t i=0; i < n_pts; i++)
        {
            depthGradX_sal(i) = _depthRefGradX[valid_pixels(i)];
            depthGradY_sal(i) = _depthRefGradY[valid_pixels(i)];
        }
    }
}

/*! Only for the sensor RGBD360. Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera */
void DirectRegistration::maskJoints_RGBD360(const int fringeFraction)
{
    cout << " DirectRegistration::maskJoints_RGBD360 ... \n";

    int minBlackFringe = 2;
    int width_sensor = ProjModel->nCols / NUM_ASUS_SENSORS;
    int nPixBlackFringe = max(minBlackFringe, width_sensor/fringeFraction);
    for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
    {
        //cout << "region_of_interest " << int((0.5f+sensor_id)*width_sensor)-nPixBlackFringe/2 << " " << 0 << " " << nPixBlackFringe << " " << ProjModel->nRows << endl;
        //cv::Rect region_of_interest = cv::Rect(int((0.5f+sensor_id)*width_sensor)-nPixBlackFringe/2, 0, nPixBlackFringe, ProjModel->nRows);
        cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackFringe/2, 0, nPixBlackFringe, ProjModel->nRows);
        ref.depthPyr[currPyrLevel](region_of_interest) = cv::Mat::zeros(ProjModel->nRows,nPixBlackFringe, ref.depthPyr[currPyrLevel].type());
        //trg.depthPyr[currPyrLevel](region_of_interest) = cv::Mat::zeros(ProjModel->nRows,nPixBlackFringe, ref.depthPyr[currPyrLevel].type());
    }
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 *  This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 *  between the ref and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 */
void DirectRegistration::doRegistration(const Matrix4f pose_guess, const costFuncType method) //, const int occlusion )
{
#if PRINT_PROFILING
    cout << " DirectRegistration::doRegistration ... \n";
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    setCostFunction(method);

    if(visualize)
        initWindows();

    num_iters.resize(nPyrLevels+1); // Store the number of iterations
    std::fill(num_iters.begin(), num_iters.end(), 0);
    Matrix4f pose_estim_temp, pose_estim = pose_guess;
    for(currPyrLevel = nPyrLevels; currPyrLevel >= 0; currPyrLevel--)
    {
        // Set the camera calibration parameters
        //ProjModel->scaleCameraParams(ref.depthPyr, currPyrLevel);
        ProjModel_ref->scaleCameraParams(ref.depthPyr, currPyrLevel);
        ProjModel_trg->scaleCameraParams(trg.depthPyr, currPyrLevel);
        //ProjModel_trg->scaleCameraParams(trg.grayPyr, currPyrLevel);

        //if(sensor_type == RGBD360_INDOOR)
        if(sensor_type_ref == STEREO_OUTDOOR)
            maskJoints_RGBD360();

        // Make LUT to store the values of the 3D points of the ref image
        if(use_salient_pixels)
        {
            ProjModel_ref->reconstruct3D_saliency( ref.depthPyr[currPyrLevel], itRef.xyz, itRef.validPixels, (int)method,
                                                   ref.depthPyr_GradX[currPyrLevel], ref.depthPyr_GradY[currPyrLevel], _max_depth_grad, thres_saliency_depth_,
                                                   ref.grayPyr[currPyrLevel], ref.grayPyr_GradX[currPyrLevel], ref.grayPyr_GradY[currPyrLevel], thres_saliency_gray_);

            extractGradSalient(itRef.validPixels);
        }
        else
        {
            ProjModel_ref->reconstruct3D(ref.depthPyr[currPyrLevel], itRef.xyz, itRef.validPixels);
        }

        if(method == DIRECT_ICP || (compositional == IC && (method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH) ) )
            ProjModel_trg->reconstruct3D(trg.depthPyr[currPyrLevel], itTrg.xyz, itTrg.validPixels);

        double error;
        if(compositional != IC)
            error = computeError(pose_estim);
        else
            error = calcHessGradError_IC(pose_estim);

        double diff_error = error;
        Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        while(num_iters[currPyrLevel] < max_iters_ && update_pose.norm() > tol_update_ && diff_error > tol_residual_) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
            cv::TickMeter tm; tm.start();

            //cout << "calcHessianAndGradient_sphere " << endl;
            if(compositional != IC)
            {
                hessian.setZero();
                gradient.setZero();
                calcHessGrad();
            }
            else
            {
                if(num_iters[currPyrLevel] > 0)
                {
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        //updateHessianAndGradient(itRef.jacobiansPhoto, itRef.residualsPhoto, itRef.wEstimPhoto, itRef.validPixelsPhoto);
                        updateGrad(itRef.jacobiansPhoto, itRef.residualsPhoto, itRef.validPixelsPhoto);
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //updateHessianAndGradient(itRef.jacobiansDepth, itRef.residualsDepth, itRef.wEstimDepth, itRef.validPixelsDepth);
                        updateGrad(itRef.jacobiansDepth, itRef.residualsDepth, itRef.validPixelsDepth);
                    }
                    if(method == DIRECT_ICP)
                        updateHessianAndGradient3D(itRef.jacobiansDepth, itRef.residualsDepth, itRef.validPixelsDepth);
                    //updateGrad3D(itRef.jacobiansDepth, itRef.residualsDepth, itRef.validPixelsDepth);
                }
            }
            //cout << "hessian \n" << hessian.transpose() << endl << "gradient \n" << gradient.transpose() << endl;

            //                assert(hessian.rank() == 6); // Make sure that the problem is observable
            if( hessian.rank() != 6 )
            //if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                cout << "\t The problem is ILL-POSED \n";
                cout << "hessian \n" << hessian << endl;
                cout << "gradient \n" << gradient.transpose() << endl;
                //cout << "itRef.jacobiansDepth \n" << itRef.jacobiansDepth << endl;
//                cout << "itRef.residualsDepth: " << itRef.residualsDepth.transpose() << endl;
//                cout << "itRef.wEstimDepth: " << itRef.wEstimDepth.transpose() << endl;
//                cout << "itRef.validPixelsDepth: " << itRef.validPixelsDepth.transpose() << endl;
                registered_pose = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -hessian.inverse() * gradient;
            // update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Matrix<double,6,1> update_pose_d = update_pose.cast<double>();

            if(compositional != IC)
                pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
            else
            {
                pose_estim_temp = pose_estim * mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>();
                cout << "pose_estim_temp \n" << pose_estim_temp << endl;
            }
            //cout << "pose_estim_temp \n" << pose_estim_temp << endl;

            double new_error = computeError(pose_estim_temp);
            diff_error = error - new_error;
            if(diff_error > 0) //  > -1e-2)
            {
                // cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
                //lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                ++num_iters[currPyrLevel];
            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            tm.stop();
            cout << "Iterations " << num_iters[currPyrLevel] << " time = " << tm.getTimeSec()*1000 << " ms" << endl;
            cout << "update_pose \n" << update_pose.transpose() << endl;
            cout << "diff_error " << diff_error << endl;
#endif

            if(visualize && diff_error > 0)
                showImgDiff();
        }
    }

    if(visualize)
        closeWindows();

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    cout << "Iterations: ";
    for(currPyrLevel = nPyrLevels; currPyrLevel >= 0; currPyrLevel--)
        cout << num_iters[currPyrLevel] << " ";
    cout << endl;
    //#endif

    registered_pose = pose_estim;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " doRegistration took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Update the Hessian and the Gradient from a list of jacobians and residuals. */
void DirectRegistration::updateHessianAndGradient(const MatrixXf & pixel_jacobians, const MatrixXf & pixel_residuals, const MatrixXi & valid_pixels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    assert( pixel_jacobians.rows() == pixel_residuals.rows() && pixel_residuals.rows() == valid_pixels.rows() );
    assert( pixel_jacobians.cols() == 6 && pixel_residuals.cols() == 1 && valid_pixels.cols() == 1);

    float h11=0,h12=0,h13=0,h14=0,h15=0,h16=0,h22=0,h23=0,h24=0,h25=0,h26=0,h33=0,h34=0,h35=0,h36=0,h44=0,h45=0,h46=0,h55=0,h56=0,h66=0;
    float g1=0,g2=0,g3=0,g4=0,g5=0,g6=0;

//    if(use_salient_pixels)
//    {
//    #if ENABLE_OPENMP
//    #pragma omp parallel for reduction (+:h11,h12,h13,h14,h15,h16,h22,h23,h24,h25,h26,h33,h34,h35,h36,h44,h45,h46,h55,h56,h66,g1,g2,g3,g4,g5,g6) // Cannot reduce on Eigen types
//    #endif
//        for(int i=0; i < pixel_jacobians.rows(); i++)
//        {
//            h11 += pixel_jacobians(i,0)*pixel_jacobians(i,0);
//            h12 += pixel_jacobians(i,0)*pixel_jacobians(i,1);
//            h13 += pixel_jacobians(i,0)*pixel_jacobians(i,2);
//            h14 += pixel_jacobians(i,0)*pixel_jacobians(i,3);
//            h15 += pixel_jacobians(i,0)*pixel_jacobians(i,4);
//            h16 += pixel_jacobians(i,0)*pixel_jacobians(i,5);
//            h22 += pixel_jacobians(i,1)*pixel_jacobians(i,1);
//            h23 += pixel_jacobians(i,1)*pixel_jacobians(i,2);
//            h24 += pixel_jacobians(i,1)*pixel_jacobians(i,3);
//            h25 += pixel_jacobians(i,1)*pixel_jacobians(i,4);
//            h26 += pixel_jacobians(i,1)*pixel_jacobians(i,5);
//            h33 += pixel_jacobians(i,2)*pixel_jacobians(i,2);
//            h34 += pixel_jacobians(i,2)*pixel_jacobians(i,3);
//            h35 += pixel_jacobians(i,2)*pixel_jacobians(i,4);
//            h36 += pixel_jacobians(i,2)*pixel_jacobians(i,5);
//            h44 += pixel_jacobians(i,3)*pixel_jacobians(i,3);
//            h45 += pixel_jacobians(i,3)*pixel_jacobians(i,4);
//            h46 += pixel_jacobians(i,3)*pixel_jacobians(i,5);
//            h55 += pixel_jacobians(i,4)*pixel_jacobians(i,4);
//            h56 += pixel_jacobians(i,4)*pixel_jacobians(i,5);
//            h66 += pixel_jacobians(i,5)*pixel_jacobians(i,5);

//            g1 += pixel_jacobians(i,0)*pixel_residuals(i);
//            g2 += pixel_jacobians(i,1)*pixel_residuals(i);
//            g3 += pixel_jacobians(i,2)*pixel_residuals(i);
//            g4 += pixel_jacobians(i,3)*pixel_residuals(i);
//            g5 += pixel_jacobians(i,4)*pixel_residuals(i);
//            g6 += pixel_jacobians(i,5)*pixel_residuals(i);
//        }
//    }
//    else
    {
    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:h11,h12,h13,h14,h15,h16,h22,h23,h24,h25,h26,h33,h34,h35,h36,h44,h45,h46,h55,h56,h66,g1,g2,g3,g4,g5,g6) // Cannot reduce on Eigen types
    #endif
        for(int i=0; i < pixel_jacobians.rows(); i++)
            if(valid_pixels(i))
            {
//                hessian += pixel_jacobians.block(i,0,1,6).transpose() * pixel_jacobians.block(i,0,1,6);
//                gradient += pixel_jacobians.block(i,0,1,6).transpose() * pixel_residuals.(i);
                h11 += pixel_jacobians(i,0)*pixel_jacobians(i,0);
                h12 += pixel_jacobians(i,0)*pixel_jacobians(i,1);
                h13 += pixel_jacobians(i,0)*pixel_jacobians(i,2);
                h14 += pixel_jacobians(i,0)*pixel_jacobians(i,3);
                h15 += pixel_jacobians(i,0)*pixel_jacobians(i,4);
                h16 += pixel_jacobians(i,0)*pixel_jacobians(i,5);
                h22 += pixel_jacobians(i,1)*pixel_jacobians(i,1);
                h23 += pixel_jacobians(i,1)*pixel_jacobians(i,2);
                h24 += pixel_jacobians(i,1)*pixel_jacobians(i,3);
                h25 += pixel_jacobians(i,1)*pixel_jacobians(i,4);
                h26 += pixel_jacobians(i,1)*pixel_jacobians(i,5);
                h33 += pixel_jacobians(i,2)*pixel_jacobians(i,2);
                h34 += pixel_jacobians(i,2)*pixel_jacobians(i,3);
                h35 += pixel_jacobians(i,2)*pixel_jacobians(i,4);
                h36 += pixel_jacobians(i,2)*pixel_jacobians(i,5);
                h44 += pixel_jacobians(i,3)*pixel_jacobians(i,3);
                h45 += pixel_jacobians(i,3)*pixel_jacobians(i,4);
                h46 += pixel_jacobians(i,3)*pixel_jacobians(i,5);
                h55 += pixel_jacobians(i,4)*pixel_jacobians(i,4);
                h56 += pixel_jacobians(i,4)*pixel_jacobians(i,5);
                h66 += pixel_jacobians(i,5)*pixel_jacobians(i,5);

                g1 += pixel_jacobians(i,0)*pixel_residuals(i);
                g2 += pixel_jacobians(i,1)*pixel_residuals(i);
                g3 += pixel_jacobians(i,2)*pixel_residuals(i);
                g4 += pixel_jacobians(i,3)*pixel_residuals(i);
                g5 += pixel_jacobians(i,4)*pixel_residuals(i);
                g6 += pixel_jacobians(i,5)*pixel_residuals(i);
            }
    }

    // Assign the values for the hessian and gradient
    hessian(0,0) += h11;
    hessian(1,0) += h12;
    hessian(0,1) = hessian(1,0);
    hessian(2,0) += h13;
    hessian(0,2) = hessian(2,0);
    hessian(3,0) += h14;
    hessian(0,3) = hessian(3,0);
    hessian(4,0) += h15;
    hessian(0,4) = hessian(4,0);
    hessian(5,0) += h16;
    hessian(0,5) = hessian(5,0);
    hessian(1,1) += h22;
    hessian(2,1) += h23;
    hessian(1,2) = hessian(2,1);
    hessian(3,1) += h24;
    hessian(1,3) = hessian(3,1);
    hessian(4,1) += h25;
    hessian(1,4) = hessian(4,1);
    hessian(5,1) += h26;
    hessian(1,5) = hessian(5,1);
    hessian(2,2) += h33;
    hessian(3,2) += h34;
    hessian(2,3) = hessian(3,2);
    hessian(4,2) += h35;
    hessian(2,4) = hessian(4,2);
    hessian(5,2) += h36;
    hessian(2,5) = hessian(5,2);
    hessian(3,3) += h44;
    hessian(4,3) += h45;
    hessian(3,4) = hessian(4,3);
    hessian(5,3) += h46;
    hessian(3,5) = hessian(5,3);
    hessian(4,4) += h55;
    hessian(5,4) += h56;
    hessian(4,5) = hessian(5,4);
    hessian(5,5) += h66;

    gradient(0) += g1;
    gradient(1) += g2;
    gradient(2) += g3;
    gradient(3) += g4;
    gradient(4) += g5;
    gradient(5) += g6;

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " DirectRegistration::updateHessianAndGradient " << pixel_jacobians.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Update the Hessian and the Gradient from a list of jacobians and residuals. */
void DirectRegistration::updateHessianAndGradient(const MatrixXf & pixel_jacobians, const MatrixXf & pixel_residuals, const MatrixXf & pixel_weights, const MatrixXi & valid_pixels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    //cout << "updateHessianAndGradient rows " << pixel_jacobians.rows() << " " << pixel_residuals.rows() << " " << valid_pixels.rows()  << endl;
    assert( pixel_jacobians.rows() == pixel_residuals.rows() && pixel_residuals.rows() == valid_pixels.rows() );
    assert( pixel_jacobians.cols() == 6 && pixel_residuals.cols() == 1 && valid_pixels.cols() == 1);

    float h11=0,h12=0,h13=0,h14=0,h15=0,h16=0,h22=0,h23=0,h24=0,h25=0,h26=0,h33=0,h34=0,h35=0,h36=0,h44=0,h45=0,h46=0,h55=0,h56=0,h66=0;
    float g1=0,g2=0,g3=0,g4=0,g5=0,g6=0;

    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:h11,h12,h13,h14,h15,h16,h22,h23,h24,h25,h26,h33,h34,h35,h36,h44,h45,h46,h55,h56,h66,g1,g2,g3,g4,g5,g6) // Cannot reduce on Eigen types
    #endif
        for(int i=0; i < pixel_jacobians.rows(); i++)
            if(valid_pixels(i))
            {
                Matrix<float,1,6> jacobian = pixel_weights(i) * pixel_jacobians.block(i,0,1,6);
                h11 += jacobian(0)*jacobian(0);
                h12 += jacobian(0)*jacobian(1);
                h13 += jacobian(0)*jacobian(2);
                h14 += jacobian(0)*jacobian(3);
                h15 += jacobian(0)*jacobian(4);
                h16 += jacobian(0)*jacobian(5);
                h22 += jacobian(1)*jacobian(1);
                h23 += jacobian(1)*jacobian(2);
                h24 += jacobian(1)*jacobian(3);
                h25 += jacobian(1)*jacobian(4);
                h26 += jacobian(1)*jacobian(5);
                h33 += jacobian(2)*jacobian(2);
                h34 += jacobian(2)*jacobian(3);
                h35 += jacobian(2)*jacobian(4);
                h36 += jacobian(2)*jacobian(5);
                h44 += jacobian(3)*jacobian(3);
                h45 += jacobian(3)*jacobian(4);
                h46 += jacobian(3)*jacobian(5);
                h55 += jacobian(4)*jacobian(4);
                h56 += jacobian(4)*jacobian(5);
                h66 += jacobian(5)*jacobian(5);

                g1 += jacobian(0)*pixel_residuals(i);
                g2 += jacobian(1)*pixel_residuals(i);
                g3 += jacobian(2)*pixel_residuals(i);
                g4 += jacobian(3)*pixel_residuals(i);
                g5 += jacobian(4)*pixel_residuals(i);
                g6 += jacobian(5)*pixel_residuals(i);

//                g1 += pixel_jacobians(i,0);
//                g2 += pixel_jacobians(i,1);
//                g3 += pixel_jacobians(i,2);
//                g4 += pixel_jacobians(i,3);
//                g5 += pixel_jacobians(i,4);
//                g6 += pixel_jacobians(i,5);
            }

    // Assign the values for the hessian and gradient
    hessian(0,0) += h11;
    hessian(1,0) += h12;
    hessian(0,1) = hessian(1,0);
    hessian(2,0) += h13;
    hessian(0,2) = hessian(2,0);
    hessian(3,0) += h14;
    hessian(0,3) = hessian(3,0);
    hessian(4,0) += h15;
    hessian(0,4) = hessian(4,0);
    hessian(5,0) += h16;
    hessian(0,5) = hessian(5,0);
    hessian(1,1) += h22;
    hessian(2,1) += h23;
    hessian(1,2) = hessian(2,1);
    hessian(3,1) += h24;
    hessian(1,3) = hessian(3,1);
    hessian(4,1) += h25;
    hessian(1,4) = hessian(4,1);
    hessian(5,1) += h26;
    hessian(1,5) = hessian(5,1);
    hessian(2,2) += h33;
    hessian(3,2) += h34;
    hessian(2,3) = hessian(3,2);
    hessian(4,2) += h35;
    hessian(2,4) = hessian(4,2);
    hessian(5,2) += h36;
    hessian(2,5) = hessian(5,2);
    hessian(3,3) += h44;
    hessian(4,3) += h45;
    hessian(3,4) = hessian(4,3);
    hessian(5,3) += h46;
    hessian(3,5) = hessian(5,3);
    hessian(4,4) += h55;
    hessian(5,4) += h56;
    hessian(4,5) = hessian(5,4);
    hessian(5,5) += h66;

    gradient(0) += g1;
    gradient(1) += g2;
    gradient(2) += g3;
    gradient(3) += g4;
    gradient(4) += g5;
    gradient(5) += g6;

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " DirectRegistration::updateHessianAndGradient " << pixel_jacobians.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

//void DirectRegistration::updateHessianAndGradient(const MatrixXf & pixel_jacobians, const MatrixXf & pixel_residuals, const MatrixXi & warp_pixels)
//{
//#if PRINT_PROFILING
//    double time_start = pcl::getTime();
//    //for(size_t ii=0; ii<100; ii++)
//    {
//#endif

//    assert( pixel_jacobians.rows() == pixel_residuals.rows() && pixel_residuals.rows() == valid_pixels.rows() );
//    assert( pixel_jacobians.cols() == 6 && pixel_residuals.cols() == 1 && valid_pixels.cols() == 1);

//    float h11=0,h12=0,h13=0,h14=0,h15=0,h16=0,h22=0,h23=0,h24=0,h25=0,h26=0,h33=0,h34=0,h35=0,h36=0,h44=0,h45=0,h46=0,h55=0,h56=0,h66=0;
//    float g1=0,g2=0,g3=0,g4=0,g5=0,g6=0;

//    #if ENABLE_OPENMP
//    #pragma omp parallel for reduction (+:h11,h12,h13,h14,h15,h16,h22,h23,h24,h25,h26,h33,h34,h35,h36,h44,h45,h46,h55,h56,h66,g1,g2,g3,g4,g5,g6) // Cannot reduce on Eigen types
//    #endif
//    for(int i=0; i < pixel_jacobians.rows(); i++)
//    {
//        if( warp_pixels(i) != -1 ) //Compute the jacobian only for the visible points
//        {
//            h11 += pixel_jacobians(i,0)*pixel_jacobians(i,0);
//            h12 += pixel_jacobians(i,0)*pixel_jacobians(i,1);
//            h13 += pixel_jacobians(i,0)*pixel_jacobians(i,2);
//            h14 += pixel_jacobians(i,0)*pixel_jacobians(i,3);
//            h15 += pixel_jacobians(i,0)*pixel_jacobians(i,4);
//            h16 += pixel_jacobians(i,0)*pixel_jacobians(i,5);
//            h22 += pixel_jacobians(i,1)*pixel_jacobians(i,1);
//            h23 += pixel_jacobians(i,1)*pixel_jacobians(i,2);
//            h24 += pixel_jacobians(i,1)*pixel_jacobians(i,3);
//            h25 += pixel_jacobians(i,1)*pixel_jacobians(i,4);
//            h26 += pixel_jacobians(i,1)*pixel_jacobians(i,5);
//            h33 += pixel_jacobians(i,2)*pixel_jacobians(i,2);
//            h34 += pixel_jacobians(i,2)*pixel_jacobians(i,3);
//            h35 += pixel_jacobians(i,2)*pixel_jacobians(i,4);
//            h36 += pixel_jacobians(i,2)*pixel_jacobians(i,5);
//            h44 += pixel_jacobians(i,3)*pixel_jacobians(i,3);
//            h45 += pixel_jacobians(i,3)*pixel_jacobians(i,4);
//            h46 += pixel_jacobians(i,3)*pixel_jacobians(i,5);
//            h55 += pixel_jacobians(i,4)*pixel_jacobians(i,4);
//            h56 += pixel_jacobians(i,4)*pixel_jacobians(i,5);
//            h66 += pixel_jacobians(i,5)*pixel_jacobians(i,5);

//            g1 += pixel_jacobians(i,0)*pixel_residuals(i);
//            g2 += pixel_jacobians(i,1)*pixel_residuals(i);
//            g3 += pixel_jacobians(i,2)*pixel_residuals(i);
//            g4 += pixel_jacobians(i,3)*pixel_residuals(i);
//            g5 += pixel_jacobians(i,4)*pixel_residuals(i);
//            g6 += pixel_jacobians(i,5)*pixel_residuals(i);
//        }
//    }

//    // Assign the values for the hessian and gradient
//    hessian(0,0) += h11;
//    hessian(1,0) += h12;
//    hessian(0,1) = hessian(1,0);
//    hessian(2,0) += h13;
//    hessian(0,2) = hessian(2,0);
//    hessian(3,0) += h14;
//    hessian(0,3) = hessian(3,0);
//    hessian(4,0) += h15;
//    hessian(0,4) = hessian(4,0);
//    hessian(5,0) += h16;
//    hessian(0,5) = hessian(5,0);
//    hessian(1,1) += h22;
//    hessian(2,1) += h23;
//    hessian(1,2) = hessian(2,1);
//    hessian(3,1) += h24;
//    hessian(1,3) = hessian(3,1);
//    hessian(4,1) += h25;
//    hessian(1,4) = hessian(4,1);
//    hessian(5,1) += h26;
//    hessian(1,5) = hessian(5,1);
//    hessian(2,2) += h33;
//    hessian(3,2) += h34;
//    hessian(2,3) = hessian(3,2);
//    hessian(4,2) += h35;
//    hessian(2,4) = hessian(4,2);
//    hessian(5,2) += h36;
//    hessian(2,5) = hessian(5,2);
//    hessian(3,3) += h44;
//    hessian(4,3) += h45;
//    hessian(3,4) = hessian(4,3);
//    hessian(5,3) += h46;
//    hessian(3,5) = hessian(5,3);
//    hessian(4,4) += h55;
//    hessian(5,4) += h56;
//    hessian(4,5) = hessian(5,4);
//    hessian(5,5) += h66;

//    gradient(0) += g1;
//    gradient(1) += g2;
//    gradient(2) += g3;
//    gradient(3) += g4;
//    gradient(4) += g5;
//    gradient(5) += g6;

//    #if PRINT_PROFILING
//    }
//    double time_end = pcl::getTime();
//    cout << " DirectRegistration::updateHessianAndGradient " << pixel_jacobians.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
//    #endif
//}

void DirectRegistration::updateHessianAndGradient3D(const MatrixXf & pixel_jacobians, const MatrixXf & pixel_residuals, const MatrixXi & valid_pixels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    ASSERT_( 3*valid_pixels.rows() == pixel_jacobians.rows() );

    for(int i=0; i < valid_pixels.rows(); i++)
        if(valid_pixels(i))
        {
            //MatrixXf jacobian = pixel_jacobians.block(i,0,3,6);
            MatrixXf jacobian = pixel_jacobians.block(i*3,0,3,6);
            hessian += jacobian.transpose() * jacobian;
            gradient += jacobian.transpose() * pixel_residuals.block(i*3,0,3,1);
            //gradient += jacobian.transpose() * pixel_residuals.block(i*3,0,3,1).norm();
        }

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " DirectRegistration::updateHessianAndGradient3D " << valid_pixels.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

void DirectRegistration::updateHessianAndGradient3D(const MatrixXf & pixel_jacobians, const MatrixXf & pixel_residuals, const MatrixXf & pixel_weights, const MatrixXi & valid_pixels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    assert( 3*valid_pixels.rows() == pixel_jacobians.rows() );

    //cout << "INIT hessian \n" << hessian.transpose() << endl << "gradient \n" << gradient.transpose() << endl;
    for(int i=0; i < valid_pixels.rows(); i++)
        if(valid_pixels(i))
        {
            MatrixXf jacobian = pixel_weights(i) * pixel_jacobians.block(i*3,0,3,6);
            hessian += jacobian.transpose() * jacobian;
            gradient += jacobian.transpose() * pixel_residuals.block(i*3,0,3,1);
            //cout << i << " hessian \n" << hessian.transpose() << endl << "gradient \n" << gradient.transpose() << endl;
        }

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " DirectRegistration::updateHessianAndGradient3D " << valid_pixels.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

void DirectRegistration::updateGrad(const MatrixXf & pixel_jacobians, const MatrixXf & pixel_residuals, const MatrixXi & valid_pixels)
{
    #if PRINT_PROFILING
        double time_start = pcl::getTime();
        //for(size_t ii=0; ii<100; ii++)
        {
    #endif

    assert( pixel_jacobians.rows() == pixel_residuals.rows() && pixel_residuals.rows() == valid_pixels.rows() );
    assert( pixel_jacobians.cols() == 6 && pixel_residuals.cols() == 1 && valid_pixels.cols() == 1);

    float g1=0,g2=0,g3=0,g4=0,g5=0,g6=0;

    #if ENABLE_OPENMP
    #pragma omp parallel for reduction (+:g1,g2,g3,g4,g5,g6) // Cannot reduce on Eigen types
    #endif
    for(int i=0; i < pixel_jacobians.rows(); i++)
    {
        if(valid_pixels(i))
        {
            g1 += pixel_jacobians(i,0)*pixel_residuals(i);
            g2 += pixel_jacobians(i,1)*pixel_residuals(i);
            g3 += pixel_jacobians(i,2)*pixel_residuals(i);
            g4 += pixel_jacobians(i,3)*pixel_residuals(i);
            g5 += pixel_jacobians(i,4)*pixel_residuals(i);
            g6 += pixel_jacobians(i,5)*pixel_residuals(i);
        }
    }

    // Assign the values for the gradient
    gradient(0) += g1;
    gradient(1) += g2;
    gradient(2) += g3;
    gradient(3) += g4;
    gradient(4) += g5;
    gradient(5) += g6;

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " DirectRegistration::updateGrad " << pixel_jacobians.rows() << " pts took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

///*! Get a list of salient points from a list of Jacobians corresponding to a set of 3D points */
//void DirectRegistration::getSalientPts(const MatrixXf & jacobians, std::vector<size_t> & salient_pixels, const float r_salient )
//{
//    //cout << " DirectRegistration::getSalientPts " << input_pts.rows() << " pts \n";
//#if PRINT_PROFILING
//    double time_start = pcl::getTime();
//    //for(size_t ii=0; ii<100; ii++)
//    {
//#endif

//   assert( jacobians.cols() == 6 && jacobians.rows() > 6 );
//   salient_pixels.resize(0);
//   if( jacobians.rows() < r_salient ) // If the user ask for more salient points than current points, do nothing
//       return;


////   const size_t n_pts = jacobians.rows();
////   size_t n_salient;
////   if(r_salient < 2)
////       n_salient = floor(r_salient * n_pts);
////   else // set the number of points
////       n_salient = size_t(r_salient);

////   std::vector<size_t> sorted_idx;
////   sorted_idx.reserve(6*n_pts);
////   // [U,S,V] = svd(J,'econ');

////   for(size_t i = 0 ; i < 6 ; ++i)
////   {
////       std::vector<size_t> idx = sort_indexes_<float>(jacobians.col(i));
////       sorted_idx.insert( sorted_idx.end(), idx.begin(), idx.end() );
////   }

////   //std::vector<size_t> salient_pixels(n_pts);
////   salient_pixels.resize(n_pts);

////   size_t *tabetiq = new size_t [n_pts*6];
////   size_t *tabi = new size_t [6];

////   for(size_t j = 0 ; j < n_pts ; ++j)
////        salient_pixels[j] = 0;

////   for(size_t i = 0 ; i < 6 ; ++i)
////   {
////       tabi[i] = 0;
////       for(size_t j = 0 ; j < n_pts ; ++j)
////            tabetiq[j*6+i] = 0;
////   }

////    size_t k = 0;
////    size_t line;
////    for(size_t i = 0 ; i < floor(n_pts/6) ; ++i) // n_pts/6 is Weird!!
////    {
////        for(size_t j = 0 ; j < 6 ; ++j)
////        {
////             while(tabi[j] < n_pts)
////             {
////                 line = (size_t)(sorted_idx[tabi[j]*6 + j]);
////                 if(tabetiq[j + line*6] == 0)
////                 {
////                    for( size_t jj = 0 ; jj < 6 ; ++jj)
////                        tabetiq[(size_t)(line*6 + jj)] = 1;

////                    salient_pixels[k] =  sorted_idx[tabi[j]*6+ j];
////                    ++k;
////                    tabi[j]++;
////                    //deg[k] = j;
////                    if(k == n_salient) // Stop arranging the indices
////                    {
////                        j = 6;
////                        i = floor(n_pts/6);
////                    }
////                    break;
////                 }
////                 else
////                 {
////                     tabi[j]++;
////                 }

////             }
////        }
////    }

////    delete [] tabetiq;
////    delete [] tabi;
////    salient_pixels.resize(n_salient);


//    // Input Npixels*Ndof sorted columns
//    const size_t Npixels = jacobians.rows();
//    const size_t Ndof = jacobians.cols();
//    size_t N_desired;
//    if(r_salient < 2)
//        N_desired = floor(r_salient * Npixels);
//    else // set the number of points
//        N_desired = size_t(r_salient);

//    std::vector<size_t> sorted_idx;
//    sorted_idx.reserve(Ndof*Npixels);
//    // [U,S,V] = svd(J,'econ');

//    for(size_t i = 0 ; i < Ndof ; ++i)
//    {
//        std::vector<size_t> idx = sort_indexes_<float>(jacobians.col(i));
//        sorted_idx.insert( sorted_idx.end(), idx.begin(), idx.end() );
//    }

//    // Npixels output //
//    salient_pixels.resize(N_desired);
//    std::fill(salient_pixels.begin(), salient_pixels.end(), 0);

//    // Mask for selected pixels
//    bool *mask = new bool [Npixels];
//    std::fill(mask, mask+Npixels, 0);

//    // Current dof counter //
//    size_t tabi[Ndof];
//    std::fill(tabi, tabi+Ndof, 0);

//    int current_idx = 0;

//    // Pick (N_desired/Ndof) pixels per dof

//    //while(current_idx<N_desired)
//    for(int i = 0 ; i < floor((double)(N_desired)/Ndof) ; ++i)
//    {
//        for(int j = 0; j < Ndof ; ++j) // For each dof
//        {
//            // Get the best unselected pixel for dof j
//            while(tabi[j] < Npixels)
//            {
//                // Get pixel idx for dof j
//                int pixel_idx = (int)(sorted_idx[tabi[j]+j*Npixels]);

//                // If pixel is unselected, store index and switch dof
//                if(!mask[pixel_idx])
//                {
//                    // Store pixel idx
//                    salient_pixels[current_idx] = pixel_idx;

//                    // Set pixel as selected
//                    mask[pixel_idx] = 1;

//                    // Increment counters
//                    ++tabi[j];
//                    ++current_idx;
//                    break;
//                }
//                else
//                    ++tabi[j];
//            }
//        }
//    }

//    delete [] mask;

//    #if PRINT_PROFILING
//    }
//    double time_end = pcl::getTime();
//    cout << " DirectRegistration::getSalientPts " << jacobians.rows() << " pts took " << (time_end - time_start)*1000 << " ms. \n";
//    #endif
//}

#define NBINS 256
typedef std::pair<float,float> min_max;
void histogram(uint *hist, min_max &scale,const float *vec,const int size,const bool *mask)
{
    scale.first = 1e15;
    scale.second = 0.0;

    // Compute min/max
    for(int i = 0 ; i < size ; ++i)
    {
        if(!mask[i])
        {
            scale.second = std::max(scale.second,float(fabs(vec[i])));
            scale.first = std::min(scale.first,float(fabs(vec[i])));
        }
    }

    // Avoid dividing by 0
    if(scale.second==0.0 && scale.first==0.0)
        scale.second=1.0;


  /*  mexPrintf("Min %f\n",scale.first);
    mexPrintf("Max %f\n",scale.second);*/

    // Compute histogram
    std::fill(hist,hist+NBINS,0);
    for(int i = 0 ; i < size ; ++i)
    {
       if(!mask[i])
         hist[(int)((float)(fabs(vec[i])-scale.first)/(scale.second-scale.first)*(NBINS-1))]++;
    }
}

int median_idx_from_hist(uint *hist,const int stop)
{
    int i = 0, cumul0 = 0,cumul1=0;
    for(i = NBINS-1 ; i >= 0 ; --i)
    {
         cumul1+=hist[i];
         if(cumul1>=stop)
             break;
         cumul0 = cumul1;
    }
    // Return the closest idx
    if(abs(cumul0-stop) < abs(cumul1-stop))
        return (int)std::max((int)0,i-1);
    else
        return i;
}
float histogram_value_from_idx(const int idx,min_max &scale)
{
    return (float)(idx)*(scale.second-scale.first)/(NBINS-1)+scale.first;
}

/*! Get a list of salient points from a list of Jacobians corresponding to a set of 3D points */
void DirectRegistration::getSalientPts(const MatrixXf & jacobians, std::vector<size_t> & salient_pixels, const float r_salient )
{
    //cout << " DirectRegistration::getSalientPts " << input_pts.rows() << " pts \n";
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

   assert( jacobians.cols() == 6 && jacobians.rows() > 6 );
   salient_pixels.resize(0);
   if( jacobians.rows() < r_salient ) // If the user ask for more salient points than current points, do nothing
       return;

   // Input Npixels*Ndof sorted columns
   const size_t Npixels = jacobians.rows();
   const size_t Ndof = jacobians.cols();
   size_t N_desired;
   if(r_salient < 2)
       N_desired = floor(r_salient * Npixels);
   else // set the number of points
       N_desired = size_t(r_salient);

   // Npixels output //
   salient_pixels.resize(N_desired);
   std::fill(salient_pixels.begin(), salient_pixels.end(), 0);

   uint hist[NBINS];
   bool *mask = new bool [Npixels];
   std::fill(mask, mask+Npixels, 0);

   min_max scale;

   size_t current_idx = 0;

   for(size_t i = 0 ; i < Ndof ; ++i)
   {
       // Get column pointer //
       const float *vec = &jacobians(0,i);

       // Compute min_max and histogram
       histogram(hist,scale,vec,Npixels,mask);

       // Extract the N median
       int N_median_idx = median_idx_from_hist(hist,floor(double(N_desired)/Ndof));
       float N_median = histogram_value_from_idx(N_median_idx,scale);

       // Now pick up the indices //
       size_t j=0, k=0;

       while(j<Npixels && k < floor(double(N_desired)/Ndof))
       {
           // Get next pixel //
           if(!mask[j] && (fabs(vec[j]) >= N_median))
           {
               salient_pixels[current_idx] = j;
               mask[j] = 1;
               ++current_idx;
               ++k;
           }
           else
           {
               ++j;
           }
       }

       if(current_idx>=N_desired)
           break;
    }
    delete [] mask;

   #if PRINT_PROFILING
   }
   double time_end = pcl::getTime();
   cout << " DirectRegistration::getSalientPts HISTOGRAM APPROXIMATION " << jacobians.rows() << " pts took " << (time_end - time_start)*1000 << " ms. \n";
   #endif
}

void DirectRegistration::trimValidPoints(MatrixXf & LUT_xyz, VectorXi & validPixels, MatrixXf & xyz_transf,
                                        VectorXi & validPixelsPhoto, VectorXi & validPixelsDepth,
                                        const costFuncType method,
                                        std::vector<size_t> &salient_pixels, std::vector<size_t> &salient_pixels_photo, std::vector<size_t> &salient_pixels_depth)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //cout << " DirectRegistration::trimValidPoints pts " << LUT_xyz.rows() << " pts " << validPixelsPhoto.rows() << " pts "<< salient_pixels_photo.size()<< " - " << salient_pixels_depth.size() << endl;
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    if( salient_pixels_photo.empty() && salient_pixels_depth.empty() ){ cout << " DirectRegistration::trimValidPoints EMPTY set of salient points \n";
        return;}

    // Arrange all the list of points and indices, by now this is done outside this function
    if(method == PHOTO_DEPTH)
    {
//            //salient_pixels = new std::vector<size_t>( salient_pixels_photo.size() + salient_pixels_depth.size() );
//            salient_pixels.resize( salient_pixels_photo.size() + salient_pixels_depth.size() );
//            size_t countValidPix = 0, i = 0, j = 0;
//            while( i < salient_pixels_photo.size() && j < salient_pixels_depth.size() )
//            {
//                if( salient_pixels_photo[i] == salient_pixels_depth[j] )
//                {
//                    salient_pixels[countValidPix] = salient_pixels_photo[i];
//                    ++countValidPix;
//                    ++i;
//                    ++j;
//                }
//                else if(salient_pixels_photo[i] < salient_pixels_depth[j])
//                {
//                    salient_pixels[countValidPix] = salient_pixels_photo[i];
//                    ++countValidPix;
//                    ++i;
//                }
//                else
//                {
//                    salient_pixels[countValidPix] = salient_pixels_depth[j];
//                    ++countValidPix;
//                    ++j;
//                }
//            }; // TODO: add the elements of the unfinished list
//            salient_pixels.resize(countValidPix);

        std::set<size_t> s_salient_pixels(salient_pixels_photo.begin(), salient_pixels_photo.end());
        s_salient_pixels.insert(salient_pixels_depth.begin(), salient_pixels_depth.end());
        salient_pixels.resize(0);
        salient_pixels.insert(salient_pixels.end(), s_salient_pixels.begin(), s_salient_pixels.end());
    }
    else if(method == PHOTO_CONSISTENCY)
        salient_pixels = salient_pixels_photo;
    else if(method == DEPTH_CONSISTENCY)
        salient_pixels = salient_pixels_depth;

    size_t aligned_pts = salient_pixels.size() - salient_pixels.size() % 4;
    salient_pixels.resize(aligned_pts);

    if(use_salient_pixels)
    {
        // Arrange salient pixels
        VectorXi validPixels_tmp(salient_pixels.size());
        MatrixXf LUT_xyz_tmp(salient_pixels.size(),3);
        //MatrixXf xyz_transf_tmp(salient_pixels.size(),3);
//        VectorXi validPixelsPhoto_tmp(salient_pixels.size());
//        VectorXi validPixelsDepth_tmp(salient_pixels.size());
        for(size_t i=0; i < salient_pixels.size(); ++i)
        {
            //cout << i << " " << salient_pixels[i] << " \n";
            validPixels_tmp(i) = validPixels(salient_pixels[i]);
            LUT_xyz_tmp.block(i,0,1,3) = LUT_xyz.block(salient_pixels[i],0,1,3);
            //xyz_transf_tmp.block(i,0,1,3) = xyz_transf.block(salient_pixels[i],0,1,3);
//            validPixelsPhoto_tmp(i) = validPixelsPhoto(salient_pixels[i]);
//            validPixelsDepth_tmp(i) = validPixelsDepth(salient_pixels[i]);
        }
        validPixels = validPixels_tmp;
        LUT_xyz = LUT_xyz_tmp;
        //xyz_transf = xyz_transf_tmp;

//        validPixelsPhoto = validPixelsPhoto_tmp;
//        validPixelsDepth = validPixelsDepth_tmp;
    }
    else
    {
        validPixels.setZero();
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        {
            //validPixelsPhoto.setZero(); // VectorXi::Zero(LUT_xyz.rows());
            for(size_t j = 0 ; j < salient_pixels_photo.size() ; ++j)
            {
                validPixels(salient_pixels_photo[j]) = 1;
                //validPixelsPhoto(salient_pixels_photo[j]) = 1;
            }
        }
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        {
            //validPixelsDepth.setZero();
            for(size_t j = 0 ; j < salient_pixels_depth.size() ; ++j)
            {
                validPixels(salient_pixels_depth[j]) = 1;
                //validPixelsDepth(salient_pixels_depth[j]) = 1;
            }
        }
    }

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " DirectRegistration::trimValidPoints took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}
