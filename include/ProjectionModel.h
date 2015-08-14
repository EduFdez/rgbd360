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

#ifndef PROJECTION_MODEL_H
#define PROJECTION_MODEL_H

#include <Miscellaneous.h>
//#include <Saliency.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
//#include "/usr/local/include/eigen3/Eigen/Core"

#if PRINT_PROFILING
    #include <pcl/common/time.h>
#endif

#if _TEST_SIMD
    #include <mrpt/system/os.h>
    #include <mrpt/utils/mrpt_macros.h>
    //#include <assert.h>
    #include <bitset>

    #define MAX_ULPS 2097152 //1000000000

// This function is explained in: http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
// Usable AlmostEqual function
inline bool AlmostEqual2sComplement(float A, float B, int maxUlps)
{
    if( abs(A - B) < 1e-4)
        return true;
    else
        return false;

    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.
    assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    int aInt = *(int*)&A;
    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0)
        aInt = 0x80000000 - aInt;
    // Make bInt lexicographically ordered as a twos-complement int
    int bInt = *(int*)&B;
    if (bInt < 0)
        bInt = 0x80000000 - bInt;
    int intDiff = abs(aInt - bInt);
    if (intDiff <= maxUlps)
        return true;
    return false;
}
#endif

/*! This class encapsulates different projection models including both perspective and spherical.
 *  It implements the functionality to project and reproject from the image domain to 3D and viceversa.
 *
 * TODO: This class could be split using polymorphism, now it uses function pointers and it is compatible with C.
 * Polymorphism has the advantage that the code is easier to read and tu reuse, but it may have a lower performance.
 */
class ProjectionModel
{
public:
//protected:

//    /*! Projection's mathematical model */
//    enum projectionType
//    {
//        PINHOLE,
//        SPHERICAL
//        //,CYLINRICAL
//    } projection_model;

    /*! Image's height and width */
    int nRows, nCols;

    /*! Image's size (number of pixels) */
    size_t imgSize;

    /*! Minimum allowed depth to consider a depth pixel valid.*/
    float min_depth_;

    /*! Maximum allowed depth to consider a depth pixel valid.*/
    float max_depth_;

//public:

    ProjectionModel() :
        //projection_model(PINHOLE),  // SPHERICAL
        min_depth_(0.3f),
        max_depth_(20.f)
    {
    };

    virtual ~ProjectionModel(){};

    /*! Set the minimum depth distance (m) to consider a certain pixel valid.*/
    inline void setMinDepth(const float minD)
    {
        min_depth_ = minD;
    };

    /*! Set the maximum depth distance (m) to consider a certain pixel valid.*/
    inline void setMaxDepth(const float maxD)
    {
        max_depth_ = maxD;
    };

    /*! Scale the intrinsic calibration parameters according to the image resolution (i.e. the reduced resolution being used). */
    virtual void scaleCameraParams(std::vector<cv::Mat> &depthPyr, const int pyrLevel) = 0;

    /*! Check if a pixel is within the image limits. */
    template<typename T>
    inline bool isInImage(const T x, const T y)
    {
        return ( x >= 0 && x < nCols && y >= 0 && y < nRows);
    }

    /*! Return the depth value of the 3D point projected on the image.*/
    virtual float getDepth(const Eigen::Vector3f &xyz) = 0;

    /*! Get a 3D points corresponding to the pixel "idx" in the given range image.*/
    virtual inline void getPoint3D(const float *depth_img, const int idx, Eigen::Vector3f & xyz) = 0;
    virtual inline void getPoint3D(const cv::Mat & depth_img, const cv::Point2f warped_pixel, Eigen::Vector3f & xyz) = 0;

    /*! Project 3D points XYZ. */
    virtual inline cv::Point2f project2Image(Eigen::Vector3f & xyz) = 0;

    /*! Re-project the warping image into the reference one. The input points 'xyz' represent the reference point cloud as seen by the target image. */
    virtual void reproject(const Eigen::MatrixXf & xyz, const cv::Mat & gray, cv::Mat & warped_gray, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible) = 0;

    /*! Project 3D points XYZ according to the pinhole camera model (3D -> 2D). */
    virtual void project(const Eigen::MatrixXf & xyz, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible) = 0;

    /*! Project 3D points XYZ according to the pinhole camera model (3D -> 1D nearest neighbor). */
    virtual void projectNN(const Eigen::MatrixXf & xyz, Eigen::VectorXi & valid_pixels, Eigen::VectorXi & warped_pixels) = 0;

    /*! Compute the 3D points XYZ according to the pinhole camera model. */
    virtual void reconstruct3D(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels) = 0;

    /*! Compute the 3D points XYZ according to the pinhole camera model. */
    virtual void reconstruct3D_saliency ( const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels, const int method,
                                          const cv::Mat & depth_gradX, const cv::Mat & depth_gradY, const float max_depth_grad, const float thres_saliency_depth,
                                          const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY, const float thres_saliency_gray
                                        ) = 0;

    /*! Warp the image according to a given geometric transformation. */
    //void warpImage ( const int pyrLevel, const Eigen::Matrix4f &pose_guess, costFuncType method );

    /*! Compute the Jacobian composition of the transformed point: T(x)Tp */
    inline void
    //Eigen::Matrix<float,3,6>
    computeJacobian36_xT_p(const Eigen::Vector3f & xyz, Eigen::Matrix<float,3,6> &jacobianRt)
    {
        //Eigen::Matrix<float,3,6> jacobianWarpRt;

        jacobianRt.block(0,0,3,3) = Eigen::Matrix3f::Identity();
        jacobianRt.block(0,3,3,3) = -skew(xyz);

//        jacobianRt = Eigen::Matrix<float,3,6>::Zero();
//        jacobianRt(0,0) = 1.f;
//        jacobianRt(1,1) = 1.f;
//        jacobianRt(2,2) = 1.f;
//        jacobianRt(0,4) = xyz(2);
//        jacobianRt(1,3) = -xyz(2);
//        jacobianRt(0,5) = -xyz(1);
//        jacobianRt(2,3) = xyz(1);
//        jacobianRt(1,5) = xyz(0);
//        jacobianRt(2,4) = -xyz(0);
    }

    /*! Compute the Jacobian composition of the transformed point: TT(x)p */
    inline void
    //Eigen::Matrix<float,3,6>
    computeJacobian36_Tx_p(const Eigen::Matrix3f & rot, const Eigen::Vector3f & xyz, Eigen::Matrix<float,3,6> &jacobianRt)
    {
        //Eigen::Matrix<float,3,6> jacobianWarpRt;

        jacobianRt.block(0,0,3,3) = Eigen::Matrix3f::Identity();
        jacobianRt.block(0,3,3,3) = -skew(xyz);

        jacobianRt = rot * jacobianRt;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    virtual inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT(const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt) = 0;

    /*! Compute the Jacobian of the warp */
    virtual inline void computeJacobian23_warp(const Eigen::Vector3f & xyz, Eigen::Matrix<float,2,3> &jacobianWarp) = 0;

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    virtual inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wTTx(const Eigen::Matrix4f & Rt, const Eigen::Vector3f & xyz, const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt) = 0;

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation). */
    virtual void computeJacobiansPhoto(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians, float *_grayGradX, float *_grayGradY) = 0;

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation). */
    virtual void computeJacobiansDepth(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians, float *_depthGradX, float *_depthGradY) = 0;

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation). */
    virtual void computeJacobiansPhotoDepth(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights,
                                            Eigen::MatrixXf & jacobians_photo, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY) = 0;

    /*! Compute the 3Nx6 jacobian matrices of the ICP using the spherical camera model. */
    //void computeJacobiansICP(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXf & residualsDepth, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY)
    void computeJacobiansICP(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_depth) //, float *_depthGradX, float *_depthGradY)
    {
    #if PRINT_PROFILING
        double time_start = pcl::getTime();
        //for(size_t ii=0; ii<100; ii++)
        {
    #endif

        //jacobians_depth.resize(xyz_tf.rows(), 6);
        jacobians_depth.resize(3*xyz_tf.rows(), 6);

    //    #if _ENABLE_OPENMP
    //    #pragma omp parallel for
    //    #endif
        for(int i=0; i < xyz_tf.rows(); i++)
        {
            Eigen::Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
            Eigen::Matrix<float,3,6> jacobianRt;
            computeJacobian36_xT_p(pt_xyz, jacobianRt);
            jacobians_depth.block(3*i,0,3,6) = (weights(i) * stdDevError_inv(i)) * jacobianRt;
            //jacobians_depth.block(i,0,3,6) = jacobianRt * residualsDepth.block(3*i,0,3,1);
        }

    #if PRINT_PROFILING
        }
        double time_end = pcl::getTime();
        cout << " SphericalModel::computeJacobiansICP " << xyz_tf.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
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


    ///*! Function to obtain a pixel value with bilinear interpolation. Unsafe function Unsafe (it does not check that the pixel is inside the image limits) */
    //float getPixelBilinear(const float* img, float x, float y)
    //{
    //#if !(_SSE3) // # ifdef __SSE3__
    //    int px = (int)x; // floor of x
    //    int py = (int)y; // floor of y
    //    const int stride = img->width;
    //    const Pixel* p0 = img->data + px + py * stride; // pointer to first pixel

    //    // load the four neighboring pixels
    //    const Pixel& p1 = p0[0 + 0 * stride];
    //    const Pixel& p2 = p0[1 + 0 * stride];
    //    const Pixel& p3 = p0[0 + 1 * stride];
    //    const Pixel& p4 = p0[1 + 1 * stride];

    //    // Calculate the weights for each pixel
    //    float fx = x - px;
    //    float fy = y - py;
    //    float fx1 = 1.0f - fx;
    //    float fy1 = 1.0f - fy;

    //    int w1 = fx1 * fy1 * 256.0f;
    //    int w2 = fx  * fy1 * 256.0f;
    //    int w3 = fx1 * fy  * 256.0f;
    //    int w4 = fx  * fy  * 256.0f;

    //    // Calculate the weighted sum of pixels (for each color channel)
    //    int outr = p1.r * w1 + p2.r * w2 + p3.r * w3 + p4.r * w4;
    //    int outg = p1.g * w1 + p2.g * w2 + p3.g * w3 + p4.g * w4;
    //    int outb = p1.b * w1 + p2.b * w2 + p3.b * w3 + p4.b * w4;
    //    int outa = p1.a * w1 + p2.a * w2 + p3.a * w3 + p4.a * w4;

    //    return Pixel(outr >> 8, outg >> 8, outb >> 8, outa >> 8);

    //#else // SSE optimzed
    //    const int stride = img->width;
    //    const Pixel* p0 = img->data + (int)x + (int)y * stride; // pointer to first pixel

    //    // Load the data (2 pixels in one load)
    //    __m128i p12 = _mm_loadl_epi64((const __m128i*)&p0[0 * stride]);
    //    __m128i p34 = _mm_loadl_epi64((const __m128i*)&p0[1 * stride]);

    //    __m128 weight = CalcWeights(x, y);

    //    // convert RGBA RGBA RGBA RGAB to RRRR GGGG BBBB AAAA (AoS to SoA)
    //    __m128i p1234 = _mm_unpacklo_epi8(p12, p34);
    //    __m128i p34xx = _mm_unpackhi_epi64(p1234, _mm_setzero_si128());
    //    __m128i p1234_8bit = _mm_unpacklo_epi8(p1234, p34xx);

    //    // extend to 16bit
    //    __m128i pRG = _mm_unpacklo_epi8(p1234_8bit, _mm_setzero_si128());
    //    __m128i pBA = _mm_unpackhi_epi8(p1234_8bit, _mm_setzero_si128());

    //    // convert weights to integer
    //    weight = _mm_mul_ps(weight, CONST_256);
    //    __m128i weighti = _mm_cvtps_epi32(weight); // w4 w3 w2 w1
    //    weighti = _mm_packs_epi32(weighti, weighti); // 32->2x16bit

    //    //outRG = [w1*R1 + w2*R2 | w3*R3 + w4*R4 | w1*G1 + w2*G2 | w3*G3 + w4*G4]
    //    __m128i outRG = _mm_madd_epi16(pRG, weighti);
    //    //outBA = [w1*B1 + w2*B2 | w3*B3 + w4*B4 | w1*A1 + w2*A2 | w3*A3 + w4*A4]
    //    __m128i outBA = _mm_madd_epi16(pBA, weighti);

    //    // horizontal add that will produce the output values (in 32bit)
    //    __m128i out = _mm_hadd_epi32(outRG, outBA);
    //    out = _mm_srli_epi32(out, 8); // divide by 256

    //    // convert 32bit->8bit
    //    out = _mm_packus_epi32(out, _mm_setzero_si128());
    //    out = _mm_packus_epi16(out, _mm_setzero_si128());

    //    // return
    //    return _mm_cvtsi128_si32(out);
    //#endif
    //}

};

#endif
