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

#include <Miscellaneous.h>
//#include <Saliency.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
//#include "/usr/local/include/eigen3/Eigen/Core"

#ifndef PROJECTION_MODEL_H
#define PROJECTION_MODEL_H

/*! This class encapsulates different projection models including both perspective and spherical.
 *  It implements the functionality to project and reproject from the image domain to 3D and viceversa.
 */
class ProjectionModel
{
//public:
protected:

    /*! Projection's mathematical model */
    enum projectionType
    {
        PINHOLE,
        SPHERICAL
        //,CYLINRICAL
    } projection_model;

    /*! Camera matrix (intrinsic parameters). This is only required for pinhole perspective sensors */
    Eigen::Matrix3f cameraMatrix;

    /*! Image's height and width */
    int nRows, nCols;

    /*! Image's size (number of pixels) */
    size_t imgSize;

    /*! Camera intrinsic parameters */
    float fx, fy, ox, oy;
    float inv_fx, inv_fy;

    /*! Spherical model parameters */
    float pixel_angle;
    float pixel_angle_inv;
    float half_width;
    float phi_start;

//    /* Vertical field of view in the sphere (in) .*/
//    float phi_FoV;

    /*! Minimum allowed depth to consider a depth pixel valid.*/
    float min_depth_;

    /*! Maximum allowed depth to consider a depth pixel valid.*/
    float max_depth_;

    /*! Pointer to reconstruct3D function .*/
    void (*reconstruct3D)(cv::Mat &, Eigen::MatrixXf &, Eigen::VectorXi &);

    /*! Pointer to reconstruct3D function using saliency .*/
    void (*reconstruct3D_saliency) (Eigen::MatrixXf &, Eigen::VectorXi &, const cv::Mat &, const cv::Mat &, const cv::Mat &, const cv::Mat &, const cv::Mat &, const cv::Mat &, const float, const float);

    /*! Project 3D points XYZ according to the pinhole camera model. */
    cv::Point2f (*project2Image)(Eigen::Vector3f &);

    /*! Check if a pixel is within the image limits. */
    bool (*isInImage)(const int, const int);

    /*! Project 3D points to 2D image coordinates. */
    void (*project)(Eigen::MatrixXf &, Eigen::MatrixXf &);

    /*! Project 3D points to 1D image pixels using nearest neighbor interpolation. */
    void (*projectNN)(Eigen::MatrixXf &, Eigen::MatrixXi &, Eigen::MatrixXi &);

    /*! Return the depth value of the 3D point projected on the image.*/
    inline float (*getDepth)(const Eigen::Vector3f &);

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    void (*computeJacobian26_wT)(const Eigen::Vector3f & , Eigen::Matrix<float,2,6> &);

    /*! Compute the 2x6 jacobian matrices of the composition (warping+rigidTransformation). */
    void (*computeJacobians)(Eigen::MatrixXf &, Eigen::MatrixXf &);

    void (*computeJacobiansPhoto)(Eigen::MatrixXf &, const float, Eigen::VectorXf &, Eigen::MatrixXf &, float *, float *);

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
    void (*computeJacobiansDepth)(Eigen::MatrixXf &, Eigen::VectorXf &, Eigen::VectorXf &, Eigen::MatrixXf &, float *, float *);

    /*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
    void (*computeJacobiansPhotoDepth)( Eigen::MatrixXf &, const float, Eigen::VectorXf &, Eigen::VectorXf &,
                                        Eigen::MatrixXf &, Eigen::MatrixXf &, float *, float *, float *, float *);

    // TODO separate between depth/intensity

public:

    ProjectionModel();

    /*! Set Projection Model .*/
    inline void setProjectionModel(const projectionType proj)
    {
        projection_model = proj;
        if(projection_model == PINHOLE)
        {
            project = &project_pinhole;
            project2Image = &project2Image_pinhole;
            projectNN = &projectNN_pinhole;
            getDepth = &getDepth_pinhole;
            isInImage = &isInImage_pinhole<int>;
            reconstruct3D = &reconstruct3D_pinhole;
            reconstruct3D_saliency = &reconstruct3D_pinhole_saliency;
            computeJacobian26_wT = &computeJacobian26_wT_pinhole;
            computeJacobians = &computeJacobians_pinhole;
            computeJacobiansPhoto = &computeJacobiansPhoto_pinhole;
            computeJacobiansDepth = &computeJacobiansDepth_pinhole;
            computeJacobiansPhotoDepth = &computeJacobiansPhotoDepth_pinhole;
        }
        else // SPHERICAL
        {
            project = &project_spherical;
            project2Image = &project2Image_spherical;
            projectNN = &projectNN_spherical;
            getDepth = &getDepth_spherical;
            isInImage = &isInImage_spherical<int>;
            reconstruct3D = &reconstruct3D_spherical;
            reconstruct3D_saliency = &reconstruct3D_pinhole_spherical;
            computeJacobian26_wT = &computeJacobian26_wT_sphere;
            computeJacobians = &computeJacobians_spherical;
            computeJacobiansPhoto = &computeJacobiansPhoto_spherical;
            computeJacobiansDepth = &computeJacobiansDepth_spherical;
            computeJacobiansPhotoDepth = &computeJacobiansPhotoDepth_spherical;
        }
    };

    /*! Set the 3x3 matrix of (pinhole) camera intrinsic parameters used to obtain the 3D colored point cloud from the RGB and depth images.*/
    inline void setCameraMatrix(const Eigen::Matrix3f & camMat)
    {
        cameraMatrix = camMat;
    };

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

    /*! Return the depth value of the 3D point projected on the image.*/
    inline float getDepth_pinhole(const Eigen::Vector3f &xyz)
    {
        return xyz(2);
    }

    /*! Return the depth value of the 3D point projected on the image.*/
    inline float getDepth_spherical(const Eigen::Vector3f &xyz)
    {
        return xyz.norm();
    }

    /*! Scale the intrinsic calibration parameters according to the image resolution (i.e. the reduced resolution being used). */
    void scaleCameraParams(const float scaleFactor);

    template<typename T>
    inline bool isInImage_pinhole(const T x, const T y)
    {
        return ( y >= 0 && y < nRows && x >= 0 && x < nCols );
    };

    /*! Check if a pixel is within the image limits. */
    template<typename T>
    inline bool isInImage_spherical(const T x, const T y)
    {
        return ( y >= 0 && y < nRows );
    };

    /*! Project 3D points XYZ according to the pinhole camera model. */
    inline cv::Point2f project2Image_pinhole(Eigen::Vector3f & xyz)
    {
        //Project the 3D point to the 2D plane
        float inv_transf_z = 1.f/xyz(2);
        // 2D coordinates of the transformed pixel(r,c) of frame 1
        float transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
        float transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
        cv::Point2f pixel(transformed_r, transformed_c);
        return pixel;
    };

    /*! Project 3D points XYZ according to the pinhole camera model. */
    inline cv::Point2f project2Image_spherical(Eigen::Vector3f & xyz)
    {
        //Project the 3D point to the 2D plane
        float dist = xyz.norm();
        float dist_inv = 1.f / dist;
        float phi = asin(xyz(1)*dist_inv);
        float theta = atan2(xyz(0),xyz(2));
        //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
        int transformed_r = (phi-phi_start)*pixel_angle_inv;
        int transformed_c = (half_width + theta*pixel_angle_inv) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);

        cv::Point2f pixel(transformed_r, transformed_c);
        return pixel;
    };

    /*! Project 3D points XYZ according to the pinhole camera model (3D -> 2D). */
    void project_pinhole(Eigen::MatrixXf & xyz, Eigen::MatrixXf & pixels);

    /*! Project 3D points XYZ according to the spherical camera model (3D -> 2D). */
    void project_spherical(Eigen::MatrixXf & xyz, Eigen::MatrixXf & pixels);

    /*! Project 3D points XYZ according to the pinhole camera model (3D -> 1D nearest neighbor). */
    void projectNN_pinhole(Eigen::MatrixXf & xyz, Eigen::MatrixXi & pixels, Eigen::MatrixXi &visible);

    /*! Project 3D points XYZ according to the spherical camera model (3D -> 1D nearest neighbor). */
    void projectNN_spherical(Eigen::MatrixXf & xyz, Eigen::MatrixXi & pixels, Eigen::MatrixXi &visible);

    /*! Compute the 3D points XYZ according to the pinhole camera model. */
    void reconstruct3D_pinhole(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels);

    void reconstruct3D_pinhole_saliency ( Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                          const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                          const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY,
                                          const float thres_saliency_gray, const float thres_saliency_depth
                                        );

    /*! Compute the unit sphere for the given spherical image dimmensions. This serves as a LUT to speed-up calculations. */
    void reconstruct3D_unitSphere();

    /*! Compute the 3D points XYZ by multiplying the unit sphere by the spherical depth image. */
    void reconstruct3D_spherical(const cv::Mat & depth_img, Eigen::MatrixXf & sphere_xyz, Eigen::VectorXi & validPixels);

    /*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
    void reconstruct3D_spherical_saliency( Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                           const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                           const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY,
                                           const float thres_saliency_gray, const float thres_saliency_depth
                                         ); // TODO extend this function to employ only depth

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


    /*! Project pixel spherical */
    inline void projectSphere(const int nCols, const int nRows, const float phi_FoV, const int c, const int r, const float depth,
                              const Eigen::Matrix4f & poseGuess,
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
        Eigen::Vector4f xyz = poseGuess*point3D;
        float depth_trg = sqrt(xyz(0)*xyz(0) + xyz(1)*xyz(1) + xyz(2)*xyz(2));

        //Project the 3D point to the S2 sphere
        float phi_trg = asin(xyz(2)/depth_trg);
        float theta_trg = atan2(-xyz(1),-xyz(2));
        transformed_r_int = round(phi_trg*nRows/phi_FoV + nRows/2);
        transformed_c_int = round(theta_trg*nCols/(2*PI));
    };

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_sphere(const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        const float dist = xyz_transf.norm();
        float dist2 = dist * dist;
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
    computeJacobian26_wT_sphere_inv(const Eigen::Vector3f & xyz_transf, const Eigen::Vector3f & xyz_orig, const Eigen::Matrix3f & rotation, const float & dist, const float & pixel_angle_inv, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        // The Jacobian of the spherical projection
        Eigen::Matrix<float,2,3> jacobianProj23;
        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz_transf(1)*xyz_transf(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
        jacobianProj23(0,0) = commonDer_c * xyz_transf(2);
        jacobianProj23(0,1) = 0;
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

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_pinhole(const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
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
    computeJacobian23_warp_sphere(const Eigen::Vector3f & xyz_transf, const float dist, const float pixel_angle_inv, Eigen::Matrix<float,2,3> &jacobianWarp)
    {
        // The Jacobian of the spherical projection
        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz_transf(1)*xyz_transf(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
        jacobianWarp(0,0) = commonDer_c * xyz_transf(2);
        jacobianWarp(0,1) = 0;
        jacobianWarp(0,2) =-commonDer_c * xyz_transf(0);
//        jacobianWarp(1,0) = commonDer_r * xyz_transf(0) * xyz_transf(1);
        jacobianWarp(1,1) =-commonDer_r * x2_z2;
//        jacobianWarp(1,2) = commonDer_r * xyz_transf(2) * xyz_transf(1);
        float commonDer_r_y = commonDer_r * xyz_transf(1);
        jacobianWarp(1,0) = commonDer_r_y * xyz_transf(0);
        jacobianWarp(1,2) = commonDer_r_y * xyz_transf(2);
    }

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
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wTTx_pinhole(const Eigen::Matrix4f & Rt, const Eigen::Vector3f & xyz, const Eigen::Vector3f & xyz_transf, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        Eigen::Matrix<float,2,3> jacobianWarp;
        Eigen::Matrix<float,3,6> jacobianRt;

        computeJacobian23_warp_pinhole(xyz_transf, jacobianWarp);
        computeJacobian36_Tx_p(Rt.block(0,0,3,3), xyz, jacobianRt);

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
        computeJacobian36_Tx_p(Rt.block(0,0,3,3), xyz, jacobianRt);

        jacobianWarpRt = jacobianWarp * jacobianRt;
    }

    /*! Warp the image according to a given geometric transformation. */
    //void warpImage ( const int pyrLevel, const Eigen::Matrix4f &poseGuess, costFuncType method );

};

/*! Compute the 2x6 jacobian matrices of the composition (warping+rigidTransformation) using the pinhole camera model. */
void computeJacobians26_pinhole(Eigen::MatrixXf & xyz_tf, Eigen::MatrixXf & jacobians_aligned);

/*! Compute the 2x6 jacobian matrices of the composition (warping+rigidTransformation) using the spherical camera model. */
void computeJacobians26_spherical(Eigen::MatrixXf & xyz_tf, Eigen::MatrixXf & jacobians_aligned);

inline void getJacobian_pinhole(Eigen::MatrixXf & jacobians_aligned, const int i, Eigen::Matrix<float,2,6> & jacobianWarpRt)
{
    jacobianWarpRt(0,0) = jacobians_aligned(i,0);
    jacobianWarpRt(1,0) = 0.f;
    jacobianWarpRt(0,1) = 0.f;
    jacobianWarpRt(1,1) = jacobians_aligned(i,1);
    jacobianWarpRt(0,2) = jacobians_aligned(i,2);
    jacobianWarpRt(1,2) = jacobians_aligned(i,3);
    jacobianWarpRt(0,3) = jacobians_aligned(i,4);
    jacobianWarpRt(1,3) = jacobians_aligned(i,5);
    jacobianWarpRt(0,4) = jacobians_aligned(i,6);
    jacobianWarpRt(1,4) = jacobians_aligned(i,7);
    jacobianWarpRt(0,5) = jacobians_aligned(i,8);
    jacobianWarpRt(1,5) = jacobians_aligned(i,9);
}

inline void getJacobian_spherical(Eigen::MatrixXf & jacobians_aligned, const int i, Eigen::Matrix<float,2,6> & jacobianWarpRt)
{
    jacobianWarpRt(0,0) = jacobians_aligned(i,0);
    jacobianWarpRt(1,0) = jacobians_aligned(i,1);
    jacobianWarpRt(0,1) = 0.f;
    jacobianWarpRt(1,1) = jacobians_aligned(i,2);
    jacobianWarpRt(0,2) = jacobians_aligned(i,3);
    jacobianWarpRt(1,2) = jacobians_aligned(i,4);
    jacobianWarpRt(0,3) = jacobians_aligned(i,5);
    jacobianWarpRt(1,3) = jacobians_aligned(i,6);
    jacobianWarpRt(0,4) = jacobians_aligned(i,7);
    jacobianWarpRt(1,4) = jacobians_aligned(i,8);
    jacobianWarpRt(0,5) = jacobians_aligned(i,9);
    jacobianWarpRt(1,5) = jacobians_aligned(i,10);
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
void computeJacobiansPhoto_pinhole(Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians, float *_gradX, float *_gradY);

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void computeJacobiansPhoto_spherical(Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians, float *_gradX, float *_gradY);

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
void computeJacobiansDepth_pinhole(Eigen::MatrixXf & xyz_tf, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians, float *_gradDepthX, float *_gradDepthY);

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void computeJacobiansDepth_spherical(Eigen::MatrixXf & xyz_tf, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians, float *_gradDepthX, float *_gradDepthY);

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
void computeJacobiansPhotoDepth_pinhole(Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & weights,
                                        Eigen::MatrixXf & jacobians_photo, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY);

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void computeJacobiansPhotoDepth_spherical ( Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & weights,
                                            Eigen::MatrixXf & jacobians_photo, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY);

///*! Compute the 3Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
//void computeJacobiansICP_pinhole(Eigen::MatrixXf & xyz_tf, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians);

///*! Compute the 3Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
//void computeJacobiansICP_spherical(Eigen::MatrixXf & xyz_tf, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians);

#endif
