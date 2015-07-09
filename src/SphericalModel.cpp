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

#include <SphericalModel.h>
#include <pcl/common/time.h>
#include <mrpt/utils/mrpt_macros.h>
//#include <ASSERT__.h>
//#include "/usr/local/include/eigen3/Eigen/Core"

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

#define ENABLE_OPENMP 0
#define PRINT_PROFILING 1
#define TEST_SIMD 1

using namespace std;
using namespace Eigen;

SphericalModel::SphericalModel()
{
}

/*! Scale the intrinsic calibration parameters according to the image resolution (i.e. the reduced resolution being used). */
void SphericalModel::scaleCameraParams(const int pyrLevel)
{
    //nCols = nCols / scaleFactor;
}

/*! Compute the unit sphere for the given spherical image dimmensions. This serves as a LUT to speed-up calculations. */
void SphericalModel::reconstruct3D_unitSphere()
{
    // Make LUT to store the values of the 3D points of the source sphere
    Eigen::MatrixXf unit_sphere;
    unit_sphere.resize(nRows*nCols,3);
    const float pixel_angle = 2*PI/nCols;
    std::vector<float> v_sinTheta(nCols);
    std::vector<float> v_cosTheta(nCols);
    for(int c=0; c < nCols; c++)
    {
        float theta = c*pixel_angle;
        v_sinTheta[c] = sin(theta);
        v_cosTheta[c] = cos(theta);
    }
    const float half_height = 0.5*nRows-0.5;
    for(int r=0; r < nRows; r++)
    {
        float phi = (half_height-r)*pixel_angle;
        float sin_phi = sin(phi);
        float cos_phi = cos(phi);

        for(int c=0;c<nCols;c++)
        {
            size_t i = r*nCols + c;
            unit_sphere(i,0) = sin_phi;
            unit_sphere(i,1) = -cos_phi*v_sinTheta[c];
            unit_sphere(i,2) = -cos_phi*v_cosTheta[c];
        }
    }
}


/*! Compute the 3D points XYZ by multiplying the unit sphere by the spherical depth image.
 * X -> Left
 * Z -> Up
 * Z -> Forward
 * In spherical coordinates Theata is in the range [-pi,pi) and Phi in [-pi/2,pi/2)
 */
void SphericalModel::reconstruct3D(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, VectorXi & validPixels) // , std::vector<int> & validPixels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    nRows = depth_img.rows;
    nCols = depth_img.cols;
    imgSize = nRows*nCols;

    pixel_angle = 2*PI/nCols;
    pixel_angle_inv = 1/pixel_angle;
    half_width = nCols/2 - 0.5f;
    phi_start = -(0.5f*nRows-0.5f)*pixel_angle;

    const float half_width = nCols/2;
    const float pixel_angle = 2*PI/nCols;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5*nRows-0.5)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    float *_depth = reinterpret_cast<float*>(depth_img.data);

    xyz.resize(imgSize,3);
    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);
    validPixels.resize(imgSize);
    //float *_valid_pt = reinterpret_cast<float*>(&validPixels(0));

    // Compute the Unit Sphere: store the values of the trigonometric functions
    Eigen::VectorXf v_sinTheta(nCols);
    Eigen::VectorXf v_cosTheta(nCols);
    float *sinTheta = &v_sinTheta[0];
    float *cosTheta = &v_cosTheta[0];
    for(int col_theta=-half_width; col_theta < half_width; ++col_theta)
    {
        //float theta = col_theta*pixel_angle;
        float theta = (col_theta+0.5f)*pixel_angle;
        *(sinTheta++) = sin(theta);
        *(cosTheta++) = cos(theta);
        //cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << endl;
    }
    size_t start_row = (nCols-nRows) / 2;
    Eigen::VectorXf v_sinPhi( v_sinTheta.block(start_row,0,nRows,1) );
    Eigen::VectorXf v_cosPhi( v_cosTheta.block(start_row,0,nRows,1) );

    // Test SSE
    Eigen::MatrixXf xyz2(imgSize,3);
    Eigen::VectorXi validPixels2(imgSize);
    for(int r=0; r < nRows;r++)
    {
        size_t i = r*nCols;
        for(int c=0; c < nCols;c++,i++)
        {
            float depth1 = _depth[i];
            if(min_depth_ < depth1 && depth1 < max_depth_) //Compute the jacobian only for the valid points
            {
                //cout << " depth1 " << depth1 << " phi " << phi << " v_sinTheta[c] " << v_sinTheta[c] << endl;
                validPixels2(i) = i;
                xyz2(i,0) = depth1 * v_cosPhi[r] * v_sinTheta[c];
                xyz(i,1) = depth1 * v_sinPhi[r];
                xyz(i,2) = depth1 * v_cosPhi[r] * v_cosTheta[c];
            }
            else
                validPixels2(i) = -1;
        }
    }

//Compute the 3D coordinates of the depth image
#if !(_SSE3) // # ifdef __SSE3__

    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(int r=0; r < nRows;r++)
    {
        size_t i = r*nCols;
        for(int c=0; c < nCols;c++,i++)
        {
            float depth1 = _depth[i];
            if(min_depth_ < depth1 && depth1 < max_depth_) //Compute the jacobian only for the valid points
            {
                //cout << " depth1 " << depth1 << " phi " << phi << " v_sinTheta[c] " << v_sinTheta[c] << endl;
                validPixels(i) = i;
                xyz(i,0) = depth1 * v_cosPhi[r] * v_sinTheta[c];
                xyz(i,1) = depth1 * v_sinPhi[r];
                xyz(i,2) = depth1 * v_cosPhi[r] * v_cosTheta[c];
//                _x[i] = depth1 * v_cosPhi[r] * v_sinTheta[c];
//                _y[i] = depth1 * v_sinPhi[r];
//                _z[i] = depth1 * v_cosPhi[r] * v_cosTheta[c];
                //cout << " xyz " << xyz [i].transpose() << " xyz_eigen " << xyz_eigen.block(c*nRows+r,0,1,3) << endl;
                //mrpt::system::pause();
            }
            else
                validPixels(i) = -1;
                //_valid_pt[i] = -1;
        }
    }

#else
//#elif !(_AVX) // # ifdef __AVX__
    cout << " reconstruct3D _SSE3 " << imgSize << " pts \n";

    ASSERT_(nCols % 4 == 0); // Make sure that the image columns are aligned
    ASSERT_(nRows % 2 == 0);

    __m128 _zero_ = _mm_set1_ps(0.f);
    __m128i _idx_zero_ = _mm_setr_epi32(0.f, 1.f, 2.f, 3.f);
    __m128i __minus_one = _mm_set1_epi32(-1);
    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
    if(imgSize > 1e5)
    {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
        //for(size_t i=0; i < __end; i++)
        for(int r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t i = r*nCols;
            for(int c=0; c < nCols; c+=4, i+=4)
            {
                __m128 __depth = _mm_load_ps(_depth+i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 __x = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 __y = _mm_mul_ps( __depth, sin_phi );
                __m128 __z = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+i, __x);
                _mm_store_ps(_y+i, __y);
                _mm_store_ps(_z+i, __z);

                __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) );
                //_mm_store_ps(_valid_pt+i, valid_depth_pts ); // Store 0 or -1

                __m128i __pos = _mm_set1_epi32(i);
                __m128i __idx = _mm_add_epi32(__pos, _idx_zero_);
                __m128i *_v = reinterpret_cast<__m128i*>(&validPixels(i));
                _mm_store_si128(_v, __idx);

                __m128i invalid_pts = _mm_cvtps_epi32( _mm_and_ps(_zero_, valid_depth_pts) );
    //            cout << "invalid_pts " << i  << " "
    //                 << _mm_extract_epi32(invalid_pts,0) << " " << _mm_extract_epi32(invalid_pts,1) << " " << _mm_extract_epi32(invalid_pts,2) << " " << _mm_extract_epi32(invalid_pts,3) << endl;
                char *_vv = reinterpret_cast<char*>(&validPixels(i));
                _mm_maskmoveu_si128(__minus_one, invalid_pts, _vv);

    //            for(int j=0; j < 4; j++)
    //            {
    //                const int jj = j;
    //                if(!valid_depth_pts[jj])
    //                    validPixels(i+j) = -1;
    //            }

            }
        }
    }
    else
    {
        for(int r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t i = r*nCols;
            for(int c=0; c < nCols; c+=4, i+=4)
            {
                __m128 __depth = _mm_load_ps(_depth+i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 __x = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 __y = _mm_mul_ps( __depth, sin_phi );
                __m128 __z = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+i, __x);
                _mm_store_ps(_y+i, __y);
                _mm_store_ps(_z+i, __z);

                __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) );
                //_mm_store_ps(_valid_pt+i, valid_depth_pts ); // Store 0 or -1

                __m128i __pos = _mm_set1_epi32(i);
                __m128i __idx = _mm_add_epi32(__pos, _idx_zero_);
                __m128i *_v = reinterpret_cast<__m128i*>(&validPixels(i));
                _mm_store_si128(_v, __idx);

                __m128i invalid_pts = _mm_cvtps_epi32( _mm_and_ps(_zero_, valid_depth_pts) );
    //            cout << "invalid_pts " << i  << " "
    //                 << _mm_extract_epi32(invalid_pts,0) << " " << _mm_extract_epi32(invalid_pts,1) << " " << _mm_extract_epi32(invalid_pts,2) << " " << _mm_extract_epi32(invalid_pts,3) << endl;
                char *_vv = reinterpret_cast<char*>(&validPixels(i));
                _mm_maskmoveu_si128(__minus_one, invalid_pts, _vv);

    //            for(int j=0; j < 4; j++)
    //            {
    //                const int jj = j;
    //                if(!valid_depth_pts[jj])
    //                    validPixels(i+j) = -1;
    //            }
            }
        }
    }
//    // Compute the transformation of those points which do not enter in a block
//    const Matrix3f rotation_transposed = Rt.block(0,0,3,3).transpose();
//    const Matrix<float,1,3> translation_transposed = Rt.block(0,3,3,1).transpose();
//    for(int i=__end; i < imgSize; i++)
//    {
//        ***********
//        output_pts.block(i,0,1,3) = input_pts.block(i,0,1,3) * rotation_transposed + translation_transposed;
//    }

#endif

    // Test SSE
    //cout << " Check result " << validPixels.size() << endl;
    for(int i=0; i < validPixels.size(); i++)
    {
        //cout << " Check result " << i << " " << pixels.size() << endl;
        ASSERT_( validPixels(i) == validPixels2(i) );
        ASSERT_( xyz(i,0) == xyz2(i,0) );
        ASSERT_( xyz(i,1) == xyz2(i,1) );
        ASSERT_( xyz(i,2) == xyz2(i,2) );
    }

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::reconstruct3D " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
void SphericalModel::reconstruct3D_saliency(Eigen::MatrixXf & xyz, VectorXi & validPixels,
                                            const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                            const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY,
                                            const float thres_saliency_gray, const float thres_saliency_depth
                                            ) // TODO extend this function to employ only depth
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    nRows = depth_img.rows;
    nCols = depth_img.cols;
    imgSize = nRows*nCols;

    pixel_angle = 2*PI/nCols;
    pixel_angle_inv = 1/pixel_angle;
    half_width = nCols/2 - 0.5f;
    phi_start = -(0.5f*nRows-0.5f)*pixel_angle;

    const float half_width = nCols/2;
    const float pixel_angle = 2*PI/nCols;

//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5*nRows-0.5)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    // Compute the Unit Sphere: store the values of the trigonometric functions
    Eigen::VectorXf v_sinTheta(nCols);
    Eigen::VectorXf v_cosTheta(nCols);
    float *sinTheta = &v_sinTheta[0];
    float *cosTheta = &v_cosTheta[0];
    for(int col_theta=-half_width; col_theta < half_width; ++col_theta)
    {
        //float theta = col_theta*pixel_angle;
        float theta = (col_theta+0.5f)*pixel_angle;
        *(sinTheta++) = sin(theta);
        *(cosTheta++) = cos(theta);
        //cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << endl;
    }
    const size_t start_row = (nCols-nRows) / 2;
    Eigen::VectorXf v_sinPhi( v_sinTheta.block(start_row,0,nRows,1) );
    Eigen::VectorXf v_cosPhi( v_cosTheta.block(start_row,0,nRows,1) );

    float *_depthGradXPyr = reinterpret_cast<float*>(depth_gradX.data);
    float *_depthGradYPyr = reinterpret_cast<float*>(depth_gradY.data);
    float *_grayGradXPyr = reinterpret_cast<float*>(intensity_gradX.data);
    float *_grayGradYPyr = reinterpret_cast<float*>(intensity_gradY.data);

    //Compute the 3D coordinates of the 3D points in source frame
    validPixels.resize(imgSize);
    xyz.resize(imgSize,3);
    float *_depth = reinterpret_cast<float*>(depth_img.data);

#if !(_SSE3) // # ifdef __SSE3__

    size_t count_valid_pixels = 0;
    for(size_t r=0;r<nRows;r++)
    {
        size_t i = r*nCols;
        for(size_t c=0;c<nCols;c++,i++)
        {
            if(min_depth_ < _depth[i] && _depth[i] < max_depth_) //Compute the jacobian only for the valid points
                if( fabs(_grayGradXPyr[i]) > thres_saliency_gray || fabs(_grayGradYPyr[i]) > thres_saliency_gray ||
                    fabs(_depthGradXPyr[i]) > thres_saliency_depth || fabs(_depthGradYPyr[i]) > thres_saliency_depth )
                {
                    validPixels(count_valid_pixels) = i;
                    //cout << " depth " << _depth[i] << " validPixels " << validPixels(count_valid_pixels) << " count_valid_pixels " << count_valid_pixels << endl;
                    xyz(count_valid_pixels,0) = _depth[i] * v_cosPhi[r] * v_sinTheta[c];
                    xyz(count_valid_pixels,1) = _depth[i] * v_sinPhi[r];
                    xyz(count_valid_pixels,2) = _depth[i] * v_cosPhi[r] * v_cosTheta[c];
                    //cout << " xyz " << xyz.block(count_valid_pixels,0,1,3) << endl;
                    ++count_valid_pixels;
                    //mrpt::system::pause();
                }
        }
    }
    size_t valid_pixels_aligned = count_valid_pixels - count_valid_pixels % 4;
    validPixels.conservativeResize(valid_pixels_aligned);
    xyz.conservativeResize(valid_pixels_aligned,3);
    //validPixels = validPixels.block(0,0,count_valid_pixels,1);
    //xyz = xyz.block(0,0,count_valid_pixels,3);

#else
//#elif !(_AVX) // # ifdef __AVX__
    cout << " reconstruct3D _SSE3 " << imgSize << " pts \n";

    //Compute the 3D coordinates of the pij of the source frame
    Eigen::MatrixXf xyz_tmp(imgSize,3);
    VectorXi validPixels_tmp(imgSize);
    float *_x = &xyz_tmp(0,0);
    float *_y = &xyz_tmp(0,1);
    float *_z = &xyz_tmp(0,2);

    float *_valid_pt = reinterpret_cast<float*>(&validPixels_tmp(0));
    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
    __m128 _depth_saliency_ = _mm_set1_ps(thres_saliency_depth);
    __m128 _gray_saliency_ = _mm_set1_ps(thres_saliency_gray);
    __m128 _depth_saliency_neg = _mm_set1_ps(-thres_saliency_depth);
    __m128 _gray_saliency_neg = _mm_set1_ps(-thres_saliency_gray);

    if(imgSize > 1e5)
    {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
        for(int r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t i = r*nCols;
            for(int c=0; c < nCols; c+=4, i+=4)
            {
                __m128 __depth = _mm_load_ps(_depth+i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 __x = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 __y = _mm_mul_ps( __depth, sin_phi );
                __m128 __z = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+i, __x);
                _mm_store_ps(_y+i, __y);
                _mm_store_ps(_z+i, __z);

                //__m128 mask = _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) );
                //mi cmpeq_epi8(mi a,mi b)

                //_mm_store_ps(_valid_pt+i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) ) );
                __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) );
                __m128 __gradDepthX = _mm_load_ps(_depthGradXPyr+i);
                __m128 __gradDepthY = _mm_load_ps(_depthGradYPyr+i);
                __m128 __gradGrayX = _mm_load_ps(_grayGradXPyr+i);
                __m128 __gradGrayY = _mm_load_ps(_grayGradYPyr+i);
                __m128 salient_pts = _mm_or_ps( _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(__gradDepthX, _depth_saliency_), _mm_cmplt_ps(__gradDepthX, _depth_saliency_neg) ),
                                                           _mm_or_ps( _mm_cmpgt_ps(__gradDepthY, _depth_saliency_), _mm_cmplt_ps(__gradDepthY, _depth_saliency_neg) ) ),
                                                _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( __gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayX, _gray_saliency_neg ) ),
                                                           _mm_or_ps( _mm_cmpgt_ps( __gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayY, _gray_saliency_neg ) ) ) );
                _mm_store_ps(_valid_pt+i, _mm_and_ps( valid_depth_pts, salient_pts ) );
            }
        }
    }
    else
    {
        for(int r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t i = r*nCols;
            for(int c=0; c < nCols; c+=4, i+=4)
            {
                __m128 __depth = _mm_load_ps(_depth+i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 __x = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 __y = _mm_mul_ps( __depth, sin_phi );
                __m128 __z = _mm_mul_ps( __depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+i, __x);
                _mm_store_ps(_y+i, __y);
                _mm_store_ps(_z+i, __z);

                //_mm_store_ps(_valid_pt+i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) ) );
                __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) );
                __m128 __gradDepthX = _mm_load_ps(_depthGradXPyr+i);
                __m128 __gradDepthY = _mm_load_ps(_depthGradYPyr+i);
                __m128 __gradGrayX = _mm_load_ps(_grayGradXPyr+i);
                __m128 __gradGrayY = _mm_load_ps(_grayGradYPyr+i);
                __m128 salient_pts = _mm_or_ps( _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(__gradDepthX, _depth_saliency_), _mm_cmplt_ps(__gradDepthX, _depth_saliency_neg) ), _mm_or_ps( _mm_cmpgt_ps(__gradDepthY, _depth_saliency_), _mm_cmplt_ps(__gradDepthY, _depth_saliency_neg) ) ),
                                                _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( __gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayX, _gray_saliency_neg ) ), _mm_or_ps( _mm_cmpgt_ps( __gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayY, _gray_saliency_neg ) ) ) );
                _mm_store_ps(_valid_pt+i, _mm_and_ps( valid_depth_pts, salient_pts ) );
            }
        }
    }
    // Select only the salient points
    size_t count_valid_pixels = 0;
    //cout << " " << LUT_xyz_source.rows() << " " << xyz_tmp.rows() << " size \n";
    for(size_t i=0; i < imgSize; i++)
    {
        if( validPixels_tmp(i) )
        {
            validPixels(count_valid_pixels) = i;
            xyz.block(count_valid_pixels,0,1,3) = xyz_tmp.block(i,0,1,3);
            ++count_valid_pixels;
        }
    }
    size_t salient_pixels_aligned = count_valid_pixels - count_valid_pixels % 4;
    //cout << salient_pixels_aligned << " salient_pixels_aligned \n";
    validPixels.conservativeResize(salient_pixels_aligned);
    xyz.conservativeResize(salient_pixels_aligned,3);

#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::reconstruct3D " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Project 3D points XYZ according to the spherical camera model. */
void SphericalModel::project(const Eigen::MatrixXf & xyz, Eigen::MatrixXf & pixels, VectorXi & visible)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    pixels.resize(xyz.rows(),2);
    float *_r = &pixels(0,0);
    float *_c = &pixels(0,1);
    visible.resize(xyz.rows());
    float *_v = reinterpret_cast<float*>(&visible(0));

    const float *_x = &xyz(0,0);
    const float *_y = &xyz(0,1);
    const float *_z = &xyz(0,2);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < pixels.size(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image(pt_xyz);
        pixels(i,0) = warped_pixel.y;
        pixels(i,1) = warped_pixel.x;
    }

#else

    ASSERT_(__SSE4_1__); // For _mm_extract_epi32

    __m128 _phi_start = _mm_set1_ps(phi_start);
    __m128 _half_width = _mm_set1_ps(half_width);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    //__m128 _nCols = _mm_set1_ps(nCols);
    __m128 _nRows = _mm_set1_ps(nRows);
    __m128 _zero = _mm_set1_ps(0.f);
    for(int i=0; i < pixels.size(); i+=4)
    {
        //__m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);

        union // float_sse
        {
            __m128 sse;    // SSE 4 x float vector
            float m128_f32[4];  // scalar array of 4 floats
        } y_div_z;
        y_div_z.sse = _mm_div_ps( __y, __z );
        //__m128 _y_z = _mm_div_ps( __y, __z );
        //float *x_z = reinterpret_cast<float*>(&_x_z);
        float theta[4];
        float phi[4];
        for(int j=0; j < 4; j++)
        {
            //phi[j] = asin( _mm_extract_epi32(_y_z,j) );
            phi[j] = asin(y_div_z.m128_f32[j]);
            theta[j] = atan2(_x[i+j], _z[i+j]);
        }
        __m128 _phi = _mm_load_ps(phi);
        __m128 _theta = _mm_load_ps(theta);

        __m128 __r = _mm_mul_ps( _mm_sub_ps(_phi, _phi_start ), _pixel_angle_inv );
        __m128 __c = _mm_add_ps( _mm_mul_ps(_theta, _pixel_angle_inv ), _half_width );

        _mm_store_ps(_r+i, __r);
        _mm_store_ps(_c+i, __c);

        __m128 __v = _mm_and_ps( _mm_cmplt_ps(_zero, __r), _mm_cmplt_ps(__r, _nRows) );
                                    //,_mm_and_ps( _mm_cmplt_ps(_zero, __c), _mm_cmplt_ps(__c, _nCols) ) );
        _mm_store_ps(_v+i, __v);
    }

    for(int i=0; i < pixels.size(); i++)
        ASSERT_(pixels(i,1) < nCols);

#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::project " << xyz.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Project 3D points XYZ according to the spherical camera model. */
void SphericalModel::projectNN(const Eigen::MatrixXf & xyz, VectorXi & pixels, VectorXi & visible)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    pixels.resize(xyz.rows());
    visible.resize(xyz.rows());
    //int *_p = &pixels(0);
    //float *_v = reinterpret_cast<float*>(&visible(0));
    const float *_x = &xyz(0,0);
    const float *_y = &xyz(0,1);
    const float *_z = &xyz(0,2);

#if TEST_SIMD
    // Test SSE
    Eigen::VectorXi pixels2(xyz.rows());
    Eigen::VectorXi visible2(xyz.rows());
    for(int i=0; i < pixels.size(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image(pt_xyz);
        int r_transf = round(warped_pixel.y);
        int c_transf = round(warped_pixel.x);
        // cout << i << " Pixel transform " << i/nCols << " " << i%nCols << " " << r_transf << " " << c_transf << endl;
        pixels2(i) = r_transf * nCols + c_transf;
        visible2(i) = isInImage(r_transf, c_transf) ? 1 : 0;
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < pixels.size(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image(pt_xyz);
        int r_transf = round(warped_pixel.y);
        int c_transf = round(warped_pixel.x);
        // cout << i << " Pixel transform " << i/nCols << " " << i%nCols << " " << r_transf << " " << c_transf << endl;
        pixels(i) = r_transf * nCols + c_transf;
        visible(i) = isInImage(r_transf, c_transf) ? 1 : 0;
    }

#else

    ASSERT_(__SSE4_1__); // For _mm_extract_epi32

    __m128i _nCols = _mm_set1_epi32(nCols);
    __m128i _nRows = _mm_set1_epi32(nRows);
    __m128i _minus_one = _mm_set1_epi32(-1);
    //__m128 _nCols = _mm_set1_ps(nCols);
    __m128 _phi_start = _mm_set1_ps(phi_start);
    __m128 _half_width = _mm_set1_ps(half_width);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(int i=0; i < pixels.size(); i+=4)
    {
        //__m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);

        union // float_sse
        {
            __m128 sse;    // SSE 4 x float vector
            float m128_f32[4];  // scalar array of 4 floats
        } y_div_z;
        y_div_z.sse = _mm_div_ps( __y, __z );
        //__m128 _y_z = _mm_div_ps( __y, __z );
        //float *x_z = reinterpret_cast<float*>(&_x_z);
        float theta[4];
        float phi[4];
        for(int j=0; j < 4; j++)
        {
            //phi[j] = asin( _mm_extract_epi32(_y_z,j) );
            phi[j] = asin(y_div_z.m128_f32[j]);
            theta[j] = atan2(_x[i+j], _z[i+j]);
        }
        __m128 _phi = _mm_load_ps(phi);
        __m128 _theta = _mm_load_ps(theta);

        __m128 __r = _mm_mul_ps( _mm_sub_ps(_phi, _phi_start ), _pixel_angle_inv );
        __m128 __c = _mm_add_ps( _mm_mul_ps(_theta, _pixel_angle_inv ), _half_width );
        //__m128 __p = _mm_add_epi32( _mm_cvtps_epi32(_nCols, __r ), __c);

        __m128i __r_int = _mm_cvtps_epi32(__r);
        __m128i __c_int = _mm_cvtps_epi32(__c);
//        for(int j=0; j < 4; j++)
//        {
//            const int jj = j;
//            cout << j << " __r " << __r[jj] << endl;
//        }
//        cout << "__r_int " << i << " " << _mm_extract_epi32(__r_int,0) << " " << _mm_extract_epi32(__r_int,1) << " " << _mm_extract_epi32(__r_int,2) << " " << _mm_extract_epi32(__r_int,3) << endl;
//        cout << "__c_int " << i << " " << _mm_extract_epi32(__c_int,0) << " " << _mm_extract_epi32(__c_int,1) << " " << _mm_extract_epi32(__c_int,2) << " " << _mm_extract_epi32(__c_int,3) << endl;
//        //cout << "__r_int " << i << " " << _mm_extract_epi32(__r_int,0) << " " << _mm_extract_epi32(__r_int,1) << " " << _mm_extract_epi32(__r_int,2) << " " << _mm_extract_epi32(__r_int,3) << endl;

        __m128i __p = _mm_add_epi32( _mm_mullo_epi32(_nCols, __r_int ), __c_int );
//        cout << "Compute warped pixel " << i  << " "
//             << _mm_extract_epi32(__p,0) << " " << _mm_extract_epi32(__p,1) << " " << _mm_extract_epi32(__p,2) << " " << _mm_extract_epi32(__p,3) << endl;

        __m128i __v = _mm_and_si128( _mm_cmplt_epi32(_minus_one, __r_int), _mm_cmplt_epi32(__r_int, _nRows) );
//        __m128i valid_row = _mm_and_si128( _mm_cmplt_epi32(_minus_one, __r_int), _mm_cmplt_epi32(__r_int, _nRows) );
//        __m128i valid_col = _mm_and_si128( _mm_cmplt_epi32(_minus_one, __c_int), _mm_cmplt_epi32(__c_int, _nCols) );
//        cout << "valid_row " << " " << _mm_extract_epi32(valid_row,0) << " " << _mm_extract_epi32(valid_row,1) << " " << _mm_extract_epi32(valid_row,2) << " " << _mm_extract_epi32(valid_row,3) << endl;
//        cout << "valid_col " << " " << _mm_extract_epi32(valid_col,0) << " " << _mm_extract_epi32(valid_col,1) << " " << _mm_extract_epi32(valid_col,2) << " " << _mm_extract_epi32(valid_col,3) << endl;

        __m128i *_p = reinterpret_cast<__m128i*>(&pixels(i));
        _mm_store_si128(_p, __p);
        __m128i *_v = reinterpret_cast<__m128i*>(&visible(i));
        _mm_store_si128(_v, __v);
    }

    for(int i=0; i < pixels.size(); i++)
        ASSERT_(pixels(i,1) < nCols);

#endif

#if TEST_SIMD
    // Test SSE
    cout << " Check result " << endl;
    for(int i=0; i < pixels.size(); i++)
    {
        //cout << " Check result " << i << " " << pixels.size() << endl;
        ASSERT_( visible(i) == visible2(i) );
        ASSERT_( pixels(i) == pixels2(i) );
    }
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::projectNN " << xyz.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

///*! Compute the 2x6 jacobian matrices of the composition (warping+rigidTransformation) using the spherical camera model. */
//void SphericalModel::computeJacobians26(Eigen::MatrixXf & xyz_tf, Eigen::MatrixXf & jacobians_aligned)
//{
//    jacobians_aligned.resize(xyz_tf.rows(), 11); // It has 10 columns instead of 10 because the positions (1,0) and (0,1) of the 2x6 jacobian are alwatys 0

//    float *_x = &xyz_tf(0,0);
//    float *_y = &xyz_tf(0,1);
//    float *_z = &xyz_tf(0,2);

//#if !(_SSE3) // # ifdef !__SSE3__

////    #if ENABLE_OPENMP
////    #pragma omp parallel for
////    #endif
//    for(int i=0; i < xyz_tf.rows(); i++)
//    {
//        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
//        float inv_transf_z = 1.0/xyz_transf(2);

//        float dist = xyz_transf.norm();
//        float dist2 = dist * dist;
//        float x2_z2 = dist2 - xyz_transf(1)*xyz_transf(1);
//        float x2_z2_sqrt = sqrt(x2_z2);
//        float commonDer_c = pixel_angle_inv / x2_z2;
//        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );

//        jacobians_aligned(i,0) = commonDer_c * xyz_transf(2);
//        //jacobianWarpRt(0,1) = 0.f;
//        jacobians_aligned(i,3) = -commonDer_c * xyz_transf(0);
//    //        jacobianWarpRt(1,0) = commonDer_r * xyz_transf(0) * xyz_transf(1);
//        jacobians_aligned(i,2) =-commonDer_r * x2_z2;
//    //        jacobianWarpRt(1,2) = commonDer_r * xyz_transf(2) * xyz_transf(1);
//        float commonDer_r_y = commonDer_r * xyz_transf(1);
//        jacobians_aligned(i,1) = commonDer_r_y * xyz_transf(0);
//        jacobians_aligned(i,4) = commonDer_r_y * xyz_transf(2);

//        jacobians_aligned(i,5) = jacobians_aligned(i,3) * xyz_transf(1);
//        jacobians_aligned(i,7) = jacobians_aligned(i,0) * xyz_transf(2) - jacobians_aligned(i,3) * xyz_transf(0);
//        jacobians_aligned(i,9) =-jacobians_aligned(i,0) * xyz_transf(1);
//        jacobians_aligned(i,6) =-jacobians_aligned(i,2) * xyz_transf(2) + jacobians_aligned(i,4) * xyz_transf(1);
//        jacobians_aligned(1,8) = jacobians_aligned(i,1) * xyz_transf(2) - jacobians_aligned(i,4) * xyz_transf(0);
//        jacobians_aligned(1,10) =-jacobians_aligned(i,1) * xyz_transf(1) + jacobians_aligned(i,2) * xyz_transf(0);
//    }

//#else
//    // Pointers to the jacobian elements
//    float *_j00 = &jacobians_aligned(0,0);
//    float *_j10 = &jacobians_aligned(0,1);
//    float *_j11 = &jacobians_aligned(0,2);
//    float *_j02 = &jacobians_aligned(0,3);
//    float *_j12 = &jacobians_aligned(0,4);
//    float *_j03 = &jacobians_aligned(0,5);
//    float *_j13 = &jacobians_aligned(0,6);
//    float *_j04 = &jacobians_aligned(0,7);
//    float *_j14 = &jacobians_aligned(0,8);
//    float *_j05 = &jacobians_aligned(0,9);
//    float *_j15 = &jacobians_aligned(0,10);

//    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
//    for(int i=0; i < xyz_tf.rows(); i+=4)
//    {
//        __m128 __x = _mm_load_ps(_x+i);
//        __m128 __y = _mm_load_ps(_y+i);
//        __m128 __z = _mm_load_ps(_z+i);
//        __m128 _x2_z2 = _mm_add_ps(_mm_mul_ps(__x, __x), _mm_mul_ps(__z, __z) );
//        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
//        __m128 _dist2 = _mm_add_ps(_mm_mul_ps(__y, __y), _x2_z2 );
//        __m128 _dist = _mm_sqrt_ps(_dist2);
//        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
//        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
//        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, __y);

//        __m128 __j00 = _mm_mul_ps(_commonDer_c, __z);
//        __m128 __j10 = _mm_mul_ps(_commonDer_r_y, __x);
//        __m128 __j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
//        __m128 __j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, __x), _mm_set1_ps(-0.f));
//        __m128 __j12 = _mm_mul_ps(_commonDer_r_y, __z );

//        _mm_store_ps(_j00+i, __j00 );
//        _mm_store_ps(_j10+i, __j10 );
//        _mm_store_ps(_j11+i, __j11 );
//        _mm_store_ps(_j02+i, __j02 );
//        _mm_store_ps(_j12+i, __j12 );
//        _mm_store_ps(_j03+i, _mm_mul_ps( __j02, __y ) );
//        _mm_store_ps(_j13+i, _mm_sub_ps(_mm_mul_ps(__j12, __y), _mm_mul_ps(__j11, __z) ) );
//        _mm_store_ps(_j04+i, _mm_sub_ps(_mm_mul_ps(__j00, __z), _mm_mul_ps(__j02, __x) ) );
//        _mm_store_ps(_j14+i, _mm_sub_ps(_mm_mul_ps(__j10, __z), _mm_mul_ps(__j12, __x) ) );
//        _mm_store_ps(_j05+i, _mm_xor_ps( _mm_mul_ps(__j00, __y), _mm_set1_ps(-0.f) ) );
//        _mm_store_ps(_j15+i, _mm_sub_ps(_mm_mul_ps(__j11, __x), _mm_mul_ps(__j10, __y) ) );
//    }
//#endif
//}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void SphericalModel::computeJacobiansPhoto(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_photo, float *_grayGradX, float *_grayGradY)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    jacobians_photo.resize(xyz_tf.rows(), 6);
    const float *_x = &xyz_tf(0,0);
    const float *_y = &xyz_tf(0,1);
    const float *_z = &xyz_tf(0,2);
    const float *_weight = &weights(0,0);

#if TEST_SIMD
    Eigen::MatrixXf jacobians_photo2(xyz_tf.rows(), 6);
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(xyz_transf, jacobianWarpRt);
        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo2.block(i,0,1,6) = ((sqrt(weights(i)) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(xyz_transf, jacobianWarpRt);
        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo.block(i,0,1,6) = ((sqrt(weights(i)) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;
    }

#else
    // Pointers to the jacobian elements
    float *_j0_gray = &jacobians_photo(0,0);
    float *_j1_gray = &jacobians_photo(0,1);
    float *_j2_gray = &jacobians_photo(0,2);
    float *_j3_gray = &jacobians_photo(0,3);
    float *_j4_gray = &jacobians_photo(0,4);
    float *_j5_gray = &jacobians_photo(0,5);

    __m128 __stdDevInv = _mm_set1_ps(stdDevPhoto_inv);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(int i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 __gradX = _mm_load_ps(_grayGradX+i);
        __m128 __gradY = _mm_load_ps(_grayGradY+i);
        __m128 __weight = _mm_load_ps(_weight+i);
        __m128 __weight_stdDevInv = _mm_mul_ps(__stdDevInv, __weight);
        __m128 __gradX_weight = _mm_mul_ps(__weight_stdDevInv, __gradX);
        __m128 __gradY_weight = _mm_mul_ps(__weight_stdDevInv, __gradY);

        __m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);
        __m128 _x2_z2 = _mm_add_ps(_mm_mul_ps(__x, __x), _mm_mul_ps(__z, __z) );
        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
        __m128 _dist2 = _mm_add_ps(_mm_mul_ps(__y, __y), _x2_z2 );
        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, __y);

        __m128 __j00 = _mm_mul_ps(_commonDer_c, __z);
        __m128 __j10 = _mm_mul_ps(_commonDer_r_y, __x);
        __m128 __j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
        //__m128 __j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, __x), _mm_set1_ps(-0.f));
        __m128 __j02_neg = _mm_mul_ps(_commonDer_c, __x);
        __m128 __j12 = _mm_mul_ps(_commonDer_r_y, __z );
        __m128 __j03_neg = _mm_mul_ps( __j02_neg, __y );
        __m128 __j13 = _mm_sub_ps(_mm_mul_ps(__j12, __y), _mm_mul_ps(__j11, __z) );
        __m128 __j04 = _mm_add_ps(_mm_mul_ps(__j00, __z), _mm_mul_ps(__j02_neg, __x) );
        __m128 __j14 = _mm_sub_ps(_mm_mul_ps(__j10, __z), _mm_mul_ps(__j12, __x) );
        __m128 __j05_neg = _mm_mul_ps(__j00, __y);
        __m128 __j15 = _mm_sub_ps(_mm_mul_ps(__j11, __x), _mm_mul_ps(__j10, __y) );

        _mm_store_ps(_j0_gray+i, _mm_add_ps(_mm_mul_ps(__gradX_weight, __j00), _mm_mul_ps(__gradY_weight, __j10) ) );
        _mm_store_ps(_j1_gray+i, _mm_mul_ps(__gradY_weight, __j11) );
        _mm_store_ps(_j2_gray+i, _mm_sub_ps(_mm_mul_ps(__gradY_weight, __j12), _mm_mul_ps(__gradX_weight, __j02_neg) ) );
        _mm_store_ps(_j3_gray+i, _mm_sub_ps(_mm_mul_ps(__gradY_weight, __j13), _mm_mul_ps(__gradX_weight, __j03_neg) ) );
        _mm_store_ps(_j4_gray+i, _mm_add_ps(_mm_mul_ps(__gradY_weight, __j14), _mm_mul_ps(__gradX_weight, __j04) ) );
        _mm_store_ps(_j5_gray+i, _mm_sub_ps(_mm_mul_ps(__gradY_weight, __j15), _mm_mul_ps(__gradX_weight, __j05_neg) ) );
    }
#endif

#if TEST_SIMD
    // Test SSE
    for(int i=0; i < xyz_tf.rows(); i++)
        for(size_t j=0; j < 6; j++)
            ASSERT_(jacobians_photo(i,j) == jacobians_photo2(i,j));
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::computeJacobiansPhoto " << xyz_tf.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void SphericalModel::computeJacobiansDepth(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    jacobians_depth.resize(xyz_tf.rows(), 6);
    const float *_x = &xyz_tf(0,0);
    const float *_y = &xyz_tf(0,1);
    const float *_z = &xyz_tf(0,2);
    const float *_weight = &weights(0,0);
    const float *_stdDevInv = &stdDevError_inv(0);

#if TEST_SIMD
    // Test SSE
    Eigen::MatrixXf jacobians_depth2(xyz_tf.rows(), 6);
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(xyz_transf, jacobianWarpRt);
        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / xyz_tf.norm()) * xyz_transf.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth.block(i,0,1,6) = (sqrt(weights(i)) * stdDevError_inv(i) * (depth_gradient) * jacobianWarpRt - jacobian16_depthT);
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(xyz_transf, jacobianWarpRt);
        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / xyz_tf.norm()) * xyz_transf.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth.block(i,0,1,6) = (sqrt(weights(i)) * stdDevError_inv(i) * (depth_gradient) * jacobianWarpRt - jacobian16_depthT);
    }

#else
    // Pointers to the jacobian elements
    float *_j0_depth = &jacobians_depth(0,0);
    float *_j1_depth = &jacobians_depth(0,1);
    float *_j2_depth = &jacobians_depth(0,2);
    float *_j3_depth = &jacobians_depth(0,3);
    float *_j4_depth = &jacobians_depth(0,4);
    float *_j5_depth = &jacobians_depth(0,5);

    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(int i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 __gradX = _mm_load_ps(_depthGradX+i);
        __m128 __gradY = _mm_load_ps(_depthGradY+i);
        __m128 __weight = _mm_load_ps(_weight+i);
        __m128 __stdDevInv = _mm_load_ps(_stdDevInv+i);
        __m128 __weight_stdDevDepthInv = _mm_mul_ps(__stdDevInv, __weight);
        __m128 __gradDepthX_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradX);
        __m128 __gradDepthY_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradY);

        __m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);
        __m128 _x2_z2 = _mm_add_ps(_mm_mul_ps(__x, __x), _mm_mul_ps(__z, __z) );
        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
        __m128 _dist2 = _mm_add_ps(_mm_mul_ps(__y, __y), _x2_z2 );
        __m128 _dist = _mm_sqrt_ps(_dist2);
        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, __y);

        __m128 __j00 = _mm_mul_ps(_commonDer_c, __z);
        __m128 __j10 = _mm_mul_ps(_commonDer_r_y, __x);
        __m128 __j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
        //__m128 __j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, __x), _mm_set1_ps(-0.f));
        __m128 __j02_neg = _mm_mul_ps(_commonDer_c, __x);
        __m128 __j12 = _mm_mul_ps(_commonDer_r_y, __z );
        __m128 __j03_neg = _mm_mul_ps( __j02_neg, __y );
        __m128 __j13 = _mm_sub_ps(_mm_mul_ps(__j12, __y), _mm_mul_ps(__j11, __z) );
        __m128 __j04 = _mm_add_ps(_mm_mul_ps(__j00, __z), _mm_mul_ps(__j02_neg, __x) );
        __m128 __j14 = _mm_sub_ps(_mm_mul_ps(__j10, __z), _mm_mul_ps(__j12, __x) );
        __m128 __j05_neg = _mm_mul_ps(__j00, __y);
        __m128 __j15 = _mm_sub_ps(_mm_mul_ps(__j11, __x), _mm_mul_ps(__j10, __y) );

        _mm_store_ps(_j0_depth+i, _mm_sub_ps(_mm_add_ps(_mm_mul_ps(__gradDepthX_weight, __j00), _mm_mul_ps(__gradDepthY_weight, __j10) ), _mm_mul_ps(__weight_stdDevDepthInv, _mm_div_ps(__x, _dist) ) ) );
        _mm_store_ps(_j1_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j11), _mm_mul_ps(__weight_stdDevDepthInv, _mm_div_ps( __y, _dist ) ) ) );
        _mm_store_ps(_j2_depth+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j12), _mm_mul_ps(__gradDepthX_weight, __j02_neg) ), _mm_mul_ps(__weight_stdDevDepthInv, _mm_div_ps(__z, _dist) ) ) );
        _mm_store_ps(_j3_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j13), _mm_mul_ps(__gradDepthX_weight, __j03_neg) ) );
        _mm_store_ps(_j4_depth+i, _mm_add_ps(_mm_mul_ps(__gradDepthY_weight, __j14), _mm_mul_ps(__gradDepthX_weight, __j04) ) );
        _mm_store_ps(_j5_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j15), _mm_mul_ps(__gradDepthX_weight, __j05_neg) ) );
    }
#endif

#if TEST_SIMD
    // Test SSE
    for(int i=0; i < xyz_tf.rows(); i++)
        for(size_t j=0; j < 6; j++)
            ASSERT_(jacobians_depth(i,j) == jacobians_depth2(i,j));
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::computeJacobiansPhoto " << xyz_tf.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void SphericalModel::computeJacobiansPhotoDepth(const Eigen::MatrixXf & xyz_tf, const float stdDevPhoto_inv, const Eigen::VectorXf & stdDevError_inv, const Eigen::VectorXf & weights,
                                                Eigen::MatrixXf & jacobians_photo, Eigen::MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    jacobians_photo.resize(xyz_tf.rows(), 6);
    jacobians_depth.resize(xyz_tf.rows(), 6);
    const float *_x = &xyz_tf(0,0);
    const float *_y = &xyz_tf(0,1);
    const float *_z = &xyz_tf(0,2);
    const float *_weight = &weights(0,0);
    const float *_stdDevInv = &stdDevError_inv(0);

#if TEST_SIMD
    // Test SSE
    Eigen::MatrixXf jacobians_photo2(xyz_tf.rows(), 6);
    Eigen::MatrixXf jacobians_depth2(xyz_tf.rows(), 6);
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(xyz_transf, jacobianWarpRt);

        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo.block(i,0,1,6) = ((sqrt(weights(i)) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;

        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / xyz_tf.norm()) * xyz_transf.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth.block(i,0,1,6) = (sqrt(weights(i)) * stdDevError_inv(i) * (depth_gradient) * jacobianWarpRt - jacobian16_depthT);
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(xyz_transf, jacobianWarpRt);

        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo.block(i,0,1,6) = ((sqrt(weights(i)) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;

        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / xyz_tf.norm()) * xyz_transf.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth.block(i,0,1,6) = (sqrt(weights(i)) * stdDevError_inv(i) * (depth_gradient) * jacobianWarpRt - jacobian16_depthT);
    }

#else
    // Pointers to the jacobian elements
    float *_j0_gray = &jacobians_photo(0,0);
    float *_j1_gray = &jacobians_photo(0,1);
    float *_j2_gray = &jacobians_photo(0,2);
    float *_j3_gray = &jacobians_photo(0,3);
    float *_j4_gray = &jacobians_photo(0,4);
    float *_j5_gray = &jacobians_photo(0,5);

    float *_j0_depth = &jacobians_depth(0,0);
    float *_j1_depth = &jacobians_depth(0,1);
    float *_j2_depth = &jacobians_depth(0,2);
    float *_j3_depth = &jacobians_depth(0,3);
    float *_j4_depth = &jacobians_depth(0,4);
    float *_j5_depth = &jacobians_depth(0,5);

    __m128 __stdDevPhotoInv = _mm_set1_ps(stdDevPhoto_inv);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(int i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 __weight = _mm_load_ps(_weight+i);

        __m128 __gradGrayX = _mm_load_ps(_grayGradX+i);
        __m128 __gradGrayY = _mm_load_ps(_grayGradY+i);
        __m128 __weight_stdDevPhotoInv = _mm_mul_ps(__stdDevPhotoInv, __weight);
        __m128 __gradX_weight = _mm_mul_ps(__weight_stdDevPhotoInv, __gradGrayX);
        __m128 __gradY_weight = _mm_mul_ps(__weight_stdDevPhotoInv, __gradGrayY);

        __m128 __gradX = _mm_load_ps(_depthGradX+i);
        __m128 __gradY = _mm_load_ps(_depthGradY+i);
        __m128 __stdDevDepthInv = _mm_load_ps(_stdDevInv+i);
        __m128 __weight_stdDevDepthInv = _mm_mul_ps(__stdDevDepthInv, __weight);
        __m128 __gradDepthX_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradX);
        __m128 __gradDepthY_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradY);

        __m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);
        __m128 _x2_z2 = _mm_add_ps(_mm_mul_ps(__x, __x), _mm_mul_ps(__z, __z) );
        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
        __m128 _dist2 = _mm_add_ps(_mm_mul_ps(__y, __y), _x2_z2 );
        __m128 _dist = _mm_sqrt_ps(_dist2);
        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, __y);

        __m128 __j00 = _mm_mul_ps(_commonDer_c, __z);
        __m128 __j10 = _mm_mul_ps(_commonDer_r_y, __x);
        __m128 __j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
        //__m128 __j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, __x), _mm_set1_ps(-0.f));
        __m128 __j02_neg = _mm_mul_ps(_commonDer_c, __x);
        __m128 __j12 = _mm_mul_ps(_commonDer_r_y, __z );
        __m128 __j03_neg = _mm_mul_ps( __j02_neg, __y );
        __m128 __j13 = _mm_sub_ps(_mm_mul_ps(__j12, __y), _mm_mul_ps(__j11, __z) );
        __m128 __j04 = _mm_add_ps(_mm_mul_ps(__j00, __z), _mm_mul_ps(__j02_neg, __x) );
        __m128 __j14 = _mm_sub_ps(_mm_mul_ps(__j10, __z), _mm_mul_ps(__j12, __x) );
        __m128 __j05_neg = _mm_mul_ps(__j00, __y);
        __m128 __j15 = _mm_sub_ps(_mm_mul_ps(__j11, __x), _mm_mul_ps(__j10, __y) );

        _mm_store_ps(_j0_gray+i, _mm_add_ps(_mm_mul_ps(__gradX_weight, __j00), _mm_mul_ps(__gradY_weight, __j10) ) );
        _mm_store_ps(_j1_gray+i, _mm_mul_ps(__gradY_weight, __j11) );
        _mm_store_ps(_j2_gray+i, _mm_sub_ps(_mm_mul_ps(__gradY_weight, __j12), _mm_mul_ps(__gradX_weight, __j02_neg) ) );
        _mm_store_ps(_j3_gray+i, _mm_sub_ps(_mm_mul_ps(__gradY_weight, __j13), _mm_mul_ps(__gradX_weight, __j03_neg) ) );
        _mm_store_ps(_j4_gray+i, _mm_add_ps(_mm_mul_ps(__gradY_weight, __j14), _mm_mul_ps(__gradX_weight, __j04) ) );
        _mm_store_ps(_j5_gray+i, _mm_sub_ps(_mm_mul_ps(__gradY_weight, __j15), _mm_mul_ps(__gradX_weight, __j05_neg) ) );

        _mm_store_ps(_j0_depth+i, _mm_sub_ps(_mm_add_ps(_mm_mul_ps(__gradDepthX_weight, __j00), _mm_mul_ps(__gradDepthY_weight, __j10) ), _mm_mul_ps(__weight_stdDevDepthInv, _mm_div_ps(__x, _dist) ) ) );
        _mm_store_ps(_j1_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j11), _mm_mul_ps(__weight_stdDevDepthInv, _mm_div_ps( __y, _dist ) ) ) );
        _mm_store_ps(_j2_depth+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j12), _mm_mul_ps(__gradDepthX_weight, __j02_neg) ), _mm_mul_ps(__weight_stdDevDepthInv, _mm_div_ps(__z, _dist) ) ) );
        _mm_store_ps(_j3_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j13), _mm_mul_ps(__gradDepthX_weight, __j03_neg) ) );
        _mm_store_ps(_j4_depth+i, _mm_add_ps(_mm_mul_ps(__gradDepthY_weight, __j14), _mm_mul_ps(__gradDepthX_weight, __j04) ) );
        _mm_store_ps(_j5_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j15), _mm_mul_ps(__gradDepthX_weight, __j05_neg) ) );
    }
#endif

#if TEST_SIMD
    // Test SSE
    for(int i=0; i < xyz_tf.rows(); i++)
        for(size_t j=0; j < 6; j++)
        {
            ASSERT_(jacobians_photo(i,j) == jacobians_photo2(i,j));
            ASSERT_(jacobians_depth(i,j) == jacobians_depth2(i,j));
        }
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::computeJacobiansPhotoDepth " << xyz_tf.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

///*! Warp the image according to a given geometric transformation. */
//void SphericalModel::warpImage(cv::Mat img,                // The original image
//                                const Matrix4f & Rt,        // The relative pose of the robot between the two frames
//                                const costFuncType method ) //,  const bool use_bilinear )
//{
//    cout << " SphericalModel::warpImage \n";

//#if PRINT_PROFILING
//    double time_start = pcl::getTime();
//    //for(size_t i=0; i<1000; i++)
//    {
//#endif

//    nRows = graySrcPyr[pyrLevel].rows;
//    nRows = graySrcPyr[pyrLevel].cols;
//    imgSize = graySrcPyr[pyrLevel].size().area();
//    const float pixel_angle = 2*PI/nCols;
//    const float pixel_angle_inv = 1/pixel_angle;
//    const float half_width = nCols/2 - 0.5f;

//    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
////    float phi_start;
////    if(sensor_type == RGBD360_INDOOR)
////        phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
////    else
////        phi_start = float(174-512)/512 *PI/2 + 0.5f*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

//    // depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

//    //reconstruct3D(depthSrcPyr[pyrLevel], LUT_xyz_source, validPixels);
//    transformPts3D(LUT_xyz_source, Rt, xyz_transf);

//    //float *_depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyrLevel].data);
//    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyrLevel].data);
//    //float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyrLevel].data);

//    //    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyrLevel].data);
//    //    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyrLevel].data);
//    //    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyrLevel].data);
//    //    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyrLevel].data);

//    //float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyrLevel].data);
//    //float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyrLevel].data);
//    //float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyrLevel].data);
//    //float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyrLevel].data);

//    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//        //warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyrLevel].type());
//        warped_gray = cv::Mat(nRows,nCols,graySrcPyr[pyrLevel].type(),-1000);
//    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//        //warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyrLevel].type());
//        warped_depth = cv::Mat(nRows,nCols,depthSrcPyr[pyrLevel].type(),-1000);

//    float *_warpedGray = reinterpret_cast<float*>(warped_gray.data);
//    float *_warpedDepth = reinterpret_cast<float*>(warped_depth.data);

////    if( use_bilinear_ )
////    {
////         cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << endl;
////#if ENABLE_OPENMP
////#pragma omp parallel for
////#endif
////        for(size_t i=0; i < imgSize; i++)
////        {
////            //Transform the 3D point using the transformation matrix Rt
////            Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
////            // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

////            //Project the 3D point to the S2 sphere
////            float dist = xyz.norm();
////            float dist_inv = 1.f / dist;
////            float phi = asin(xyz(1)*dist_inv);
////            float transformed_r = (phi-phi_start)*pixel_angle_inv;
////            int transformed_r_int = round(transformed_r);
////            // cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << endl;
////            //Asign the intensity value to the warped image and compute the difference between the transformed
////            //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
////            if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
////            {
////                //visible_pixels(i) = 1;
////                float theta = atan2(xyz(0),xyz(2));
////                float transformed_c = half_width + theta*pixel_angle_inv; ASSERT_(transformed_c <= nCols_1); //transformed_c -= half_width;
////                int transformed_c_int = int(round(transformed_c)); ASSERT_(transformed_c_int<nCols);// % half_width;
////                cv::Point2f warped_pixel(transformed_r, transformed_c);

////                size_t warped_i = transformed_r_int * nCols + transformed_c_int;
////                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
////                    _warpedGray[warped_i] = bilinearInterp( grayTrgPyr[pyrLevel], warped_pixel );
////                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
////                    _warpedDepth[warped_i] = dist;
////            }
////        }
////    }
////    else
//    {
//         cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << endl;
//#if ENABLE_OPENMP
//#pragma omp parallel for
//#endif
//        for(size_t i=0; i < imgSize; i++)
//        {
//            //Transform the 3D point using the transformation matrix Rt
//            Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
//            // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

//            //Project the 3D point to the S2 sphere
//            float dist = xyz.norm();
//            float dist_inv = 1.f / dist;
//            float phi = asin(xyz(1)*dist_inv);
//            //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
//            int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
//            // cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << endl;
//            //Asign the intensity value to the warped image and compute the difference between the transformed
//            //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//            if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
//            {
//                //visible_pixels(i) = 1;
//                float theta = atan2(xyz(0),xyz(2));
//                int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //ASSERT_(transformed_c_int<nCols); //ASSERT_(transformed_c_int<nCols);
//                size_t warped_i = transformed_r_int * nCols + transformed_c_int;
//                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    _warpedGray[warped_i] = _graySrcPyr[i];
//                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                    _warpedDepth[warped_i] = dist;
//            }
//        }
//    }

//#if PRINT_PROFILING
//    }
//    double time_end = pcl::getTime();
//    cout << pyrLevel << " SphericalModel::warpImage took " << double (time_end - time_start)*1000 << " ms. \n";
//#endif
//}
