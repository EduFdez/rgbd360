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

#include <config.h>
#include <SphericalModel.h>
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

using namespace std;
using namespace Eigen;

SphericalModel::SphericalModel()
{
}

/*! Scale the intrinsic calibration parameters according to the image resolution (i.e. the reduced resolution being used). */
void SphericalModel::scaleCameraParams(std::vector<cv::Mat> & depthPyr, const int pyrLevel)
{
    const float scaleFactor = 1.0/pow(2,pyrLevel);
    nRows = depthPyr[0].rows*scaleFactor;
    nCols = depthPyr[0].cols*scaleFactor;
    imgSize = nRows*nCols;

    pixel_angle = 2*PI/nCols;
    pixel_angle_inv = 1/pixel_angle;
    row_phi_start = 0.5f*nRows;//-0.5f;

    assert(nRows == depthPyr[pyrLevel].rows);
    ASSERT_(nRows == depthPyr[pyrLevel].rows);
}

/*! Compute the unit sphere for the given spherical image dimmensions. This serves as a LUT to speed-up calculations. */
void SphericalModel::reconstruct3D_unitSphere()
{
    // Make LUT to store the values of the 3D points of the ref sphere
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
    const float half_height = 0.5f*nRows;//-0.5f;
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
    //cout << "SphericalModel::reconstruct3D... " << depth_img.rows*depth_img.cols << " pts \n";
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    nRows = depth_img.rows;
    nCols = depth_img.cols;
    imgSize = nRows*nCols;

    pixel_angle = 2*PI/nCols;
    pixel_angle_inv = 1/pixel_angle;
    int half_width_int = nCols/2;
    float half_width = half_width_int;// - 0.5f;
    const size_t start_row = (nCols-nRows) / 2;

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
    for(int col_theta=-half_width_int; col_theta < half_width_int; ++col_theta)
    {
//        float theta = (col_theta+0.5f)*pixel_angle;
        float theta = col_theta*pixel_angle;
        *(sinTheta++) = sin(theta);
        *(cosTheta++) = cos(theta);
        //cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << endl;
    }
    Eigen::VectorXf v_sinPhi( v_sinTheta.block(start_row,0,nRows,1) );
    Eigen::VectorXf v_cosPhi( v_cosTheta.block(start_row,0,nRows,1) );

#if TEST_SIMD
    // Test SSE
    Eigen::MatrixXf xyz2(imgSize,3);
    Eigen::VectorXi validPixels2(imgSize);
    for(int r=0, i=0; r < nRows;r++)
    {
        for(int c=0; c < nCols;c++,i++)
        {
            float depth1 = _depth[i];
            if(min_depth_ < depth1 && depth1 < max_depth_) //Compute the jacobian only for the valid points
            {
                validPixels2(i) = i;
                xyz2(i,0) = depth1 * v_cosPhi[r] * v_sinTheta[c];
                xyz2(i,1) = depth1 * v_sinPhi[r];
                xyz2(i,2) = depth1 * v_cosPhi[r] * v_cosTheta[c];
                //cout << " depth1 " << depth1 << " phi " << phi << " v_sinTheta[c] " << v_sinTheta[c] << endl;
                //cout << i << " c,r " << c << " " << r << " depth " << depth1 << " xyz " << xyz2(i,0) << " " << xyz2(i,1) << " " << xyz2(i,2) << " validPixels " << validPixels(i) << endl;
            }
            else
                validPixels2(i) = -1;

            //cout << i << " depth " << depth1 << " validPixels2(i) " << validPixels2(i) << " xyz2 " << xyz2(i,0) << " " << xyz2(i,1) << " " << xyz2(i,2) << endl;
        }
    }
    //mrpt::system::pause();
#endif

//Compute the 3D coordinates of the depth image
#if !(_SSE3) // # ifdef __SSE3__

    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(int r=0, i=0; r < nRows;r++)
    {
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
    //cout << " reconstruct3D _SSE3 " << imgSize << " pts \n";

    ASSERT_(nCols % 4 == 0); // Make sure that the image columns are aligned
    ASSERT_(nRows % 2 == 0);

    __m128i _idx_zero_ = _mm_setr_epi32(0.f, 1.f, 2.f, 3.f);
    __m128i __minus_one = _mm_set1_epi32(-1);
    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
//    if(imgSize > 1e5)
//    {
//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif

//    }
//    else
    {
        for(int r=0, i=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

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

                //__m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, __depth), _mm_cmplt_ps(__depth, _max_depth_) );
                //_mm_store_ps(_valid_pt+i, valid_depth_pts ); // Store 0 or -1

                __m128i __pos = _mm_set1_epi32(i);
                __m128i __idx = _mm_add_epi32(__pos, _idx_zero_);
                __m128 _invalid_pts = _mm_or_ps( _mm_cmpgt_ps(_min_depth_, __depth), _mm_cmpgt_ps(__depth, _max_depth_) );
                __m128i __idx_mask = _mm_or_si128(__idx, reinterpret_cast<__m128i>(_invalid_pts));
                __m128i *_v = reinterpret_cast<__m128i*>(&validPixels(i));
                _mm_store_si128(_v, __idx_mask);

//                for(int j=0; j < 4; j++)
//                {
//                    const int jj = j;
//                    cout << i+j << " depth " << __depth[jj] << " _invalid_pts " << (int)(_invalid_pts[jj]) << " xyz " << __x[jj] << " " << __y[jj] << " " << __z[jj] << " validPixels " << validPixels(i+j) << endl;
//    //                if(!valid_depth_pts[jj])
//    //                    validPixels(i+j) = -1;
//                }
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

#if TEST_SIMD
    // Test SSE
    for(int i=0; i < validPixels.size(); i++)
        if(validPixels(i) != -1)
        {
            if( validPixels(i) != validPixels2(i) )
                cout << i << " validPixels(i) " << validPixels(i) << " " << validPixels2(i) << " xyz " << xyz(i,0) << " " << xyz(i,1) << " " << xyz(i,2) << " xyz2 " << xyz2(i,0) << " " << xyz2(i,1) << " " << xyz2(i,2) << endl;
            if( !AlmostEqual2sComplement(xyz(i,0), xyz2(i,0), MAX_ULPS) || !AlmostEqual2sComplement(xyz(i,1), xyz2(i,1), MAX_ULPS) || !AlmostEqual2sComplement(xyz(i,2), xyz2(i,2), MAX_ULPS) )
                cout << i << " xyz " << xyz(i,0) << " " << xyz(i,1) << " " << xyz(i,2) << " xyz2 " << xyz2(i,0) << " " << xyz2(i,1) << " " << xyz2(i,2) << endl;
    //        cout << " diff " << xyz(i,0) - xyz2(i,0) << " " << 1e-5 << endl;
            ASSERT_( validPixels(i) == validPixels2(i) );
            ASSERT_( AlmostEqual2sComplement(xyz(i,0), xyz2(i,0), MAX_ULPS) );
            ASSERT_( AlmostEqual2sComplement(xyz(i,1), xyz2(i,1), MAX_ULPS) );
            ASSERT_( AlmostEqual2sComplement(xyz(i,2), xyz2(i,2), MAX_ULPS) );
        }
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::reconstruct3D " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
void SphericalModel::reconstruct3D_saliency( const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels, const int method,
                                             const cv::Mat & depth_gradX, const cv::Mat & depth_gradY, const float max_depth_grad, const float thres_saliency_depth,
                                             const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY, const float thres_saliency_gray
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
    int half_width_int = nCols/2;
    float half_width = half_width_int;// - 0.5f;

    // Compute the Unit Sphere: store the values of the trigonometric functions
    Eigen::VectorXf v_sinTheta(nCols);
    Eigen::VectorXf v_cosTheta(nCols);
    float *sinTheta = &v_sinTheta[0];
    float *cosTheta = &v_cosTheta[0];
    for(int col_theta=-half_width_int; col_theta < half_width_int; ++col_theta)
    {
        float theta = col_theta*pixel_angle;
//        float theta = (col_theta+0.5f)*pixel_angle;
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

    //Compute the 3D coordinates of the 3D points in ref frame
    validPixels.resize(imgSize);
    xyz.resize(imgSize,3);
    float *_depth = reinterpret_cast<float*>(depth_img.data);

#if TEST_SIMD
    Eigen::VectorXf validPixels2(imgSize);
    Eigen::MatrixXf xyz2(imgSize,3);
    size_t count_valid_pixels2 = 0;
    for(size_t r=0, i=0; r<nRows; r++)
    {
        for(size_t c=0; c<nCols; c++,i++)
        {
            //if(min_depth_ < _depth[i] && _depth[i] < max_depth_) //Compute only for the valid points
            if( min_depth_ < _depth[i] && _depth[i] < max_depth_ &&
                fabs(_depthGradXPyr[i]) < max_depth_grad && fabs(_depthGradYPyr[i]) < max_depth_grad ) //Compute only for the valid points
                if( fabs(_grayGradXPyr[i]) > thres_saliency_gray || fabs(_grayGradYPyr[i]) > thres_saliency_gray  )
                //    || fabs(_depthGradXPyr[i]) > thres_saliency_depth || fabs(_depthGradYPyr[i]) > thres_saliency_depth )
                {
                    validPixels2(count_valid_pixels2) = i;
                    //cout << " depth " << _depth[i] << " validPixels " << validPixels(count_valid_pixels2) << " count_valid_pixels2 " << count_valid_pixels2 << endl;
                    xyz2(count_valid_pixels2,0) = _depth[i] * v_cosPhi[r] * v_sinTheta[c];
                    xyz2(count_valid_pixels2,1) = _depth[i] * v_sinPhi[r];
                    xyz2(count_valid_pixels2,2) = _depth[i] * v_cosPhi[r] * v_cosTheta[c];
                    //cout << " xyz " << xyz.block(count_valid_pixels2,0,1,3) << endl;
                    ++count_valid_pixels2;
                    //mrpt::system::pause();
                }
        }
    }
    size_t valid_pixels_aligned = count_valid_pixels2 - count_valid_pixels2 % 4;
    validPixels2.conservativeResize(valid_pixels_aligned);
    xyz2.conservativeResize(valid_pixels_aligned,3);
#endif

#if !(_SSE3) // # ifdef __SSE3__

    size_t count_valid_pixels = 0;
    for(size_t r=0, i=0;r<nRows;r++)
    {
        for(size_t c=0;c<nCols;c++,i++)
        {
            //if(min_depth_ < _depth[i] && _depth[i] < max_depth_) //Compute only for the valid points
            if( min_depth_ < _depth[i] && _depth[i] < max_depth_ &&
                fabs(_depthGradXPyr[i]) < max_depth_grad && fabs(_depthGradYPyr[i]) < max_depth_grad ) //Compute only for the valid points
                if( fabs(_grayGradXPyr[i]) > thres_saliency_gray || fabs(_grayGradYPyr[i]) > thres_saliency_gray  )
                //    || fabs(_depthGradXPyr[i]) > thres_saliency_depth || fabs(_depthGradYPyr[i]) > thres_saliency_depth )
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

    //Compute the 3D coordinates of the pij of the ref frame
    Eigen::MatrixXf xyz_tmp(imgSize,3);
    VectorXi validPixels_tmp(imgSize);
    float *_x = &xyz_tmp(0,0);
    float *_y = &xyz_tmp(0,1);
    float *_z = &xyz_tmp(0,2);

    float *_valid_pt = reinterpret_cast<float*>(&validPixels_tmp(0));
    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
    __m128 _max_depth_grad = _mm_set1_ps(max_depth_grad);
    __m128 _max_depth_grad_neg = _mm_set1_ps(-max_depth_grad);
    __m128 _gray_saliency_ = _mm_set1_ps(thres_saliency_gray);
    __m128 _gray_saliency_neg = _mm_set1_ps(-thres_saliency_gray);
    __m128 _depth_saliency_ = _mm_set1_ps(thres_saliency_depth);
    __m128 _depth_saliency_neg = _mm_set1_ps(-thres_saliency_depth);

//    if(imgSize > 1e5)
//    {
//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif

//    }
//    else
    {
        for(int r=0, i=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

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
                __m128 __gradDepthX = _mm_load_ps(_depthGradXPyr+i);
                __m128 __gradDepthY = _mm_load_ps(_depthGradYPyr+i);
                __m128 __gradGrayX = _mm_load_ps(_grayGradXPyr+i);
                __m128 __gradGrayY = _mm_load_ps(_grayGradYPyr+i);
                __m128 valid_depth_grad = _mm_and_ps(_mm_and_ps( _mm_cmpgt_ps(_max_depth_grad, __gradDepthX), _mm_cmplt_ps(_max_depth_grad_neg, __gradDepthX) ),
                                                     _mm_and_ps( _mm_cmpgt_ps(_max_depth_grad, __gradDepthY), _mm_cmplt_ps(_max_depth_grad_neg, __gradDepthY) ) );
                //__m128 salient_pts = _mm_or_ps( _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(__gradDepthX, _depth_saliency_), _mm_cmplt_ps(__gradDepthX, _depth_saliency_neg) ),
                //                                           _mm_or_ps( _mm_cmpgt_ps(__gradDepthY, _depth_saliency_), _mm_cmplt_ps(__gradDepthY, _depth_saliency_neg) ) ),
                //                                _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( __gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayX, _gray_saliency_neg ) ),
                //                                           _mm_or_ps( _mm_cmpgt_ps( __gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayY, _gray_saliency_neg ) ) ) );
                //_mm_store_ps(_valid_pt+i, _mm_and_ps( valid_depth_pts, salient_pts ) );

                __m128 salient_pts;
                if(method == 0) // PhotoDepth
                    salient_pts = _mm_or_ps(_mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(__gradDepthX, _depth_saliency_), _mm_cmplt_ps(__gradDepthX, _depth_saliency_neg) ),
                                                       _mm_or_ps( _mm_cmpgt_ps(__gradDepthY, _depth_saliency_), _mm_cmplt_ps(__gradDepthY, _depth_saliency_neg) ) ),
                                            _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( __gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayX, _gray_saliency_neg ) ),
                                                       _mm_or_ps( _mm_cmpgt_ps( __gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayY, _gray_saliency_neg ) ) ) );
                else if(method == 1)
                    salient_pts =_mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( __gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayX, _gray_saliency_neg ) ),
                                            _mm_or_ps( _mm_cmpgt_ps( __gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( __gradGrayY, _gray_saliency_neg ) ) );
                else
                    salient_pts = _mm_or_ps(_mm_or_ps( _mm_cmpgt_ps(__gradDepthX, _depth_saliency_), _mm_cmplt_ps(__gradDepthX, _depth_saliency_neg) ),
                                            _mm_or_ps( _mm_cmpgt_ps(__gradDepthY, _depth_saliency_), _mm_cmplt_ps(__gradDepthY, _depth_saliency_neg) ) );

                _mm_store_ps(_valid_pt+i, _mm_and_ps( _mm_and_ps(valid_depth_pts, valid_depth_grad), salient_pts ) );
            }
        }
    }
    // Select only the salient points
    size_t count_valid_pixels = 0;
    //cout << " " << LUT_xyz_ref.rows() << " " << xyz_tmp.rows() << " size \n";
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

//#if TEST_SIMD
//    // Test SSE
//    for(int i=0, ii=0; i < validPixels.size(); i++, ii++)
//    {
//        while( validPixels(i) != validPixels2(ii) )
//        {
//            cout << i << " validPixels(i) " << validPixels(i) << " " << validPixels2(ii) << " xyz " << xyz(i,0) << " " << xyz(i,1) << " " << xyz(i,2) << " xyz2 " << xyz2(ii,0) << " " << xyz2(ii,1) << " " << xyz2(ii,2) << endl;
//            ++ii;
//            ASSERT_(ii < validPixels2.size());
//            continue;
//        }
//        if( !AlmostEqual2sComplement(xyz(i,0), xyz2(i,0), MAX_ULPS) || !AlmostEqual2sComplement(xyz(i,1), xyz2(i,1), MAX_ULPS) || !AlmostEqual2sComplement(xyz(i,2), xyz2(i,2), MAX_ULPS) )
//            cout << i << " xyz " << xyz(i,0) << " " << xyz(i,1) << " " << xyz(i,2) << " xyz2 " << xyz2(ii,0) << " " << xyz2(ii,1) << " " << xyz2(ii,2) << endl;
////        cout << " diff " << xyz(i,0) - xyz2(i,0) << " " << 1e-5 << endl;
//        ASSERT_( validPixels(i) == validPixels2(ii) );
//        ASSERT_( AlmostEqual2sComplement(xyz(i,0), xyz2(ii,0), MAX_ULPS) );
//        ASSERT_( AlmostEqual2sComplement(xyz(i,1), xyz2(ii,1), MAX_ULPS) );
//        ASSERT_( AlmostEqual2sComplement(xyz(i,2), xyz2(ii,2), MAX_ULPS) );
//    }
//#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::reconstruct3D_saliency " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Project 3D points XYZ according to the spherical camera model. */
void SphericalModel::reproject(const Eigen::MatrixXf & xyz, const cv::Mat & gray, cv::Mat & warped_gray, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible)
{
#if PRINT_PROFILING
    cout << " SphericalModel::reproject ... " << xyz.rows() << endl;
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    pixels.resize(xyz.rows(),2);
    float *_r = &pixels(0,0);
    float *_c = &pixels(0,1);
    visible.resize(xyz.rows());
    //float *_v = reinterpret_cast<float*>(&visible(0));

    const float *_x = &xyz(0,0);
    const float *_y = &xyz(0,1);
    const float *_z = &xyz(0,2);
    const float *_gray = reinterpret_cast<float*>(gray.data);
    warped_gray = cv::Mat(gray.rows, gray.cols, CV_32FC1);
    float *_warped = reinterpret_cast<float*>(warped_gray.data);

#if TEST_SIMD
    // Test SSE
    Eigen::VectorXi visible2(xyz.rows());
    Eigen::MatrixXf pixels2(xyz.rows(),2);
    for(int i=0; i < pixels.rows(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image(pt_xyz);
        pixels2(i,0) = warped_pixel.y;
        pixels2(i,1) = warped_pixel.x;
        visible2(i) = isInImage(warped_pixel.y) ? 1 : -1;
        //cout << i << " Pixel transform " << i/nCols << " " << i%nCols << " " << warped_pixel.y << " " << warped_pixel.x << " visible2(i) " << visible2(i) << endl;
//         mrpt::system::pause();
    }
#endif

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
        visible(i) = isInImage(warped_pixel.y) ? 1 : -1;
        _warped[i] = bilinearInterp(gray, warped_pixel);
    }

#else

    ASSERT_(__SSE4_1__); // For _mm_extract_epi32

    float half_width = 0.5f*nCols;// -0.5f;
    __m128 _row_phi_start = _mm_set1_ps(row_phi_start);
    __m128 _half_width = _mm_set1_ps(half_width);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    //__m128 _nCols = _mm_set1_ps(nCols);
    __m128 _nRows = _mm_set1_ps(nRows);
    __m128 _nRows_1 = _mm_set1_ps(nRows-1);
    __m128 _zero = _mm_set1_ps(0.f);
    for(int i=0; i < pixels.rows(); i+=4)
    {
        __m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);
        __m128 _dist = _mm_sqrt_ps( _mm_add_ps( _mm_add_ps(_mm_mul_ps(__x, __x), _mm_mul_ps(__y, __y) ), _mm_mul_ps(__z, __z) ) );

        __m128 _y_dist = _mm_div_ps( __y, _dist );
        float theta[4];
        float phi[4];
        for(int j=0; j < 4; j++)
        {
            const int jj = j;
            phi[j] = asin(_y_dist[jj]);
            theta[j] = atan2(_x[i+j], _z[i+j]);
        }
        __m128 _phi = _mm_load_ps(phi);
        __m128 _theta = _mm_load_ps(theta);
//        cout << "__y " << i << " " << __y[0] << " " << __y[1] << " " << __y[2] << " " << __y[3] << endl;
//        cout << "_phi " << i << " " << _phi[0] << " " << _phi[1] << " " << _phi[2] << " " << _phi[3] << endl;
//        cout << "_theta " << i << " " << _theta[0] << " " << _theta[1] << " " << _theta[2] << " " << _theta[3] << endl;

        __m128 __r = _mm_add_ps( _mm_mul_ps(_phi, _pixel_angle_inv ), _row_phi_start );
        __m128 __c = _mm_add_ps( _mm_mul_ps(_theta, _pixel_angle_inv ), _half_width );
//        cout << "__r " << i << " " << __r[0] << " " << __r[1] << " " << __r[2] << " " << __r[3] << endl;
//        cout << "__c " << i << " " << __c[0] << " " << __c[1] << " " << __c[2] << " " << __c[3] << " nCols " << nCols << endl;
//        for(int j=0; j < 4; j++)
//            ASSERT_(pixels(i+j,1) < nCols);

        _mm_store_ps(_r+i, __r);
        _mm_store_ps(_c+i, __c);

        //__m128 __v = _mm_and_ps( _mm_cmplt_ps(_zero, __r), _mm_cmplt_ps(__r, _nRows) );
        __m128 __invalid =  _mm_or_ps( _mm_cmpgt_ps(_zero, __r), _mm_cmpgt_ps(__r, _nRows_1) );
        __m128i __v_mask = _mm_or_si128(reinterpret_cast<__m128i>(__invalid), _mm_set1_epi32(1));
        //cout << "__v_mask " << i << " " << _mm_extract_epi32(__v_mask,0) << " " << _mm_extract_epi32(__v_mask,1) << " " << _mm_extract_epi32(__v_mask,2) << " " << _mm_extract_epi32(__v_mask,3) << endl;
        __m128i *_vv = reinterpret_cast<__m128i*>(&visible(i));
        _mm_store_si128(_vv, __v_mask);

        float warped_pix[4];
        for(int j=0; j < 4; j++)
            if( visible(i+j) != -1 )
            {
                cv::Point2f warped_pixel(__c[j],__r[j]);
                //_warped[i+j] = bilinearInterp(gray, warped_pixel);
                warped_pix[j] = bilinearInterp(gray, warped_pixel);
            }
        _mm_store_ps(_warped+i, _mm_load_ps(warped_pix) );
    }

#endif

#if TEST_SIMD
    // Test SSE
    cout << " Check result " << endl;
    for(int i=0; i < pixels.rows(); i++)
    {
        //cout << " Check result " << i << " " << pixels.size() << endl;
        if( !(visible(i) == visible2(i) && AlmostEqual2sComplement(pixels(i,0), pixels2(i,0), MAX_ULPS) && AlmostEqual2sComplement(pixels(i,1), pixels2(i,1), MAX_ULPS) ) )
        {
            Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
            cv::Point2f warped_pixel = project2Image(pt_xyz);
            cout << i << " Pixel transform " //<< i/nCols << " " << i%nCols << " "
                 << " warped_pixel " << warped_pixel.x << " " << warped_pixel.y << endl;
            cout << i << " pixels " << pixels(i,1) << " " << pixels(i,0) << " visible " << visible(i) << " " << visible2(i) << endl;
        }
//        ASSERT_( AlmostEqual2sComplement(pixels(i,0), pixels2(i,0), MAX_ULPS) );
//        ASSERT_( AlmostEqual2sComplement(pixels(i,1), pixels2(i,1), MAX_ULPS) );
//        ASSERT_( visible(i) == visible2(i) );
    }
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::reproject " << xyz.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Project 3D points XYZ according to the spherical camera model. */
void SphericalModel::project(const Eigen::MatrixXf & xyz, Eigen::MatrixXf & pixels, Eigen::VectorXi & visible)
{
#if PRINT_PROFILING
    //cout << " SphericalModel::project ... " << xyz.rows() << endl;
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

#if TEST_SIMD
    // Test SSE
    Eigen::VectorXi visible2(xyz.rows());
    Eigen::MatrixXf pixels2(xyz.rows(),2);
    for(int i=0; i < pixels.rows(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image(pt_xyz);
        pixels2(i,0) = warped_pixel.y;
        pixels2(i,1) = warped_pixel.x;
        visible2(i) = isInImage(warped_pixel.y) ? 1 : -1;
        //cout << i << " Pixel transform " << i/nCols << " " << i%nCols << " " << warped_pixel.y << " " << warped_pixel.x << " visible2(i) " << visible2(i) << endl;
//         mrpt::system::pause();
    }
#endif

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
        visible(i) = isInImage(warped_pixel.y) ? 1 : -1;
    }

#else

    ASSERT_(__SSE4_1__); // For _mm_extract_epi32

    float half_width = 0.5f*nCols;// - 0.5f;
    __m128 _row_phi_start = _mm_set1_ps(row_phi_start);
    __m128 _half_width = _mm_set1_ps(half_width);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    //__m128 _nCols = _mm_set1_ps(nCols);
    __m128 _nRows = _mm_set1_ps(nRows);
    __m128 _nRows_1 = _mm_set1_ps(nRows-1);
    __m128 _zero = _mm_set1_ps(0.f);
    __m128i _minus_one = _mm_set1_epi32(-1);
    __m128i __one = _mm_set1_epi32(1);
    for(int i=0; i < pixels.rows(); i+=4)
    {
        __m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);
        __m128 _dist = _mm_sqrt_ps( _mm_add_ps( _mm_add_ps(_mm_mul_ps(__x, __x), _mm_mul_ps(__y, __y) ), _mm_mul_ps(__z, __z) ) );

        __m128 _y_dist = _mm_div_ps( __y, _dist );
        float theta[4];
        float phi[4];
        for(int j=0; j < 4; j++)
        {
            const int jj = j;
            phi[j] = asin(_y_dist[jj]);
            theta[j] = atan2(_x[i+j], _z[i+j]);
        }
        __m128 _phi = _mm_load_ps(phi);
        __m128 _theta = _mm_load_ps(theta);
//        cout << "__y " << i << " " << __y[0] << " " << __y[1] << " " << __y[2] << " " << __y[3] << endl;
//        cout << "_phi " << i << " " << _phi[0] << " " << _phi[1] << " " << _phi[2] << " " << _phi[3] << endl;
//        cout << "_theta " << i << " " << _theta[0] << " " << _theta[1] << " " << _theta[2] << " " << _theta[3] << endl;

        __m128 __r = _mm_add_ps( _mm_mul_ps(_phi, _pixel_angle_inv ), _row_phi_start );
        __m128 __c = _mm_add_ps( _mm_mul_ps(_theta, _pixel_angle_inv ), _half_width );
//        cout << "__r " << i << " " << __r[0] << " " << __r[1] << " " << __r[2] << " " << __r[3] << endl;
//        cout << "__c " << i << " " << __c[0] << " " << __c[1] << " " << __c[2] << " " << __c[3] << " nCols " << nCols << endl;
//        for(int j=0; j < 4; j++)
//            ASSERT_(pixels(i+j,1) < nCols);

        _mm_store_ps(_r+i, __r);
        _mm_store_ps(_c+i, __c);

        __m128 __valid = _mm_and_ps( _mm_cmplt_ps(_zero, __r), _mm_cmplt_ps(__r, _nRows) );
//        __m128i __v_mask = _mm_and_si128(reinterpret_cast<__m128i>(__valid), _minus_one);
//        __m128 __invalid =  _mm_or_ps( _mm_cmpgt_ps(_zero, __r), _mm_cmpgt_ps(__r, _nRows_1) );
//        //_mm_store_ps(_v+i, __invalid); // Warning, the bit to int conversion is: 00000000 -> nan, 11111111 -> -1
//        __m128i __v_mask = _mm_or_si128(reinterpret_cast<__m128i>(__invalid), __one);
        __m128i __v_mask = _mm_or_si128( _mm_xor_si128(reinterpret_cast<__m128i>(__valid), _minus_one), __one);
        //cout << "__v_mask " << i << " " << _mm_extract_epi32(__v_mask,0) << " " << _mm_extract_epi32(__v_mask,1) << " " << _mm_extract_epi32(__v_mask,2) << " " << _mm_extract_epi32(__v_mask,3) << endl;
        __m128i *_vv = reinterpret_cast<__m128i*>(&visible(i));
        _mm_store_si128(_vv, __v_mask);
        //cout << "visible " << i << " " << visible(i) << " " << visible(i+1) << " " << visible(i+2) << " " << visible(i+3) << endl;

//        for(int j=0; j < 4; j++)
//            if( !(visible(i+j) == visible2(i+j)) )
//            {
//                const int jj = j;
//                cout << i << " pixels " << pixels(i+j,1) << " " << pixels(i+j,0) << " pixels2 " << pixels2(i+j,1) << " " << pixels2(i+j,0) << " visible " << visible(i+j) << " vs " << __v[jj] << " " << visible2(i+j) << " __r " << __r[jj] << endl;
//                ASSERT_(0);
//            }
    }

#endif

#if TEST_SIMD
    // Test SSE
    //cout << " Check result " << endl;
    for(int i=0; i < pixels.rows(); i++)
    {
        if( visible(i) == -1 )
            continue;

        //cout << " Check result " << i << " " << pixels.size() << endl;
        if( !(visible(i) == visible2(i) && AlmostEqual2sComplement(pixels(i,0), pixels2(i,0), MAX_ULPS) && AlmostEqual2sComplement(pixels(i,1), pixels2(i,1), MAX_ULPS) ) )
        {
            Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
            cv::Point2f warped_pixel = project2Image(pt_xyz);
            cout << i << " pt_xyz " << pt_xyz.transpose() << " Pixel transform " //<< i/nCols << " " << i%nCols << " "
                 << " warped_pixel " << warped_pixel.x << " " << warped_pixel.y << endl;
            cout << i << " pixels " << pixels(i,1) << " " << pixels(i,0) << " visible " << visible(i) << " " << visible2(i) << endl;
        }
//        ASSERT_( AlmostEqual2sComplement(pixels(i,0), pixels2(i,0), MAX_ULPS) );
//        ASSERT_( AlmostEqual2sComplement(pixels(i,1), pixels2(i,1), MAX_ULPS) );
        ASSERT_( visible(i) == visible2(i) );
    }
    //mrpt::system::pause();
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::project " << xyz.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Project 3D points XYZ according to the spherical camera model. */
void SphericalModel::projectNN(const Eigen::MatrixXf & xyz, VectorXi & valid_pixels, VectorXi & warped_pixels)
{
#if PRINT_PROFILING
    //cout << " SphericalModel::projectNN ... " << xyz.rows() << " points. \n";
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    warped_pixels.resize(xyz.rows());
//    visible.resize(xyz.rows());
    //int *_p = &pixels(0);
    //float *_v = reinterpret_cast<float*>(&visible(0));
    const float *_x = &xyz(0,0);
    const float *_y = &xyz(0,1);
    const float *_z = &xyz(0,2);

#if TEST_SIMD
    // Test SSE
    Eigen::VectorXi warped_pixels2(xyz.rows());
    for(int i=0; i < warped_pixels.size(); i++)
    {
        if(valid_pixels(i) == -1)
        {
            warped_pixels2(i) = -1;
            continue;
        }
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image(pt_xyz);
        int r_transf = round(warped_pixel.y);
        int c_transf = round(warped_pixel.x);
        if( isInImage(r_transf) )
            warped_pixels2(i) = r_transf * nCols + c_transf;
        else
            warped_pixels2(i) = -1;
        // cout << i << " Pixel transform " << i/nCols << " " << i%nCols << " " << r_transf << " " << c_transf << " pixel " << warped_pixels2(i) << endl;
        // mrpt::system::pause();
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(int i=0; i < warped_pixels.size(); i++)
    {
        if(valid_pixels(i) == -1)
        {
            warped_pixels(i) = -1;
            continue;
        }
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image(pt_xyz);
        int r_transf = round(warped_pixel.y);
        int c_transf = round(warped_pixel.x);
        if( isInImage(r_transf) )
            warped_pixels(i) = r_transf * nCols + c_transf;
        else
            warped_pixels(i) = -1;
        // cout << i << " valid " << valid_pixels(i) << " Pixel transform " << i/nCols << " " << i%nCols << " " << r_transf << " " << c_transf << endl;
//        visible(i) = isInImage(r_transf) ? 1 : 0;
    }

#else

    ASSERT_(__SSE4_1__); // For _mm_extract_epi32

    float half_width = 0.5f*nCols;// - 0.5f;
    __m128i _nCols = _mm_set1_epi32(nCols);
    //__m128i _nRows = _mm_set1_epi32(nRows);
    __m128i _nRows_1 = _mm_set1_epi32(nRows-1);
    __m128i _minus_one = _mm_set1_epi32(-1);
    __m128i _zero = _mm_set1_epi32(0);
    //__m128 _nCols = _mm_set1_ps(nCols);
    __m128 _row_phi_start = _mm_set1_ps(row_phi_start);
    __m128 _half_width = _mm_set1_ps(half_width);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(int i=0; i < warped_pixels.size(); i+=4)
    {
        __m128 __x = _mm_load_ps(_x+i);
        __m128 __y = _mm_load_ps(_y+i);
        __m128 __z = _mm_load_ps(_z+i);
        __m128 _dist = _mm_sqrt_ps( _mm_add_ps( _mm_add_ps(_mm_mul_ps(__x, __x), _mm_mul_ps(__y, __y) ), _mm_mul_ps(__z, __z) ) );

        __m128 _y_dist = _mm_div_ps( __y, _dist );
        float theta[4];
        float phi[4];
        for(int j=0; j < 4; j++)
        {
            const int jj = j;
            phi[j] = asin(_y_dist[jj]);
            theta[j] = atan2(_x[i+j], _z[i+j]);
        }
        __m128 _phi = _mm_load_ps(phi);
        __m128 _theta = _mm_load_ps(theta);

        __m128 __r = _mm_add_ps( _mm_mul_ps(_phi, _pixel_angle_inv ), _row_phi_start);
        __m128 __c = _mm_add_ps( _mm_mul_ps(_theta, _pixel_angle_inv ), _half_width );
        //__m128 __p = _mm_add_epi32( _mm_cvtps_epi32(_nCols, __r ), __c);
        //cout << "__r " << i << " " << __r[0] << " " << __r[1] << " " << __r[2] << " " << __r[3] << endl;
//        cout << "__c " << i << " " << __c[0] << " " << __c[1] << " " << __c[2] << " " << __c[3] << endl;

        __m128i __r_int = _mm_cvtps_epi32( _mm_round_ps(__r, 0x0) );
        __m128i __c_int = _mm_cvtps_epi32( _mm_round_ps(__c, 0x0) );
        //cout << "__r_int " << i << " " << _mm_extract_epi32(__r_int,0) << " " << _mm_extract_epi32(__r_int,1) << " " << _mm_extract_epi32(__r_int,2) << " " << _mm_extract_epi32(__r_int,3) << endl;
//        cout << "__c_int " << i << " " << _mm_extract_epi32(__c_int,0) << " " << _mm_extract_epi32(__c_int,1) << " " << _mm_extract_epi32(__c_int,2) << " " << _mm_extract_epi32(__c_int,3) << endl;

        __m128i __p = _mm_add_epi32( _mm_mullo_epi32(_nCols, __r_int ), __c_int );
        __m128i _outOfImg = _mm_or_si128( _mm_cmpgt_epi32(_zero, __r_int), _mm_cmpgt_epi32(__r_int, _nRows_1) );
        const __m128i *_valid = reinterpret_cast<__m128i*>(&valid_pixels(i));
        __m128i __v = _mm_load_si128(_valid);
//        for(int j=0; j < 4; j++)
//        {
//            const int jj = j;
//            if(_mm_extract_epi32(__v,jj) == -1)
//                ASSERT_( _mm_extract_epi32(_outOfImg,jj) == _mm_extract_epi32(__v,jj) );
//        }

        __m128i _invalid = _mm_or_si128( _outOfImg, _mm_cmpeq_epi32(__v, _minus_one) );
        __m128i __p_mask = _mm_or_si128(__p, reinterpret_cast<__m128i>(_invalid));
//        cout << " __v " << i << ": " << _mm_extract_epi32(__v,0) << " " << _mm_extract_epi32(__v,1) << " " << _mm_extract_epi32(__v,2) << " " << _mm_extract_epi32(__v,3) << endl;
//        cout << " _invalid " << i << ": " << _mm_extract_epi32(_invalid,0) << " " << _mm_extract_epi32(_invalid,1) << " " << _mm_extract_epi32(_invalid,2) << " " << _mm_extract_epi32(_invalid,3) << endl;
//        cout << " __p " << i << " " << _mm_extract_epi32(__p,0) << " " << _mm_extract_epi32(__p,1) << " " << _mm_extract_epi32(__p,2) << " " << _mm_extract_epi32(__p,3) << endl;
//        cout << " __p_mask " << i << ": " << _mm_extract_epi32(__p_mask,0) << " " << _mm_extract_epi32(__p_mask,1) << " " << _mm_extract_epi32(__p_mask,2) << " " << _mm_extract_epi32(__p_mask,3) << endl;
//        cout << "_outOfImg " << i << ": " << _mm_extract_epi32(_outOfImg,0) << " " << _mm_extract_epi32(_outOfImg,1) << " " << _mm_extract_epi32(_outOfImg,2) << " " << _mm_extract_epi32(_outOfImg,3) << endl;

        __m128i *_p = reinterpret_cast<__m128i*>(&warped_pixels(i));
        _mm_store_si128(_p, __p_mask);
//        cout << "stored warped __p  " << i << " " << warped_pixels(i) << " " << warped_pixels(i+1) << " " << warped_pixels(i+2) << " " << warped_pixels(i+3) << endl;
//        mrpt::system::pause();

            //        __m128i __v = _mm_and_si128( _mm_cmplt_epi32(_minus_one, __r_int), _mm_cmplt_epi32(__r_int, _nRows) );
////        __m128i valid_row = _mm_and_si128( _mm_cmplt_epi32(_minus_one, __r_int), _mm_cmplt_epi32(__r_int, _nRows) );
////        __m128i valid_col = _mm_and_si128( _mm_cmplt_epi32(_minus_one, __c_int), _mm_cmplt_epi32(__c_int, _nCols) );
////        cout << "valid_row " << " " << _mm_extract_epi32(valid_row,0) << " " << _mm_extract_epi32(valid_row,1) << " " << _mm_extract_epi32(valid_row,2) << " " << _mm_extract_epi32(valid_row,3) << endl;
////        cout << "valid_col " << " " << _mm_extract_epi32(valid_col,0) << " " << _mm_extract_epi32(valid_col,1) << " " << _mm_extract_epi32(valid_col,2) << " " << _mm_extract_epi32(valid_col,3) << endl;
//        __m128i *_v = reinterpret_cast<__m128i*>(&visible(i));
//        _mm_store_si128(_v, __v);
    }

#endif

#if TEST_SIMD
    // Test SSE
    for(int i=0; i < warped_pixels.rows(); i++)
    {
        if( warped_pixels(i) != warped_pixels2(i) )
        {
            Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
            cv::Point2f warped_pixel = project2Image(pt_xyz);
            int r_transf = round(warped_pixel.y);
            int c_transf = round(warped_pixel.x);
            cout << i << " Pixel transform " //<< i/nCols << " " << i%nCols << " "
                 << c_transf << " " << r_transf << " warped_pixel " << warped_pixel.x << " " << warped_pixel.y << endl;
            cout << i << " warped_pixels " << warped_pixels(i) << " " << warped_pixels2(i) << endl; // << " visible " << visible(i) << " " << visible2(i) <<
//            cout << " pt_xyz " << pt_xyz.transpose() << endl; // << " visible " << visible(i) << " " << visible2(i) <<
//            float dist_inv = 1.f / pt_xyz.norm();
//            float phi = asin(pt_xyz(1)*dist_inv);
//            float theta = atan2(pt_xyz(0),pt_xyz(2));
//            float transformed_r = phi*pixel_angle_inv + row_phi_start;
//            float transformed_c = theta*pixel_angle_inv + nCols/2; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
//            cout << " pixel_angle_inv " << pixel_angle_inv << " pixel_angle_inv " << pixel_angle_inv << " theta " << theta  << " project2Image " << transformed_c << " " << transformed_r << endl;
//            cout << " _y_dist " << pt_xyz(1)*dist_inv << " phi " << phi << " theta " << theta  << " project2Image " << transformed_c << " " << transformed_r << endl;
        }
        //ASSERT_( warped_pixels(i) == warped_pixels2(i) );
    }
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::projectNN " << xyz.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

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
    const float *_weight = &weights(0);

#if TEST_SIMD
    Eigen::MatrixXf jacobians_photo2(xyz_tf.rows(), 6);
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(pt_xyz, jacobianWarpRt);
        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo2.block(i,0,1,6) = ((weights(i) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(pt_xyz, jacobianWarpRt);
        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo.block(i,0,1,6) = ((weights(i) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;
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
    {
        //cout << i << " weights(i) " << weights(i) << "\n  jacobians_photo(i) " << jacobians_photo.block(i,0,1,6) << "\n jacobians_photo2(i) " << jacobians_photo2.block(i,0,1,6) << endl;
        for(size_t j=0; j < 6; j++)
        {
            if(weights(i) > 0.f)
            {
                if( !AlmostEqual2sComplement(jacobians_photo(i,j), jacobians_photo2(i,j), MAX_ULPS) )
                    cout << i << " weights(i) " << weights(i) << "\n  jacobians_photo(i) " << jacobians_photo.block(i,0,1,6) << "\n jacobians_photo2(i) " << jacobians_photo2.block(i,0,1,6) << endl;
                ASSERT_( AlmostEqual2sComplement(jacobians_photo(i,j), jacobians_photo2(i,j), MAX_ULPS) );
            }
        }
    }
    //mrpt::system::pause();
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::computeJacobiansPhoto " << xyz_tf.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void SphericalModel::computeJacobiansPhoto2(const Eigen::MatrixXf & xyz_tf, const Eigen::VectorXi & warped_pixels, const float stdDevPhoto_inv, const Eigen::VectorXf & weights, Eigen::MatrixXf & jacobians_photo, float *_grayGradX, float *_grayGradY)
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
    const float *_weight = &weights(0);

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(pt_xyz, jacobianWarpRt);
        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[warped_pixels(i)];
        img_gradient(0,1) = _grayGradY[warped_pixels(i)];
        jacobians_photo.block(i,0,1,6) = ((weights(i) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;
    }

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
    const float *_weight = &weights(0);
    const float *_stdDevInv = &stdDevError_inv(0);

#if TEST_SIMD
    // Test SSE
    Eigen::MatrixXf jacobians_depth2(xyz_tf.rows(), 6);
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(pt_xyz, jacobianWarpRt);
        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / pt_xyz.norm()) * pt_xyz.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth2.block(i,0,1,6) = (weights(i) * stdDevError_inv(i)) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(pt_xyz, jacobianWarpRt);
        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / xyz_tf.norm()) * pt_xyz.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth.block(i,0,1,6) = (weights(i) * stdDevError_inv(i)) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);
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
        __m128 __gradDepthX = _mm_load_ps(_depthGradX+i);
        __m128 __gradDepthY = _mm_load_ps(_depthGradY+i);
        __m128 __stdDevDepthInv = _mm_load_ps(_stdDevInv+i);
        __m128 __weight = _mm_load_ps(_weight+i);
        __m128 __weight_stdDevDepthInv = _mm_mul_ps(__stdDevDepthInv, __weight);
        __m128 __gradDepthX_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradDepthX);
        __m128 __gradDepthY_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradDepthY);

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
    {
        //cout << i << " weights(i) " << weights(i) << "\n  jacobians_depth(i) " << jacobians_depth.block(i,0,1,6) << "\n jacobians_depth2(i) " << jacobians_depth2.block(i,0,1,6) << endl;
        for(size_t j=0; j < 6; j++)
        {
            if(weights(i) > 0.f)
            {
                if( !AlmostEqual2sComplement(jacobians_depth(i,j), jacobians_depth2(i,j), MAX_ULPS) )
                    cout << i << " weights(i) " << weights(i) << "\n  jacobians_depth(i) " << jacobians_depth.block(i,0,1,6) << "\n jacobians_depth2(i) " << jacobians_depth2.block(i,0,1,6) << endl;
                ASSERT_( AlmostEqual2sComplement(jacobians_depth(i,j), jacobians_depth2(i,j), MAX_ULPS) );
            }
        }
    }
    //mrpt::system::pause();
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
    const float *_weight = &weights(0);
    const float *_stdDevInv = &stdDevError_inv(0);

#if TEST_SIMD
    // Test SSE
    Eigen::MatrixXf jacobians_photo2(xyz_tf.rows(), 6);
    Eigen::MatrixXf jacobians_depth2(xyz_tf.rows(), 6);
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(pt_xyz, jacobianWarpRt);

        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo2.block(i,0,1,6) = ((weights(i) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;

        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / pt_xyz.norm()) * pt_xyz.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth2.block(i,0,1,6) = (weights(i) * stdDevError_inv(i)) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);
//        cout << " test_jx " << i << " " << weights(i) * stdDevError_inv(i) * depth_gradient * jacobianWarpRt.block(0,0,2,1) << endl;
//        cout << " test_jx_ " << i << " " << weights(i) * stdDevError_inv(i) * pt_xyz(0) / pt_xyz.norm() << endl;
    }
#endif

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        Vector3f pt_xyz = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT(pt_xyz, jacobianWarpRt);

        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians_photo.block(i,0,1,6) = ((weights(i) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;

        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / pt_xyz.norm()) * pt_xyz.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth.block(i,0,1,6) = (weights(i) * stdDevError_inv(i)) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);
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

        __m128 __gradDepthX = _mm_load_ps(_depthGradX+i);
        __m128 __gradDepthY = _mm_load_ps(_depthGradY+i);
        __m128 __stdDevDepthInv = _mm_load_ps(_stdDevInv+i);
        __m128 __weight_stdDevDepthInv = _mm_mul_ps(__stdDevDepthInv, __weight);
        __m128 __gradDepthX_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradDepthX);
        __m128 __gradDepthY_weight = _mm_mul_ps(__weight_stdDevDepthInv, __gradDepthY);

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

        //jacobians_depth.block(i,0,1,6) = (weights(i) * stdDevError_inv(i)) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);
        __m128 _weight_dist = _mm_div_ps(__weight_stdDevDepthInv, _dist);
//        __m128 test_j0 = _mm_add_ps(_mm_mul_ps(__gradDepthX_weight, __j00), _mm_mul_ps(__gradDepthY_weight, __j10) );
//        __m128 test_j0_ = _mm_mul_ps(__x, _weight_dist);
//        cout << " test_j0 " << i << " " << test_j0[0] << " " << test_j0[1] << " " << test_j0[2] << " " << test_j0[3] << endl;
//        cout << " test_j0_ " << i << " " << test_j0_[0] << " " << test_j0_[1] << " " << test_j0_[2] << " " << test_j0_[3] << endl;
        _mm_store_ps(_j0_depth+i, _mm_sub_ps(_mm_add_ps(_mm_mul_ps(__gradDepthX_weight, __j00), _mm_mul_ps(__gradDepthY_weight, __j10) ), _mm_mul_ps(__x, _weight_dist) ) );
        _mm_store_ps(_j1_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j11), _mm_mul_ps(__y, _weight_dist ) ) );
        _mm_store_ps(_j2_depth+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j12), _mm_mul_ps(__gradDepthX_weight, __j02_neg) ), _mm_mul_ps(__z, _weight_dist) ) );
        _mm_store_ps(_j3_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j13), _mm_mul_ps(__gradDepthX_weight, __j03_neg) ) );
        _mm_store_ps(_j4_depth+i, _mm_add_ps(_mm_mul_ps(__gradDepthY_weight, __j14), _mm_mul_ps(__gradDepthX_weight, __j04) ) );
        _mm_store_ps(_j5_depth+i, _mm_sub_ps(_mm_mul_ps(__gradDepthY_weight, __j15), _mm_mul_ps(__gradDepthX_weight, __j05_neg) ) );
    }
#endif

#if TEST_SIMD
    // Test SSE
    for(int i=0; i < xyz_tf.rows(); i++)
    {
        //cout << i << " weights(i) " << weights(i) << "\n  jac_photo " << jacobians_photo.block(i,0,1,6) << "\n                    (i) " << jacobians_photo.block(i,0,1,6) << endl;
        for(size_t j=0; j < 6; j++)
        {
            if(weights(i) > 0.f)
            {
                //cout << i << " jacobians_photo(i) " << jacobians_photo.block(i,0,1,6) << "\n  jacobians_photo2(i) " << jacobians_photo2.block(i,0,1,6) << endl;
                if( !AlmostEqual2sComplement(jacobians_photo(i,j), jacobians_photo2(i,j), MAX_ULPS) )
                    cout << i << " weights(i) " << weights(i) << "\n  jacobians_photo(i) " << jacobians_photo.block(i,0,1,6) << "\n jacobians_photo2(i) " << jacobians_photo2.block(i,0,1,6) << endl;
                ASSERT_( AlmostEqual2sComplement(jacobians_photo(i,j), jacobians_photo2(i,j), MAX_ULPS) );
                if( !AlmostEqual2sComplement(jacobians_depth(i,j), jacobians_depth2(i,j), MAX_ULPS) )
                    cout << i << " weights(i) " << weights(i) << "\n  jacobians_depth(i) " << jacobians_depth.block(i,0,1,6) << "\n jacobians_depth2(i) " << jacobians_depth2.block(i,0,1,6) << endl;
                ASSERT_( AlmostEqual2sComplement(jacobians_depth(i,j), jacobians_depth2(i,j), MAX_ULPS) );
            }
        }
    }
#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " SphericalModel::computeJacobiansPhotoDepth " << xyz_tf.rows() << " points took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}
