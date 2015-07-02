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

#include <ProjectionModel.h>
#include <pcl/common/time.h>
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
#define INVALID_POINT -10000

using namespace std;

ProjectionModel::ProjectionModel() :
    min_depth_(0.3f),
    max_depth_(20.f),
    sensor_type(STEREO_OUTDOOR) //RGBD360_INDOOR
{
    // For sensor_type = KINECT
    cameraMatrix << 262.5, 0., 1.5950e+02,
                    0., 262.5, 1.1950e+02,
                    0., 0., 1.;
}

/*! Scale the intrinsic calibration parameters according to the image resolution (i.e. the reduced resolution being used). */
void ProjectionModel::scaleCameraParams(const float scaleFactor)
{
    fx = cameraMatrix(0,0)*scaleFactor;
    fy = cameraMatrix(1,1)*scaleFactor;
    ox = cameraMatrix(0,2)*scaleFactor;
    oy = cameraMatrix(1,2)*scaleFactor;
    inv_fx = 1.f/fx;
    inv_fy = 1.f/fy;
}

/*! Compute the unit sphere for the given spherical image dimmensions. This serves as a LUT to speed-up calculations. */
void ProjectionModel::computeUnitSphere(const size_t nRows, const size_t nCols)
{
    // Make LUT to store the values of the 3D points of the source sphere
    Eigen::MatrixXf unit_sphere;
    unit_sphere.resize(nRows*nCols,3);
    const float pixel_angle = 2*PI/nCols;
    std::vector<float> v_sinTheta(nCols);
    std::vector<float> v_cosTheta(nCols);
    for(size_t c=0; c < nCols; c++)
    {
        float theta = c*pixel_angle;
        v_sinTheta[c] = sin(theta);
        v_cosTheta[c] = cos(theta);
    }
    const float half_height = 0.5*nRows-0.5;
    for(size_t r=0; r < nRows; r++)
    {
        float phi = (half_height-r)*pixel_angle;
        float sin_phi = sin(phi);
        float cos_phi = cos(phi);

        for(size_t c=0;c<nCols;c++)
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
void ProjectionModel::computeSphereXYZ(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels)
{
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const float half_width = nCols/2;
    const float pixel_angle = 2*PI/nCols;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5*nRows-0.5)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    xyz.resize(imgSize,3);

#if !(_SSE3) // # ifdef __SSE3__
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

    //Compute the 3D coordinates of the pij of the source frame
    validPixels = Eigen::VectorXi::Ones(imgSize);
    float *_depth_src = reinterpret_cast<float*>(depth_img.data);
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(size_t r=0;r<nRows;r++)
    {
        size_t i = r*nCols;
        for(size_t c=0;c<nCols;c++,i++)
        {
            float depth1 = _depth_src[i];
            if(min_depth_ < depth1 && depth1 < max_depth_) //Compute the jacobian only for the valid points
            {
                //cout << " depth1 " << depth1 << " phi " << phi << " v_sinTheta[c] " << v_sinTheta[c] << endl;
                xyz(i,0) = depth1 * v_cosPhi[r] * v_sinTheta[c];
                xyz(i,1) = depth1 * v_sinPhi[r];
                xyz(i,2) = depth1 * v_cosPhi[r] * v_cosTheta[c];
                //cout << " xyz " << xyz [i].transpose() << " xyz_eigen " << xyz_eigen.block(c*nRows+r,0,1,3) << endl;
                //mrpt::system::pause();
            }
            else
                validPixels(i) = 0;
        }
    }

#else
//#elif !(_AVX) // # ifdef __AVX__
    cout << " computeSphereXYZ _SSE3 " << imgSize << " pts \n";

    assert(nCols % 4 == 0); // Make sure that the image columns are aligned
    assert(nRows % 2 == 0);

    validPixels = Eigen::VectorXi::Zero(imgSize);
    //validPixels = Eigen::VectorXi::Ones(imgSize);

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

    //Compute the 3D coordinates of the pij of the source frame
    float *_depth = reinterpret_cast<float*>(depth_img.data);
    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);
    float *_valid_pt = reinterpret_cast<float*>(&validPixels(0));
    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
    if(imgSize > 1e5)
    {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
        //for(size_t i=0; i < block_end; i++)
        for(size_t r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t block_i = r*nCols;
            for(size_t c=0; c < nCols; c+=4, block_i+=4)
            {
                __m128 block_depth = _mm_load_ps(_depth+block_i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 block_x = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 block_y = _mm_mul_ps( block_depth, sin_phi );
                __m128 block_z = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+block_i, block_x);
                _mm_store_ps(_y+block_i, block_y);
                _mm_store_ps(_z+block_i, block_z);

                //__m128 mask = _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) );
                //mi cmpeq_epi8(mi a,mi b)
                _mm_store_ps(_valid_pt+block_i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) );
                //_mm_stream_si128
            }
        }
    }
    else
    {
        for(size_t r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t block_i = r*nCols;
            for(size_t c=0; c < nCols; c+=4, block_i+=4)
            {
                __m128 block_depth = _mm_load_ps(_depth+block_i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 block_x = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 block_y = _mm_mul_ps( block_depth, sin_phi );
                __m128 block_z = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+block_i, block_x);
                _mm_store_ps(_y+block_i, block_y);
                _mm_store_ps(_z+block_i, block_z);

                //__m128 mask = _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) );
                _mm_store_ps(_valid_pt+block_i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) );
            }
        }
    }
//    // Compute the transformation of those points which do not enter in a block
//    const Eigen::Matrix3f rotation_transposed = Rt.block(0,0,3,3).transpose();
//    const Eigen::Matrix<float,1,3> translation_transposed = Rt.block(0,3,3,1).transpose();
//    for(int i=block_end; i < imgSize; i++)
//    {
//        ***********
//        output_pts.block(i,0,1,3) = input_pts.block(i,0,1,3) * rotation_transposed + translation_transposed;
//    }

#endif

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " ProjectionModel::computeSphereXYZ_sse " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
void ProjectionModel::computeSphereXYZ_saliency(Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                            const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                            const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY,
                                            const float thres_saliency_gray, const float thres_saliency_depth
                                            ) // TODO extend this function to employ only depth
{
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

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
    cout << " computeSphereXYZ _SSE3 " << imgSize << " pts \n";

    //Compute the 3D coordinates of the pij of the source frame
    Eigen::MatrixXf xyz_tmp(imgSize,3);
    Eigen::VectorXi validPixels_tmp(imgSize);
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
        //for(size_t i=0; i < block_end; i++)
        for(size_t r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t block_i = r*nCols;
            for(size_t c=0; c < nCols; c+=4, block_i+=4)
            {
                __m128 block_depth = _mm_load_ps(_depth+block_i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 block_x = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 block_y = _mm_mul_ps( block_depth, sin_phi );
                __m128 block_z = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+block_i, block_x);
                _mm_store_ps(_y+block_i, block_y);
                _mm_store_ps(_z+block_i, block_z);

                //__m128 mask = _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) );
                //mi cmpeq_epi8(mi a,mi b)

                //_mm_store_ps(_valid_pt+block_i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) );
                __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) );
                __m128 block_gradDepthX = _mm_load_ps(_depthGradXPyr+block_i);
                __m128 block_gradDepthY = _mm_load_ps(_depthGradYPyr+block_i);
                __m128 block_gradGrayX = _mm_load_ps(_grayGradXPyr+block_i);
                __m128 block_gradGrayY = _mm_load_ps(_grayGradYPyr+block_i);
                __m128 salient_pts = _mm_or_ps( _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(block_gradDepthX, _depth_saliency_), _mm_cmplt_ps(block_gradDepthX, _depth_saliency_neg) ), _mm_or_ps( _mm_cmpgt_ps(block_gradDepthY, _depth_saliency_), _mm_cmplt_ps(block_gradDepthY, _depth_saliency_neg) ) ),
                                                _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( block_gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayX, _gray_saliency_neg ) ), _mm_or_ps( _mm_cmpgt_ps( block_gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayY, _gray_saliency_neg ) ) ) );
                _mm_store_ps(_valid_pt+block_i, _mm_and_ps( valid_depth_pts, salient_pts ) );
            }
        }
    }
    else
    {
        for(size_t r=0; r < nRows; r++)
        {
            __m128 sin_phi = _mm_set1_ps(v_sinPhi[r]);
            __m128 cos_phi = _mm_set1_ps(v_cosPhi[r]);

            size_t block_i = r*nCols;
            for(size_t c=0; c < nCols; c+=4, block_i+=4)
            {
                __m128 block_depth = _mm_load_ps(_depth+block_i);
                __m128 sin_theta = _mm_load_ps(&v_sinTheta[c]);
                __m128 cos_theta = _mm_load_ps(&v_cosTheta[c]);

                __m128 block_x = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, sin_theta) );
                __m128 block_y = _mm_mul_ps( block_depth, sin_phi );
                __m128 block_z = _mm_mul_ps( block_depth, _mm_mul_ps(cos_phi, cos_theta) );
                _mm_store_ps(_x+block_i, block_x);
                _mm_store_ps(_y+block_i, block_y);
                _mm_store_ps(_z+block_i, block_z);

                //_mm_store_ps(_valid_pt+block_i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) );
                __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) );
                __m128 block_gradDepthX = _mm_load_ps(_depthGradXPyr+block_i);
                __m128 block_gradDepthY = _mm_load_ps(_depthGradYPyr+block_i);
                __m128 block_gradGrayX = _mm_load_ps(_grayGradXPyr+block_i);
                __m128 block_gradGrayY = _mm_load_ps(_grayGradYPyr+block_i);
                __m128 salient_pts = _mm_or_ps( _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(block_gradDepthX, _depth_saliency_), _mm_cmplt_ps(block_gradDepthX, _depth_saliency_neg) ), _mm_or_ps( _mm_cmpgt_ps(block_gradDepthY, _depth_saliency_), _mm_cmplt_ps(block_gradDepthY, _depth_saliency_neg) ) ),
                                                _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( block_gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayX, _gray_saliency_neg ) ), _mm_or_ps( _mm_cmpgt_ps( block_gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayY, _gray_saliency_neg ) ) ) );
                _mm_store_ps(_valid_pt+block_i, _mm_and_ps( valid_depth_pts, salient_pts ) );
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
    cout << " ProjectionModel::computeSphereXYZ " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}


/*! Compute the 3D points XYZ according to the pinhole camera model. */
void ProjectionModel::computePinholeXYZ(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels)
{
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    float *_depth = reinterpret_cast<float*>(depth_img.data);

    // Make LUT to store the values of the 3D points of the source sphere
    xyz.resize(imgSize,3);

#if !(_SSE3) // # ifdef !__SSE3__

    validPixels = Eigen::VectorXi::Ones(imgSize);

    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(size_t r=0;r<nRows; r++)
    {
        size_t row_pix = r*nCols;
        for(size_t c=0;c<nCols; c++)
        {
            int i = row_pix + c;
            xyz(i,2) = _depth[i]; //xyz(i,2) = 0.001f*depthSrcPyr[pyrLevel].at<unsigned short>(r,c);

            //Compute the 3D coordinates of the pij of the source frame
            //cout << depthSrcPyr[pyrLevel].type() << " xyz " << i << " x " << xyz(i,2) << " thres " << min_depth_ << " " << max_depth_ << endl;
            if(min_depth_ < xyz(i,2) && xyz(i,2) < max_depth_) //Compute the jacobian only for the valid points
            {
                xyz(i,0) = (c - ox) * xyz(i,2) * inv_fx;
                xyz(i,1) = (r - oy) * xyz(i,2) * inv_fy;
            }
            else
                validPixels(i) = 0;

//            cout << i << " pt " << xyz.block(i,0,1,3) << " c " << c << " ox " << ox << " inv_fx " << inv_fx
//                      << " min_depth_ " << min_depth_ << " max_depth_ " << max_depth_ << endl;
//            mrpt::system::pause();
        }
    }

#else

    validPixels.resize(imgSize);
    float *_valid_pt = reinterpret_cast<float*>(&validPixels(0));

    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);

    std::vector<float> idx(nCols);
    std::iota(idx.begin(), idx.end(), 0.f);
    float *_idx = &idx[0];

//#if !(_AVX) // Use _SSE3
    cout << " computePinholeXYZ _SSE3 " << nRows << "x" << nCols << " = " << imgSize << " pts \n";
    assert(nCols % 4 == 0); // Make sure that the image columns are aligned

//    cout << " alignment 16 " << (((unsigned long)_x & 15) == 0) << " \n";
//    cout << " alignment 16 " << (((unsigned long)_y & 15) == 0) << " \n";
//    cout << " alignment 16 " << (((unsigned long)_z & 15) == 0) << " \n";

    __m128 _inv_fx = _mm_set1_ps(inv_fx);
    __m128 _inv_fy = _mm_set1_ps(inv_fy);
    __m128 _ox = _mm_set1_ps(ox);
    __m128 _oy = _mm_set1_ps(oy);
    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
    for(size_t r=0; r < nRows; r++)
    {
        __m128 _r = _mm_set1_ps(r);
        size_t block_i = r*nCols;
        for(size_t c=0; c < nCols; c+=4, block_i+=4)
        {
            __m128 block_depth = _mm_load_ps(_depth+block_i);
            __m128 block_c = _mm_load_ps(_idx+c);

            __m128 block_x = _mm_mul_ps( block_depth, _mm_mul_ps(_inv_fx, _mm_sub_ps(block_c, _ox) ) );
            __m128 block_y = _mm_mul_ps( block_depth, _mm_mul_ps(_inv_fy, _mm_sub_ps(_r, _oy) ) );
            _mm_store_ps(_x+block_i, block_x);
            _mm_store_ps(_y+block_i, block_y);
            _mm_store_ps(_z+block_i, block_depth);

            __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) );
            _mm_store_ps(_valid_pt+block_i, valid_depth_pts );
        }
    }

//#else // Use _AVX
//    cout << " computePinholeXYZ _AVX " << imgSize << " pts \n";
//    assert(nCols % 8 == 0); // Make sure that the image columns are aligned

//    // Hack for unaligned (32bytes). Eigen should have support for AVX soon
//    bool b_depth_aligned = (((unsigned long)_depth & 31) == 0);
//    size_t hack_padding_xyz = 0, hack_padding_validPixels = 0;
//    bool b_xyz_aligned = (((unsigned long)_x & 31) == 0);
//    if(!b_xyz_aligned)
//    {
//        xyz.resize(imgSize+8,3);
//        hack_padding_xyz = 4;
//    }
//    bool b_validPixels_aligned = (((unsigned long)_valid_pt & 31) == 0);
//    if(!b_validPixels_aligned)
//    {
//        validPixels.resize(imgSize+8);
//        hack_padding_validPixels = 4;
//    }

//    cout << " alignment 32 depth " << (((unsigned long)_depth & 31) == 0) << " \n";
//    cout << " alignment 32 x " << (((unsigned long)_x & 31) == 0) << " \n";
//    cout << " alignment 32 y " << (((unsigned long)_y & 31) == 0) << " \n";
//    cout << " alignment 32 z " << (((unsigned long)_z & 31) == 0) << " \n";
//    cout << " alignment 32 validPixels " << (((unsigned long)_valid_pt & 31) == 0) << " \n";
//    cout << " alignment 16 validPixels " << (((unsigned long)_valid_pt & 15) == 0) << " \n";
////    cout << " alignment 64 " << (((unsigned long)_x & 63) == 0) << " \n";

//    __m256 _inv_fx = _mm256_set1_ps(inv_fx);
//    __m256 _inv_fy = _mm256_set1_ps(inv_fy);
//    __m256 _ox = _mm256_set1_ps(ox);
//    __m256 _oy = _mm256_set1_ps(oy);
//    __m256 _min_depth_ = _mm256_set1_ps(min_depth_);
//    __m256 _max_depth_ = _mm256_set1_ps(max_depth_);
//    if(b_depth_aligned)
//    {
//        for(size_t r=0; r < nRows; r++)
//        {
//            __m256 _r = _mm256_set1_ps(r);
////            cout << " Set _r \n";
//            size_t block_i = r*nCols;
//            size_t block_i_xyz = r*nCols + hack_padding_xyz;
//            size_t block_i_valid = r*nCols + hack_padding_validPixels;
//            for(size_t c=0; c < nCols; c+=8, block_i+=8, block_i_xyz+=8, block_i_valid+=8)
//            {
//                __m256 block_depth = _mm256_load_ps(_depth+block_i);
//                __m256 block_c = _mm256_load_ps(_idx+c);
////                cout << " Load depth \n";

//                __m256 block_x = _mm256_mul_ps( block_depth, _mm256_mul_ps(_inv_fx, _mm256_sub_ps(block_c, _ox) ) );
////                cout << " operation \n";
//                __m256 block_y = _mm256_mul_ps( block_depth, _mm256_mul_ps(_inv_fy, _mm256_sub_ps(_r, _oy) ) );
//                _mm256_store_ps(_x+block_i_xyz, block_x);
////                cout << " store \n";
////                cout << " alignment 32 " << (((unsigned long)(_x+block_i) & 31) == 0) << " \n";
//                _mm256_store_ps(_y+block_i_xyz, block_y);
//                _mm256_store_ps(_z+block_i_xyz, block_depth);

////                cout << " calc valid_depth_pts \n";

//                __m256 valid_depth_pts = _mm256_and_ps( _mm256_cmp_ps(_min_depth_, block_depth, _CMP_LT_OQ), _mm256_cmp_ps(block_depth, _max_depth_, _CMP_LT_OQ) );
////                cout << " store valid_depth_pts \n";
//                _mm256_store_ps(_valid_pt+block_i_valid, valid_depth_pts );
////                cout << " cycle \n";
//            }
//        }
//        if(!b_xyz_aligned)
//            xyz = xyz.block(4,0,imgSize,3);
//        if(!b_validPixels_aligned)
//            validPixels = validPixels.block(4,0,imgSize,1);
//    }
//    else // !b_depth_aligned
//    {
//        assert(!b_xyz_aligned);

//        for(size_t r=0; r < nRows; r++)
//        {
//            __m256 _r = _mm256_set1_ps(r);
////            cout << " Set _r \n";
//            size_t block_i = r*nCols+4;
////            size_t block_i_xyz = r*nCols + hack_padding_xyz;
//            size_t block_i_valid = r*nCols + hack_padding_validPixels;
//            for(size_t c=4; c < nCols-4; c+=8, block_i+=8, block_i_valid+=8)//, block_i_xyz+=8)
//            {
//                __m256 block_depth = _mm256_load_ps(_depth+block_i);
//                __m256 block_c = _mm256_load_ps(_idx+c);
////                cout << " Load depth \n";

//                __m256 block_x = _mm256_mul_ps( block_depth, _mm256_mul_ps(_inv_fx, _mm256_sub_ps(block_c, _ox) ) );
////                cout << " operation \n";
//                __m256 block_y = _mm256_mul_ps( block_depth, _mm256_mul_ps(_inv_fy, _mm256_sub_ps(_r, _oy) ) );
//                _mm256_store_ps(_x+block_i, block_x);
////                cout << " store \n";
////                cout << " alignment 32 " << (((unsigned long)(_x+block_i) & 31) == 0) << " \n";
//                _mm256_store_ps(_y+block_i, block_y);
//                _mm256_store_ps(_z+block_i, block_depth);

////                cout << " calc valid_depth_pts \n";

//                __m256 valid_depth_pts = _mm256_and_ps( _mm256_cmp_ps(_min_depth_, block_depth, _CMP_LT_OQ), _mm256_cmp_ps(block_depth, _max_depth_, _CMP_LT_OQ) );
////                cout << " store valid_depth_pts \n";
//                _mm256_store_ps(_valid_pt+block_i_valid, valid_depth_pts );

//                cout << " cycle " << block_i << " " << block_i_valid << "\n";

////                if(block_i_valid==112)
////                {
////                    for(size_t i=0; i < 8; i++)
////                        cout << i << " pt " << xyz.block(block_i_valid+i,0,1,3) << " valid " << validPixels(block_i_valid+i) << endl;

////                    float *b_depth = reinterpret_cast<float*>(&block_depth);
////                    int *b_valid = reinterpret_cast<int*>(&valid_depth_pts);
////                    cout << "b_depth " << b_depth[0] << " " << b_depth[1] << " " << b_depth[2] << " " << b_depth[3] << " " << b_depth[4] << " " << b_depth[5] << " " << b_depth[6] << " " << b_depth[7] << " \n";
////                    cout << "b_valid " << b_valid[0] << " " << b_valid[1] << " " << b_valid[2] << " " << b_valid[3] << " " << b_valid[4] << " " << b_valid[5] << " " << b_valid[6] << " " << b_valid[7] << " \n";
////                    cout << "min_depth_ " << min_depth_ << " max_depth_ " << max_depth_ << " \n";
////                }

//                // Compute unaligned pixels
//                size_t row_pix = r*nCols;
//                for(size_t c=0;c<4; c++)
//                {
//                    int i = row_pix + c;
//                    xyz(i,2) = _depth[i]; //xyz(i,2) = 0.001f*depthSrcPyr[pyrLevel].at<unsigned short>(r,c);

//                    //Compute the 3D coordinates of the pij of the source frame
//                    //cout << depthSrcPyr[pyrLevel].type() << " xyz " << i << " x " << xyz(i,2) << " thres " << min_depth_ << " " << max_depth_ << endl;
//                    if(min_depth_ < xyz(i,2) && xyz(i,2) < max_depth_) //Compute the jacobian only for the valid points
//                    {
//                        xyz(i,0) = (c - ox) * xyz(i,2) * inv_fx;
//                        xyz(i,1) = (r - oy) * xyz(i,2) * inv_fy;
//                        validPixels(i) = -1;
//                    }
//                    else
//                        validPixels(i) = 0;
//                }
//                for(size_t c=nCols-4;c<nCols; c++)
//                {
//                    int i = row_pix + c;
//                    xyz(i,2) = _depth[i]; //xyz(i,2) = 0.001f*depthSrcPyr[pyrLevel].at<unsigned short>(r,c);

//                    //Compute the 3D coordinates of the pij of the source frame
//                    //cout << depthSrcPyr[pyrLevel].type() << " xyz " << i << " x " << xyz(i,2) << " thres " << min_depth_ << " " << max_depth_ << endl;
//                    if(min_depth_ < xyz(i,2) && xyz(i,2) < max_depth_) //Compute the jacobian only for the valid points
//                    {
//                        xyz(i,0) = (c - ox) * xyz(i,2) * inv_fx;
//                        xyz(i,1) = (r - oy) * xyz(i,2) * inv_fy;
//                        validPixels(i) = -1;
//                    }
//                    else
//                        validPixels(i) = 0;
//                }

//            }
//        }
//    }

//    // Test the AVX result
//    Eigen::MatrixXf xyz_2(imgSize,3);
//    Eigen::VectorXf validPixels_2(imgSize);
//    for(size_t r=0;r<nRows; r++)
//    {
//        size_t row_pix = r*nCols;
//        for(size_t c=0;c<nCols; c++)
//        {
//            int i = row_pix + c;
//            xyz_2(i,2) = _depth[i]; //xyz(i,2) = 0.001f*depthSrcPyr[pyrLevel].at<unsigned short>(r,c);

//            //Compute the 3D coordinates of the pij of the source frame
//            if(min_depth_ < xyz_2(i,2) && xyz_2(i,2) < max_depth_) //Compute the jacobian only for the valid points
//            {
//                xyz_2(i,0) = (c - ox) * xyz_2(i,2) * inv_fx;
//                xyz_2(i,1) = (r - oy) * xyz_2(i,2) * inv_fy;
//                validPixels_2(i) = -1;
//            }
//            else
//                validPixels_2(i) = 0;

//            cout << i << " pt " << xyz.block(i,0,1,3) << " vs " << xyz_2.block(i,0,1,3) << " valid " << validPixels(i) << " " << validPixels_2(i) << endl;
////            assert(validPixels_2(i) == validPixels(i));
////            if(validPixels(i))
////            {
////                //cout << i << " pt3D " << xyz.block(i,0,1,3) << " vs " << xyz_2.block(i,0,1,3) << " valid " << validPixels(i) << " " << validPixels_2(i) << endl;
////                assert(xyz_2(i,0) - xyz(i,0) < 1e-6);
////                assert(xyz_2(i,1) - xyz(i,1) < 1e-6);
////                assert(xyz_2(i,2) - xyz(i,2) < 1e-6);
////            }
//        }
//    }
//#endif // endif vectorization options
#endif // endif vectorization

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " ProjectionModel::computePinholeXYZ " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the 3D points XYZ according to the pinhole camera model. */
void ProjectionModel::computePinholeXYZ_saliency(Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                                const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                                const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY,
                                                const float thres_saliency_gray, const float thres_saliency_depth)
{
#if !(_SSE)
    assert(0); // TODO: implement regular (non SSE)
#endif

    // TODO: adapt the sse formulation
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    assert(nCols % 4 == 0); // Make sure that the image columns are aligned

    float *_depth = reinterpret_cast<float*>(depth_img.data);
    float *_depthGradXPyr = reinterpret_cast<float*>(depth_gradX.data);
    float *_depthGradYPyr = reinterpret_cast<float*>(depth_gradY.data);
    float *_grayGradXPyr = reinterpret_cast<float*>(intensity_gradX.data);
    float *_grayGradYPyr = reinterpret_cast<float*>(intensity_gradY.data);

    // Make LUT to store the values of the 3D points of the source sphere
    xyz.resize(imgSize,3);
    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);

    //validPixels = Eigen::VectorXi::Ones(imgSize);
    validPixels.resize(imgSize);
    float *_valid_pt = reinterpret_cast<float*>(&validPixels(0));

    std::vector<float> idx(nCols);
    std::iota(idx.begin(), idx.end(), 0.f);
    float *_idx = &idx[0];

    __m128 _inv_fx = _mm_set1_ps(inv_fx);
    __m128 _inv_fy = _mm_set1_ps(inv_fy);
    __m128 _ox = _mm_set1_ps(ox);
    __m128 _oy = _mm_set1_ps(oy);

    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
    __m128 _depth_saliency_ = _mm_set1_ps(thres_saliency_depth);
    __m128 _gray_saliency_ = _mm_set1_ps(thres_saliency_gray);
    __m128 _depth_saliency_neg = _mm_set1_ps(-thres_saliency_depth);
    __m128 _gray_saliency_neg = _mm_set1_ps(-thres_saliency_gray);

    for(size_t r=0; r < nRows; r++)
    {
        __m128 _r = _mm_set1_ps(r);
        size_t block_i = r*nCols;
        for(size_t c=0; c < nCols; c+=4, block_i+=4)
        {
            __m128 block_depth = _mm_load_ps(_depth+block_i);
            __m128 block_c = _mm_load_ps(_idx+c);

            __m128 block_x = _mm_mul_ps( block_depth, _mm_mul_ps(_inv_fx, _mm_sub_ps(block_c, _ox) ) );
            __m128 block_y = _mm_mul_ps( block_depth, _mm_mul_ps(_inv_fy, _mm_sub_ps(_r, _oy) ) );
            _mm_store_ps(_x+block_i, block_x);
            _mm_store_ps(_y+block_i, block_y);
            _mm_store_ps(_z+block_i, block_depth);

            //_mm_store_ps(_valid_pt+block_i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) );
            __m128 valid_depth_pts = _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) );
            __m128 block_gradDepthX = _mm_load_ps(_depthGradXPyr+block_i);
            __m128 block_gradDepthY = _mm_load_ps(_depthGradYPyr+block_i);
            __m128 block_gradGrayX = _mm_load_ps(_grayGradXPyr+block_i);
            __m128 block_gradGrayY = _mm_load_ps(_grayGradYPyr+block_i);
            __m128 salient_pts = _mm_or_ps( _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(block_gradDepthX, _depth_saliency_), _mm_cmplt_ps(block_gradDepthX, _depth_saliency_neg) ), _mm_or_ps( _mm_cmpgt_ps(block_gradDepthY, _depth_saliency_), _mm_cmplt_ps(block_gradDepthY, _depth_saliency_neg) ) ),
                                            _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( block_gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayX, _gray_saliency_neg ) ), _mm_or_ps( _mm_cmpgt_ps( block_gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayY, _gray_saliency_neg ) ) ) );
            _mm_store_ps(_valid_pt+block_i, _mm_and_ps( valid_depth_pts, salient_pts ) );
        }
    }

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " ProjectionModel::computePinholeXYZ_sse SALIENT " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}
