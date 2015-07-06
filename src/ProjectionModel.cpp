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

using namespace std;

ProjectionModel::ProjectionModel() :
    projection_model(PINHOLE),  // SPHERICAL
    min_depth_(0.3f),
    max_depth_(20.f)
{
    // For sensor_type = KINECT
    cameraMatrix << 262.5, 0., 1.5950e+02,
                    0., 262.5, 1.1950e+02,
                    0., 0., 1.;

    setProjectionModel(projection_model);
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
void ProjectionModel::reconstruct3D_unitSphere()
{
    // Make LUT to store the values of the 3D points of the source sphere
    MatrixXf unit_sphere;
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
void ProjectionModel::reconstruct3D_spherical(const cv::Mat & depth_img, MatrixXf & xyz, VectorXi & validPixels) // , std::vector<int> & validPixels)
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
    int *_valid_pt = reinterpret_cast<int*>(&validPixels(0));
    for(int i=0; i < imgSize; i++, _valid_pt++)
        _valid_pt[i] = i;  // *(_valid_pt++) = i;
    int *_valid_pt = reinterpret_cast<int*>(&validPixels(0));

    // Compute the Unit Sphere: store the values of the trigonometric functions
    VectorXf v_sinTheta(nCols);
    VectorXf v_cosTheta(nCols);
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
    VectorXf v_sinPhi( v_sinTheta.block(start_row,0,nRows,1) );
    VectorXf v_cosPhi( v_cosTheta.block(start_row,0,nRows,1) );

//Compute the 3D coordinates of the depth image
#if !(_SSE3) // # ifdef __SSE3__

    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(size_t r=0;r<nRows;r++)
    {
        size_t i = r*nCols;
        for(size_t c=0;c<nCols;c++,i++)
        {
            float depth1 = _depth[i];
            if(min_depth_ < depth1 && depth1 < max_depth_) //Compute the jacobian only for the valid points
            {
                //cout << " depth1 " << depth1 << " phi " << phi << " v_sinTheta[c] " << v_sinTheta[c] << endl;
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
    cout << " reconstruct3D_spherical _SSE3 " << imgSize << " pts \n";

    assert(nCols % 4 == 0); // Make sure that the image columns are aligned
    assert(nRows % 2 == 0);

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

                __m128 block_valid = _mm_load_ps(_valid_pt+block_i);
                _mm_store_ps(_valid_pt+block_i, _mm_and_ps( block_valid, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) ) );
                //_mm_store_ps(_valid_pt+block_i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) );
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

                __m128 block_valid = _mm_load_ps(_valid_pt+block_i);
                _mm_store_ps(_valid_pt+block_i, _mm_and_ps( block_valid, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) ) );
                //_mm_store_ps(_valid_pt+block_i, _mm_and_ps( _mm_cmplt_ps(_min_depth_, block_depth), _mm_cmplt_ps(block_depth, _max_depth_) ) );
            }
        }
    }
//    // Compute the transformation of those points which do not enter in a block
//    const Matrix3f rotation_transposed = Rt.block(0,0,3,3).transpose();
//    const Matrix<float,1,3> translation_transposed = Rt.block(0,3,3,1).transpose();
//    for(int i=block_end; i < imgSize; i++)
//    {
//        ***********
//        output_pts.block(i,0,1,3) = input_pts.block(i,0,1,3) * rotation_transposed + translation_transposed;
//    }

#endif

    for(size_t i=0; i < imgSize; i++)
        cout << _valid_pt[i] << " ";
    assert(0);

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " ProjectionModel::reconstruct3D_spherical " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
void ProjectionModel::reconstruct3D_spherical_saliency(MatrixXf & xyz, VectorXi & validPixels,
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
    VectorXf v_sinTheta(nCols);
    VectorXf v_cosTheta(nCols);
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
    VectorXf v_sinPhi( v_sinTheta.block(start_row,0,nRows,1) );
    VectorXf v_cosPhi( v_cosTheta.block(start_row,0,nRows,1) );

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
    cout << " reconstruct3D_spherical _SSE3 " << imgSize << " pts \n";

    //Compute the 3D coordinates of the pij of the source frame
    MatrixXf xyz_tmp(imgSize,3);
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
                __m128 salient_pts = _mm_or_ps( _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps(block_gradDepthX, _depth_saliency_), _mm_cmplt_ps(block_gradDepthX, _depth_saliency_neg) ),
                                                           _mm_or_ps( _mm_cmpgt_ps(block_gradDepthY, _depth_saliency_), _mm_cmplt_ps(block_gradDepthY, _depth_saliency_neg) ) ),
                                                _mm_or_ps( _mm_or_ps( _mm_cmpgt_ps( block_gradGrayX, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayX, _gray_saliency_neg ) ),
                                                           _mm_or_ps( _mm_cmpgt_ps( block_gradGrayY, _gray_saliency_ ), _mm_cmplt_ps( block_gradGrayY, _gray_saliency_neg ) ) ) );
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
    cout << " ProjectionModel::reconstruct3D_spherical " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the 3D points XYZ according to the pinhole camera model. */
void ProjectionModel::reconstruct3D_pinhole(const cv::Mat & depth_img, MatrixXf & xyz, VectorXi & validPixels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    nRows = depth_img.rows;
    nCols = depth_img.cols;
    imgSize = nRows*nCols;

    float *_depth = reinterpret_cast<float*>(depth_img.data);

    // Make LUT to store the values of the 3D points of the source sphere
    xyz.resize(imgSize,3);
    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);

    validPixels.resize(imgSize);
    int *_valid_pt = reinterpret_cast<int*>(&validPixels(0));
    for(int i=0; i < imgSize; i++, _valid_pt++)
        _valid_pt[i] = i;  // *(_valid_pt++) = i;
    int *_valid_pt = reinterpret_cast<int*>(&validPixels(0));

#if !(_SSE3) // # ifdef !__SSE3__

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
                validPixels(i) = -1;

//            cout << i << " pt " << xyz.block(i,0,1,3) << " c " << c << " ox " << ox << " inv_fx " << inv_fx
//                      << " min_depth_ " << min_depth_ << " max_depth_ " << max_depth_ << endl;
//            mrpt::system::pause();
        }
    }

#else

    std::vector<float> idx(nCols);
    std::iota(idx.begin(), idx.end(), 0.f);
    float *_idx = &idx[0];

//#if !(_AVX) // Use _SSE3
    cout << " reconstruct3D_pinhole _SSE3 " << nRows << "x" << nCols << " = " << imgSize << " pts \n";
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
            //_mm_store_ps(_valid_pt+block_i, valid_depth_pts );
            __m128 block_valid = _mm_load_ps(_valid_pt+block_i);
            _mm_store_ps(_valid_pt+block_i, _mm_and_ps(block_valid, valid_depth_pts) );
        }
    }

//#else // Use _AVX
//    cout << " reconstruct3D_pinhole _AVX " << imgSize << " pts \n";
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
//    MatrixXf xyz_2(imgSize,3);
//    VectorXf validPixels_2(imgSize);
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
    cout << " ProjectionModel::reconstruct3D_pinhole " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the 3D points XYZ according to the pinhole camera model. */
void ProjectionModel::reconstruct3D_pinhole_saliency(MatrixXf & xyz, VectorXi & validPixels,
                                                const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                                const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY,
                                                const float thres_saliency_gray, const float thres_saliency_depth)
{
#if !(_SSE)
    assert(0); // TODO: implement regular (non SSE)
#endif

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    // TODO: adapt the sse formulation
    nRows = depth_img.rows;
    nCols = depth_img.cols;
    imgSize = nRows*nCols;
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

    //validPixels = VectorXi::Ones(imgSize);
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
    cout << " ProjectionModel::reconstruct3D_pinhole_sse SALIENT " << depth_img.rows*depth_img.cols << " (" << depth_img.rows << "x" << depth_img.cols << ")" << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Project 3D points XYZ according to the pinhole camera model. */
void ProjectionModel::project_pinhole(MatrixXf & xyz, MatrixXf & pixels)
{
    pixels.resize(xyz.rows(),2);
    float *_r = &pixels(0,0);
    float *_c = &pixels(0,1);

    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < pixels.size(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image_pinhole(pt_xyz);
        pixels(i,0) = warped_pixel.y;
        pixels(i,1) = warped_pixel.x;
    }

#else

    __m128 _fx = _mm_set1_ps(fx);
    __m128 _fy = _mm_set1_ps(fy);
    __m128 _ox = _mm_set1_ps(ox);
    __m128 _oy = _mm_set1_ps(oy);
    for(size_t i=0; i < pixels.size(); i+=4)
    {
        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);

        __m128 block_r = _mm_sum_ps( _mm_div_ps( _mm_mul_ps(_fy, block_y ), block_z ), _oy );
        __m128 block_c = _mm_sum_ps( _mm_div_ps( _mm_mul_ps(_fx, block_x ), block_z ), _ox );

        _mm_store_ps(_r+i, block_r);
        _mm_store_ps(_c+i, block_c);
    }
#endif
};

/*! Project 3D points XYZ according to the spherical camera model. */
void ProjectionModel::project_spherical(MatrixXf & xyz, MatrixXf & pixels)
{
    pixels.resize(xyz.rows(),2);
    float *_r = &pixels(0,0);
    float *_c = &pixels(0,1);

    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < pixels.size(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image_spherical(pt_xyz);
        pixels(i,0) = warped_pixel.y;
        pixels(i,1) = warped_pixel.x;
    }

#else

    assert(__SSE4_1__); // For _mm_extract_epi32

    __m128 _nCols = _mm_set1_ps(nCols);
    __m128 _phi_start = _mm_set1_ps(phi_start);
    __m128 _half_width = _mm_set1_ps(half_width);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(size_t i=0; i < pixels.size(); i+=4)
    {
        //__m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);

        __m128 _y_z = _mm_div_ps( block_y, block_z );
        //float *x_z = reinterpret_cast<float*>(&_x_z);
        float theta[4];
        float phi[4];
        for(size_t j=0; j < 4; j++)
        {
            phi[j] = asin( _mm_extract_epi32(_y_z,j) );
            theta[j] = atan2(_x[i+j], _z[i+j]);
        }
        __m128 _phi = _mm_load_ps(phi);
        __m128 _theta = _mm_load_ps(theta);

        __m128 block_r = _mm_mul_ps( _mm_sub_ps(_phi, _phi_start ), _pixel_angle_inv );
        __m128 block_c = _mm_sum_ps( _mm_mul_ps(_theta, _pixel_angle_inv ), _half_width );

        _mm_store_ps(_r+i, block_r);
        _mm_store_ps(_c+i, block_c);
    }

    for(size_t i=0; i < pixels.size(); i++)
        assert(pixels(i,1) < nCols);

#endif
};

/*! Project 3D points XYZ according to the pinhole camera model. */
void ProjectionModel::projectNN_pinhole(MatrixXf & xyz, MatrixXi & pixels, MatrixXi & visible)
{
    pixels.resize(xyz.rows(),1);
    visible.resize(xyz.rows(),1);
    float *_p = &pixels(0);
    int *_v = &visible(0);

    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < pixels.size(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image_pinhole(pt_xyz);

        int r_transf = round(warped_pixel.y);
        int c_transf = round(warped_pixel.x);
        // cout << i << " Pixel transform " << i/nCols << " " << i%nCols << " " << r_transf << " " << c_transf << endl;
        pixels(i) = r_transf * nCols + c_transf;
        visible(i) = isInImage_pinhole(r_transf, c_transf) ? 1 : 0;
    }

#else

    __m128 _fx = _mm_set1_ps(fx);
    __m128 _fy = _mm_set1_ps(fy);
    __m128 _ox = _mm_set1_ps(ox);
    __m128 _oy = _mm_set1_ps(oy);
    __m128 _nCols = _mm_set1_ps(nCols);
    __m128 _nRows = _mm_set1_ps(nRows);
    __m128 _nzero = _mm_set1_ps(0.f);
    for(size_t i=0; i < pixels.size(); i+=4)
    {
        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);

        __m128 block_r = _mm_sum_ps( _mm_div_ps( _mm_mul_ps(_fy, block_y ), block_z ), _oy );
        __m128 block_c = _mm_sum_ps( _mm_div_ps( _mm_mul_ps(_fx, block_x ), block_z ), _ox );
        __m128 block_p = _mm_sum_ps( _mm_mul_ps(_nCols, block_r ), block_c);

        __m128 block_v =_mm_and_ps( _mm_and_ps( _mm_cmplt_ps(_nzero, block_r), _mm_cmplt_ps(block_r, _nRows) ),
                                    _mm_and_ps( _mm_cmplt_ps(_nzero, block_c), _mm_cmplt_ps(block_c, _nCols) ) );

        _mm_store_ps(_p+i, block_p);
        _mm_store_ps(_v+i, block_v);
    }
#endif
};

/*! Project 3D points XYZ according to the spherical camera model. */
void ProjectionModel::projectNN_spherical(MatrixXf & xyz, MatrixXi & pixels, MatrixXi & visible)
{
    pixels.resize(xyz.rows(),1);
    visible.resize(xyz.rows(),1);
    float *_p = &pixels(0);
    int *_v = &visible(0);

    float *_x = &xyz(0,0);
    float *_y = &xyz(0,1);
    float *_z = &xyz(0,2);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < pixels.size(); i++)
    {
        Vector3f pt_xyz = xyz.block(i,0,1,3).transpose();
        cv::Point2f warped_pixel = project2Image_spherical(pt_xyz);
        int r_transf = round(warped_pixel.y);
        int c_transf = round(warped_pixel.x);
        // cout << i << " Pixel transform " << i/nCols << " " << i%nCols << " " << r_transf << " " << c_transf << endl;
        pixels(i) = r_transf * nCols + c_transf;
        visible(i) = isInImage_pinhole(r_transf, c_transf) ? 1 : 0;
    }

#else

    assert(__SSE4_1__); // For _mm_extract_epi32

    __m128 _nCols = _mm_set1_ps(nCols);
    __m128 _phi_start = _mm_set1_ps(phi_start);
    __m128 _half_width = _mm_set1_ps(half_width);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(size_t i=0; i < pixels.size(); i+=4)
    {
        //__m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);

        __m128 _y_z = _mm_div_ps( block_y, block_z );
        //float *x_z = reinterpret_cast<float*>(&_x_z);
        float theta[4];
        float phi[4];
        for(size_t j=0; j < 4; j++)
        {
            phi[j] = asin( _mm_extract_epi32(_y_z,j) );
            theta[j] = atan2(_x[i+j], _z[i+j]);
        }
        __m128 _phi = _mm_load_ps(phi);
        __m128 _theta = _mm_load_ps(theta);

        __m128 block_r = _mm_mul_ps( _mm_sub_ps(_phi, _phi_start ), _pixel_angle_inv );
        __m128 block_c = _mm_sum_ps( _mm_mul_ps(_theta, _pixel_angle_inv ), _half_width );
        __m128 block_p = _mm_sum_ps( _mm_mul_ps(_nCols, block_r ), block_c);

        __m128 block_v =_mm_and_ps( _mm_and_ps( _mm_cmplt_ps(_nzero, block_r), _mm_cmplt_ps(block_r, _nRows) ) );
                                    //,_mm_and_ps( _mm_cmplt_ps(_nzero, block_c), _mm_cmplt_ps(block_c, _nCols) ) );

        _mm_store_ps(_p+i, block_p);
        _mm_store_ps(_v+i, block_v);
    }

    for(size_t i=0; i < pixels.size(); i++)
        assert(pixels(i,1) < nCols);

#endif
};

/*! Compute the 2x6 jacobian matrices of the composition (warping+rigidTransformation) using the pinhole camera model. */
void ProjectionModel::computeJacobians26_pinhole(MatrixXf & xyz_tf, MatrixXf & jacobians_aligned)
{
    jacobians_aligned.resize(xyz_tf.rows(), 10); // It has 10 columns instead of 10 because the positions (1,0) and (0,1) of the 2x6 jacobian are alwatys 0

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        float inv_transf_z = 1.0/xyz_transf(2);

        //Derivative with respect to x
        jacobians_aligned(i,0)=fx*inv_transf_z;
        //jacobianWarpRt(1,0)=0.f;

        //Derivative with respect to y
        //jacobianWarpRt(0,1)=0.f;
        jacobians_aligned(i,1)=fy*inv_transf_z;

        //Derivative with respect to z
        float inv_transf_z_2 = inv_transf_z*inv_transf_z;
        jacobians_aligned(i,2)=-fx*xyz_transf(0)*inv_transf_z_2;
        jacobians_aligned(i,3)=-fy*xyz_transf(1)*inv_transf_z_2;

        //Derivative with respect to \w_x
        jacobians_aligned(i,4)=-fx*xyz_transf(1)*xyz_transf(0)*inv_transf_z_2;
        jacobians_aligned(i,5)=-fy*(1+xyz_transf(1)*xyz_transf(1)*inv_transf_z_2);

        //Derivative with respect to \w_y
        jacobians_aligned(i,6)= fx*(1+xyz_transf(0)*xyz_transf(0)*inv_transf_z_2);
        jacobians_aligned(i,7)= fy*xyz_transf(0)*xyz_transf(1)*inv_transf_z_2;

        //Derivative with respect to \w_z
        jacobians_aligned(i,8)=-fx*xyz_transf(1)*inv_transf_z;
        jacobians_aligned(i,9)= fy*xyz_transf(0)*inv_transf_z;
    }

#else
    // Pointers to the jacobian elements
    float *_j00 = &jacobians_aligned(0,0);
    float *_j11 = &jacobians_aligned(0,1);
    float *_j02 = &jacobians_aligned(0,2);
    float *_j12 = &jacobians_aligned(0,3);
    float *_j03 = &jacobians_aligned(0,4);
    float *_j13 = &jacobians_aligned(0,5);
    float *_j04 = &jacobians_aligned(0,6);
    float *_j14 = &jacobians_aligned(0,7);
    float *_j05 = &jacobians_aligned(0,8);
    float *_j15 = &jacobians_aligned(0,9);

    __m128 _fx = _mm_set1_ps(fx);
    __m128 _fy = _mm_set1_ps(fy);
    __m128 _one = _mm_set1_ps(1.f);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _inv_z = _mm_div_ps(_one, block_z);

        __m128 block_j00 = _mm_mul_ps(_fx, _inv_z);
        __m128 block_j11 = _mm_mul_ps(_fy, _inv_z);
        __m128 block_j02 = _mm_xor_ps( _mm_mul_ps(block_x, _mm_mul_ps(_inv_z, block_j00) ), _mm_set1_ps(-0.f) );
        __m128 block_j12_neg = _mm_mul_ps(block_y, _mm_mul_ps(_inv_z, block_j11) );

        _mm_store_ps(_j00+i, block_j00 );
        _mm_store_ps(_j11+i, block_j11 );
        _mm_store_ps(_j02+i, block_j02);
        _mm_store_ps(_j12+i, _mm_xor_ps(block_j12, _mm_set1_ps(-0.f)) );
        _mm_store_ps(_j03+i, _mm_mul_ps( block_y, block_j02) );
        _mm_store_ps(_j13+i, _mm_sub_ps(_mm_mul_ps( block_y, block_j12), _fy );
        _mm_store_ps(_j04+i, _mm_sub_ps(_fx, _mm_mul_ps( block_x, block_j02) );
        _mm_store_ps(_j14+i, _mm_mul_ps( block_0, block_j12_neg) );
        _mm_store_ps(_j05+i, _mm_xor_ps( _mm_mul_ps(block_y, block_j00), _mm_set1_ps(-0.f) ) );
        _mm_store_ps(_j16+i, _mm_mul_ps(block_x, block_j11));
    }
#endif
}

/*! Compute the 2x6 jacobian matrices of the composition (warping+rigidTransformation) using the spherical camera model. */
void ProjectionModel::computeJacobians26_spherical(MatrixXf & xyz_tf, MatrixXf & jacobians_aligned)
{
    jacobians_aligned.resize(xyz_tf.rows(), 11); // It has 10 columns instead of 10 because the positions (1,0) and (0,1) of the 2x6 jacobian are alwatys 0

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        float inv_transf_z = 1.0/xyz_transf(2);

        float dist = xyz_transf.norm();
        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz_transf(1)*xyz_transf(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );

        jacobians_aligned(i,0) = commonDer_c * xyz_transf(2);
        //jacobianWarpRt(0,1) = 0.f;
        jacobians_aligned(i,3) = -commonDer_c * xyz_transf(0);
    //        jacobianWarpRt(1,0) = commonDer_r * xyz_transf(0) * xyz_transf(1);
        jacobians_aligned(i,2) =-commonDer_r * x2_z2;
    //        jacobianWarpRt(1,2) = commonDer_r * xyz_transf(2) * xyz_transf(1);
        float commonDer_r_y = commonDer_r * xyz_transf(1);
        jacobians_aligned(i,1) = commonDer_r_y * xyz_transf(0);
        jacobians_aligned(i,4) = commonDer_r_y * xyz_transf(2);

        jacobians_aligned(i,5) = jacobians_aligned(i,3) * xyz_transf(1);
        jacobians_aligned(i,7) = jacobians_aligned(i,0) * xyz_transf(2) - jacobians_aligned(i,3) * xyz_transf(0);
        jacobians_aligned(i,9) =-jacobians_aligned(i,0) * xyz_transf(1);
        jacobians_aligned(i,6) =-jacobians_aligned(i,2) * xyz_transf(2) + jacobians_aligned(i,4) * xyz_transf(1);
        jacobians_aligned(1,8) = jacobians_aligned(i,1) * xyz_transf(2) - jacobians_aligned(i,4) * xyz_transf(0);
        jacobians_aligned(1,10) =-jacobians_aligned(i,1) * xyz_transf(1) + jacobians_aligned(i,2) * xyz_transf(0);
    }

#else
    // Pointers to the jacobian elements
    float *_j00 = &jacobians_aligned(0,0);
    float *_j10 = &jacobians_aligned(0,1);
    float *_j11 = &jacobians_aligned(0,2);
    float *_j02 = &jacobians_aligned(0,3);
    float *_j12 = &jacobians_aligned(0,4);
    float *_j03 = &jacobians_aligned(0,5);
    float *_j13 = &jacobians_aligned(0,6);
    float *_j04 = &jacobians_aligned(0,7);
    float *_j14 = &jacobians_aligned(0,8);
    float *_j05 = &jacobians_aligned(0,9);
    float *_j15 = &jacobians_aligned(0,10);

    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _x2_z2 = _mm_sum_ps(_mm_mul_ps(block_x, block_x), _mm_mul_ps(block_z, block_z) );
        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
        __m128 _dist2 = _mm_sum_ps(_mm_mul_ps(block_y, block_y), _x2_z2 );
        __m128 _dist = _mm_sqrt_ps(_dist2);
        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, block_y);

        __m128 block_j00 = _mm_mul_ps(_commonDer_c, block_z);
        __m128 block_j10 = _mm_mul_ps(_commonDer_r_y, block_x);
        __m128 block_j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
        __m128 block_j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, block_x), _mm_set1_ps(-0.f));
        __m128 block_j12 = _mm_mul_ps(_commonDer_r_y, block_z );

        _mm_store_ps(_j00+i, block_j00 );
        _mm_store_ps(_j10+i, block_j10 );
        _mm_store_ps(_j11+i, block_j11 );
        _mm_store_ps(_j02+i, block_j02 );
        _mm_store_ps(_j12+i, block_j12 );
        _mm_store_ps(_j03+i, _mm_mul_ps( block_j02, block_y ) );
        _mm_store_ps(_j13+i, _mm_sub_ps(_mm_mul_ps(block_j12, block_y), _mm_mul_ps(block_j11, block_z) ) );
        _mm_store_ps(_j04+i, _mm_sub_ps(_mm_mul_ps(block_j00, block_z), _mm_mul_ps(block_j02, block_x) ) );
        _mm_store_ps(_j14+i, _mm_sub_ps(_mm_mul_ps(block_j10, block_z), _mm_mul_ps(block_j12, block_x) ) );
        _mm_store_ps(_j05+i, _mm_xor_ps( _mm_mul_ps(block_j00, block_y), _mm_set1_ps(-0.f) ) );
        _mm_store_ps(_j15+i, _mm_sub_ps(_mm_mul_ps(block_j11, block_x), _mm_mul_ps(block_j10, block_y) ) );
    }
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
void ProjectionModel::computeJacobiansPhoto_pinhole(MatrixXf & xyz_tf, const float stdDevPhoto_inv, VectorXf & weights, MatrixXf & jacobians, float *_grayGradX, float *_grayGradY)
{
    jacobians.resize(xyz_tf.rows(), 6);

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);
    float *_weight = &weights(0,0);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT_pinhole(xyz_transf, jacobianWarpRt);
        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians.block(i,0,1,6) = ((sqrt(wEstimDepth(i)) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;
    }

#else
    // Pointers to the jacobian elements
    float *_j0 = &jacobians(0,0);
    float *_j1 = &jacobians(0,1);
    float *_j2 = &jacobians(0,2);
    float *_j3 = &jacobians(0,3);
    float *_j4 = &jacobians(0,4);
    float *_j5 = &jacobians(0,5);

    __m128 _fx = _mm_set1_ps(fx);
    __m128 _fy = _mm_set1_ps(fy);
    __m128 _one = _mm_set1_ps(1.f);
    __m128 block_stdDevInv = _mm_set1_ps(stdDevPhoto_inv);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_gradX = _mm_load_ps(_grayGradX+i);
        __m128 block_gradY = _mm_load_ps(_grayGradY+i);
        __m128 block_weight = _mm_load_ps(_weight+i);
        __m128 block_weight_stdDevInv = _mm_mul_ps(block_stdDevInv, block_weight);
        __m128 block_gradX_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradX);
        __m128 block_gradY_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradY);

        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _inv_z = _mm_div_ps(_one, block_z);

        __m128 block_j00 = _mm_mul_ps(_fx, _inv_z);
        __m128 block_j11 = _mm_mul_ps(_fy, _inv_z);
        __m128 block_j02 = _mm_xor_ps( _mm_mul_ps(block_x, _mm_mul_ps(_inv_z, block_j00) ), _mm_set1_ps(-0.f) );
        __m128 block_j12_neg = _mm_mul_ps(block_y, _mm_mul_ps(_inv_z, block_j11) );
        __m128 block_j03 = _mm_mul_ps( block_y, block_j02);
        __m128 block_j13_neg = _mm_sum_ps(_mm_mul_ps( block_y, block_j12_neg), _fy );
        __m128 block_j04_neg = _mm_sub_ps(_mm_mul_ps( block_x, block_j02), _fx );
        __m128 block_j14 = _mm_mul_ps( block_x, block_j12_neg);
        __m128 block_j05_neg = _mm_mul_ps(block_y, block_j00);
        __m128 block_j15 = _mm_mul_ps(block_x, block_j11);

        _mm_store_ps(_j0+i, _mm_mul_ps(block_gradX_weight, block_j00) );
        _mm_store_ps(_j1+i, _mm_mul_ps(block_gradY_weight, block_j11) );
        _mm_store_ps(_j2+i, _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j02), _mm_mul_ps(block_gradY_weight, block_j12_neg) ) );
        _mm_store_ps(_j3+i, _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j03), _mm_mul_ps(block_gradY_weight, block_j13) ) );
        _mm_store_ps(_j4+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j14), _mm_mul_ps(block_gradX_weight, block_j04_neg) ) );
        _mm_store_ps(_j5+i, _mm_sub_ps(block_gradY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j05_neg) ) );
    }
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void ProjectionModel::computeJacobiansPhoto_spherical(MatrixXf & xyz_tf, const float stdDevPhoto_inv, VectorXf & weights, MatrixXf & jacobians, float *_grayGradX, float *_grayGradY)
{
    jacobians.resize(xyz_tf.rows(), 6);

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);
    float *_weight = &weights(0,0);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        float dist = xyz_transf.norm();
        computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);
        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians.block(i,0,1,6) = ((sqrt(wEstimDepth(i)) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;
    }

#else
    // Pointers to the jacobian elements
    float *_j0 = &jacobians(0,0);
    float *_j1 = &jacobians(0,1);
    float *_j2 = &jacobians(0,2);
    float *_j3 = &jacobians(0,3);
    float *_j4 = &jacobians(0,4);
    float *_j5 = &jacobians(0,5);
\
    __m128 block_stdDevInv = _mm_set1_ps(stdDevPhoto_inv);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_gradX = _mm_load_ps(_grayGradX+i);
        __m128 block_gradY = _mm_load_ps(_grayGradY+i);
        __m128 block_weight = _mm_load_ps(_weight+i);
        __m128 block_weight_stdDevInv = _mm_mul_ps(block_stdDevInv, block_weight);
        __m128 block_gradX_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradX);
        __m128 block_gradY_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradY);

        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _x2_z2 = _mm_sum_ps(_mm_mul_ps(block_x, block_x), _mm_mul_ps(block_z, block_z) );
        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
        __m128 _dist2 = _mm_sum_ps(_mm_mul_ps(block_y, block_y), _x2_z2 );
        __m128 _dist = _mm_sqrt_ps(_dist2);
        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, block_y);

        __m128 block_j00 = _mm_mul_ps(_commonDer_c, block_z);
        __m128 block_j10 = _mm_mul_ps(_commonDer_r_y, block_x);
        __m128 block_j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
        //__m128 block_j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, block_x), _mm_set1_ps(-0.f));
        __m128 block_j02_neg = _mm_mul_ps(_commonDer_c, block_x);
        __m128 block_j12 = _mm_mul_ps(_commonDer_r_y, block_z );
        __m128 block_j03_neg = _mm_mul_ps( block_j02, block_y );
        __m128 block_j13 = _mm_sub_ps(_mm_mul_ps(block_j12, block_y), _mm_mul_ps(block_j11, block_z) );
        __m128 block_j04 = _mm_sum_ps(_mm_mul_ps(block_j00, block_z), _mm_mul_ps(block_j02_neg, block_x) );
        __m128 block_j14 = _mm_sub_ps(_mm_mul_ps(block_j10, block_z), _mm_mul_ps(block_j12, block_x) );
        __m128 block_j05_neg = _mm_mul_ps(block_j00, block_y);
        __m128 block_j15 = _mm_sub_ps(_mm_mul_ps(block_j11, block_x), _mm_mul_ps(block_j10, block_y) );

        _mm_store_ps(_j0+i, _mm_sum_ps(_mm_mul_ps(block_gradX_weight, block_j00), _mm_mul_ps(block_gradY_weight, block_j10) ) );
        _mm_store_ps(_j1+i, _mm_mul_ps(block_gradY_weight, block_j11) );
        _mm_store_ps(_j2+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j12), _mm_mul_ps(block_gradX_weight, block_j02_neg) ) );
        _mm_store_ps(_j3+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j13), _mm_mul_ps(block_gradX_weight, block_j03_neg) ) );
        _mm_store_ps(_j4+i, _mm_sum_ps(_mm_mul_ps(block_gradY_weight, block_j14), _mm_mul_ps(block_gradX_weight, block_j04) ) );
        _mm_store_ps(_j5+i, _mm_sub_ps(block_gradY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j05_neg) ) );
    }
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
void ProjectionModel::computeJacobiansDepth_pinhole(MatrixXf & xyz_tf, VectorXf & stdDevError_inv, VectorXf & weights, MatrixXf & jacobians, float *_depthGradX, float *_depthGradY)
{
    jacobians.resize(xyz_tf.rows(), 6);

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);
    float *_weight = &weights(0,0);
    float *_stdDevInv = &stdDevError_inv(0);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT_pinhole(xyz_transf, jacobianWarpRt);
        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT(0,2) = 1.f;
        jacobian16_depthT(0,3) = xyz_transf(1);
        jacobian16_depthT(0,4) =-xyz_transf(0);
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians.block(i,0,1,6) = (sqrt(wEstimDepth(i)) * stdDevError_inv(i) ) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);

        //residualsDepth_src(i) *= weight_estim_sqrt;
    }

#else
    // Pointers to the jacobian elements
    float *_j0 = &jacobians(0,0);
    float *_j1 = &jacobians(0,1);
    float *_j2 = &jacobians(0,2);
    float *_j3 = &jacobians(0,3);
    float *_j4 = &jacobians(0,4);
    float *_j5 = &jacobians(0,5);

    __m128 _fx = _mm_set1_ps(fx);
    __m128 _fy = _mm_set1_ps(fy);
    __m128 _one = _mm_set1_ps(1.f);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_gradX = _mm_load_ps(_depthGradX+i);
        __m128 block_gradY = _mm_load_ps(_depthGradY+i);
        __m128 block_weight = _mm_load_ps(_weight+i);
        __m128 block_stdDevInv = _mm_load_ps(_stdDevInv+i);
        __m128 block_weight_stdDevInv = _mm_mul_ps(block_stdDevInv, block_weight);
        __m128 block_gradX_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradX);
        __m128 block_gradY_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradY);

        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _inv_z = _mm_div_ps(_one, block_z);

        __m128 block_j00 = _mm_mul_ps(_fx, _inv_z);
        __m128 block_j11 = _mm_mul_ps(_fy, _inv_z);
        __m128 block_j02 = _mm_xor_ps( _mm_mul_ps(block_x, _mm_mul_ps(_inv_z, block_j00) ), _mm_set1_ps(-0.f) );
        __m128 block_j12_neg = _mm_mul_ps(block_y, _mm_mul_ps(_inv_z, block_j11) );
        __m128 block_j03 = _mm_mul_ps( block_y, block_j02);
        __m128 block_j13_neg = _mm_sum_ps(_mm_mul_ps( block_y, block_j12_neg), _fy );
        __m128 block_j04_neg = _mm_sub_ps(_mm_mul_ps( block_x, block_j02), _fx );
        __m128 block_j14 = _mm_mul_ps( block_x, block_j12_neg);
        __m128 block_j05_neg = _mm_mul_ps(block_y, block_j00);
        __m128 block_j15 = _mm_mul_ps(block_x, block_j11);

        _mm_store_ps(_j0+i, _mm_mul_ps(block_gradX_weight, block_j00) );
        _mm_store_ps(_j1+i, _mm_mul_ps(block_gradY_weight, block_j11) );
        _mm_store_ps(_j2+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j02), _mm_mul_ps(block_gradY_weight, block_j12_neg) ), block_weight_stdDevInv) );
        _mm_store_ps(_j3+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j03), _mm_mul_ps(block_gradY_weight, block_j13) ), _mm_mul_ps(block_weight_stdDevInv, block_y) ) );
        _mm_store_ps(_j4+i, _mm_sum_ps(_mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j14), _mm_mul_ps(block_gradX_weight, block_j04_neg) ), _mm_mul_ps(block_weight_stdDevInv, block_x) ) );
        _mm_store_ps(_j5+i, _mm_sub_ps(block_gradY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j05_neg) ) );
    }
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void ProjectionModel::computeJacobiansDepth_spherical(MatrixXf & xyz_tf, VectorXf & stdDevError_inv, VectorXf & weights, MatrixXf & jacobians, float *_depthGradX, float *_depthGradY)
{
    jacobians.resize(xyz_tf.rows(), 6);

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);
    float *_weight = &weights(0,0);
    float *_stdDevInv = &stdDevError_inv(0);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        float dist = xyz_transf.norm();
        computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);
        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz_transf.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians.block(i,0,1,6) = (sqrt(wEstimDepth(i)) * stdDevError_inv(i) * (depth_gradient) * jacobianWarpRt - jacobian16_depthT);

        //residualsDepth_src(i) *= weight_estim_sqrt;
    }

#else
    // Pointers to the jacobian elements
    float *_j0 = &jacobians(0,0);
    float *_j1 = &jacobians(0,1);
    float *_j2 = &jacobians(0,2);
    float *_j3 = &jacobians(0,3);
    float *_j4 = &jacobians(0,4);
    float *_j5 = &jacobians(0,5);

    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_gradX = _mm_load_ps(_depthGradX+i);
        __m128 block_gradY = _mm_load_ps(_depthGradY+i);
        __m128 block_weight = _mm_load_ps(_weight+i);
        __m128 block_stdDevInv = _mm_load_ps(_stdDevInv+i);
        __m128 block_weight_stdDevInv = _mm_mul_ps(block_stdDevInv, block_weight);
        __m128 block_gradX_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradX);
        __m128 block_gradY_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradY);

        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _x2_z2 = _mm_sum_ps(_mm_mul_ps(block_x, block_x), _mm_mul_ps(block_z, block_z) );
        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
        __m128 _dist2 = _mm_sum_ps(_mm_mul_ps(block_y, block_y), _x2_z2 );
        __m128 _dist = _mm_sqrt_ps(_dist2);
        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, block_y);

        __m128 block_j00 = _mm_mul_ps(_commonDer_c, block_z);
        __m128 block_j10 = _mm_mul_ps(_commonDer_r_y, block_x);
        __m128 block_j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
        //__m128 block_j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, block_x), _mm_set1_ps(-0.f));
        __m128 block_j02_neg = _mm_mul_ps(_commonDer_c, block_x);
        __m128 block_j12 = _mm_mul_ps(_commonDer_r_y, block_z );
        __m128 block_j03_neg = _mm_mul_ps( block_j02, block_y );
        __m128 block_j13 = _mm_sub_ps(_mm_mul_ps(block_j12, block_y), _mm_mul_ps(block_j11, block_z) );
        __m128 block_j04 = _mm_sum_ps(_mm_mul_ps(block_j00, block_z), _mm_mul_ps(block_j02_neg, block_x) );
        __m128 block_j14 = _mm_sub_ps(_mm_mul_ps(block_j10, block_z), _mm_mul_ps(block_j12, block_x) );
        __m128 block_j05_neg = _mm_mul_ps(block_j00, block_y);
        __m128 block_j15 = _mm_sub_ps(_mm_mul_ps(block_j11, block_x), _mm_mul_ps(block_j10, block_y) );

        _mm_store_ps(_j0+i, _mm_sub_ps(_mm_sum_ps(_mm_mul_ps(block_gradX_weight, block_j00), _mm_mul_ps(block_gradY_weight, block_j10) ), _mm_mul_ps(block_weight_stdDevInv, _mm_div_ps(block_x, _dist) ) ) );
        _mm_store_ps(_j1+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j11), _mm_mul_ps(block_weight_stdDevInv, _mm_div_ps( block_y, _dist ) ) ) );
        _mm_store_ps(_j2+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j12), _mm_mul_ps(block_gradX_weight, block_j02_neg) ), _mm_mul_ps(block_weight_stdDevInv, _mm_div_ps(block_z, _dist) ) ) );
        _mm_store_ps(_j3+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j13), _mm_mul_ps(block_gradX_weight, block_j03_neg) ) );
        _mm_store_ps(_j4+i, _mm_sum_ps(_mm_mul_ps(block_gradY_weight, block_j14), _mm_mul_ps(block_gradX_weight, block_j04) ) );
        _mm_store_ps(_j5+i, _mm_sub_ps(block_gradY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j05_neg) ) );
    }
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
void ProjectionModel::computeJacobiansPhotoDepth_pinhole(MatrixXf & xyz_tf, const float stdDevPhoto_inv, VectorXf & stdDevError_inv, VectorXf & weights,
                                                         MatrixXf & jacobians_photo, MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY)
{
    jacobians.resize(xyz_tf.rows(), 6);

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);
    float *_weight = &weights(0,0);
    float *_stdDevInv = &stdDevError_inv(0);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        computeJacobian26_wT_pinhole(xyz_transf, jacobianWarpRt);

        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        float sqrt_weight = sqrt(wEstimDepth(i));
        jacobians_photo.block(i,0,1,6) = ((sqrt_weight * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;

        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT(0,2) = 1.f;
        jacobian16_depthT(0,3) = xyz_transf(1);
        jacobian16_depthT(0,4) =-xyz_transf(0);
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians_depth.block(i,0,1,6) = ( (sqrt_weight * stdDevError_inv(i) ) * (depth_gradient * jacobianWarpRt - jacobian16_depthT);
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

    __m128 _fx = _mm_set1_ps(fx);
    __m128 _fy = _mm_set1_ps(fy);
    __m128 _one = _mm_set1_ps(1.f);
    __m128 block_stdDevPhotoInv = _mm_set1_ps(stdDevPhoto_inv);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_weight = _mm_load_ps(_weight+i);

        __m128 block_gradGrayX = _mm_load_ps(_grayGradX+i);
        __m128 block_gradGrayY = _mm_load_ps(_grayGradY+i);
        __m128 block_weight_stdDevPhotoInv = _mm_mul_ps(block_stdDevPhotoInv, block_weight);
        __m128 block_gradX_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradX);
        __m128 block_gradY_weight = _mm_mul_ps(block_weight_stdDevPhotoInv, block_gradY);

        __m128 block_gradX = _mm_load_ps(_depthGradX+i);
        __m128 block_gradY = _mm_load_ps(_depthGradY+i);
        __m128 block_stdDevDepthInv = _mm_load_ps(_stdDevInv+i);
        __m128 block_weight_stdDevDepthInv = _mm_mul_ps(block_stdDevDepthInv, block_weight);
        __m128 block_gradDepthX_weight = _mm_mul_ps(block_weight_stdDevDepthInv, block_gradX);
        __m128 block_gradDepthY_weight = _mm_mul_ps(block_weight_stdDevDepthInv, block_gradY);

        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _inv_z = _mm_div_ps(_one, block_z);

        __m128 block_j00 = _mm_mul_ps(_fx, _inv_z);
        __m128 block_j11 = _mm_mul_ps(_fy, _inv_z);
        __m128 block_j02 = _mm_xor_ps( _mm_mul_ps(block_x, _mm_mul_ps(_inv_z, block_j00) ), _mm_set1_ps(-0.f) );
        __m128 block_j12_neg = _mm_mul_ps(block_y, _mm_mul_ps(_inv_z, block_j11) );
        __m128 block_j03 = _mm_mul_ps( block_y, block_j02);
        __m128 block_j13_neg = _mm_sum_ps(_mm_mul_ps( block_y, block_j12_neg), _fy );
        __m128 block_j04_neg = _mm_sub_ps(_mm_mul_ps( block_x, block_j02), _fx );
        __m128 block_j14 = _mm_mul_ps( block_x, block_j12_neg);
        __m128 block_j05_neg = _mm_mul_ps(block_y, block_j00);
        __m128 block_j15 = _mm_mul_ps(block_x, block_j11);

        _mm_store_ps(_j0_gray+i, _mm_mul_ps(block_gradX_weight, block_j00) );
        _mm_store_ps(_j1_gray+i, _mm_mul_ps(block_gradY_weight, block_j11) );
        _mm_store_ps(_j2_gray+i, _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j02), _mm_mul_ps(block_gradY_weight, block_j12_neg) ) );
        _mm_store_ps(_j3_gray+i, _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j03), _mm_mul_ps(block_gradY_weight, block_j13) ) );
        _mm_store_ps(_j4_gray+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j14), _mm_mul_ps(block_gradX_weight, block_j04_neg) ) );
        _mm_store_ps(_j5_gray+i, _mm_sub_ps(block_gradY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j05_neg) ) );

        _mm_store_ps(_j0_depth+i, _mm_mul_ps(block_gradDepthX_weight, block_j00) );
        _mm_store_ps(_j1_depth+i, _mm_mul_ps(block_gradDepthY_weight, block_j11) );
        _mm_store_ps(_j2_depth+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradDepthX_weight, block_j02), _mm_mul_ps(block_gradDepthY_weight, block_j12_neg) ), block_weight_stdDevInv) );
        _mm_store_ps(_j3_depth+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradDepthX_weight, block_j03), _mm_mul_ps(block_gradDepthY_weight, block_j13) ), _mm_mul_ps(block_weight_stdDevInv, block_y) ) );
        _mm_store_ps(_j4_depth+i, _mm_sum_ps(_mm_sub_ps(_mm_mul_ps(block_gradDepthY_weight, block_j14), _mm_mul_ps(block_gradDepthX_weight, block_j04_neg) ), _mm_mul_ps(block_weight_stdDevInv, block_x) ) );
        _mm_store_ps(_j5_depth+i, _mm_sub_ps(block_gradDepthY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradDepthX_weight, block_j05_neg) ) );
    }
#endif
}

/*! Compute the Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
void ProjectionModel::computeJacobiansPhotoDepth_spherical(MatrixXf & xyz_tf, const float stdDevPhoto_inv, VectorXf & stdDevError_inv, VectorXf & weights,
                                                           MatrixXf & jacobians_photo, MatrixXf & jacobians_depth, float *_depthGradX, float *_depthGradY, float *_grayGradX, float *_grayGradY)
{
    jacobians.resize(xyz_tf.rows(), 6);

    float *_x = &xyz_tf(0,0);
    float *_y = &xyz_tf(0,1);
    float *_z = &xyz_tf(0,2);
    float *_weight = &weights(0,0);
    float *_stdDevInv = &stdDevError_inv(0);

#if !(_SSE3) // # ifdef !__SSE3__

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
    for(size_t i=0; i < xyz_transf.rows(); i++)
    {
        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
        Matrix<float,2,6> jacobianWarpRt;
        float dist = xyz_transf.norm();
        computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

        Matrix<float,1,2> img_gradient;
        img_gradient(0,0) = _grayGradX[i];
        img_gradient(0,1) = _grayGradY[i];
        jacobians.block(i,0,1,6) = ((sqrt(wEstimDepth(i)) * stdDevPhoto_inv ) * img_gradient) * jacobianWarpRt;

        Matrix<float,1,6> jacobian16_depthT = Matrix<float,1,6>::Zero();
        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz_transf.transpose();
        Matrix<float,1,2> depth_gradient;
        depth_gradient(0,0) = _depthGradX[i];
        depth_gradient(0,1) = _depthGradY[i];
        jacobians.block(i,0,1,6) = (sqrt(wEstimDepth(i)) * stdDevError_inv(i) * (depth_gradient) * jacobianWarpRt - jacobian16_depthT);
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

    __m128 block_stdDevPhotoInv = _mm_set1_ps(stdDevPhoto_inv);
    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
    for(size_t i=0; i < xyz_tf.rows(); i+=4)
    {
        __m128 block_weight = _mm_load_ps(_weight+i);

        __m128 block_gradGrayX = _mm_load_ps(_grayGradX+i);
        __m128 block_gradGrayY = _mm_load_ps(_grayGradY+i);
        __m128 block_weight_stdDevPhotoInv = _mm_mul_ps(block_stdDevPhotoInv, block_weight);
        __m128 block_gradX_weight = _mm_mul_ps(block_weight_stdDevInv, block_gradX);
        __m128 block_gradY_weight = _mm_mul_ps(block_weight_stdDevPhotoInv, block_gradY);

        __m128 block_gradX = _mm_load_ps(_depthGradX+i);
        __m128 block_gradY = _mm_load_ps(_depthGradY+i);
        __m128 block_stdDevDepthInv = _mm_load_ps(_stdDevInv+i);
        __m128 block_weight_stdDevDepthInv = _mm_mul_ps(block_stdDevDepthInv, block_weight);
        __m128 block_gradDepthX_weight = _mm_mul_ps(block_weight_stdDevDepthInv, block_gradX);
        __m128 block_gradDepthY_weight = _mm_mul_ps(block_weight_stdDevDepthInv, block_gradY);

        __m128 block_x = _mm_load_ps(_x+i);
        __m128 block_y = _mm_load_ps(_y+i);
        __m128 block_z = _mm_load_ps(_z+i);
        __m128 _x2_z2 = _mm_sum_ps(_mm_mul_ps(block_x, block_x), _mm_mul_ps(block_z, block_z) );
        __m128 _x2_z2_sqrt = _mm_sqrt_ps(_x2_z2);
        __m128 _dist2 = _mm_sum_ps(_mm_mul_ps(block_y, block_y), _x2_z2 );
        __m128 _dist = _mm_sqrt_ps(_dist2);
        __m128 _commonDer_c = _mm_div_ps(_pixel_angle_inv, _x2_z2);
        __m128 _commonDer_r = _mm_xor_ps( _mm_div_ps(_pixel_angle_inv, _mm_mul_ps(_dist2, _x2_z2_sqrt) ), _mm_set1_ps(-0.f) );
        __m128 _commonDer_r_y = _mm_mul_ps(_commonDer_r, block_y);

        __m128 block_j00 = _mm_mul_ps(_commonDer_c, block_z);
        __m128 block_j10 = _mm_mul_ps(_commonDer_r_y, block_x);
        __m128 block_j11 = _mm_xor_ps( _mm_mul_ps(_commonDer_r, _x2_z2), _mm_set1_ps(-0.f) );
        //__m128 block_j02 = _mm_xor_ps( _mm_mul_ps(_commonDer_c, block_x), _mm_set1_ps(-0.f));
        __m128 block_j02_neg = _mm_mul_ps(_commonDer_c, block_x);
        __m128 block_j12 = _mm_mul_ps(_commonDer_r_y, block_z );
        __m128 block_j03_neg = _mm_mul_ps( block_j02, block_y );
        __m128 block_j13 = _mm_sub_ps(_mm_mul_ps(block_j12, block_y), _mm_mul_ps(block_j11, block_z) );
        __m128 block_j04 = _mm_sum_ps(_mm_mul_ps(block_j00, block_z), _mm_mul_ps(block_j02_neg, block_x) );
        __m128 block_j14 = _mm_sub_ps(_mm_mul_ps(block_j10, block_z), _mm_mul_ps(block_j12, block_x) );
        __m128 block_j05_neg = _mm_mul_ps(block_j00, block_y);
        __m128 block_j15 = _mm_sub_ps(_mm_mul_ps(block_j11, block_x), _mm_mul_ps(block_j10, block_y) );

        _mm_store_ps(_j0+i, _mm_sum_ps(_mm_mul_ps(block_gradX_weight, block_j00), _mm_mul_ps(block_gradY_weight, block_j10) ) );
        _mm_store_ps(_j1+i, _mm_mul_ps(block_gradY_weight, block_j11) );
        _mm_store_ps(_j2+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j12), _mm_mul_ps(block_gradX_weight, block_j02_neg) ) );
        _mm_store_ps(_j3+i, _mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j13), _mm_mul_ps(block_gradX_weight, block_j03_neg) ) );
        _mm_store_ps(_j4+i, _mm_sum_ps(_mm_mul_ps(block_gradY_weight, block_j14), _mm_mul_ps(block_gradX_weight, block_j04) ) );
        _mm_store_ps(_j5+i, _mm_sub_ps(block_gradY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j05_neg) ) );

        _mm_store_ps(_j0_depth+i, _mm_sub_ps(_mm_sum_ps(_mm_mul_ps(block_gradDepthX_weight, block_j00), _mm_mul_ps(block_gradDepthY_weight, block_j10) ), _mm_mul_ps(block_weight_stdDevInv, _mm_div_ps(block_x, _dist) ) ) );
        _mm_store_ps(_j1_depth+i, _mm_sub_ps(_mm_mul_ps(block_gradDepthY_weight, block_j11), _mm_mul_ps(block_weight_stdDevInv, _mm_div_ps( block_y, _dist ) ) ) );
        _mm_store_ps(_j2_depth+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradDepthY_weight, block_j12), _mm_mul_ps(block_gradDepthX_weight, block_j02_neg) ), _mm_mul_ps(block_weight_stdDevInv, _mm_div_ps(block_z, _dist) ) ) );
        _mm_store_ps(_j3_depth+i, _mm_sub_ps(_mm_mul_ps(block_gradDepthY_weight, block_j13), _mm_mul_ps(block_gradDepthX_weight, block_j03_neg) ) );
        _mm_store_ps(_j4_depth+i, _mm_sum_ps(_mm_mul_ps(block_gradDepthY_weight, block_j14), _mm_mul_ps(block_gradDepthX_weight, block_j04) ) );
        _mm_store_ps(_j5_depth+i, _mm_sub_ps(block_gradDepthY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradDepthX_weight, block_j05_neg) ) );
    }
#endif
}

///*! Compute the 3Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the pinhole camera model. */
//void ProjectionModel::computeJacobiansICP_pinhole(Eigen::MatrixXf & xyz_tf, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians)
//{
//    jacobians.resize(3*xyz_tf.rows(), 6);

//    float *_x = &xyz_tf(0,0);
//    float *_y = &xyz_tf(0,1);
//    float *_z = &xyz_tf(0,2);
//    float *_weight = &weights(0,0);
//    float *_stdDevInv = &stdDevError_inv(0);

//#if !(_SSE3) // # ifdef !__SSE3__

////    #if ENABLE_OPENMP
////    #pragma omp parallel for
////    #endif
//    for(size_t i=0; i < xyz_transf.rows(); i++)
//    {
//        Vector3f xyz_transf = xyz_tf.block(i,0,1,3).transpose();
//        Matrix<float,3,6> jacobianRt;
//        computeJacobian36_xT_p(xyz, jacobianRt);
//        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
//        jacobiansDepth.block(3*i,0,3,6) = sqrt(wEstimDepth(i)) * stdDevError_inv_src(i) * jacobianRt;
//        //residualsDepth_src(i) *= weight_estim_sqrt;
//    }

//#else
//    // Pointers to the jacobian elements
//    float *_j0 = &jacobians(0,0);
//    float *_j1 = &jacobians(0,1);
//    float *_j2 = &jacobians(0,2);
//    float *_j3 = &jacobians(0,3);
//    float *_j4 = &jacobians(0,4);
//    float *_j5 = &jacobians(0,5);

//    __m128 _fx = _mm_set1_ps(fx);
//    __m128 _fy = _mm_set1_ps(fy);
//    __m128 _one = _mm_set1_ps(1.f);
//    for(size_t i=0; i < xyz_tf.rows(); i+=4)
//    {
//        __m128 block_weight = _mm_load_ps(_weight+i);
//        __m128 block_stdDevInv = _mm_load_ps(_stdDevInv+i);
//        __m128 block_weight_stdDevInv = _mm_mul_ps(block_stdDevInv, block_weight);

//        __m128 block_x = _mm_load_ps(_x+i);
//        __m128 block_y = _mm_load_ps(_y+i);
//        __m128 block_z = _mm_load_ps(_z+i);
//        __m128 _inv_z = _mm_div_ps(_one, block_z);

//        __m128 block_j00 = _mm_mul_ps(_fx, _inv_z);
//        __m128 block_j11 = _mm_mul_ps(_fy, _inv_z);
//        __m128 block_j02 = _mm_xor_ps( _mm_mul_ps(block_x, _mm_mul_ps(_inv_z, block_j00) ), _mm_set1_ps(-0.f) );
//        __m128 block_j12_neg = _mm_mul_ps(block_y, _mm_mul_ps(_inv_z, block_j11) );
//        __m128 block_j03 = _mm_mul_ps( block_y, block_j02);
//        __m128 block_j13_neg = _mm_sum_ps(_mm_mul_ps( block_y, block_j12_neg), _fy );
//        __m128 block_j04_neg = _mm_sub_ps(_mm_mul_ps( block_x, block_j02), _fx );
//        __m128 block_j14 = _mm_mul_ps( block_x, block_j12_neg);
//        __m128 block_j05_neg = _mm_mul_ps(block_y, block_j00);
//        __m128 block_j15 = _mm_mul_ps(block_x, block_j11);

//        _mm_store_ps(_j0+i, _mm_mul_ps(block_gradX_weight, block_j00) );
//        _mm_store_ps(_j1+i, _mm_mul_ps(block_gradY_weight, block_j11) );
//        _mm_store_ps(_j2+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j02), _mm_mul_ps(block_gradY_weight, block_j12_neg) ), block_weight_stdDevInv) );
//        _mm_store_ps(_j3+i, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j03), _mm_mul_ps(block_gradY_weight, block_j13) ), _mm_mul_ps(block_weight_stdDevInv, block_y) ) );
//        _mm_store_ps(_j4+i, _mm_sum_ps(_mm_sub_ps(_mm_mul_ps(block_gradY_weight, block_j14), _mm_mul_ps(block_gradX_weight, block_j04_neg) ), _mm_mul_ps(block_weight_stdDevInv, block_x) ) );
//        _mm_store_ps(_j5+i, _mm_sub_ps(block_gradY_weight, block_j15), _mm_sub_ps(_mm_mul_ps(block_gradX_weight, block_j05_neg) ) );
//    }
//#endif
//}

///*! Compute the 3Nx6 jacobian matrices of the composition (imgGrad+warping+rigidTransformation) using the spherical camera model. */
//void ProjectionModel::computeJacobiansICP_spherical(Eigen::MatrixXf & xyz_tf, Eigen::VectorXf & stdDevError_inv, Eigen::VectorXf & wEstimDepth, Eigen::MatrixXf & jacobians)
//{

//}


///*! Warp the image according to a given geometric transformation. */
//void ProjectionModel::warpImage(cv::Mat img,                // The original image
//                                const Matrix4f & Rt,        // The relative pose of the robot between the two frames
//                                const costFuncType method ) //,  const bool use_bilinear )
//{
//    cout << " ProjectionModel::warpImage \n";

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

//    //reconstruct3D_spherical(depthSrcPyr[pyrLevel], LUT_xyz_source, validPixels);
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
////                float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
////                int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
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
//                int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
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
//    cout << pyrLevel << " ProjectionModel::warpImage took " << double (time_end - time_start)*1000 << " ms. \n";
//#endif
//}
