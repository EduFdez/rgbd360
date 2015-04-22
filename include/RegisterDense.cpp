/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include <RegisterDense.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
//#include <iostream>
//#include <fstream>

#include <pcl/common/time.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP
#include <pcl/registration/warp_point_rigid.h>

#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/obs/CObservation2DRangeScan.h>
//#include <mrpt/slam/CSimplePointsMap.h>
//#include <mrpt/slam/CObservation2DRangeScan.h>
#include <mrpt/slam/CICP.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/poses/CPosePDF.h>
#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/math/utils.h>
#include <mrpt/system/os.h>

// For SIMD performance
//#include "cvalarray.hpp"
//#include "veclib.h"
//#include <x86intrin.h>
#include <mmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>

#define ENABLE_OPENMP 0
#define PRINT_PROFILING 1
#define ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS 1
#define INVALID_POINT -10000
#define SSE_AVAILABLE 1

RegisterDense::RegisterDense() :
    min_depth_(0.3f),
    max_depth_(20.f),
    nPyrLevels(4),
    use_salient_pixels_(false),
    compute_MAD_stdDev_(false),
    use_bilinear_(false),
    visualizeIterations(false)
{
    sensor_type = STEREO_OUTDOOR; //RGBD360_INDOOR

    stdDevPhoto = 8./255;
    varPhoto = stdDevPhoto*stdDevPhoto;

    stdDevDepth = 0.01;
    varDepth = stdDevDepth*stdDevDepth;

    min_depth_Outliers = 2*stdDevDepth; // in meters
    max_depth_Outliers = 1; // in meters

    thresSaliency = 0.04f;
    thres_saliency_gray_ = 0.04f;
    thres_saliency_depth_ = 0.04f;
//    thres_saliency_gray_ = 0.f;
//    thres_saliency_depth_ = 0.f;

    registered_pose_ = Eigen::Matrix4f::Identity();
};


/*! Build a pyramid of nLevels of image resolutions from the input image.
 * The resolution of each layer is 2x2 times the resolution of its image above.*/
void RegisterDense::buildPyramid( const cv::Mat & img, std::vector<cv::Mat> & pyramid, const int nLevels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif
    //Create space for all the images // ??
    pyramid.resize(nLevels);
    pyramid[0] = img;
    //std::cout << "types " << pyramid[0].type() << " " << img.type() << std::endl;

    for(size_t level=1; level < nLevels; level++)
    {
        assert(pyramid[0].rows % 2 == 0 && pyramid[0].cols % 2 == 0 );

        // Assign the resized image to the current level of the pyramid
        pyrDown( pyramid[level-1], pyramid[level], cv::Size( pyramid[level-1].cols/2, pyramid[level-1].rows/2 ) );

        //cv::imshow("pyramid", pyramid[level]);
        //cv::waitKey(0);
    }
#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "RegisterDense::buildPyramid " << (time_end - time_start)*1000 << " ms. \n";
#endif
};

/*! Build a pyramid of nLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
void RegisterDense::buildPyramidRange( const cv::Mat & img, std::vector<cv::Mat> & pyramid, const int nLevels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();\
    //for(size_t i=0; i<1000; i++)
    {
#endif
    //Create space for all the images // ??
    pyramid.resize(nLevels);
    if(img.type() == CV_16U) // If the image is in milimetres, convert it to meters
        img.convertTo(pyramid[0], CV_32FC1, 0.001 );
    else
        pyramid[0] = img;
    //    std::cout << "types " << pyramid[0].type() << " " << img.type() << std::endl;

    for(size_t level=1; level < nLevels; level++)
    {
        //Create an auxiliar image of factor times the size of the original image
        size_t nCols = pyramid[level-1].cols;
        size_t nRows = pyramid[level-1].rows;
        size_t img_size = nRows * nCols;
        pyramid[level] = cv::Mat::zeros(cv::Size( nCols/2, nRows/2 ), pyramid[0].type() );
        //            cv::Mat imgAux = cv::Mat::zeros(cv::Size( nCols/2, pyramid[level-1].rows/2 ), pyramid[0].type() );
        float *_z = reinterpret_cast<float*>(pyramid[level-1].data);
        float *_z_sub = reinterpret_cast<float*>(pyramid[level].data);
        if(img_size > 4*1e4) // Apply multicore only to the bigger images
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t r=0; r < pyramid[level-1].rows; r+=2)
                for(size_t c=0; c < nCols; c+=2)
                {
                    float avDepth = 0.f;
                    unsigned nvalidPixels_src = 0;
                    for(size_t i=0; i < 2; i++)
                        for(size_t j=0; j < 2; j++)
                        {
                            size_t pixel = (r+i)*nCols + c + j;
                            float z = _z[pixel];
                            //                        float z = pyramid[level-1].at<float>(r+i,c+j);
                            //cout << "z " << z << endl;
                            if(z > min_depth_ && z < max_depth_)
                            {
                                avDepth += z;
                                ++nvalidPixels_src;
                            }
                        }
                    if(nvalidPixels_src > 0)
                    {
                        //pyramid[level].at<float>(r/2,c/2) = avDepth / nvalidPixels_src;
                        size_t pixel_sub = r*nCols/4 + c/2;
                        _z_sub[pixel_sub] = avDepth / nvalidPixels_src;
                    }
                }
        }
        else
        {
            for(size_t r=0; r < pyramid[level-1].rows; r+=2)
                for(size_t c=0; c < nCols; c+=2)
                {
                    float avDepth = 0.f;
                    unsigned nvalidPixels_src = 0;
                    for(size_t i=0; i < 2; i++)
                        for(size_t j=0; j < 2; j++)
                        {
                            size_t pixel = (r+i)*nCols + c + j;
                            float z = _z[pixel];
                            //                        float z = pyramid[level-1].at<float>(r+i,c+j);
                            //cout << "z " << z << endl;
                            if(z > min_depth_ && z < max_depth_)
                            {
                                avDepth += z;
                                ++nvalidPixels_src;
                            }
                        }
                    if(nvalidPixels_src > 0)
                    {
                        //pyramid[level].at<float>(r/2,c/2) = avDepth / nvalidPixels_src;
                        size_t pixel_sub = r*nCols/4 + c/2;
                        _z_sub[pixel_sub] = avDepth / nvalidPixels_src;
                    }
                }
        }
    }
#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "RegisterDense::buildPyramidRange " << (time_end - time_start)*1000 << " ms. \n";
#endif
};


/*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
void RegisterDense::calcGradientXY(const cv::Mat & src, cv::Mat & gradX, cv::Mat & gradY)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    //int dataType = src.type();
    gradX = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );
    gradY = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );

    float *_pixel = reinterpret_cast<float*>(src.data);
    float *_pixel_gradX = reinterpret_cast<float*>(gradX.data);
    float *_pixel_gradY = reinterpret_cast<float*>(gradY.data);

#if SSE_AVAILABLE
    // Use SIMD instructions -> x4 performance
    assert( src.cols % 4 == 0);
    //size_t img_size = src.rows * src.cols;
    size_t img_size = src.rows * src.cols;
    size_t block_start = src.cols, block_end = img_size - src.cols;
    const __m128 scalar2 = _mm_set1_ps(2.f); //float f2 = 2.f;
    if(img_size > 4*1e4) // Apply multicore only to the bigger images
    {
        #if ENABLE_OPENMP
        #pragma omp parallel for // schedule(static) // schedule(dynamic)
        #endif
        for(size_t b=block_start; b < block_end; b+=4)
        {
            __m128 block_ = _mm_load_ps(_pixel+b);
            __m128 block_x1 = _mm_loadu_ps(_pixel+b+1);
            __m128 block_x_1 = _mm_loadu_ps(_pixel+b-1);
            __m128 diff1 = _mm_sub_ps(block_x1, block_);
            __m128 diff_1= _mm_sub_ps(block_, block_x_1);
            __m128 den = _mm_add_ps(diff1, diff_1);
            __m128 mul = _mm_mul_ps(diff1, diff_1);
            __m128 num = _mm_mul_ps(scalar2, mul);
            __m128 res = _mm_div_ps(num, den);
            //__m128 res = _mm_div_ps(_mm_mul_ps(scalar2, mul), den);
            //_mm_store_ps(_pixel_gradX+b, res);
            __m128 mask = _mm_or_ps(
                                    _mm_and_ps( _mm_cmplt_ps(block_x_1, block_), _mm_cmplt_ps(block_, block_x1) ),
                                    _mm_and_ps( _mm_cmplt_ps(block_x1, block_), _mm_cmplt_ps(block_, block_x_1) ) );
            //_mm_maskstore_ps(_pixel_gradX+b, mask, res);
            __m128 gradX = _mm_and_ps(mask, res);
            _mm_store_ps(_pixel_gradX+b, gradX);

            __m128 block_y1 = _mm_load_ps(_pixel+b+src.cols);
            __m128 block_y_1 = _mm_load_ps(_pixel+b-src.cols);
            diff1 = _mm_sub_ps(block_y1, block_);
            diff_1= _mm_sub_ps(block_, block_y_1);
            den = _mm_add_ps(diff1, diff_1);
            mul = _mm_mul_ps(diff1, diff_1);
            num = _mm_mul_ps(scalar2, mul);
            res = _mm_div_ps(num, den);
            //res = _mm_div_ps(_mm_mul_ps(scalar2, mul), den);
            //_mm_store_ps(_pixel_gradY+b, res);
            mask = _mm_or_ps( _mm_and_ps( _mm_cmplt_ps(block_y_1, block_), _mm_cmplt_ps(block_, block_y1) ),
                              _mm_and_ps( _mm_cmplt_ps(block_y1, block_), _mm_cmplt_ps(block_, block_y_1) ) );
            //_mm_maskstore_ps(_pixel_gradY+b, mask, res);
            __m128 gradY = _mm_and_ps(mask, res);
            _mm_store_ps(_pixel_gradY+b, gradY);
        }
        // Compute the gradint at the image border
        #if ENABLE_OPENMP
        #pragma omp parallel for // schedule(static) // schedule(dynamic)
        #endif
        for(unsigned r=1; r < src.rows-1; ++r)
        {
            size_t row_pix = r*src.cols;
            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
        }
        size_t last_row_pix = (src.rows - 1) * src.cols;
        #if ENABLE_OPENMP
        #pragma omp parallel for // schedule(static) // schedule(dynamic)
        #endif
        for(size_t c=1; c < src.cols-1; ++c)
        {
            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
        }
    }
    else
    {
        for(size_t b=block_start; b < block_end; b+=4)
        {
            __m128 block_ = _mm_load_ps(_pixel+b);
            __m128 block_x1 = _mm_loadu_ps(_pixel+b+1);
            __m128 block_x_1 = _mm_loadu_ps(_pixel+b-1);
            __m128 diff1 = _mm_sub_ps(block_x1, block_);
            __m128 diff_1= _mm_sub_ps(block_, block_x_1);
            __m128 den = _mm_add_ps(diff1, diff_1);
            __m128 mul = _mm_mul_ps(diff1, diff_1);
            __m128 num = _mm_mul_ps(scalar2, mul);
            __m128 res = _mm_div_ps(num, den);
            //__m128 res = _mm_div_ps(_mm_mul_ps(scalar2, mul), den);
            //_mm_store_ps(_pixel_gradX+b, res);
            __m128 mask = _mm_or_ps(
                        _mm_and_ps( _mm_cmplt_ps(block_x_1, block_), _mm_cmplt_ps(block_, block_x1) ),
                        _mm_and_ps( _mm_cmplt_ps(block_x1, block_), _mm_cmplt_ps(block_, block_x_1) ) );
            //_mm_maskstore_ps(_pixel_gradX+b, mask, res);
            __m128 gradX = _mm_and_ps(mask, res);
            _mm_store_ps(_pixel_gradX+b, gradX);

            __m128 block_y1 = _mm_load_ps(_pixel+b+src.cols);
            __m128 block_y_1 = _mm_load_ps(_pixel+b-src.cols);
            diff1 = _mm_sub_ps(block_y1, block_);
            diff_1= _mm_sub_ps(block_, block_y_1);
            den = _mm_add_ps(diff1, diff_1);
            mul = _mm_mul_ps(diff1, diff_1);
            num = _mm_mul_ps(scalar2, mul);
            res = _mm_div_ps(num, den);
            //res = _mm_div_ps(_mm_mul_ps(scalar2, mul), den);
            //_mm_store_ps(_pixel_gradY+b, res);
            mask = _mm_or_ps( _mm_and_ps( _mm_cmplt_ps(block_y_1, block_), _mm_cmplt_ps(block_, block_y1) ),
                              _mm_and_ps( _mm_cmplt_ps(block_y1, block_), _mm_cmplt_ps(block_, block_y_1) ) );
            //_mm_maskstore_ps(_pixel_gradY+b, mask, res);
            __m128 gradY = _mm_and_ps(mask, res);
            _mm_store_ps(_pixel_gradY+b, gradY);
        }
        // Compute the gradint at the image border
        for(unsigned r=1; r < src.rows-1; ++r)
        {
            size_t row_pix = r*src.cols;
            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
        }
        size_t last_row_pix = (src.rows - 1) * src.cols;
        for(size_t c=1; c < src.cols-1; ++c)
        {
            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
        }
    }

#else
    if(img_size > 4*1e4) // Apply multicore only to the bigger images
    {
        #if ENABLE_OPENMP
        #pragma omp parallel for // schedule(static) // schedule(dynamic)
        #endif
        for(size_t r=1; r < src.rows-1; ++r)
        {
            size_t row_pix = r*src.cols;
            for(size_t c=1; c < src.cols-1; ++c)
            {
                // Efficient pointer-based gradient computation
                //unsigned i = r*src.cols + c;
                size_t i = row_pix + c;

                if( (_pixel[i+1] > _pixel[i] && _pixel[i] > _pixel[i-1]) || (_pixel[i+1] < _pixel[i] && _pixel[i] < _pixel[i-1]) )
                    _pixel_gradX[i] = 2.f * (_pixel[i+1] - _pixel[i]) * (_pixel[i] - _pixel[i-1]) / (_pixel[i+1] - _pixel[i-1]);

                if( (_pixel[i+src.cols] > _pixel[i] && _pixel[i] > _pixel[i-src.cols]) || (_pixel[i+src.cols] < _pixel[i] && _pixel[i] < _pixel[i-src.cols]) )
                    _pixel_gradY[i] = 2.f * (_pixel[i+src.cols] - _pixel[i]) * (_pixel[i] - _pixel[i-src.cols]) / (_pixel[i+src.cols] - _pixel[i-src.cols]);

    //            if( (src.at<float>(r,c) > src.at<float>(r,c+1) && src.at<float>(r,c) < src.at<float>(r,c-1) ) ||
    //                (src.at<float>(r,c) < src.at<float>(r,c+1) && src.at<float>(r,c) > src.at<float>(r,c-1) )   )//{
    //                    gradX.at<float>(r,c) = 2.f * (src.at<float>(r,c+1)-src.at<float>(r,c)) * (src.at<float>(r,c)-src.at<float>(r,c-1)) / (src.at<float>(r,c+1)-src.at<float>(r,c-1));
    //                    //std::cout << "GradX " << gradX.at<float>(r,c) << " " << gradX_.at<float>(r,c) << std::endl;}

    //            if( (src.at<float>(r,c) > src.at<float>(r+1,c) && src.at<float>(r,c) < src.at<float>(r-1,c) ) ||
    //                (src.at<float>(r,c) < src.at<float>(r+1,c) && src.at<float>(r,c) > src.at<float>(r-1,c) )   )
    //                    gradY.at<float>(r,c) = 2.f * (src.at<float>(r+1,c)-src.at<float>(r,c)) * (src.at<float>(r,c)-src.at<float>(r-1,c)) / (src.at<float>(r+1,c)-src.at<float>(r-1,c));
            }
            // Compute the gradint at the image border
            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
        }
        // Compute the gradint at the image border
        size_t last_row_pix = (src.rows - 1) * src.cols;
        #if ENABLE_OPENMP
        #pragma omp parallel for // schedule(static) // schedule(dynamic)
        #endif
        for(size_t c=1; c < src.cols-1; ++c)
        {
            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
        }
    }
    else
    {
        for(size_t r=1; r < src.rows-1; ++r)
        {
            size_t row_pix = r*src.cols;
            for(size_t c=1; c < src.cols-1; ++c)
            {
                // Efficient pointer-based gradient computation
                //unsigned i = r*src.cols + c;
                size_t i = row_pix + c;

                if( (_pixel[i+1] > _pixel[i] && _pixel[i] > _pixel[i-1]) || (_pixel[i+1] < _pixel[i] && _pixel[i] < _pixel[i-1]) )
                    _pixel_gradX[i] = 2.f * (_pixel[i+1] - _pixel[i]) * (_pixel[i] - _pixel[i-1]) / (_pixel[i+1] - _pixel[i-1]);

                if( (_pixel[i+src.cols] > _pixel[i] && _pixel[i] > _pixel[i-src.cols]) || (_pixel[i+src.cols] < _pixel[i] && _pixel[i] < _pixel[i-src.cols]) )
                    _pixel_gradY[i] = 2.f * (_pixel[i+src.cols] - _pixel[i]) * (_pixel[i] - _pixel[i-src.cols]) / (_pixel[i+src.cols] - _pixel[i-src.cols]);

    //            if( (src.at<float>(r,c) > src.at<float>(r,c+1) && src.at<float>(r,c) < src.at<float>(r,c-1) ) ||
    //                (src.at<float>(r,c) < src.at<float>(r,c+1) && src.at<float>(r,c) > src.at<float>(r,c-1) )   )//{
    //                    gradX.at<float>(r,c) = 2.f * (src.at<float>(r,c+1)-src.at<float>(r,c)) * (src.at<float>(r,c)-src.at<float>(r,c-1)) / (src.at<float>(r,c+1)-src.at<float>(r,c-1));
    //                    //std::cout << "GradX " << gradX.at<float>(r,c) << " " << gradX_.at<float>(r,c) << std::endl;}

    //            if( (src.at<float>(r,c) > src.at<float>(r+1,c) && src.at<float>(r,c) < src.at<float>(r-1,c) ) ||
    //                (src.at<float>(r,c) < src.at<float>(r+1,c) && src.at<float>(r,c) > src.at<float>(r-1,c) )   )
    //                    gradY.at<float>(r,c) = 2.f * (src.at<float>(r+1,c)-src.at<float>(r,c)) * (src.at<float>(r,c)-src.at<float>(r-1,c)) / (src.at<float>(r+1,c)-src.at<float>(r-1,c));
            }
            // Compute the gradint at the image border
            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
        }
        // Compute the gradint at the image border
        size_t last_row_pix = (src.rows - 1) * src.cols;
        for(size_t c=1; c < src.cols-1; ++c)
        {
            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
        }
    }
#endif

//    cv::imshow("DerY", gradY);
//    cv::imshow("DerX", gradX);
//    cv::waitKey(0);

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << src.rows << "rows. RegisterDense::calcGradientXY " << (time_end - time_start)*1000 << " ms. \n";
#endif
};

/*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
void RegisterDense::calcGradientXY_saliency(const cv::Mat & src, cv::Mat & gradX, cv::Mat & gradY, std::vector<int> & vSalientPixels_)
{
    //int dataType = src.type();
    gradX = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );
    gradY = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );

    for(unsigned r=1; r < src.rows-1; r++)
        for(unsigned c=1; c < src.cols-1; c++)
        {
            if( (src.at<float>(r,c) > src.at<float>(r,c+1) && src.at<float>(r,c) < src.at<float>(r,c-1) ) ||
                (src.at<float>(r,c) < src.at<float>(r,c+1) && src.at<float>(r,c) > src.at<float>(r,c-1) )   )
                    gradX.at<float>(r,c) = 2.f / (1/(src.at<float>(r,c+1)-src.at<float>(r,c)) + 1/(src.at<float>(r,c)-src.at<float>(r,c-1)));

            if( (src.at<float>(r,c) > src.at<float>(r+1,c) && src.at<float>(r,c) < src.at<float>(r-1,c) ) ||
                (src.at<float>(r,c) < src.at<float>(r+1,c) && src.at<float>(r,c) > src.at<float>(r-1,c) )   )
                    gradY.at<float>(r,c) = 2.f / (1/(src.at<float>(r+1,c)-src.at<float>(r,c)) + 1/(src.at<float>(r,c)-src.at<float>(r-1,c)));
        }

    vSalientPixels_.clear();
    for(unsigned r=1; r < src.rows-1; r++)
        for(unsigned c=1; c < src.cols-1; c++)
            if( (fabs(gradX.at<float>(r,c)) > thresSaliency) || (fabs(gradY.at<float>(r,c)) > thresSaliency) )
                vSalientPixels_.push_back(src.cols*r+c); //vector index
};

/*! Compute the gradient images for each pyramid level. */
void RegisterDense::buildGradientPyramids(const std::vector<cv::Mat> & grayPyr, std::vector<cv::Mat> & grayGradXPyr, std::vector<cv::Mat> & grayGradYPyr,
                                          const std::vector<cv::Mat> & depthPyr, std::vector<cv::Mat> & depthGradXPyr, std::vector<cv::Mat> & depthGradYPyr)
{
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)

    //Compute image gradients
    //double scale = 1./255;
    //double delta = 0;
    //int dataType = CV_32FC1; // grayPyr[level].type();

    //Create space for all the derivatives images
    grayGradXPyr.resize(grayPyr.size());
    grayGradYPyr.resize(grayPyr.size());

    depthGradXPyr.resize(grayPyr.size());
    depthGradYPyr.resize(grayPyr.size());

    for(unsigned level=0; level < nPyrLevels; level++)
    {
        double time_start_ = pcl::getTime();

        calcGradientXY(grayPyr[level], grayGradXPyr[level], grayGradYPyr[level]);

//#if PRINT_PROFILING
//        double time_end_ = pcl::getTime();
//        std::cout << level << " PyramidPhoto " << (time_end_ - time_start_) << std::endl;

//        time_start_ = pcl::getTime();
//#endif

        calcGradientXY(depthPyr[level], depthGradXPyr[level], depthGradYPyr[level]);

//#if PRINT_PROFILING
//        time_end_ = pcl::getTime();
//        std::cout << level << " PyramidDepth " << (time_end_ - time_start_) << std::endl;
//#endif


        //time_start_ = pcl::getTime();

        // Compute the gradient in x
        //grayGradXPyr[level] = cv::Mat(cv::Size( grayPyr[level].cols, grayPyr[level].rows), grayPyr[level].type() );
        //cv::Scharr( grayPyr[level], grayGradXPyr[level], dataType, 1, 0, scale, delta, cv::BORDER_DEFAULT );
        //cv::Sobel( grayPyr[level], grayGradXPyr[level], dataType, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );

        // Compute the gradient in y
        //grayGradYPyr[level] = cv::Mat(cv::Size( grayPyr[level].cols, grayPyr[level].rows), grayPyr[level].type() );
        //cv::Scharr( grayPyr[level], grayGradYPyr[level], dataType, 0, 1, scale, delta, cv::BORDER_DEFAULT );
        //cv::Sobel( grayPyr[level], grayGradYPyr[level], dataType, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

        //double time_end_ = pcl::getTime();
        //std::cout << level << " PyramidPhoto " << (time_end_ - time_start_) << std::endl;

        //            cv::Mat imgNormalizedDepth;
        //            imagePyramid[level].convertTo(imgNormalizedDepth, CV_32FC1,1./max_depth_);

        // Compute the gradient in x
        //            cv::Scharr( depthPyr[level], depthGradXPyr[level], dataType, 1, 0, scale, delta, cv::BORDER_DEFAULT );

        // Compute the gradient in y
        //            cv::Scharr( depthPyr[level], depthGradYPyr[level], dataType, 0, 1, scale, delta, cv::BORDER_DEFAULT );

        //            cv::imshow("DerX", grayGradXPyr[level]);
        //            cv::imshow("DerY", grayGradYPyr[level]);
        //            cv::waitKey(0);
        //            cv::imwrite(mrpt::format("/home/edu/gradX_%d.png",level), grayGradXPyr[level]);
        //            cv::imwrite(mrpt::format("/home/edu/gray_%d.png",level), grayPyr[level]);
    }

//#if PRINT_PROFILING
    double time_end = pcl::getTime();
    std::cout << "RegisterDense::buildGradientPyramids " << (time_end - time_start)*1000 << " ms. \n";
//#endif
};

/*! Sets the source (Intensity+Depth) frame.*/
void RegisterDense::setSourceFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth)
{
    #if PRINT_PROFILING
    double time_start = pcl::getTime();
    #endif

    //Create a float auxialiary image from the imput image
//    cv::cvtColor(imgRGB, graySrc, CV_RGB2GRAY);
    cv::cvtColor(imgRGB, graySrc, cv::COLOR_RGB2GRAY);
    graySrc.convertTo(graySrc, CV_32FC1, 1./255 );

    //Compute image pyramids for the grayscale and depth images
    buildPyramid(graySrc, graySrcPyr, nPyrLevels);
    buildPyramidRange(imgDepth, depthSrcPyr, nPyrLevels);

    //Compute image pyramids for the gradients images
    buildGradientPyramids( graySrcPyr, graySrcGradXPyr, graySrcGradYPyr,
                           depthSrcPyr, depthSrcGradXPyr, depthSrcGradYPyr );
//    // This is intended to show occlussions
//    rgbSrc = imgRGB;
//    buildPyramid(rgbSrc, colorSrcPyr, nPyrLevels);

    #if PRINT_PROFILING
    double time_end = pcl::getTime();
    std::cout << "RegisterDense::setSourceFrame construction " << (time_end - time_start)*1000 << " ms. \n";
    #endif
};

/*! Sets the source (Intensity+Depth) frame. Depth image is ignored*/
void RegisterDense::setTargetFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth)
{
    #if PRINT_PROFILING
    double time_start = pcl::getTime();
    #endif

    //Create a float auxialiary image from the imput image
    // grayTrg.create(imgRGB.rows, imgRGB.cols, CV_32FC1);
//    cv::cvtColor(imgRGB, grayTrg, CV_RGB2GRAY);
    cv::cvtColor(imgRGB, grayTrg, cv::COLOR_RGB2GRAY);
    grayTrg.convertTo(grayTrg, CV_32FC1, 1./255 );

    //Compute image pyramids for the grayscale and depth images
    buildPyramid(grayTrg, grayTrgPyr, nPyrLevels);
    buildPyramidRange(imgDepth, depthTrgPyr, nPyrLevels);

    //Compute image pyramids for the gradients images
    buildGradientPyramids( grayTrgPyr, grayTrgGradXPyr, grayTrgGradYPyr,
                           depthTrgPyr, depthTrgGradXPyr, depthTrgGradYPyr );

    //        cv::imwrite("/home/efernand/test.png", grayTrgGradXPyr[nPyrLevels-1]);
    //        cv::imshow("GradX_pyr ", grayTrgGradXPyr[nPyrLevels-1]);
    //        cv::imshow("GradY_pyr ", grayTrgGradYPyr[nPyrLevels-1]);
    //        cv::imshow("GradX ", grayTrgGradXPyr[0]);
    //        cv::imshow("GradY ", grayTrgGradYPyr[0]);
    //        cv::imshow("GradX_d ", depthTrgGradXPyr[0]);
    //        cv::waitKey(0);

#if PRINT_PROFILING
    double time_end = pcl::getTime();
    std::cout << "RegisterDense::setTargetFrame construction " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Swap the source and target images */
void RegisterDense::swapSourceTarget()
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

//    std::vector<cv::Mat> graySrcPyr_tmp = graySrcPyr;
//    std::vector<cv::Mat> graySrcGradXPyr_tmp = graySrcGradXPyr;
//    std::vector<cv::Mat> graySrcGradYPyr_tmp = graySrcGradYPyr;
//    std::vector<cv::Mat> depthSrcPyr_tmp = depthSrcPyr;
//    std::vector<cv::Mat> depthSrcGradXPyr_tmp = depthSrcGradXPyr;
//    std::vector<cv::Mat> depthSrcGradYPyr_tmp = depthSrcGradYPyr;

//    graySrcPyr = grayTrgPyr;
//    graySrcGradXPyr = grayTrgGradXPyr;
//    graySrcGradYPyr = grayTrgGradYPyr;
//    depthSrcPyr = depthTrgPyr;
//    depthSrcGradXPyr = depthTrgGradXPyr;
//    depthSrcGradYPyr = depthTrgGradYPyr;

//    grayTrgPyr = graySrcPyr_tmp;
//    grayTrgGradXPyr = graySrcGradXPyr_tmp;
//    grayTrgGradYPyr = graySrcGradYPyr_tmp;
//    depthTrgPyr = depthSrcPyr_tmp;
//    depthTrgGradXPyr = depthSrcGradXPyr_tmp;
//    depthTrgGradYPyr = depthSrcGradYPyr_tmp;

    cv::imshow( "sphereGray", graySrcPyr[0] );
    cv::imshow( "sphereGray1", graySrcPyr[1] );
    cv::imshow( "sphereDepth2", depthSrcPyr[2] );

    cv::waitKey(0);

    grayTrgPyr = graySrcPyr;
    grayTrgGradXPyr = graySrcGradXPyr;
    grayTrgGradYPyr = graySrcGradYPyr;
    depthTrgPyr = depthSrcPyr;
    depthTrgGradXPyr = depthSrcGradXPyr;
    depthTrgGradYPyr = depthSrcGradYPyr;

    cv::imshow( "sphereGray", grayTrgPyr[1] );
    cv::imshow( "sphereGray1", grayTrgPyr[2] );
    cv::imshow( "sphereDepth2", depthTrgPyr[3] );

    cv::waitKey(0);

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "RegisterDense::swapSourceTarget took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
double RegisterDense::errorDense( const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method )
{
    //std::cout << " RegisterDense::errorDense \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;
//    size_t n_ptsPhoto = 0;
//    size_t n_ptsDepth = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    float stdDevPhoto_inv = 1./stdDevPhoto;
    float stdDevDepth_inv = 1./stdDevDepth;

    transformPts3D_sse(LUT_xyz_source, poseGuess, xyz_src_transf);

    warp_pixels_src.resize( xyz_src_transf.rows() );
    warp_img_src.resize( xyz_src_transf.rows(), 2 );
    residualsPhoto_src = Eigen::VectorXf::Zero(imgSize);
    residualsDepth_src = Eigen::VectorXf::Zero(imgSize);
    stdDevError_inv_src = Eigen::VectorXf::Zero(imgSize);
    wEstimPhoto_src = Eigen::VectorXf::Zero(imgSize);
    wEstimDepth_src = Eigen::VectorXf::Zero(imgSize);
    validPixelsPhoto_src = Eigen::VectorXi::Zero(imgSize);
    validPixelsDepth_src = Eigen::VectorXi::Zero(imgSize);

//    _residualsPhoto_src = Eigen::VectorXf::Zero(imgSize);
//    _residualsDepth_src = Eigen::VectorXf::Zero(imgSize);
//    _stdDevError_inv_src = Eigen::VectorXf::Zero(imgSize);
//    _wEstimPhoto_src = Eigen::VectorXf::Zero(imgSize);
//    _wEstimDepth_src = Eigen::VectorXf::Zero(imgSize);
//    _validPixelsPhoto_src = Eigen::VectorXi::Zero(imgSize);
//    _validPixelsDepth_src = Eigen::VectorXi::Zero(imgSize);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    std::vector<float> v_AD_intensity(imgSize);

    float *_depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyramidLevel].data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);
    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    if( !use_bilinear_ || pyramidLevel !=0 )
    {
        // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
        for(size_t i=0; i < xyz_src_transf.rows(); i++)
        {
            if( validPixels_src(i) ) //Compute the jacobian only for the valid points
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();

                //Project the 3D point to the 2D plane
                float inv_transf_z = 1.f/xyz(2);
                // 2D coordinates of the transformed pixel(r,c) of frame 1
                float transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
                float transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
                int transformed_r_int = round(transformed_r);
                int transformed_c_int = round(transformed_c);
                // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( (transformed_r_int>=0 && transformed_r_int < nRows) && (transformed_c_int>=0 && transformed_c_int < nCols) )
                {
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    //if(compute_MAD_stdDev_)
                    //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
                    ++numVisiblePts;

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //                        float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                            //                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            validPixelsPhoto_src(i) = 1;
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
                            //                        _validPixelsPhoto_src(n_ptsPhoto) = i;
                            //                        _residualsPhoto_src(n_ptsPhoto) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsPhoto]) * stdDevPhoto_inv;
                            //                        _wEstimPhoto_src(n_ptsPhoto) = weightMEstimator(_residualsPhoto_src(n_ptsPhoto)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            //                        error2 += _wEstimPhoto_src(n_ptsPhoto) * _residualsPhoto_src(n_ptsPhoto) * _residualsPhoto_src(n_ptsPhoto);

                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                        float depth = _depthTrgPyr[warped_i];
                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(xyz(2)*xyz(2)+depth*depth), 2*stdDevDepth);
                                residualsDepth_src(i) = (depth - xyz(2)) * stdDevError_inv_src(i);
                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;

                                //                            _validPixelsDepth_src(n_ptsDepth) = i;
                                //                            _stdDevError_inv_src(n_ptsDepth) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                //                            _residualsDepth_src(n_ptsDepth) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsDepth]) * stdDevError_inv_src(n_ptsDepth);
                                //                            _wEstimDepth_src(n_ptsDepth) = weightMEstimator(_residualsDepth_src(n_ptsDepth)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                //                            error2 += _wEstimDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth);
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
        for(size_t i=0; i < xyz_src_transf.rows(); i++)
        {
            //Compute the 3D coordinates of the pij of the source frame
            if( validPixels_src(i) ) //Compute the jacobian only for the valid points
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                //Project the 3D point to the 2D plane
                float inv_transf_z = 1.f/xyz(2);
                // 2D coordinates of the transformed pixel(r,c) of frame 1
                float transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
                float transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
                int transformed_r_int = round(transformed_r);
                int transformed_c_int = round(transformed_c);
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( (transformed_r>=0 && transformed_r < nRows) && (transformed_c>=0 && transformed_c < nCols) )
                {
                    ++numVisiblePts;
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    warp_img_src(i,0) = transformed_r;
                    warp_img_src(i,1) = transformed_c;
                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_src(i) = 1;
                            float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _graySrcPyr[i];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
                            // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(xyz(2)*xyz(2)+depth*depth), 2*stdDevDepth);
                                residualsDepth_src(i) = (depth - xyz(2)) * stdDevError_inv_src(i);
                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
    }

    SSO = (float)numVisiblePts / imgSize;
    //        std::cout << "numVisiblePixels " << numVisiblePixels << " imgSize " << imgSize << " sso " << SSO << std::endl;

//    _validPixelsPhoto_src.conservativeResize(n_ptsPhoto);
//    _residualsPhoto_src.conservativeResize(n_ptsPhoto);
//    _wEstimPhoto_src.conservativeResize(n_ptsPhoto);

//    _validPixelsDepth_src.conservativeResize(n_ptsPhoto);
//    _stdDevError_inv_src.conservativeResize(n_ptsPhoto);
//    _residualsDepth_src.conservativeResize(n_ptsPhoto);
//    _wEstimDepth_src.conservativeResize(n_ptsPhoto);

    // Compute the median absulute deviation of the projection of reference image onto the target one to update the value of the standard deviation of the intesity error
//    if(error2_photo > 0 && compute_MAD_stdDev_)
//    {
//        std::cout << " stdDevPhoto PREV " << stdDevPhoto << std::endl;
//        size_t count_valid_pix = 0;
//        std::vector<float> v_AD_intensity(n_ptsPhoto);
//        for(size_t i=0; i < imgSize; i++)
//            if( validPixelsPhoto_src(i) ) //Compute the jacobian only for the valid points
//            {
//                v_AD_intensity[count_valid_pix] = v_AD_intensity_[i];
//                ++count_valid_pix;
//            }
//        //v_AD_intensity.conservativeResize(n_pts);
//        v_AD_intensity.conservativeResize(n_ptsPhoto);
//        float stdDevPhoto_updated = 1.4826 * median(v_AD_intensity);
//        error2_photo *= stdDevPhoto*stdDevPhoto / (stdDevPhoto_updated*stdDevPhoto_updated);
//        stdDevPhoto = stdDevPhoto_updated;
//        std::cout << " stdDevPhoto_updated    " << stdDevPhoto_updated << std::endl;
//    }

    error2 = error2_photo + error2_depth;

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif
#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " errorDense took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//    std::cout << "error_av " << (error2 / n_pts) << " error2 " << error2 << " n_pts " << n_pts << " stdDevPhoto " << stdDevPhoto << std::endl;
//#endif

    return (error2 / numVisiblePts);
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::computeJacobian(const int &pyramidLevel,
                                    const Eigen::Matrix4f poseGuess,
                                    costFuncType method )
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    float stdDevPhoto_inv = 1./stdDevPhoto;
    float stdDevDepth_inv = 1./stdDevDepth;

    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
    Eigen::MatrixXf jacobiansDepth(imgSize,6);

    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    if( !use_bilinear_ || pyramidLevel !=0 )
    {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(size_t i=0; i < xyz_src_transf.rows(); i++)
        {
            //Compute the 3D coordinates of the pij of the source frame
            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) ) //Compute the jacobian only for the valid points
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_pinhole(xyz, jacobianWarpRt);

                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                    jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                    // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                    jacobian16_depthT(0,2) = 1.f;
                    jacobian16_depthT(0,2) = xyz(1);
                    jacobian16_depthT(0,2) =-xyz(0);
                    float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                    jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                    residualsDepth_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << "residualsDepth_src " << residualsDepth_src(i) << std::endl;
                }
            }
        }
    }
    else
    {
        std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
        // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i < xyz_src_transf.rows(); i++)
        {
            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) ) //Compute the jacobian only for the valid points
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;

                //Compute the pixel jacobian
                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_pinhole(xyz, jacobianWarpRt);

                cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                    img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                    jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                    depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                    jacobian16_depthT(0,2) = 1.f;
                    jacobian16_depthT(0,2) = xyz(1);
                    jacobian16_depthT(0,2) =-xyz(0);
                    float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                    jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                    residualsDepth_src(i) *= weight_estim_sqrt;
                    // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                }
            }
        }
    }

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " computeJacobian took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::calcHessGrad(const int &pyramidLevel,
                                    const Eigen::Matrix4f poseGuess,
                                    costFuncType method )
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    float stdDevPhoto_inv = 1./stdDevPhoto;
    float stdDevDepth_inv = 1./stdDevDepth;

    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
    Eigen::MatrixXf jacobiansDepth(imgSize,6);

    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if( !use_bilinear_ || pyramidLevel !=0 )
    {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(size_t i=0; i < xyz_src_transf.rows(); i++)
        {
            //Compute the 3D coordinates of the pij of the source frame
            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) ) //Compute the jacobian only for the valid points
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_pinhole(xyz, jacobianWarpRt);

                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                    //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                    jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_depth.at<float>(warp_pixels_src(i)) = xyz(2);

                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                    // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                    jacobian16_depthT(0,2) = 1.f;
                    jacobian16_depthT(0,2) = xyz(1);
                    jacobian16_depthT(0,2) =-xyz(0);
                    float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                    jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                    residualsDepth_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << "residualsDepth_src " << residualsDepth_src(i) << std::endl;
                }
            }
        }
    }
    else
    {
        std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
        // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i < xyz_src_transf.rows(); i++)
        {
            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) ) //Compute the jacobian only for the valid points
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;

                //Compute the pixel jacobian
                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_pinhole(xyz, jacobianWarpRt);

                cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                    img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                    jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_depth.at<float>(warp_pixels_src(i)) = xyz(2);

                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                    depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                    jacobian16_depthT(0,2) = 1.f;
                    jacobian16_depthT(0,2) = xyz(1);
                    jacobian16_depthT(0,2) =-xyz(0);
                    float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                    jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                    residualsDepth_src(i) *= weight_estim_sqrt;
                    // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
    //std::cout << "hessian \n" << hessian << std::endl;
    //std::cout << "gradient \n" << gradient.transpose() << std::endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGrad took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
double RegisterDense::errorDense_inv( const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method )
{
    //std::cout << " RegisterDense::errorDense \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;
//    size_t n_ptsPhoto = 0;
//    size_t n_ptsDepth = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    float stdDevPhoto_inv = 1./stdDevPhoto;
    float stdDevDepth_inv = 1./stdDevDepth;

    //    std::cout << "poseGuess \n" << poseGuess << std::endl;

    const Eigen::Matrix4f poseGuess_inv = poseGuess.inverse();
    transformPts3D_sse(LUT_xyz_target, poseGuess_inv, xyz_trg_transf);

    warp_pixels_trg.resize(imgSize);
    warp_img_trg.resize(imgSize,2);
    residualsPhoto_trg = Eigen::VectorXf::Zero(imgSize);
    residualsDepth_trg = Eigen::VectorXf::Zero(imgSize);
    stdDevError_inv_trg = Eigen::VectorXf::Zero(imgSize);
    wEstimPhoto_trg = Eigen::VectorXf::Zero(imgSize);
    wEstimDepth_trg = Eigen::VectorXf::Zero(imgSize);
    validPixelsPhoto_trg = Eigen::VectorXi::Zero(imgSize);
    validPixelsDepth_trg = Eigen::VectorXi::Zero(imgSize);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    std::vector<float> v_AD_intensity(imgSize);

    float *_depthSrcPyr = reinterpret_cast<float*>(depthSrcPyr[pyramidLevel].data);
    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if( !use_bilinear_ || pyramidLevel !=0 )
    {
        // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
        for(size_t i=0; i < xyz_trg_transf.rows(); i++)
        {
            if( validPixels_trg(i) ) //Compute the jacobian only for the valid points
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();

                //Project the 3D point to the 2D plane
                float inv_transf_z = 1.f/xyz(2);
                // 2D coordinates of the transformed pixel(r,c) of frame 1
                float transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
                float transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
                int transformed_r_int = round(transformed_r);
                int transformed_c_int = round(transformed_c);
                // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( (transformed_r_int>=0 && transformed_r_int < nRows) && (transformed_c_int>=0 && transformed_c_int < nCols) )
                {
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_trg(i) = warped_i;
                    //if(compute_MAD_stdDev_)
                    //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
                    ++numVisiblePts;

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        if( fabs(_graySrcGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[warped_i]) > thres_saliency_gray_)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //                        float pixel_trg = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                            //                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            validPixelsPhoto_trg(i) = 1;
                            float diff = _graySrcPyr[warped_i] - _grayTrgPyr[i];
                            residualsPhoto_trg(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_trg(i) = weightMEstimator(residualsPhoto_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_trg(i) * residualsPhoto_trg(i) * residualsPhoto_trg(i);
                            //                        _validPixelsPhoto_trg(n_ptsPhoto) = i;
                            //                        _residualsPhoto_trg(n_ptsPhoto) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsPhoto]) * stdDevPhoto_inv;
                            //                        _wEstimPhoto_trg(n_ptsPhoto) = weightMEstimator(_residualsPhoto_trg(n_ptsPhoto)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            //                        error2 += _wEstimPhoto_trg(n_ptsPhoto) * _residualsPhoto_trg(n_ptsPhoto) * _residualsPhoto_trg(n_ptsPhoto);

                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                        float depth = _depthSrcPyr[warped_i];
                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            if( fabs(_depthSrcGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[warped_i]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_trg(i) = 1;
                                stdDevError_inv_trg(i) = 1 / std::max (stdDevDepth*(xyz(2)*xyz(2)+depth*depth), 2*stdDevDepth);
                                residualsDepth_trg(i) = (depth - xyz(2)) * stdDevError_inv_trg(i);
                                wEstimDepth_trg(i) = weightMEstimator(residualsDepth_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_trg(i) * residualsDepth_trg(i) * residualsDepth_trg(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;

                                //                            _validPixelsDepth_trg(n_ptsDepth) = i;
                                //                            _stdDevError_inv_trg(n_ptsDepth) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                //                            _residualsDepth_trg(n_ptsDepth) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsDepth]) * stdDevError_inv_trg(n_ptsDepth);
                                //                            _wEstimDepth_trg(n_ptsDepth) = weightMEstimator(_residualsDepth_trg(n_ptsDepth)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                //                            error2 += _wEstimDepth_trg(n_ptsDepth) * _residualsDepth_trg(n_ptsDepth) * _residualsDepth_trg(n_ptsDepth);
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
        for(size_t i=0; i < xyz_trg_transf.rows(); i++)
        {
            if( validPixels_trg(i) ) //Compute the jacobian only for the valid points
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                //Project the 3D point to the 2D plane
                float inv_transf_z = 1.f/xyz(2);
                // 2D coordinates of the transformed pixel(r,c) of frame 1
                float transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
                float transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
                int transformed_r_int = round(transformed_r);
                int transformed_c_int = round(transformed_c);
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( (transformed_r>=0 && transformed_r < nRows) && (transformed_c>=0 && transformed_c < nCols) )
                {
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_trg(i) = warped_i;
                    warp_img_trg(i,0) = transformed_r;
                    warp_img_trg(i,1) = transformed_c;
                    cv::Point2f warped_pixel(warp_img_trg(i,0), warp_img_trg(i,1));
                    // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);
                    ++numVisiblePts;

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        if( fabs(_graySrcGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[warped_i]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_trg(i) = 1;
                            float intensity = bilinearInterp( graySrcPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _grayTrgPyr[i];
                            residualsPhoto_trg(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_trg(i) = weightMEstimator(residualsPhoto_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_trg(i) * residualsPhoto_trg(i) * residualsPhoto_trg(i);
                            // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        float depth = bilinearInterp_depth( graySrcPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            if( fabs(_depthSrcGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[warped_i]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_trg(i) = 1;
                                stdDevError_inv_trg(i) = 1 / std::max (stdDevDepth*(xyz(2)*xyz(2)+depth*depth), 2*stdDevDepth);
                                residualsDepth_trg(i) = (depth - xyz(2)) * stdDevError_inv_trg(i);
                                wEstimDepth_trg(i) = weightMEstimator(residualsDepth_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_trg(i) * residualsDepth_trg(i) * residualsDepth_trg(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
    }

    SSO = (float)numVisiblePts / imgSize;
    //        std::cout << "numVisiblePixels " << numVisiblePixels << " imgSize " << imgSize << " sso " << SSO << std::endl;

    error2 = error2_photo + error2_depth;

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif
#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " errorDense took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//    std::cout << "error_av " << (error2 / n_pts) << " error2 " << error2 << " n_pts " << n_pts << " stdDevPhoto " << stdDevPhoto << std::endl;
//#endif

    return (error2 / numVisiblePts);
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::calcHessGrad_inv(const int &pyramidLevel,
                                    const Eigen::Matrix4f poseGuess,
                                    costFuncType method )
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    float stdDevPhoto_inv = 1./stdDevPhoto;
    float stdDevDepth_inv = 1./stdDevDepth;

    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
    Eigen::MatrixXf jacobiansDepth(imgSize,6);

    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if( !use_bilinear_ || pyramidLevel !=0 )
    {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(size_t i=0; i < xyz_trg_transf.rows(); i++)
        {
            if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) ) //Compute the jacobian only for the valid points
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_pinhole(xyz, jacobianWarpRt);

                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(i); // We keep the wrong name to 'source' to avoid duplicating more structures

                    //std::cout << "warp_pixels_trg(i) " << warp_pixels_trg(i) << std::endl;

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = _graySrcGradXPyr[warp_pixels_trg(i)];
                    img_gradient(0,1) = _graySrcGradYPyr[warp_pixels_trg(i)];

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                    jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_trg(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " weightedErrorPhoto " << residualsPhoto_trg(i) << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_depth.at<float>(warp_pixels_trg(i)) = xyz(2);

                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = _depthSrcGradXPyr[warp_pixels_trg(i)];
                    depth_gradient(0,1) = _depthSrcGradYPyr[warp_pixels_trg(i)];
                    // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                    jacobian16_depthT(0,2) = 1.f;
                    jacobian16_depthT(0,2) = xyz(1);
                    jacobian16_depthT(0,2) =-xyz(0);
                    float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                    jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                    residualsDepth_trg(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << "residualsDepth_trg " << residualsDepth_trg(i) << std::endl;
                }
            }
        }
    }
    else
    {
        std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
        // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(size_t i=0; i < xyz_trg_transf.rows(); i++)
        {
            if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) ) //Compute the jacobian only for the valid points
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;

                //Compute the pixel jacobian
                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_pinhole(xyz, jacobianWarpRt);

                cv::Point2f warped_pixel(warp_img_trg(i,0), warp_img_trg(i,1));
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = bilinearInterp( graySrcGradXPyr[pyramidLevel], warped_pixel );
                    img_gradient(0,1) = bilinearInterp( graySrcGradYPyr[pyramidLevel], warped_pixel );

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                    jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_trg(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_depth.at<float>(warp_pixels_trg(i)) = xyz(2);

                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                    depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                    jacobian16_depthT(0,2) = 1.f;
                    jacobian16_depthT(0,2) = xyz(1);
                    jacobian16_depthT(0,2) =-xyz(0);
                    float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                    jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                    residualsDepth_trg(i) *= weight_estim_sqrt;
                    // std::cout << "residualsDepth_trg \n " << residualsDepth_trg << std::endl;
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);

    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
    //std::cout << "hessian \n" << hessian << std::endl;
    //std::cout << "gradient \n" << gradient.transpose() << std::endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGrad took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

///*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method. */
//double RegisterDense::errorDense_Occ1(const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method )
//{
//    //double error2 = 0.0; // Squared error
//    double PhotoResidual = 0.0;
//    double DepthResidual = 0.0;
//    int n_ptsPhoto = 0;
//    int n_ptsDepth = 0;

//    const size_t nRows = graySrcPyr[pyramidLevel].rows;
//    const size_t nCols = graySrcPyr[pyramidLevel].cols;
//    const size_t imgSize = nRows*nCols;

//    Eigen::VectorXf residualsPhoto_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::VectorXf residualsDepth_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::VectorXf invDepthBuffer = Eigen::VectorXf::Zero(imgSize);

//    const float scaleFactor = 1.0/pow(2,pyramidLevel);
//    fx = cameraMatrix(0,0)*scaleFactor;
//    fy = cameraMatrix(1,1)*scaleFactor;
//    ox = cameraMatrix(0,2)*scaleFactor;
//    oy = cameraMatrix(1,2)*scaleFactor;
//    const float inv_fx = 1./fx;
//    const float inv_fy = 1./fy;

//    //        float varianceRegularization = 1; // 63%
//    //        float stdDevReg = sqrt(varianceRegularization);
//    float weight_estim; // The weight computed from an M-estimator
//    float stdDevPhoto_inv = 1./stdDevPhoto;
//    float stdDevDepth_inv = 1./stdDevDepth;

//    //    std::cout << "poseGuess \n" << poseGuess << std::endl;

//    Eigen::Matrix3f rotation = poseGuess.block(0,0,3,3);
//    Eigen::Vector3f translation = poseGuess.block(0,3,3,1);

//    //        if(use_salient_pixels_)
//    //        {

//    //#if ENABLE_OPENMP
//    //#pragma omp parallel for reduction (+:error2)
//    //#endif
//    //            for (int i=0; i < vSalientPixels[pyramidLevel].size(); i++)
//    //            {
//    ////                //                int i = nCols*r+c; //vector index
//    ////                int r = vSalientPixels[pyramidLevel][i] / nCols;
//    ////                int c = vSalientPixels[pyramidLevel][i] % nCols;

//    ////                //Compute the 3D coordinates of the pij of the source frame
//    ////                Eigen::Vector3f point3D;
//    ////                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//    ////                if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//    ////                {
//    ////                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
//    ////                    point3D(1)=(r - oy) * point3D(2) * inv_fy;

//    ////                    //Transform the 3D point using the transformation matrix Rt
//    ////                    Eigen::Vector3f xyz = rotation*point3D + translation;

//    //                if(LUT_xyz_source[vSalientPixels[pyramidLevel][i]](0) != INVALID_POINT) //Compute the jacobian only for the valid points
//    //                {
//    //                    Eigen::Vector3f xyz = rotation*LUT_xyz_source[vSalientPixels[pyramidLevel][i]] + translation;

//    //                    //Project the 3D point to the 2D plane
//    //                    float inv_transf_z = 1.f/xyz(2);
//    //                    float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//    //                    transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//    //                    transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//    //                    int transformed_r_int = round(transformed_r);
//    //                    int transformed_c_int = round(transformed_c);

//    //                    //Asign the intensity value to the warped image and compute the difference between the transformed
//    //                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//    //                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//    //                        (transformed_c_int>=0 && transformed_c_int < nCols) )
//    //                    {
//    //                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//    //                        {
//    //                            //Obtain the pixel values that will be used to compute the pixel residual
//    //                            // float pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//    //                            float pixel_src = graySrcPyr[pyramidLevel].data[vSalientPixels[pyramidLevel][i]]; // Intensity value of the pixel(r,c) of source frame
//    //                            float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//    //                            float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//    //                            float weight_estim = weightMEstimator(weightedError);
//    //                            float weightedErrorPhoto = weight_estim * weightedError;
//    //                            // Apply M-estimator weighting
//    ////                            if(weightedError2 > varianceRegularization)
//    ////                            {
//    //////                                float weightedError2_norm = sqrt(weightedError2);
//    //////                                double weight = sqrt(2*stdDevReg*weightedError2_norm-varianceRegularization) / weightedError2_norm;
//    ////                                weightedError2 = 2*stdDevReg*sqrt(weightedError2)-varianceRegularization;
//    ////                            }
//    //                            error2 += weightedErrorPhoto*weightedErrorPhoto;

//    //                        }
//    //                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//    //                        {
//    //                            float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//    //                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//    //                            {
//    //                                //Obtain the depth values that will be used to the compute the depth residual
//    //                                float depth1 = xyz(2);
//    //                                float weightedError = depth - depth1;
//    //                                //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//    //                                float stdDevError = stdDevDepth*depth1;
//    //                                float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//    //                                float weight_estim = weightMEstimator(weightedError);
//    //                                float weightedErrorDepth = weight_estim * weightedError;
//    //                                error2 += weightedErrorDepth*weightedErrorDepth;
//    //                            }
//    //                        }
//    //                    }
//    //                }
//    //            }
//    //        }
//    //        else
//    {
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:n_ptsPhoto,n_ptsDepth) // n_ptsPhoto,  //for reduction (+:error2)
//#endif
//        for (int i=0; i < LUT_xyz_source.size(); i++)
//        {
//            //int i = r*nCols + c;
//            // The depth is represented by the 'z' coordinate of LUT_xyz_source[i]
//            if(LUT_xyz_source(i,0) != INVALID_POINT) //Compute the jacobian only for the valid points
//            {
//                Eigen::Vector3f xyz = rotation*LUT_xyz_source[i] + translation;

//                //Project the 3D point to the 2D plane
//                float inv_transf_z = 1.f/xyz(2);
//                float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//                transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//                int transformed_r_int = round(transformed_r);
//                int transformed_c_int = round(transformed_c);

//                //Asign the intensity value to the warped image and compute the difference between the transformed
//                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                        (transformed_c_int>=0 && transformed_c_int < nCols) )
//                {
//                    // Discard occluded points
//                    int ii = transformed_r_int*nCols + transformed_c_int; // The index of the pixel in the target image
//                    if(invDepthBuffer(ii) > 0 && inv_transf_z < invDepthBuffer(ii)) // the current pixel is occluded
//                        continue;
//                    invDepthBuffer(ii) = inv_transf_z;

//                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        if(fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_gray_ &&
//                                fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_gray_)
//                            continue;

//                        //Obtain the pixel values that will be used to compute the pixel residual
//                        float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
//                        //float pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                        float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                        float weight_estim = weightMEstimator(weightedError);
//                        float weightedErrorPhoto = weight_estim * weightedError;
//                        // Apply M-estimator weighting
//                        //                            if(weightedError2 > varianceRegularization)
//                        //                            {
//                        ////                                float weightedError2_norm = sqrt(weightedError2);
//                        ////                                double weight = sqrt(2*stdDevReg*weightedError2_norm-varianceRegularization) / weightedError2_norm;
//                        //                                weightedError2 = 2*stdDevReg*sqrt(weightedError2)-varianceRegularization;
//                        //                            }
//                        residualsPhoto_src(ii) = weightedErrorPhoto*weightedErrorPhoto;
//                        ++n_ptsPhoto;
//                        // error2 += weightedErrorPhoto*weightedErrorPhoto;
//                        //                            std::cout << "error2 " << error2 << " weightedErrorPhoto " << weightedErrorPhoto << " " << weight_estim << " " << weightedError << std::endl;
//                    }
//                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                        {
//                            if( fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_depth_ &&
//                                    fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_depth_)
//                                continue;

//                            //Obtain the depth values that will be used to the compute the depth residual
//                            float depth1 = xyz(2);
//                            float weightedError = depth - depth1;
//                            //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//                            //                                    float stdDevError = stdDevDepth*depth1;
//                            float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                            float weight_estim = weightMEstimator(weightedError);
//                            float weightedErrorDepth = weight_estim * weightedError;
//                            //error2 += weightedErrorDepth*weightedErrorDepth;
//                            residualsDepth_src(ii) = weightedErrorDepth*weightedErrorDepth;
//                            ++n_ptsDepth;
//                        }
//                    }
//                }
//            }
//        }
//        //}
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:PhotoResidual,DepthResidual) // n_ptsPhoto, n_ptsDepth
//#endif
//        for(int i=0; i < imgSize; i++)
//        {
//            PhotoResidual += residualsPhoto_src(i);
//            DepthResidual += residualsDepth_src(i);
//            //                if(residualsPhoto_src(i) > 0) ++n_ptsPhoto;
//            //                if(residualsDepth_src(i) > 0) ++n_ptsDepth;
//        }
//    }

//    avPhotoResidual = sqrt(PhotoResidual / n_ptsPhoto);
//    //avPhotoResidual = sqrt(PhotoResidual / n_ptsDepth);
//    avDepthResidual = sqrt(DepthResidual / n_ptsDepth);
//    avResidual = avPhotoResidual + avDepthResidual;

//    // std::cout << "PhotoResidual " << PhotoResidual << " DepthResidual " << DepthResidual << std::endl;
//    // std::cout << "n_ptsPhoto " << n_ptsPhoto << " n_ptsDepth " << n_ptsDepth << std::endl;
//    // std::cout << "avPhotoResidual " << avPhotoResidual << " avDepthResidual " << avDepthResidual << std::endl;

//    return avResidual;
//    //return error2;
//}

///*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient. */
//void RegisterDense::calcHessGrad_Occ1(const int &pyramidLevel,
//                                         const Eigen::Matrix4f poseGuess,
//                                         costFuncType method )
//{
//    int nRows = graySrcPyr[pyramidLevel].rows;
//    int nCols = graySrcPyr[pyramidLevel].cols;
//    const size_t imgSize = nRows*nCols;

//    const float scaleFactor = 1.0/pow(2,pyramidLevel);
//    fx = cameraMatrix(0,0)*scaleFactor;
//    fy = cameraMatrix(1,1)*scaleFactor;
//    ox = cameraMatrix(0,2)*scaleFactor;
//    oy = cameraMatrix(1,2)*scaleFactor;
//    const float inv_fx = 1./fx;
//    const float inv_fy = 1./fy;

//    Eigen::MatrixXf jacobiansPhoto = Eigen::MatrixXf::Zero(imgSize,6);
//    Eigen::VectorXf residualsPhoto_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::MatrixXf jacobiansDepth = Eigen::MatrixXf::Zero(imgSize,6);
//    Eigen::VectorXf residualsDepth_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::VectorXf invDepthBuffer = Eigen::VectorXf::Zero(imgSize);

//    hessian = Eigen::Matrix<float,6,6>::Zero();
//    gradient = Eigen::Matrix<float,6,1>::Zero();

//    float weight_estim; // The weight computed from an M-estimator
//    //        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

//    //        double varianceRegularization = 1; // 63%
//    //        double stdDevReg = sqrt(varianceRegularization);
//    float stdDevPhoto_inv = 1./stdDevPhoto;
//    float stdDevDepth_inv = 1./stdDevDepth;

//    Eigen::Matrix3f rotation = poseGuess.block(0,0,3,3);
//    Eigen::Vector3f translation = poseGuess.block(0,3,3,1);

//    if(visualizeIterations)
//    {
//        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
//        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());

//        //            // Initialize the mask to segment the dynamic objects and the occlusions
//        //            mask_dynamic_occlusion = cv::Mat::zeros(nRows, nCols, CV_8U);
//    }

//    Eigen::VectorXf correspTrgSrc = Eigen::VectorXf::Zero(imgSize);
//    //std::vector<float> weightedError_(imgSize,-1);
//    int numVisiblePixels = 0;

//#if ENABLE_OPENMP
//#pragma omp parallel for reduction(+:numVisiblePixels)
//#endif
//    for (int i=0; i < LUT_xyz_source.size(); i++)
//        //            for (int r=0;r<nRows;r++)
//        //            {
//        //                for (int c=0;c<nCols;c++)
//    {
//        //int i = r*nCols + c;
//        if(LUT_xyz_source(i,0) != INVALID_POINT) //Compute the jacobian only for the valid points
//        {
//            Eigen::Vector3f xyz = rotation*LUT_xyz_source[i] + translation;

//            // Project the 3D point to the 2D plane
//            float inv_transf_z = 1.f/xyz(2);
//            float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//            transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//            transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//            int transformed_r_int = round(transformed_r);
//            int transformed_c_int = round(transformed_c);

//            // Asign the intensity value to the warped image and compute the difference between the transformed
//            // pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//            if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                    (transformed_c_int>=0 && transformed_c_int < nCols) )
//            {
//                // Discard occluded points
//                int ii = transformed_r_int*nCols + transformed_c_int; // The index of the pixel in the target image
//                if(invDepthBuffer(ii) == 0)
//                    ++numVisiblePixels;
//                else
//                {
//                    if(inv_transf_z < invDepthBuffer(ii)) // the current pixel is occluded
//                    {
//                        //                                    mask_dynamic_occlusion.at<uchar>(i) = 55;
//                        continue;
//                    }
//                    //                                else // The previous pixel used was occluded
//                    //                                    mask_dynamic_occlusion.at<uchar>(correspTrgSrc[ii]) = 55;
//                }

//                ++numVisiblePixels;
//                invDepthBuffer(ii) = inv_transf_z;
//                //correspTrgSrc(ii) = i;

//                // Compute the pixel jacobian
//                Eigen::Matrix<float,2,6> jacobianWarpRt;

//                // Derivative with respect to x
//                jacobianWarpRt(0,0)=fx*inv_transf_z;
//                jacobianWarpRt(1,0)=0;

//                // Derivative with respect to y
//                jacobianWarpRt(0,1)=0;
//                jacobianWarpRt(1,1)=fy*inv_transf_z;

//                // Derivative with respect to z
//                float inv_transf_z_2 = inv_transf_z*inv_transf_z;
//                jacobianWarpRt(0,2)=-fx*xyz(0)*inv_transf_z_2;
//                jacobianWarpRt(1,2)=-fy*xyz(1)*inv_transf_z_2;

//                // Derivative with respect to \w_x
//                jacobianWarpRt(0,3)=-fx*xyz(1)*xyz(0)*inv_transf_z_2;
//                jacobianWarpRt(1,3)=-fy*(1+xyz(1)*xyz(1)*inv_transf_z_2);

//                // Derivative with respect to \w_y
//                jacobianWarpRt(0,4)= fx*(1+xyz(0)*xyz(0)*inv_transf_z_2);
//                jacobianWarpRt(1,4)= fy*xyz(0)*xyz(1)*inv_transf_z_2;

//                // Derivative with respect to \w_z
//                jacobianWarpRt(0,5)=-fx*xyz(1)*inv_transf_z;
//                jacobianWarpRt(1,5)= fy*xyz(0)*inv_transf_z;

//                float pixel_src, intensity, depth1, depth;
//                float weightedErrorPhoto, weightedErrorDepth;
//                Eigen::Matrix<float,1,6> jacobianPhoto, jacobianDepth;

//                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                {
//                    if(visualizeIterations)
//                        warped_gray.at<float>(transformed_r_int,transformed_c_int) = graySrcPyr[pyramidLevel].at<float>(i);

//                    Eigen::Matrix<float,1,2> img_gradient;
//                    img_gradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
//                    img_gradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    if(fabs(img_gradient(0,0)) < thres_saliency_gray_ && fabs(img_gradient(0,1)) < thres_saliency_gray_)
//                        continue;

//                    //Obtain the pixel values that will be used to compute the pixel residual
//                    pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
//                    intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                    //                        cout << "pixel_src " << pixel_src << " intensity " << intensity << endl;
//                    float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                    float weight_estim = weightMEstimator(weightedError);
//                    //                            if(weightedError2 > varianceRegularization)
//                    //                                float weight_estim = sqrt(2*stdDevReg*abs(weightedError2)-varianceRegularization) / weightedError2_norm;
//                    weightedErrorPhoto = weight_estim * weightedError;

//                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
//                    jacobianPhoto = weight_estim * img_gradient*jacobianWarpRt;
//                    //                        cout << "img_gradient " << img_gradient << endl;
//                    //cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << endl;

//                    jacobiansPhoto.block(i,0,1,6) = jacobianPhoto;
//                    residualsPhoto_src(i) = weightedErrorPhoto;
//                }
//                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                {
//                    if(visualizeIterations)
//                        warped_depth.at<float>(transformed_r_int,transformed_c_int) = xyz(2);

//                    Eigen::Matrix<float,1,2> depth_gradient;
//                    depth_gradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    depth_gradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    if(fabs(depth_gradient(0,0)) < thres_saliency_depth_ && fabs(depth_gradient(0,1)) < thres_saliency_depth_)
//                        continue;

//                    //Obtain the depth values that will be used to the compute the depth residual
//                    depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                    {
//                        float depth1 = xyz(2);
//                        float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                        float weightedError = (depth - depth1)/stdDevError;
//                        float weight_estim = weightMEstimator(weightedError);
//                        weightedErrorDepth = weight_estim * weightedError;

//                        //Depth jacobian:
//                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
//                        Eigen::Matrix<float,1,6> jacobianRt_z;
//                        jacobianRt_z << 0,0,1,xyz(1),-xyz(0),0;
//                        jacobianDepth = weight_estim * (depth_gradient*jacobianWarpRt-jacobianRt_z);
//                        //                            cout << "jacobianDepth " << jacobianDepth << " weightedErrorDepth " << weightedErrorDepth << endl;

//                        jacobiansDepth.block(i,0,1,6) = jacobianDepth;
//                        residualsDepth_src(i) = weightedErrorDepth;
//                    }
//                }
//            }
//        }
//    }
//    //}

//    //        std::cout << "jacobiansPhoto \n" << jacobiansPhoto << std::endl;
//    //        std::cout << "residualsPhoto_src \n" << residualsPhoto_src.transpose() << std::endl;
//    //        std::cout << "jacobiansDepth \n" << jacobiansDepth << std::endl;
//    //        std::cout << "residualsDepth_src \n" << residualsDepth_src.transpose() << std::endl;

//    //#if ENABLE_OPENMP
//    //#pragma omp parallel for reduction (+:hessian,gradient) // Cannot reduce on Eigen types
//    //#endif
//    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//        for (int i=0; i<imgSize; i++)
//            if(residualsPhoto_src(i) != 0)
//            {
//                hessian += jacobiansPhoto.block(i,0,1,6).transpose() * jacobiansPhoto.block(i,0,1,6);
//                gradient += jacobiansPhoto.block(i,0,1,6).transpose() * residualsPhoto_src(i);
//            }
//    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//        for (int i=0; i<imgSize; i++)
//            if(residualsPhoto_src(i) != 0)
//            {
//                hessian += jacobiansDepth.block(i,0,1,6).transpose() * jacobiansDepth.block(i,0,1,6);
//                gradient += jacobiansDepth.block(i,0,1,6).transpose() * residualsDepth_src(i);
//            }
//}

///*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method. */
//double RegisterDense::errorDense_Occ2(const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method )
//{
//    //double error2 = 0.0; // Squared error
//    double PhotoResidual = 0.0;
//    double DepthResidual = 0.0;
//    int n_ptsPhoto = 0;
//    int n_ptsDepth = 0;

//    const size_t nRows = graySrcPyr[pyramidLevel].rows;
//    const size_t nCols = graySrcPyr[pyramidLevel].cols;
//    const size_t imgSize = nRows*nCols;

//    Eigen::VectorXf residualsPhoto_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::VectorXf residualsDepth_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::VectorXf invDepthBuffer = Eigen::VectorXf::Zero(imgSize);

//    const float scaleFactor = 1.0/pow(2,pyramidLevel);
//    fx = cameraMatrix(0,0)*scaleFactor;
//    fy = cameraMatrix(1,1)*scaleFactor;
//    ox = cameraMatrix(0,2)*scaleFactor;
//    oy = cameraMatrix(1,2)*scaleFactor;
//    const float inv_fx = 1./fx;
//    const float inv_fy = 1./fy;

//    //        float varianceRegularization = 1; // 63%
//    //        float stdDevReg = sqrt(varianceRegularization);
//    float weight_estim; // The weight computed from an M-estimator
//    float stdDevPhoto_inv = 1./stdDevPhoto;
//    float stdDevDepth_inv = 1./stdDevDepth;

//    //    std::cout << "poseGuess \n" << poseGuess << std::endl;

//    Eigen::Matrix3f rotation = poseGuess.block(0,0,3,3);
//    Eigen::Vector3f translation = poseGuess.block(0,3,3,1);

//    //        if(use_salient_pixels_)
//    //        {

//    //#if ENABLE_OPENMP
//    //#pragma omp parallel for reduction (+:error2)
//    //#endif
//    //            for (int i=0; i < vSalientPixels[pyramidLevel].size(); i++)
//    //            {
//    ////                //                int i = nCols*r+c; //vector index
//    ////                int r = vSalientPixels[pyramidLevel][i] / nCols;
//    ////                int c = vSalientPixels[pyramidLevel][i] % nCols;

//    ////                //Compute the 3D coordinates of the pij of the source frame
//    ////                Eigen::Vector3f point3D;
//    ////                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//    ////                if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//    ////                {
//    ////                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
//    ////                    point3D(1)=(r - oy) * point3D(2) * inv_fy;

//    ////                    //Transform the 3D point using the transformation matrix Rt
//    ////                    Eigen::Vector3f xyz = rotation*point3D + translation;

//    //                if(LUT_xyz_source[vSalientPixels[pyramidLevel][i]](0) != INVALID_POINT) //Compute the jacobian only for the valid points
//    //                {
//    //                    Eigen::Vector3f xyz = rotation*LUT_xyz_source[vSalientPixels[pyramidLevel][i]] + translation;

//    //                    //Project the 3D point to the 2D plane
//    //                    float inv_transf_z = 1.f/xyz(2);
//    //                    float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//    //                    transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//    //                    transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//    //                    int transformed_r_int = round(transformed_r);
//    //                    int transformed_c_int = round(transformed_c);

//    //                    //Asign the intensity value to the warped image and compute the difference between the transformed
//    //                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//    //                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//    //                        (transformed_c_int>=0 && transformed_c_int < nCols) )
//    //                    {
//    //                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//    //                        {
//    //                            //Obtain the pixel values that will be used to compute the pixel residual
//    //                            // float pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//    //                            float pixel_src = graySrcPyr[pyramidLevel].data[vSalientPixels[pyramidLevel][i]]; // Intensity value of the pixel(r,c) of source frame
//    //                            float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//    //                            float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//    //                            float weight_estim = weightMEstimator(weightedError);
//    //                            float weightedErrorPhoto = weight_estim * weightedError;
//    //                            // Apply M-estimator weighting
//    ////                            if(weightedError2 > varianceRegularization)
//    ////                            {
//    //////                                float weightedError2_norm = sqrt(weightedError2);
//    //////                                double weight = sqrt(2*stdDevReg*weightedError2_norm-varianceRegularization) / weightedError2_norm;
//    ////                                weightedError2 = 2*stdDevReg*sqrt(weightedError2)-varianceRegularization;
//    ////                            }
//    //                            error2 += weightedErrorPhoto*weightedErrorPhoto;

//    //                        }
//    //                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//    //                        {
//    //                            float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//    //                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//    //                            {
//    //                                //Obtain the depth values that will be used to the compute the depth residual
//    //                                float depth1 = xyz(2);
//    //                                float weightedError = depth - depth1;
//    //                                //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//    //                                //float stdDevError = stdDevDepth*depth1;
//    //                                float stdDevError = std::max (stdDevDepth*(xyz(2)*xyz(2)+depth*depth), 2*stdDevDepth);
//    //                                weight_estim = weightMEstimator(weightedError);
//    //                                float weightedErrorDepth = weight_estim * weightedError;
//    //                                error2 += weightedErrorDepth*weightedErrorDepth;
//    //                            }
//    //                        }
//    //                    }
//    //                }
//    //            }
//    //        }
//    //        else
//    {
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:n_ptsPhoto,n_ptsDepth) // n_ptsPhoto,  //for reduction (+:error2)
//#endif
//        for (int i=0; i < LUT_xyz_source.size(); i++)
//        {
//            //int i = r*nCols + c;
//            // The depth is represented by the 'z' coordinate of LUT_xyz_source[i]
//            if(LUT_xyz_source(i,0) != INVALID_POINT) //Compute the jacobian only for the valid points
//            {
//                Eigen::Vector3f xyz = rotation*LUT_xyz_source[i] + translation;

//                //Project the 3D point to the 2D plane
//                float inv_transf_z = 1.f/xyz(2);
//                float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//                transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//                int transformed_r_int = round(transformed_r);
//                int transformed_c_int = round(transformed_c);

//                //Asign the intensity value to the warped image and compute the difference between the transformed
//                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                        (transformed_c_int>=0 && transformed_c_int < nCols) )
//                {
//                    // Discard outliers (occluded and moving points)
//                    float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    float weightedError = depth - inv_transf_z;
//                    if(fabs(weightedError) > thresDepthOutliers)
//                        continue;

//                    // Discard occluded points
//                    int ii = transformed_r_int*nCols + transformed_c_int; // The index of the pixel in the target image
//                    if(invDepthBuffer(ii) > 0 && inv_transf_z < invDepthBuffer(ii)) // the current pixel is occluded
//                        continue;
//                    invDepthBuffer(ii) = inv_transf_z;


//                    //                            // Discard outlies: both from occlusions and moving objects
//                    //                            if( fabs() )

//                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        if(fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_gray_ &&
//                                fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_gray_)
//                            continue;

//                        //Obtain the pixel values that will be used to compute the pixel residual
//                        float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
//                        //float pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                        float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                        float weight_estim = weightMEstimator(weightedError);
//                        float weightedErrorPhoto = weight_estim * weightedError;
//                        // Apply M-estimator weighting
//                        //                            if(weightedError2 > varianceRegularization)
//                        //                            {
//                        ////                                float weightedError2_norm = sqrt(weightedError2);
//                        ////                                double weight = sqrt(2*stdDevReg*weightedError2_norm-varianceRegularization) / weightedError2_norm;
//                        //                                weightedError2 = 2*stdDevReg*sqrt(weightedError2)-varianceRegularization;
//                        //                            }
//                        residualsPhoto_src(ii) = weightedErrorPhoto*weightedErrorPhoto;
//                        ++n_ptsPhoto;
//                        // error2 += weightedErrorPhoto*weightedErrorPhoto;
//                        //                            std::cout << "error2 " << error2 << " weightedErrorPhoto " << weightedErrorPhoto << " " << weight_estim << " " << weightedError << std::endl;
//                    }
//                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                        {
//                            if( fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_depth_ &&
//                                    fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) < thres_saliency_depth_)
//                                continue;

//                            //Obtain the depth values that will be used to the compute the depth residual
//                            float depth1 = xyz(2);
//                            float weightedError = depth - depth1;
//                            //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//                            //float stdDevError = stdDevDepth*depth1;
//                            float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                            float weight_estim = weightMEstimator(weightedError);
//                            float weightedErrorDepth = weight_estim * weightedError;
//                            //error2 += weightedErrorDepth*weightedErrorDepth;
//                            residualsDepth_src(ii) = weightedErrorDepth*weightedErrorDepth;
//                            ++n_ptsDepth;
//                        }
//                    }
//                }
//            }
//        }
//        //}
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:PhotoResidual,DepthResidual) // n_ptsPhoto, n_ptsDepth
//#endif
//        for(int i=0; i < imgSize; i++)
//        {
//            PhotoResidual += residualsPhoto_src(i);
//            DepthResidual += residualsDepth_src(i);
//            //                if(residualsPhoto_src(i) > 0) ++n_ptsPhoto;
//            //                if(residualsDepth_src(i) > 0) ++n_ptsDepth;
//        }
//    }

//    avPhotoResidual = sqrt(PhotoResidual / n_ptsPhoto);
//    //avPhotoResidual = sqrt(PhotoResidual / n_ptsDepth);
//    avDepthResidual = sqrt(DepthResidual / n_ptsDepth);
//    avResidual = avPhotoResidual + avDepthResidual;

//    // std::cout << "PhotoResidual " << PhotoResidual << " DepthResidual " << DepthResidual << std::endl;
//    // std::cout << "n_ptsPhoto " << n_ptsPhoto << " n_ptsDepth " << n_ptsDepth << std::endl;
//    // std::cout << "avPhotoResidual " << avPhotoResidual << " avDepthResidual " << avDepthResidual << std::endl;

//    return avResidual;
//}

///*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient. */
//void RegisterDense::calcHessGrad_Occ2(const int &pyramidLevel,
//                                         const Eigen::Matrix4f poseGuess,
//                                         costFuncType method )
//{
//    int nRows = graySrcPyr[pyramidLevel].rows;
//    int nCols = graySrcPyr[pyramidLevel].cols;
//    const size_t imgSize = nRows*nCols;

//    const float scaleFactor = 1.0/pow(2,pyramidLevel);
//    fx = cameraMatrix(0,0)*scaleFactor;
//    fy = cameraMatrix(1,1)*scaleFactor;
//    ox = cameraMatrix(0,2)*scaleFactor;
//    oy = cameraMatrix(1,2)*scaleFactor;
//    const float inv_fx = 1./fx;
//    const float inv_fy = 1./fy;

//    Eigen::MatrixXf jacobiansPhoto = Eigen::MatrixXf::Zero(imgSize,6);
//    Eigen::VectorXf residualsPhoto_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::MatrixXf jacobiansDepth = Eigen::MatrixXf::Zero(imgSize,6);
//    Eigen::VectorXf residualsDepth_src = Eigen::VectorXf::Zero(imgSize);
//    Eigen::VectorXf invDepthBuffer = Eigen::VectorXf::Zero(imgSize);

//    hessian = Eigen::Matrix<float,6,6>::Zero();
//    gradient = Eigen::Matrix<float,6,1>::Zero();

//    float weight_estim; // The weight computed from an M-estimator
//    //        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

//    //        double varianceRegularization = 1; // 63%
//    //        double stdDevReg = sqrt(varianceRegularization);
//    float stdDevPhoto_inv = 1./stdDevPhoto;
//    float stdDevDepth_inv = 1./stdDevDepth;

//    Eigen::Matrix3f rotation = poseGuess.block(0,0,3,3);
//    Eigen::Vector3f translation = poseGuess.block(0,3,3,1);

//    if(visualizeIterations)
//    {
//        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
//        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());

//        // Initialize the mask to segment the dynamic objects and the occlusions
//        mask_dynamic_occlusion = cv::Mat::zeros(nRows, nCols, CV_8U);
//    }

//    Eigen::VectorXf correspTrgSrc = Eigen::VectorXf::Zero(imgSize);
//    std::vector<float> weightedError_(imgSize,-1);
//    int numVisiblePixels = 0;
//    float pixel_src, intensity, depth1, depth;

//#if ENABLE_OPENMP
//#pragma omp parallel for reduction(+:numVisiblePixels)
//#endif
//    for (int i=0; i < LUT_xyz_source.size(); i++)
//        //            for (int r=0;r<nRows;r++)
//        //            {
//        //                for (int c=0;c<nCols;c++)
//    {
//        //int i = r*nCols + c;
//        if(LUT_xyz_source(i,0) != INVALID_POINT) //Compute the jacobian only for the valid points
//        {
//            Eigen::Vector3f xyz = rotation*LUT_xyz_source[i] + translation;

//            // Project the 3D point to the 2D plane
//            float inv_transf_z = 1.f/xyz(2);
//            float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//            transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//            transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//            int transformed_r_int = round(transformed_r);
//            int transformed_c_int = round(transformed_c);

//            // Asign the intensity value to the warped image and compute the difference between the transformed
//            // pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//            if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                    (transformed_c_int>=0 && transformed_c_int < nCols) )
//            {
//                // Discard outliers (occluded and moving points)
//                //float dist_src = xyz.norm();
//                depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                depth1 = xyz(2);
//                float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                float weightedError = (depth - depth1)/stdDevError;
//                if(fabs(weightedError) > thresDepthOutliers)
//                {
//                    //                            if(visualizeIterations)
//                    //                                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    //                                {
//                    //                                    if(weightedError > 0)
//                    //                                        mask_dynamic_occlusion.at<uchar>(i) = 255;
//                    //                                    else
//                    //                                        mask_dynamic_occlusion.at<uchar>(i) = 155;
//                    //                                }
//                    //assert(false);
//                    continue;
//                }

//                // Discard occluded points
//                int ii = transformed_r_int*nCols + transformed_c_int; // The index of the pixel in the target image
//                if(invDepthBuffer(ii) == 0)
//                    ++numVisiblePixels;
//                else
//                {
//                    if(inv_transf_z < invDepthBuffer(ii)) // the current pixel is occluded
//                    {
//                        mask_dynamic_occlusion.at<uchar>(i) = 55;
//                        continue;
//                    }
//                    else // The previous pixel used was occluded
//                        mask_dynamic_occlusion.at<uchar>(correspTrgSrc[ii]) = 55;
//                }

//                ++numVisiblePixels;
//                invDepthBuffer(ii) = inv_transf_z;
//                correspTrgSrc(ii) = i;

//                // Compute the pixel jacobian
//                Eigen::Matrix<float,2,6> jacobianWarpRt;

//                // Derivative with respect to x
//                jacobianWarpRt(0,0)=fx*inv_transf_z;
//                jacobianWarpRt(1,0)=0;

//                // Derivative with respect to y
//                jacobianWarpRt(0,1)=0;
//                jacobianWarpRt(1,1)=fy*inv_transf_z;

//                // Derivative with respect to z
//                float inv_transf_z_2 = inv_transf_z*inv_transf_z;
//                jacobianWarpRt(0,2)=-fx*xyz(0)*inv_transf_z_2;
//                jacobianWarpRt(1,2)=-fy*xyz(1)*inv_transf_z_2;

//                // Derivative with respect to \w_x
//                jacobianWarpRt(0,3)=-fx*xyz(1)*xyz(0)*inv_transf_z_2;
//                jacobianWarpRt(1,3)=-fy*(1+xyz(1)*xyz(1)*inv_transf_z_2);

//                // Derivative with respect to \w_y
//                jacobianWarpRt(0,4)= fx*(1+xyz(0)*xyz(0)*inv_transf_z_2);
//                jacobianWarpRt(1,4)= fy*xyz(0)*xyz(1)*inv_transf_z_2;

//                // Derivative with respect to \w_z
//                jacobianWarpRt(0,5)=-fx*xyz(1)*inv_transf_z;
//                jacobianWarpRt(1,5)= fy*xyz(0)*inv_transf_z;

//                float weightedErrorPhoto, weightedErrorDepth;
//                Eigen::Matrix<float,1,6> jacobianPhoto, jacobianDepth;

//                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                {
//                    if(visualizeIterations)
//                        warped_gray.at<float>(transformed_r_int,transformed_c_int) = graySrcPyr[pyramidLevel].at<float>(i);

//                    Eigen::Matrix<float,1,2> img_gradient;
//                    img_gradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
//                    img_gradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    if(fabs(img_gradient(0,0)) < thres_saliency_gray_ && fabs(img_gradient(0,1)) < thres_saliency_gray_)
//                        continue;

//                    //Obtain the pixel values that will be used to compute the pixel residual
//                    pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
//                    intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                    //                        cout << "pixel_src " << pixel_src << " intensity " << intensity << endl;
//                    float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                    float weight_estim = weightMEstimator(weightedError);
//                    //                            if(weightedError2 > varianceRegularization)
//                    //                                float weight_estim = sqrt(2*stdDevReg*abs(weightedError2)-varianceRegularization) / weightedError2_norm;
//                    weightedErrorPhoto = weight_estim * weightedError;

//                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
//                    jacobianPhoto = weight_estim * img_gradient*jacobianWarpRt;
//                    //                        cout << "img_gradient " << img_gradient << endl;

//                    jacobiansPhoto.block(i,0,1,6) = jacobianPhoto;
//                    residualsPhoto_src(i) = weightedErrorPhoto;
//                }
//                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                {
//                    if(visualizeIterations)
//                        warped_depth.at<float>(transformed_r_int,transformed_c_int) = xyz(2);

//                    Eigen::Matrix<float,1,2> depth_gradient;
//                    depth_gradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    depth_gradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    if(fabs(depth_gradient(0,0)) < thres_saliency_depth_ && fabs(depth_gradient(0,1)) < thres_saliency_depth_)
//                        continue;

//                    //Obtain the depth values that will be used to the compute the depth residual
//                    depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                    if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                    {
//                        float depth1 = xyz(2);
//                        float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                        float weightedError = (depth - depth1)/stdDevError;
//                        float weight_estim = weightMEstimator(weightedError);
//                        weightedErrorDepth = weight_estim * weightedError;

//                        //Depth jacobian:
//                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
//                        Eigen::Matrix<float,1,6> jacobianRt_z;
//                        jacobianRt_z << 0,0,1,xyz(1),-xyz(0),0;
//                        jacobianDepth = weight_estim * (depth_gradient*jacobianWarpRt-jacobianRt_z);
//                        //                            cout << "jacobianDepth " << jacobianDepth << " weightedErrorDepth " << weightedErrorDepth << endl;

//                        jacobiansDepth.block(i,0,1,6) = jacobianDepth;
//                        residualsDepth_src(i) = weightedErrorDepth;
//                        weightedError_[ii] = fabs(weightedError);
//                    }
//                    else
//                        assert(false);
//                }
//            }
//        }
//    }
//    //}
//    //#if ENABLE_OPENMP
//    //#pragma omp parallel for reduction (+:hessian,gradient) // Cannot reduce on Eigen types
//    //#endif
//    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//        for (int i=0; i<imgSize; i++)
//            if(residualsPhoto_src(i) != 0)
//            {
//                hessian += jacobiansPhoto.block(i,0,1,6).transpose() * jacobiansPhoto.block(i,0,1,6);
//                gradient += jacobiansPhoto.block(i,0,1,6).transpose() * residualsPhoto_src(i);
//            }
//    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//        for (int i=0; i<imgSize; i++)
//            if(residualsPhoto_src(i) != 0)
//            {
//                hessian += jacobiansDepth.block(i,0,1,6).transpose() * jacobiansDepth.block(i,0,1,6);
//                gradient += jacobiansDepth.block(i,0,1,6).transpose() * residualsDepth_src(i);
//            }

//    SSO = (float)numVisiblePixels / imgSize;
//    //        std::cout << "numVisiblePixels " << numVisiblePixels << " imgSize " << imgSize << " sso " << SSO << std::endl;

//    std::vector<float> diffDepth(imgSize);
//    int validPt = 0;
//    for(int i=0; i < imgSize; i++)
//        if(weightedError_[i] >= 0)
//            diffDepth[validPt++] = weightedError_[i];
//    float diffDepthMean, diffDepthStDev;
//    calcMeanAndStDev(diffDepth, diffDepthMean, diffDepthStDev);
//    std::cout << "diffDepthMean " << diffDepthMean << " diffDepthStDev " << diffDepthStDev << " trans " << poseGuess.block(0,3,3,1).norm() << " sso " << SSO << std::endl;
//}



///*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        This is done following the work in:
//        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
//        in Computer Vision Workshops (ICCV Workshops), 2011. */
//void RegisterDense::calcHessianAndGradient(const int &pyramidLevel,
//                                              const Eigen::Matrix4f poseGuess,
//                                              costFuncType method )
//{
//    int nRows = graySrcPyr[pyramidLevel].rows;
//    int nCols = graySrcPyr[pyramidLevel].cols;

//    float scaleFactor = 1.0/pow(2,pyramidLevel);
//    float fx = cameraMatrix(0,0)*scaleFactor;
//    float fy = cameraMatrix(1,1)*scaleFactor;
//    float ox = cameraMatrix(0,2)*scaleFactor;
//    float oy = cameraMatrix(1,2)*scaleFactor;
//    float inv_fx = 1./fx;
//    float inv_fy = 1./fy;

//    hessian = Eigen::Matrix<float,6,6>::Zero();
//    gradient = Eigen::Matrix<float,6,1>::Zero();

//    float weight_estim; // The weight computed from an M-estimator
//    //        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

//    //        double varianceRegularization = 1; // 63%
//    //        double stdDevReg = sqrt(varianceRegularization);
//    float stdDevPhoto_inv = 1./stdDevPhoto;
//    float stdDevDepth_inv = 1./stdDevDepth;

//    Eigen::Matrix3f rotation = poseGuess.block(0,0,3,3);
//    Eigen::Vector3f translation = poseGuess.block(0,3,3,1);

//    if(visualizeIterations)
//    {
//        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
//        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
//    }

//    //        if(use_salient_pixels_)
//    //        {
//    //#if ENABLE_OPENMP
//    //#pragma omp parallel for
//    //#endif
//    //            for (int i=0; i < vSalientPixels[pyramidLevel].size(); i++)
//    //            {
//    //                int r = vSalientPixels[pyramidLevel][i] / nCols;
//    //                int c = vSalientPixels[pyramidLevel][i] % nCols;
//    ////            cout << "vSalientPixels[pyramidLevel][i] " << vSalientPixels[pyramidLevel][i] << " " << r << " " << c << endl;

//    //                //Compute the 3D coordinates of the pij of the source frame
//    //                Eigen::Vector3f point3D;
//    //                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//    ////            cout << "z " << depthSrcPyr[pyramidLevel].at<float>(r,c) << endl;
//    //                if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//    //                {
//    //                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
//    //                    point3D(1)=(r - oy) * point3D(2) * inv_fy;

//    //                    //Transform the 3D point using the transformation matrix Rt
//    //                    Eigen::Vector3f rotatedPoint3D = rotation*point3D;
//    //                    Eigen::Vector3f xyz = rotatedPoint3D + translation;
//    //                    rotatedPoint3D = xyz;
//    ////                    Eigen::Vector3f xyz = rotation*point3D + translation;

//    //                    //Project the 3D point to the 2D plane
//    //                    float inv_transf_z = 1.f/xyz(2);
//    //                    float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//    //                    transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//    //                    transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//    //                    int transformed_r_int = round(transformed_r);
//    //                    int transformed_c_int = round(transformed_c);

//    //                    //Asign the intensity value to the warped image and compute the difference between the transformed
//    //                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//    //                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//    //                        (transformed_c_int>=0 && transformed_c_int < nCols) )
//    //                    {
//    //                        //Compute the pixel jacobian
//    //                        Eigen::Matrix<float,2,6> jacobianWarpRt;

//    //                        //Derivative with respect to x
//    //                        jacobianWarpRt(0,0)=fx*inv_transf_z;
//    //                        jacobianWarpRt(1,0)=0;

//    //                        //Derivative with respect to y
//    //                        jacobianWarpRt(0,1)=0;
//    //                        jacobianWarpRt(1,1)=fy*inv_transf_z;

//    //                        //Derivative with respect to z
//    //                        jacobianWarpRt(0,2)=-fx*xyz(0)*inv_transf_z*inv_transf_z;
//    //                        jacobianWarpRt(1,2)=-fy*xyz(1)*inv_transf_z*inv_transf_z;

//    //                        //Derivative with respect to \lambda_x
//    //                        jacobianWarpRt(0,3)=-fx*rotatedPoint3D(1)*xyz(0)*inv_transf_z*inv_transf_z;
//    //                        jacobianWarpRt(1,3)=-fy*(rotatedPoint3D(2)+rotatedPoint3D(1)*xyz(1)*inv_transf_z)*inv_transf_z;

//    //                        //Derivative with respect to \lambda_y
//    //                        jacobianWarpRt(0,4)= fx*(rotatedPoint3D(2)+rotatedPoint3D(0)*xyz(0)*inv_transf_z)*inv_transf_z;
//    //                        jacobianWarpRt(1,4)= fy*rotatedPoint3D(0)*xyz(1)*inv_transf_z*inv_transf_z;

//    //                        //Derivative with respect to \lambda_z
//    //                        jacobianWarpRt(0,5)=-fx*rotatedPoint3D(1)*inv_transf_z;
//    //                        jacobianWarpRt(1,5)= fy*rotatedPoint3D(0)*inv_transf_z;

//    ////                        float weight_estim; // The weight computed from an M-estimator
//    ////                        float weightedError2, weightedError2;
//    //                        float pixel_src, intensity, depth1, depth;
//    //                        float weightedErrorPhoto, weightedErrorDepth;
//    //                        Eigen::Matrix<float,1,6> jacobianPhoto, jacobianDepth;

//    //                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//    //                        {
//    //                            //Obtain the pixel values that will be used to compute the pixel residual
//    //                            pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//    //                            intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//    ////                        cout << "pixel_src " << pixel_src << " intensity " << intensity << endl;
//    //                            float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//    //                            float weight_estim = weightMEstimator(weightedError);
//    ////                            if(weightedError2 > varianceRegularization)
//    ////                                weight_estim = sqrt(2*stdDevReg*abs(weightedError2)-varianceRegularization) / weightedError2_norm;
//    //                            weightedErrorPhoto = weight_estim * weightedError;

//    //                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
//    //                            Eigen::Matrix<float,1,2> img_gradient;
//    //                            img_gradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
//    //                            img_gradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//    //                            jacobianPhoto = weight_estim * img_gradient*jacobianWarpRt;
//    ////                        cout << "img_gradient " << img_gradient << endl;

//    //                            if(visualizeIterations)
//    //                                warped_gray.at<float>(transformed_r_int,transformed_c_int) = pixel_src;
//    //                        }
//    //                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//    //                        {
//    //                            //Obtain the depth values that will be used to the compute the depth residual
//    //                            depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//    //                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//    //                            {
//    //                                depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
//    //                                float weightedError = depth - depth1;
//    //                                //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//    //                                float stdDevError = stdDevDepth*depth1;
//    //                                float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//    //                                float weight_estim = weightMEstimator(weightedError);
//    //                                weightedErrorDepth = weight_estim * weightedError;

//    //                                //Depth jacobian:
//    //                                //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
//    //                                Eigen::Matrix<float,1,2> depth_gradient;
//    //                                depth_gradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//    //                                depth_gradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);

//    //                                Eigen::Matrix<float,1,6> jacobianRt_z;
//    //                                jacobianRt_z << 0,0,1,rotatedPoint3D(1),-rotatedPoint3D(0),0;
//    //                                jacobianDepth = weight_estim * (depth_gradient*jacobianWarpRt-jacobianRt_z);
//    ////                            cout << "jacobianDepth " << jacobianDepth << " weightedErrorDepth " << weightedErrorDepth << endl;

//    //                                if(visualizeIterations)
//    //                                    warped_depth.at<float>(transformed_r_int,transformed_c_int) = depth1;
//    //                            }
//    //                        }

//    //                        //Assign the pixel residual and jacobian to its corresponding row
//    //#if ENABLE_OPENMP
//    //#pragma omp critical
//    //#endif
//    //                        {
//    //                            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//    //                            {
//    //                                // Photometric component
//    ////                                hessian += jacobianPhoto.transpose()*jacobianPhoto / varPhoto;
//    //                                hessian += jacobianPhoto.transpose()*jacobianPhoto;
//    //                                gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
//    //                            }
//    //                            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//    //                                if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//    //                            {
//    //                                // Depth component (Plane ICL like)
//    //                                hessian += jacobianDepth.transpose()*jacobianDepth;
//    //                                gradient += jacobianDepth.transpose()*weightedErrorDepth;
//    //                            }
//    //                        }

//    //                    }
//    //                }
//    //            }
//    //        }
//    //        else
//    {
//#if ENABLE_OPENMP
//#pragma omp parallel for
//#endif
//        for (int r=0;r<nRows;r++)
//        {
//            for (int c=0;c<nCols;c++)
//            {
//                //Compute the 3D coordinates of the pij of the source frame
//                Eigen::Vector3f point3D;
//                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//                //            cout << "z " << depthSrcPyr[pyramidLevel].at<float>(r,c) << endl;
//                if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//                {
//                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
//                    point3D(1)=(r - oy) * point3D(2) * inv_fy;

//                    //Transform the 3D point using the transformation matrix Rt
//                    //                    Eigen::Vector3f rotatedPoint3D = rotation*point3D;
//                    //                    Eigen::Vector3f xyz = rotatedPoint3D + translation;
//                    Eigen::Vector3f xyz = rotation*point3D + translation;

//                    //Project the 3D point to the 2D plane
//                    float inv_transf_z = 1.f/xyz(2);
//                    float transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                    transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//                    transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//                    int transformed_r_int = round(transformed_r);
//                    int transformed_c_int = round(transformed_c);

//                    //Asign the intensity value to the warped image and compute the difference between the transformed
//                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                            (transformed_c_int>=0 && transformed_c_int < nCols) )
//                    {
//                        //Compute the pixel jacobian
//                        Eigen::Matrix<float,2,6> jacobianWarpRt;

//                        //Derivative with respect to x
//                        jacobianWarpRt(0,0)=fx*inv_transf_z;
//                        jacobianWarpRt(1,0)=0;

//                        //Derivative with respect to y
//                        jacobianWarpRt(0,1)=0;
//                        jacobianWarpRt(1,1)=fy*inv_transf_z;

//                        //Derivative with respect to z
//                        float inv_transf_z_2 = inv_transf_z*inv_transf_z;
//                        jacobianWarpRt(0,2)=-fx*xyz(0)*inv_transf_z_2;
//                        jacobianWarpRt(1,2)=-fy*xyz(1)*inv_transf_z_2;

//                        //Derivative with respect to \lambda_x
//                        jacobianWarpRt(0,3)=-fx*xyz(1)*xyz(0)*inv_transf_z_2;
//                        jacobianWarpRt(1,3)=-fy*(1+xyz(1)*xyz(1)*inv_transf_z_2);

//                        //Derivative with respect to \lambda_y
//                        jacobianWarpRt(0,4)= fx*(1+xyz(0)*xyz(0)*inv_transf_z_2);
//                        jacobianWarpRt(1,4)= fy*xyz(0)*xyz(1)*inv_transf_z_2;

//                        //Derivative with respect to \lambda_z
//                        jacobianWarpRt(0,5)=-fx*xyz(1)*inv_transf_z;
//                        jacobianWarpRt(1,5)= fy*xyz(0)*inv_transf_z;

//                        float pixel_src, intensity, depth1, depth;
//                        float weightedErrorPhoto, weightedErrorDepth;
//                        Eigen::Matrix<float,1,6> jacobianPhoto, jacobianDepth;

//                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            if(visualizeIterations)
//                                warped_gray.at<float>(transformed_r_int,transformed_c_int) = graySrcPyr[pyramidLevel].at<float>(r,c);

//                            Eigen::Matrix<float,1,2> img_gradient;
//                            img_gradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
//                            img_gradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            if(fabs(img_gradient(0,0)) < thres_saliency_gray_ && fabs(img_gradient(0,1)) < thres_saliency_gray_)
//                                continue;

//                            //Obtain the pixel values that will be used to compute the pixel residual
//                            pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                            intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                            //                        cout << "pixel_src " << pixel_src << " intensity " << intensity << endl;
//                            float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                            float weight_estim = weightMEstimator(weightedError);
//                            //                            if(weightedError2 > varianceRegularization)
//                            //                                float weight_estim = sqrt(2*stdDevReg*abs(weightedError2)-varianceRegularization) / weightedError2_norm;
//                            weightedErrorPhoto = weight_estim * weightedError;

//                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
//                            jacobianPhoto = weight_estim * img_gradient*jacobianWarpRt;
//                            //                        cout << "img_gradient " << img_gradient << endl;
//                        }
//                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            if(visualizeIterations)
//                                warped_depth.at<float>(transformed_r_int,transformed_c_int) = xyz(2);

//                            Eigen::Matrix<float,1,2> depth_gradient;
//                            depth_gradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            depth_gradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            if(fabs(depth_gradient(0,0)) < thres_saliency_depth_ && fabs(depth_gradient(0,1)) < thres_saliency_depth_)
//                                continue;

//                            //Obtain the depth values that will be used to the compute the depth residual
//                            depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                            {
//                                float depth1 = xyz(2);
//                                float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                                float weightedError = (depth - depth1)/stdDevError;
//                                float weight_estim = weightMEstimator(weightedError);
//                                weightedErrorDepth = weight_estim * weightedError;

//                                //Depth jacobian:
//                                //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
//                                Eigen::Matrix<float,1,6> jacobianRt_z;
//                                jacobianRt_z << 0,0,1,xyz(1),-xyz(0),0;
//                                jacobianDepth = weight_estim * (depth_gradient*jacobianWarpRt-jacobianRt_z);
//                                //                            cout << "jacobianDepth " << jacobianDepth << " weightedErrorDepth " << weightedErrorDepth << endl;
//                            }
//                        }

//                        //Assign the pixel residual and jacobian to its corresponding row
//#if ENABLE_OPENMP
//#pragma omp critical
//#endif
//                        {
//                            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                            {
//                                // Photometric component
//                                //                                hessian += jacobianPhoto.transpose()*jacobianPhoto / varPhoto;
//                                hessian += jacobianPhoto.transpose()*jacobianPhoto;
//                                gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
//                            }
//                            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                                if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                                {
//                                    // Depth component (Plane ICL like)
//                                    hessian += jacobianDepth.transpose()*jacobianDepth;
//                                    gradient += jacobianDepth.transpose()*weightedErrorDepth;
//                                }
//                        }

//                    }
//                }
//            }
//        }
//    }
//}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::warpImage(const int &pyramidLevel,
                              const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                              costFuncType method ) //,  const bool use_bilinear )
{
    std::cout << " RegisterDense::warpImage \n";

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = graySrcPyr[pyramidLevel].size().area();
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    const float half_width = nCols/2 - 0.5f;

    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5f*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    // depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];
    const float stdDevPhoto_inv = 1.f/stdDevPhoto;
    const float stdDevDepth_inv = 1.f/stdDevDepth;

    //computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
    transformPts3D_sse(LUT_xyz_source, poseGuess, xyz_src_transf);

    float *_depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);

    //    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    //    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    //    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    //    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        //warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type(),-1000);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        //warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
        warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type(),-1000);

    float *_warpedGray = reinterpret_cast<float*>(warped_gray.data);
    float *_warpedDepth = reinterpret_cast<float*>(warped_depth.data);

//    if( use_bilinear_ )
//    {
//         std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
//#if ENABLE_OPENMP
//#pragma omp parallel for
//#endif
//        for(size_t i=0; i < imgSize; i++)
//        {
//            //Transform the 3D point using the transformation matrix Rt
//            Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
//            // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

//            //Project the 3D point to the S2 sphere
//            float dist = xyz.norm();
//            float dist_inv = 1.f / dist;
//            float phi = asin(xyz(1)*dist_inv);
//            float transformed_r = (phi-phi_start)*pixel_angle_inv;
//            int transformed_r_int = round(transformed_r);
//            // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
//            //Asign the intensity value to the warped image and compute the difference between the transformed
//            //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//            if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
//            {
//                //visible_pixels_src(i) = 1;
//                float theta = atan2(xyz(0),xyz(2));
//                float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
//                int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
//                cv::Point2f warped_pixel(transformed_r, transformed_c);

//                size_t warped_i = transformed_r_int * nCols + transformed_c_int;
//                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    _warpedGray[warped_i] = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel );
//                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                    _warpedDepth[warped_i] = dist;
//            }
//        }
//    }
//    else
    {       
         std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(size_t i=0; i < imgSize; i++)
        {
            //Transform the 3D point using the transformation matrix Rt
            Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
            // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

            //Project the 3D point to the S2 sphere
            float dist = xyz.norm();
            float dist_inv = 1.f / dist;
            float phi = asin(xyz(1)*dist_inv);
            //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
            int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
            // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
            //Asign the intensity value to the warped image and compute the difference between the transformed
            //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
            if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
            {
                //visible_pixels_src(i) = 1;
                float theta = atan2(xyz(0),xyz(2));
                int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    _warpedGray[warped_i] = _graySrcPyr[i];
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    _warpedDepth[warped_i] = dist;
            }
        }
    }

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " warpImage took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
double RegisterDense::computeReprojError_spherical (  const int &pyramidLevel,
                                                      const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                                      costFuncType method,
                                                      const int direction ) //,  const bool use_bilinear )
{
    //std::cout << " RegisterDense::errorDense_sphere \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    assert(direction == 1 || direction == -1);

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = graySrcPyr[pyramidLevel].size().area();
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    const float half_width = nCols/2 - 0.5f;

    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5f*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    // depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];
    const float stdDevPhoto_inv = 1.f/stdDevPhoto;
    const float stdDevDepth_inv = 1.f/stdDevDepth;

    const size_t n_pts; // The amount of points to be reprojected (it may be smaller than imgSize if salient points are employed)
    float *_residualsPhoto, *_residualsDepth, *_stdDevError_inv, *_wEstimPhoto, *_wEstimDepth; // List of residuals and M-Estimator weights to be stored for future operation
    int *_warp_pixels, *_validPixelsPhoto, *_validPixelsDepth; // List of warped pixel locations
    float *_warp_img; // (x,y)_warped pixels -> For the bilinear interpolation
    float *_depthTrgPyr, *_graySrcPyr, *_grayTrgPyr; // Pointers to the data of the source and target images
    float *_depthSrcGradXPyr, *_depthSrcGradYPyr, *_graySrcGradXPyr, *_graySrcGradYPyr; // Pointers to the image gradients
    Eigen::MatrixXf &LUT_xyz, &xyz_transf;

    if(direction == 1)
    {
        n_pts = LUT_xyz_source.rows();
        //cout << "n_pts " << n_pts << " / " << imgSize << endl;

        //transformPts3D(LUT_xyz_source, poseGuess, xyz_src_transf);
        transformPts3D_sse(LUT_xyz_source, poseGuess, xyz_src_transf);
        LUT_xyz = LUT_xyz_source;
        xyz_transf = xyz_src_transf;

        residualsPhoto_src.resize(n_pts);
        residualsDepth_src.resize(n_pts);
        stdDevError_inv_src.resize(n_pts);
        wEstimPhoto_src.resize(n_pts);
        wEstimDepth_src.resize(n_pts);
        warp_pixels_src = Eigen::VectorXi::Constant(n_pts,-1);
        //visible_pixels_src = Eigen::VectorXi::Zero(n_pts);
        validPixelsPhoto_src = Eigen::VectorXi::Zero(n_pts);
        validPixelsDepth_src = Eigen::VectorXi::Zero(n_pts);

        *_residualsPhoto = &residualsPhoto_src(0);
        *_residualsDepth = &residualsDepth_src(0);
        *_stdDevError_inv = &stdDevError_inv_src(0);
        *_wEstimPhoto = &wEstimPhoto_src(0);
        *_wEstimDepth = &wEstimDepth_src(0);
        *_warp_pixels = &warp_pixels_src(0);
        *_validPixelsPhoto = &validPixelsPhoto_src(0);
        *_validPixelsDepth = &validPixelsDepth_src(0);
        if( use_bilinear_ && pyramidLevel ==0 )
        {
            warp_img_src.resize(2,n_pts);
            _warp_img = &warp_img_src(0,0);
        }

        _depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyramidLevel].data);
        _graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
        _grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);

        //    _depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
        //    _depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
        //    _grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
        //    _grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

        _depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
        _depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
        _graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
        _graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);
    }
    else
    {
        n_pts = LUT_xyz_target.rows();
        //cout << "n_pts " << n_pts << " / " << imgSize << endl;

        transformPts3D_sse(LUT_xyz_target, poseGuess.inverse(), xyz_trg_transf);
        LUT_xyz = LUT_xyz_target;
        xyz_transf = xyz_trg_transf;

        residualsPhoto_trg.resize(n_pts);
        residualsDepth_trg.resize(n_pts);
        stdDevError_inv_trg.resize(n_pts);
        wEstimPhoto_trg.resize(n_pts);
        wEstimDepth_trg.resize(n_pts);
        warp_pixels_trg = Eigen::VectorXi::Constant(n_pts,-1);
        //visible_pixels_trg = Eigen::VectorXi::Zero(n_pts);
        validPixelsPhoto_trg = Eigen::VectorXi::Zero(n_pts);
        validPixelsDepth_trg = Eigen::VectorXi::Zero(n_pts);

        *_residualsPhoto = &residualsPhoto_trg(0);
        *_residualsDepth = &residualsDepth_trg(0);
        *_stdDevError_inv = &stdDevError_inv_trg(0);
        *_wEstimPhoto = &wEstimPhoto_trg(0);
        *_wEstimDepth = &wEstimDepth_trg(0);
        *_warp_pixels = &warp_pixels_trg(0);
        *_validPixelsPhoto = &validPixelsPhoto_trg(0);
        *_validPixelsDepth = &validPixelsDepth_trg(0);
        if( use_bilinear_ && pyramidLevel ==0 )
        {
            warp_img_trg.resize(2,n_pts);
            _warp_img = &warp_img_trg(0,0);
        }

        // The cource and target references are all inverted following pointers
        _depthTrgPyr = reinterpret_cast<float*>(depthSrcPyr[pyramidLevel].data);
        _graySrcPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);
        _grayTrgPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);

        //    _depthTrgGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
        //    _depthTrgGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
        //    _grayTrgGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
        //    _grayTrgGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

        _depthSrcGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
        _depthSrcGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
        _graySrcGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
        _graySrcGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);
    }

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    //std::vector<float> v_AD_intensity_(imgSize);

    if(method == PHOTO_DEPTH)
    {
        if(use_salient_pixels_)
        {
            if( !use_bilinear_ || pyramidLevel !=0 )
            {
                //std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < n_pts; i++)
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        //visible_pixels(i) = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        _warp_pixels[i] = warped_i;

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //if( fabs(_graySrcGradXPyr[_validPixels[i]]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[_validPixels[i]]) > thres_saliency_gray_)
                        {
                            _validPixelsPhoto[i] = 1;
                            // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[_validPixels[i]];
                            _residualsPhoto[i] = diff * stdDevPhoto_inv;
                            _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                            //                            std::cout << i << " " << _validPixels[i] << " warped_i " << warped_i << " error2_photo " << error2_photo << " residualsPhoto " << _residualsPhoto[i] << " weight_estim " << _wEstimPhoto[i] << std::endl;
                            //                            std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[_validPixels[i]] " << _graySrcPyr[_validPixels[i]] << std::endl;
                            //                            mrpt::system::pause();
                            //v_AD_intensity[i] = fabs(diff);
                        }

                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                        float depth = _depthTrgPyr[warped_i];
                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            //if( fabs(_depthSrcGradXPyr[_validPixels[i]]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[_validPixels[i]]) > thres_saliency_depth_)
                            {
                                _validPixelsDepth[i] = 1;
                                _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);

                                _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];

                                _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                //std::cout << i << " error2_depth " << error2_depth << " wDepthError " << _residualsDepth[i] << " weight_estim " << _wEstimDepth[i] << std::endl;
                            }
                        }
                    }
                }
            }
            else // BILINEAR
            {
                const float nCols_1 = nCols-1;
                std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
                    int transformed_r_int = round(transformed_r);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        //_visible_pixels[i] = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        float transformed_c = half_width + theta*pixel_angle_inv;
                        if(transformed_c > nCols_1);
                        continue;
                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        _warp_pixels[i] = warped_i;
                        _warp_img[2*i] = transformed_r;
                        _warp_img[2*i+1] = transformed_c;
                        cv::Point2f warped_pixel(transformed_r,transformed_c);
                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);


                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        //if( fabs(_graySrcGradXPyr[_validPixels[i]]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[_validPixels[i]]) > thres_saliency_gray_)
                        {
                            _validPixelsPhoto[i] = 1;
                            float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _graySrcPyr[i];
                            _residualsPhoto[i] = diff * stdDevPhoto_inv;
                            _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                            //std::cout << i << " error2_photo " << error2_photo << " wDepthPhoto " << _residualsPhoto[i] << " weight_estim " << _wEstimPhoto[i] << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }

                        float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                            //if( fabs(_depthSrcGradXPyr[_validPixels[i]]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[_validPixels[i]]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                _validPixelsDepth[i] = 1;
                                _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];
                                _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if( !use_bilinear_ || pyramidLevel !=0 )
            {
                // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    if( _validPixels[i] ) //Compute the jacobian only for the valid points
                    {
                        //Transform the 3D point using the transformation matrix Rt
                        Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                        // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

                        //Project the 3D point to the S2 sphere
                        float dist = xyz.norm();
                        float dist_inv = 1.f / dist;
                        float phi = asin(xyz(1)*dist_inv);
                        //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                        int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                        // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                        //Asign the intensity value to the warped image and compute the difference between the transformed
                        //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                        if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                        {
                            //_visible_pixels[i] = 1;
                            float theta = atan2(xyz(0),xyz(2));
                            int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                            size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                            _warp_pixels[i] = warped_i;
                            //if(compute_MAD_stdDev_)
                            //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
                            ++numVisiblePts;

                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            if( fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f)
                            {
                                //Obtain the pixel values that will be used to compute the pixel residual
                                // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                _validPixelsPhoto[i] = 1;
                                float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                                _residualsPhoto[i] = diff * stdDevPhoto_inv;
                                _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                                //                                std::cout << i << " " << _validPixels[i] << " error2_photo " << error2_photo << " wDepthPhoto " << _residualsPhoto[i] << " i " << i << " w " << warped_i << " c " << transformed_c_int << " " << theta*pixel_angle_inv << " theta " << theta << " weight_estim " << _wEstimPhoto[i] << std::endl;
                                //                                std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[i] " << _graySrcPyr[i] << std::endl;
                                //                                mrpt::system::pause();

                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }

                            //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float depth = _depthTrgPyr[warped_i];
                            if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                if( fabs(_depthSrcGradXPyr[i]) > 0.f || fabs(_depthSrcGradYPyr[i]) > 0.f )
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    _validPixelsDepth[i] = 1;
                                    _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];
                                    _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;

                                }
                            }
                        }
                    }
                }
            }
            else
            {
                std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
                const float nCols_1 = nCols-1;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    //Compute the 3D coordinates of the pij of the source frame
                    if( _validPixels[i] ) //Compute the jacobian only for the valid points
                    {
                        Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                        // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                        //Project the 3D point to the S2 sphere
                        float dist = xyz.norm();
                        float dist_inv = 1.f / dist;
                        float phi = asin(xyz(1)*dist_inv);
                        float transformed_r = (phi-phi_start)*pixel_angle_inv;
                        int transformed_r_int = round(transformed_r);
                        //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                        // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                        //Asign the intensity value to the warped image and compute the difference between the transformed
                        //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                        if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                        {
                            //_visible_pixels[i] = 1;
                            ++numVisiblePts;
                            float theta = atan2(xyz(0),xyz(2));
                            float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
                            int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                            size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                            _warp_pixels[i] = warped_i;
                            _warp_img[2*i] = transformed_r;
                            _warp_img[2*i+1] = transformed_c;
                            cv::Point2f warped_pixel(transformed_r,transformed_c);
                            // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                            // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            {
                                _validPixelsPhoto[i] = 1;
                                float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                float diff = intensity - _graySrcPyr[i];
                                _residualsPhoto[i] = diff * stdDevPhoto_inv;
                                _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                                // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }

                            float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    _validPixelsDepth[i] = 1;
                                    _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];
                                    _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
    } ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(method == PHOTO_CONSISTENCY)
    {
        if(use_salient_pixels_)
        {
            if( !use_bilinear_ || pyramidLevel !=0 )
            {
                //std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < n_pts; i++)
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        //visible_pixels(i) = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        _warp_pixels[i] = warped_i;

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //if( fabs(_graySrcGradXPyr[_validPixels[i]]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[_validPixels[i]]) > thres_saliency_gray_)
                        {
                            _validPixelsPhoto[i] = 1;
                            // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[_validPixels[i]];
                            _residualsPhoto[i] = diff * stdDevPhoto_inv;
                            _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                            //                            std::cout << i << " " << _validPixels[i] << " warped_i " << warped_i << " error2_photo " << error2_photo << " residualsPhoto " << _residualsPhoto[i] << " weight_estim " << _wEstimPhoto[i] << std::endl;
                            //                            std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[_validPixels[i]] " << _graySrcPyr[_validPixels[i]] << std::endl;
                            //                            mrpt::system::pause();
                            //v_AD_intensity[i] = fabs(diff);
                        }
                    }
                }
            }
            else // BILINEAR
            {
                const float nCols_1 = nCols-1;
                std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
                    int transformed_r_int = round(transformed_r);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        //_visible_pixels[i] = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        float transformed_c = half_width + theta*pixel_angle_inv;
                        if(transformed_c > nCols_1);
                        continue;
                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        _warp_pixels[i] = warped_i;
                        _warp_img[2*i] = transformed_r;
                        _warp_img[2*i+1] = transformed_c;
                        cv::Point2f warped_pixel(transformed_r,transformed_c);
                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);


                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        //if( fabs(_graySrcGradXPyr[_validPixels[i]]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[_validPixels[i]]) > thres_saliency_gray_)
                        {
                            _validPixelsPhoto[i] = 1;
                            float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _graySrcPyr[i];
                            _residualsPhoto[i] = diff * stdDevPhoto_inv;
                            _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                            //std::cout << i << " error2_photo " << error2_photo << " wDepthPhoto " << _residualsPhoto[i] << " weight_estim " << _wEstimPhoto[i] << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                }
            }
        }
        else
        {
            if( !use_bilinear_ || pyramidLevel !=0 )
            {
                // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    if( _validPixels[i] ) //Compute the jacobian only for the valid points
                    {
                        //Transform the 3D point using the transformation matrix Rt
                        Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                        // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

                        //Project the 3D point to the S2 sphere
                        float dist = xyz.norm();
                        float dist_inv = 1.f / dist;
                        float phi = asin(xyz(1)*dist_inv);
                        //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                        int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                        // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                        //Asign the intensity value to the warped image and compute the difference between the transformed
                        //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                        if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                        {
                            //_visible_pixels[i] = 1;
                            float theta = atan2(xyz(0),xyz(2));
                            int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                            size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                            _warp_pixels[i] = warped_i;
                            //if(compute_MAD_stdDev_)
                            //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
                            ++numVisiblePts;

                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            if( fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f)
                            {
                                //Obtain the pixel values that will be used to compute the pixel residual
                                // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                _validPixelsPhoto[i] = 1;
                                float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                                _residualsPhoto[i] = diff * stdDevPhoto_inv;
                                _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                                //                                std::cout << i << " " << _validPixels[i] << " error2_photo " << error2_photo << " wDepthPhoto " << _residualsPhoto[i] << " i " << i << " w " << warped_i << " c " << transformed_c_int << " " << theta*pixel_angle_inv << " theta " << theta << " weight_estim " << _wEstimPhoto[i] << std::endl;
                                //                                std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[i] " << _graySrcPyr[i] << std::endl;
                                //                                mrpt::system::pause();

                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                    }
                }
            }
            else
            {
                std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
                const float nCols_1 = nCols-1;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    //Compute the 3D coordinates of the pij of the source frame
                    if( _validPixels[i] ) //Compute the jacobian only for the valid points
                    {
                        Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                        // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                        //Project the 3D point to the S2 sphere
                        float dist = xyz.norm();
                        float dist_inv = 1.f / dist;
                        float phi = asin(xyz(1)*dist_inv);
                        float transformed_r = (phi-phi_start)*pixel_angle_inv;
                        int transformed_r_int = round(transformed_r);
                        //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                        // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                        //Asign the intensity value to the warped image and compute the difference between the transformed
                        //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                        if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                        {
                            //_visible_pixels[i] = 1;
                            ++numVisiblePts;
                            float theta = atan2(xyz(0),xyz(2));
                            float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
                            int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                            size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                            _warp_pixels[i] = warped_i;
                            _warp_img[2*i] = transformed_r;
                            _warp_img[2*i+1] = transformed_c;
                            cv::Point2f warped_pixel(transformed_r,transformed_c);
                            // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                            // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            {
                                _validPixelsPhoto[i] = 1;
                                float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                float diff = intensity - _graySrcPyr[i];
                                _residualsPhoto[i] = diff * stdDevPhoto_inv;
                                _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                                // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                    }
                }
            }
        }
    } ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(method == DEPTH_CONSISTENCY)
    {
        if(use_salient_pixels_)
        {
            if( !use_bilinear_ || pyramidLevel !=0 )
            {
                //std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < n_pts; i++)
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        //visible_pixels(i) = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        _warp_pixels[i] = warped_i;

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //if( fabs(_graySrcGradXPyr[_validPixels[i]]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[_validPixels[i]]) > thres_saliency_gray_)
                        {
                            _validPixelsPhoto[i] = 1;
                            // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[_validPixels[i]];
                            _residualsPhoto[i] = diff * stdDevPhoto_inv;
                            _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                            //                            std::cout << i << " " << _validPixels[i] << " warped_i " << warped_i << " error2_photo " << error2_photo << " residualsPhoto " << _residualsPhoto[i] << " weight_estim " << _wEstimPhoto[i] << std::endl;
                            //                            std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[_validPixels[i]] " << _graySrcPyr[_validPixels[i]] << std::endl;
                            //                            mrpt::system::pause();
                            //v_AD_intensity[i] = fabs(diff);
                        }

                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                        float depth = _depthTrgPyr[warped_i];
                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            //if( fabs(_depthSrcGradXPyr[_validPixels[i]]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[_validPixels[i]]) > thres_saliency_depth_)
                            {
                                _validPixelsDepth[i] = 1;
                                _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];
                                _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                //std::cout << i << " error2_depth " << error2_depth << " wDepthError " << _residualsDepth[i] << " weight_estim " << _wEstimDepth[i] << std::endl;
                            }
                        }
                    }
                }
            }
            else // BILINEAR
            {
                const float nCols_1 = nCols-1;
                std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
                    int transformed_r_int = round(transformed_r);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        //_visible_pixels[i] = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        float transformed_c = half_width + theta*pixel_angle_inv;
                        if(transformed_c > nCols_1);
                        continue;
                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        _warp_pixels[i] = warped_i;
                        _warp_img[2*i] = transformed_r;
                        _warp_img[2*i+1] = transformed_c;
                        cv::Point2f warped_pixel(transformed_r,transformed_c);
                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);


                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        //if( fabs(_graySrcGradXPyr[_validPixels[i]]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[_validPixels[i]]) > thres_saliency_gray_)
                        {
                            _validPixelsPhoto[i] = 1;
                            float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _graySrcPyr[i];
                            _residualsPhoto[i] = diff * stdDevPhoto_inv;
                            _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                            //std::cout << i << " error2_photo " << error2_photo << " wDepthPhoto " << _residualsPhoto[i] << " weight_estim " << _wEstimPhoto[i] << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }

                        float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                            //if( fabs(_depthSrcGradXPyr[_validPixels[i]]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[_validPixels[i]]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                _validPixelsDepth[i] = 1;
                                _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];
                                _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if( !use_bilinear_ || pyramidLevel !=0 )
            {
                // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    if( _validPixels[i] ) //Compute the jacobian only for the valid points
                    {
                        //Transform the 3D point using the transformation matrix Rt
                        Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                        // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_transf.block(i,0,1,3) << endl;

                        //Project the 3D point to the S2 sphere
                        float dist = xyz.norm();
                        float dist_inv = 1.f / dist;
                        float phi = asin(xyz(1)*dist_inv);
                        //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                        int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                        // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                        //Asign the intensity value to the warped image and compute the difference between the transformed
                        //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                        if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                        {
                            //_visible_pixels[i] = 1;
                            float theta = atan2(xyz(0),xyz(2));
                            int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                            size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                            _warp_pixels[i] = warped_i;
                            //if(compute_MAD_stdDev_)
                            //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
                            ++numVisiblePts;

                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            if( fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f)
                            {
                                //Obtain the pixel values that will be used to compute the pixel residual
                                // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                _validPixelsPhoto[i] = 1;
                                float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                                _residualsPhoto[i] = diff * stdDevPhoto_inv;
                                _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                                //                                std::cout << i << " " << _validPixels[i] << " error2_photo " << error2_photo << " wDepthPhoto " << _residualsPhoto[i] << " i " << i << " w " << warped_i << " c " << transformed_c_int << " " << theta*pixel_angle_inv << " theta " << theta << " weight_estim " << _wEstimPhoto[i] << std::endl;
                                //                                std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[i] " << _graySrcPyr[i] << std::endl;
                                //                                mrpt::system::pause();

                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }

                            //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float depth = _depthTrgPyr[warped_i];
                            if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                if( fabs(_depthSrcGradXPyr[i]) > 0.f || fabs(_depthSrcGradYPyr[i]) > 0.f )
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    _validPixelsDepth[i] = 1;
                                    _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];
                                    _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;

                                }
                            }
                        }
                    }
                }
            }
            else
            {
                std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
                const float nCols_1 = nCols-1;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
                for(size_t i=0; i < xyz_transf.rows(); i++)
                {
                    //Compute the 3D coordinates of the pij of the source frame
                    if( _validPixels[i] ) //Compute the jacobian only for the valid points
                    {
                        Eigen::Vector3f xyz = xyz_transf.block(i,0,1,3).transpose();
                        // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                        //Project the 3D point to the S2 sphere
                        float dist = xyz.norm();
                        float dist_inv = 1.f / dist;
                        float phi = asin(xyz(1)*dist_inv);
                        float transformed_r = (phi-phi_start)*pixel_angle_inv;
                        int transformed_r_int = round(transformed_r);
                        //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                        // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                        //Asign the intensity value to the warped image and compute the difference between the transformed
                        //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                        if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                        {
                            //_visible_pixels[i] = 1;
                            ++numVisiblePts;
                            float theta = atan2(xyz(0),xyz(2));
                            float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
                            int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                            size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                            _warp_pixels[i] = warped_i;
                            _warp_img[2*i] = transformed_r;
                            _warp_img[2*i+1] = transformed_c;
                            cv::Point2f warped_pixel(transformed_r,transformed_c);
                            // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                            // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            {
                                _validPixelsPhoto[i] = 1;
                                float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                float diff = intensity - _graySrcPyr[i];
                                _residualsPhoto[i] = diff * stdDevPhoto_inv;
                                _wEstimPhoto[i] = weightMEstimator(_residualsPhoto[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += _wEstimPhoto[i] * _residualsPhoto[i] * _residualsPhoto[i];
                                // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }

                            float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    _validPixelsDepth[i] = 1;
                                    _stdDevError_inv[i] = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    _residualsDepth[i] = (depth - dist) * _stdDevError_inv[i];
                                    _wEstimDepth[i] = weightMEstimator(_residualsDepth[i]); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += _wEstimDepth[i] * _residualsDepth[i] * _residualsDepth[i];
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    SSO = (float)numVisiblePts / n_pts;
    error2 = (error2_photo + error2_depth) / numVisiblePts;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " errorDense_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif

    return error2;
}

///*! Compute the median absulute deviation of the projection of reference image onto the target one */
//float computeMAD(const int &pyramidLevel)
//{
//}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
double RegisterDense::errorDense_sphere ( const int &pyramidLevel,
                                          const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                          costFuncType method ) //,  const bool use_bilinear )
{
    //std::cout << " RegisterDense::errorDense_sphere \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = graySrcPyr[pyramidLevel].size().area();
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    const float half_width = nCols/2 - 0.5f;

    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5f*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    // depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];
    const float stdDevPhoto_inv = 1.f/stdDevPhoto;
    const float stdDevDepth_inv = 1.f/stdDevDepth;

    const size_t n_pts = LUT_xyz_source.rows();
    //cout << "n_pts " << n_pts << " / " << imgSize << endl;

    //transformPts3D(LUT_xyz_source, poseGuess, xyz_src_transf);
    transformPts3D_sse(LUT_xyz_source, poseGuess, xyz_src_transf);

    warp_pixels_src = Eigen::VectorXi::Constant(n_pts,-1);
    residualsPhoto_src.resize(n_pts);
    residualsDepth_src.resize(n_pts);
    stdDevError_inv_src.resize(n_pts);
    wEstimPhoto_src.resize(n_pts);
    wEstimDepth_src.resize(n_pts);
    //visible_pixels_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsPhoto_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsDepth_src = Eigen::VectorXi::Zero(n_pts);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    //std::vector<float> v_AD_intensity_(imgSize);

    float *_depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);

    //    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    //    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    //    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    //    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
             //std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    //visible_pixels_src(i) = 1;
                    float theta = atan2(xyz(0),xyz(2));
                    int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    //if(compute_MAD_stdDev_)
                    //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)]);

                    // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " numVisiblePts " << numVisiblePts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);
                    //                    if(method == PHOTO_CONSISTENCY && method == PHOTO_DEPTH)
                    //                    {

                    //                    }
                    //                    else
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //Obtain the pixel values that will be used to compute the pixel residual
                        //if( fabs(_graySrcGradXPyr[validPixels_src(i)]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[validPixels_src(i)]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_src(i) = 1;
                            // float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                            // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//                            std::cout << i << " " << validPixels_src(i) << " warped_i " << warped_i << " error2_photo " << error2_photo << " residualsPhoto " << residualsPhoto_src(i) << " weight_estim " << wEstimPhoto_src(i) << std::endl;
//                            std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[validPixels_src(i)] " << _graySrcPyr[validPixels_src(i)] << std::endl;
//                            mrpt::system::pause();
                            //v_AD_intensity[i] = fabs(diff);
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                        float depth = _depthTrgPyr[warped_i];
                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            //if( fabs(_depthSrcGradXPyr[validPixels_src(i)]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[validPixels_src(i)]) > thres_saliency_depth_)
                            {
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                //std::cout << i << " error2_depth " << error2_depth << " wDepthError " << residualsDepth_src(i) << " weight_estim " << wEstimDepth_src(i) << std::endl;

                                //                            _validPixelsDepth_src(n_ptsDepth) = i;
                                //                            _stdDevError_inv_src(n_ptsDepth) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                //                            _residualsDepth_src(n_ptsDepth) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsDepth]) * stdDevError_inv_src(n_ptsDepth);
                                //                            _wEstimDepth_src(n_ptsDepth) = weightMEstimator(_residualsDepth_src(n_ptsDepth)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                //                            error2 += _wEstimDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth);
                            }
                        }
                    }
                }

//                Eigen::VectorXf v_theta(n_pts);
//                if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
//                {
//                    ++numVisiblePts;
//                    //visible_pixels_src(i) = 1;
//                    v_theta(i) = atan2(xyz(0),xyz(2));

//                    float *_theta = &v_theta(0);
//                    float *_warp_pixels = &warp_pixels_src(0);

//                    __m128 _pixel_angle_inv = _mm_set1_ps(pixel_angle_inv);
//                    __m128 _half_width = _mm_set1_ps(half_width);

//                    int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
//                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
//                    warp_pixels_src(i) = warped_i;
//                    //if(compute_MAD_stdDev_)
//                    //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)]);

//                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        //Obtain the pixel values that will be used to compute the pixel residual
//                        //if( fabs(_graySrcGradXPyr[validPixels_src(i)]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[validPixels_src(i)]) > thres_saliency_gray_)
//                        {
//                            validPixelsPhoto_src(i) = 1;
//                            // float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
//                            // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)];
//                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
//                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
//                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//                            //std::cout << i << " " << validPixels_src(i) << " error2_photo " << error2_photo << " wDepthPhoto " << residualsPhoto_src(i) << " i " << validPixels_src(i) << " w " << warped_i << " c " << transformed_c_int << " " << theta*pixel_angle_inv << " theta " << theta << " weight_estim " << wEstimPhoto_src(i) << std::endl;
//                            //v_AD_intensity[i] = fabs(diff);
//                        }
//                    }
//                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                        float depth = _depthTrgPyr[warped_i];
//                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                        {
//                            //Obtain the depth values that will be used to the compute the depth residual
//                            //if( fabs(_depthSrcGradXPyr[validPixels_src(i)]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[validPixels_src(i)]) > thres_saliency_depth_)
//                            {
//                                validPixelsDepth_src(i) = 1;
//                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
//                                residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
//                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
//                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
//                                //std::cout << i << " error2_depth " << error2_depth << " wDepthError " << residualsDepth_src(i) << " weight_estim " << wEstimDepth_src(i) << std::endl;

//                                //                            _validPixelsDepth_src(n_ptsDepth) = i;
//                                //                            _stdDevError_inv_src(n_ptsDepth) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
//                                //                            _residualsDepth_src(n_ptsDepth) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsDepth]) * stdDevError_inv_src(n_ptsDepth);
//                                //                            _wEstimDepth_src(n_ptsDepth) = weightMEstimator(_residualsDepth_src(n_ptsDepth)); // Apply M-estimator weighting // The weight computed by an M-estimator
//                                //                            error2 += _wEstimDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth);
//                            }
//                        }
//                    }
//                }
            }
        }
        else // BILINEAR
        {
            const float nCols_1 = nCols-1;
            std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
            warp_img_src.resize(imgSize, 2);
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_src_transf.rows(); i++)
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                float transformed_r = (phi-phi_start)*pixel_angle_inv;
                int transformed_r_int = round(transformed_r);
                //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    //visible_pixels_src(i) = 1;
                    float theta = atan2(xyz(0),xyz(2));
                    float transformed_c = half_width + theta*pixel_angle_inv;
                    if(transformed_c > nCols_1);
                        continue;
                    int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    warp_img_src(i,0) = transformed_r;
                    warp_img_src(i,1) = transformed_c;
                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        //if( fabs(_graySrcGradXPyr[validPixels_src(i)]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[validPixels_src(i)]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_src(i) = 1;
                            float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _graySrcPyr[i];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
                            //std::cout << i << " error2_photo " << error2_photo << " wDepthPhoto " << residualsPhoto_src(i) << " weight_estim " << wEstimPhoto_src(i) << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                            //if( fabs(_depthSrcGradXPyr[validPixels_src(i)]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[validPixels_src(i)]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
//        thres_saliency_gray_ = 0.0f;
//        thres_saliency_depth_ = 0.0f;
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
            // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_src_transf.rows(); i++)
            {
                if( validPixels_src(i) ) //Compute the jacobian only for the valid points
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                    {
                        //visible_pixels_src(i) = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_src(i) = warped_i;
                        //if(compute_MAD_stdDev_)
                        //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
                        ++numVisiblePts;

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            if( fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f)
                            {
                                //Obtain the pixel values that will be used to compute the pixel residual
                                //                        float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                                //                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                validPixelsPhoto_src(i) = 1;
                                float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                                residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                                wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//                                std::cout << i << " " << validPixels_src(i) << " error2_photo " << error2_photo << " wDepthPhoto " << residualsPhoto_src(i) << " i " << i << " w " << warped_i << " c " << transformed_c_int << " " << theta*pixel_angle_inv << " theta " << theta << " weight_estim " << wEstimPhoto_src(i) << std::endl;
//                                std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[i] " << _graySrcPyr[i] << std::endl;
//                                mrpt::system::pause();

                                //                        _validPixelsPhoto_src(n_ptsPhoto) = i;
                                //                        _residualsPhoto_src(n_ptsPhoto) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsPhoto]) * stdDevPhoto_inv;
                                //                        _wEstimPhoto_src(n_ptsPhoto) = weightMEstimator(_residualsPhoto_src(n_ptsPhoto)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                //                        error2 += _wEstimPhoto_src(n_ptsPhoto) * _residualsPhoto_src(n_ptsPhoto) * _residualsPhoto_src(n_ptsPhoto);

                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float depth = _depthTrgPyr[warped_i];
                            if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                if( fabs(_depthSrcGradXPyr[i]) > 0.f || fabs(_depthSrcGradYPyr[i]) > 0.f )
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    validPixelsDepth_src(i) = 1;
                                    stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                    wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;

                                    //                            _validPixelsDepth_src(n_ptsDepth) = i;
                                    //                            _stdDevError_inv_src(n_ptsDepth) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    //                            _residualsDepth_src(n_ptsDepth) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsDepth]) * stdDevError_inv_src(n_ptsDepth);
                                    //                            _wEstimDepth_src(n_ptsDepth) = weightMEstimator(_residualsDepth_src(n_ptsDepth)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    //                            error2 += _wEstimDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth) * _residualsDepth_src(n_ptsDepth);
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
            warp_img_src.resize(imgSize, 2);
            const float nCols_1 = nCols-1;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_src_transf.rows(); i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                if( validPixels_src(i) ) //Compute the jacobian only for the valid points
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
                    int transformed_r_int = round(transformed_r);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                    {
                        //visible_pixels_src(i) = 1;
                        ++numVisiblePts;
                        float theta = atan2(xyz(0),xyz(2));
                        float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_src(i) = warped_i;
                        warp_img_src(i,0) = transformed_r;
                        warp_img_src(i,1) = transformed_c;
                        cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            {
                                validPixelsPhoto_src(i) = 1;
                                float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                float diff = intensity - _graySrcPyr[i];
                                residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                                wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
                                // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    validPixelsDepth_src(i) = 1;
                                    stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                    wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

//    _validPixelsPhoto_src.resize(n_ptsPhoto);
//    _residualsPhoto_src.resize(n_ptsPhoto);
//    _wEstimPhoto_src.resize(n_ptsPhoto);

//    _validPixelsDepth_src.resize(n_ptsPhoto);
//    _stdDevError_inv_src.resize(n_ptsPhoto);
//    _residualsDepth_src.resize(n_ptsPhoto);
//    _wEstimDepth_src.resize(n_ptsPhoto);

    // Compute the median absulute deviation of the projection of reference image onto the target one to update the value of the standard deviation of the intesity error
//    if(error2_photo > 0 && compute_MAD_stdDev_)
//    {
//        std::cout << " stdDevPhoto PREV " << stdDevPhoto << std::endl;
//        size_t count_valid_pix = 0;
//        std::vector<float> v_AD_intensity(n_ptsPhoto);
//        for(size_t i=0; i < imgSize; i++)
//            if( validPixelsPhoto_src(i) )
//            {
//                v_AD_intensity[count_valid_pix] = v_AD_intensity_[i];
//                ++count_valid_pix;
//            }
//        //v_AD_intensity.resize(n_pts);
//        v_AD_intensity.resize(n_ptsPhoto);
//        float stdDevPhoto_updated = 1.4826 * median(v_AD_intensity);
//        error2_photo *= stdDevPhoto*stdDevPhoto / (stdDevPhoto_updated*stdDevPhoto_updated);
//        stdDevPhoto = stdDevPhoto_updated;
//        std::cout << " stdDevPhoto_updated    " << stdDevPhoto_updated << std::endl;
//    }

//    //computeMAD()
//    //std::cout << " computeMAD \n";
//    std::vector<float> v_diff_intensity(n_ptsPhoto);
//    size_t count_valid_pix = 0;
//    for(size_t i=0; i < imgSize; i++)
//        if( validPixelsPhoto_src(i) )
//        {
//            v_diff_intensity[count_valid_pix] = residualsPhoto_src(i) * stdDevPhoto;
//            ++count_valid_pix;
//        }
//    float median_diff = median(v_diff_intensity);
//    std::cout << " median_diff " << median_diff << " \n";
//    float median_diff_scaled = median_diff * stdDevPhoto_inv;
//    error2_photo = 0;
//    for(size_t i=0; i < imgSize; i++)
//        if( validPixelsPhoto_src(i) )
//        {
//            residualsPhoto_src(i) -= median_diff_scaled;
//            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//        }

    SSO = (float)numVisiblePts / n_pts;
    error2 = (error2_photo + error2_depth) / numVisiblePts;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " errorDense_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif

    return error2;
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
double RegisterDense::errorDenseWarp_sphere ( const int &pyramidLevel,
                                              const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                              costFuncType method ) //,  const bool use_bilinear )
{
    //std::cout << " RegisterDense::errorDense_sphere \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = graySrcPyr[pyramidLevel].size().area();
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    const float half_width = nCols/2 - 0.5f;

    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5f*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    // depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];
    const float stdDevPhoto_inv = 1.f/stdDevPhoto;
    const float stdDevDepth_inv = 1.f/stdDevDepth;

    const size_t n_pts = LUT_xyz_source.rows();
    //cout << "n_pts " << n_pts << " / " << imgSize << endl;

    //transformPts3D(LUT_xyz_source, poseGuess, xyz_src_transf);
    //transformPts3D_sse(LUT_xyz_source, poseGuess, xyz_src_transf);

    warp_pixels_src = Eigen::VectorXi::Constant(n_pts,-1);
    residualsPhoto_src.resize(n_pts);
    residualsDepth_src.resize(n_pts);
    stdDevError_inv_src.resize(n_pts);
    wEstimPhoto_src.resize(n_pts);
    wEstimDepth_src.resize(n_pts);
    //visible_pixels_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsPhoto_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsDepth_src = Eigen::VectorXi::Zero(n_pts);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    //std::vector<float> v_AD_intensity_(imgSize);

    float *_depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);

    //    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    //    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    //    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    //    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        //warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type(),-1000);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        //warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
        warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type(),-1000);

    float *_warpedGray = reinterpret_cast<float*>(warped_gray.data);
    float *_warpedDepth = reinterpret_cast<float*>(warped_depth.data);

//    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
             //std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    //visible_pixels_src(i) = 1;
                    float theta = atan2(xyz(0),xyz(2));
                    int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    //if(compute_MAD_stdDev_)
                    //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)]);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        _warpedGray[warped_i] = _graySrcPyr[i];
                        //Obtain the pixel values that will be used to compute the pixel residual
                        //if( fabs(_graySrcGradXPyr[validPixels_src(i)]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[validPixels_src(i)]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_src(i) = 1;
                            // float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                            // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//                            std::cout << i << " " << validPixels_src(i) << " warped_i " << warped_i << " error2_photo " << error2_photo << " residualsPhoto " << residualsPhoto_src(i) << " weight_estim " << wEstimPhoto_src(i) << std::endl;
//                            std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[validPixels_src(i)] " << _graySrcPyr[validPixels_src(i)] << std::endl;
//                            mrpt::system::pause();
                            //v_AD_intensity[i] = fabs(diff);
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                        _warpedDepth[warped_i] = dist;
                        float depth = _depthTrgPyr[warped_i];
                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            //if( fabs(_depthSrcGradXPyr[validPixels_src(i)]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[validPixels_src(i)]) > thres_saliency_depth_)
                            {
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                //std::cout << i << " error2_depth " << error2_depth << " wDepthError " << residualsDepth_src(i) << " weight_estim " << wEstimDepth_src(i) << std::endl;

                            }
                        }
                    }
                }
            }
        }
        else // BILINEAR
        {
            const float nCols_1 = nCols-1;
            std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
            warp_img_src.resize(imgSize, 2);
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_src_transf.rows(); i++)
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                float transformed_r = (phi-phi_start)*pixel_angle_inv;
                int transformed_r_int = round(transformed_r);
                //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    //visible_pixels_src(i) = 1;
                    float theta = atan2(xyz(0),xyz(2));
                    float transformed_c = half_width + theta*pixel_angle_inv;
                    if(transformed_c > nCols_1);
                        continue;
                    int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    warp_img_src(i,0) = transformed_r;
                    warp_img_src(i,1) = transformed_c;
                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        _warpedGray[warped_i] = _graySrcPyr[i];
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        //if( fabs(_graySrcGradXPyr[validPixels_src(i)]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[validPixels_src(i)]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_src(i) = 1;
                            float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _graySrcPyr[i];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
                            //std::cout << i << " error2_photo " << error2_photo << " wDepthPhoto " << residualsPhoto_src(i) << " weight_estim " << wEstimPhoto_src(i) << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        _warpedDepth[warped_i] = dist;
                        float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                            //if( fabs(_depthSrcGradXPyr[validPixels_src(i)]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[validPixels_src(i)]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
    }
//    else
//    {
//        if( !use_bilinear_ || pyramidLevel !=0 )
//        {
//            // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
//#endif
//            for(size_t i=0; i < xyz_src_transf.rows(); i++)
//            {
//                if( validPixels_src(i) ) //Compute the jacobian only for the valid points
//                {
//                    //Transform the 3D point using the transformation matrix Rt
//                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
//                    // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

//                    //Project the 3D point to the S2 sphere
//                    float dist = xyz.norm();
//                    float dist_inv = 1.f / dist;
//                    float phi = asin(xyz(1)*dist_inv);
//                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
//                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
//                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
//                    //Asign the intensity value to the warped image and compute the difference between the transformed
//                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
//                    {
//                        //visible_pixels_src(i) = 1;
//                        float theta = atan2(xyz(0),xyz(2));
//                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
//                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
//                        warp_pixels_src(i) = warped_i;
//                        //if(compute_MAD_stdDev_)
//                        //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
//                        ++numVisiblePts;

//                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
//                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
//                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
//                            if( fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f)
//                            {
//                                //Obtain the pixel values that will be used to compute the pixel residual
//                                //                        float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
//                                //                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                                validPixelsPhoto_src(i) = 1;
//                                float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
//                                residualsPhoto_src(i) = diff * stdDevPhoto_inv;
//                                wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
//                                error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
////                                std::cout << i << " " << validPixels_src(i) << " error2_photo " << error2_photo << " wDepthPhoto " << residualsPhoto_src(i) << " i " << i << " w " << warped_i << " c " << transformed_c_int << " " << theta*pixel_angle_inv << " theta " << theta << " weight_estim " << wEstimPhoto_src(i) << std::endl;
////                                std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[i] " << _graySrcPyr[i] << std::endl;
////                                mrpt::system::pause();
//                            }
//                        }
//                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            float depth = _depthTrgPyr[warped_i];
//                            if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                            {
//                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
//                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
//                                if( fabs(_depthSrcGradXPyr[i]) > 0.f || fabs(_depthSrcGradYPyr[i]) > 0.f )
//                                {
//                                    //Obtain the depth values that will be used to the compute the depth residual
//                                    validPixelsDepth_src(i) = 1;
//                                    stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
//                                    residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
//                                    wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
//                                    error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
//                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//        else
//        {
//            std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
//            warp_img_src.resize(imgSize, 2);
//            const float nCols_1 = nCols-1;
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
//#endif
//            for(size_t i=0; i < xyz_src_transf.rows(); i++)
//            {
//                //Compute the 3D coordinates of the pij of the source frame
//                if( validPixels_src(i) ) //Compute the jacobian only for the valid points
//                {
//                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
//                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

//                    //Project the 3D point to the S2 sphere
//                    float dist = xyz.norm();
//                    float dist_inv = 1.f / dist;
//                    float phi = asin(xyz(1)*dist_inv);
//                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
//                    int transformed_r_int = round(transformed_r);
//                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
//                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

//                    //Asign the intensity value to the warped image and compute the difference between the transformed
//                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
//                    {
//                        //visible_pixels_src(i) = 1;
//                        ++numVisiblePts;
//                        float theta = atan2(xyz(0),xyz(2));
//                        float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
//                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
//                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
//                        warp_pixels_src(i) = warped_i;
//                        warp_img_src(i,0) = transformed_r;
//                        warp_img_src(i,1) = transformed_c;
//                        cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
//                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
//                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

//                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
//                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
//                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
//                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
//                            {
//                                validPixelsPhoto_src(i) = 1;
//                                float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                                float diff = intensity - _graySrcPyr[i];
//                                residualsPhoto_src(i) = diff * stdDevPhoto_inv;
//                                wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
//                                error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//                                // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
//                                //v_AD_intensity[i] = fabs(diff);
//                                //++n_ptsPhoto;
//                            }
//                        }
//                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                            {
//                                // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
//                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
//                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
//                                {
//                                    //Obtain the depth values that will be used to the compute the depth residual
//                                    validPixelsDepth_src(i) = 1;
//                                    stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
//                                    residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
//                                    wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
//                                    error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
//                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
//                                    //++n_ptsDepth;
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }

    SSO = (float)numVisiblePts / n_pts;
    error2 = (error2_photo + error2_depth) / numVisiblePts;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " errorDenseWarp_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif

    return error2;
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
double RegisterDense::errorDenseIC_sphere(const int &pyramidLevel,
                                          const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                          costFuncType method ) //,  const bool use_bilinear )
{
    //std::cout << " RegisterDense::errorDense_sphere \n";
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = graySrcPyr[pyramidLevel].size().area();
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    const float half_width = nCols/2 - 0.5f;

    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5f*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    // depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];
    const float stdDevPhoto_inv = 1.f/stdDevPhoto;
    const float stdDevDepth_inv = 1.f/stdDevDepth;

    const size_t n_pts = LUT_xyz_source.rows();
    //cout << "n_pts " << n_pts << " / " << imgSize << endl;

    //transformPts3D(LUT_xyz_source, poseGuess, xyz_src_transf);
    transformPts3D_sse(LUT_xyz_source, poseGuess, xyz_src_transf);

    warp_pixels_src = Eigen::VectorXi::Constant(n_pts,-1);
    residualsPhoto_src.resize(n_pts);
    residualsDepth_src.resize(n_pts);
    stdDevError_inv_src.resize(n_pts);
    wEstimPhoto_src.resize(n_pts);
    wEstimDepth_src.resize(n_pts);
    //visible_pixels_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsPhoto_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsDepth_src = Eigen::VectorXi::Zero(n_pts);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    //std::vector<float> v_AD_intensity_(imgSize);

    float *_depthSrcPyr = reinterpret_cast<float*>(depthSrcPyr[pyramidLevel].data);
    float *_depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);

    //    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    //    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    //    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    //    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
             //std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.rows() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    //visible_pixels_src(i) = 1;
                    float theta = atan2(xyz(0),xyz(2));
                    int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    //if(compute_MAD_stdDev_)
                    //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)]);

                    // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " numVisiblePts " << numVisiblePts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);
                    //                    if(method == PHOTO_CONSISTENCY && method == PHOTO_DEPTH)
                    //                    {

                    //                    }
                    //                    else
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //Obtain the pixel values that will be used to compute the pixel residual
                        //if( fabs(_graySrcGradXPyr[validPixels_src(i)]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[validPixels_src(i)]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_src(i) = 1;
                            // float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                            // float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[validPixels_src(i)];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//                            std::cout << i << " " << validPixels_src(i) << " warped_i " << warped_i << " error2_photo " << error2_photo << " residualsPhoto " << residualsPhoto_src(i) << " weight_estim " << wEstimPhoto_src(i) << std::endl;
//                            std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[validPixels_src(i)] " << _graySrcPyr[validPixels_src(i)] << std::endl;
//                            mrpt::system::pause();
                            //v_AD_intensity[i] = fabs(diff);
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                        float dist_trg = _depthTrgPyr[warped_i];
                        if(dist_trg > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            //if( fabs(_depthSrcGradXPyr[validPixels_src(i)]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[validPixels_src(i)]) > thres_saliency_depth_)
                            {
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+dist_trg*dist_trg), 2*stdDevDepth);
                                //residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                Eigen::Vector3f xyz_trg = LUT_xyz_target.block(warped_i,0,1,3).transpose();
                                Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                                float dist_src = _depthSrcPyr[validPixels_src(i)];
                                residualsDepth_src(i) = (((xyz_src .dot (xyz_trg - poseGuess.block(0,3,3,1))) / dist_src) - dist_src) * stdDevError_inv_src(i);
                                //Eigen::Matrix<float,1,3> xyz_src = LUT_xyz_source.block(i,0,1,3);

                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                //std::cout << i << " error2_depth " << error2_depth << " wDepthError " << residualsDepth_src(i) << " weight_estim " << wEstimDepth_src(i) << std::endl;
                            }
                        }
                    }
                }
            }
        }
        else // BILINEAR
        {
            const float nCols_1 = nCols-1;
            std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
            warp_img_src.resize(imgSize, 2);
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_src_transf.rows(); i++)
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                float transformed_r = (phi-phi_start)*pixel_angle_inv;
                int transformed_r_int = round(transformed_r);
                //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    //visible_pixels_src(i) = 1;
                    float theta = atan2(xyz(0),xyz(2));
                    float transformed_c = half_width + theta*pixel_angle_inv;
                    if(transformed_c > nCols_1);
                        continue;
                    int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;
                    warp_img_src(i,0) = transformed_r;
                    warp_img_src(i,1) = transformed_c;
                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                        //if( fabs(_graySrcGradXPyr[validPixels_src(i)]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[validPixels_src(i)]) > thres_saliency_gray_)
                        {
                            validPixelsPhoto_src(i) = 1;
                            float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float diff = intensity - _graySrcPyr[i];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
                            //std::cout << i << " error2_photo " << error2_photo << " wDepthPhoto " << residualsPhoto_src(i) << " weight_estim " << wEstimPhoto_src(i) << std::endl;
                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        float dist_trg = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                        if(dist_trg > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                            //if( fabs(_depthSrcGradXPyr[validPixels_src(i)]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[validPixels_src(i)]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_src(i) = 1;
                                stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+dist_trg*dist_trg), 2*stdDevDepth);
                                //residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                Eigen::Vector3f xyz_trg;// = LUT_xyz_target.block(warped_i,0,1,3).transpose();
                                float theta = (transformed_r - half_width)*pixel_angle;
                                float phi = transformed_r * pixel_angle + phi_start;
                                float cos_phi = cos(phi);
                                xyz_trg(0) = dist_trg * cos_phi * sin(theta);
                                xyz_trg(1) = dist_trg * sin(phi);
                                xyz_trg(2) = dist_trg * cos_phi * cos(theta);
                                // TODO: change the sin / cos operations tp dp interpolation on precomputed vectors
                                Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                                float dist_src = _depthSrcPyr[validPixels_src(i)];
                                residualsDepth_src(i) = (((xyz_src .dot (xyz_trg - poseGuess.block(0,3,3,1))) / dist_src) - dist_src) * stdDevError_inv_src(i);                                wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
//        thres_saliency_gray_ = 0.0f;
//        thres_saliency_depth_ = 0.0f;
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
            // std::cout << " Standart Nearest-Neighbor LUT " << LUT_xyz_source.size() << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_src_transf.rows(); i++)
            {
                if( validPixels_src(i) ) //Compute the jacobian only for the valid points
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                    {
                        //visible_pixels_src(i) = 1;
                        float theta = atan2(xyz(0),xyz(2));
                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_src(i) = warped_i;
                        //if(compute_MAD_stdDev_)
                        //    v_AD_intensity[i] = fabs(_grayTrgPyr[warped_i] - _graySrcPyr[i]);
                        ++numVisiblePts;

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            if( fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f)
                            {
                                //Obtain the pixel values that will be used to compute the pixel residual
                                //                        float pixel_src = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                                //                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                validPixelsPhoto_src(i) = 1;
                                float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                                residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                                wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
//                                std::cout << i << " " << validPixels_src(i) << " error2_photo " << error2_photo << " wDepthPhoto " << residualsPhoto_src(i) << " i " << i << " w " << warped_i << " c " << transformed_c_int << " " << theta*pixel_angle_inv << " theta " << theta << " weight_estim " << wEstimPhoto_src(i) << std::endl;
//                                std::cout << " _grayTrgPyr[warped_i] " << _grayTrgPyr[warped_i] << " _graySrcPyr[i] " << _graySrcPyr[i] << std::endl;
//                                mrpt::system::pause();
                            }
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float dist_trg = _depthTrgPyr[warped_i];
                            if(dist_trg > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                if( fabs(_depthSrcGradXPyr[i]) > 0.f || fabs(_depthSrcGradYPyr[i]) > 0.f )
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    validPixelsDepth_src(i) = 1;
                                    stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+dist_trg*dist_trg), 2*stdDevDepth);
                                    //residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                    Eigen::Vector3f xyz_trg = LUT_xyz_target.block(warped_i,0,1,3).transpose();
                                    Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                                    float dist_src = _depthSrcPyr[i];
                                    residualsDepth_src(i) = (((xyz_src .dot (xyz_trg - poseGuess.block(0,3,3,1))) / dist_src) - dist_src) * stdDevError_inv_src(i);                                    wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            std::cout << " BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
            warp_img_src.resize(imgSize, 2);
            const float nCols_1 = nCols-1;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_src_transf.rows(); i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                if( validPixels_src(i) ) //Compute the jacobian only for the valid points
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
                    int transformed_r_int = round(transformed_r);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                    {
                        //visible_pixels_src(i) = 1;
                        ++numVisiblePts;
                        float theta = atan2(xyz(0),xyz(2));
                        float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_src(i) = warped_i;
                        warp_img_src(i,0) = transformed_r;
                        warp_img_src(i,1) = transformed_c;
                        cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                            //if( fabs(_grayTrgGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_)
                            {
                                validPixelsPhoto_src(i) = 1;
                                float intensity = bilinearInterp( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                float diff = intensity - _graySrcPyr[i];
                                residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                                wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += wEstimPhoto_src(i) * residualsPhoto_src(i) * residualsPhoto_src(i);
                                // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;
                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            float dist_trg = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            if(dist_trg > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                                //if( fabs(_depthTrgGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_)
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    validPixelsDepth_src(i) = 1;
                                    stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+dist_trg*dist_trg), 2*stdDevDepth);
                                    //residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                                    Eigen::Vector3f xyz_trg;// = LUT_xyz_target.block(warped_i,0,1,3).transpose();
                                    float theta = (transformed_r - half_width)*pixel_angle;
                                    float phi = transformed_r * pixel_angle + phi_start;
                                    float cos_phi = cos(phi);
                                    xyz_trg(0) = dist_trg * cos_phi * sin(theta);
                                    xyz_trg(1) = dist_trg * sin(phi);
                                    xyz_trg(2) = dist_trg * cos_phi * cos(theta);
                                    // TODO: change the sin / cos operations tp dp interpolation on precomputed vectors
                                    Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                                    float dist_src = _depthSrcPyr[i];
                                    residualsDepth_src(i) = (((xyz_src .dot (xyz_trg - poseGuess.block(0,3,3,1))) / dist_src) - dist_src) * stdDevError_inv_src(i);                                    wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += wEstimDepth_src(i) * residualsDepth_src(i) * residualsDepth_src(i);
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    SSO = (float)numVisiblePts / n_pts;
    error2 = (error2_photo + error2_depth) / numVisiblePts;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " errorDense_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif

    return error2;
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
double RegisterDense::computeJacobian_sphere(const int &pyramidLevel,
                                        const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                        costFuncType method ) //,const bool use_bilinear )
{
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

    // std::cout << " RegisterDense::calcHessGrad_sphere() method " << method << " use_bilinear " << use_bilinear_ << std::endl;
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const size_t n_pts = LUT_xyz_source.rows();

    const float half_width = nCols/2 - 0.5f;
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;

//    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
//    Eigen::MatrixXf jacobiansDepth(imgSize,6);
//    jacobiansPhoto.resize(n_pts,6);
//    jacobiansDepth.resize(n_pts,6);
    jacobiansPhoto = Eigen::MatrixXf::Zero(n_pts,6);
    jacobiansDepth = Eigen::MatrixXf::Zero(n_pts,6);

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    transformPts3D_sse(LUT_xyz_source, poseGuess, xyz_src_transf);

    cout << "n_pts " << n_pts << " / " << imgSize << endl;
    warp_pixels_src = Eigen::VectorXi::Constant(n_pts,-1);
    residualsPhoto_src.resize(n_pts);
    residualsDepth_src.resize(n_pts);
    stdDevError_inv_src.resize(n_pts);
    wEstimPhoto_src.resize(n_pts);
    wEstimDepth_src.resize(n_pts);
    //visible_pixels_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsPhoto_src = Eigen::VectorXi::Zero(n_pts);
    validPixelsDepth_src = Eigen::VectorXi::Zero(n_pts);

    float *_depthTrgPyr = reinterpret_cast<float*>(depthTrgPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);

    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);

    // For ESM
    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    if( !(fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f || fabs(_depthSrcGradXPyr[i]) > 0.f || fabs(_depthSrcGradYPyr[i]) > 0.f) )
                        continue;
                    float theta = atan2(xyz(0),xyz(2));
                    int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_src(i) = warped_i;

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;
                        validPixelsPhoto_src(i) = 1;
                        float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                        residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                        wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator

                        Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        //img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                        //img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];
                        //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                        //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                        //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                        //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        error2_photo += residualsPhoto_src(i) * residualsPhoto_src(i);
                        //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                        //mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        validPixelsDepth_src(i) = 1;
                        float depth = _depthTrgPyr[warped_i];
                        stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                        residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                        wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator

                        Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                        //depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                        //depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                        depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                        depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];
                        //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                        //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                        //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                        //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        error2_depth += residualsDepth_src(i) * residualsDepth_src(i);
                        //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                    }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                    {
                        Eigen::Matrix<float,1,2> img_gradient;
                        //img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                        //img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );
                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        error2_photo += residualsPhoto_src(i) * residualsPhoto_src(i);
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                    {
                        Eigen::Matrix<float,1,2> depth_gradient;
                        //depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                        //depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );
                        depth_gradient(0,0) = _depthTrgGradXPyr[validPixels_src(i)];
                        depth_gradient(0,1) = _depthTrgGradYPyr[validPixels_src(i)];

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        error2_depth += residualsDepth_src(i) * residualsDepth_src(i);
                        // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                {
                    //Compute the 3D coordinates of the pij of the source frame
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz_src_transf.block(i,0,1,3) << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i << " transformed_r_int " << transformed_r_int << " " << nRows << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        if( !(fabs(_graySrcGradXPyr[i]) > 0.f || fabs(_graySrcGradYPyr[i]) > 0.f || fabs(_depthSrcGradXPyr[i]) > 0.f || fabs(_depthSrcGradYPyr[i]) > 0.f) )
                            continue;
                        float theta = atan2(xyz(0),xyz(2));
                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_src(i) = warped_i;

                        Eigen::Matrix<float,2,6> jacobianWarpRt;
                        computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;
                            validPixelsPhoto_src(i) = 1;
                            float diff = _grayTrgPyr[warped_i] - _graySrcPyr[i];
                            residualsPhoto_src(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_src(i) = weightMEstimator(residualsPhoto_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator

                            Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                            //img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                            //img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                            img_gradient(0,0) = _graySrcGradXPyr[i];
                            img_gradient(0,1) = _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                            //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                            //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            error2_photo += residualsPhoto_src(i) * residualsPhoto_src(i);
                            //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                            //mrpt::system::pause();
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            validPixelsDepth_src(i) = 1;
                            float depth = _depthTrgPyr[warped_i];
                            stdDevError_inv_src(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                            residualsDepth_src(i) = (depth - dist) * stdDevError_inv_src(i);
                            wEstimDepth_src(i) = weightMEstimator(residualsDepth_src(i)); // Apply M-estimator weighting // The weight computed by an M-estimator

                            Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            //depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                            //depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                            depth_gradient(0,0) = _depthSrcGradXPyr[i];
                            depth_gradient(0,1) = _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                            //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                            //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            error2_depth += residualsDepth_src(i) * residualsDepth_src(i);
                            //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        }
                    }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            Eigen::Matrix<float,1,2> img_gradient;
                            //img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                            //img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );
                            img_gradient(0,0) = _graySrcGradXPyr[i];
                            img_gradient(0,1) = _graySrcGradYPyr[i];

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            error2_photo += residualsPhoto_src(i) * residualsPhoto_src(i);
                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                            // mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            Eigen::Matrix<float,1,2> depth_gradient;
                            //depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                            //depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );
                            depth_gradient(0,0) = _depthTrgGradXPyr[i];
                            depth_gradient(0,1) = _depthTrgGradYPyr[i];

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            error2_depth += residualsDepth_src(i) * residualsDepth_src(i);
                            // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                        }
                }
            }
        }
    }

    error2 = (error2_photo + error2_depth) / numVisiblePts;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " computeJacobian_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif

    return error2;
}

/*! Compute the averaged squared error of the salient points. */
double RegisterDense::computeErrorHessGrad_salient( std::vector<size_t> & salient_pixels_photo,
                                                    std::vector<size_t> & salient_pixels_depth,
                                                    costFuncType method ) //,const bool use_bilinear )
{
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif
    //const size_t n_pts = LUT_xyz_source.rows();
//    if(use_salient_pixels_)
//    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        {
            for(size_t i=0; i < salient_pixels_photo.size(); i++)
            {
                size_t salient_pix = salient_pixels_photo[i];
                error2_photo += wEstimPhoto_src(salient_pix) * residualsPhoto_src(salient_pix) * residualsPhoto_src(salient_pix);
                hessian += jacobiansPhoto.block(salient_pix,0,1,6).transpose() * jacobiansPhoto.block(salient_pix,0,1,6);
                gradient += jacobiansPhoto.block(salient_pix,0,1,6).transpose() * residualsPhoto_src(salient_pix);
            }
            error2_photo /= salient_pixels_photo.size();
        }
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        {
            for(size_t i=0; i < salient_pixels_depth.size(); i++)
            {
                size_t salient_pix = salient_pixels_depth[i];
                error2_depth += wEstimDepth_src(salient_pix) * residualsDepth_src(salient_pix) * residualsDepth_src(salient_pix);
                hessian += jacobiansDepth.block(salient_pix,0,1,6).transpose() * jacobiansDepth.block(salient_pix,0,1,6);
                gradient += jacobiansDepth.block(salient_pix,0,1,6).transpose() * residualsDepth_src(salient_pix);
            }
            error2_depth /= salient_pixels_depth.size();
        }
//    }
//    else
//    {

//    }

    error2 = error2_photo + error2_depth;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " computeErrorHessGrad_salient took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

    return error2;
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::calcHessGrad_sphere(const int &pyramidLevel,
                                        const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                        costFuncType method ) //,const bool use_bilinear )
{
    // std::cout << " RegisterDense::calcHessGrad_sphere() method " << method << " use_bilinear " << use_bilinear_ << std::endl;
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const size_t n_pts = LUT_xyz_source.rows();

    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;

//    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
//    Eigen::MatrixXf jacobiansDepth(imgSize,6);
//    jacobiansPhoto.resize(n_pts,6);
//    jacobiansDepth.resize(n_pts,6);
    jacobiansPhoto = Eigen::MatrixXf::Zero(n_pts,6);
    jacobiansDepth = Eigen::MatrixXf::Zero(n_pts,6);

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);

    // For ESM
    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                        Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        //img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                        //img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];
                        //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                        //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                        //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                        //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                        //mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                        //depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                        //depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                        depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                        depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];
                        //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                        //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                        //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                        //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        //std::cout << "jacobianDepth_Rt " << weight_estim_sqrt * stdDevError_inv_src(i) * depth_gradient*jacobianWarpRt << std::endl;
                        //std::cout << "jacobianDepth_t " << weight_estim_sqrt * stdDevError_inv_src(i) * jacobian16_depthT << std::endl;
                        //mrpt::system::pause();
                    }
                    //Assign the pixel residual and jacobian to its corresponding row
                    //#if ENABLE_OPENMP
                    //#pragma omp critical
                    //#endif
                    //                            {
                    //                                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    //                                {
                    //                                    // Photometric component
                    //                                    hessian += jacobianPhoto.transpose()*jacobianPhoto;
                    //                                    gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
                    //                                }
                    //                                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    //                                    if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                    //                                    {
                    //                                        // Depth component (Plane ICL like)
                    //                                        hessian += jacobianDepth.transpose()*jacobianDepth;
                    //                                        gradient += jacobianDepth.transpose()*weightedErrorDepth;
                    //                                    }
                    //                            }
                    //                        }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        Eigen::Matrix<float,1,2> img_gradient;
                        img_gradient(0,0) = bilinearInterp( graySrcGradXPyr[pyramidLevel], warped_pixel );
                        img_gradient(0,1) = bilinearInterp( graySrcGradYPyr[pyramidLevel], warped_pixel );
//                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
//                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient;
                        depth_gradient(0,0) = bilinearInterp_depth( depthSrcGradXPyr[pyramidLevel], warped_pixel );
                        depth_gradient(0,1) = bilinearInterp_depth( depthSrcGradYPyr[pyramidLevel], warped_pixel );
//                        depth_gradient(0,0) = _depthTrgGradXPyr[validPixels_src(i)];
//                        depth_gradient(0,1) = _depthTrgGradYPyr[validPixels_src(i)];

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                            Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                            //img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                            //img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                            img_gradient(0,0) = _graySrcGradXPyr[i];
                            img_gradient(0,1) = _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                            //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                            //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                            //mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            //depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                            //depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                            depth_gradient(0,0) = _depthSrcGradXPyr[i];
                            depth_gradient(0,1) = _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                            //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                            //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        }
                    //Assign the pixel residual and jacobian to its corresponding row
                    //#if ENABLE_OPENMP
                    //#pragma omp critical
                    //#endif
                    //                            {
                    //                                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    //                                {
                    //                                    // Photometric component
                    //                                    hessian += jacobianPhoto.transpose()*jacobianPhoto;
                    //                                    gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
                    //                                }
                    //                                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    //                                    if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                    //                                    {
                    //                                        // Depth component (Plane ICL like)
                    //                                        hessian += jacobianDepth.transpose()*jacobianDepth;
                    //                                        gradient += jacobianDepth.transpose()*weightedErrorDepth;
                    //                                    }
                    //                            }
                    //                        }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            Eigen::Matrix<float,1,2> img_gradient;
                            img_gradient(0,0) = bilinearInterp( graySrcGradXPyr[pyramidLevel], warped_pixel );
                            img_gradient(0,1) = bilinearInterp( graySrcGradYPyr[pyramidLevel], warped_pixel );
//                            img_gradient(0,0) = _graySrcGradXPyr[i];
//                            img_gradient(0,1) = _graySrcGradYPyr[i];

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                            // mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient;
                            depth_gradient(0,0) = bilinearInterp_depth( depthSrcGradXPyr[pyramidLevel], warped_pixel );
                            depth_gradient(0,1) = bilinearInterp_depth( depthSrcGradYPyr[pyramidLevel], warped_pixel );
//                            depth_gradient(0,0) = _depthTrgGradXPyr[i];
//                            depth_gradient(0,1) = _depthTrgGradYPyr[i];

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                        }
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
    //std::cout << "hessian \n" << hessian << std::endl;
    //std::cout << "gradient \n" << gradient.transpose() << std::endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGrad_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

void RegisterDense::calcHessGrad_warp_sphere(const int &pyramidLevel,
                                            const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                            costFuncType method ) //,const bool use_bilinear )
{
    // std::cout << " RegisterDense::calcHessGrad_sphere() method " << method << " use_bilinear " << use_bilinear_ << std::endl;
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const size_t n_pts = LUT_xyz_source.rows();

    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;

//    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
//    Eigen::MatrixXf jacobiansDepth(imgSize,6);
//    jacobiansPhoto.resize(n_pts,6);
//    jacobiansDepth.resize(n_pts,6);
    jacobiansPhoto = Eigen::MatrixXf::Zero(n_pts,6);
    jacobiansDepth = Eigen::MatrixXf::Zero(n_pts,6);

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    float *_grayTrgGradXPyr = reinterpret_cast<float*>(warped_gray_gradX.data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(warped_gray_gradY.data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(warped_depth_gradX.data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(warped_depth_gradY.data);

    // For ESM
    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_src(i) )
                    {
                        //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                        Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                        img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                        //img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        //img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];
                        //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                        //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                        //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                        //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                        //mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                        depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                        depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                        //depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                        //depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];
                        //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                        //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                        //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                        //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        //std::cout << "jacobianDepth_Rt " << weight_estim_sqrt * stdDevError_inv_src(i) * depth_gradient*jacobianWarpRt << std::endl;
                        //std::cout << "jacobianDepth_t " << weight_estim_sqrt * stdDevError_inv_src(i) * jacobian16_depthT << std::endl;
                        //mrpt::system::pause();
                    }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        Eigen::Matrix<float,1,2> img_gradient;
                        img_gradient(0,0) = bilinearInterp( warped_gray_gradX, warped_pixel );
                        img_gradient(0,1) = bilinearInterp( warped_gray_gradY, warped_pixel );
//                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
//                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient;
                        depth_gradient(0,0) = bilinearInterp_depth( warped_depth_gradX, warped_pixel );
                        depth_gradient(0,1) = bilinearInterp_depth( warped_depth_gradY, warped_pixel );
//                        depth_gradient(0,0) = _depthTrgGradXPyr[validPixels_src(i)];
//                        depth_gradient(0,1) = _depthTrgGradYPyr[validPixels_src(i)];

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                            Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                            img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                            img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                            //img_gradient(0,0) = _graySrcGradXPyr[i];
                            //img_gradient(0,1) = _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                            //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                            //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                            //mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                            depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                            //depth_gradient(0,0) = _depthSrcGradXPyr[i];
                            //depth_gradient(0,1) = _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                            //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                            //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            Eigen::Matrix<float,1,2> img_gradient;
                            img_gradient(0,0) = bilinearInterp( warped_gray_gradX, warped_pixel );
                            img_gradient(0,1) = bilinearInterp( warped_gray_gradY, warped_pixel );
//                            img_gradient(0,0) = _graySrcGradXPyr[i];
//                            img_gradient(0,1) = _graySrcGradYPyr[i];

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                            // mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            Eigen::Matrix<float,1,2> depth_gradient;
                            depth_gradient(0,0) = bilinearInterp_depth( warped_depth_gradX, warped_pixel );
                            depth_gradient(0,1) = bilinearInterp_depth( warped_depth_gradY, warped_pixel );
//                            depth_gradient(0,0) = _depthTrgGradXPyr[i];
//                            depth_gradient(0,1) = _depthTrgGradYPyr[i];

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                        }
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
    //std::cout << "hessian \n" << hessian << std::endl;
    //std::cout << "gradient \n" << gradient.transpose() << std::endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGradWarp_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::calcHessGrad_side_sphere(const int &pyramidLevel,
                                         const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                         costFuncType method,
                                         const int side) // side is an aproximation parameter, 0 -> optimization starts at the identity and gradients are computed at the source; 1 at the target
{
    // std::cout << " RegisterDense::calcHessGrad_sphere() method " << method << " use_bilinear " << use_bilinear_ << std::endl;
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const size_t n_pts = LUT_xyz_source.rows();

    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;

//    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
//    Eigen::MatrixXf jacobiansDepth(imgSize,6);
//    jacobiansPhoto.resize(n_pts,6);
//    jacobiansDepth.resize(n_pts,6);
    jacobiansPhoto = Eigen::MatrixXf::Zero(n_pts,6);
    jacobiansDepth = Eigen::MatrixXf::Zero(n_pts,6);

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);

    // For ESM
    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                        Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        if(side)
                        {
                            img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                            img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                        }
                        else
                        {
                            img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                            img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];
                        }

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                        //mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                        if(side)
                        {
                            depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                            depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                        }
                        else
                        {
                            depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                            depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];
                        }
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                    }
                    //Assign the pixel residual and jacobian to its corresponding row
                    //#if ENABLE_OPENMP
                    //#pragma omp critical
                    //#endif
                    //                            {
                    //                                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    //                                {
                    //                                    // Photometric component
                    //                                    hessian += jacobianPhoto.transpose()*jacobianPhoto;
                    //                                    gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
                    //                                }
                    //                                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    //                                    if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                    //                                    {
                    //                                        // Depth component (Plane ICL like)
                    //                                        hessian += jacobianDepth.transpose()*jacobianDepth;
                    //                                        gradient += jacobianDepth.transpose()*weightedErrorDepth;
                    //                                    }
                    //                            }
                    //                        }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        Eigen::Matrix<float,1,2> img_gradient;
                        if(side)
                        {
                            img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                            img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );
                        }
                        else
                        {
                            img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                            img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];
                        }

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient;
                        if(side)
                        {
                            depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                            depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );
                        }
                        else
                        {
                            depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                            depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];
                        }

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                            Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                            if(side)
                            {
                                img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                                img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                            }
                            else
                            {
                                img_gradient(0,0) = _graySrcGradXPyr[i];
                                img_gradient(0,1) = _graySrcGradYPyr[i];
                            }

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                            //mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            if(side)
                            {
                                depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                                depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                            }
                            else
                            {
                                depth_gradient(0,0) = _depthSrcGradXPyr[i];
                                depth_gradient(0,1) = _depthSrcGradYPyr[i];
                            }
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            Eigen::Matrix<float,1,2> img_gradient;
                            if(side)
                            {
                                img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                                img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );
                            }
                            else
                            {
                                img_gradient(0,0) = _graySrcGradXPyr[i];
                                img_gradient(0,1) = _graySrcGradYPyr[i];
                            }

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                            // mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient;
                            if(side)
                            {
                                depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                                depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );
                            }
                            else
                            {
                                depth_gradient(0,0) = _depthSrcGradXPyr[i];
                                depth_gradient(0,1) = _depthSrcGradYPyr[i];
                            }

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                        }
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
    //std::cout << "hessian \n" << hessian << std::endl;
    //std::cout << "gradient \n" << gradient.transpose() << std::endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGrad_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::calcHessGradRot_sphere(const int &pyramidLevel,
                                        const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                        costFuncType method ) //,const bool use_bilinear )
{
    // std::cout << " RegisterDense::calcHessGrad_sphere() method " << method << " use_bilinear " << use_bilinear_ << std::endl;
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const size_t n_pts = LUT_xyz_source.rows();

    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;

    hessian_rot = Eigen::Matrix<float,3,3>::Zero();
    gradient_rot = Eigen::Matrix<float,3,1>::Zero();
    jacobiansPhoto = Eigen::MatrixXf::Zero(n_pts,3);
    jacobiansDepth = Eigen::MatrixXf::Zero(n_pts,3);

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);

    // For ESM
    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                        Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        //img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                        //img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];
                        //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                        //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                        //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                        //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,3) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt.block(0,3,2,3);
                        //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        hessian_rot += jacobiansPhoto.block(i,0,1,3).transpose() * jacobiansPhoto.block(i,0,1,3);
                        gradient_rot += jacobiansPhoto.block(i,0,1,3).transpose() * residualsPhoto_src(i);

                        //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                        //mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                        //depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                        //depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                        depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                        depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];
                        //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                        //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                        //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                        //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,3) = (weight_estim_sqrt * stdDevError_inv_src(i) * depth_gradient) * jacobianWarpRt.block(0,3,2,3);
                        //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        hessian_rot += jacobiansDepth.block(i,0,1,3).transpose() * jacobiansDepth.block(i,0,1,3);
                        gradient_rot += jacobiansDepth.block(i,0,1,3).transpose() * residualsDepth_src(i);
                        //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                    }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        Eigen::Matrix<float,1,2> img_gradient;
                        //img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                        //img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );
                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,3) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt.block(0,3,2,3);
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        hessian_rot += jacobiansPhoto.block(i,0,1,3).transpose() * jacobiansPhoto.block(i,0,1,3);
                        gradient_rot += jacobiansPhoto.block(i,0,1,3).transpose() * residualsPhoto_src(i);
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient;
                        //depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                        //depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );
                        depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                        depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,3) = (weight_estim_sqrt * stdDevError_inv_src(i) * depth_gradient) * jacobianWarpRt.block(0,3,2,3);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        hessian_rot += jacobiansDepth.block(i,0,1,3).transpose() * jacobiansDepth.block(i,0,1,3);
                        gradient_rot += jacobiansDepth.block(i,0,1,3).transpose() * residualsDepth_src(i);
                        // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                    //Projected 3D point to the S2 sphere
                    float dist = xyz.norm();

                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                            Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                            //img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                            //img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];
                            img_gradient(0,0) = _graySrcGradXPyr[i];
                            img_gradient(0,1) = _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]; // ESM
                            //                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i];
                            //                    img_gradient(0,0) = 0.5f*(_grayTrgGradXPyr[warp_pixels_src(i)] + _graySrcGradXPyr[i]); // ESM
                            //                    img_gradient(0,1) = 0.5f*(_grayTrgGradYPyr[warp_pixels_src(i)] + _graySrcGradYPyr[i]);
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,3) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt.block(0,3,2,3);
                            //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            hessian_rot += jacobiansPhoto.block(i,0,1,3).transpose() * jacobiansPhoto.block(i,0,1,3);
                            gradient_rot += jacobiansPhoto.block(i,0,1,3).transpose() * residualsPhoto_src(i);
                            //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                            //mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            //depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                            //depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                            depth_gradient(0,0) = _depthSrcGradXPyr[i];
                            depth_gradient(0,1) = _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]; // ESM
                            //                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i];
                            //                    depth_gradient(0,0) = 0.5f*(_depthTrgGradXPyr[warp_pixels_src(i)] + _depthSrcGradXPyr[i]); // ESM
                            //                    depth_gradient(0,1) = 0.5f*(_depthTrgGradYPyr[warp_pixels_src(i)] + _depthSrcGradYPyr[i]);
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,3) = (weight_estim_sqrt * stdDevError_inv_src(i) * depth_gradient) * jacobianWarpRt.block(0,3,2,3);
                            //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            hessian_rot += jacobiansDepth.block(i,0,1,3).transpose() * jacobiansDepth.block(i,0,1,3);
                            gradient_rot += jacobiansDepth.block(i,0,1,3).transpose() * residualsDepth_src(i);
                            //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    //Compute the pixel jacobian
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            Eigen::Matrix<float,1,2> img_gradient;
                            //img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                            //img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );
                            img_gradient(0,0) = _graySrcGradXPyr[i];
                            img_gradient(0,1) = _graySrcGradYPyr[i];

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,3) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt.block(0,3,2,3);
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            hessian_rot += jacobiansPhoto.block(i,0,1,3).transpose() * jacobiansPhoto.block(i,0,1,3);
                            gradient_rot += jacobiansPhoto.block(i,0,1,3).transpose() * residualsPhoto_src(i);
                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                            // mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient;
                            //depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                            //depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );
                            depth_gradient(0,0) = _depthSrcGradXPyr[i];
                            depth_gradient(0,1) = _depthSrcGradYPyr[i];

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,3) = (weight_estim_sqrt * stdDevError_inv_src(i) * depth_gradient) * jacobianWarpRt.block(0,3,2,3);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            hessian_rot += jacobiansDepth.block(i,0,1,3).transpose() * jacobiansDepth.block(i,0,1,3);
                            gradient_rot += jacobiansDepth.block(i,0,1,3).transpose() * residualsDepth_src(i);
                            // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                        }
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
    //std::cout << "hessian \n" << hessian << std::endl;
    //std::cout << "gradient \n" << gradient.transpose() << std::endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGrad_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
void RegisterDense::calcHessGradIC_sphere(const int &pyramidLevel,
                                        const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                        costFuncType method ) //,const bool use_bilinear )
{
    // WARNING: The negative Jacobians are computed, thus it is not necesary to invert the construction of the SE3 pose update

    // std::cout << " RegisterDense::calcHessGrad_sphere() method " << method << " use_bilinear " << use_bilinear_ << std::endl;
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const size_t n_pts = LUT_xyz_source.rows();

    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;

//    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
//    Eigen::MatrixXf jacobiansDepth(imgSize,6);
//    jacobiansPhoto.resize(n_pts,6);
//    jacobiansDepth.resize(n_pts,6);
    jacobiansPhoto = Eigen::MatrixXf::Zero(n_pts,6);
    jacobiansDepth = Eigen::MatrixXf::Zero(n_pts,6);

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    float *_depthSrcPyr = reinterpret_cast<float*>(depthSrcPyr[pyramidLevel].data);

    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);

    // For ESM
    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                    float dist_src = _depthSrcPyr[validPixels_src(i)];
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz_src, dist_src, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                        Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                        //mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_src(i) )
                    {
                        Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                        depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                        depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist_src) * LUT_xyz_source.block(i,0,1,3);
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        //mrpt::system::pause();
                    }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                    float dist_src = _depthSrcPyr[validPixels_src(i)];
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz_src, dist_src, pixel_angle_inv, jacobianWarpRt);

                    //cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_src(i));

                        Eigen::Matrix<float,1,2> img_gradient;
                        img_gradient(0,0) = _graySrcGradXPyr[validPixels_src(i)];
                        img_gradient(0,1) = _graySrcGradYPyr[validPixels_src(i)];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_src(i) *= weight_estim_sqrt;
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                    {
                        Eigen::Matrix<float,1,2> depth_gradient;
                        depth_gradient(0,0) = _depthSrcGradXPyr[validPixels_src(i)];
                        depth_gradient(0,1) = _depthSrcGradYPyr[validPixels_src(i)];

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist_src) * LUT_xyz_source.block(i,0,1,3);
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //Compute the 3D coordinates of the pij of the source frame
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( validPixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                    float dist_src = _depthSrcPyr[i];
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz_src, dist_src, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                            Eigen::Matrix<float,1,2> img_gradient; // This is an approximation of the gradient of the warped image
                            img_gradient(0,0) = _graySrcGradXPyr[i];
                            img_gradient(0,1) = _graySrcGradYPyr[i];
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            //jacobiansPhoto.block(i,0,1,6).noalias() = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            //std::cout << img_gradient(0,0) << " " << img_gradient(0,1) << " salient jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " \t weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                            //mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            //if(visualizeIterations)
                            //    warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient; // This is an approximation of the gradient of the warped image
                            depth_gradient(0,0) = _depthSrcGradXPyr[i];
                            depth_gradient(0,1) = _depthSrcGradYPyr[i];
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist_src) * LUT_xyz_source.block(i,0,1,3);
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            //jacobiansDepth.block(i,0,1,6).noalias() = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            //std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << " \t residualsDepth_src " << residualsDepth_src(i) << std::endl;
                        }
                }
            }
        }
        else
        {
            std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( warp_pixels_src(i) != -1 ) //Compute the jacobian only for the visible points
                //if( visible_pixels_src(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
                {
                    Eigen::Vector3f xyz_src = LUT_xyz_source.block(i,0,1,3).transpose();
                    float dist_src = _depthSrcPyr[i];
                    Eigen::Matrix<float,2,6> jacobianWarpRt;
                    computeJacobian26_wT_sphere(xyz_src, dist_src, pixel_angle_inv, jacobianWarpRt);

                    cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsPhoto_src(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                            Eigen::Matrix<float,1,2> img_gradient;
                            img_gradient(0,0) = _graySrcGradXPyr[i];
                            img_gradient(0,1) = _graySrcGradYPyr[i];

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_src(i) *= weight_estim_sqrt;
                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                            // mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_src(i) )
                        {
                            //if(visualizeIterations)
                            //    warped_depth.at<float>(warp_pixels_src(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient;
                            depth_gradient(0,0) = _depthSrcGradXPyr[i];
                            depth_gradient(0,1) = _depthSrcGradYPyr[i];

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                            jacobian16_depthT.block(0,0,1,3) = (1 / dist_src) * LUT_xyz_source.block(i,0,1,3);
                            float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                            residualsDepth_src(i) *= weight_estim_sqrt;
                            // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                        }
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
    //std::cout << "hessian \n" << hessian << std::endl;
    //std::cout << "gradient \n" << gradient.transpose() << std::endl;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " IC calcHessGradIC_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Compute the residuals of the target image projected onto the source one. */
double RegisterDense::errorDenseInv_sphere ( const int &pyramidLevel,
                                              const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                              costFuncType method )//,const bool use_bilinear )
{
    double error2 = 0.0;
    double error2_photo = 0.0;
    double error2_depth = 0.0;
    size_t numVisiblePts = 0;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    const float half_width = nCols/2 - 0.5f;

    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5*nRows-0.5)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    const Eigen::Matrix4f poseGuess_inv = poseGuess.inverse();
    //transformPts3D(LUT_xyz_target, poseGuess_inv, xyz_trg_transf);
    transformPts3D_sse(LUT_xyz_target, poseGuess_inv, xyz_trg_transf);

    const size_t n_pts = LUT_xyz_target.rows();

    warp_pixels_trg.resize(n_pts);
    warp_img_trg.resize(n_pts,2);
    residualsPhoto_trg = Eigen::VectorXf::Zero(n_pts);
    residualsDepth_trg = Eigen::VectorXf::Zero(n_pts);
    stdDevError_inv_trg = Eigen::VectorXf::Zero(n_pts);
    wEstimPhoto_trg = Eigen::VectorXf::Zero(n_pts);
    wEstimDepth_trg = Eigen::VectorXf::Zero(n_pts);
    validPixelsPhoto_trg = Eigen::VectorXi::Zero(n_pts);
    validPixelsDepth_trg = Eigen::VectorXi::Zero(n_pts);

    // Container to compute the MAD, which is used to update the intensity (or brightness) standard deviation
    //std::vector<float> v_AD_intensity_(n_pts);

    float *_depthSrcPyr = reinterpret_cast<float*>(depthSrcPyr[pyramidLevel].data);
    float *_grayTrgPyr = reinterpret_cast<float*>(grayTrgPyr[pyramidLevel].data);
    float *_graySrcPyr = reinterpret_cast<float*>(graySrcPyr[pyramidLevel].data);

//    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
//    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
//    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
//    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_trg_transf.rows(); i++)
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << LUT_xyz_target.block(i,0,1,3) << " transformed " << xyz_trg_transf.block(i,0,1,3) << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;
                float phi = asin(xyz(1)*dist_inv);
                int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                {
                    ++numVisiblePts;
                    float theta = atan2(xyz(0),xyz(2));
                    int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                    size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                    warp_pixels_trg(i) = warped_i;
                    //std::cout << "Pixel transform_ " << validPixels_trg(i)/nCols << " " << validPixels_trg(i)%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " numVisiblePts " << numVisiblePts << endl;
                    // assert(transformed_c_int >= 0 && transformed_c_int < nCols);
                    //                    if(method == PHOTO_CONSISTENCY && method == PHOTO_DEPTH)
                    //                    {

                    //                    }
                    //                    else
                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                        //if( fabs(_graySrcGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[warped_i]) > thres_saliency_gray_)
                        //if( fabs(_grayTrgGradXPyr[validPixels_trg(i)]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[validPixels_trg(i)]) > thres_saliency_gray_)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            //                        float pixel_trg = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                            //                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            validPixelsPhoto_trg(i) = 1;
                            float diff = _graySrcPyr[warped_i] - _grayTrgPyr[validPixels_trg(i)];
                            residualsPhoto_trg(i) = diff * stdDevPhoto_inv;
                            wEstimPhoto_trg(i) = weightMEstimator(residualsPhoto_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            error2_photo += wEstimPhoto_trg(i) * residualsPhoto_trg(i) * residualsPhoto_trg(i);
                            //std::cout << i << " " << validPixels_trg(i) << " warped_i " << warped_i << " error2_photo " << error2_photo << " residualsPhoto " << residualsPhoto_trg(i) << " weight_estim " << wEstimPhoto_trg(i) << std::endl;
                            //mrpt::system::pause();
                            //                        _validPixelsPhoto_trg(n_ptsPhoto) = i;
                            //                        _residualsPhoto_trg(n_ptsPhoto) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsPhoto]) * stdDevPhoto_inv;
                            //                        _wEstimPhoto_trg(n_ptsPhoto) = weightMEstimator(_residualsPhoto_trg(n_ptsPhoto)); // Apply M-estimator weighting // The weight computed by an M-estimator
                            //                        error2 += _wEstimPhoto_trg(n_ptsPhoto) * _residualsPhoto_trg(n_ptsPhoto) * _residualsPhoto_trg(n_ptsPhoto);

                            //v_AD_intensity[i] = fabs(diff);
                            //++n_ptsPhoto;
                        }
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    {
                        float depth = _depthSrcPyr[warped_i];
                        if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                        {
                            //if( fabs(_depthSrcGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[warped_i]) > thres_saliency_depth_)
                            //if( fabs(_depthTrgGradXPyr[validPixels_trg(i)]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[validPixels_trg(i)]) > thres_saliency_depth_)
                            {
                                //Obtain the depth values that will be used to the compute the depth residual
                                validPixelsDepth_trg(i) = 1;
                                stdDevError_inv_trg(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                residualsDepth_trg(i) = (depth - dist) * stdDevError_inv_trg(i);
                                wEstimDepth_trg(i) = weightMEstimator(residualsDepth_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_depth += wEstimDepth_trg(i) * residualsDepth_trg(i) * residualsDepth_trg(i);
                                // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;

                                //                            _validPixelsDepth_trg(n_ptsDepth) = i;
                                //                            _stdDevError_inv_trg(n_ptsDepth) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                //                            _residualsDepth_trg(n_ptsDepth) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsDepth]) * stdDevError_inv_trg(n_ptsDepth);
                                //                            _wEstimDepth_trg(n_ptsDepth) = weightMEstimator(_residualsDepth_trg(n_ptsDepth)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                //                            error2 += _wEstimDepth_trg(n_ptsDepth) * _residualsDepth_trg(n_ptsDepth) * _residualsDepth_trg(n_ptsDepth);
                                //++n_ptsDepth;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            std::cout << "inverse BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
            const float nCols_1 = nCols-1;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_trg_transf.rows(); i++)
            {
                if( validPixels_trg(i) ) //Compute the jacobian only for the valid points
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
                    int transformed_r_int = round(transformed_r);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        float theta = atan2(xyz(0),xyz(2));
                        float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_trg(i) = warped_i;
                        warp_img_trg(i,0) = transformed_r;
                        warp_img_trg(i,1) = transformed_c;
                        cv::Point2f warped_pixel = cv::Point2f(warp_img_trg(i,0), warp_img_trg(i,1));
                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                            //if( fabs(_graySrcGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_grayTrgGradXPyr[validPixels_trg(i)]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[validPixels_trg(i)]) > thres_saliency_gray_)
                            {
                                validPixelsPhoto_trg(i) = 1;
                                float intensity = bilinearInterp( graySrcPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                float diff = intensity - _graySrcPyr[validPixels_trg(i)];
                                residualsPhoto_trg(i) = diff * stdDevPhoto_inv;
                                wEstimPhoto_trg(i) = weightMEstimator(residualsPhoto_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += wEstimPhoto_trg(i) * residualsPhoto_trg(i) * residualsPhoto_trg(i);
                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            float depth = bilinearInterp_depth( depthSrcPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                                //if( fabs(_depthSrcGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthTrgGradXPyr[validPixels_trg(i)]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[validPixels_trg(i)]) > thres_saliency_depth_)
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    validPixelsDepth_trg(i) = 1;
                                    stdDevError_inv_trg(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    residualsDepth_trg(i) = (depth - dist) * stdDevError_inv_trg(i);
                                    wEstimDepth_trg(i) = weightMEstimator(residualsDepth_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += wEstimDepth_trg(i) * residualsDepth_trg(i) * residualsDepth_trg(i);
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_trg_transf.rows(); i++)
            {
                if( validPixels_trg(i) ) //Compute the jacobian only for the valid points
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    int transformed_r_int = int(round((phi-phi_start)*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r_int>=0 && transformed_r_int < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        float theta = atan2(xyz(0),xyz(2));
                        int transformed_c_int = int(round(half_width + theta*pixel_angle_inv)) % nCols; //assert(transformed_c_int<nCols); //assert(transformed_c_int<nCols);
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_trg(i) = warped_i;
                        //std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);
                        //                    if(method == PHOTO_CONSISTENCY && method == PHOTO_DEPTH)
                        //                    {

                        //                    }
                        //                    else
                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            //if( fabs(_graySrcGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_grayTrgGradXPyr[i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[i]) > thres_saliency_gray_)
                            if( fabs(_grayTrgGradXPyr[i]) > 0.f || fabs(_grayTrgGradYPyr[i]) > 0.f)
                            {
                                //Obtain the pixel values that will be used to compute the pixel residual
                                //                        float pixel_trg = graySrcPyr[pyramidLevel].at<float>(i); // Intensity value of the pixel(r,c) of source frame
                                //                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                validPixelsPhoto_trg(i) = 1;
                                float diff = _graySrcPyr[warped_i] - _grayTrgPyr[i];
                                residualsPhoto_trg(i) = diff * stdDevPhoto_inv;
                                wEstimPhoto_trg(i) = weightMEstimator(residualsPhoto_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += wEstimPhoto_trg(i) * residualsPhoto_trg(i) * residualsPhoto_trg(i);

                                //                        _validPixelsPhoto_trg(n_ptsPhoto) = i;
                                //                        _residualsPhoto_trg(n_ptsPhoto) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsPhoto]) * stdDevPhoto_inv;
                                //                        _wEstimPhoto_trg(n_ptsPhoto) = weightMEstimator(_residualsPhoto_trg(n_ptsPhoto)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                //                        error2 += _wEstimPhoto_trg(n_ptsPhoto) * _residualsPhoto_trg(n_ptsPhoto) * _residualsPhoto_trg(n_ptsPhoto);

                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            float depth = _depthSrcPyr[warped_i];
                            if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                //if( fabs(_depthSrcGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthTrgGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[i]) > thres_saliency_depth_)
                                if( fabs(_depthTrgGradXPyr[i]) > 0.f || fabs(_depthTrgGradYPyr[i]) > 0.f)
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    validPixelsDepth_trg(i) = 1;
                                    stdDevError_inv_trg(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    residualsDepth_trg(i) = (depth - dist) * stdDevError_inv_trg(i);
                                    wEstimDepth_trg(i) = weightMEstimator(residualsDepth_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += wEstimDepth_trg(i) * residualsDepth_trg(i) * residualsDepth_trg(i);
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;

                                    //                            _validPixelsDepth_trg(n_ptsDepth) = i;
                                    //                            _stdDevError_inv_trg(n_ptsDepth) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    //                            _residualsDepth_trg(n_ptsDepth) = (_grayTrgPyr[warped_i] - _graySrcPyr[n_ptsDepth]) * stdDevError_inv_trg(n_ptsDepth);
                                    //                            _wEstimDepth_trg(n_ptsDepth) = weightMEstimator(_residualsDepth_trg(n_ptsDepth)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    //                            error2 += _wEstimDepth_trg(n_ptsDepth) * _residualsDepth_trg(n_ptsDepth) * _residualsDepth_trg(n_ptsDepth);
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            std::cout << "inverse BILINEAR TRANSF -> SUBPIXEL TRANSFORMATION \n " << std::endl;
            const float nCols_1 = nCols-1;
#if ENABLE_OPENMP
#pragma omp parallel for reduction (+:error2_photo,error2_depth,numVisiblePts)//,n_ptsPhoto,n_ptsDepth) // error2, n_ptsPhoto, n_ptsDepth
#endif
            for(size_t i=0; i < xyz_trg_transf.rows(); i++)
            {
                if( validPixels_trg(i) ) //Compute the jacobian only for the valid points
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << xyz.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;
                    float phi = asin(xyz(1)*dist_inv);
                    float transformed_r = (phi-phi_start)*pixel_angle_inv;
                    int transformed_r_int = round(transformed_r);
                    //int transformed_r_int = half_height + int(round(phi*pixel_angle_inv));
                    // std::cout << "Pixel transform " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << std::endl;
                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( transformed_r>=0 && transformed_r < nRows) // && transformed_c_int < nCols )
                    {
                        ++numVisiblePts;
                        float theta = atan2(xyz(0),xyz(2));
                        float transformed_c = half_width + theta*pixel_angle_inv; assert(transformed_c <= nCols_1); //transformed_c -= half_width;
                        int transformed_c_int = int(round(transformed_c)); assert(transformed_c_int<nCols);// % half_width;
                        size_t warped_i = transformed_r_int * nCols + transformed_c_int;
                        warp_pixels_trg(i) = warped_i;
                        warp_img_trg(i,0) = transformed_r;
                        warp_img_trg(i,1) = transformed_c;
                        cv::Point2f warped_pixel = cv::Point2f(warp_img_trg(i,0), warp_img_trg(i,1));
                        // std::cout << "Pixel transform_ " << i/nCols << " " << i%nCols << " " << transformed_r_int << " " << transformed_c_int << " " << nRows << "x" << nCols << " n_pts " << n_pts << endl;
                        // assert(transformed_c_int >= 0 && transformed_c_int < nCols);

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            // std::cout << thres_saliency_gray_ << " Grad " << fabs(grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                            // For higher performance: Interpolation is not applied here (only when computing the Hessian)
                            //if( fabs(_graySrcGradXPyr[warped_i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[warped_i]) > thres_saliency_gray_)
                            //if( fabs(_grayTrgGradXPyr[i]) > thres_saliency_gray_ || fabs(_grayTrgGradYPyr[i]) > thres_saliency_gray_)
                            if( fabs(_grayTrgGradXPyr[i]) > 0.f || fabs(_grayTrgGradYPyr[i]) > 0.f)
                            {
                                validPixelsPhoto_trg(i) = 1;
                                float intensity = bilinearInterp( graySrcPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                                float diff = intensity - _graySrcPyr[i];
                                residualsPhoto_trg(i) = diff * stdDevPhoto_inv;
                                wEstimPhoto_trg(i) = weightMEstimator(residualsPhoto_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                error2_photo += wEstimPhoto_trg(i) * residualsPhoto_trg(i) * residualsPhoto_trg(i);
                                // std::cout << i << " error2 " << error2 << " wEPhoto " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevPhoto << std::endl;

                                //v_AD_intensity[i] = fabs(diff);
                                //++n_ptsPhoto;
                            }
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            float depth = bilinearInterp_depth( depthSrcPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            if(depth > min_depth_) // if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                            {
                                // std::cout << thres_saliency_depth_ << " Grad-Depth " << fabs(depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << " " << fabs(depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int)) << std::endl;
                                //if( fabs(_depthSrcGradXPyr[warped_i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[warped_i]) > thres_saliency_depth_)
                                //if( fabs(_depthTrgGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthTrgGradYPyr[i]) > thres_saliency_depth_)
                                if( fabs(_depthTrgGradXPyr[i]) > 0.f || fabs(_depthTrgGradYPyr[i]) > 0.f)
                                {
                                    //Obtain the depth values that will be used to the compute the depth residual
                                    validPixelsDepth_trg(i) = 1;
                                    stdDevError_inv_trg(i) = 1 / std::max (stdDevDepth*(dist*dist+depth*depth), 2*stdDevDepth);
                                    residualsDepth_trg(i) = (depth - dist) * stdDevError_inv_trg(i);
                                    wEstimDepth_trg(i) = weightMEstimator(residualsDepth_trg(i)); // Apply M-estimator weighting // The weight computed by an M-estimator
                                    error2_depth += wEstimDepth_trg(i) * residualsDepth_trg(i) * residualsDepth_trg(i);
                                    // std::cout << i << " error2 " << error2 << " wDepthError " << weightedError << " weight_estim " << weight_estim << " diff " << diff << " " << stdDevError << std::endl;
                                    //++n_ptsDepth;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    SSO = (float)numVisiblePts / n_pts;
    error2 = (error2_photo + error2_depth) / numVisiblePts;

    // Compute the median absulute deviation of the projection of reference image onto the target one to update the value of the standard deviation of the intesity error
//    if(error2_photo > 0 && compute_MAD_stdDev_)
//    {
//        std::cout << " stdDevPhoto PREV " << stdDevPhoto << std::endl;
//        size_t count_valid_pix = 0;
//        std::vector<float> v_AD_intensity(n_ptsPhoto);
//        for(size_t i=0; i < imgSize; i++)
//            if( validPixelsPhoto_trg(i) ) //Compute the jacobian only for the valid points
//            {
//                v_AD_intensity[count_valid_pix] = v_AD_intensity_[i];
//                ++count_valid_pix;
//            }
//        //v_AD_intensity.resize(n_pts);
//        v_AD_intensity.resize(n_ptsPhoto);
//        float stdDevPhoto_updated = 1.4826 * median(v_AD_intensity);
//        error2_photo *= stdDevPhoto*stdDevPhoto / (stdDevPhoto_updated*stdDevPhoto_updated);
//        stdDevPhoto = stdDevPhoto_updated;
//        std::cout << " stdDevPhoto_updated    " << stdDevPhoto_updated << std::endl;
//    }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "INV error2 " << error2 << " error2_photo " << error2_photo << " error2_depth " << error2_depth << " numVisiblePts " << numVisiblePts << std::endl;
#endif
#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " errorDenseInv_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif

    return error2;
}

/*! Compute the residuals and the jacobians corresponding to the target image projected onto the source one. */
void RegisterDense::calcHessGradInv_sphere( const int &pyramidLevel,
                                            const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                            costFuncType method )//,const bool use_bilinear )
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;
    const size_t n_pts = xyz_trg_transf.rows();
    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;
    const float half_width = nCols/2 - 0.5f;

    float phi_start = -(0.5f*nRows-0.5f)*pixel_angle;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5*nRows-0.5)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    float stdDevPhoto_inv = 1./stdDevPhoto;
    float stdDevDepth_inv = 1./stdDevDepth;

    const Eigen::Matrix3f rotation = poseGuess.block(0,0,3,3);
    const Eigen::Matrix4f poseGuess_inv = poseGuess.inverse();
    const Eigen::Matrix3f rotation_inv = poseGuess_inv.block(0,0,3,3);
    const Eigen::Vector3f translation_inv = poseGuess_inv.block(0,3,3,1);

//    Eigen::MatrixXf jacobiansPhoto(imgSize,6);
//    Eigen::MatrixXf jacobiansDepth(imgSize,6);
//    jacobiansPhoto.resize(n_pts,6);
//    jacobiansDepth.resize(n_pts,6);
    jacobiansPhoto = Eigen::MatrixXf::Zero(n_pts,6);
    jacobiansDepth = Eigen::MatrixXf::Zero(n_pts,6);
//    assert(residualsPhoto_trg.rows() == imgSize && residualsDepth_trg.rows() == imgSize);

    //    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    //    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    //    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    //    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

        float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
        float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);
        float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
        float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);

    if(visualizeIterations)
    {
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
    }

    if(use_salient_pixels_)
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
            // int countSalientPix = 0;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( validPixels_trg(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) )
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    // The Jacobian of the inverse pixel transformation.
                    // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
                    //Eigen::Vector3f xyz_inv = rotation_inv*LUT_xyz_target[i] + translation_inv;
                    Eigen::Matrix<float,3,6> jacobianT36_inv;
                    jacobianT36_inv.block(0,0,3,3) = -rotation_inv;
                    jacobianT36_inv.block(0,3,3,1) = LUT_xyz_target(i,2)*rotation.block(0,1,3,1) - LUT_xyz_target(i,1)*rotation.block(0,2,3,1);
                    jacobianT36_inv.block(0,4,3,1) = LUT_xyz_target(i,0)*rotation.block(0,2,3,1) - LUT_xyz_target(i,2)*rotation.block(0,0,3,1);
                    jacobianT36_inv.block(0,5,3,1) = LUT_xyz_target(i,1)*rotation.block(0,0,3,1) - LUT_xyz_target(i,0)*rotation.block(0,1,3,1);
                    //Eigen::Map<Eigen::Vector3f>(&jacobianT36_inv(0,3)) = LUT_xyz_target(i,2)*Eigen::Vector3f(&rotation(0,1)) - Eigen::Vector3f(&rotation(0,2));
                    //Eigen::Map<Eigen::Vector3f>(&jacobianT36_inv(0,4)) = LUT_xyz_target(i,0)*Eigen::Vector3f(&rotation(0,2)) - LUT_xyz_target(i,2)*Eigen::Vector3f(&rotation(0,0));
                    //Eigen::Map<Eigen::Vector3f>(&jacobianT36_inv(0,5)) = LUT_xyz_target(i,1)*Eigen::Vector3f(&rotation(0,0)) - LUT_xyz_target(i,0)*Eigen::Vector3f(&rotation(0,1));

                    // The Jacobian of the spherical projection
                    Eigen::Matrix<float,2,3> jacobianProj23;
                    float dist2 = dist * dist;
                    float x2_z2 = dist2 - xyz(1)*xyz(1);
                    float x2_z2_sqrt = sqrt(x2_z2);
                    float commonDer_c = pixel_angle_inv / x2_z2;
                    float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
                    jacobianProj23(0,0) = commonDer_c * xyz(2);
                    jacobianProj23(0,1) = 0;
                    jacobianProj23(0,2) = -commonDer_c * xyz(0);
                    //jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
                    jacobianProj23(1,1) =-commonDer_r * x2_z2;
                    //jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);
                    float commonDer_r_y = commonDer_r * xyz(1);
                    jacobianProj23(1,0) = commonDer_r_y * xyz(0);
                    jacobianProj23(1,2) = commonDer_r_y * xyz(2);

                    Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

                    // Eigen::Matrix<float,2,6> jacobianWarpRt;
                    // computeJacobian26_wT_sphere_inv(xyz, LUT_xyz_target.block(i,0,1,3).transpose(), rotation, dist, pixel_angle_inv, jacobianWarpRt);

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                            if( validPixelsPhoto_trg(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_trg(i)); // We keep the wrong name to 'source' to avoid duplicating more structures
                            //warped_target_grayImage.at<float>(warp_pixels_trg(i)) = grayTrgPyr[pyramidLevel].at<float>(i);

                            Eigen::Matrix<float,1,2> img_gradient;
                            //img_gradient(0,0) = _graySrcGradXPyr[warp_pixels_trg(i)];
                            //img_gradient(0,1) = _graySrcGradYPyr[warp_pixels_trg(i)];
                            img_gradient(0,0) = _grayTrgGradXPyr[validPixels_trg(i)];
                            img_gradient(0,1) = _grayTrgGradYPyr[validPixels_trg(i)];

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_trg(i) *= weight_estim_sqrt;
                            //std::cout << i << " validPixels_trg(i) " << validPixels_trg(i) << " img_gradient " << img_gradient << std::endl;
                            //std::cout << "jacobianWarpRt \n" << jacobianWarpRt << std::endl;
                            //std::cout << "jacobianT36_inv \n" << jacobianT36_inv << std::endl;
                            // std::cout << "jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " residualsPhoto_trg(i) " << residualsPhoto_trg(i) << std::endl;
                            // mrpt::system::pause();
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        if( validPixelsDepth_trg(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_trg(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient;
                            //depth_gradient(0,0) = _depthSrcGradXPyr[warp_pixels_trg(i)];
                            //depth_gradient(0,1) = _depthSrcGradYPyr[warp_pixels_trg(i)];
                            depth_gradient(0,0) = _depthTrgGradXPyr[validPixels_trg(i)];
                            depth_gradient(0,1) = _depthTrgGradYPyr[validPixels_trg(i)];
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,3> jacobianDepthSrc = dist_inv * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt-jacobianDepthSrc*jacobianT36_inv);
                            residualsDepth_trg(i) *= weight_estim_sqrt;
                            // std::cout << "residualsDepth_trg \n " << residualsDepth_trg << std::endl;
                        }
                }
            }
        }
        else
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( validPixels_trg(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) ) //Compute the jacobian only for the valid points
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    // The Jacobian of the inverse pixel transformation.
                    // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
                    //Eigen::Vector3f xyz_inv = rotation_inv*LUT_xyz_target[i] + translation_inv;
                    Eigen::Matrix<float,3,6> jacobianT36_inv;
                    jacobianT36_inv.block(0,0,3,3) = -rotation_inv;
                    jacobianT36_inv.block(0,3,3,1) = LUT_xyz_target(i,2)*rotation.block(0,1,3,1) - LUT_xyz_target(i,1)*rotation.block(0,2,3,1);
                    jacobianT36_inv.block(0,4,3,1) = LUT_xyz_target(i,0)*rotation.block(0,2,3,1) - LUT_xyz_target(i,2)*rotation.block(0,0,3,1);
                    jacobianT36_inv.block(0,5,3,1) = LUT_xyz_target(i,1)*rotation.block(0,0,3,1) - LUT_xyz_target(i,0)*rotation.block(0,1,3,1);

                    // The Jacobian of the spherical projection
                    Eigen::Matrix<float,2,3> jacobianProj23;
                    float dist2 = dist * dist;
                    float x2_z2 = dist2 - xyz(1)*xyz(1);
                    float x2_z2_sqrt = sqrt(x2_z2);
                    float commonDer_c = pixel_angle_inv / x2_z2;
                    float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
                    jacobianProj23(0,0) = commonDer_c * xyz(2);
                    jacobianProj23(0,1) = 0;
                    jacobianProj23(0,2) = -commonDer_c * xyz(0);
                    jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
                    jacobianProj23(1,1) =-commonDer_r * x2_z2;
                    jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);

                    Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

                    //                    Eigen::Matrix<float,2,6> jacobianWarpRt_;
                    //                    computeJacobian26_wT_sphere_inv(xyz, LUT_xyz_target.block(i,0,1,3).transpose(), rotation, dist, pixel_angle_inv, jacobianWarpRt_);
                    //                     std::cout << "jacobianWarpRt_ \n" << jacobianWarpRt_ << " jacobianWarpRt \n" << jacobianWarpRt << std::endl;
                    //                     mrpt::system::pause();

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_trg(i) )
                        {
                            if(visualizeIterations)
                                warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(validPixels_trg(i)); // We keep the wrong name to 'source' to avoid duplicating more structures
                            //warped_target_grayImage.at<float>(warp_pixels_trg(i)) = grayTrgPyr[pyramidLevel].at<float>(i);

                            Eigen::Matrix<float,1,2> img_gradient;
                            //img_gradient(0,0) = _graySrcGradXPyr[warp_pixels_trg(i)];
                            //img_gradient(0,1) = _graySrcGradYPyr[warp_pixels_trg(i)];
                            img_gradient(0,0) = _grayTrgGradXPyr[validPixels_trg(i)];
                            img_gradient(0,1) = _grayTrgGradYPyr[validPixels_trg(i)];

                            //Obtain the pixel values that will be used to compute the pixel residual
                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                            jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                            residualsPhoto_trg(i) *= weight_estim_sqrt;
                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                            // mrpt::system::pause();
                        }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_trg(i) )
                        {
                            if(visualizeIterations)
                                warped_depth.at<float>(warp_pixels_trg(i)) = dist;

                            Eigen::Matrix<float,1,2> depth_gradient;
                            //depth_gradient(0,0) = _depthSrcGradXPyr[warp_pixels_trg(i)];
                            //depth_gradient(0,1) = _depthSrcGradYPyr[warp_pixels_trg(i)];
                            depth_gradient(0,0) = _depthTrgGradXPyr[validPixels_trg(i)];
                            depth_gradient(0,1) = _depthTrgGradYPyr[validPixels_trg(i)];
                            // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<float,1,3> jacobianDepthSrc = dist_inv * xyz.transpose();
                            float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                            jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt-jacobianDepthSrc*jacobianT36_inv);
                            residualsDepth_trg(i) *= weight_estim_sqrt;
                            // std::cout << "residualsDepth_trg \n " << residualsDepth_trg << std::endl;
                        }
                }
            }
        }
    }
    else
    {
        if( !use_bilinear_ || pyramidLevel !=0 )
        {
            // int countSalientPix = 0;
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
                //if( validPixels_trg(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) )
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    // The Jacobian of the inverse pixel transformation.
                    // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
                    //Eigen::Vector3f xyz_inv = rotation_inv*LUT_xyz_target[i] + translation_inv;
                    Eigen::Matrix<float,3,6> jacobianT36_inv;
                    jacobianT36_inv.block(0,0,3,3) = -rotation_inv;
                    jacobianT36_inv.block(0,3,3,1) = LUT_xyz_target(i,2)*rotation.block(0,1,3,1) - LUT_xyz_target(i,1)*rotation.block(0,2,3,1);
                    jacobianT36_inv.block(0,4,3,1) = LUT_xyz_target(i,0)*rotation.block(0,2,3,1) - LUT_xyz_target(i,2)*rotation.block(0,0,3,1);
                    jacobianT36_inv.block(0,5,3,1) = LUT_xyz_target(i,1)*rotation.block(0,0,3,1) - LUT_xyz_target(i,0)*rotation.block(0,1,3,1);

                    // The Jacobian of the spherical projection
                    Eigen::Matrix<float,2,3> jacobianProj23;
                    float dist2 = dist * dist;
                    float x2_z2 = dist2 - xyz(1)*xyz(1);
                    float x2_z2_sqrt = sqrt(x2_z2);
                    float commonDer_c = pixel_angle_inv / x2_z2;
                    float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
                    jacobianProj23(0,0) = commonDer_c * xyz(2);
                    jacobianProj23(0,1) = 0;
                    jacobianProj23(0,2) = -commonDer_c * xyz(0);
                    jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
                    jacobianProj23(1,1) =-commonDer_r * x2_z2;
                    jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);

                    Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

    //                Eigen::Matrix<float,2,6> jacobianWarpRt;
    //                computeJacobian26_wT_sphere_inv(xyz, LUT_xyz_target.block(i,0,1,3).transpose(), rotation, dist, pixel_angle_inv, jacobianWarpRt);

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_trg(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(i); // We keep the wrong name to 'source' to avoid duplicating more structures
                            //warped_target_grayImage.at<float>(warp_pixels_trg(i)) = grayTrgPyr[pyramidLevel].at<float>(i);

                        Eigen::Matrix<float,1,2> img_gradient;
                        //img_gradient(0,0) = _graySrcGradXPyr[warp_pixels_trg(i)];
                        //img_gradient(0,1) = _graySrcGradYPyr[warp_pixels_trg(i)];
                        img_gradient(0,0) = _grayTrgGradXPyr[i];
                        img_gradient(0,1) = _grayTrgGradYPyr[i];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_trg(i) *= weight_estim_sqrt;
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_trg(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_trg(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient;
                        //depth_gradient(0,0) = _depthSrcGradXPyr[warp_pixels_trg(i)];
                        //depth_gradient(0,1) = _depthSrcGradYPyr[warp_pixels_trg(i)];
                        depth_gradient(0,0) = _depthTrgGradXPyr[i];
                        depth_gradient(0,1) = _depthTrgGradYPyr[i];
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,3> jacobianDepthSrc = dist_inv * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt-jacobianDepthSrc*jacobianT36_inv);
                        residualsDepth_trg(i) *= weight_estim_sqrt;
                        // std::cout << "residualsDepth_trg \n " << residualsDepth_trg << std::endl;
                    }
                }
            }
        }
        else
        {
    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
            for(size_t i=0; i < n_pts; i++)
            {
//                if( validPixels_trg(i) ) //Compute the jacobian only for the visible points
                if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) ) //Compute the jacobian only for the valid points
                {
                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                    // cout << "3D pts " << point3D.transpose() << " transformed " << xyz.transpose() << endl;

                    //Project the 3D point to the S2 sphere
                    float dist = xyz.norm();
                    float dist_inv = 1.f / dist;

                    // The Jacobian of the inverse pixel transformation.
                    // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
                    //Eigen::Vector3f xyz_inv = rotation_inv*LUT_xyz_target[i] + translation_inv;
                    Eigen::Matrix<float,3,6> jacobianT36_inv;
                    jacobianT36_inv.block(0,0,3,3) = -rotation_inv;
                    jacobianT36_inv.block(0,3,3,1) = LUT_xyz_target(i,2)*rotation.block(0,1,3,1) - LUT_xyz_target(i,1)*rotation.block(0,2,3,1);
                    jacobianT36_inv.block(0,4,3,1) = LUT_xyz_target(i,0)*rotation.block(0,2,3,1) - LUT_xyz_target(i,2)*rotation.block(0,0,3,1);
                    jacobianT36_inv.block(0,5,3,1) = LUT_xyz_target(i,1)*rotation.block(0,0,3,1) - LUT_xyz_target(i,0)*rotation.block(0,1,3,1);

                    // The Jacobian of the spherical projection
                    Eigen::Matrix<float,2,3> jacobianProj23;
                    float dist2 = dist * dist;
                    float x2_z2 = dist2 - xyz(1)*xyz(1);
                    float x2_z2_sqrt = sqrt(x2_z2);
                    float commonDer_c = pixel_angle_inv / x2_z2;
                    float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
                    jacobianProj23(0,0) = commonDer_c * xyz(2);
                    jacobianProj23(0,1) = 0;
                    jacobianProj23(0,2) = -commonDer_c * xyz(0);
                    jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
                    jacobianProj23(1,1) =-commonDer_r * x2_z2;
                    jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);

                    Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

    //                    Eigen::Matrix<float,2,6> jacobianWarpRt_;
    //                    computeJacobian26_wT_sphere_inv(xyz, LUT_xyz_target.block(i,0,1,3).transpose(), rotation, dist, pixel_angle_inv, jacobianWarpRt_);
    //                     std::cout << "jacobianWarpRt_ \n" << jacobianWarpRt_ << " jacobianWarpRt \n" << jacobianWarpRt << std::endl;
    //                     mrpt::system::pause();

                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsPhoto_trg(i) )
                    {
                        if(visualizeIterations)
                            warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(i); // We keep the wrong name to 'source' to avoid duplicating more structures
                            //warped_target_grayImage.at<float>(warp_pixels_trg(i)) = grayTrgPyr[pyramidLevel].at<float>(i);

                        Eigen::Matrix<float,1,2> img_gradient;
                        //img_gradient(0,0) = _graySrcGradXPyr[warp_pixels_trg(i)];
                        //img_gradient(0,1) = _graySrcGradYPyr[warp_pixels_trg(i)];
                        img_gradient(0,0) = _grayTrgGradXPyr[i];
                        img_gradient(0,1) = _grayTrgGradYPyr[i];

                        //Obtain the pixel values that will be used to compute the pixel residual
                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                        float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                        jacobiansPhoto.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                        residualsPhoto_trg(i) *= weight_estim_sqrt;
                        // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                        // mrpt::system::pause();
                    }
                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                    if( validPixelsDepth_trg(i) )
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_trg(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient;
                        //depth_gradient(0,0) = _depthSrcGradXPyr[warp_pixels_trg(i)];
                        //depth_gradient(0,1) = _depthSrcGradYPyr[warp_pixels_trg(i)];
                        depth_gradient(0,0) = _depthTrgGradXPyr[i];
                        depth_gradient(0,1) = _depthTrgGradYPyr[i];
                        // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,3> jacobianDepthSrc = dist_inv * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                        jacobiansDepth.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt-jacobianDepthSrc*jacobianT36_inv);
                        residualsDepth_trg(i) *= weight_estim_sqrt;
                        // std::cout << "residualsDepth_trg \n " << residualsDepth_trg << std::endl;
                    }
                }
            }
        }
    }

    // Update the Hessian and the Gradient
    //hessian.setZero();
    //gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansPhoto, residualsPhoto_trg, validPixelsPhoto_trg);

    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        updateHessianAndGradient(jacobiansDepth, residualsDepth_trg, validPixelsDepth_trg);

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGradInv_sphere took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 *  This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 *  between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 */
void RegisterDense::registerRGBD(const Eigen::Matrix4f pose_guess, costFuncType method, const int occlusion )
{
    //std::cout << " RegisterDense::register \n";
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    if(occlusion == 2)
    {
        min_depth_Outliers = 2*stdDevDepth; // in meters
        thresDepthOutliers = max_depth_Outliers;
    }

    num_iterations.resize(nPyrLevels); // Store the number of iterations
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;

        // Make LUT to store the values of the 3D points of the source image
        computePinholeXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);

        LUT_xyz_source.resize(imgSize,3);
        const float scaleFactor = 1.0/pow(2,pyramidLevel);
        fx = cameraMatrix(0,0)*scaleFactor;
        fy = cameraMatrix(1,1)*scaleFactor;
        ox = cameraMatrix(0,2)*scaleFactor;
        oy = cameraMatrix(1,2)*scaleFactor;
        const float inv_fx = 1./fx;
        const float inv_fy = 1./fy;

        for(size_t r=0;r<nRows; r++)
        {
            for(size_t c=0;c<nCols; c++)
            {
                int i = r*nCols + c;
                LUT_xyz_source(i,2) = depthSrcPyr[pyramidLevel].at<float>(r,c); //LUT_xyz_source(i,2) = 0.001f*depthSrcPyr[pyramidLevel].at<unsigned short>(r,c);

                //Compute the 3D coordinates of the pij of the source frame
                //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
                //std::cout << depthSrcPyr[pyramidLevel].type() << " LUT_xyz_source " << i << " x " << LUT_xyz_source(i,2) << " thres " << min_depth_ << " " << max_depth_ << std::endl;
                if(min_depth_ < LUT_xyz_source(i,2) && LUT_xyz_source(i,2) < max_depth_) //Compute the jacobian only for the valid points
                {
                    LUT_xyz_source(i,0) = (c - ox) * LUT_xyz_source(i,2) * inv_fx;
                    LUT_xyz_source(i,1) = (r - oy) * LUT_xyz_source(i,2) * inv_fy;
                }
                else
                    LUT_xyz_source(i,0) = INVALID_POINT;
            }
        }

        double lambda = 0.01; // Levenberg-Marquardt (LM) lambda
        double step = 10; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-4;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        double error = errorDense(pyramidLevel, pose_estim, method);
//        double error, new_error;
//        if(occlusion == 0)
//            error = errorDense(pyramidLevel, pose_estim, method);
//        else if(occlusion == 1)
//            error = errorDense_Occ1(pyramidLevel, pose_estim, method);
//        else if(occlusion == 2)
//            error = errorDense_Occ2(pyramidLevel, pose_estim, method);

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "error2 " << error << std::endl;
        std::cout << "Level " << pyramidLevel << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm; tm.start();
//#endif

            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            calcHessGrad(pyramidLevel, pose_estim, method);
//            if(occlusion == 0)
//                calcHessGrad(pyramidLevel, pose_estim, method);
//            else if(occlusion == 1)
//                calcHessGrad_Occ1(pyramidLevel, pose_estim, method);
//            else if(occlusion == 2)
//                calcHessGrad_Occ2(pyramidLevel, pose_estim, method);
//            else
//                assert(false);

            //                assert(hessian.rank() == 6); // Make sure that the problem is observable
            if( hessian.rank() != 6 )
            //if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -hessian.inverse() * gradient;
            //                update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
            //            std::cout << "update_pose \n" << update_pose.transpose() << std::endl;

            double new_error = errorDense(pyramidLevel, pose_estim_temp, method);
//            if(occlusion == 0)
//                new_error = errorDense(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 1)
//                new_error = errorDense_Occ1(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 2)
//                new_error = errorDense_Occ2(pyramidLevel, pose_estim_temp, method);

            diff_error = error - new_error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            //std::cout << "update_pose \n" << update_pose.transpose() << std::endl;
            std::cout << "diff_error " << diff_error << std::endl;
#endif
            if(diff_error > 0)
            {
                // cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }

//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop(); std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
//#endif

            if(visualizeIterations)
            {
                //std::cout << "visualizeIterations\n";
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    // std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    // cout << "type " << grayTrgPyr[pyramidLevel].type() << " " << warped_gray.type() << endl;

                    //                        cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                    //                        cv::imshow("src", graySrcPyr[pyramidLevel]);
                    //                        cv::imshow("optimize::imgDiff", imgDiff);
                    //                        cv::imshow("warp", warped_gray);

                    cv::Mat DispImage = cv::Mat(2*grayTrgPyr[pyramidLevel].rows+4, 2*grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].type(), cv::Scalar(255)); // cv::Scalar(100, 100, 200)
                    grayTrgPyr[pyramidLevel].copyTo(DispImage(cv::Rect(0, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    graySrcPyr[pyramidLevel].copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    warped_gray.copyTo(DispImage(cv::Rect(0, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    imgDiff.copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    //cv::namedWindow("Photoconsistency", cv::WINDOW_AUTOSIZE );// Create a window for display.
                    cv::imshow("Photoconsistency", DispImage);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    //std::cout << "sizes " << nRows << " " << nCols << " " << "sizes " << depthTrgPyr[pyramidLevel].rows << " " << depthTrgPyr[pyramidLevel].cols << " " << "sizes " << warped_depth.rows << " " << warped_depth.cols << " " << grayTrgPyr[pyramidLevel].type() << std::endl;
                    cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    //cv::imshow("weightedError", weightedError);

                    cv::Mat DispImage = cv::Mat(2*grayTrgPyr[pyramidLevel].rows+4, 2*grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].type(), cv::Scalar(255)); // cv::Scalar(100, 100, 200)
                    depthTrgPyr[pyramidLevel].copyTo(DispImage(cv::Rect(0, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    depthSrcPyr[pyramidLevel].copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    warped_depth.copyTo(DispImage(cv::Rect(0, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    weightedError.copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    DispImage.convertTo(DispImage, CV_8U, 22.5);

                    //cv::namedWindow("Depth-consistency", cv::WINDOW_AUTOSIZE );// Create a window for display.
                    cv::imshow("Depth-consistency", DispImage);
                }
                if(occlusion == 2)
                {
                    // Draw the segmented features: pixels moving forward and backward and occlusions
                    cv::Mat segmentedSrcImg = colorSrcPyr[pyramidLevel].clone(); // cv::Mat segmentedSrcImg(colorSrcPyr[pyramidLevel],true); // Both should be the same
                    //std::cout << "imgSize  " << imgSize << " nRows*nCols " << nRows << "x" << nCols << " types " << segmentedSrcImg.type() << " " << CV_8UC3 << std::endl;
                    for(unsigned i=0; i < imgSize; i++)
                    {
                        if(mask_dynamic_occlusion.at<uchar>(i) == 255) // Draw in Red (BGR)
                        {
                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 255;
                        }
                        else if(mask_dynamic_occlusion.at<uchar>(i) == 155)
                        {
                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 255;
                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
                        }
                        else if(mask_dynamic_occlusion.at<uchar>(i) == 55)
                        {
                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 255;
                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
                        }
                    }
                    cv::imshow("SegmentedSRC", segmentedSrcImg);
                }
                cv::waitKey(0);
            }
        }

        num_iterations[pyramidLevel] = it;
    }

    //        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
    //            cv::destroyWindow("Photoconsistency");
    //        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
    //            cv::destroyWindow("Depth-consistency");

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: ";
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " registerRGBD took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 *  This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 *  between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 */
void RegisterDense::registerRGBD_bidirectional(const Eigen::Matrix4f pose_guess, costFuncType method, const int occlusion )
{
    //std::cout << " RegisterDense::register \n";
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    if(occlusion == 2)
    {
        min_depth_Outliers = 2*stdDevDepth; // in meters
        thresDepthOutliers = max_depth_Outliers;
    }

    num_iterations.resize(nPyrLevels); // Store the number of iterations
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;

        // Make LUT to store the values of the 3D points of the source image
        computePinholeXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
        computePinholeXYZ(depthTrgPyr[pyramidLevel], LUT_xyz_target, validPixels_trg);

        LUT_xyz_source.resize(imgSize,3);
        const float scaleFactor = 1.0/pow(2,pyramidLevel);
        fx = cameraMatrix(0,0)*scaleFactor;
        fy = cameraMatrix(1,1)*scaleFactor;
        ox = cameraMatrix(0,2)*scaleFactor;
        oy = cameraMatrix(1,2)*scaleFactor;
        const float inv_fx = 1./fx;
        const float inv_fy = 1./fy;

        for(size_t r=0;r<nRows; r++)
        {
            for(size_t c=0;c<nCols; c++)
            {
                int i = r*nCols + c;
                LUT_xyz_source(i,2) = depthSrcPyr[pyramidLevel].at<float>(r,c); //LUT_xyz_source(i,2) = 0.001f*depthSrcPyr[pyramidLevel].at<unsigned short>(r,c);

                //Compute the 3D coordinates of the pij of the source frame
                //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
                //std::cout << depthSrcPyr[pyramidLevel].type() << " LUT_xyz_source " << i << " x " << LUT_xyz_source(i,2) << " thres " << min_depth_ << " " << max_depth_ << std::endl;
                if(min_depth_ < LUT_xyz_source(i,2) && LUT_xyz_source(i,2) < max_depth_) //Compute the jacobian only for the valid points
                {
                    LUT_xyz_source(i,0) = (c - ox) * LUT_xyz_source(i,2) * inv_fx;
                    LUT_xyz_source(i,1) = (r - oy) * LUT_xyz_source(i,2) * inv_fy;
                }
                else
                    LUT_xyz_source(i,0) = INVALID_POINT;
            }
        }

        double lambda = 0.01; // Levenberg-Marquardt (LM) lambda
        double step = 10; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-4;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        double error = errorDense(pyramidLevel, pose_estim, method) + errorDense_inv(pyramidLevel, pose_estim, method);
//        double error, new_error;
//        if(occlusion == 0)
//            error = errorDense(pyramidLevel, pose_estim, method);
//        else if(occlusion == 1)
//            error = errorDense_Occ1(pyramidLevel, pose_estim, method);
//        else if(occlusion == 2)
//            error = errorDense_Occ2(pyramidLevel, pose_estim, method);

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "error2 " << error << std::endl;
        std::cout << "Level " << pyramidLevel << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm; tm.start();
//#endif

            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            calcHessGrad(pyramidLevel, pose_estim, method);
            calcHessGrad_inv(pyramidLevel, pose_estim, method);

//            if(occlusion == 0)
//                calcHessGrad(pyramidLevel, pose_estim, method);
//            else if(occlusion == 1)
//                calcHessGrad_Occ1(pyramidLevel, pose_estim, method);
//            else if(occlusion == 2)
//                calcHessGrad_Occ2(pyramidLevel, pose_estim, method);
//            else
//                assert(false);

            //                assert(hessian.rank() == 6); // Make sure that the problem is observable
            if( hessian.rank() != 6 )
            //if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -hessian.inverse() * gradient;
            //                update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
            //            std::cout << "update_pose \n" << update_pose.transpose() << std::endl;

            double new_error = errorDense(pyramidLevel, pose_estim_temp, method) + errorDense_inv(pyramidLevel, pose_estim_temp, method);
//            if(occlusion == 0)
//                new_error = errorDense(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 1)
//                new_error = errorDense_Occ1(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 2)
//                new_error = errorDense_Occ2(pyramidLevel, pose_estim_temp, method);

            diff_error = error - new_error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            //std::cout << "update_pose \n" << update_pose.transpose() << std::endl;
            std::cout << "diff_error " << diff_error << std::endl;
#endif
            if(diff_error > 0)
            {
                // cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }
//            else
//            {
//                unsigned LM_it = 0;
//                while(LM_it < LM_maxIters && diff_error < 0)
//                {
//                    lambda = lambda * step;

//                    update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
//                    Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
//                    pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
//                    //new_error = errorDense(pyramidLevel, pose_estim_temp, method);
//                    if(occlusion == 0)
//                        new_error = errorDense(pyramidLevel, pose_estim_temp, method);
//                    else if(occlusion == 1)
//                        new_error = errorDense_Occ1(pyramidLevel, pose_estim_temp, method);
//                    else if(occlusion == 2)
//                        new_error = errorDense_Occ2(pyramidLevel, pose_estim_temp, method);
//                    diff_error = error - new_error;
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//                    std::cout << "diff_error LM " << diff_error << std::endl;
//#endif
//                    if(diff_error > 0)
//                    {
//                        //                        cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
//                        pose_estim = pose_estim_temp;
//                        error = new_error;
//                        it = it+1;
//                    }
//                    else
//                        LM_it = LM_it + 1;
//                }
//            }

//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop(); std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
//#endif

            if(visualizeIterations)
            {
                //std::cout << "visualizeIterations\n";
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    // std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    // cout << "type " << grayTrgPyr[pyramidLevel].type() << " " << warped_gray.type() << endl;

                    //                        cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                    //                        cv::imshow("src", graySrcPyr[pyramidLevel]);
                    //                        cv::imshow("optimize::imgDiff", imgDiff);
                    //                        cv::imshow("warp", warped_gray);

                    cv::Mat DispImage = cv::Mat(2*grayTrgPyr[pyramidLevel].rows+4, 2*grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].type(), cv::Scalar(255)); // cv::Scalar(100, 100, 200)
                    grayTrgPyr[pyramidLevel].copyTo(DispImage(cv::Rect(0, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    graySrcPyr[pyramidLevel].copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    warped_gray.copyTo(DispImage(cv::Rect(0, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    imgDiff.copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    //cv::namedWindow("Photoconsistency", cv::WINDOW_AUTOSIZE );// Create a window for display.
                    cv::imshow("Photoconsistency", DispImage);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    //std::cout << "sizes " << nRows << " " << nCols << " " << "sizes " << depthTrgPyr[pyramidLevel].rows << " " << depthTrgPyr[pyramidLevel].cols << " " << "sizes " << warped_depth.rows << " " << warped_depth.cols << " " << grayTrgPyr[pyramidLevel].type() << std::endl;
                    cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    //cv::imshow("weightedError", weightedError);

                    cv::Mat DispImage = cv::Mat(2*grayTrgPyr[pyramidLevel].rows+4, 2*grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].type(), cv::Scalar(255)); // cv::Scalar(100, 100, 200)
                    depthTrgPyr[pyramidLevel].copyTo(DispImage(cv::Rect(0, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    depthSrcPyr[pyramidLevel].copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, 0, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    warped_depth.copyTo(DispImage(cv::Rect(0, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    weightedError.copyTo(DispImage(cv::Rect(grayTrgPyr[pyramidLevel].cols+4, grayTrgPyr[pyramidLevel].rows+4, grayTrgPyr[pyramidLevel].cols, grayTrgPyr[pyramidLevel].rows)));
                    DispImage.convertTo(DispImage, CV_8U, 22.5);

                    //cv::namedWindow("Depth-consistency", cv::WINDOW_AUTOSIZE );// Create a window for display.
                    cv::imshow("Depth-consistency", DispImage);
                }
                if(occlusion == 2)
                {
                    // Draw the segmented features: pixels moving forward and backward and occlusions
                    cv::Mat segmentedSrcImg = colorSrcPyr[pyramidLevel].clone(); // cv::Mat segmentedSrcImg(colorSrcPyr[pyramidLevel],true); // Both should be the same
                    //std::cout << "imgSize  " << imgSize << " nRows*nCols " << nRows << "x" << nCols << " types " << segmentedSrcImg.type() << " " << CV_8UC3 << std::endl;
                    for(unsigned i=0; i < imgSize; i++)
                    {
                        if(mask_dynamic_occlusion.at<uchar>(i) == 255) // Draw in Red (BGR)
                        {
                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 255;
                        }
                        else if(mask_dynamic_occlusion.at<uchar>(i) == 155)
                        {
                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 255;
                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
                        }
                        else if(mask_dynamic_occlusion.at<uchar>(i) == 55)
                        {
                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 255;
                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
                        }
                    }
                    cv::imshow("SegmentedSRC", segmentedSrcImg);
                }
                cv::waitKey(0);
            }
        }

        num_iterations[pyramidLevel] = it;
    }

    //        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
    //            cv::destroyWindow("Photoconsistency");
    //        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
    //            cv::destroyWindow("Depth-consistency");

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: ";
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " registerRGBD took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "RegisterDense::register360 " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    double error;
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    num_iterations.resize(nPyrLevels); // Store the number of iterations
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
//                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
//                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
//                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
//                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
//                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
//                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
//                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());

                graySrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradXPyr[pyramidLevel].type());
                graySrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradYPyr[pyramidLevel].type());
                depthSrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradXPyr[pyramidLevel].type());
                depthSrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
        if(use_salient_pixels_)
        {
            getSalientPoints_sphere(LUT_xyz_source, validPixels_src,
                                    depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
                                    graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth

//            getSalientPoints_sphere_sse(LUT_xyz_source, validPixels_src,
//                                        depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
//                                        graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel] );
        }
        else
        {
            //computePointsXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source);
            computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
        }
        //cout << LUT_xyz_source.rows() << " pts LUT_xyz_source \n";

        double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        error = errorDense_sphere(pyramidLevel, pose_estim, method);
//        if(occlusion == 0)
//            error = errorDense_sphere(pyramidLevel, pose_estim, method);
//        else if(occlusion == 1)
//            error = errorDense_sphereOcc1(pyramidLevel, pose_estim, method);
//        else if(occlusion == 2)
//            error = errorDense_sphereOcc2(pyramidLevel, pose_estim, method);

//        for(size_t i=0; i < LUT_xyz_source.rows(); i++)
//            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
//                std::cout << i << " " << validPixels_src(i) << " pt " << xyz_src_transf.block(i,0,1,3) << " " << validPixelsPhoto_src(i) << " resPhoto " << residualsPhoto_src(i) << " " << validPixelsPhoto_src(i) << " resDepth " << residualsDepth_src(i) << std::endl;

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif
            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            calcHessGrad_sphere(pyramidLevel, pose_estim, method);
//            std::cout << "hessian \n" << hessian << std::endl;
//            std::cout << "gradient \n" << gradient.transpose() << std::endl;

//            if(occlusion == 0)
//                calcHessGrad_sphere(pyramidLevel, pose_estim, method);
//            else if(occlusion == 1)
//                calcHessGrad_sphereOcc1(pyramidLevel, pose_estim, method);
//            else if(occlusion == 2)
//                calcHessGrad_sphereOcc2(pyramidLevel, pose_estim, method);
//            else
//                assert(false);

            if(visualizeIterations)
            {
                cv::Mat imgDiff, weightedError;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
//                    // Draw the segmented features: pixels moving forward and backward and occlusions
//                    cv::Mat segmentedSrcImg = colorSrcPyr[pyramidLevel].clone();
//                    //std::cout << "imgSize  " << imgSize << " nRows*nCols " << nRows << "x" << nCols << " types " << segmentedSrcImg.type() << " " << CV_8UC3 << std::endl;
//                    for(unsigned i=0; i < imgSize; i++)
//                    {
//                        if(mask_dynamic_occlusion.at<uchar>(i) == 255) // Draw in Red (BGR)
//                        {
//                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 255;
//                        }
//                        else if(mask_dynamic_occlusion.at<uchar>(i) == 155)
//                        {
//                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 255;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
//                        }
//                        //                        else if(mask_dynamic_occlusion.at<uchar>(i) == 55)
//                        //                        {
//                        //                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 255;
//                        //                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
//                        //                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
//                        //                        }
//                    }
//                    cv::imshow("SegmentedSRC", segmentedSrcImg);

                    cv::imshow("trg", grayTrgPyr[pyramidLevel]);
                    cv::imshow("src", graySrcPyr[pyramidLevel]);

                    imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    //                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    cv::imshow("intensityDiff", imgDiff);
                    cv::imshow("warp", warped_gray);
                    //                    cv::Mat imgDiffsave;
                    //                    imgDiff.convertTo(imgDiffsave, CV_8UC1, 255);
                    //                    cv::imwrite( mrpt::format("/home/edu/weightedError.png"), imgDiffsave);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    cv::imshow("weightedError", weightedError);
                }
                cv::waitKey(0);
            }

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -(hessian.inverse() * gradient);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            double new_error;
//            if(occlusion == 0)
//                new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 1)
//                new_error = errorDense_sphereOcc1(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 2)
//                new_error = errorDense_sphereOcc2(pyramidLevel, pose_estim_temp, method);

            diff_error = error - new_error;
            if(diff_error > 0.0)
            {
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }
//            else
//            {
//                unsigned LM_it = 0;
//                while(LM_it < LM_maxIters && diff_error < 0)
//                {
//                    lambda = lambda * step;
//                    update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
//                    Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
//                    pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
//                    double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//                    diff_error = error - new_error;
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//                    cout << "diff_error LM " << diff_error << endl;
//#endif
//                    if(diff_error > 0.0)
//                    {
//                        cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
//                        pose_estim = pose_estim_temp;
//                        error = new_error;
//                        it = it+1;
//                    }
//                    LM_it = LM_it + 1;
//                }
//            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
            cout << " pose_estim_temp \n" << pose_estim_temp << endl;
//            cout << " pose_estim \n" << pose_estim << endl;
            // cout << "error " << errorDense_sphere(pyramidLevel, pose_estim, method) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, method) << endl;
            //mrpt::system::pause();
#endif

            if(visualizeIterations)
            {
                cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
//                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                cv::imshow("optimize::imgDiff", imgDiff);

                cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                cv::imshow("warp", warped_gray);

                cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                cv::imshow("weightedError", weightedError);

                cv::waitKey(0);
            }
        }

        num_iterations[pyramidLevel] = it;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

//#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment 360 took " << (time_end - time_start)*1000 << " ms. \n";
//#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360_rot(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "RegisterDense::register360 " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    double error;
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    num_iterations.resize(nPyrLevels); // Store the number of iterations
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
//                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
//                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
//                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
//                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
//                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
//                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
//                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());

                graySrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradXPyr[pyramidLevel].type());
                graySrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradYPyr[pyramidLevel].type());
                depthSrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradXPyr[pyramidLevel].type());
                depthSrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
        if(use_salient_pixels_)
        {
            getSalientPoints_sphere(LUT_xyz_source, validPixels_src,
                                    depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
                                    graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth

//            getSalientPoints_sphere_sse(LUT_xyz_source, validPixels_src,
//                                        depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
//                                        graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel] );
        }
        else
        {
            //computePointsXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source);
            computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
        }
        //cout << LUT_xyz_source.rows() << " pts LUT_xyz_source \n";

        double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        error = errorDense_sphere(pyramidLevel, pose_estim, method);

        // The first iteration optimizes only the rotation parameters
        calcHessGradRot_sphere(pyramidLevel, pose_estim, method);

        if(hessian_rot.rank() != 3)
        {
            std::cout << "\t The SALIENT problem is ILL-POSED \n";
            std::cout << "hessian_rot \n" << hessian_rot << std::endl;
            std::cout << "gradient_rot \n" << gradient_rot.transpose() << std::endl;
            registered_pose_ = pose_estim;
            return;
        }

        // The first iteration is done outside the loop because

        // Compute the pose update
        //Eigen::Matrix<float,3,1>
        update_pose.setZero();
        update_pose.block(3,0,3,1) = -(hessian_rot.inverse() * gradient_rot);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
        Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
        pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;
        //pose_estim_temp.block(0,0,3,3) = mrpt::poses::CPose3D::exp_rotation(mrpt::math::CArrayNumeric<double,3>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;
        cout << " pose_estim_temp \n" << pose_estim_temp << endl;
        std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";

        double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
        double diff_error = error - new_error;
        std::cout << "diff_error " << diff_error << " \n ";
        if(diff_error < 0.0)
        {
            std::cout << "\t Rotation Init: First iteration does not converge \n";
            //continue;
        }
        else
        {
            pose_estim = pose_estim_temp;
            error = new_error;
            it = it+1;
        }

        diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif
            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            calcHessGrad_sphere(pyramidLevel, pose_estim, method);
//            std::cout << "hessian \n" << hessian << std::endl;
//            std::cout << "gradient \n" << gradient.transpose() << std::endl;

            if(visualizeIterations)
            {
                cv::Mat imgDiff, weightedError;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    cv::imshow("trg", grayTrgPyr[pyramidLevel]);
                    cv::imshow("src", graySrcPyr[pyramidLevel]);

                    imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    //                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    cv::imshow("intensityDiff", imgDiff);
                    cv::imshow("warp", warped_gray);
                    //                    cv::Mat imgDiffsave;
                    //                    imgDiff.convertTo(imgDiffsave, CV_8UC1, 255);
                    //                    cv::imwrite( mrpt::format("/home/edu/weightedError.png"), imgDiffsave);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    cv::imshow("weightedError", weightedError);
                }
                cv::waitKey(0);
            }

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -(hessian.inverse() * gradient);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            double new_error;
//            if(occlusion == 0)
//                new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 1)
//                new_error = errorDense_sphereOcc1(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 2)
//                new_error = errorDense_sphereOcc2(pyramidLevel, pose_estim_temp, method);

            diff_error = error - new_error;
            if(diff_error > 0.0)
            {
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
            cout << " pose_estim_temp \n" << pose_estim_temp << endl;
//            cout << " pose_estim \n" << pose_estim << endl;
            // cout << "error " << errorDense_sphere(pyramidLevel, pose_estim, method) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, method) << endl;
            //mrpt::system::pause();
#endif

            if(visualizeIterations)
            {
                cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
//                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                cv::imshow("optimize::imgDiff", imgDiff);

                cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                cv::imshow("warp", warped_gray);

                cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                cv::imshow("weightedError", weightedError);

                cv::waitKey(0);
            }
        }

        num_iterations[pyramidLevel] = it;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

//#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment 360 took " << (time_end - time_start)*1000 << " ms. \n";
//#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360_warp(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "RegisterDense::register360 " << std::endl;
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    double error;
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    int side = 0;
    num_iterations.resize(nPyrLevels); // Store the number of iterations
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
//                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
//                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
//                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
//                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
//                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
//                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
//                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());

                graySrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradXPyr[pyramidLevel].type());
                graySrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradYPyr[pyramidLevel].type());
                depthSrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradXPyr[pyramidLevel].type());
                depthSrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

//        // Make LUT to store the values of the 3D points of the source sphere
//        if(use_salient_pixels_)
//        {
//            getSalientPoints_sphere(LUT_xyz_source, validPixels_src,
//                                    depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
//                                    graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
//                                    ); // TODO extend this function to employ only depth

////            getSalientPoints_sphere_sse(LUT_xyz_source, validPixels_src,
////                                        depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
////                                        graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel] );
//        }
//        else
        {
            //computePointsXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source);
            computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
        }
        //cout << LUT_xyz_source.rows() << " pts LUT_xyz_source \n";

        //double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        //double step = 5; // Update step
        //unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        error = errorDenseWarp_sphere(pyramidLevel, pose_estim, method);

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif

            // Compute the gradient images
            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                calcGradientXY(warped_gray, warped_gray_gradX, warped_gray_gradY);
            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                calcGradientXY(warped_depth, warped_depth_gradX, warped_depth_gradY);

            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            calcHessGrad_warp_sphere(pyramidLevel, pose_estim, method, side);
//            std::cout << "hessian \n" << hessian << std::endl;
//            std::cout << "gradient \n" << gradient.transpose() << std::endl;

            if(visualizeIterations)
            {
                cv::Mat imgDiff, weightedError;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    cv::imshow("trg", grayTrgPyr[pyramidLevel]);
                    cv::imshow("src", graySrcPyr[pyramidLevel]);

                    imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    //                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    cv::imshow("intensityDiff", imgDiff);
                    cv::imshow("warp", warped_gray);
                    //                    cv::Mat imgDiffsave;
                    //                    imgDiff.convertTo(imgDiffsave, CV_8UC1, 255);
                    //                    cv::imwrite( mrpt::format("/home/edu/weightedError.png"), imgDiffsave);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    cv::imshow("weightedError", weightedError);
                }
                cv::waitKey(0);
            }

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -(hessian.inverse() * gradient);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDenseWarp_sphere(pyramidLevel, pose_estim_temp, method);
//            double new_error;
//            if(occlusion == 0)
//                new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 1)
//                new_error = errorDense_sphereOcc1(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 2)
//                new_error = errorDense_sphereOcc2(pyramidLevel, pose_estim_temp, method);

            diff_error = error - new_error;
            if(diff_error > 0.0)
            {
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
            cout << " pose_estim_temp \n" << pose_estim_temp << endl;
//            cout << " pose_estim \n" << pose_estim << endl;
            // cout << "error " << errorDense_sphere(pyramidLevel, pose_estim, method) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, method) << endl;
            //mrpt::system::pause();
#endif

            if(visualizeIterations)
            {
                cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
//                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                cv::imshow("optimize::imgDiff", imgDiff);

                cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                cv::imshow("warp", warped_gray);

                cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                cv::imshow("weightedError", weightedError);

                cv::waitKey(0);
            }
        }

        num_iterations[pyramidLevel] = it;
        side = 1;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "register360_warp Dense alignment 360 took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360_side(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "RegisterDense::register360 " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    double error;
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    int side = 0;
    num_iterations.resize(nPyrLevels); // Store the number of iterations
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
//                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
//                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
//                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
//                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
//                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
//                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
//                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());

                graySrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradXPyr[pyramidLevel].type());
                graySrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradYPyr[pyramidLevel].type());
                depthSrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradXPyr[pyramidLevel].type());
                depthSrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
        if(use_salient_pixels_)
        {
            getSalientPoints_sphere(LUT_xyz_source, validPixels_src,
                                    depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
                                    graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth

//            getSalientPoints_sphere_sse(LUT_xyz_source, validPixels_src,
//                                        depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
//                                        graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel] );
        }
        else
        {
            //computePointsXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source);
            computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
        }
        //cout << LUT_xyz_source.rows() << " pts LUT_xyz_source \n";

        double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        error = errorDense_sphere(pyramidLevel, pose_estim, method);

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif
            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            calcHessGrad_side_sphere(pyramidLevel, pose_estim, method, side);
//            std::cout << "hessian \n" << hessian << std::endl;
//            std::cout << "gradient \n" << gradient.transpose() << std::endl;

            if(visualizeIterations)
            {
                cv::Mat imgDiff, weightedError;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    cv::imshow("trg", grayTrgPyr[pyramidLevel]);
                    cv::imshow("src", graySrcPyr[pyramidLevel]);

                    imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    //                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    cv::imshow("intensityDiff", imgDiff);
                    cv::imshow("warp", warped_gray);
                    //                    cv::Mat imgDiffsave;
                    //                    imgDiff.convertTo(imgDiffsave, CV_8UC1, 255);
                    //                    cv::imwrite( mrpt::format("/home/edu/weightedError.png"), imgDiffsave);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    cv::imshow("weightedError", weightedError);
                }
                cv::waitKey(0);
            }

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -(hessian.inverse() * gradient);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            double new_error;
//            if(occlusion == 0)
//                new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 1)
//                new_error = errorDense_sphereOcc1(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 2)
//                new_error = errorDense_sphereOcc2(pyramidLevel, pose_estim_temp, method);

            diff_error = error - new_error;
            if(diff_error > 0.0)
            {
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
            cout << " pose_estim_temp \n" << pose_estim_temp << endl;
//            cout << " pose_estim \n" << pose_estim << endl;
            // cout << "error " << errorDense_sphere(pyramidLevel, pose_estim, method) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, method) << endl;
            //mrpt::system::pause();
#endif

            if(visualizeIterations)
            {
                cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
//                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                cv::imshow("optimize::imgDiff", imgDiff);

                cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                cv::imshow("warp", warped_gray);

                cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                cv::imshow("weightedError", weightedError);

                cv::waitKey(0);
            }
        }

        num_iterations[pyramidLevel] = it;
        side = 1;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

//#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment 360 took " << (time_end - time_start)*1000 << " ms. \n";
//#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360_salientJ(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "RegisterDense::register360 " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();\
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    double error;
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    num_iterations.resize(nPyrLevels); // Store the number of iterations
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
//                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
//                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
//                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
//                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
//                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
//                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
//                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());

                graySrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradXPyr[pyramidLevel].type());
                graySrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradYPyr[pyramidLevel].type());
                depthSrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradXPyr[pyramidLevel].type());
                depthSrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
        if(use_salient_pixels_)
        {
            getSalientPoints_sphere(LUT_xyz_source, validPixels_src,
                                    depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
                                    graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth

//            getSalientPoints_sphere_sse(LUT_xyz_source, validPixels_src,
//                                        depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
//                                        graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
//                                        ); // TODO extend this function to employ only depth
        }
        else
        {
            //computePointsXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source);
            computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
        }

        double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        error = computeJacobian_sphere(pyramidLevel, pose_estim, method);
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
            //getSalientPts(jacobiansPhoto, salient_pixels_photo, 0.1f );
            getSalientPts(jacobiansPhoto, salient_pixels_photo, 1000 );
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
            //getSalientPts(jacobiansDepth, salient_pixels_depth, 0.1f );
            getSalientPts(jacobiansDepth, salient_pixels_depth, 1000 );
        trimValidPoints(LUT_xyz_source, validPixels_src, xyz_src_transf, validPixelsPhoto_src, validPixelsDepth_src,
                        method, salient_pixels_photo, salient_pixels_depth);//, validPixels_src);

        hessian.setZero();
        gradient.setZero();
        if( !salient_pixels_photo.empty() || !salient_pixels_depth.empty() )
        {
            error = computeErrorHessGrad_salient(salient_pixels_photo, salient_pixels_depth, method);
        }
        else
        {
            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                updateHessianAndGradient(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                updateHessianAndGradient(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
        }

        if(hessian.rank() != 6)
        {
            std::cout << "\t The SALIENT problem is ILL-POSED \n";
            std::cout << "hessian \n" << hessian << std::endl;
            std::cout << "gradient \n" << gradient.transpose() << std::endl;
            registered_pose_ = pose_estim;
            return;
        }

        // The first iteration is done outside the loop because

        // Compute the pose update
        update_pose = -(hessian.inverse() * gradient);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
        Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
        pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

        double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
        double diff_error = error - new_error;
        if(diff_error < 0.0)
        {
            std::cout << "\t SALIENT: First iteration does not converge \n";
            continue;
        }
        else
        {
            pose_estim = pose_estim_temp;
            error = new_error;
            it = it+1;
        }

//        for(size_t i=0; i < LUT_xyz_source.rows(); i++)
//            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) )
//                std::cout << i << " " << validPixels_src(i) << " pt " << xyz_src_transf.block(i,0,1,3) << " " << validPixelsPhoto_src(i) << " resPhoto " << residualsPhoto_src(i) << " " << validPixelsPhoto_src(i) << " resDepth " << residualsDepth_src(i) << std::endl;

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "/" << imgSize << " error " << error << std::endl;
        cout << " 1st it pose_estim_temp \n" << pose_estim_temp << endl;
        std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
        cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif
            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            calcHessGrad_sphere(pyramidLevel, pose_estim, method);
//            std::cout << "hessian \n" << hessian << std::endl;
//            std::cout << "gradient \n" << gradient.transpose() << std::endl;

//            if(occlusion == 0)
//                calcHessGrad_sphere(pyramidLevel, pose_estim, method);
//            else if(occlusion == 1)
//                calcHessGrad_sphereOcc1(pyramidLevel, pose_estim, method);
//            else if(occlusion == 2)
//                calcHessGrad_sphereOcc2(pyramidLevel, pose_estim, method);
//            else
//                assert(false);

            if(visualizeIterations)
            {
                cv::Mat imgDiff, weightedError;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
//                    // Draw the segmented features: pixels moving forward and backward and occlusions
//                    cv::Mat segmentedSrcImg = colorSrcPyr[pyramidLevel].clone();
//                    //std::cout << "imgSize  " << imgSize << " nRows*nCols " << nRows << "x" << nCols << " types " << segmentedSrcImg.type() << " " << CV_8UC3 << std::endl;
//                    for(unsigned i=0; i < imgSize; i++)
//                    {
//                        if(mask_dynamic_occlusion.at<uchar>(i) == 255) // Draw in Red (BGR)
//                        {
//                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 255;
//                        }
//                        else if(mask_dynamic_occlusion.at<uchar>(i) == 155)
//                        {
//                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 0;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 255;
//                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
//                        }
//                        //                        else if(mask_dynamic_occlusion.at<uchar>(i) == 55)
//                        //                        {
//                        //                            segmentedSrcImg.at<cv::Vec3b>(i)[0] = 255;
//                        //                            segmentedSrcImg.at<cv::Vec3b>(i)[1] = 0;
//                        //                            segmentedSrcImg.at<cv::Vec3b>(i)[2] = 0;
//                        //                        }
//                    }
//                    cv::imshow("SegmentedSRC", segmentedSrcImg);

                    cv::imshow("trg", grayTrgPyr[pyramidLevel]);
                    cv::imshow("src", graySrcPyr[pyramidLevel]);

                    imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    //                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    cv::imshow("intensityDiff", imgDiff);
                    cv::imshow("warp", warped_gray);
                    //                    cv::Mat imgDiffsave;
                    //                    imgDiff.convertTo(imgDiffsave, CV_8UC1, 255);
                    //                    cv::imwrite( mrpt::format("/home/edu/weightedError.png"), imgDiffsave);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    cv::imshow("weightedError", weightedError);
                }
                cv::waitKey(0);
            }

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -(hessian.inverse() * gradient);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            double new_error;
//            if(occlusion == 0)
//                new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 1)
//                new_error = errorDense_sphereOcc1(pyramidLevel, pose_estim_temp, method);
//            else if(occlusion == 2)
//                new_error = errorDense_sphereOcc2(pyramidLevel, pose_estim_temp, method);

            diff_error = error - new_error;
            if(diff_error > 0.0)
            {
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }
//            else
//            {
//                unsigned LM_it = 0;
//                while(LM_it < LM_maxIters && diff_error < 0)
//                {
//                    lambda = lambda * step;
//                    update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
//                    Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
//                    pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
//                    double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//                    diff_error = error - new_error;
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//                    cout << "diff_error LM " << diff_error << endl;
//#endif
//                    if(diff_error > 0.0)
//                    {
//                        cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
//                        pose_estim = pose_estim_temp;
//                        error = new_error;
//                        it = it+1;
//                    }
//                    LM_it = LM_it + 1;
//                }
//            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
            cout << " pose_estim_temp \n" << pose_estim_temp << endl;
//            cout << " pose_estim \n" << pose_estim << endl;
            // cout << "error " << errorDense_sphere(pyramidLevel, pose_estim, method) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, method) << endl;
            //mrpt::system::pause();
#endif

            if(visualizeIterations)
            {
                cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
//                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                cv::imshow("optimize::imgDiff", imgDiff);

                cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                cv::imshow("warp", warped_gray);

                cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                cv::imshow("weightedError", weightedError);

                cv::waitKey(0);
            }
        }

        num_iterations[pyramidLevel] = it;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

//#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment 360 took " << (time_end - time_start)*1000 << " ms. \n";
//#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360_IC(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "RegisterDense::register360_IC " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();\
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    double error;
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    num_iterations.resize(nPyrLevels); // Store the number of iterations
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
        if(use_salient_pixels_)
        {
            getSalientPoints_sphere(LUT_xyz_source, validPixels_src,
                                    depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
                                    graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth

//            getSalientPoints_sphere_sse(LUT_xyz_source, validPixels_src,
//                                        depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
//                                        graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
//                                        ); // TODO extend this function to employ only depth

            if(use_bilinear_)
                getSalientPoints_sphere(LUT_xyz_target, validPixels_trg,
                                        depthTrgPyr[0], depthTrgGradXPyr[0], depthTrgGradYPyr[0],
                                        grayTrgPyr[0], grayTrgGradXPyr[0], grayTrgGradYPyr[0]
                                        ); // TODO extend this function to employ only depth
        }
        else
        {
            //computePointsXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source);
            computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);

            if(use_bilinear_)
                computeSphereXYZ_sse(depthTrgPyr[0], LUT_xyz_target, validPixels_trg);
        }

        //double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        //double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        error = errorDense_sphere(pyramidLevel, pose_estim, method);
        hessian.setZero();
        calcHessGradIC_sphere(pyramidLevel, pose_estim, method);

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif
            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            gradient.setZero();
            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                updateGrad(jacobiansPhoto, residualsPhoto_src, validPixelsPhoto_src);
            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                updateGrad(jacobiansDepth, residualsDepth_src, validPixelsDepth_src);
            std::cout << "gradient \n" << gradient.transpose() << std::endl;

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -(hessian.inverse() * gradient);
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            std::cout << "new_error " << new_error << std::endl;
#endif

            diff_error = error - new_error;
            if(diff_error > 0.0)
            {
                //lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
            cout << " pose_estim_temp \n" << pose_estim_temp << endl;
//            cout << " pose_estim \n" << pose_estim << endl;
            // cout << "error " << errorDense_sphere(pyramidLevel, pose_estim, method) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, method) << endl;
            //mrpt::system::pause();
#endif

        }

        num_iterations[pyramidLevel] = it;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment IC 360 took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

void RegisterDense::register360_depthPyr(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "register360_depthPyr " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();\
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    double error;
    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    num_iterations.resize(nPyrLevels); // Store the number of iterations
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
//        computePointsXYZ(depthSrcPyr[pyramidLevel], LUT_xyz_source);
        computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);

        double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-4;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        if(pyramidLevel == 0)
            error = errorDense_sphere(pyramidLevel, pose_estim, PHOTO_DEPTH);
        else
            error = errorDense_sphere(pyramidLevel, pose_estim, DEPTH_CONSISTENCY);

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif
            //std::cout << "calcHessianAndGradient_sphere " << std::endl;
            hessian.setZero();
            gradient.setZero();
            if(pyramidLevel == 0)
                calcHessGrad_sphere(pyramidLevel, pose_estim, PHOTO_DEPTH);
            else
                calcHessGrad_sphere(pyramidLevel, pose_estim, DEPTH_CONSISTENCY);
//            std::cout << "hessian \n" << hessian << std::endl;
//            std::cout << "gradient \n" << gradient.transpose() << std::endl;

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -hessian.inverse() * gradient;
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error;
            if(pyramidLevel == 0)
                new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, PHOTO_DEPTH);
            else
                new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, DEPTH_CONSISTENCY);

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            std::cout << "new_error " << new_error << std::endl;
//            cout << "dense error " << errorDense_sphere(pyramidLevel, pose_estim, PHOTO_DEPTH) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, PHOTO_DEPTH) << endl;
#endif

            diff_error = error - new_error;
            if(diff_error > 0.0)
            {
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
            cout << " pose_estim_temp \n" << pose_estim_temp << endl;
//            cout << " pose_estim \n" << pose_estim << endl;
            // cout << "error " << errorDense_sphere(pyramidLevel, pose_estim, method) << " new_error " << errorDense_sphere(pyramidLevel, pose_estim_temp, method) << endl;
            //mrpt::system::pause();
#endif
        }

        num_iterations[pyramidLevel] = it;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

//#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment depthPyr 360 took " << (time_end - time_start)*1000 << " ms. \n";
//#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360_inv(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "register360 " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();\
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    thresDepthOutliers = 0.3;

    num_iterations.resize(nPyrLevels); // Store the number of iterations

    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());

//                graySrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradXPyr[pyramidLevel].type());
//                graySrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradYPyr[pyramidLevel].type());
//                depthSrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradXPyr[pyramidLevel].type());
//                depthSrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", graySrcGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
        if(use_salient_pixels_)
        {
            getSalientPoints_sphere(LUT_xyz_target, validPixels_trg,
                                    depthTrgPyr[pyramidLevel], depthTrgGradXPyr[pyramidLevel], depthTrgGradYPyr[pyramidLevel],
                                    grayTrgPyr[pyramidLevel], grayTrgGradXPyr[pyramidLevel], grayTrgGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth
        }
        else
        {
            //computeSphereXYZ(depthTrgPyr[pyramidLevel], LUT_xyz_target, validPixels_trg);
            computeSphereXYZ_sse(depthTrgPyr[pyramidLevel], LUT_xyz_target, validPixels_trg);
        }

        double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        double error = errorDenseInv_sphere(pyramidLevel, pose_estim, method);
        //            std::cout << "error  " << error << std::endl;

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        //            std::cout << "pose_estim \n " << pose_estim << std::endl;
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_trg.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif

            hessian.setZero();
            gradient.setZero();
            calcHessGradInv_sphere(pyramidLevel, pose_estim, method);

            if(visualizeIterations)
            {
                cv::Mat imgDiff, weightedError;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    cv::imshow("trg", grayTrgPyr[pyramidLevel]);
                    cv::imshow("src", graySrcPyr[pyramidLevel]);

                    imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    //                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    cv::imshow("intensityDiff", imgDiff);
                    cv::imshow("warp", warped_gray);
                    //                    cv::Mat imgDiffsave;
                    //                    imgDiff.convertTo(imgDiffsave, CV_8UC1, 255);
                    //                    cv::imwrite( mrpt::format("/home/edu/weightedError.png"), imgDiffsave);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    cv::imshow("weightedError", weightedError);
                }
                cv::waitKey(0);
            }

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                std::cout << "hessian \n" << hessian << std::endl;
                std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -hessian.inverse() * gradient;
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            //                update_pose_d.block(0,0,3,1) = -update_pose_d.block(0,0,3,1);
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDenseInv_sphere(pyramidLevel, pose_estim_temp, method);
            diff_error = error - new_error;

            if(diff_error > 0.0)
            {
                //                cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }
//            else
//            {
//                unsigned LM_it = 0;
//                while(LM_it < LM_maxIters && diff_error < 0)
//                {
//                    lambda = lambda * step;
//                    update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
//                    Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
//                    pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
//                    double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method);
//                    diff_error = error - new_error;
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//                    cout << "diff_error LM " << diff_error << endl;
//#endif
//                    if(diff_error > 0.0)
//                    {
//                        cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
//                        pose_estim = pose_estim_temp;
//                        error = new_error;
//                        it = it+1;
//                    }
//                    LM_it = LM_it + 1;
//                }
//            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
#endif

            //                if(visualizeIterations)
            //                {
            //                    cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
            //                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
            ////                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
            //                    cv::imshow("optimize::imgDiff", imgDiff);

            //                    cv::imshow("orig", grayTrgPyr[pyramidLevel]);
            //                    cv::imshow("warp", warped_gray);

            //                    cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
            //                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
            //                    cv::imshow("weightedError", weightedError);

            //                    cv::waitKey(0);
            //                }
        }

        num_iterations[pyramidLevel] = it;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

//#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment inverse 360 took " << (time_end - time_start)*1000 << " ms. \n";
//#endif
}


/*! Compute the residuals and the jacobians corresponding to the target image projected onto the source one. */
void RegisterDense::calcHessGrad_sphere_bidirectional( const int &pyramidLevel,
                                            const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                            costFuncType method )//,const bool use_bilinear )
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    const size_t nRows = graySrcPyr[pyramidLevel].rows;
    const size_t nCols = graySrcPyr[pyramidLevel].cols;
    const size_t imgSize = nRows*nCols;

    const float pixel_angle = 2*PI/nCols;
    const float pixel_angle_inv = 1/pixel_angle;

    const float stdDevPhoto_inv = 1./stdDevPhoto;
    const float stdDevDepth_inv = 1./stdDevDepth;

    const Eigen::Matrix3f rotation = poseGuess.block(0,0,3,3);
    const Eigen::Matrix4f poseGuess_inv = poseGuess.inverse();
    const Eigen::Matrix3f rotation_inv = poseGuess_inv.block(0,0,3,3);
    const Eigen::Vector3f translation_inv = poseGuess_inv.block(0,3,3,1);

    Eigen::MatrixXf jacobiansPhoto_src(imgSize,6);
    Eigen::MatrixXf jacobiansDepth_src(imgSize,6);
    Eigen::MatrixXf jacobiansPhoto_trg(imgSize,6);
    Eigen::MatrixXf jacobiansDepth_trg(imgSize,6);

    float *_grayTrgGradXPyr = reinterpret_cast<float*>(grayTrgGradXPyr[pyramidLevel].data);
    float *_grayTrgGradYPyr = reinterpret_cast<float*>(grayTrgGradYPyr[pyramidLevel].data);
    float *_depthTrgGradXPyr = reinterpret_cast<float*>(depthTrgGradXPyr[pyramidLevel].data);
    float *_depthTrgGradYPyr = reinterpret_cast<float*>(depthTrgGradYPyr[pyramidLevel].data);

    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depthSrcGradXPyr[pyramidLevel].data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depthSrcGradYPyr[pyramidLevel].data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(graySrcGradXPyr[pyramidLevel].data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(graySrcGradYPyr[pyramidLevel].data);

    if( !use_bilinear_ || pyramidLevel !=0 )
    {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(size_t i=0; i < xyz_src_transf.rows(); i++)
        {
            //Compute the 3D coordinates of the pij of the source frame
            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) ) //Compute the jacobian only for the valid points
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << i << " 3D pts " << LUT_xyz_source.block(i,0,1,3) << " transformed " << xyz.transpose() << endl;

                //Projected 3D point to the S2 sphere
                float dist = xyz.norm();

                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                    //std::cout << "warp_pixels_src(i) " << warp_pixels_src(i) << std::endl;

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = _grayTrgGradXPyr[warp_pixels_src(i)];
                    img_gradient(0,1) = _grayTrgGradYPyr[warp_pixels_src(i)];

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                    jacobiansPhoto_src.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobiansPhoto.block(i,0,1,6) << " weightedErrorPhoto " << residualsPhoto_src(i) << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_depth.at<float>(warp_pixels_src(i)) = dist;

                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = _depthTrgGradXPyr[warp_pixels_src(i)];
                    depth_gradient(0,1) = _depthTrgGradYPyr[warp_pixels_src(i)];
                    // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                    jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                    float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                    jacobiansDepth_src.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                    residualsDepth_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianDepth " << jacobiansDepth.block(i,0,1,6) << "residualsDepth_src " << residualsDepth_src(i) << std::endl;
                }
            }
            if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) ) //Compute the jacobian only for the valid points
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << point3D.transpose() << " transformed " << xyz.transpose() << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;

                // The Jacobian of the inverse pixel transformation.
                // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
                //Eigen::Vector3f xyz_inv = rotation_inv*LUT_xyz_target[i] + translation_inv;
                Eigen::Matrix<float,3,6> jacobianT36_inv;
                jacobianT36_inv.block(0,0,3,3) = -rotation_inv;
                jacobianT36_inv.block(0,3,3,1) = LUT_xyz_target(i,2)*rotation.block(0,1,3,1) - LUT_xyz_target(i,1)*rotation.block(0,2,3,1);
                jacobianT36_inv.block(0,4,3,1) = LUT_xyz_target(i,0)*rotation.block(0,2,3,1) - LUT_xyz_target(i,2)*rotation.block(0,0,3,1);
                jacobianT36_inv.block(0,5,3,1) = LUT_xyz_target(i,1)*rotation.block(0,0,3,1) - LUT_xyz_target(i,0)*rotation.block(0,1,3,1);

                // The Jacobian of the spherical projection
                Eigen::Matrix<float,2,3> jacobianProj23;
                float dist2 = dist * dist;
                float x2_z2 = dist2 - xyz(1)*xyz(1);
                float x2_z2_sqrt = sqrt(x2_z2);
                float commonDer_c = pixel_angle_inv / x2_z2;
                float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
                jacobianProj23(0,0) = commonDer_c * xyz(2);
                jacobianProj23(0,1) = 0;
                jacobianProj23(0,2) = -commonDer_c * xyz(0);
                jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
                jacobianProj23(1,1) =-commonDer_r * x2_z2;
                jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);

                Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

//                Eigen::Matrix<float,2,6> jacobianWarpRt;
//                computeJacobian26_wT_sphere_inv(xyz, LUT_xyz_target.block(i,0,1,3).transpose(), rotation, dist, pixel_angle_inv, jacobianWarpRt);

                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(i); // We keep the wrong name to 'source' to avoid duplicating more structures
                        //warped_target_grayImage.at<float>(warp_pixels_trg(i)) = grayTrgPyr[pyramidLevel].at<float>(i);

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = _graySrcGradXPyr[warp_pixels_trg(i)];
                    img_gradient(0,1) = _graySrcGradYPyr[warp_pixels_trg(i)];

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                    jacobiansPhoto_trg.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_trg(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_depth.at<float>(warp_pixels_trg(i)) = dist;

                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = _depthSrcGradXPyr[warp_pixels_trg(i)];
                    depth_gradient(0,1) = _depthSrcGradYPyr[warp_pixels_trg(i)];
                    // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,3> jacobianDepthSrc = dist_inv * xyz.transpose();
                    float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                    jacobiansDepth_trg.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt-jacobianDepthSrc*jacobianT36_inv);
                    residualsDepth_trg(i) *= weight_estim_sqrt;
                    // std::cout << "residualsDepth_trg \n " << residualsDepth_trg << std::endl;
                }
            }
        }
    }
    else
    {
        std::cout << "use_bilinear_ " << use_bilinear_ << " " << pyramidLevel << std::endl;
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i < xyz_src_transf.rows(); i++)
        {
            if( validPixelsPhoto_src(i) || validPixelsDepth_src(i) ) //Compute the jacobian only for the valid points
            {
                Eigen::Vector3f xyz = xyz_src_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << point3D.transpose() << " trnasformed " << xyz.transpose() << endl;
                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;

                //Compute the pixel jacobian
                Eigen::Matrix<float,2,6> jacobianWarpRt;
                computeJacobian26_wT_sphere(xyz, dist, pixel_angle_inv, jacobianWarpRt);

                cv::Point2f warped_pixel(warp_img_src(i,0), warp_img_src(i,1));
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_src(i)) = graySrcPyr[pyramidLevel].at<float>(i);

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = bilinearInterp( grayTrgGradXPyr[pyramidLevel], warped_pixel );
                    img_gradient(0,1) = bilinearInterp( grayTrgGradYPyr[pyramidLevel], warped_pixel );

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_src(i));
                    jacobiansPhoto_src.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_src(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    //Obtain the depth values that will be used to the compute the depth residual
                    float depth = bilinearInterp_depth( grayTrgPyr[pyramidLevel], warped_pixel ); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                    if(depth > min_depth_) // Make sure this point has depth (not a NaN)
                    {
                        if(visualizeIterations)
                            warped_depth.at<float>(warp_pixels_src(i)) = dist;

                        Eigen::Matrix<float,1,2> depth_gradient;
                        depth_gradient(0,0) = bilinearInterp_depth( depthTrgGradXPyr[pyramidLevel], warped_pixel );
                        depth_gradient(0,1) = bilinearInterp_depth( depthTrgGradYPyr[pyramidLevel], warped_pixel );

                        //Depth jacobian:
                        //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                        Eigen::Matrix<float,1,6> jacobian16_depthT = Eigen::Matrix<float,1,6>::Zero();
                        jacobian16_depthT.block(0,0,1,3) = (1 / dist) * xyz.transpose();
                        float weight_estim_sqrt = sqrt(wEstimDepth_src(i));
                        jacobiansDepth_src.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_src(i) * (depth_gradient*jacobianWarpRt - jacobian16_depthT);
                        residualsDepth_src(i) *= weight_estim_sqrt;
                        // std::cout << "residualsDepth_src \n " << residualsDepth_src << std::endl;
                    }
                }
            }
            if( validPixelsPhoto_trg(i) || validPixelsDepth_trg(i) ) //Compute the jacobian only for the valid points
            {
                //Transform the 3D point using the transformation matrix Rt
                Eigen::Vector3f xyz = xyz_trg_transf.block(i,0,1,3).transpose();
                // cout << "3D pts " << point3D.transpose() << " transformed " << xyz.transpose() << endl;

                //Project the 3D point to the S2 sphere
                float dist = xyz.norm();
                float dist_inv = 1.f / dist;

                // The Jacobian of the inverse pixel transformation.
                // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
                //Eigen::Vector3f xyz_inv = rotation_inv*LUT_xyz_target[i] + translation_inv;
                Eigen::Matrix<float,3,6> jacobianT36_inv;
                jacobianT36_inv.block(0,0,3,3) = -rotation_inv;
                jacobianT36_inv.block(0,3,3,1) = LUT_xyz_target(i,2)*rotation.block(0,1,3,1) - LUT_xyz_target(i,1)*rotation.block(0,2,3,1);
                jacobianT36_inv.block(0,4,3,1) = LUT_xyz_target(i,0)*rotation.block(0,2,3,1) - LUT_xyz_target(i,2)*rotation.block(0,0,3,1);
                jacobianT36_inv.block(0,5,3,1) = LUT_xyz_target(i,1)*rotation.block(0,0,3,1) - LUT_xyz_target(i,0)*rotation.block(0,1,3,1);

                // The Jacobian of the spherical projection
                Eigen::Matrix<float,2,3> jacobianProj23;
                float dist2 = dist * dist;
                float x2_z2 = dist2 - xyz(1)*xyz(1);
                float x2_z2_sqrt = sqrt(x2_z2);
                float commonDer_c = pixel_angle_inv / x2_z2;
                float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
                jacobianProj23(0,0) = commonDer_c * xyz(2);
                jacobianProj23(0,1) = 0;
                jacobianProj23(0,2) = -commonDer_c * xyz(0);
                jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
                jacobianProj23(1,1) =-commonDer_r * x2_z2;
                jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);

                Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

//                    Eigen::Matrix<float,2,6> jacobianWarpRt_;
//                    computeJacobian26_wT_sphere_inv(xyz, LUT_xyz_target.block(i,0,1,3).transpose(), rotation, dist, pixel_angle_inv, jacobianWarpRt_);
//                     std::cout << "jacobianWarpRt_ \n" << jacobianWarpRt_ << " jacobianWarpRt \n" << jacobianWarpRt << std::endl;
//                     mrpt::system::pause();

                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_gray.at<float>(warp_pixels_trg(i)) = graySrcPyr[pyramidLevel].at<float>(i); // We keep the wrong name to 'source' to avoid duplicating more structures
                        //warped_target_grayImage.at<float>(warp_pixels_trg(i)) = grayTrgPyr[pyramidLevel].at<float>(i);

                    Eigen::Matrix<float,1,2> img_gradient;
                    img_gradient(0,0) = _graySrcGradXPyr[warp_pixels_trg(i)];
                    img_gradient(0,1) = _graySrcGradYPyr[warp_pixels_trg(i)];

                    //Obtain the pixel values that will be used to compute the pixel residual
                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    float weight_estim_sqrt = sqrt(wEstimPhoto_trg(i));
                    jacobiansPhoto_trg.block(i,0,1,6) = ((weight_estim_sqrt * stdDevPhoto_inv) * img_gradient) * jacobianWarpRt;
                    residualsPhoto_trg(i) *= weight_estim_sqrt;
                    // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
                    // mrpt::system::pause();
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    if(visualizeIterations)
                        warped_depth.at<float>(warp_pixels_trg(i)) = dist;

                    Eigen::Matrix<float,1,2> depth_gradient;
                    depth_gradient(0,0) = _depthSrcGradXPyr[warp_pixels_trg(i)];
                    depth_gradient(0,1) = _depthSrcGradYPyr[warp_pixels_trg(i)];
                    // std::cout << "depth_gradient \n " << depth_gradient << std::endl;

                    //Depth jacobian:
                    //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                    Eigen::Matrix<float,1,3> jacobianDepthSrc = dist_inv * xyz.transpose();
                    float weight_estim_sqrt = sqrt(wEstimDepth_trg(i));
                    jacobiansDepth_trg.block(i,0,1,6) = weight_estim_sqrt * stdDevError_inv_trg(i) * (depth_gradient*jacobianWarpRt-jacobianDepthSrc*jacobianT36_inv);
                    residualsDepth_trg(i) *= weight_estim_sqrt;
                    // std::cout << "residualsDepth_trg \n " << residualsDepth_trg << std::endl;
                }
            }
        }
    }

    // Compute hessian and gradient
    hessian.setZero();
    gradient.setZero();
    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
    {
        updateHessianAndGradient(jacobiansPhoto_src, residualsPhoto_src, validPixelsPhoto_src);
        updateHessianAndGradient(jacobiansPhoto_src, residualsPhoto_src, validPixelsPhoto_src);
    }
    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
    {
        updateHessianAndGradient(jacobiansDepth_trg, residualsDepth_trg, validPixelsDepth_trg);
        updateHessianAndGradient(jacobiansDepth_trg, residualsDepth_trg, validPixelsDepth_trg);
    }

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << pyramidLevel << " calcHessGrad_sphere_bidirectional took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
 * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
 * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
 * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
 */
void RegisterDense::register360_bidirectional(const Eigen::Matrix4f pose_guess, costFuncType method , const int occlusion )
{
    //    std::cout << "register360_bidirectional " << std::endl;
//#if PRINT_PROFILING
    double time_start = pcl::getTime();\
    //for(size_t ii=0; ii<100; ii++)
    {
//#endif

    thresDepthOutliers = 0.3;

    num_iterations.resize(nPyrLevels); // Store the number of iterations

    Eigen::Matrix4f pose_estim_temp, pose_estim = pose_guess;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t nRows = graySrcPyr[pyramidLevel].rows;
        const size_t nCols = graySrcPyr[pyramidLevel].cols;
        const size_t imgSize = nRows*nCols;
        const float pixel_angle = 2*PI/nCols;

        if(sensor_type == RGBD360_INDOOR)
        {
            // HACK: Mask the joints between the different images to avoid the high gradients that are the res of using auto-shutter for each camera
            int nPixBlackBand = 2;
            int width_sensor = nCols / 8;
            for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            {
                if( graySrcGradXPyr[pyramidLevel].cols > 400 )
                {
                    nPixBlackBand = 4;
                    if( graySrcGradXPyr[pyramidLevel].cols > 800 )
                        nPixBlackBand = 8;
                }

                cv::Rect region_of_interest = cv::Rect((0.5f+sensor_id)*width_sensor-nPixBlackBand/2, 0, nPixBlackBand, nRows);
                //                cv::Mat image_roi = grayTrgGradXPyr[pyramidLevel](region_of_interest);
                //                image_roi = cv::Mat::zeros(image_roi.cols, image_roi.rows, image_roi.type());
                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradXPyr[pyramidLevel].type());
                //                grayTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat(nRows, 20, grayTrgGradXPyr[pyramidLevel].type(), cv::Scalar(255.f));
                grayTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, grayTrgGradYPyr[pyramidLevel].type());
                depthTrgGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradXPyr[pyramidLevel].type());
                depthTrgGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthTrgGradYPyr[pyramidLevel].type());

                graySrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradXPyr[pyramidLevel].type());
                graySrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, graySrcGradYPyr[pyramidLevel].type());
                depthSrcGradXPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradXPyr[pyramidLevel].type());
                depthSrcGradYPyr[pyramidLevel](region_of_interest) = cv::Mat::zeros(nRows,nPixBlackBand, depthSrcGradYPyr[pyramidLevel].type());
            }
            //            cv::imshow("test_grad", grayTrgGradXPyr[pyramidLevel]);
            //        cv::waitKey(0);
        }

        // Make LUT to store the values of the 3D points of the source sphere
        if(use_salient_pixels_)
        {
            getSalientPoints_sphere(LUT_xyz_source, validPixels_src,
                                    depthSrcPyr[pyramidLevel], depthSrcGradXPyr[pyramidLevel], depthSrcGradYPyr[pyramidLevel],
                                    graySrcPyr[pyramidLevel], graySrcGradXPyr[pyramidLevel], graySrcGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth

            getSalientPoints_sphere(LUT_xyz_target, validPixels_trg,
                                    depthTrgPyr[pyramidLevel], depthTrgGradXPyr[pyramidLevel], depthTrgGradYPyr[pyramidLevel],
                                    grayTrgPyr[pyramidLevel], grayTrgGradXPyr[pyramidLevel], grayTrgGradYPyr[pyramidLevel]
                                    ); // TODO extend this function to employ only depth
        }
        else
        {
            computeSphereXYZ_sse(depthSrcPyr[pyramidLevel], LUT_xyz_source, validPixels_src);
            //computeSphereXYZ(depthTrgPyr[pyramidLevel], LUT_xyz_target, validPixels_trg);
            computeSphereXYZ_sse(depthTrgPyr[pyramidLevel], LUT_xyz_target, validPixels_trg);
        }

        double lambda = 1e0; // Levenberg-Marquardt (LM) lambda
        double step = 5; // Update step
        unsigned LM_maxIters = 1;

        int it = 0, maxIters = 10;
        double tol_residual = 1e-3;
        double tol_update = 1e-3;
        Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
        double error = errorDense_sphere(pyramidLevel, pose_estim, method) + errorDenseInv_sphere(pyramidLevel, pose_estim, method);

        double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        //            std::cout << "pose_estim \n " << pose_estim << std::endl;
        std::cout << "Level " << pyramidLevel << " imgSize " << validPixels_src.rows() << "+" << validPixels_trg.rows() << "/" << imgSize << " error " << error << std::endl;
#endif
        while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
        {
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            cv::TickMeter tm;tm.start();
//#endif

            hessian.setZero();
            gradient.setZero();
            //calcHessGrad_sphere_bidirectional(pyramidLevel, pose_estim, method);
            calcHessGrad_sphere(pyramidLevel, pose_estim, method);
            calcHessGradInv_sphere(pyramidLevel, pose_estim, method);

            if(visualizeIterations)
            {
                cv::Mat imgDiff, weightedError;
                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    cv::imshow("trg", grayTrgPyr[pyramidLevel]);
                    cv::imshow("src", graySrcPyr[pyramidLevel]);

                    imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
                    //                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
                    cv::imshow("intensityDiff", imgDiff);
                    cv::imshow("warp", warped_gray);
                    //                    cv::Mat imgDiffsave;
                    //                    imgDiff.convertTo(imgDiffsave, CV_8UC1, 255);
                    //                    cv::imwrite( mrpt::format("/home/edu/weightedError.png"), imgDiffsave);
                }
                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                {
                    weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
                    cv::imshow("weightedError", weightedError);
                }
                cv::waitKey(0);
            }

            if(hessian.rank() != 6)
//            if((hessian + lambda*getDiagonalMatrix(hessian)).rank() != 6)
            {
                std::cout << "\t The problem is ILL-POSED \n";
                //                    std::cout << "hessian \n" << hessian << std::endl;
                //                    std::cout << "gradient \n" << gradient.transpose() << std::endl;
                registered_pose_ = pose_estim;
                return;
            }

            // Compute the pose update
            update_pose = -hessian.inverse() * gradient;
//            update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
            Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
            //                update_pose_d.block(0,0,3,1) = -update_pose_d.block(0,0,3,1);
            pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() * pose_estim;

            double new_error = errorDense_sphere(pyramidLevel, pose_estim_temp, method) + errorDenseInv_sphere(pyramidLevel, pose_estim_temp, method);
            diff_error = error - new_error;

            if(diff_error > 0.0)
            {
                //                cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
                lambda /= step;
                pose_estim = pose_estim_temp;
                error = new_error;
                it = it+1;
            }
//            else
//            {
//                unsigned LM_it = 0;
//                while(LM_it < LM_maxIters && diff_error < 0)
//                {
//                    lambda = lambda * step;
//                    update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
//                    Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
//                    pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
//                    double new_error = errorDense_sphere_bidirectional(pyramidLevel, pose_estim_temp, method);
//                    diff_error = error - new_error;
//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//                    cout << "diff_error LM " << diff_error << endl;
//#endif
//                    if(diff_error > 0.0)
//                    {
//                        cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
//                        pose_estim = pose_estim_temp;
//                        error = new_error;
//                        it = it+1;
//                    }
//                    LM_it = LM_it + 1;
//                }
//            }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            tm.stop();
//            std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
            std::cout << "update_pose \n" << update_pose.transpose() << " norm " << update_pose.norm() << " tol_update " << tol_update << " \n ";// << mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d), true).getHomogeneousMatrixVal().cast<float>() << std::endl;
            cout << "diff_error " << diff_error << " tol_residual " << tol_residual << endl;
#endif

            //                if(visualizeIterations)
            //                {
            //                    cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
            //                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_gray, imgDiff);
            ////                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_gray.at<float>(20,20) << std::endl;
            //                    cv::imshow("optimize::imgDiff", imgDiff);

            //                    cv::imshow("orig", grayTrgPyr[pyramidLevel]);
            //                    cv::imshow("warp", warped_gray);

            //                    cv::Mat weightedError = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
            //                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_depth, weightedError);
            //                    cv::imshow("weightedError", weightedError);

            //                    cv::waitKey(0);
            //                }
        }

        num_iterations[pyramidLevel] = it;
    }

    //#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Iterations: "; //<< std::endl;
    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        std::cout << num_iterations[pyramidLevel] << " ";
    std::cout << std::endl;
    //#endif

    registered_pose_ = pose_estim;

//#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << "Dense alignment bidirectional 360 took " << (time_end - time_start)*1000 << " ms. \n";
//#endif
}

/*! Compute the unit sphere for the given spherical image dimmensions. This serves as a LUT to speed-up calculations. */
void RegisterDense::computeUnitSphere()
{
    const size_t nRows = graySrc.rows;
    const size_t nCols = graySrc.cols;

    // Make LUT to store the values of the 3D points of the source sphere
    Eigen::MatrixXf unit_sphere;
    unit_sphere.resize(nRows*nCols,3);
    const float pixel_angle = 2*PI/nCols;
    std::vector<float> v_sinTheta(nCols);
    std::vector<float> v_cosTheta(nCols);
    for(int c=0;c<nCols;c++)
    {
        float theta = c*pixel_angle;
        v_sinTheta[c] = sin(theta);
        v_cosTheta[c] = cos(theta);
    }
    const float half_height = 0.5*nRows-0.5;
    for(int r=0;r<nRows;r++)
    {
        float phi = (half_height-r)*pixel_angle;
        float sin_phi = sin(phi);
        float cos_phi = cos(phi);

        for(int c=0;c<nCols;c++)
        {
            int i = r*nCols + c;
            unit_sphere(i,0) = sin_phi;
            unit_sphere(i,1) = -cos_phi*v_sinTheta[c];
            unit_sphere(i,2) = -cos_phi*v_cosTheta[c];
        }
    }
}

/*! Align depth frames applying ICP in different pyramid scales. */
double RegisterDense::alignPyramidICP(Eigen::Matrix4f poseGuess)
{
    //        vector<pcl::PointCloud<PointT> > pyrCloudSrc(nPyrLevels);
    //        vector<pcl::PointCloud<PointT> > pyrCloudTrg(nPyrLevels);

    // ICP alignement
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;

    icp.setMaxCorrespondenceDistance (0.3);
    icp.setMaximumIterations (10);
    icp.setTransformationEpsilon (1e-6);
    //  icp.setEuclideanFitnessEpsilon (1);
    icp.setRANSACOutlierRejectionThreshold (0.1);

    for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
    {
        const size_t height = depthSrcPyr[pyramidLevel].rows;
        const size_t width = depthSrcPyr[pyramidLevel].cols;

        const float res_factor_VGA = width / 640.0;
        const float focal_length = 525 * res_factor_VGA;
        const float inv_fx = 1.f/focal_length;
        const float inv_fy = 1.f/focal_length;
        const float ox = width/2 - 0.5;
        const float oy = height/2 - 0.5;

        pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloudPtr(new pcl::PointCloud<pcl::PointXYZ>());
        srcCloudPtr->height = height;
        srcCloudPtr->width = width;
        srcCloudPtr->is_dense = false;
        srcCloudPtr->points.resize(height*width);

        pcl::PointCloud<pcl::PointXYZ>::Ptr trgCloudPtr(new pcl::PointCloud<pcl::PointXYZ>());
        trgCloudPtr->height = height;
        trgCloudPtr->width = width;
        trgCloudPtr->is_dense = false;
        trgCloudPtr->points.resize(height*width);

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for( int y = 0; y < height; y++ )
        {
            for( int x = 0; x < width; x++ )
            {
                float z = depthSrcPyr[pyramidLevel].at<float>(y,x); //convert from milimeters to meters
                //std::cout << "Build " << z << std::endl;
                //if(z>0 && z>=min_depth_ && z<=max_depth_) //If the point has valid depth information assign the 3D point to the point cloud
                if(z>=min_depth_ && z<=max_depth_) //If the point has valid depth information assign the 3D point to the point cloud
                {
                    srcCloudPtr->points[width*y+x].x = (x - ox) * z * inv_fx;
                    srcCloudPtr->points[width*y+x].y = (y - oy) * z * inv_fy;
                    srcCloudPtr->points[width*y+x].z = z;
                }
                else //else, assign a NAN value
                {
                    srcCloudPtr->points[width*y+x].x = std::numeric_limits<float>::quiet_NaN ();
                    srcCloudPtr->points[width*y+x].y = std::numeric_limits<float>::quiet_NaN ();
                    srcCloudPtr->points[width*y+x].z = std::numeric_limits<float>::quiet_NaN ();
                }

                z = depthTrgPyr[pyramidLevel].at<float>(y,x); //convert from milimeters to meters
                //std::cout << "Build " << z << std::endl;
                //if(z>0 && z>=min_depth_ && z<=max_depth_) //If the point has valid depth information assign the 3D point to the point cloud
                if(z>=min_depth_ && z<=max_depth_) //If the point has valid depth information assign the 3D point to the point cloud
                {
                    trgCloudPtr->points[width*y+x].x = (x - ox) * z * inv_fx;
                    trgCloudPtr->points[width*y+x].y = (y - oy) * z * inv_fy;
                    trgCloudPtr->points[width*y+x].z = z;
                }
                else //else, assign a NAN value
                {
                    trgCloudPtr->points[width*y+x].x = std::numeric_limits<float>::quiet_NaN ();
                    trgCloudPtr->points[width*y+x].y = std::numeric_limits<float>::quiet_NaN ();
                    trgCloudPtr->points[width*y+x].z = std::numeric_limits<float>::quiet_NaN ();
                }
            }
        }

        // Remove NaN points
        pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloudPtr_(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr trgCloudPtr_(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<int> nan_indices;
        pcl::removeNaNFromPointCloud(*srcCloudPtr,*srcCloudPtr_,nan_indices);
        //std::cout << " pts " << srcCloudPtr->size() << " pts " << srcCloudPtr_->size() << std::endl;
        pcl::removeNaNFromPointCloud(*trgCloudPtr,*trgCloudPtr_,nan_indices);

        // ICP registration:
        icp.setInputSource(srcCloudPtr_);
        icp.setInputTarget(trgCloudPtr_);
        pcl::PointCloud<pcl::PointXYZ>::Ptr alignedICP(new pcl::PointCloud<pcl::PointXYZ>);
        icp.align(*alignedICP, poseGuess);
        poseGuess = icp.getFinalTransformation();

        // std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
        //std::cout << pyramidLevel << " PyrICP has converged: " << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    }
    registered_pose_ = poseGuess;
}


///*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
//        This is done following the work in:
//        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
//        in Computer Vision Workshops (ICCV Workshops), 2011. */
//double RegisterDense::calcDenseError_rgbd360_singlesensor(const int &pyramidLevel,
//                                                 const Eigen::Matrix4f poseGuess,
//                                                 const Eigen::Matrix4f &poseCamRobot,
//                                                 costFuncType method )
//{
//    double error2 = 0.0; // Squared error

//    const size_t nRows = graySrcPyr[pyramidLevel].rows;
//    const size_t nCols = graySrcPyr[pyramidLevel].cols;

//    const float scaleFactor = 1.0/pow(2,pyramidLevel);
//    fx = cameraMatrix(0,0)*scaleFactor;
//    fy = cameraMatrix(1,1)*scaleFactor;
//    ox = cameraMatrix(0,2)*scaleFactor;
//    oy = cameraMatrix(1,2)*scaleFactor;
//    const float inv_fx = 1./fx;
//    const float inv_fy = 1./fy;

//    const Eigen::Matrix4f poseCamRobot_inv = poseCamRobot.inverse();
//    const Eigen::Matrix4f registered_pose_Cam = poseCamRobot_inv*poseGuess*poseCamRobot;

//    double weight_estim; // The weight computed by an M-estimator
//    const float stdDevPhoto_inv = 1./stdDevPhoto;
//    const float stdDevDepth_inv = 1./stdDevDepth;

//    if(use_salient_pixels_)
//    {
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:error2)
//#endif
//        for (int i=0; i < vSalientPixels[pyramidLevel].size(); i++)
//        {
//            int r = vSalientPixels[pyramidLevel][i] / nCols;
//            int c = vSalientPixels[pyramidLevel][i] % nCols;
//            //Compute the 3D coordinates of the pij of the source frame
//            Eigen::Vector4f point3D;
//            point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//            if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//            {
//                point3D(0)=(c - ox) * point3D(2) * inv_fx;
//                point3D(1)=(r - oy) * point3D(2) * inv_fy;
//                point3D(3)=1;

//                //Transform the 3D point using the transformation matrix Rt
//                Eigen::Vector4f xyz = registered_pose_Cam*point3D;

//                //Project the 3D point to the 2D plane
//                double inv_transf_z = 1.f/xyz(2);
//                double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//                transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//                int transformed_r_int = round(transformed_r);
//                int transformed_c_int = round(transformed_c);

//                //Asign the intensity value to the warped image and compute the difference between the transformed
//                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                        (transformed_c_int>=0 && transformed_c_int < nCols) )
//                {
//                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        //Obtain the pixel values that will be used to compute the pixel residual
//                        float pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                        float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                        float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                        float weight_estim = weightMEstimator(weightedError); // Apply M-estimator weighting
//                        float weightedErrorPhoto = weight_estim * weightedError;
//                        //                            if(weightedError2 > varianceRegularization)
//                        //                            {
//                        ////                                float weightedError2_norm = sqrt(weightedError2);
//                        ////                                double weight = sqrt(2*stdDevReg*weightedError2_norm-varianceRegularization) / weightedError2_norm;
//                        //                                weightedError2 = 2*stdDevReg*sqrt(weightedError2)-varianceRegularization;
//                        //                            }
//                        error2 += weightedErrorPhoto*weightedErrorPhoto;
//                    }
//                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                        {
//                            //Obtain the depth values that will be used to the compute the depth residual
//                            float depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
//                            float weightedError = depth - depth1;
//                            //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//                            //                                float stdDevError = stdDevDepth*depth1;
//                            float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                            float weight_estim = weightMEstimator(weightedError);
//                            float weightedErrorDepth = weight_estim * weightedError;
//                            error2 += weightedErrorDepth*weightedErrorDepth;
//                        }
//                    }
//                }
//            }
//        }
//    }
//    else
//    {
//        std::cout << "calcDenseError_rgbd360_singlesensor error2 " << error2 << std::endl;
//#if ENABLE_OPENMP
//#pragma omp parallel for reduction (+:error2)
//#endif
//        for (int r=0;r<nRows;r++)
//        {
//            for (int c=0;c<nCols;c++)
//            {
//                // int i = nCols*r+c; //vector index for sorting salient pixels

//                //Compute the 3D coordinates of the pij of the source frame
//                Eigen::Vector4f point3D;
//                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//                if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//                {
//                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
//                    point3D(1)=(r - oy) * point3D(2) * inv_fy;
//                    point3D(3)=1;

//                    //Transform the 3D point using the transformation matrix Rt
//                    Eigen::Vector4f xyz = registered_pose_Cam*point3D;

//                    //Project the 3D point to the 2D plane
//                    double inv_transf_z = 1.f/xyz(2);
//                    double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                    transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//                    transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//                    int transformed_r_int = round(transformed_r);
//                    int transformed_c_int = round(transformed_c);

//                    //Asign the intensity value to the warped image and compute the difference between the transformed
//                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                            (transformed_c_int>=0 && transformed_c_int < nCols) )
//                    {
//                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            //Obtain the pixel values that will be used to compute the pixel residual
//                            float pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                            float intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                            float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                            float weight_estim = weightMEstimator(weightedError); // Apply M-estimator weighting
//                            float weightedErrorPhoto = weight_estim * weightedError;
//                            //                            if(weightedError2 > varianceRegularization)
//                            //                            {
//                            ////                                float weightedError2_norm = sqrt(weightedError2);
//                            ////                                double weight = sqrt(2*stdDevReg*weightedError2_norm-varianceRegularization) / weightedError2_norm;
//                            //                                weightedError2 = 2*stdDevReg*sqrt(weightedError2)-varianceRegularization;
//                            //                            }
//                            error2 += weightedErrorPhoto*weightedErrorPhoto;
//                            // std::cout << "error2 " << error2 << " weightedErrorPhoto " << weightedErrorPhoto << " " << weight_estim << " " << weightedError << std::endl;
//                        }
//                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            float depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                            {
//                                //Obtain the depth values that will be used to the compute the depth residual
//                                float depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
//                                float weightedError = depth - depth1;
//                                //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//                                //float stdDevError = stdDevDepth*depth1;
//                                float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                                float weight_estim = weightMEstimator(weightedError);
//                                float weightedErrorDepth = weight_estim * weightedError;
//                                error2 += weightedErrorDepth*weightedErrorDepth;
//                            }
//                        }
//                        //                            std::cout << " error2 " << error2 << std::endl;
//                    }
//                }
//            }
//        }
//    }

//    return error2;
//}


///*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        This is done following the work in:
//        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
//        in Computer Vision Workshops (ICCV Workshops), 2011. */
//void RegisterDense::calcHessGrad_rgbd360_singlesensor( const int &pyramidLevel,
//                                                  const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
//                                                  const Eigen::Matrix4f &poseCamRobot, // The pose of the camera wrt to the Robot (fixed beforehand through calibration) // Maybe calibration can be computed at the same time
//                                                  costFuncType method )
//{
//    const size_t nRows = graySrcPyr[pyramidLevel].rows;
//    const size_t nCols = graySrcPyr[pyramidLevel].cols;

//    const double scaleFactor = 1.0/pow(2,pyramidLevel);
//    const double fx = cameraMatrix(0,0)*scaleFactor;
//    const double fy = cameraMatrix(1,1)*scaleFactor;
//    const double ox = cameraMatrix(0,2)*scaleFactor;
//    const double oy = cameraMatrix(1,2)*scaleFactor;
//    const double inv_fx = 1./fx;
//    const double inv_fy = 1./fy;

//    hessian = Eigen::Matrix<float,6,6>::Zero();
//    gradient = Eigen::Matrix<float,6,1>::Zero();

//    double weight_estim; // The weight computed by an M-estimator
//    //        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];
//    const float stdDevPhoto_inv = 1./stdDevPhoto;
//    const float stdDevDepth_inv = 1./stdDevDepth;

//    const Eigen::Matrix4f poseCamRobot_inv = poseCamRobot.inverse();
//    const Eigen::Matrix4f registered_pose_Cam2 = poseGuess*poseCamRobot;

//    //    std::cout << "poseCamRobot \n" << poseCamRobot << std::endl;

//    if(visualizeIterations)
//    {
//        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_gray = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
//        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//            warped_depth = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
//        //            cout << "type__ " << grayTrgPyr[pyramidLevel].type() << " " << warped_gray.type() << endl;
//    }

//    if(use_salient_pixels_)
//    {
//        //#if ENABLE_OPENMP
//        //#pragma omp parallel for
//        //#endif
//        for (int i=0; i < vSalientPixels[pyramidLevel].size(); i++)
//        {
//            int r = vSalientPixels[pyramidLevel][i] / nCols;
//            int c = vSalientPixels[pyramidLevel][i] % nCols;
//            // int i = nCols*r+c; //vector index

//            //Compute the 3D coordinates of the pij of the source frame
//            Eigen::Vector4f point3D;
//            point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//            if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//            {
//                point3D(0)=(c - ox) * point3D(2) * inv_fx;
//                point3D(1)=(r - oy) * point3D(2) * inv_fy;
//                point3D(3)=1;

//                //Transform the 3D point using the transformation matrix Rt
//                //                    Eigen::Vector4f point3D_robot = poseCamRobot*point3D;
//                Eigen::Vector4f point3D_robot2 = registered_pose_Cam2*point3D;
//                Eigen::Vector4f xyz = poseCamRobot_inv*point3D_robot2;
//                //                std::cout << "xyz " << xyz.transpose() << std::endl;

//                //Project the 3D point to the 2D plane
//                double inv_transf_z = 1.f/xyz(2);
//                double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//                transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//                int transformed_r_int = round(transformed_r);
//                int transformed_c_int = round(transformed_c);
//                //                std::cout << "transformed_r_int " << transformed_r_int << " transformed_c_int " << transformed_c_int << std::endl;

//                //Asign the intensity value to the warped image and compute the difference between the transformed
//                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                        (transformed_c_int>=0 && transformed_c_int < nCols) )
//                {
//                    //Compute the pixel jacobian
//                    Eigen::Matrix<float,3,6> jacobianT36;
//                    jacobianT36.block(0,0,3,3) = Eigen::Matrix<float,3,3>::Identity();
//                    Eigen::Vector3f rotatedPoint3D = point3D_robot2.block(0,0,3,1);// - poseGuess.block(0,3,3,1); // TODO: make more efficient
//                    jacobianT36.block(0,3,3,3) = -skew( rotatedPoint3D );
//                    jacobianT36 = poseCamRobot_inv.block(0,0,3,3) * jacobianT36;

//                    Eigen::Matrix<float,2,3> jacobianProj23;
//                    //Derivative with respect to x
//                    jacobianProj23(0,0)=fx*inv_transf_z;
//                    jacobianProj23(1,0)=0;
//                    //Derivative with respect to y
//                    jacobianProj23(0,1)=0;
//                    jacobianProj23(1,1)=fy*inv_transf_z;
//                    //Derivative with respect to z
//                    jacobianProj23(0,2)=-fx*xyz(0)*inv_transf_z*inv_transf_z;
//                    jacobianProj23(1,2)=-fy*xyz(1)*inv_transf_z*inv_transf_z;

//                    Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36;


//                    float pixel_src, intensity, depth1, depth;
//                    double weightedErrorPhoto, weightedErrorDepth;
//                    Eigen::Matrix<float,1,6> jacobianPhoto, jacobianDepth;

//                    if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                        if(visualizeIterations)
//                            warped_gray.at<float>(transformed_r_int,transformed_c_int) = pixel_src;

//                        //Obtain the pixel values that will be used to compute the pixel residual
//                        intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                        float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                        float weight_estim = weightMEstimator(weightedError);
//                        weightedErrorPhoto = weight_estim * weightedError;

//                        //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
//                        Eigen::Matrix<float,1,2> img_gradient;
//                        img_gradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
//                        img_gradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                        jacobianPhoto = weight_estim * img_gradient*jacobianWarpRt;
//                    }
//                    if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                    {
//                        //Obtain the depth values that will be used to the compute the depth residual
//                        depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                        if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                        {
//                            depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
//                            if(visualizeIterations)
//                                warped_depth.at<float>(transformed_r_int,transformed_c_int) = depth1;

//                            float weightedError = depth - depth1;
//                            //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//                            //float stdDevError = stdDevDepth*depth1;
//                            float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                            float weight_estim = weightMEstimator(weightedError);
//                            weightedErrorDepth = weight_estim * weightedError;

//                            //Depth jacobian:
//                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
//                            Eigen::Matrix<float,1,2> depth_gradient;
//                            depth_gradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            depth_gradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);

//                            Eigen::Matrix<float,1,6> jacobianRt_z;
//                            jacobianT36.block(2,0,1,6);
//                            jacobianDepth = weight_estim * (depth_gradient*jacobianWarpRt-jacobianRt_z);
//                            // cout << "jacobianDepth " << jacobianDepth << " weightedErrorDepth " << weightedErrorDepth << endl;
//                        }
//                    }

//                    //Assign the pixel residual and jacobian to its corresponding row
//                    //#if ENABLE_OPENMP
//                    //#pragma omp critical
//                    //#endif
//                    {
//                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            // Photometric component
//                            //                                hessian += jacobianPhoto.transpose()*jacobianPhoto / varPhoto;
//                            hessian += jacobianPhoto.transpose()*jacobianPhoto;
//                            gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
//                        }
//                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                            {
//                                // Depth component (Plane ICL like)
//                                hessian += jacobianDepth.transpose()*jacobianDepth;
//                                gradient += jacobianDepth.transpose()*weightedErrorDepth;
//                            }
//                    }
//                }
//            }
//        }
//    }
//    else // Use all points
//    {
//        //#if ENABLE_OPENMP
//        //#pragma omp parallel for
//        //#endif
//        for (int r=0;r<nRows;r++)
//        {
//            for (int c=0;c<nCols;c++)
//            {
//                //                int i = nCols*r+c; //vector index

//                //Compute the 3D coordinates of the pij of the source frame
//                Eigen::Vector4f point3D;
//                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//                if(min_depth_ < point3D(2) && point3D(2) < max_depth_) //Compute the jacobian only for the valid points
//                {
//                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
//                    point3D(1)=(r - oy) * point3D(2) * inv_fy;
//                    point3D(3)=1;

//                    //Transform the 3D point using the transformation matrix Rt
//                    Eigen::Vector4f point3D_robot = poseCamRobot*point3D;
//                    Eigen::Vector4f point3D_robot2 = poseGuess*point3D_robot;
//                    Eigen::Vector4f xyz = poseCamRobot_inv*point3D_robot2;
//                    //                    Eigen::Vector3f  xyz = poseGuess.block(0,0,3,3)*point3D;
//                    //                    std::cout << "xyz " << xyz.transpose() << std::endl;

//                    //Project the 3D point to the 2D plane
//                    double inv_transf_z = 1.f/xyz(2);
//                    double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                    transformed_c = (xyz(0) * fx)*inv_transf_z + ox; //transformed x (2D)
//                    transformed_r = (xyz(1) * fy)*inv_transf_z + oy; //transformed y (2D)
//                    int transformed_r_int = round(transformed_r);
//                    int transformed_c_int = round(transformed_c);
//                    //                std::cout << "transformed_r_int " << transformed_r_int << " transformed_c_int " << transformed_c_int << std::endl;

//                    //Asign the intensity value to the warped image and compute the difference between the transformed
//                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
//                            (transformed_c_int>=0 && transformed_c_int < nCols) )
//                    {
//                        //Compute the pixel jacobian
//                        Eigen::Matrix<float,3,6> jacobianT36;
//                        jacobianT36.block(0,0,3,3) = Eigen::Matrix<float,3,3>::Identity();
//                        Eigen::Vector3f rotatedPoint3D = point3D_robot2.block(0,0,3,1);// - poseGuess.block(0,3,3,1); // TODO: make more efficient
//                        jacobianT36.block(0,3,3,3) = -skew( rotatedPoint3D );
//                        jacobianT36 = poseCamRobot_inv.block(0,0,3,3) * jacobianT36;

//                        Eigen::Matrix<float,2,3> jacobianProj23;
//                        //Derivative with respect to x
//                        jacobianProj23(0,0)=fx*inv_transf_z;
//                        jacobianProj23(1,0)=0;
//                        //Derivative with respect to y
//                        jacobianProj23(0,1)=0;
//                        jacobianProj23(1,1)=fy*inv_transf_z;
//                        //Derivative with respect to z
//                        jacobianProj23(0,2)=-fx*xyz(0)*inv_transf_z*inv_transf_z;
//                        jacobianProj23(1,2)=-fy*xyz(1)*inv_transf_z*inv_transf_z;

//                        Eigen::Matrix<float,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36;


//                        float pixel_src, intensity, depth1, depth;
//                        double weightedErrorPhoto, weightedErrorDepth;
//                        Eigen::Matrix<float,1,6> jacobianPhoto, jacobianDepth;

//                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            pixel_src = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                            if(visualizeIterations)
//                                warped_gray.at<float>(transformed_r_int,transformed_c_int) = pixel_src;

//                            Eigen::Matrix<float,1,2> img_gradient;
//                            img_gradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
//                            img_gradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            if(fabs(img_gradient(0,0)) < thres_saliency_gray_ && fabs(img_gradient(0,1)) < thres_saliency_gray_)
//                                continue;

//                            //Obtain the pixel values that will be used to compute the pixel residual
//                            intensity = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                            float weightedError = (intensity - pixel_src)*stdDevPhoto_inv;
//                            float weight_estim = weightMEstimator(weightedError);
//                            weightedErrorPhoto = weight_estim * weightedError;

//                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
//                            jacobianPhoto = weight_estim * img_gradient*jacobianWarpRt;

//                            // std::cout << "weight_estim " << weight_estim << " img_gradient " << img_gradient << "\njacobianWarpRt\n" << jacobianWarpRt << std::endl;
//                            // std::cout << "jacobianPhoto " << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
//                            // std::cout << "hessian " << hessian << std::endl;
//                        }
//                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                        {
//                            //Obtain the depth values that will be used to the compute the depth residual
//                            depth = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                            if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                                depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
//                            {
//                                if(visualizeIterations)
//                                    warped_depth.at<float>(transformed_r_int,transformed_c_int) = depth1;

//                                Eigen::Matrix<float,1,2> depth_gradient;
//                                depth_gradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                                depth_gradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
//                                if(fabs(depth_gradient(0,0)) < thres_saliency_depth_ && fabs(depth_gradient(0,1)) < thres_saliency_depth_)
//                                    continue;

//                                float weightedError = depth - depth1;
//                                //float weight_estim = weightMEstimator(weightedError,stdDevDepth)*stdDevPhoto_inv;
//                                //float stdDevError = stdDevDepth*depth1;
//                                float stdDevError = std::max (stdDevDepth*(depth1*depth1+depth*depth), 2*stdDevDepth);
//                                float weight_estim = weightMEstimator(weightedError);
//                                weightedErrorDepth = weight_estim * weightedError;

//                                //Depth jacobian:
//                                //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
//                                Eigen::Matrix<float,1,6> jacobianRt_z;
//                                jacobianT36.block(2,0,1,6);
//                                jacobianDepth = weight_estim * (depth_gradient*jacobianWarpRt-jacobianRt_z);
//                                // cout << "jacobianDepth " << jacobianDepth << " weightedErrorDepth " << weightedErrorDepth << endl;
//                            }
//                        }

//                        //Assign the pixel residual and jacobian to its corresponding row
//                        //    #if ENABLE_OPENMP
//                        //    #pragma omp critical
//                        //    #endif
//                        {
//                            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                            {
//                                // Photometric component
//                                //                                    std::cout << "c " << c << " r " << r << std::endl;
//                                //                                    std::cout << "hessian \n" << hessian << std::endl;
//                                hessian += jacobianPhoto.transpose()*jacobianPhoto;
//                                gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
//                                //                                    std::cout << "jacobianPhoto \n" << jacobianPhoto << " weightedErrorPhoto " << weightedErrorPhoto << std::endl;
//                                //                                    std::cout << "hessian \n" << hessian << std::endl;
//                            }
//                            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                                if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                                {
//                                    // Depth component (Plane ICL like)
//                                    hessian += jacobianDepth.transpose()*jacobianDepth;
//                                    gradient += jacobianDepth.transpose()*weightedErrorDepth;
//                                }
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

/*! Compute the 3D points XYZ by multiplying the unit sphere by the spherical depth image. */
void RegisterDense::computeSphereXYZ(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels)
{
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    for(size_t ii=0; ii<100; ii++)
    {
#endif

    const float half_width = nCols/2;
    const float pixel_angle = 2*PI/nCols;

    float phi_start;
    if(sensor_type == RGBD360_INDOOR)
        phi_start = -(0.5*nRows-0.5)*pixel_angle;
    else
        phi_start = float(174-512)/512 *PI/2 + 0.5*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)


    xyz.resize(imgSize,3);

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
        //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
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
                //std::cout << " depth1 " << depth1 << " phi " << phi << " v_sinTheta[c] " << v_sinTheta[c] << std::endl;
                xyz(i,0) = depth1 * v_cosPhi[r] * v_sinTheta[c];
                xyz(i,1) = depth1 * v_sinPhi[r];
                xyz(i,2) = depth1 * v_cosPhi[r] * v_cosTheta[c];
                //std::cout << " xyz " << xyz [i].transpose() << " xyz_eigen " << xyz_eigen.block(c*nRows+r,0,1,3) << std::endl;
                //mrpt::system::pause();
            }
            else
                validPixels(i) = 0;
        }
    }

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::computeSphereXYZ_sse " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Compute the 3D points XYZ by multiplying the unit sphere by the spherical depth image. */
void RegisterDense::computeSphereXYZ_sse(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels)
{
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    assert(nCols % 4 == 0); // Make sure that the image columns are aligned
    assert(nRows % 2 == 0);
    const float half_width = nCols/2;
    const float pixel_angle = 2*PI/nCols;
//    float phi_start;
//    if(sensor_type == RGBD360_INDOOR)
//        phi_start = -(0.5*nRows-0.5)*pixel_angle;
//    else
//        phi_start = float(174-512)/512 *PI/2 + 0.5*pixel_angle; // The images must be 640 pixels height to compute the pyramids efficiently (we substract 8 pixels from the top and 7 from the lower part)

    xyz.resize(imgSize,3);
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
        //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
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

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::computeSphereXYZ_sse " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
void RegisterDense::getSalientPoints_sphere(Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                            const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                            const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY
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
        //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
    }
    const size_t start_row = (nCols-nRows) / 2;
    Eigen::VectorXf v_sinPhi( v_sinTheta.block(start_row,0,nRows,1) );
    Eigen::VectorXf v_cosPhi( v_cosTheta.block(start_row,0,nRows,1) );

    float *_depthSrcGradXPyr = reinterpret_cast<float*>(depth_gradX.data);
    float *_depthSrcGradYPyr = reinterpret_cast<float*>(depth_gradY.data);
    float *_graySrcGradXPyr = reinterpret_cast<float*>(intensity_gradX.data);
    float *_graySrcGradYPyr = reinterpret_cast<float*>(intensity_gradY.data);

    //Compute the 3D coordinates of the 3D points in source frame
    validPixels.resize(imgSize);
    xyz.resize(imgSize,3);
    float *_depth_src = reinterpret_cast<float*>(depth_img.data);
    size_t count_valid_pixels = 0;
    for(size_t r=0;r<nRows;r++)
    {
        size_t i = r*nCols;
        for(size_t c=0;c<nCols;c++,i++)
        {
            if(min_depth_ < _depth_src[i] && _depth_src[i] < max_depth_) //Compute the jacobian only for the valid points
                if( fabs(_graySrcGradXPyr[i]) > thres_saliency_gray_ || fabs(_graySrcGradYPyr[i]) > thres_saliency_gray_ ||
                    fabs(_depthSrcGradXPyr[i]) > thres_saliency_depth_ || fabs(_depthSrcGradYPyr[i]) > thres_saliency_depth_ )
                {
                    validPixels(count_valid_pixels) = i;
                    //std::cout << " depth " << _depth_src[i] << " validPixels " << validPixels(count_valid_pixels) << " count_valid_pixels " << count_valid_pixels << std::endl;
                    xyz(count_valid_pixels,0) = _depth_src[i] * v_cosPhi[r] * v_sinTheta[c];
                    xyz(count_valid_pixels,1) = _depth_src[i] * v_sinPhi[r];
                    xyz(count_valid_pixels,2) = _depth_src[i] * v_cosPhi[r] * v_cosTheta[c];
                    //std::cout << " xyz " << xyz.block(count_valid_pixels,0,1,3) << std::endl;
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

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::getSalientPoints_sphere " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
void RegisterDense::getSalientPoints_sphere_sse(Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                                const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                                const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY
                                                ) // TODO extend this function to employ only depth
{
    //std::cout << " RegisterDense::getSalientPoints_sphere_sse \n";
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

    assert(nCols % 4 == 0);

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
        //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
    }
    const size_t start_row = (nCols-nRows) / 2;
    Eigen::VectorXf v_sinPhi( v_sinTheta.block(start_row,0,nRows,1) );
    Eigen::VectorXf v_cosPhi( v_cosTheta.block(start_row,0,nRows,1) );

    //Compute the 3D coordinates of the pij of the source frame
    Eigen::MatrixXf xyz_tmp(imgSize,3);
    Eigen::VectorXi validPixels_tmp(imgSize);
    float *_depth = reinterpret_cast<float*>(depth_img.data);
    float *_x = &xyz_tmp(0,0);
    float *_y = &xyz_tmp(0,1);
    float *_z = &xyz_tmp(0,2);
    float *_depthGradXPyr = reinterpret_cast<float*>(depth_gradX.data);
    float *_depthGradYPyr = reinterpret_cast<float*>(depth_gradY.data);
    float *_grayGradXPyr = reinterpret_cast<float*>(intensity_gradX.data);
    float *_grayGradYPyr = reinterpret_cast<float*>(intensity_gradY.data);
    float *_valid_pt = reinterpret_cast<float*>(&validPixels_tmp(0));
    __m128 _min_depth_ = _mm_set1_ps(min_depth_);
    __m128 _max_depth_ = _mm_set1_ps(max_depth_);
    __m128 _depth_saliency_ = _mm_set1_ps(thres_saliency_depth_);
    __m128 _gray_saliency_ = _mm_set1_ps(thres_saliency_gray_);
    __m128 _depth_saliency_neg = _mm_set1_ps(-thres_saliency_depth_);
    __m128 _gray_saliency_neg = _mm_set1_ps(-thres_saliency_gray_);

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
    size_t count_valid = 0;
    xyz.resize(imgSize,3);
    validPixels.resize(imgSize);
    //std::cout << " " << LUT_xyz_source.rows() << " " << xyz_tmp.rows() << " size \n";
    for(size_t i=0; i < imgSize; i++)
    {
        if( validPixels_tmp(i) )
        {
            validPixels(count_valid) = i;
            xyz.block(count_valid,0,1,3) = xyz_tmp.block(i,0,1,3);
            ++count_valid;
        }
    }
    size_t salient_pixels_aligned = count_valid - count_valid % 4;
    //std::cout << salient_pixels_aligned << " salient_pixels_aligned \n";
    validPixels.conservativeResize(salient_pixels_aligned);
    xyz.conservativeResize(salient_pixels_aligned,3);

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::getSalientPoints_sphere_sse " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}


/*! Compute the 3D points XYZ according to the pinhole camera model. */
void RegisterDense::computePinholeXYZ(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels)
{
    const size_t nRows = depth_img.rows;
    const size_t nCols = depth_img.cols;
    const size_t imgSize = nRows*nCols;

#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    // Make LUT to store the values of the 3D points of the source sphere
    LUT_xyz_source.resize(imgSize,3);
    validPixels = Eigen::VectorXi::Ones(imgSize);

    const float inv_fx = 1./fx;
    const float inv_fy = 1./fy;
    float *_depth_src = reinterpret_cast<float*>(depth_img.data);

    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(size_t r=0;r<nRows; r++)
    {
        for(size_t c=0;c<nCols; c++)
        {
            int i = r*nCols + c;
            LUT_xyz_source(i,2) = _depth_src[i]; //LUT_xyz_source(i,2) = 0.001f*depthSrcPyr[pyramidLevel].at<unsigned short>(r,c);

            //Compute the 3D coordinates of the pij of the source frame
            //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
            //std::cout << depthSrcPyr[pyramidLevel].type() << " LUT_xyz_source " << i << " x " << LUT_xyz_source(i,2) << " thres " << min_depth_ << " " << max_depth_ << std::endl;
            if(min_depth_ < LUT_xyz_source(i,2) && LUT_xyz_source(i,2) < max_depth_) //Compute the jacobian only for the valid points
            {
                LUT_xyz_source(i,0) = (c - ox) * LUT_xyz_source(i,2) * inv_fx;
                LUT_xyz_source(i,1) = (r - oy) * LUT_xyz_source(i,2) * inv_fy;
            }
            else
                validPixels(i) = 0;
        }
    }

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::computePinholeXYZ " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Compute the 3D points XYZ according to the pinhole camera model. */
void RegisterDense::computePinholeXYZ_sse(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels)
{
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

    // Make LUT to store the values of the 3D points of the source sphere
    LUT_xyz_source.resize(imgSize,3);
    validPixels = Eigen::VectorXi::Ones(imgSize);

    const float inv_fx = 1./fx;
    const float inv_fy = 1./fy;
    float *_depth_src = reinterpret_cast<float*>(depth_img.data);

    #if ENABLE_OPENMP
    #pragma omp parallel for
    #endif
    for(size_t r=0;r<nRows; r++)
    {
        for(size_t c=0;c<nCols; c++)
        {
            int i = r*nCols + c;
            LUT_xyz_source(i,2) = _depth_src[i]; //LUT_xyz_source(i,2) = 0.001f*depthSrcPyr[pyramidLevel].at<unsigned short>(r,c);

            //Compute the 3D coordinates of the pij of the source frame
            //std::cout << " theta " << theta << " phi " << phi << " rc " << r << " " << c << std::endl;
            //std::cout << depthSrcPyr[pyramidLevel].type() << " LUT_xyz_source " << i << " x " << LUT_xyz_source(i,2) << " thres " << min_depth_ << " " << max_depth_ << std::endl;
            if(min_depth_ < LUT_xyz_source(i,2) && LUT_xyz_source(i,2) < max_depth_) //Compute the jacobian only for the valid points
            {
                LUT_xyz_source(i,0) = (c - ox) * LUT_xyz_source(i,2) * inv_fx;
                LUT_xyz_source(i,1) = (r - oy) * LUT_xyz_source(i,2) * inv_fy;
            }
            else
                validPixels(i) = 0;
        }
    }

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::computePinholeXYZ_sse " << imgSize << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Transform 'input_pts', a set of 3D points according to the given rigid transformation 'Rt'. The output set of points is 'output_pts' */
void RegisterDense::transformPts3D(const Eigen::MatrixXf & input_pts, const Eigen::Matrix4f & Rt, Eigen::MatrixXf & output_pts)
{
    //std::cout << " RegisterDense::transformPts3D " << input_pts.rows() << " pts \n";
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    Eigen::MatrixXf input_xyz_transp = Eigen::MatrixXf::Ones(4,input_pts.rows());
    input_xyz_transp = input_pts.block(0,0,input_pts.rows(),3).transpose();
    Eigen::MatrixXf aux = Rt * input_xyz_transp;
    output_pts = aux.block(0,0,3,input_pts.rows()).transpose();

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::transformPts3D " << input_pts.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Transform 'input_pts', a set of 3D points according to the given rigid transformation 'Rt'. The output set of points is 'output_pts' */
//void RegisterDense::transformPts3D_sse(const Eigen::MatrixXf & input_pts, const Eigen::Matrix4f & Rt, Eigen::MatrixXf & output_xyz_transp)
void RegisterDense::transformPts3D_sse(const Eigen::MatrixXf & input_pts, const Eigen::Matrix4f & Rt, Eigen::MatrixXf & output_pts)
{
    //std::cout << " RegisterDense::transformPts3D_sse " << input_pts.rows() << " pts \n";
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    // Eigen default ColMajor is assumed
    assert(input_pts.cols() == 3);
    output_pts.resize( input_pts.rows(), input_pts.cols() );

    const __m128 r11 = _mm_set1_ps( Rt(0,0) );
    const __m128 r12 = _mm_set1_ps( Rt(0,1) );
    const __m128 r13 = _mm_set1_ps( Rt(0,2) );
    const __m128 r21 = _mm_set1_ps( Rt(1,0) );
    const __m128 r22 = _mm_set1_ps( Rt(1,1) );
    const __m128 r23 = _mm_set1_ps( Rt(1,2) );
    const __m128 r31 = _mm_set1_ps( Rt(2,0) );
    const __m128 r32 = _mm_set1_ps( Rt(2,1) );
    const __m128 r33 = _mm_set1_ps( Rt(2,2) );
    const __m128 t1 = _mm_set1_ps( Rt(0,3) );
    const __m128 t2 = _mm_set1_ps( Rt(1,3) );
    const __m128 t3 = _mm_set1_ps( Rt(2,3) );

    //Map<MatrixType>
    const float *input_x = &input_pts(0,0);
    const float *input_y = &input_pts(0,1);
    const float *input_z = &input_pts(0,2);
    float *output_x = &output_pts(0,0);
    float *output_y = &output_pts(0,1);
    float *output_z = &output_pts(0,2);

    size_t block_end;
    const Eigen::MatrixXf *input_pts_aligned;

    // Take into account that the total number of points might not be a multiple of 4
    if(input_pts.rows() % 4 == 0) // Data must be aligned
    {
        block_end = input_pts.rows();
    }
    else
    {
        std::cout << " RegisterDense::transformPts3D_sse UNALIGNED MATRIX pts \n";
        size_t block_end = input_pts.rows() - input_pts.rows() % 4;
        input_pts_aligned = new Eigen::MatrixXf(input_pts.block(0,0,block_end,3));
        output_pts.resize( block_end, input_pts.cols() );
        const float *input_x = &(*input_pts_aligned)(0,0);
        const float *input_y = &(*input_pts_aligned)(0,1);
        const float *input_z = &(*input_pts_aligned)(0,2);
    }

    if(block_end > 1e2)
    {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(size_t b=0; b < input_pts.rows(); b+=4)
        {
            __m128 block_x_in = _mm_load_ps(input_x+b);
            __m128 block_y_in = _mm_load_ps(input_y+b);
            __m128 block_z_in = _mm_load_ps(input_z+b);

            __m128 block_x_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r11, block_x_in), _mm_mul_ps(r12, block_y_in) ), _mm_mul_ps(r13, block_z_in) ), t1);
            __m128 block_y_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r21, block_x_in), _mm_mul_ps(r22, block_y_in) ), _mm_mul_ps(r23, block_z_in) ), t2);
            __m128 block_z_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r31, block_x_in), _mm_mul_ps(r32, block_y_in) ), _mm_mul_ps(r33, block_z_in) ), t3);

            _mm_store_ps(output_x+b, block_x_out);
            _mm_store_ps(output_y+b, block_y_out);
            _mm_store_ps(output_z+b, block_z_out);
            //std::cout << b << " b " << input_x[b] << " " << input_y[b] << " " << input_z[b] << " pt " << output_x[b] << " " << output_y[b] << " " << output_z[b] << "\n";
        }
    }
    else
    {
        for(size_t b=0; b < block_end; b+=4)
        {
            __m128 block_x_in = _mm_load_ps(input_x+b);
            __m128 block_y_in = _mm_load_ps(input_y+b);
            __m128 block_z_in = _mm_load_ps(input_z+b);

            __m128 block_x_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r11, block_x_in), _mm_mul_ps(r12, block_y_in) ), _mm_mul_ps(r13, block_z_in) ), t1);
            __m128 block_y_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r21, block_x_in), _mm_mul_ps(r22, block_y_in) ), _mm_mul_ps(r23, block_z_in) ), t2);
            __m128 block_z_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r31, block_x_in), _mm_mul_ps(r32, block_y_in) ), _mm_mul_ps(r33, block_z_in) ), t3);

            _mm_store_ps(output_x+b, block_x_out);
            _mm_store_ps(output_y+b, block_y_out);
            _mm_store_ps(output_z+b, block_z_out);
        }
    }

    if(input_pts.rows() % 4 != 0) // Compute the transformation of the unaligned points
    {
        delete input_pts_aligned;

        // Compute the transformation of those points which do not enter in a block
        const Eigen::Matrix3f rotation_transposed = Rt.block(0,0,3,3).transpose();
        const Eigen::Matrix<float,1,3> translation_transposed = Rt.block(0,3,3,1).transpose();
        output_pts.conservativeResize( input_pts.rows(), input_pts.cols() );
        for(size_t i=block_end; i < input_pts.rows(); i++)
        {
            output_pts.block(i,0,1,3) = input_pts.block(i,0,1,3) * rotation_transposed + translation_transposed;
        }
    }

//    // Get the points as a 3xN matrix, where N is the number of points
//    output_xyz_transp = output_pts.transpose();

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::transformPts3D_sse " << input_pts.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

//                                if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
//                                {
//                                    // Photometric component
//                                    hessian += jacobianPhoto.transpose()*jacobianPhoto;
//                                    gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
//                                }
//                                if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
//                                    if(depth > min_depth_) // Make sure this point has depth (not a NaN)
//                                    {
//                                        // Depth component (Plane ICL like)
//                                        hessian += jacobianDepth.transpose()*jacobianDepth;
//                                        gradient += jacobianDepth.transpose()*weightedErrorDepth;
//                                    }
//                            }

/*! Update the Hessian and the Gradient from a list of jacobians and residuals. */
void RegisterDense::updateHessianAndGradient(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi & valid_pixels)
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

//    if(use_salient_pixels_)
//    {
//    #if ENABLE_OPENMP
//    #pragma omp parallel for reduction (+:h11,h12,h13,h14,h15,h16,h22,h23,h24,h25,h26,h33,h34,h35,h36,h44,h45,h46,h55,h56,h66,g1,g2,g3,g4,g5,g6) // Cannot reduce on Eigen types
//    #endif
//        for(size_t i=0; i < pixel_jacobians.rows(); i++)
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
        for(size_t i=0; i < pixel_jacobians.rows(); i++)
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
    std::cout << " RegisterDense::updateHessianAndGradient " << pixel_jacobians.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

//void RegisterDense::updateHessianAndGradient(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi & warp_pixels)
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
//    std::cout << " RegisterDense::updateHessianAndGradient " << pixel_jacobians.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
//    #endif
//}

void RegisterDense::updateGrad(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi & valid_pixels)
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
    std::cout << " RegisterDense::updateGrad " << pixel_jacobians.rows() << " pts took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

/*! Get a list of salient points from a list of Jacobians corresponding to a set of 3D points */
void RegisterDense::getSalientPts(const Eigen::MatrixXf & jacobians, std::vector<size_t> & salient_pixels, const float r_salient )
{
    //std::cout << " RegisterDense::getSalientPts " << input_pts.rows() << " pts \n";
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

   assert( jacobians.cols() == 6 && jacobians.rows() > 6 );
   salient_pixels.resize(0);
   if( jacobians.rows() < r_salient ) // If the user ask for more salient points than current points, do nothing
       return;

   const size_t n_pts = jacobians.rows();
   size_t n_salient;
   if(r_salient < 2)
       n_salient = floor(r_salient * n_pts);
   else // set the number of points
       n_salient = size_t(r_salient);

   std::vector<size_t> sorted_idx;
   sorted_idx.reserve(6*n_pts);
   // [U,S,V] = svd(J,'econ');

   for(size_t i = 0 ; i < 6 ; ++i)
   {
       std::vector<size_t> idx = sort_indexes_<float>(jacobians.col(i));
       sorted_idx.insert( sorted_idx.end(), idx.begin(), idx.end() );
   }

   //std::vector<size_t> salient_pixels(n_pts);
   salient_pixels.resize(n_pts);

   size_t *tabetiq = new size_t [n_pts*6];
   size_t *tabi = new size_t [6];

   for(size_t j = 0 ; j < n_pts ; ++j)
        salient_pixels[j] = 0;

   for(size_t i = 0 ; i < 6 ; ++i)
   {
       tabi[i] = 0;
       for(size_t j = 0 ; j < n_pts ; ++j)
            tabetiq[j*6+i] = 0;
   }

    size_t k = 0;
    size_t line;
    for(size_t i = 0 ; i < floor(n_pts/6) ; ++i) // n_pts/6 is Weird!!
    {
        for(size_t j = 0 ; j < 6 ; ++j)
        {
             while(tabi[j] < n_pts)
             {
                 line = (size_t)(sorted_idx[tabi[j]*6 + j]);
                 if(tabetiq[j + line*6] == 0)
                 {
                    for( size_t jj = 0 ; jj < 6 ; ++jj)
                        tabetiq[(size_t)(line*6 + jj)] = 1;

                    salient_pixels[k] =  sorted_idx[tabi[j]*6+ j];
                    ++k;
                    tabi[j]++;
                    //deg[k] = j;
                    if(k == n_salient) // Stop arranging the indices
                    {
                        j = 6;
                        i = floor(n_pts/6);
                    }
                    break;
                 }
                 else
                 {
                     tabi[j]++;
                 }

             }
        }
    }

    delete [] tabetiq;
    delete [] tabi;
    salient_pixels.resize(n_salient);

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::getSalientPts " << jacobians.rows() << " pts took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}

void RegisterDense::trimValidPoints(Eigen::MatrixXf & LUT_xyz, Eigen::VectorXi & validPixels, Eigen::MatrixXf & xyz_transf,
                                     Eigen::VectorXi & validPixelsPhoto, Eigen::VectorXi & validPixelsDepth,
                                     costFuncType method, std::vector<size_t> &salient_pixels_photo, std::vector<size_t> &salient_pixels_depth)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //std::cout << " RegisterDense::trimValidPoints pts " << LUT_xyz.rows() << " pts " << validPixelsPhoto.rows() << " pts "<< salient_pixels_photo.size()<< " - " << salient_pixels_depth.size() << std::endl;
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    if( salient_pixels_photo.empty() && salient_pixels_depth.empty() ){ std::cout << " RegisterDense::trimValidPoints EMPTY set of salient points \n";
        return;}

    if(use_salient_pixels_)
    {
        // Arrange all the list of points and indices, by now this is done outside this function
        std::vector<size_t> salient_pixels(0); // Make it a pointer
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
            salient_pixels.insert(salient_pixels.end(), s_salient_pixels.begin(), s_salient_pixels.end());
        }
        else if(method == PHOTO_CONSISTENCY)
            salient_pixels = salient_pixels_photo;
        else if(method == DEPTH_CONSISTENCY)
            salient_pixels = salient_pixels_depth;

        size_t aligned_pts = salient_pixels.size() - salient_pixels.size() % 4;
        salient_pixels.resize(aligned_pts);

        // Arrange salient pixels
        Eigen::MatrixXf LUT_xyz_tmp(salient_pixels.size(),3);
        Eigen::MatrixXf xyz_transf_tmp(salient_pixels.size(),3);
//        Eigen::VectorXi validPixelsPhoto_tmp(salient_pixels.size());
//        Eigen::VectorXi validPixelsDepth_tmp(salient_pixels.size());
        for(size_t i=0; i < salient_pixels.size(); ++i)
        {
            //std::cout << i << " " << salient_pixels[i] << " \n";
            LUT_xyz_tmp.block(i,0,1,3) = LUT_xyz.block(salient_pixels[i],0,1,3);
            xyz_transf_tmp.block(i,0,1,3) = xyz_transf.block(salient_pixels[i],0,1,3);
//            validPixelsPhoto_tmp(i) = validPixelsPhoto(salient_pixels[i]);
//            validPixelsDepth_tmp(i) = validPixelsDepth(salient_pixels[i]);
        }
        LUT_xyz = LUT_xyz_tmp;
        xyz_transf = xyz_transf_tmp;
//        validPixelsPhoto = validPixelsPhoto_tmp;
//        validPixelsDepth = validPixelsDepth_tmp;
    }
    else
    {
        validPixels.setZero();
        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
        {
            validPixelsPhoto.setZero(); // Eigen::VectorXi::Zero(LUT_xyz.rows());
            for(size_t j = 0 ; j < salient_pixels_photo.size() ; ++j)
            {
                validPixels(salient_pixels_photo[j]) = 1;
                validPixelsPhoto(salient_pixels_photo[j]) = 1;
            }
        }
        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
        {
            validPixelsDepth.setZero();
            for(size_t j = 0 ; j < salient_pixels_depth.size() ; ++j)
            {
                validPixels(salient_pixels_depth[j]) = 1;
                validPixelsDepth(salient_pixels_depth[j]) = 1;
            }
        }
    }

    #if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    std::cout << " RegisterDense::trimValidPoints took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}
