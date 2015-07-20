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

#include <Pyramid.h>
#include <config.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
//#include <iostream>
//#include <fstream>

#include <pcl/common/time.h>
#include <mrpt/utils/mrpt_macros.h>

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

/*! Build a pyramid of nPyrLevels of image resolutions from the input image.
 * The resolution of each layer is 2x2 times the resolution of its image above.*/
void Pyramid::buildPyramid(const cv::Mat & img, std::vector<cv::Mat> & pyramid, const int nPyrLevels)
{
#if PRINT_PROFILING
    cout << "Pyramid::buildPyramid... \n";
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif
    //Create space for all the images // ??
    pyramid.resize(nPyrLevels+1);
    pyramid[0] = img;
    //cout << "types " << pyramid[0].type() << " " << img.type() << endl;

//    cv::Mat img_show;
//    pyramid[0].convertTo(img_show, CV_8UC1, 255);
//    cv::imwrite(mrpt::format("/home/efernand/pyr_gray_%d.png",0), img_show);
//    //cv::imshow("pyramid", img_show);
//    //cv::waitKey(0);

    for(int level=1; level <= nPyrLevels; level++)
    {
        assert(pyramid[0].rows % 2 == 0 && pyramid[0].cols % 2 == 0 );

        // Assign the resized image to the current level of the pyramid
        pyrDown( pyramid[level-1], pyramid[level], cv::Size( pyramid[level-1].cols/2, pyramid[level-1].rows/2 ) );

//        cv::Mat img_show;
//        pyramid[level].convertTo(img_show, CV_8UC1, 255);
//        cv::imwrite(mrpt::format("/home/efernand/pyr_gray_%d.png",level), img_show);
//        cv::imshow("pyramid", pyramid[level]);
//        cv::waitKey(0);
    }
#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << "Pyramid::buildPyramid " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Build a pyramid of nPyrLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
void Pyramid::buildPyramidRange(const cv::Mat & img, std::vector<cv::Mat> & pyramid, const int nPyrLevels)
{
#if PRINT_PROFILING
    cout << "Pyramid::buildPyramidRange... \n";
    double time_start = pcl::getTime();\
    //for(size_t i=0; i<1000; i++)
    {
#endif
    //Create space for all the images // ??
    pyramid.resize(nPyrLevels+1);
    if(img.type() == CV_16U) // If the image is in milimetres, convert it to meters
        img.convertTo(pyramid[0], CV_32FC1, 0.001 );
    else
        pyramid[0] = img;
    //    cout << "types " << pyramid[0].type() << " " << img.type() << endl;

    for(int level=1; level <= nPyrLevels; level++)
    {
        //Create an auxiliar image of factor times the size of the original image
        size_t nCols = pyramid[level-1].cols;
        size_t nRows = pyramid[level-1].rows;
        size_t img_size = nRows * nCols;
        pyramid[level] = cv::Mat::zeros(cv::Size( nCols/2, nRows/2 ), pyramid[0].type() );
        //            cv::Mat imgAux = cv::Mat::zeros(cv::Size( nCols/2, pyramid[level-1].rows/2 ), pyramid[0].type() );
        float *_z = reinterpret_cast<float*>(pyramid[level-1].data);
        float *_z_sub = reinterpret_cast<float*>(pyramid[level].data);
//        if(img_size > 4*1e4) // Apply multicore only to the bigger images
//        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t r=0; r < nRows; r+=2)
            {
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
                            if(z > 0.f)
                            {
                                avDepth += z;
                                ++nvalidPixels_src;
                            }
                        }
                    if(nvalidPixels_src > 0)
                    {
                        //pyramid[level].at<float>(r/2,c/2) = avDepth / nvalidPixels_src;
                        unsigned int pixel_sub = r*nCols/4 + c/2;
                        _z_sub[pixel_sub] = avDepth / nvalidPixels_src;
                        //cout << pixel_sub << " pixel_sub " << _z_sub[pixel_sub] << endl;
                    }
                }
            }
//        }
//        else
//        {
//            for(size_t r=0, new_pixel=0; r < nRows; r+=2)
//                for(size_t c=0; c < nCols; c+=2, new_pixel++)
//                {
//                    vector<float> valid_depth(4);
//                    unsigned nvalidPixels_src = 0;
//                    for(size_t i=0; i < 2; i++)
//                        for(size_t j=0; j < 2; j++)
//                        {
//                            size_t pixel = (r+i)*nCols + c + j;
//                            float z = _z[pixel];
//                            //                        float z = pyramid[level-1].at<float>(r+i,c+j);
//                            //cout << "z " << z << endl;
//                            if(z > 0.f)
//                            {
//                                valid_depth[nvalidPixels_src] = z;
//                                ++nvalidPixels_src;
//                            }
//                        }
//                    if(nvalidPixels_src > 0)
//                    {
//                        valid_depth.resize(nvalidPixels_src);
//                        std::sort(valid_depth.begin(), valid_depth.end());
//                        //pyramid[level].at<float>(r/2,c/2) = avDepth / nvalidPixels_src;
//                        size_t id_median = nvalidPixels_src / 2;
//                        _z_sub[new_pixel] = valid_depth[id_median];
//                        //cout << new_pixel << " new_pixel " << _z_sub[new_pixel] << endl;
//                    }
//                }
//        }

//        cv::Mat img_show;
//        cv::Mat img255(pyramid[level].rows, pyramid[level].cols, CV_8U, 255);
//        const float viz_factor_meters = 82.5;
//        pyramid[level].convertTo(img_show, CV_8U, viz_factor_meters);
//        cv::Mat mask = img_show == 0;
//        img_show = img255 - img_show;
//        img_show.setTo(0, mask);
//        cv::imwrite(mrpt::format("/home/efernand/pyr_depth_%d.png",level), img_show);
    }
#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << "Pyramid::buildPyramidRange " << (time_end - time_start)*1000 << " ms. \n";
#endif
}


/*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
void Pyramid::calcGradientXY(const cv::Mat & src, cv::Mat & gradX, cv::Mat & gradY)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
    {
#endif

    ASSERT_(src.cols % 4 == 0);

    //int dataType = src.type();
    gradX = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );
    gradY = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );

    float *_pixel = reinterpret_cast<float*>(src.data);
    float *_pixel_gradX = reinterpret_cast<float*>(gradX.data);
    float *_pixel_gradY = reinterpret_cast<float*>(gradY.data);

    size_t img_size = src.rows * src.cols;

#if !(_SSE3) // # ifdef __SSE3__

    if(img_size > 4*1e4) // Apply multicore only to the bigger images
    {
        #if ENABLE_OPENMP
        #pragma omp parallel for // schedule(static) // schedule(dynamic)
        #endif
        for(size_t r=1; r < src.rows-1; ++r)
        {
            size_t row_pix = r*src.cols;
            for(int c=1; c < src.cols-1; ++c)
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
    //                    //cout << "GradX " << gradX.at<float>(r,c) << " " << gradX_.at<float>(r,c) << endl;}

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
        for(int c=1; c < src.cols-1; ++c)
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
            for(int c=1; c < src.cols-1; ++c)
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
    //                    //cout << "GradX " << gradX.at<float>(r,c) << " " << gradX_.at<float>(r,c) << endl;}

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
        for(int c=1; c < src.cols-1; ++c)
        {
            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
        }
    }

#else // Use _SSE3
//#elif !(_AVX) //

    // Use SIMD instructions -> x4 performance
    assert( src.cols % 4 == 0);
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
        for(int r=1; r < src.rows-1; ++r)
        {
            size_t row_pix = r*src.cols;
            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
        }
        size_t last_row_pix = (src.rows - 1) * src.cols;
        #if ENABLE_OPENMP
        #pragma omp parallel for // schedule(static) // schedule(dynamic)
        #endif
        for(int c=1; c < src.cols-1; ++c)
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
        for(int r=1; r < src.rows-1; ++r)
        {
            size_t row_pix = r*src.cols;
            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
        }
        size_t last_row_pix = (src.rows - 1) * src.cols;
        for(int c=1; c < src.cols-1; ++c)
        {
            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
        }
    }

//#else // Use _AVX

//    cout << "calcGradientXY AVX \n";

//    // Use SIMD instructions -> x4 performance
//    assert( src.cols % 8 == 0);
//    size_t block_start = src.cols, block_end = img_size - src.cols;
//    const __m256 scalar2 = _mm256_set1_ps(2.f); //float f2 = 2.f;
//    if(img_size > 8*1e4) // Apply multicore only to the bigger images
//    {
//        #if ENABLE_OPENMP
//        #pragma omp parallel for // schedule(static) // schedule(dynamic)
//        #endif
//        for(size_t b=block_start; b < block_end; b+=8)
//        {
////            float *_block = _pixel+b;
//            __m256 block_ = _mm256_load_ps(_pixel+b);
//            __m256 block_x1 = _mm256_loadu_ps(_pixel+b+1);
//            __m256 block_x_1 = _mm256_loadu_ps(_pixel+b-1);
//            __m256 diff1 = _mm256_sub_ps(block_x1, block_);
//            __m256 diff_1= _mm256_sub_ps(block_, block_x_1);
//            __m256 den = _mm256_add_ps(diff1, diff_1);
//            __m256 mul = _mm256_mul_ps(diff1, diff_1);
//            __m256 num = _mm256_mul_ps(scalar2, mul);
//            __m256 res = _mm256_div_ps(num, den);
//            //__m256 res = _mm256_div_ps(_mm256_mul_ps(scalar2, mul), den);
//            //_mm256_store_ps(_pixel_gradX+b, res);
//            __m256 mask = _mm256_or_ps(
//                                    _mm256_and_ps( _mm256_cmplt_ps(block_x_1, block_), _mm256_cmplt_ps(block_, block_x1) ),
//                                    _mm256_and_ps( _mm256_cmplt_ps(block_x1, block_), _mm256_cmplt_ps(block_, block_x_1) ) );

//            __m128 block_a_ = _mm_load_ps(_pixel+b);
//            __m128 block_b_ = _mm_load_ps(_pixel+b+4);
//            __m128 block_x1_a = _mm_loadu_ps(_pixel+b+1);
//            __m128 block_x1_b = _mm_loadu_ps(_pixel+b+5);

//            __m128 block_x_1 = _mm_loadu_ps(_pixel+b-1);
//            __m128 diff1 = _mm_sub_ps(block_x1, block_);
//            __m128 diff_1= _mm_sub_ps(block_, block_x_1);
//            __m128 den = _mm_add_ps(diff1, diff_1);
//            __m128 mul = _mm_mul_ps(diff1, diff_1);
//            __m128 num = _mm_mul_ps(scalar2, mul);
//            __m128 res = _mm_div_ps(num, den);
//            //__m128 res = _mm_div_ps(_mm_mul_ps(scalar2, mul), den);
//            //_mm_store_ps(_pixel_gradX+b, res);
//            __m128 mask = _mm_or_ps(
//                        _mm_and_ps( _mm_cmplt_ps(block_x_1, block_), _mm_cmplt_ps(block_, block_x1) ),



//            //_mm256_maskstore_ps(_pixel_gradX+b, mask, res);
//            __m256 gradX = _mm256_and_ps(mask, res);
//            _mm256_store_ps(_pixel_gradX+b, gradX);

//            __m256 block_y1 = _mm256_load_ps(_pixel+b+src.cols);
//            __m256 block_y_1 = _mm256_load_ps(_pixel+b-src.cols);
//            diff1 = _mm256_sub_ps(block_y1, block_);
//            diff_1= _mm256_sub_ps(block_, block_y_1);
//            den = _mm256_add_ps(diff1, diff_1);
//            mul = _mm256_mul_ps(diff1, diff_1);
//            num = _mm256_mul_ps(scalar2, mul);
//            res = _mm256_div_ps(num, den);
//            //res = _mm256_div_ps(_mm256_mul_ps(scalar2, mul), den);
//            //_mm256_store_ps(_pixel_gradY+b, res);
//            mask = _mm256_or_ps( _mm256_and_ps( _mm256_cmplt_ps(block_y_1, block_), _mm256_cmplt_ps(block_, block_y1) ),
//                              _mm256_and_ps( _mm256_cmplt_ps(block_y1, block_), _mm256_cmplt_ps(block_, block_y_1) ) );
//            //_mm256_maskstore_ps(_pixel_gradY+b, mask, res);
//            __m256 gradY = _mm256_and_ps(mask, res);
//            _mm256_store_ps(_pixel_gradY+b, gradY);
//        }
//        // Compute the gradint at the image border
//        #if ENABLE_OPENMP
//        #pragma omp parallel for // schedule(static) // schedule(dynamic)
//        #endif
//        for(int r=1; r < src.rows-1; ++r)
//        {
//            size_t row_pix = r*src.cols;
//            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
//            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
//        }
//        size_t last_row_pix = (src.rows - 1) * src.cols;
//        #if ENABLE_OPENMP
//        #pragma omp parallel for // schedule(static) // schedule(dynamic)
//        #endif
//        for(int c=1; c < src.cols-1; ++c)
//        {
//            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
//            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
//        }
//    }
//    else
//    {
//        for(size_t b=block_start; b < block_end; b+=4)
//        {
//            __m256 block_ = _mm256_load_ps(_pixel+b);
//            __m256 block_x1 = _mm256_loadu_ps(_pixel+b+1);
//            __m256 block_x_1 = _mm256_loadu_ps(_pixel+b-1);
//            __m256 diff1 = _mm256_sub_ps(block_x1, block_);
//            __m256 diff_1= _mm256_sub_ps(block_, block_x_1);
//            __m256 den = _mm256_add_ps(diff1, diff_1);
//            __m256 mul = _mm256_mul_ps(diff1, diff_1);
//            __m256 num = _mm256_mul_ps(scalar2, mul);
//            __m256 res = _mm256_div_ps(num, den);
//            //__m256 res = _mm256_div_ps(_mm256_mul_ps(scalar2, mul), den);
//            //_mm256_store_ps(_pixel_gradX+b, res);
//            __m256 mask = _mm256_or_ps(
//                        _mm256_and_ps( _mm256_cmplt_ps(block_x_1, block_), _mm256_cmplt_ps(block_, block_x1) ),
//                        _mm256_and_ps( _mm256_cmplt_ps(block_x1, block_), _mm256_cmplt_ps(block_, block_x_1) ) );
//            //_mm256_maskstore_ps(_pixel_gradX+b, mask, res);
//            __m256 gradX = _mm256_and_ps(mask, res);
//            _mm256_store_ps(_pixel_gradX+b, gradX);

//            __m256 block_y1 = _mm256_load_ps(_pixel+b+src.cols);
//            __m256 block_y_1 = _mm256_load_ps(_pixel+b-src.cols);
//            diff1 = _mm256_sub_ps(block_y1, block_);
//            diff_1= _mm256_sub_ps(block_, block_y_1);
//            den = _mm256_add_ps(diff1, diff_1);
//            mul = _mm256_mul_ps(diff1, diff_1);
//            num = _mm256_mul_ps(scalar2, mul);
//            res = _mm256_div_ps(num, den);
//            //res = _mm256_div_ps(_mm256_mul_ps(scalar2, mul), den);
//            //_mm256_store_ps(_pixel_gradY+b, res);
//            mask = _mm256_or_ps( _mm256_and_ps( _mm256_cmplt_ps(block_y_1, block_), _mm256_cmplt_ps(block_, block_y1) ),
//                              _mm256_and_ps( _mm256_cmplt_ps(block_y1, block_), _mm256_cmplt_ps(block_, block_y_1) ) );
//            //_mm256_maskstore_ps(_pixel_gradY+b, mask, res);
//            __m256 gradY = _mm256_and_ps(mask, res);
//            _mm256_store_ps(_pixel_gradY+b, gradY);
//        }
//        // Compute the gradint at the image border
//        for(int r=1; r < src.rows-1; ++r)
//        {
//            size_t row_pix = r*src.cols;
//            _pixel_gradX[row_pix] = _pixel[row_pix] - _pixel[row_pix-1];
//            _pixel_gradX[row_pix+src.cols-1] = _pixel[row_pix+src.cols-1] - _pixel[row_pix+src.cols-2];
//        }
//        size_t last_row_pix = (src.rows - 1) * src.cols;
//        for(int c=1; c < src.cols-1; ++c)
//        {
//            _pixel_gradY[c] = _pixel[c+src.cols] - _pixel[c];
//            _pixel_gradY[last_row_pix+c] = _pixel[last_row_pix+c] - _pixel[last_row_pix-src.cols+c];
//        }
//    }

#endif

//    cv::imshow("DerY", gradY);
//    cv::imshow("DerX", gradX);
//    cv::waitKey(0);

#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << src.rows << "rows. Pyramid::calcGradientXY _SSE3 " << _SSE3 << " " << (time_end - time_start)*1000 << " ms. \n";
#endif
}

/*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
void Pyramid::calcGradientXY_saliency(const cv::Mat & src, cv::Mat & gradX, cv::Mat & gradY, std::vector<int> & vSalientPixels_, const float thresSaliency)
{
    //int dataType = src.type();
    gradX = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );
    gradY = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );

    for(int r=1; r < src.rows-1; r++)
        for(int c=1; c < src.cols-1; c++)
        {
            if( (src.at<float>(r,c) > src.at<float>(r,c+1) && src.at<float>(r,c) < src.at<float>(r,c-1) ) ||
                (src.at<float>(r,c) < src.at<float>(r,c+1) && src.at<float>(r,c) > src.at<float>(r,c-1) )   )
                    gradX.at<float>(r,c) = 2.f / (1/(src.at<float>(r,c+1)-src.at<float>(r,c)) + 1/(src.at<float>(r,c)-src.at<float>(r,c-1)));

            if( (src.at<float>(r,c) > src.at<float>(r+1,c) && src.at<float>(r,c) < src.at<float>(r-1,c) ) ||
                (src.at<float>(r,c) < src.at<float>(r+1,c) && src.at<float>(r,c) > src.at<float>(r-1,c) )   )
                    gradY.at<float>(r,c) = 2.f / (1/(src.at<float>(r+1,c)-src.at<float>(r,c)) + 1/(src.at<float>(r,c)-src.at<float>(r-1,c)));
        }

    vSalientPixels_.clear();
    for(int r=1; r < src.rows-1; r++)
        for(int c=1; c < src.cols-1; c++)
            if( (fabs(gradX.at<float>(r,c)) > thresSaliency) || (fabs(gradY.at<float>(r,c)) > thresSaliency) )
                vSalientPixels_.push_back(src.cols*r+c); //vector index
}

/*! Compute the gradient images for each pyramid level. */
void Pyramid::buildGradientPyramids(const std::vector<cv::Mat> & grayPyr, std::vector<cv::Mat> & grayGradXPyr, std::vector<cv::Mat> & grayGradYPyr,
                                    const std::vector<cv::Mat> & depthPyr, std::vector<cv::Mat> & depthGradXPyr, std::vector<cv::Mat> & depthGradYPyr,
                                    const int nPyrLevels)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t i=0; i<1000; i++)
#endif

    //Compute image gradients
    //double scale = 1./255;
    //double delta = 0;
    //int dataType = CV_32FC1; // grayPyr[level].type();

    //Create space for all the derivatives images
    grayGradXPyr.resize(grayPyr.size());
    grayGradYPyr.resize(grayPyr.size());

    depthGradXPyr.resize(grayPyr.size());
    depthGradYPyr.resize(grayPyr.size());

    for(int level=0; level <= nPyrLevels; level++)
    {
        //double time_start_ = pcl::getTime();

        calcGradientXY(grayPyr[level], grayGradXPyr[level], grayGradYPyr[level]);

//#if PRINT_PROFILING
//        double time_end_ = pcl::getTime();
//        cout << level << " PyramidPhoto " << (time_end_ - time_start_) << endl;

//        time_start_ = pcl::getTime();
//#endif

        calcGradientXY(depthPyr[level], depthGradXPyr[level], depthGradYPyr[level]);

//#if PRINT_PROFILING
//        time_end_ = pcl::getTime();
//        cout << level << " PyramidDepth " << (time_end_ - time_start_) << endl;
//#endif


        //double time_start_ = pcl::getTime();

        // Compute the gradient in x
        //grayGradXPyr[level] = cv::Mat(cv::Size( grayPyr[level].cols, grayPyr[level].rows), grayPyr[level].type() );
        //cv::Scharr( grayPyr[level], grayGradXPyr[level], grayPyr[level].type(), 1, 0);//, scale, delta, cv::BORDER_DEFAULT );
        //cv::Sobel( grayPyr[level], grayGradXPyr[level], dataType, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );

        // Compute the gradient in y
        //grayGradYPyr[level] = cv::Mat(cv::Size( grayPyr[level].cols, grayPyr[level].rows), grayPyr[level].type() );
        //cv::Scharr( grayPyr[level], grayGradYPyr[level], grayPyr[level].type(), 0, 1);//, scale, delta, cv::BORDER_DEFAULT );
        //cv::Sobel( grayPyr[level], grayGradYPyr[level], dataType, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

        //double time_end_ = pcl::getTime();
        //cout << "Pyramid::Scharr derivatives " << (time_end_ - time_start_)*1000 << " ms. \n";

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

#if PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "Pyramid::buildGradientPyramids " << (time_end - time_start)*1000 << " ms. \n";
#endif
}
