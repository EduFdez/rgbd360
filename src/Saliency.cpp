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

#include <Saliency.h>

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
#if _AVX
#  include <avxintrin.h>
#endif

#define ENABLE_OPENMP 0
#define PRINT_PROFILING 1
#define ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS 1
#define INVALID_POINT -10000
#define SSE_AVAILABLE 1

using namespace std;

Saliency::Saliency() :
    thresSaliency(0.04f),
//    thres_saliency_gray_ = 0.04f;
//    thres_saliency_depth_ = 0.04f;
    thres_saliency_gray_(0.001f),
    thres_saliency_depth_(0.001f),
    _max_depth_grad(1.2f)
{
}

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
void Saliency::getSalientPts(const Eigen::MatrixXf & jacobians, std::vector<size_t> & salient_pixels, const float r_salient )
{
    //cout << " RegisterDense::getSalientPts " << input_pts.rows() << " pts \n";
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
   cout << " RegisterDense::getSalientPts HISTOGRAM APPROXIMATION " << jacobians.rows() << " pts took " << (time_end - time_start)*1000 << " ms. \n";
   #endif
}

void Saliency::trimValidPoints(Eigen::MatrixXf & LUT_xyz, Eigen::VectorXi & validPixels, Eigen::MatrixXf & xyz_transf,
                                    Eigen::VectorXi & validPixelsPhoto, Eigen::VectorXi & validPixelsDepth,
                                    const costFuncType method,
                                    std::vector<size_t> &salient_pixels, std::vector<size_t> &salient_pixels_photo, std::vector<size_t> &salient_pixels_depth)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //cout << " RegisterDense::trimValidPoints pts " << LUT_xyz.rows() << " pts " << validPixelsPhoto.rows() << " pts "<< salient_pixels_photo.size()<< " - " << salient_pixels_depth.size() << endl;
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

    if( salient_pixels_photo.empty() && salient_pixels_depth.empty() ){ cout << " RegisterDense::trimValidPoints EMPTY set of salient points \n";
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

    if(use_salient_pixels_)
    {
        // Arrange salient pixels
        Eigen::VectorXi validPixels_tmp(salient_pixels.size());
        Eigen::MatrixXf LUT_xyz_tmp(salient_pixels.size(),3);
        //Eigen::MatrixXf xyz_transf_tmp(salient_pixels.size(),3);
//        Eigen::VectorXi validPixelsPhoto_tmp(salient_pixels.size());
//        Eigen::VectorXi validPixelsDepth_tmp(salient_pixels.size());
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
            //validPixelsPhoto.setZero(); // Eigen::VectorXi::Zero(LUT_xyz.rows());
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
    cout << " RegisterDense::trimValidPoints took " << (time_end - time_start)*1000 << " ms. \n";
    #endif
}
