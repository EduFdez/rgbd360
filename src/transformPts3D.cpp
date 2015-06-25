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

#include <transformPts3D.h>
#include <pcl/common/time.h>
#include <mrpt/system/os.h> // Only for pause

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

#define ENABLE_OPENMP 0
#define PRINT_PROFILING 1

using namespace std;

/*! Transform 'input_pts', a set of 3D points according to the given rigid transformation 'Rt'. The output set of points is 'output_pts' */
void transformPts3D(const Eigen::MatrixXf & input_pts, const Eigen::Matrix4f & Rt, Eigen::MatrixXf & output_pts)
{
#if PRINT_PROFILING
    double time_start = pcl::getTime();
    //for(size_t ii=0; ii<100; ii++)
    {
#endif

        size_t n_pts = input_pts.rows();

#if !(_SSE3) // # ifdef __SSE3__
        cout << " transformPts3D " << input_pts.rows() << " pts \n";

        Eigen::MatrixXf input_xyz_transp = Eigen::MatrixXf::Ones(4,n_pts);
        input_xyz_transp.block(0,0,3,n_pts) = input_pts.block(0,0,n_pts,3).transpose();
        Eigen::MatrixXf aux = Rt * input_xyz_transp;
        output_pts = aux.block(0,0,3,n_pts).transpose();

//        Eigen::MatrixXf input_xyz_transp = input_pts.block(0,0,n_pts,3).transpose();
//        Eigen::MatrixXf aux = Rt.block(0,0,3,3) * input_xyz_transp + repmat(1, t);
//        output_pts = aux.block(0,0,3,n_pts).transpose();

#else // _SSE3
//#elif !(_AVX) // # ifdef __AVX__
        cout << " transformPts3D _SSE3 " << n_pts << " pts \n";

        // Eigen default ColMajor is assumed
        assert(input_pts.cols() == 3);
        assert(n_pts % 4 == 0);
        output_pts.resize( n_pts, input_pts.cols() );

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
        float *input_x = const_cast<float*>(&input_pts(0,0));
        float *input_y = const_cast<float*>(&input_pts(0,1));
        float *input_z = const_cast<float*>(&input_pts(0,2));
        float *output_x = &output_pts(0,0);
        float *output_y = &output_pts(0,1);
        float *output_z = &output_pts(0,2);

        size_t block_end = n_pts;
        const Eigen::MatrixXf *input_pts_aligned = NULL;

        // Take into account that the total number of points might not be a multiple of 4
        if(n_pts % 4 != 0) // Data must be aligned
        {
            cout << " RegisterDense::transformPts3D UNALIGNED MATRIX pts \n";
            size_t block_end = n_pts - n_pts % 4;
            input_pts_aligned = new Eigen::MatrixXf(input_pts.block(0,0,block_end,3));
            output_pts.resize( block_end, input_pts.cols() );
            input_x = const_cast<float*>(&(*input_pts_aligned)(0,0));
            input_y = const_cast<float*>(&(*input_pts_aligned)(0,1));
            input_z = const_cast<float*>(&(*input_pts_aligned)(0,2));
        }

        if(block_end > 1e2)
        {
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
            for(size_t b=0; b < n_pts; b+=4)
            {
                __m128 block_x_in = _mm_load_ps(input_x+b);
                __m128 block_y_in = _mm_load_ps(input_y+b);
                __m128 block_z_in = _mm_load_ps(input_z+b);

//                float check_sse[4];
////                _mm_store_ps(check_sse, block_x_in);
//                _mm_store_ps(&check_sse[0], block_x_in);
//                cout << " check_sse \n";
//                for(size_t bit=0; bit < 4; bit++)
//                    cout << check_sse[bit] << " " << input_x[b+bit] << " \n";
//                mrpt::system::pause();

                __m128 block_x_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r11, block_x_in), _mm_mul_ps(r12, block_y_in) ), _mm_mul_ps(r13, block_z_in) ), t1);
                __m128 block_y_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r21, block_x_in), _mm_mul_ps(r22, block_y_in) ), _mm_mul_ps(r23, block_z_in) ), t2);
                __m128 block_z_out = _mm_add_ps( _mm_add_ps( _mm_add_ps( _mm_mul_ps(r31, block_x_in), _mm_mul_ps(r32, block_y_in) ), _mm_mul_ps(r33, block_z_in) ), t3);

                _mm_store_ps(output_x+b, block_x_out);
                _mm_store_ps(output_y+b, block_y_out);
                _mm_store_ps(output_z+b, block_z_out);
                //cout << b << " b " << input_x[b] << " " << input_y[b] << " " << input_z[b] << " pt " << output_x[b] << " " << output_y[b] << " " << output_z[b] << "\n";
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

        if(n_pts % 4 != 0) // Compute the transformation of the unaligned points
        {
            delete input_pts_aligned;

            // Compute the transformation of those points which do not enter in a block
            const Eigen::Matrix3f rotation_transposed = Rt.block(0,0,3,3).transpose();
            const Eigen::Matrix<float,1,3> translation_transposed = Rt.block(0,3,3,1).transpose();
            output_pts.conservativeResize( n_pts, input_pts.cols() );
            for(size_t i=block_end; i < n_pts; i++)
            {
                output_pts.block(i,0,1,3) = input_pts.block(i,0,1,3) * rotation_transposed + translation_transposed;
            }
        }

//#else // _AVX
//        cout << " transformPts3D _AVX " << n_pts << " pts \n";

//        cout << " Alignment 8 " << n_pts % 8 << " pts \n";
//        Eigen::MatrixXf input_xyz_transp = Eigen::MatrixXf::Ones(4,n_pts);
//        input_xyz_transp.block(0,0,3,n_pts) = input_pts.block(0,0,n_pts,3).transpose();
//        Eigen::MatrixXf aux = Rt * input_xyz_transp;
//        Eigen::MatrixXf xyz_src_transf2 = aux.block(0,0,3,n_pts).transpose();

//        // Eigen default ColMajor is assumed
//        assert(input_pts.cols() == 3);
//        assert(n_pts % 8 == 0);
//        output_pts.resize( n_pts, input_pts.cols() );

////        cout << " transformPts3D_avx check 1 \n";

//        const __m256 r11 = _mm256_set1_ps( Rt(0,0) );
//        const __m256 r12 = _mm256_set1_ps( Rt(0,1) );
//        const __m256 r13 = _mm256_set1_ps( Rt(0,2) );
//        const __m256 r21 = _mm256_set1_ps( Rt(1,0) );
//        const __m256 r22 = _mm256_set1_ps( Rt(1,1) );
//        const __m256 r23 = _mm256_set1_ps( Rt(1,2) );
//        const __m256 r31 = _mm256_set1_ps( Rt(2,0) );
//        const __m256 r32 = _mm256_set1_ps( Rt(2,1) );
//        const __m256 r33 = _mm256_set1_ps( Rt(2,2) );
//        const __m256 t1 = _mm256_set1_ps( Rt(0,3) );
//        const __m256 t2 = _mm256_set1_ps( Rt(1,3) );
//        const __m256 t3 = _mm256_set1_ps( Rt(2,3) );
////        cout << " transformPts3D_avx check 2 \n";

//        //Map<MatrixType>
//        const float *input_x = &input_pts(0,0);
//        const float *input_y = &input_pts(0,1);
//        const float *input_z = &input_pts(0,2);
//        float *output_x = &output_pts(0,0);
//        float *output_y = &output_pts(0,1);
//        float *output_z = &output_pts(0,2);

//        size_t block_end;
//        const Eigen::MatrixXf *input_pts_aligned = NULL;

//        // Take into account that the total number of points might not be a multiple of 8
//        if(n_pts % 8 == 0) // Data must be aligned
//        {
//            block_end = n_pts;
//        }
//        else
//        {
//            cout << " RegisterDense::transformPts3D UNALIGNED MATRIX pts \n";
//            size_t block_end = n_pts - n_pts % 8;
//            input_pts_aligned = new Eigen::MatrixXf(input_pts.block(0,0,block_end,3));
//            output_pts.resize( block_end, input_pts.cols() );
//            const float *input_x = &(*input_pts_aligned)(0,0);
//            const float *input_y = &(*input_pts_aligned)(0,1);
//            const float *input_z = &(*input_pts_aligned)(0,2);
//        }

//        cout << " transformPts3D_avx check 3 \n";

//        if(block_end > 1e2)
//        {
////    #if ENABLE_OPENMP
////    #pragma omp parallel for
////    #endif
//            for(size_t b=0; b < n_pts; b+=8)
//            {
//                cout << " transformPts3D_avx check 3a \n";

//                __m256 block_x_in = _mm256_load_ps(input_x+b);
//                cout << " transformPts3D_avx check 3aa \n";

//                __m256 block_y_in = _mm256_load_ps(input_y+b);
//                __m256 block_z_in = _mm256_load_ps(input_z+b);
//                cout << " transformPts3D_avx check 3b \n";
//                float check_avx[8];
//                _mm256_store_ps(check_avx, block_x_in);
//                cout << " check_avx \n";
//                for(size_t bit=0; bit < 8; bit++)
//                    cout << check_avx[bit] << " " << input_x[b+bit] << " \n";

//                __m256 block_x_out = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(r11, block_x_in), _mm256_mul_ps(r12, block_y_in) ), _mm256_mul_ps(r13, block_z_in) ), t1);
//                __m256 block_y_out = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(r21, block_x_in), _mm256_mul_ps(r22, block_y_in) ), _mm256_mul_ps(r23, block_z_in) ), t2);
//                __m256 block_z_out = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(r31, block_x_in), _mm256_mul_ps(r32, block_y_in) ), _mm256_mul_ps(r33, block_z_in) ), t3);

//                _mm256_store_ps(output_x+b, block_x_out);
//                _mm256_store_ps(output_y+b, block_y_out);
//                _mm256_store_ps(output_z+b, block_z_out);
//                //cout << b << " b " << input_x[b] << " " << input_y[b] << " " << input_z[b] << " pt " << output_x[b] << " " << output_y[b] << " " << output_z[b] << "\n";
//                cout << " transformPts3D_avx check 3d \n";
//            }
//            cout << " transformPts3D_avx check 3E \n";
//        }
//        else
//        {
//            for(size_t b=0; b < block_end; b+=8)
//            {
//                cout << " transformPts3D_avx check 4 \n";

//                __m256 block_x_in = _mm256_load_ps(input_x+b);
//                __m256 block_y_in = _mm256_load_ps(input_y+b);
//                __m256 block_z_in = _mm256_load_ps(input_z+b);

//                __m256 block_x_out = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(r11, block_x_in), _mm256_mul_ps(r12, block_y_in) ), _mm256_mul_ps(r13, block_z_in) ), t1);
//                __m256 block_y_out = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(r21, block_x_in), _mm256_mul_ps(r22, block_y_in) ), _mm256_mul_ps(r23, block_z_in) ), t2);
//                __m256 block_z_out = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(r31, block_x_in), _mm256_mul_ps(r32, block_y_in) ), _mm256_mul_ps(r33, block_z_in) ), t3);

//                _mm256_store_ps(output_x+b, block_x_out);
//                _mm256_store_ps(output_y+b, block_y_out);
//                _mm256_store_ps(output_z+b, block_z_out);

//                cout << " transformPts3D_avx check 4a \n";
//            }
//        }

//        if(n_pts % 8 != 0) // Compute the transformation of the unaligned points
//        {
//            delete input_pts_aligned;

//            // Compute the transformation of those points which do not enter in a block
//            const Eigen::Matrix3f rotation_transposed = Rt.block(0,0,3,3).transpose();
//            const Eigen::Matrix<float,1,3> translation_transposed = Rt.block(0,3,3,1).transpose();
//            output_pts.conservativeResize( n_pts, input_pts.cols() );
//            for(size_t i=block_end; i < n_pts; i++)
//            {
//                output_pts.block(i,0,1,3) = input_pts.block(i,0,1,3) * rotation_transposed + translation_transposed;
//            }
//        }

//        for(size_t i=0; i < n_pts; i++)
//        {
//            cout << " check " << output_pts.block(i,0,1,3) << " vs " << xyz_src_transf2.block(i,0,1,3) << endl;
//            assert( output_pts.block(i,0,1,3) == xyz_src_transf2.block(i,0,1,3) );
//        }
#endif


#if PRINT_PROFILING
    }
    double time_end = pcl::getTime();
    cout << " RegisterDense::transformPts3D " << input_pts.rows() << " took " << (time_end - time_start)*1000 << " ms. \n";
#endif
}
