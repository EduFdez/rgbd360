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

#include <mrpt/utils/mrpt_macros.h>
#include <pcl/console/parse.h>

#include <Frame360.h>
#include <Frame360_Visualizer.h>

using namespace std;
using namespace mrpt::utils;

/*! This class' encapsulates some basic functionality to retrieve omnidirectional stereo images (RGB+Depth) from a given folder.
 */
class SequenceStereo360
{
private:

    /*! Path to the folder containing the images of the sequence */
    string path_images;

    /*! Number of frames of the dataset */
    int n_size;

    /*! First frame of the sequence to be considered (set by the user) */
    int n_first_frame;

    /*! Is a RGB-D sequence? */
    bool has_depth;

    /*! Mask the INRIA's car (the front part is visible in the images, and it's not static wrt the scene) */
    cv::Mat mask_car;

public:

    /*! Index of the current frame */
    int n_current_frame;

    SequenceStereo360(const string &path, const bool hasDepth = true) :
        path_images(path),
        n_first_frame(0),
        has_depth(hasDepth)
    {
        mask_car = cv::imread("/Data/Shared_Lagadic/useful_code/mask_car_.png",0);
        //  cv::imshow( "mask_car", mask_car );
        //  cv::waitKey(0);
    }

    /*! Read the dataset from the image specified by "rgb_1st" */
    void readDataset(const string & path_folder)
    {
        ASSERT_ ( fexists(path_folder.c_str()) ); // CHeck that the directory exists

        string img_rgb;
        string img_depth; //img_depth_fused

        const string rgb_prefix = "rgb";
        const string rgb_sufix = ".png";
        const string depth_prefix = "depth";
        //const string depth_fused_prefix = "depthF";
        const string depth_sufix = ".raw";

        string frame_num_s;

        // Check all the frames in the folder to retrieve the size of the dataset
        n_current_frame = n_first_frame;
        if(!has_depth)
        {
            do{
                ++n_current_frame;
                frame_num_s = mrpt::format("%07d", n_current_frame);
                img_rgb = path_folder + "/" + rgb_prefix + frame_num_s + rgb_sufix;

            }while( fexists(rgb.c_str()) );
        }
        else
        {
            do{
                ++n_current_frame;
                frame_num_s = mrpt::format("%07d", n_current_frame);
                img_rgb = path_folder + "/" + rgb_prefix + frame_num_s + rgb_sufix;
                img_depth = path_folder + "/" + depth_prefix + frame_num_s + depth_sufix;

                //std::cerr << "\n... The corresponding depth image does not exist!!! \n";

            }while( fexists(rgb.c_str()) && fexists(img_depth.c_str()) ); // && fexists(img_depth_fused.c_str())
        }

        n_size = n_current_frame - n_first_frame;
        std::cout << "The dataset contains " << n_size  << " frames. \n";
    }

    /*! Read the dataset from the image specified by "rgb_1st" */
    void readDatasetFrom1stImg(const string &rgb_1st)
    {
        ASSERT_ ( fexists(rgb1.c_str()) );
        //std::cout << "  rgb1:\t\t" << rgb1 << std::endl;

        string img_rgb = rgb_1st;
        string img_depth; //img_depth_fused

        const string rgb_prefix = "rgb";
        const string rgb_sufix = ".png";
        const string depth_prefix = "depth";
        //const string depth_fused_prefix = "depthF";
        const string depth_sufix = ".raw";
        const size_t char_prefix = 3;
        const size_t char_suffix = 4;
        const size_t idx_length = 7;

        size_t pos_last_slash = img_rgb.find_last_of("/\\");
        string path_folder = img_rgb.substr(0, pos_last_slash);
        string img_name = img_rgb.substr(pos_last_slash+1);
        string frame_num_s = img_name.substr(char_prefix, img_name.length()-char_suffix);

        if( rgb_sufix.compare( img_name.substr(img_name.length()-idx_length-char_suffix) ) != 0 ) // If the first string correspond to a pointCloud path
        {
            std::cerr << "\n... INVALID IMAGE FILE!!! \n";
            return;
        }
        if( !fexists(img_rgb.c_str()) )
        {
            std::cerr << "\n... The image provided does not exist!!! \n";
            return;
        }

        n_current_frame = n_first_frame;
        readDataset(path_folder);
        //std::cout << "The dataset contains " << n_size  << " frames from image " << frame_num_s << " to the end. \n";
    }

};
