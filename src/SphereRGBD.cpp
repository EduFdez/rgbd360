/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR and
 *                      INRIA Sophia Antipolis - LAGADIC Team
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

#include <SphereRGBD.h>

using namespace std;

/*! Constructor for the SphericalStereo sensor (outdoor sensor) */
SphereRGBD::SphereRGBD() :
    node(0),
    sphereCloud(new pcl::PointCloud<PointT>())
    //    bSphereCloudBuilt(false),
{

}

// Functions for SphereStereo images (outdoors)
/*! Load a spherical RGB-D image from the raw data stored in a binary file */
void Spieye360::loadDepth (const string &binaryDepthFile, const cv::Mat * mask)
{
#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

    ifstream file (binaryDepthFile.c_str(), ios::in | ios::binary);
    if (file)
    {
        char *header_property = new char[2]; // Read height_ and width_
        file.seekg (0, ios::beg);
        file.read (header_property, 2);
        unsigned short *height = reinterpret_cast<unsigned short*> (header_property);
        height_ = *height;

        //file.seekg (2, ios::beg);
        file.read (header_property, 2);
        unsigned short *width = reinterpret_cast<unsigned short*> (header_property);
        width_ = *width;
        //cout << "height_ " << height_ << " width_ " << width_ << endl;

        cv::Mat sphereDepth_aux(width_, height_, CV_32FC1);
        char *mem_block = reinterpret_cast<char*>(sphereDepth_aux.data);
        streampos size = height_*width_*4; // file.tellg() - streampos(4); // Header is 4 bytes: 2 unsigned short for height and width
        //file.seekg (4, ios::beg);
        file.read (mem_block, size);

        //            cv::Mat sphereDepth_aux2(height_, width_, CV_32FC1);
        //            for(int i = 0; i < height_; i++)
        //                for(int j = 0; j < width_; j++)
        //                    sphereDepth_aux2.at<float>(i,j) = *(reinterpret_cast<float*>(mem_block + 4*(j*height_+i)));
        //            cv::imshow( "sphereDepth", sphereDepth_aux2 );
        //            cv::waitKey(0);

        //Close the binary bile
        file.close();
        // sphereDepth.create(height_, width_, sphereDepth_aux.type());
        // cv::transpose(sphereDepth_aux, sphereDepth);
        //sphereDepth.create(640, width_, sphereDepth_aux.type());
        //cv::Rect region_of_interest = cv::Rect(8, 0, 640, width_); // Select only a portion of the image with height = 640 to facilitate the pyramid constructions
        sphereDepth.create(512, width_, sphereDepth_aux.type() );
        cv::Rect region_of_interest_transp = cv::Rect(90, 0, 512, width_); // Select only a portion of the image with height = width/4 (90 deg) with the FOV centered at the equator. This increases the performance of dense registration at the cost of losing some details from the upper/lower part of the images, which generally capture the sky and the floor.
        cv::transpose(sphereDepth_aux(region_of_interest_transp), sphereDepth); // The saved image is transposed wrt to the RGB img!

        //cv::imshow( "sphereDepth", sphereDepth );
        //cv::waitKey(0);

        if (mask && sphereDepth_aux.rows == mask->cols && sphereDepth_aux.cols == mask->rows){
            cv::Mat aux;
            cv::Rect region_of_interest = cv::Rect(0, 90, width_, 512); // This region of interest is the transposed of the above one (depth images are saved in disk as ColMajor)
            //cv::Rect region_of_interest = cv::Rect(0, 8, width_, 640);
            sphereDepth.copyTo(aux, (*mask)(region_of_interest) );
            sphereDepth = aux;
        }
        // cout << "height_ " << sphereDepth.rows << " width_ " << sphereDepth.cols << endl;
        //cv::imshow( "sphereDepth", sphereDepth );
        //cv::waitKey(0);
    }
    else
        cerr << "File: " << binaryDepthFile << " does NOT EXIST.\n";

    //    bSphereCloudBuilt = false; // The spherical PointCloud of the frame just loaded is not built yet

#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "loadDepth took " << double (time_end - time_start) << endl;
#endif
}

/*! Load a spherical RGB-D image from the raw data stored in a binary file */
void Spieye360::loadRGB(string &fileNamePNG)
{
#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

    ifstream file (fileNamePNG.c_str(), ios::in | ios::binary);
    if (file)
    {
        //            sphereRGB = cv::imread (fileNamePNG.c_str(), CV_LOAD_IMAGE_COLOR); // Full size 665x2048

        cv::Mat sphereRGB_aux = cv::imread (fileNamePNG.c_str(), CV_LOAD_IMAGE_COLOR);
        width_ = sphereRGB_aux.cols;
        sphereRGB.create(512, width_, sphereRGB_aux.type ());
        //sphereRGB.create(640, width_, sphereRGB_aux.type () );
        cv::Rect region_of_interest = cv::Rect(0,90, width_, 512); // Select only a portion of the image with height = width/4 (90 deg) with the FOV centered at the equator. This increases the performance of dense registration at the cost of losing some details from the upper/lower part of the images, which generally capture the sky and the floor.
        //cv::Rect region_of_interest = cv::Rect(0, 8, width_, 640); // Select only a portion of the image with height = 640 to facilitate the pyramid constructions
        sphereRGB = sphereRGB_aux (region_of_interest); // Size 640x2048
    }
    else
        cerr << "File: " << fileNamePNG << " does NOT EXIST.\n";

#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "loadRGB took " << double (time_end - time_start) << endl;
#endif
}
