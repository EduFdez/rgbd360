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

#pragma once

#include "definitions.h"
//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

/*! This class defines the omnidirectional RGB-D frame 'SphereRGBD'. */
class SphereRGBD
{
  protected:

    /*! Time-stamp of the spherical frame (it corresponds to the last capture of the NUM_ASUS_SENSORS sensors, as they are not syncronized) */
    uint64_t timeStamp;

    /*! Spherical (omnidirectional) RGB image. The term spherical means that the same solid angle is assigned to each pixel */
    cv::Mat sphereRGB;

    /*! Spherical (omnidirectional) Depth image*/
    cv::Mat sphereDepth;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> sphereDepth_;

  public:

    /*! Frame height */
    unsigned short height_;

    /*! Frame width */
    unsigned short width_;

    /*! Constructor for the SphericalStereo sensor (outdoor sensor) */
    SphereRGBD();

    /*! Set the spherical image timestamp */
    inline void setTimeStamp(uint64_t timestamp)
    {
        timeStamp = timestamp;
    }

    /*! Get the spherical image RGB */
    inline cv::Mat & getImgRGB()
    {
        return sphereRGB;
    }

    /*! Get the spherical image Depth */
    inline cv::Mat & getImgDepth()
    {
        return sphereDepth;
    }

    /*! Load a spherical RGB-D image from the raw data stored in a binary file */
    void loadFrame(std::string &binaryFile);

    /*Get the average intensity*/
    inline int getAverageIntensity(int sample = 1)
    {
        int av_intensity = 0;

        //    getIntensityImage();
        //
        //    #pragma omp parallel num_threads(NUM_ASUS_SENSORS)
        //    {
        //      int sensor_id = omp_get_thread_num();
        //      int sum_intensity[NUM_ASUS_SENSORS];
        //      std::fill(sum_intensity, sum_intensity+NUM_ASUS_SENSORS, 0);
        //      for(unsigned i=0; i < NUM_ASUS_SENSORS; i++)
        //        if(sensor_id == i)
        //        {
        //          frameRGBD_[sensor_id].getIntensityImage(); //Make sure that the intensity image has been computed
        //          sum_intensity[sensor_id] = frameRGBD_[sensor_id].getAverageIntensity(sample); //Make sure that the intensity image has been computed
        //        }
        //    }
        //    for(unsigned i=1; i < NUM_ASUS_SENSORS; i++)
        //      sum_intensity[0] += sum_intensity[i];
        //
        //    return floor(sum_intensity[0] / 8.0 + 0.5);
        return av_intensity;
    }

    /*! Load a spherical RGB-D image from the raw data stored in a binary file */
    void loadDepth (const std::string &binaryDepthFile, const cv::Mat * mask = NULL);

    /*! Load a spherical RGB-D image from the raw data stored in a binary file */
    void loadRGB(std::string &fileNamePNG);

};
