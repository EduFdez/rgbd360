/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR
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
 *  Author: Eduardo Fernandez-Moral
 *  This code is an adaptation of a previous work of Miguel Algaba
 */

#ifndef FRAME_RGBD
#define FRAME_RGBD

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*! This class contains the RGB and depth images, together with the timestamp of a RGBD observation (e.g. from Kinect).
 */
class FrameRGBD
{
 protected:

  /*!RGB image*/
  cv::Mat m_rgbImage;

  /*!Intensity image (grayscale version of the RGB image)*/
  cv::Mat m_intensityImage;

  /*!Depth image*/
  cv::Mat m_depthImage;

  /*!True if the intensity image is available, false otherwise*/
  bool intensityImageAvailable;

  /*!Timestamp of the RGBD frame*/
  uint64_t m_timeStamp;

 public:

//  /*!Frame dimensions.*/
//  int width, height;

  /*! Constructor */
  FrameRGBD() :
    intensityImageAvailable(false)
  {
  };

//  ~FrameRGBD(){};

  /*!Set a RGB image to the RGBD frame.*/
  inline void setRGBImage(const cv::Mat & rgbImage){m_rgbImage = rgbImage;}

  /*!Return the RGB image of the RGBD frame.*/
  inline cv::Mat & getRGBImage(){return m_rgbImage;}

  /*!Set a depth image to the RGBD frame.*/
  inline void setDepthImage(const cv::Mat & depthImage){m_depthImage = depthImage;}

  /*!Return the depth image of the RGBD frame.*/
  inline cv::Mat & getDepthImage(){return m_depthImage;}

  /*!Return the depth image of the RGBD frame.*/
  inline void getDepthImgMeters(cv::Mat &depthInMeters)
  {
    depthInMeters = cv::Mat(m_depthImage.rows, m_depthImage.cols, CV_32FC1);
    m_depthImage.convertTo( depthInMeters, CV_32FC1, 0.001 ); //CV_16UC1
  }

  /*!Set the RGBD frame timestamp*/
  inline void setTimeStamp(uint64_t timeStamp){m_timeStamp=timeStamp;};

  /*!Return the RGBD frame timestamp*/
  inline uint64_t getTimeStamp(){return m_timeStamp;};

  /*!Set the intensity image*/
  inline void setIntensityImage(const cv::Mat & intensityImage){m_intensityImage = intensityImage;}

  /*!Get the grayscale image from RGB*/
  inline cv::Mat & getIntensityImage()
  {
    //If the intensity image has been already computed, don't compute it again
    if(!intensityImageAvailable)
    {
      cv::cvtColor(m_rgbImage,m_intensityImage,CV_BGR2GRAY);

      //The intensity image is now available
      intensityImageAvailable = true;
    }
    return m_intensityImage;
  }

  /*!Get the average intensity*/
  inline int getAverageIntensity(int sample = 1)
  {
    //Make sure that the intensity image has been computed
    getIntensityImage();
    unsigned sum_intensity = 0;
    unsigned sample_size = (m_intensityImage.rows/sample) * (m_intensityImage.cols/sample);
    for(unsigned r=0; r < m_intensityImage.rows; r+=sample)
      for(unsigned c=0; c < m_intensityImage.cols; c+=sample)
        sum_intensity += static_cast<unsigned>(m_intensityImage.at<uint8_t>(r,c));

    return sum_intensity / sample_size;
  }
};
#endif
