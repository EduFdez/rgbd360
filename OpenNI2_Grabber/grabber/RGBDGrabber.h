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
 * Author: Eduardo Fernandez-Moral
 * This code is an adaptation of a previous work of Miguel Algaba
 */

#ifndef RGBDGRABBER
#define RGBDGRABBER

#include <FrameRGBD.h>

/*!Abstract class that especifies the functionality of a generic RGBD grabber.*/
class RGBDGrabber
{

 protected:

  /*!Frame dimensions.*/
  int width, height;
//
//  /*! Resolution mode of the device (default is QVGA = 320x240) */
//  enum Resolution
//  {
//    VGA = 1,
//    QVGA = 2,
//    QQVGA = 4,
//  } resolution;

  cv::Mat currentRGBImg;
//  cv::Mat currentBGRImg;
  cv::Mat currentDepthImg;
//  Eigen::MatrixXf currentDepthEigen;

 public:

  /*!Initializes the grabber object*/
  virtual void init()=0;

  /*!Retains the current RGBD frame.*/
  virtual void grab(FrameRGBD*)=0;

  /*!Stop grabing RGBD frames.*/
  virtual void stop()=0;
};
#endif
