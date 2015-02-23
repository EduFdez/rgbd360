/*
 *  Copyright (c) 2012, Universidad de Málaga - Grupo MAPIR
 *
 *  http://code.google.com/p/photoconsistency-visual-odometry/
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
 */

#ifndef SERIALIZE_FRAME_RGBD
#define SERIALIZE_FRAME_RGBD

#include "FrameRGBD.h"

//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <boost/lexical_cast.hpp>

//#include "../third_party/cvmat_serialization.h"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <fstream>


void inline getMatrixNumberRepresentationOf_uint64_t(const uint64_t number, cv::Mat & matrixNumberRepresentation)
{

    //Determine the number of digits of the number
    int num_digits = 0;
    uint64_t number_aux = number;
    while(number_aux > 0)
    {
      num_digits++;
      number_aux/=10;
    }

    //Compute the matrix representation of the number
    matrixNumberRepresentation = cv::Mat::zeros(1,num_digits,CV_8U);

    uint64_t remainder = number;
    for(int digitIndex=0;digitIndex<num_digits;digitIndex++)
    {
      if(remainder==0){break;}

      uint8_t greaterDigit;
      uint64_t dividend = remainder;
      uint64_t divisor = pow(10,num_digits-1-digitIndex);
      uint64_t quotient = remainder / divisor;
      greaterDigit = quotient;
      matrixNumberRepresentation.at<uint8_t>(0,digitIndex)=greaterDigit;
      remainder = dividend - divisor * quotient;
    }
}

void inline get_uint64_t_ofMatrixRepresentation(const cv::Mat & matrixNumberRepresentation, uint64_t & number)
{
    int num_digits = matrixNumberRepresentation.cols;

    number=0;
    uint64_t power10=1;
    for(int digitIndex=num_digits-1;digitIndex>=0;digitIndex--)
    {
        number += power10 * uint64_t(matrixNumberRepresentation.at<uint8_t>(0,digitIndex));

        power10 = power10 * 10;
    }
}


class SerializeFrameRGBD
{
 public:

  SerializeFrameRGBD()
  {
  }

  void saveToFile(std::string fileName, FrameRGBD &frameRGBD)
  {
    std::ofstream ofs(fileName.append(".bin").c_str(), std::ios::out | std::ios::binary);

    {   // use scope to ensure archive goes out of scope before stream

      boost::archive::binary_oarchive oa(ofs);

      cv::Mat timeStampMatrix;
      getMatrixNumberRepresentationOf_uint64_t(frameRGBD.getTimeStamp(),timeStampMatrix);
      oa << frameRGBD.getDepthImage() << frameRGBD.getRGBImage() << timeStampMatrix;
    }

    ofs.close();

  }

  void loadFromFile(std::string fileName, FrameRGBD &frameRGBD)
  {
      std::ifstream ifs(fileName.append(".bin").c_str(), std::ios::in | std::ios::binary);

      { // use scope to ensure archive goes out of scope before stream
        boost::archive::binary_iarchive ia(ifs);

        cv::Mat timeStampMatrix;
        ia >> frameRGBD.getDepthImage() >> frameRGBD.getRGBImage() >> timeStampMatrix;
        uint64_t timeStamp;
        get_uint64_t_ofMatrixRepresentation(timeStampMatrix,timeStamp);
        frameRGBD.setTimeStamp(timeStamp);
      }

      ifs.close();

      //Initialize the intensity image with an empty matrix
      cv::Mat intensityImage = cv::Mat();
      frameRGBD.setIntensityImage(intensityImage);

//      //Initialize the point cloud with an empty pointer
//      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloudPtr(new pcl::PointCloud<pcl::PointXYZRGBA>());
//      frameRGBD.setPointCloud(pointCloudPtr);
  }
};
#endif
