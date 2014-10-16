/*
 *  Copyright (c) 2012, Universidad de Málaga - Grupo MAPIR
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

#ifndef CLOUD_RGBD
#define CLOUD_RGBD

#define ENABLE_OPENMP_MULTITHREADING_FrameRGBD 1

#include "FrameRGBD.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/*! This class encapsulates a RGB-D frame (from Kinect or Asus XPL), in addition to
 *  the RGB and depth images, it contains the 3D point cloud data.
 */
class CloudRGBD : public FrameRGBD
{
 protected:

  /*!Coloured point cloud*/
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_pointCloudPtr;

  /*!Max pointcloud depth*/
  float maxDepth;

  /*!Min pointcloud depth*/
  float minDepth;

 public:

  /*!True if the coloured point cloud is available, false otherwise*/
  bool pointCloudAvailable;

  /*! Constructor */
  CloudRGBD() :
    pointCloudAvailable(false),
    minDepth(0.3), // Default min depth
    maxDepth(10.0) // Default max depth
  {
  };

//  ~CloudRGBD(){};

  /*!Set the max depth value for the point cloud points.*/
  inline void setMaxPointCloudDepth(float maxD)
  {
    maxDepth = maxD;
  }

  /*!Set the min depth value for the point cloud points.*/
  inline void setMinPointCloudDepth(float minD)
  {
    minDepth = minD;
  }

  /*!Return the max depth value for point cloud points.*/
  inline float getMaxPointCloudDepth()
  {
    return maxDepth;
  }

  /*!Return the min depth value for point cloud points.*/
  inline float getMinPointCloudDepth()
  {
    return minDepth;
  }

  /*!Set the pointCloud of the RGBD frame.*/
  inline void setPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloud)
  {
    m_pointCloudPtr = pointCloud;
  }

//  /*!Get the point cloud provided by the sensor*/
//  inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPointCloud(){return m_pointCloudPtr;}

  /*!Gets a 3D coloured point cloud from the RGBD data using the camera parameters*/
  inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPointCloud()
  {
    //If the point cloud has been already computed, don't compute it again
    if(!pointCloudAvailable)
    {
//      std::cout << "Build point cloud\n";
      assert(m_rgbImage.rows == m_depthImage.rows && m_rgbImage.cols == m_depthImage.cols);

      const int height = m_rgbImage.rows;
      const int width = m_rgbImage.cols;

      const float res_factor_VGA = width / 640.0;
      const float focal_length = 525 * res_factor_VGA;
      const float inv_fx = 1.f/focal_length;
      const float inv_fy = 1.f/focal_length;
      const float ox = width/2 - 0.5;
      const float oy = height/2 - 0.5;

      m_pointCloudPtr.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
      m_pointCloudPtr->height = height;
      m_pointCloudPtr->width = width;
      m_pointCloudPtr->is_dense = false;
      m_pointCloudPtr->points.resize(height*width);

      if(m_depthImage.type() == CV_16U) // The image pixels are presented in millimetres
      {
          float minDepth_ = minDepth*1000;
          float maxDepth_ = maxDepth*1000;
#if ENABLE_OPENMP_MULTITHREADING_FrameRGBD
#pragma omp parallel for
#endif
          for( int y = 0; y < height; y++ )
          {
            for( int x = 0; x < width; x++ )
            {
              cv::Vec3b& bgr = m_rgbImage.at<cv::Vec3b>(y,x);
              m_pointCloudPtr->points[width*y+x].r = bgr[2];
              m_pointCloudPtr->points[width*y+x].g = bgr[1];
              m_pointCloudPtr->points[width*y+x].b = bgr[0];

              float z = 0.001 * m_depthImage.at<unsigned short>(y,x); //convert from milimeters to meters
              //std::cout << "Build " << z << std::endl;
              //if(z>0 && z>=minDepth_ && z<=maxDepth_) //If the point has valid depth information assign the 3D point to the point cloud
              if(z>=minDepth_ && z<=maxDepth_) //If the point has valid depth information assign the 3D point to the point cloud
              {
                m_pointCloudPtr->points[width*y+x].x = (x - ox) * z * inv_fx;
                m_pointCloudPtr->points[width*y+x].y = (y - oy) * z * inv_fy;
    //                    m_pointCloudPtr->points[width*y+x].x = -(y - oy) * z * inv_fy;
    //                    m_pointCloudPtr->points[width*y+x].y = (x - ox) * z * inv_fx;
                m_pointCloudPtr->points[width*y+x].z = z;
              }
              else //else, assign a NAN value
              {
                m_pointCloudPtr->points[width*y+x].x = std::numeric_limits<float>::quiet_NaN ();
                m_pointCloudPtr->points[width*y+x].y = std::numeric_limits<float>::quiet_NaN ();
                m_pointCloudPtr->points[width*y+x].z = std::numeric_limits<float>::quiet_NaN ();
              }
            }
          }
      }
      else if(m_depthImage.type() == CV_32F) // The image pixels are presented in metres
      {
#if ENABLE_OPENMP_MULTITHREADING_FrameRGBD
#pragma omp parallel for
#endif
          for( int y = 0; y < height; y++ )
          {
              for( int x = 0; x < width; x++ )
              {
                  cv::Vec3b& bgr = m_rgbImage.at<cv::Vec3b>(y,x);
                  m_pointCloudPtr->points[width*y+x].r = bgr[2];
                  m_pointCloudPtr->points[width*y+x].g = bgr[1];
                  m_pointCloudPtr->points[width*y+x].b = bgr[0];

                  float z = m_depthImage.at<float>(y,x); //convert from milimeters to meters
                  //std::cout << "Build " << z << std::endl;
                  //if(z>0 && z>=minDepth && z<=maxDepth) //If the point has valid depth information assign the 3D point to the point cloud
                  if(z>=minDepth && z<=maxDepth) //If the point has valid depth information assign the 3D point to the point cloud
                  {
                      m_pointCloudPtr->points[width*y+x].x = (x - ox) * z * inv_fx;
                      m_pointCloudPtr->points[width*y+x].y = (y - oy) * z * inv_fy;
                      //                    m_pointCloudPtr->points[width*y+x].x = -(y - oy) * z * inv_fy;
                      //                    m_pointCloudPtr->points[width*y+x].y = (x - ox) * z * inv_fx;
                      m_pointCloudPtr->points[width*y+x].z = z;
                  }
                  else //else, assign a NAN value
                  {
                      m_pointCloudPtr->points[width*y+x].x = std::numeric_limits<float>::quiet_NaN ();
                      m_pointCloudPtr->points[width*y+x].y = std::numeric_limits<float>::quiet_NaN ();
                      m_pointCloudPtr->points[width*y+x].z = std::numeric_limits<float>::quiet_NaN ();
                  }
              }
          }
      }

      //The point cloud is now available
      pointCloudAvailable = true;
    }
    return m_pointCloudPtr;
  }

  /*! Get a downsampled point cloud from the RGBD frame */
  inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getDownsampledPointCloud(const int &downsamplingStep)
  {
    //If the point cloud has been already computed, don't compute it again
//    if(!pointCloudAvailable)
    {
        if(downsamplingStep == 1)
          getPointCloud();

        assert(m_rgbImage.rows == m_depthImage.rows && m_rgbImage.cols == m_depthImage.cols);

        const int height = m_rgbImage.rows;
        const int width = m_rgbImage.cols;

        const float res_factor_VGA = width / 640.0;
        const float focal_length = 525 * res_factor_VGA;
        const float inv_fx = 1.f/focal_length;
        const float inv_fy = 1.f/focal_length;
        const float ox = width/2 - 0.5;
        const float oy = height/2 - 0.5;

        m_pointCloudPtr.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
        m_pointCloudPtr->height = height/downsamplingStep;
        m_pointCloudPtr->width = width/downsamplingStep;
        m_pointCloudPtr->is_dense = false;
        m_pointCloudPtr->points.resize(height*width/(downsamplingStep*downsamplingStep));


        if(m_depthImage.type() == CV_16U)
        {
            int pt_idx = 0;
#if ENABLE_OPENMP_MULTITHREADING_FrameRGBD
#pragma omp parallel for shared(pt_idx)
#endif
            for( int y = 0; y < height; y+=downsamplingStep )
            {
                for( int x = 0; x < width; x+=downsamplingStep, ++pt_idx)
                {
                    // int pt_idx = m_pointCloudPtr->width*(y/downsamplingStep)+(x/downsamplingStep);

                    cv::Vec3b& bgr = m_rgbImage.at<cv::Vec3b>(y,x);
                    m_pointCloudPtr->points[pt_idx].r = bgr[2];
                    m_pointCloudPtr->points[pt_idx].g = bgr[1];
                    m_pointCloudPtr->points[pt_idx].b = bgr[0];

                    bool nanPoint = true;
                    for( int yy = 0; yy < downsamplingStep; yy++ )
                        for( int xx = 0; xx < downsamplingStep; xx++ )
                        {
                            float z = 0.001 * m_depthImage.at<unsigned short>(y+yy,x+xx); //convert from milimeters to meters
                            //float z = m_depthImage.at<float>(y,x); //convert from milimeters to meters
                            //std::cout << "Build " << z << " " << m_depthImage.at<float>(y,x) << std::endl;
                            if(z>=minDepth && z<=maxDepth) //If the point has valid depth information assign the 3D point to the point cloud
                            {
                                m_pointCloudPtr->points[pt_idx].x = (x - ox) * z * inv_fx;
                                m_pointCloudPtr->points[pt_idx].y = (y - oy) * z * inv_fy;
                                //                    m_pointCloudPtr->points[pt_idx].x = -(y - oy) * z * inv_fy;
                                //                    m_pointCloudPtr->points[pt_idx].y = (x - ox) * z * inv_fx;
                                m_pointCloudPtr->points[pt_idx].z = z;

                                nanPoint = false;
                                yy = downsamplingStep;
                                break;
                            }
                        }
                    if(nanPoint) // assign a NAN value
                    {
                        m_pointCloudPtr->points[pt_idx].x = std::numeric_limits<float>::quiet_NaN ();
                        m_pointCloudPtr->points[pt_idx].y = std::numeric_limits<float>::quiet_NaN ();
                        m_pointCloudPtr->points[pt_idx].z = std::numeric_limits<float>::quiet_NaN ();
                    }
                }
            }
        }
        else if(m_depthImage.type() == CV_32F)
        {
            int pt_idx = 0;
#if ENABLE_OPENMP_MULTITHREADING_FrameRGBD
#pragma omp parallel for shared(pt_idx)
#endif
            for( int y = 0; y < height; y+=downsamplingStep )
            {
                for( int x = 0; x < width; x+=downsamplingStep, ++pt_idx)
                {
                    //              int pt_idx = m_pointCloudPtr->width*(y/downsamplingStep)+(x/downsamplingStep);

                    cv::Vec3b& bgr = m_rgbImage.at<cv::Vec3b>(y,x);
                    m_pointCloudPtr->points[pt_idx].r = bgr[2];
                    m_pointCloudPtr->points[pt_idx].g = bgr[1];
                    m_pointCloudPtr->points[pt_idx].b = bgr[0];

                    bool nanPoint = true;
                    for( int yy = 0; yy < downsamplingStep; yy++ )
                        for( int xx = 0; xx < downsamplingStep; xx++ )
                        {
                            float z = m_depthImage.at<float>(y,x); //convert from milimeters to meters
                            //std::cout << "Build " << z << " " << m_depthImage.at<float>(y,x) << std::endl;
                            if(z>=minDepth && z<=maxDepth) //If the point has valid depth information assign the 3D point to the point cloud
                            {
                                m_pointCloudPtr->points[pt_idx].x = (x - ox) * z * inv_fx;
                                m_pointCloudPtr->points[pt_idx].y = (y - oy) * z * inv_fy;
                                //                    m_pointCloudPtr->points[pt_idx].x = -(y - oy) * z * inv_fy;
                                //                    m_pointCloudPtr->points[pt_idx].y = (x - ox) * z * inv_fx;
                                m_pointCloudPtr->points[pt_idx].z = z;

                                nanPoint = false;
                                yy = downsamplingStep;
                                break;
                            }
                        }
                    if(nanPoint) // assign a NAN value
                    {
                        m_pointCloudPtr->points[pt_idx].x = std::numeric_limits<float>::quiet_NaN ();
                        m_pointCloudPtr->points[pt_idx].y = std::numeric_limits<float>::quiet_NaN ();
                        m_pointCloudPtr->points[pt_idx].z = std::numeric_limits<float>::quiet_NaN ();
                    }
                }
            }
        }

        //The point cloud is now available
        pointCloudAvailable = true;
    }
    return m_pointCloudPtr;
  }
};
#endif
