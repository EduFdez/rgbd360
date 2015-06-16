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
 */

#ifndef CLOUD_RGBD_EXT
#define CLOUD_RGBD_EXT

#include "CloudRGBD.h"
#include <opencv2/core/eigen.hpp>
#include "SerializeFrameRGBD.h"

//#include <mrpt/base.h>
//#include <mrpt/pbmap.h>
//#include <pcl/features/integral_image_normal.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/ModelCoefficients.h>
//#include <pcl/segmentation/planar_region.h>
//#include <pcl/segmentation/organized_multi_plane_segmentation.h>
//#include <pcl/segmentation/organized_connected_component_segmentation.h>

/*! This class encapsulates a RGB-D frame (from Kinect or Asus XPL). In addition to
 *  the RGB and depth images, it contains the 3D point cloud data and the undistorted depth image.
 */
class CloudRGBD_Ext : public CloudRGBD
{
 protected:

//  /* ! True if the depth image is already undistorted */
//  bool undistortedDepthAvailabe;

 public:

  /*! Depth image used after intrinsic calibration */
  Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> m_depthEigUndistort;

  /*! Obtain the depth image in meters in eigen format */
  inline void loadDepthEigen()
  {
    cv::Mat depthInMeters;// = cv::Mat(m_depthImage.rows, m_depthImage.cols, CV_32FC1);
    getDepthImgMeters(depthInMeters);
    cv::cv2eigen(depthInMeters, m_depthEigUndistort);
//    cv::cv2eigen(m_depthImage, m_depthEigUndistort);
//      cv::Mat depthInMeters(m_depthImage.rows, m_depthImage.cols, CV_16UC1);
//      m_depthImage.convertTo(depthInMeters, CV_16UC1);
//      m_depthImage = depthInMeters;

//    m_depthImage.convertTo(m_depthImage, CV_16UC1);
  }

  /*! Get the undistorted point cloud from the frame RGB-D */
  inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPointCloudUndist()
  {
//    std::cout << "getPointCloudUndist \n";
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
//    std::cout << "height " << height << " width " << width << " \n";

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

          float z = m_depthEigUndistort.coeffRef(y,x);
//              std::cout << "Build " << z << " " << m_depthImage.at<float>(y,x) << std::endl;
          if(z>0 && z>=minDepth && z<=maxDepth) //If the point has valid depth information assign the 3D point to the point cloud
          {
            m_pointCloudPtr->points[width*y+x].x = (x - ox) * z * inv_fx;
            m_pointCloudPtr->points[width*y+x].y = (y - oy) * z * inv_fy;
//                    m_pointCloudPtr->points[width*y+x].x = -(y - oy) * z * inv_fy;
//                    m_pointCloudPtr->points[width*y+x].y = (x - ox) * z * inv_fx;
            m_pointCloudPtr->points[width*y+x].z = z;
          }
          else // assign a NAN value
          {
            m_pointCloudPtr->points[width*y+x].x = std::numeric_limits<float>::quiet_NaN ();
            m_pointCloudPtr->points[width*y+x].y = std::numeric_limits<float>::quiet_NaN ();
            m_pointCloudPtr->points[width*y+x].z = std::numeric_limits<float>::quiet_NaN ();
          }
        }
      }

      //save cloud...

      //The point cloud is now available
      pointCloudAvailable = true;
    }
    return m_pointCloudPtr;
  }

  /*! Get a downsampled point cloud from the RGBD frame */
  inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getDownsampledPointCloudUndist(const int &downsamplingStep)
  {
    //If the point cloud has been already computed, don't compute it again
    if(!pointCloudAvailable)
    {
        if(downsamplingStep == 1)
          getPointCloudUndist();

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
        m_pointCloudPtr->points.resize(m_pointCloudPtr->height * m_pointCloudPtr->width);

//        #if ENABLE_OPENMP_MULTITHREADING_FrameRGBD
//        #pragma omp parallel for
//        #endif
        int pt_idx = 0;
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
                float z = m_depthEigUndistort.coeffRef(y+yy,x+xx);
//              std::cout << "Build " << z << " " << m_depthImage.at<float>(y,x) << std::endl;
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

        //The point cloud is now available
        pointCloudAvailable = true;
    }
    return m_pointCloudPtr;
  }

//  /* ! Get a 3D point cloud from the RGBD data using the camera parameters*/
//  inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPointCloud(const Eigen::Matrix3f & cameraMatrix)
//  {
//    //If the point cloud has been already computed, don't compute it again
//    if(!pointCloudAvailable)
//    {
////      std::cout << "Build point cloud\n";
//        const float inv_fx = 1.f/cameraMatrix(0,0);
//        const float inv_fy = 1.f/cameraMatrix(1,1);
//        const float ox = cameraMatrix(0,2);
//        const float oy = cameraMatrix(1,2);
////
//
//        int height = m_rgbImage.rows;
//        int width = m_rgbImage.cols;
//
//        m_pointCloudPtr.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
//        m_pointCloudPtr->height = height;
//        m_pointCloudPtr->width = width;
//        m_pointCloudPtr->is_dense = false;
//        m_pointCloudPtr->points.resize(height*width);
//
//        #if ENABLE_OPENMP_MULTITHREADING_FrameRGBD
//        #pragma omp parallel for
//        #endif
//        for( int y = 0; y < height; y++ )
//        {
//            for( int x = 0; x < width; x++ )
//            {
//                int pt_idx = width*y+x;
//
//                cv::Vec3b& bgr = m_rgbImage.at<cv::Vec3b>(y,x);
//                m_pointCloudPtr->points[pt_idx].r = bgr[2];
//                m_pointCloudPtr->points[pt_idx].g = bgr[1];
//                m_pointCloudPtr->points[pt_idx].b = bgr[0];
//
//                float z = m_depthEigUndistort.coeffRef(y,x);
////                float z = m_depthImage.at<float>(y,x); //convert from milimeters to meters
////              std::cout << "Build " << z << " " << m_depthImage.at<float>(y,x) << std::endl;
//                if(z>0 && z>=minDepth && z<=maxDepth) //If the point has valid depth information assign the 3D point to the point cloud
//                {
//                    m_pointCloudPtr->points[pt_idx].x = (x - ox) * z * inv_fx;
//                    m_pointCloudPtr->points[pt_idx].y = (y - oy) * z * inv_fy;
////                    m_pointCloudPtr->points[pt_idx].x = -(y - oy) * z * inv_fy;
////                    m_pointCloudPtr->points[pt_idx].y = (x - ox) * z * inv_fx;
//                    m_pointCloudPtr->points[pt_idx].z = z;
//                }
//                else //else, assign a NAN value
//                {
//                    m_pointCloudPtr->points[pt_idx].x = std::numeric_limits<float>::quiet_NaN ();
//                    m_pointCloudPtr->points[pt_idx].y = std::numeric_limits<float>::quiet_NaN ();
//                    m_pointCloudPtr->points[pt_idx].z = std::numeric_limits<float>::quiet_NaN ();
//                }
//            }
//        }
//
//        //The point cloud is now available
//        pointCloudAvailable = true;
//    }
//    return m_pointCloudPtr;
//  }


//  /*! The PbMap of segmented */
//  mrpt::pbmap::PbMap planes;
//
//  /*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
//      in the frame of reference of the omnidirectional camera
//  */
//  void getPlanesInFrame()
//  {
//    // Segment planes
////    std::cout << "extractPlaneFeatures, size " << m_pointCloudPtr->size() << "\n";
//    double extractPlanes_start = pcl::getTime();
//  assert(m_pointCloudPtr->height > 1 && m_pointCloudPtr->width > 1);
//
//    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
////      ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);
////      ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);
//    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
////      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
//    ne.setMaxDepthChangeFactor (0.02); // For VGA: 0.02f, 10.01
//    ne.setNormalSmoothingSize (8.0f);
//    ne.setDepthDependentSmoothing (true);
//
//    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
////      mps.setMinInliers (std::max(uint32_t(40),m_pointCloudPtr->height*2));
//    mps.setMinInliers (80);
//    mps.setAngularThreshold (0.039812); // (0.017453 * 2.0) // 3 degrees
//    mps.setDistanceThreshold (0.02); //2cm
////    cout << "PointCloud size " << m_pointCloudPtr->size() << endl;
//
//    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
//    ne.setInputCloud ( m_pointCloudPtr );
//    ne.compute (*normal_cloud);
//
//    mps.setInputNormals (normal_cloud);
//    mps.setInputCloud ( m_pointCloudPtr );
//    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
//    std::vector<pcl::ModelCoefficients> model_coefficients;
//    std::vector<pcl::PointIndices> inlier_indices;
//    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
//    std::vector<pcl::PointIndices> label_indices;
//    std::vector<pcl::PointIndices> boundary_indices;
//    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);
//
//    // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
//    unsigned single_cloud_size = m_pointCloudPtr->size();
//    Eigen::Matrix4f Rt = calib->getRt_id(sensor_id);
//    for (size_t i = 0; i < regions.size (); i++)
//    {
//      mrpt::pbmap::Plane plane;
//
//      plane.v3center = regions[i].getCentroid ();
//      plane.v3normal = Eigen::Vector3f(model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);
//      if( plane.v3normal.dot(plane.v3center) > 0)
//      {
//        plane.v3normal = -plane.v3normal;
////          plane.d = -plane.d;
//      }
//      plane.curvature = regions[i].getCurvature ();
////    cout << i << " getCurvature\n";
//
////        if(plane.curvature > max_curvature_plane)
////          continue;
//
//      // Extract the planar inliers from the input cloud
//      pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
//      extract.setInputCloud ( m_pointCloudPtr );
//      extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices[i]) );
//      extract.setNegative (false);
//      extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
//      plane.inliers.resize(inlier_indices[i].indices.size());
//      for(size_t j=0; j<inlier_indices[i].indices.size(); j++)
//        plane.inliers[j] = inlier_indices[i].indices[j] + sensor_id*single_cloud_size;
//
//      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr contourPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);
//      contourPtr->points = regions[i].getContour();
//
////    cout << "Extract contour\n";
//      if(contourPtr->size() != 0)
//      {
//        plane.calcConvexHull(contourPtr);
//      }
//      else
//      {
////        assert(false);
//      std::cout << "HULL 000\n" << plane.planePointCloudPtr->size() << std::endl;
//        static pcl::VoxelGrid<pcl::PointXYZRGBA> plane_grid;
//        plane_grid.setLeafSize(0.05,0.05,0.05);
//        plane_grid.setInputCloud (plane.planePointCloudPtr);
//        plane_grid.filter (*contourPtr);
//        plane.calcConvexHull(contourPtr);
//      }
//
////        assert(contourPtr->size() > 0);
////        plane.calcConvexHull(contourPtr);
////    cout << "calcConvexHull\n";
//      plane.computeMassCenterAndArea();
////    cout << "Extract convexHull\n";
//      // Discard small planes
//      if(plane.areaHull < min_area_plane)
//        continue;
//
//      plane.d = -plane.v3normal .dot( plane.v3center );
//
//      plane.calcElongationAndPpalDir();
//      // Discard narrow planes
//      if(plane.elongation > max_elongation_plane)
//        continue;
//
////      double color_start = pcl::getTime();
//      plane.calcPlaneHistH();
//      plane.calcMainColor2();
////      double color_end = pcl::getTime();
////    std::cout << "color in " << (color_end - color_start)*1000 << " ms\n";
//
////      color_start = pcl::getTime();
//      plane.transform(Rt);
////      color_end = pcl::getTime();
////    std::cout << "transform in " << (color_end - color_start)*1000 << " ms\n";
//
//      bool isSamePlane = false;
//      if(plane.curvature < max_curvature_plane)
//        for (size_t j = 0; j < planes.vPlanes.size(); j++)
//          if( planes.vPlanes[j].curvature < max_curvature_plane && planes.vPlanes[j].isSamePlane(plane, 0.99, 0.05, 0.2) ) // The planes are merged if they are the same
//          {
////          cout << "Merge local region\n";
//            isSamePlane = true;
////            double time_start = pcl::getTime();
//            planes.vPlanes[j].mergePlane2(plane);
////            double time_end = pcl::getTime();
////          std::cout << " mergePlane2 took " << double (time_start - time_end) << std::endl;
//
//            break;
//          }
//      if(!isSamePlane)
//      {
////          plane.calcMainColor();
//        plane.id = planes.vPlanes.size();
//        planes.vPlanes.push_back(plane);
//      }
//    }
////      double extractPlanes_end = pcl::getTime();
////    std::cout << "getPlanesInFrame in " << (extractPlanes_end - extractPlanes_start)*1000 << " ms\n";
//  }
};
#endif
