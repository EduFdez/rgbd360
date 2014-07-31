/*
 *  Copyright (c) 2012, Universidad de Málaga - Grupo MAPIR and
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
 *  Author: Eduardo Fernandez-Moral
 */

#ifndef FRAME360_FILTER_H
#define FRAME360_FILTER_H

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

typedef pcl::PointXYZRGBA PointT;

/*! This class is used to filter a point cloud in place.
 */
class FilterPointCloud
{
 private:

  /*! Filter points in the direction x */
  pcl::PassThrough<PointT> filter_pass_x;

  /*! Filter points in the direction y */
  pcl::PassThrough<PointT> filter_pass_y;

  /*! Filter points in the direction z */
  pcl::PassThrough<PointT> filter_pass_z;

  /*! Voxel filter */
  pcl::VoxelGrid<pcl::PointXYZRGBA> filter_voxel;

 public:

  /*! Constructor. It sets the default parameters for the filters (these parameters are fixed) */
  FilterPointCloud(const float voxelSize = 0.05, const float euclideanBox = 4.0)
  {
    // Set narrow limits for the euclidean filters to better visualize superimposed close-by frames
    filter_pass_x.setFilterFieldName ("x"); // Vertical axis of the spherical point clouds in RGBD360 (it points towards the ceiling)
    filter_pass_x.setFilterLimits (-2.0, 1.0);
    filter_pass_z.setFilterFieldName ("z");
    filter_pass_z.setFilterLimits (-euclideanBox, euclideanBox);
    filter_pass_y.setFilterFieldName ("y");
    filter_pass_y.setFilterLimits (-euclideanBox, euclideanBox);

    filter_voxel.setLeafSize(voxelSize,voxelSize,voxelSize);
  }

  /*! This function filters the input 'cloud' by setting a maximum and minimum in the x, y and z coordinates
   *  (these thresholds are defined by this class' constructor) */
  void filterEuclidean(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud)
  {
    // Initialize the global map with the first observation
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filteredCloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);
    filter_pass_y.setInputCloud (cloud);
    filter_pass_y.filter (*filteredCloud);
    filter_pass_z.setInputCloud (filteredCloud);
    filter_pass_z.filter (*filteredCloud2);
    filter_pass_x.setInputCloud (filteredCloud2);
    filter_pass_x.filter (*cloud);
  }

  /*! This function filters the input 'cloud' leaving one pixel per voxel (the voxel is defined by this class' constructor) */
  void filterVoxel(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud)
  {
    // Initialize the global map with the first observation
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    filter_voxel.setInputCloud (cloud);
    filter_voxel.filter (*filteredCloud);
    cloud = filteredCloud;
  }

};

#endif
