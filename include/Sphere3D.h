/*
 *  Copyright (c) 2015 INRIA Sophia Antipolis - LAGADIC Team
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

#include "SphereRGBD.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGBA PointT;

/*! This class defines the omnidirectional RGB-D frame 'Frame360'. It contains a serie of attributes and methods to
 *  produce the omnidirectional images, and to obtain the spherical point cloud and a the planar representation for it
 */
class Sphere3D : public SphereRGBD
{
  protected:

    /*! Frame ID*/
    size_t id;

    /*! Topological node where this frame (keyframe) is located */
    size_t node;

//    /*! The angular resolution of a pixel. Normally it is the same along horizontal/vertical (theta/phi) axis. */
//    float pixel_angle_;

//    /*! The index referring the latitude in pixels of the first row in the image (the closest to the upper part of the sphere) */
//    int phi_start_pixel_;

    /*! Pose of this frame */
    Eigen::Matrix4f pose;

    /*! 3D Point cloud of the spherical frame */
    pcl::PointCloud<PointT>::Ptr sphereCloud;

    //  /*! Has the spherical point cloud already been built? */
    //  bool bSphereCloudBuilt;

public:

    /*! Constructor for the SphericalStereo sensor (outdoor sensor) */
    Sphere3D();

    inline pcl::PointCloud<PointT>::Ptr & getSphereCloud()
    {
        return sphereCloud;
    }

    /*! Load a spherical point cloud */
    void loadCloud(const std::string &pointCloudPath);

    /*! Save the PbMap from an omnidirectional RGB-D image */
    void savePlanes(std::string pathPbMap);

    /*! Save the pointCloud and PbMap from an omnidirectional RGB-D image */
    void save(std::string &path, unsigned &frame);

    /*! Downsample and filter the individual point clouds from the different Asus XPL */
    void buildCloudsDownsampleAndFilter();

    /*! Fast version of the method 'buildPointCloud'. This one performs more poorly for plane segmentation. */
    void buildPointCloud_fast();

    /*! Build the spherical point cloud. The reference system is the one used by the INRIA SphericalStereo sensor. Z points forward, X points to the right and Y points downwards */
    void buildPointCloud();
    //void buildPointCloud2();

    /*! Compute the normalMap from an organized cloud of normal vectors. */
    void computeNormalMap(const pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud, cv::Mat & normalMap, const bool display = false);

};
