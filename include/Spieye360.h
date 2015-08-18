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

#define USE_BILATERAL_FILTER 1
#define DOWNSAMPLE_160 1

#include <mrpt/pbmap.h>
#include <mrpt/pbmap/Miscellaneous.h>

#include "Calib360.h"
#include "Sphere3D.h"

#include <CloudRGBD_Ext.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cvmat_serialization.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv/cv.h>
//#include <opencv2/opencv.hpp> // which one should I use?

//typedef pcl::PointXYZRGBA PointT;

/*! This class defines the omnidirectional RGB-D frame 'Spieye360'. It contains a serie of attributes and methods to
 *  produce the omnidirectional images, and to obtain the spherical point cloud and a the planar representation for it
 */
class Spieye360 : public virtual Sphere3D
{
  public:

    //Sphere3D *sphere3D;

    /*! The NUM_ASUS_SENSORS sets of planes segmented from each camera */
    std::vector<mrpt::pbmap::PbMap> local_planes_;

    /*! The NUM_ASUS_SENSORS RGB-D images captured by the omnidirectional device */
    //  FrameRGBD frameRGBD_[NUM_ASUS_SENSORS];
    CloudRGBD_Ext frameRGBD_[NUM_ASUS_SENSORS];

    /*! Calibration object */
    Calib360 *calib;

  private:

    /*! The NUM_ASUS_SENSORS separate point clouds from each single Asus XPL */
    pcl::PointCloud<PointT>::Ptr cloud_[NUM_ASUS_SENSORS];

    //  /*! Has the spherical point cloud already been built? */
    //  bool bSphereCloudBuilt;

    /*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in its local frame of reference
    */
    void segmentPlanesLocalCam(int sensor_id);

    /*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in the frame of reference of the omnidirectional camera
    */
    void segmentPlanesSensor(int sensor_id);

    /*! Undistort the depth image corresponding to the sensor 'sensor_id' */
    void undistortDepthSensor(int sensor_id);

    /*! Stitch both the RGB and the depth images corresponding to the sensor 'sensor_id' */
    void stitchImage(int sensor_id);
    // Functions for SphereStereo images (outdoors)

public:
    /*! Constructor for the SphericalStereo sensor (outdoor sensor) */
    Spieye360();

    /*! Constructor for the sensor RGBD360 (NUM_ASUS_SENSORS Asus XPL)*/
    Spieye360(Calib360 *calib360);

    /*! Return the the point cloudgrabbed by the sensor 'id' */
    inline pcl::PointCloud<PointT>::Ptr getCloud_id(int id)
    {
        assert(id >= 0 && id < NUM_ASUS_SENSORS);
        return cloud_[id];
    }

    /*! Return the the point FrameRGBD by the sensor 'id' */
    inline CloudRGBD_Ext getFrameRGBD_id(int id)
    {
        assert(id >= 0 && id < NUM_ASUS_SENSORS);
        return frameRGBD_[id];
    }

    /*! Load a spherical RGB-D image from the raw data stored in a binary file */
    void loadFrame(std::string &binaryFile);

    /*! Undistort the omnidirectional depth image using the models acquired with CLAMS */
    void undistort();

    /*! Serialize the omnidirectional RGB-D image */
    void serialize(std::string &fileName);

    /*! Concatenate the different RGB-D images to obtain the omnidirectional one (stich images without using the spherical representation) */
    void fastStitchImage360(); // Parallelized with OpenMP

    /*! Stitch RGB-D image using a spherical representation */
    void stitchSphericalImage(const int depth_type = CV_16U); // Parallelized with OpenMP

    /*! Downsample and filter the individual point clouds from the different Asus XPL */
    void buildCloudsDownsampleAndFilter();

    /*! Build the cloud from the 'sensor_id' Asus XPL */
    void buildCloud_id(int sensor_id);

    /*! Build the spherical point cloud by superimposing the NUM_ASUS_SENSORS point clouds from the NUM_ASUS_SENSORS Asus XPL*/
    void buildPointCloud_rgbd360();

    /*! Fast version of the method 'buildPointCloud'. This one performs more poorly for plane segmentation. */
    void buildPointCloud_fast();

    /*! Build the spherical point cloud. The reference system is the one used by the INRIA SphericalStereo sensor. Z points forward, X points to the right and Y points downwards */
    void buildPointCloud();
    //void buildPointCloud2();

    /*! Create the PbMap of the spherical point cloud */
    void segmentPlanes();

    /*! Segment planes in each of the separate point clouds correspondint to each Asus XPL */
    void segmentPlanesLocal();

    /*! Merge the planar patches that correspond to the same surface in the sphere */
    void mergePlanes();

    /*! Group the planes segmented from each single sensor into the common PbMap 'planes' */
    void groupPlanes();

    /*! Perform bilateral filtering on the point cloud   */
    void filterCloudBilateral_stereo();

    /*! This function segments planes from the point cloud    */
    void segmentPlanesStereo();

    /*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in the frame of reference of the omnidirectional camera   */
    void segmentPlanesStereoRANSAC();

    /*! Compute the normalMap from an organized cloud of normal vectors. */
    void computeNormalMap(const pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud, cv::Mat & normalMap, const bool display = false);

};
