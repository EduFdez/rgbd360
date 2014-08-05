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
 * Author: Eduardo Fernandez-Moral
 */

#ifndef FRAME360_H
#define FRAME360_H

#ifndef _DEBUG_MSG
#define _DEBUG_MSG 0
#endif

#define USE_BILATERAL_FILTER 1
#define DOWNSAMPLE_160 1

#include <mrpt/pbmap.h>

#include "Calib360.h"
#include "Miscellaneous.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h> //Save global map as PCD file
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/planar_region.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <boost/thread/thread.hpp>

#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cvmat_serialization.h>

#include <CloudRGBD_Ext.h>
#include <DownsampleRGBD.h>

#include <Eigen/Core>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

//#include <omp.h>


typedef pcl::PointXYZRGBA PointT;

/*! This class defines the omnidirectional RGB-D frame 'Frame360'. It contains a serie of attributes and methods to
 *  produce the omnidirectional images, and to obtain the spherical point cloud and a the planar representation for it
 */
class Frame360
{
public:

    /*! Frame ID*/
    unsigned id;

    /*! Topological node where this frame (keyframe) is located */
    unsigned node;

    /*! Spherical (omnidirectional) RGB image. The term spherical means that the same solid angle is assigned to each pixel */
    cv::Mat sphereRGB;

    /*! Spherical (omnidirectional) Depth image*/
    cv::Mat sphereDepth;

    /*! The 8 sets of planes segmented from each camera */
    std::vector<mrpt::pbmap::PbMap> local_planes_;

    /*! The 8 RGB-D images captured by the omnidirectional device */
    //  FrameRGBD frameRGBD_[8];
    CloudRGBD_Ext frameRGBD_[8];

    /*! Pose of this frame */
    Eigen::Matrix4d pose;

    /*! 3D Point cloud of the spherical frame */
    pcl::PointCloud<PointT>::Ptr sphereCloud;

    /*! PbMap of the spherical frame */
    mrpt::pbmap::PbMap planes;

    /*! Calibration object */
    Calib360 *calib;

private:

    /*! Time-stamp of the spherical frame (it corresponds to the last capture of the 8 sensors, as they are not syncronized) */
    uint64_t timeStamp;

    /*! The 8 separate point clouds from each single Asus XPL */
    pcl::PointCloud<PointT>::Ptr cloud_[8];

    //  /*! Has the spherical point cloud already been built? */
    //  bool bSphereCloudBuilt;

public:
    /*! Constructor */
    Frame360(Calib360 *calib360) :
        calib(calib360),
        sphereCloud(new pcl::PointCloud<PointT>()),
        //    bSphereCloudBuilt(false),
        node(0)
    {
        local_planes_.resize(8);
        for(unsigned sensor_id=0; sensor_id<8; sensor_id++)
        {
            cloud_[sensor_id].reset(new pcl::PointCloud<PointT>());
        }

        double time_start = pcl::getTime();
    }

    /*! Return the total area of the planar patches from this frame */
    float getPlanarArea()
    {
        float planarArea = 0;
        for(unsigned i = 0; i < planes.vPlanes.size(); i++)
            planarArea += planes.vPlanes[i].areaHull;

        return planarArea;
    }

    /*! Return the the point cloudgrabbed by the sensor 'id' */
    pcl::PointCloud<PointT>::Ptr getCloud_id(int id)
    {
        assert(id >= 0 && id < 8);
        return cloud_[id];
    }

    /*! Return the the point FrameRGBD by the sensor 'id' */
    CloudRGBD_Ext getFrameRGBD_id(int id)
    {
        assert(id >= 0 && id < 8);
        return frameRGBD_[id];
    }

    /*! Set the spherical image timestamp */
    void setTimeStamp(uint64_t timestamp)
    {
        timeStamp = timestamp;
    }

    /*! Load a spherical point cloud */
    void loadCloud(std::string &pointCloudPath)
    {
        // Load pointCloud
        sphereCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PCDReader reader;
        reader.read (pointCloudPath, *sphereCloud);
    }

    /*! Load a spherical PbMap */
    void loadPbMap(std::string &pbmapPath)
    {
        // Load planes
        mrpt::utils::CFileGZInputStream serialize_planes;
        if (serialize_planes.open(pbmapPath))
        {
            serialize_planes >> planes;
#if _DEBUG_MSG
            cout << planes.vPlanes.size() << " planes loaded\n";
#endif
        }
        else
            cout << "Error: cannot open " << pbmapPath << "\n";
        serialize_planes.close();
    }

    /*! Load a spherical frame from its point cloud and its PbMap files */
    void load_PbMap_Cloud(std::string &pointCloudPath, std::string &pbmapPath)
    {
        // Load pointCloud
        loadCloud(pointCloudPath);

        // Load planes
        loadPbMap(pbmapPath);
    }

    /*! Load a spherical frame from its point cloud and its PbMap files */
    void load_PbMap_Cloud(std::string &path, unsigned &index)
    {
        std::string pointCloudPath = path + mrpt::format("/sphereCloud_%u.pcd", index);
        std::string pbmapPath = path + mrpt::format("/spherePlanes_%u.pbmap", index);
        load_PbMap_Cloud(pointCloudPath, pbmapPath);
    }

    /*! Load a spherical RGB-D image from the raw data stored in a binary file */
    void loadFrame(std::string &binaryFile)
    {
        double time_start = pcl::getTime();

        //    std::cout << "Opening binary file... " << binaryFile << std::endl;
        std::ifstream ifs(binaryFile.c_str(), std::ios::in | std::ios::binary);
        //    std::cout << " binary file... " << binaryFile << std::endl;

        try{// use scope to ensure archive goes out of scope before stream

            boost::archive::binary_iarchive binary360(ifs);

            cv::Mat timeStampMatrix;
            for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
                binary360 >> frameRGBD_[sensor_id].getRGBImage() >> frameRGBD_[sensor_id].getDepthImage();
            binary360 >> timeStampMatrix;
            get_uint64_t_ofMatrixRepresentation(timeStampMatrix, timeStamp);

        }catch(const boost::archive::archive_exception &e){}

        //Close the binary bile
        ifs.close();

#pragma omp parallel num_threads(8)
        {
            int sensor_id = omp_get_thread_num();
            frameRGBD_[sensor_id].loadDepthEigen();
        }

        //    bSphereCloudBuilt = false; // The spherical PointCloud of the frame just loaded is not built yet

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "Load Frame360 took " << double (time_end - time_start) << std::endl;
#endif
    }

    /*Get the average intensity*/
    inline int getAverageIntensity(int sample = 1)
    {

        //    getIntensityImage();
        //
        //    #pragma omp parallel num_threads(8)
        //    {
        //      int sensor_id = omp_get_thread_num();
        //      int sum_intensity[8];
        //      std::fill(sum_intensity, sum_intensity+8, 0);
        //      for(unsigned i=0; i < 8; i++)
        //        if(sensor_id == i)
        //        {
        //          frameRGBD_[sensor_id].getIntensityImage(); //Make sure that the intensity image has been computed
        //          sum_intensity[sensor_id] = frameRGBD_[sensor_id].getAverageIntensity(sample); //Make sure that the intensity image has been computed
        //        }
        //    }
        //    for(unsigned i=1; i < 8; i++)
        //      sum_intensity[0] += sum_intensity[i];
        //
        //    return floor(sum_intensity[0] / 8.0 + 0.5);
    }

    /*! Undistort the omnidirectional depth image using the models acquired with CLAMS */
    void undistort()
    {
        double time_start = pcl::getTime();
        //    std::cout << "Undistort Frame360\n";

#pragma omp parallel num_threads(8)
        {
            int sensor_id = omp_get_thread_num();
            undistortDepthSensor(sensor_id);
            //      cv::eigen2cv(frameRGBD_[sensor_id].m_depthEigUndistort, frameRGBD_[sensor_id].getDepthImage());
        }

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "Undistort Frame360 took " << double (time_end - time_start) << std::endl;
#endif

    }

    /*! Save the PbMap from an omnidirectional RGB-D image */
    void savePlanes(std::string pathPbMap)
    {
        mrpt::utils::CFileGZOutputStream serialize_planesLabeled(pathPbMap);
        serialize_planesLabeled << planes;
        serialize_planesLabeled.close();
    }

    /*! Save the pointCloud and PbMap from an omnidirectional RGB-D image */
    void save(std::string &path, unsigned &frame)
    {
        assert(!sphereCloud->empty() && planes.vPlanes.size() > 0);

        std::string cloudPath = path + mrpt::format("/sphereCloud_%d.pcd",frame);
        pcl::io::savePCDFile(cloudPath, *sphereCloud);

        std::string pbmapPath = path + mrpt::format("/spherePlanes_%d.pbmap",frame);
        savePlanes(pbmapPath);
    }

    /*! Serialize the omnidirectional RGB-D image */
    void serialize(std::string &fileName)
    {
        std::ofstream ofs_images(fileName.c_str(), std::ios::out | std::ios::binary);
        boost::archive::binary_oarchive oa_images(ofs_images);

        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            oa_images << frameRGBD_[sensor_id].getRGBImage() << frameRGBD_[sensor_id].getDepthImage();
        cv::Mat timeStampMatrix;
        getMatrixNumberRepresentationOf_uint64_t(timeStamp,timeStampMatrix);
        oa_images << timeStampMatrix;

        ofs_images.close();
    }

    /*! Concatenate the different RGB-D images to obtain the omnidirectional one (stich images without using the spherical representation) */
    void fastStitchImage360() // Parallelized with OpenMP
    {
        double time_start = pcl::getTime();

        int width_im = frameRGBD_[0].getRGBImage().rows;
        int width_SphereImg = frameRGBD_[0].getRGBImage().rows * 8;
        int height_SphereImg = frameRGBD_[0].getRGBImage().cols;
        sphereRGB = cv::Mat::zeros(height_SphereImg, width_SphereImg, CV_8UC3);
        sphereDepth = cv::Mat::zeros(height_SphereImg, width_SphereImg, CV_32FC1);

        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
            //    #pragma omp parallel num_threads(8)
        {
            //      int sensor_id = omp_get_thread_num();

            // RGB
            cv::Mat rgb_transposed, rgb_rotated;
            cv::transpose(frameRGBD_[7-sensor_id].getRGBImage(), rgb_transposed);
            cv::flip(rgb_transposed, rgb_rotated, 0);
            cv::Mat tmp = sphereRGB(cv::Rect(sensor_id*width_im, 0, frameRGBD_[0].getRGBImage().rows, frameRGBD_[0].getRGBImage().cols));
            rgb_rotated.copyTo(tmp);

            // Depth
            //      cv::Mat depth_transposed, depth_rotated;
            //      cvTranspose(frameRGBD_[7-sensor_id].getDepthImage(), depth_transposed);
            //      cv::flip(depth_transposed, depth_rotated, 0);
            //      cv::Mat tmp_depth =  sphereDepth(cv::Rect(sensor_id*width_im, 0, frameRGBD_[0].getDepthImage().rows, frameRGBD_[0].getDepthImage().cols));
            //      depth_rotated.copyTo(tmp_depth);
        }

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "fastStitchImage360 took " << double (time_end - time_start) << std::endl;
#endif

    }

    /*! Stitch RGB-D image using a spherical representation */
    void stitchSphericalImage() // Parallelized with OpenMP
    {
        double time_start = pcl::getTime();

        int width_SphereImg = frameRGBD_[0].getRGBImage().rows * 8;
        int height_SphereImg = width_SphereImg * 0.5 * 63.0/180; // Store only the part of the sphere which contains information
        sphereRGB = cv::Mat::zeros(height_SphereImg, width_SphereImg, CV_8UC3);
        sphereDepth = cv::Mat::zeros(height_SphereImg, width_SphereImg, CV_32FC1);
        cout << "stitchSphericalImage\n";
#pragma omp parallel num_threads(8)
        {
            int sensor_id = omp_get_thread_num();
            stitchImage(sensor_id);
            //      cout << "Stitcher omp " << sensor_id << endl;
        }

        double time_end = pcl::getTime();
        std::cout << "stitchSphericalImage took " << double (time_end - time_start) << std::endl;
    }

    /*! Downsample and filter the individual point clouds from the different Asus XPL */
    void buildCloudsDownsampleAndFilter()
    {
        double time_start = pcl::getTime();

#pragma omp parallel num_threads(8)
        {
            int sensor_id = omp_get_thread_num();

            // Filter pointClouds
            pcl::FastBilateralFilter<pcl::PointXYZRGBA> filter;
            filter.setSigmaS(10.0);
            filter.setSigmaR(0.05);

#if DOWNSAMPLE_160
            //        DownsampleRGBD downsampler(2);
            //        frameRGBD_[sensor_id].setPointCloud(downsampler.downsamplePointCloud(frameRGBD_[sensor_id].getPointCloudUndist()));
            frameRGBD_[sensor_id].getDownsampledPointCloudUndist(2);
#endif

#if USE_BILATERAL_FILTER
            filter.setInputCloud(frameRGBD_[sensor_id].getPointCloud());
            filter.filter(*frameRGBD_[sensor_id].getPointCloud());
#endif
        }

        double time_end = pcl::getTime();
        std::cout << "Build single clouds + downsample + filter took " << double (time_end - time_start) << std::endl;
    }

    /*! Build the cloud from the 'sensor_id' Asus XPL */
    void buildCloud_id(int sensor_id)
    {
        pcl::FastBilateralFilter<pcl::PointXYZRGBA> filter;
        filter.setSigmaS (10.0);
        filter.setSigmaR (0.05);
        //  std::cout << "buildCloud_id1 " << sensor_id << "\n";

        // #pragma openmp
#if DOWNSAMPLE_160
        //      DownsampleRGBD downsampler(2);
        ////      std::cout << "buildCloud_id1\n";
        //      frameRGBD_[sensor_id].setPointCloud(downsampler.downsamplePointCloud(frameRGBD_[sensor_id].getPointCloudUndist()));
        ////    std::cout << "buildCloud_id2\n";
        frameRGBD_[sensor_id].getDownsampledPointCloudUndist(2);
#endif

#if USE_BILATERAL_FILTER
        filter.setInputCloud(frameRGBD_[sensor_id].getDownsampledPointCloudUndist(2));
        filter.filter(*frameRGBD_[sensor_id].getDownsampledPointCloudUndist(2));
#endif

        pcl::transformPointCloud(*frameRGBD_[sensor_id].getPointCloud(),*cloud_[sensor_id],calib->getRt_id(sensor_id));

        //    buildCloud_im_[sensor_id] = true;

        //  std::cout << "buildCloud_id3 " << sensor_id << "\n";
    }

    /*! Build the spherical point cloud */
    void buildSphereCloud()
    {
        //    if(bSphereCloudBuilt) // Avoid building twice the spherical point cloud
        //      return;

        double time_start = pcl::getTime();

        //      #pragma omp parallel num_threads(8) // I still don't understand why this doesn't work
        for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
        {
            //        int sensor_id = omp_get_thread_num();
//            cout << "build " << sensor_id << endl;
#if DOWNSAMPLE_160
            DownsampleRGBD downsampler(2);
            frameRGBD_[sensor_id].setPointCloud(downsampler.downsamplePointCloud(frameRGBD_[sensor_id].getPointCloudUndist()));
            //          frameRGBD_[sensor_id].getDownsampledPointCloudUndist(2);
#else
            frameRGBD_[sensor_id].getPointCloudUndist();
#endif
        }
        //      std::cout << "Downsample image\n";

#pragma omp parallel num_threads(8)
        {
            int sensor_id = omp_get_thread_num();

#if USE_BILATERAL_FILTER
            pcl::FastBilateralFilter<pcl::PointXYZRGBA> filter;
            filter.setSigmaS (10.0);
            filter.setSigmaR (0.05);
            filter.setInputCloud(frameRGBD_[sensor_id].getPointCloud());
            filter.filter(*frameRGBD_[sensor_id].getPointCloud());
#endif

            pcl::transformPointCloud(*frameRGBD_[sensor_id].getPointCloud(),*cloud_[sensor_id],calib->getRt_id(sensor_id));
        }

        *sphereCloud = *cloud_[0];
        for(unsigned sensor_id=1; sensor_id < 8; ++sensor_id)
            *sphereCloud += *cloud_[sensor_id];

        sphereCloud->height = cloud_[0]->width;
        sphereCloud->width = 8 * cloud_[0]->height;
        sphereCloud->is_dense = false;

        //    bSphereCloudBuilt = true;

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "PointCloud sphere construction took " << double (time_end - time_start) << std::endl;
#endif

    }

    /*! Fast version of the method 'buildSphereCloud'. This one performs more poorly for plane segmentation. */
    void buildSphereCloud_fast()
    {
        double time_start = pcl::getTime();

#pragma omp parallel num_threads(8) // I still don't understand why this doesn't work
        //      for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
        {
            int sensor_id = omp_get_thread_num();
#if DOWNSAMPLE_160
            frameRGBD_[sensor_id].getDownsampledPointCloudUndist(2);
#else
            frameRGBD_[sensor_id].getPointCloudUndist();
#endif

            pcl::transformPointCloud(*frameRGBD_[sensor_id].getPointCloud(),*cloud_[sensor_id],calib->getRt_id(sensor_id));
        }

        *sphereCloud = *cloud_[0];
        for(unsigned sensor_id=1; sensor_id < 8; ++sensor_id)
            *sphereCloud += *cloud_[sensor_id];

        sphereCloud->height = cloud_[0]->width;
        sphereCloud->width = 8 * cloud_[0]->height;
        sphereCloud->is_dense = false;

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "buildSphereCloud_fast took " << double (time_end - time_start) << std::endl;
#endif

    }

    /*! Build the point cloud from the spherical image representation */
    void buildSphereCloud_fromImage()
    {
        double time_start = pcl::getTime();

        size_t height_SphereImg = sphereRGB.rows;
        size_t width_SphereImg = sphereRGB.cols;
        float angle_pixel(width_SphereImg/(2*PI));

        sphereCloud->resize(height_SphereImg*width_SphereImg);
        sphereCloud->height = height_SphereImg;
        sphereCloud->width = width_SphereImg;
        sphereCloud->is_dense = false;

        float angle_px(1/angle_pixel);
        int row_pixels = 0;
        float offset_phi = PI*31.5/180; // height_SphereImg((width_SphereImg/2) * 63.0/180),
        float PI_ = PI; // Does this spped-up the next loop?
        for(int row_phi=0; row_phi < height_SphereImg; row_phi++, row_pixels += width_SphereImg)
        {
            float phi_i = row_phi*angle_px - offset_phi;// + PI/2;

            float sin_phi = sin(phi_i);
            float cos_phi = cos(phi_i);

            // Stitch each image from each sensor
            //      #pragma omp parallel for
            for(int col_theta=0; col_theta < width_SphereImg; col_theta++)
            {
                float theta_i = col_theta*angle_px - PI_; // + PI;
                float depth = sphereDepth.at<float>(row_phi,col_theta);
                //      if(row_phi >50)
                //        cout << "pos " << row_phi << " " << col_theta << " Phi/Theta " << phi_i << " " << theta_i << " depth " << depth << endl;
                if(depth != 0)
                {
                    //            cout << " Assign point in the cloud\n";
                    sphereCloud->points[row_pixels+col_theta].x = sin(theta_i) * cos_phi * depth;
                    //            sphereCloud->points[row_pixels+col_theta].x = cos(theta_i) * cos_phi * depth;
                    sphereCloud->points[row_pixels+col_theta].y = sin_phi * depth;
                    sphereCloud->points[row_pixels+col_theta].z = cos(theta_i) * cos_phi * depth;
                    //            sphereCloud->points[row_pixels+col_theta].z = sin(theta_i) * cos_phi * depth;
                    sphereCloud->points[row_pixels+col_theta].r = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[2];
                    sphereCloud->points[row_pixels+col_theta].g = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[1];
                    sphereCloud->points[row_pixels+col_theta].b = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[0];
                }
                else
                {
                    sphereCloud->points[row_pixels+col_theta].x = std::numeric_limits<float>::quiet_NaN ();
                    sphereCloud->points[row_pixels+col_theta].y = std::numeric_limits<float>::quiet_NaN ();
                    sphereCloud->points[row_pixels+col_theta].z = std::numeric_limits<float>::quiet_NaN ();
                }
            }
        }

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "PointCloud sphere from image took " << double (time_end - time_start) << std::endl;
#endif

    }

    /*! Create the PbMap of the spherical point cloud */
    void getPlanes()
    {
        std::cout << "Frame360.getPlanes()\n";
        double extractPlanes_start = pcl::getTime();

#pragma omp parallel num_threads(8)
        {
            int sensor_id = omp_get_thread_num();
            getPlanesSensor(sensor_id);
        }

        double segmentation_end = pcl::getTime();
        std::cout << "Segmentation took " << double (segmentation_end - extractPlanes_start)*1000 << " ms.\n";

        // Merge the big planes
        groupPlanes(); // Merge planes detected from adjacent sensors, and place them in "planes"
        mergePlanes(); // Merge big planes

#if _DEBUG_MSG
        double extractPlanes_end = pcl::getTime();
        std::cout << planes.vPlanes.size() << " planes. Extraction took " << double (extractPlanes_end - extractPlanes_start)*1000 << " ms.\n";
#endif

    }

    /*! Segment planes in each of the separate point clouds correspondint to each Asus XPL */
    void getLocalPlanes()
    {
        double time_start = pcl::getTime();

        //    #pragma omp parallel num_threads(8)
        for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
        {
            //      int sensor_id = omp_get_thread_num();
            getLocalPlanesInFrame(sensor_id);
        }

        //    double time_end = pcl::getTime();
        //    std::cout << "Local-Plane extraction took " << double (time_end - time_start) << std::endl;
    }

    /*! Merge the planar patches that correspond to the same surface in the sphere */
    void mergePlanes()
    {
        double time_start = pcl::getTime();

        // Merge repeated planes
        for(size_t j = 0; j < planes.vPlanes.size(); j++) // numPrevPlanes
            if(planes.vPlanes[j].curvature < max_curvature_plane)
                for(size_t k = j+1; k < planes.vPlanes.size(); k++) // numPrevPlanes
                    if(planes.vPlanes[k].curvature < max_curvature_plane)
                    {
                        bool bSamePlane = false;
                        //        Eigen::Vector3f center_diff = planes.vPlanes[k].v3center - planes.vPlanes[j].v3center;
                        Eigen::Vector3f close_points_diff;
                        float dist, prev_dist = 1;
                        if( planes.vPlanes[j].v3normal.dot(planes.vPlanes[k].v3normal) > 0.99 )
                            if( fabs(planes.vPlanes[j].d - planes.vPlanes[k].d) < 0.45 )
                                //          if( BhattacharyyaDist_(plane1.hist_H, plane2.hist_H) > configLocaliser.hue_threshold )
                                //          if( fabs(planes.vPlanes[j].v3normal.dot(center_diff)) < std::max(0.07, 0.03*center_diff.norm() ) )
                            {
                                // Checking distances:
                                // a) Between an vertex and a vertex
                                // b) Between an edge and a vertex
                                // c) Between two edges (imagine two polygons on perpendicular planes)
                                for(unsigned i=1; i < planes.vPlanes[j].polygonContourPtr->size() && !bSamePlane; i++)
                                    for(unsigned ii=1; ii < planes.vPlanes[k].polygonContourPtr->size(); ii++)
                                    {
                                        close_points_diff = mrpt::pbmap::diffPoints(planes.vPlanes[j].polygonContourPtr->points[i], planes.vPlanes[k].polygonContourPtr->points[ii]);
                                        dist = close_points_diff.norm();
                                        //                if( dist < prev_dist )
                                        //                  prev_dist = dist;
                                        if( dist < 0.3 && fabs(planes.vPlanes[j].v3normal.dot(close_points_diff)) < 0.06)
                                        {
                                            bSamePlane = true;
                                            break;
                                        }
                                    }
                                // a) & b)
                                if(!bSamePlane)
                                    for(unsigned i=1; i < planes.vPlanes[j].polygonContourPtr->size() && !bSamePlane; i++)
                                        for(unsigned ii=1; ii < planes.vPlanes[k].polygonContourPtr->size(); ii++)
                                        {
                                            dist = sqrt(mrpt::pbmap::dist3D_Segment_to_Segment2(mrpt::pbmap::Segment(planes.vPlanes[j].polygonContourPtr->points[i],planes.vPlanes[j].polygonContourPtr->points[i-1]), mrpt::pbmap::Segment(planes.vPlanes[k].polygonContourPtr->points[ii],planes.vPlanes[k].polygonContourPtr->points[ii-1])));
                                            //                if( dist < prev_dist )
                                            //                  prev_dist = dist;
                                            if( dist < 0.3)
                                            {
                                                close_points_diff = mrpt::pbmap::diffPoints(planes.vPlanes[j].polygonContourPtr->points[i], planes.vPlanes[k].polygonContourPtr->points[ii]);
                                                if(fabs(planes.vPlanes[j].v3normal.dot(close_points_diff)) < 0.06)
                                                {
                                                    bSamePlane = true;
                                                    break;
                                                }
                                            }
                                        }
                            }

                        if( bSamePlane ) // The planes are merged if they are the same
                        {
                            // Update normal and center
                            assert(planes.vPlanes[j].inliers.size() > 0 &&  planes.vPlanes[k].inliers.size() > 0);
                            planes.vPlanes[j].mergePlane2(planes.vPlanes[k]);

                            // Update plane index
                            for(size_t h = k+1; h < planes.vPlanes.size(); h++)
                                --planes.vPlanes[h].id;

                            // Delete plane to merge
                            std::vector<mrpt::pbmap::Plane>::iterator itPlane = planes.vPlanes.begin();
                            for(size_t i = 0; i < k; i++)
                                itPlane++;
                            planes.vPlanes.erase(itPlane);

                            // Re-evaluate possible planes to merge
                            j--;
                            k = planes.vPlanes.size();
                        }
                    }
#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "Merge planes took " << double (time_end - time_start) << std::endl;
#endif

    }

    /*! Group the planes segmented from each single sensor into the common PbMap 'planes' */
    void groupPlanes()
    {
        //  cout << "groupPlanes...\n";
        double time_start = pcl::getTime();

        float maxDistHull = 0.5;
        float maxDistParallelHull = 0.09;

        //    Eigen::Matrix4d Rt = calib->getRt_id(0);
        //    planes.MergeWith(local_planes_[0], Rt);
        planes = local_planes_[0];
        std::set<unsigned> prev_planes, first_planes;
        for(size_t i=0; i < planes.vPlanes.size(); i++)
            first_planes.insert(planes.vPlanes[i].id);
        prev_planes = first_planes;

        for(unsigned sensor_id=1; sensor_id < 8; ++sensor_id)
        {
            size_t j;
            std::set<unsigned> next_prev_planes;
            for(size_t k = 0; k < local_planes_[sensor_id].vPlanes.size(); k++)
            {
                bool bSamePlane = false;
                if(local_planes_[sensor_id].vPlanes[k].areaHull > 0.5 || local_planes_[sensor_id].vPlanes[k].curvature < max_curvature_plane)
                    for(std::set<unsigned>::iterator it = prev_planes.begin(); it != prev_planes.end() && !bSamePlane; it++) // numPrevPlanes
                    {
                        j = *it;

                        if(planes.vPlanes[j].areaHull < 0.5 || planes.vPlanes[j].curvature > max_curvature_plane)
                            continue;

                        Eigen::Vector3f close_points_diff;
                        float dist, prev_dist = 1;
                        if( fabs(planes.vPlanes[j].d - local_planes_[sensor_id].vPlanes[k].d) < 0.45 )
                            if( planes.vPlanes[j].v3normal.dot(local_planes_[sensor_id].vPlanes[k].v3normal) > 0.99 )
                            {
                                // Checking distances:
                                // a) Between an vertex and a vertex
                                // b) Between an edge and a vertex
                                // c) Between two edges (imagine two polygons on perpendicular planes)
                                //            if(!planes.vPlanes[j].isPlaneNearby(local_planes_[sensor_id].vPlanes[k],0.2);
                                //              continue;

                                for(unsigned i=1; i < planes.vPlanes[j].polygonContourPtr->size() && !bSamePlane; i++)
                                    for(unsigned ii=1; ii < local_planes_[sensor_id].vPlanes[k].polygonContourPtr->size(); ii++)
                                    {
                                        close_points_diff = mrpt::pbmap::diffPoints(planes.vPlanes[j].polygonContourPtr->points[i], local_planes_[sensor_id].vPlanes[k].polygonContourPtr->points[ii]);
                                        dist = close_points_diff.norm();
                                        if( dist < maxDistHull && fabs(planes.vPlanes[j].v3normal.dot(close_points_diff)) < maxDistParallelHull)
                                        {
                                            bSamePlane = true;
                                            break;
                                        }
                                    }
                                // a) & b)
                                if(!bSamePlane)
                                    for(unsigned i=1; i < planes.vPlanes[j].polygonContourPtr->size() && !bSamePlane; i++)
                                        for(unsigned ii=1; ii < local_planes_[sensor_id].vPlanes[k].polygonContourPtr->size(); ii++)
                                        {
                                            dist = sqrt(mrpt::pbmap::dist3D_Segment_to_Segment2(mrpt::pbmap::Segment(planes.vPlanes[j].polygonContourPtr->points[i],planes.vPlanes[j].polygonContourPtr->points[i-1]), mrpt::pbmap::Segment(local_planes_[sensor_id].vPlanes[k].polygonContourPtr->points[ii],local_planes_[sensor_id].vPlanes[k].polygonContourPtr->points[ii-1])));
                                            if( dist < maxDistHull)
                                            {
                                                close_points_diff = mrpt::pbmap::diffPoints(planes.vPlanes[j].polygonContourPtr->points[i], local_planes_[sensor_id].vPlanes[k].polygonContourPtr->points[ii]);
                                                if(fabs(planes.vPlanes[j].v3normal.dot(close_points_diff)) < maxDistParallelHull)
                                                {
                                                    bSamePlane = true;
                                                    break;
                                                }
                                            }
                                        }
                            }
                        if(bSamePlane)
                            break;
                    }
                if( bSamePlane ) // The planes are merged if they are the same
                {
                    next_prev_planes.insert(planes.vPlanes[j].id);
                    planes.vPlanes[j].mergePlane2(local_planes_[sensor_id].vPlanes[k]);
                }
                else
                {
                    next_prev_planes.insert(planes.vPlanes.size());
                    local_planes_[sensor_id].vPlanes[k].id = planes.vPlanes.size();
                    planes.vPlanes.push_back(local_planes_[sensor_id].vPlanes[k]);
                }
            }
            prev_planes = next_prev_planes;
            if(sensor_id == 6)
                prev_planes.insert(first_planes.begin(), first_planes.end());
        }
    }

private:

    /*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in its local frame of reference
  */
    void getLocalPlanesInFrame(int sensor_id)
    {
        // Segment planes
        //    cout << "extractPlaneFeatures\n";
        //      double extractPlanes_start = pcl::getTime();

        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
        ne.setMaxDepthChangeFactor (0.02); // For VGA: 0.02f, 10.0f
        ne.setNormalSmoothingSize (10.0f);
        ne.setDepthDependentSmoothing (true);

        pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
        mps.setMinInliers (100);
        mps.setAngularThreshold (0.039812); // (0.017453 * 2.0) // 3 degrees
        mps.setDistanceThreshold (0.02); //2cm

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud ( frameRGBD_[sensor_id].getPointCloud() );
        ne.compute (*normal_cloud);

        mps.setInputNormals (normal_cloud);
        mps.setInputCloud ( frameRGBD_[sensor_id].getPointCloud() );
        std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
        std::vector<pcl::ModelCoefficients> model_coefficients;
        std::vector<pcl::PointIndices> inlier_indices;
        pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
        std::vector<pcl::PointIndices> label_indices;
        std::vector<pcl::PointIndices> boundary_indices;
        mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);
        //      mps.segment (model_coefficients, inlier_indices);
        //    cout << regions.size() << " planes detected\n";

        // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
        for (size_t i = 0; i < regions.size (); i++)
        {
            //      std::cout << "curv " << regions[i].getCurvature() << std::endl;
            if(regions[i].getCurvature() > max_curvature_plane)
                continue;

            mrpt::pbmap::Plane plane;

            plane.v3center = regions[i].getCentroid ();
            plane.v3normal = Eigen::Vector3f(model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);
            plane.d = model_coefficients[i].values[3];
            //        if( plane.v3normal.dot(plane.v3center) > 0)
            if( model_coefficients[i].values[3] < 0)
            {
                plane.v3normal = -plane.v3normal;
                plane.d = -plane.d;
            }
            plane.curvature = regions[i].getCurvature();
            //      cout << "normal " << plane.v3normal.transpose() << " center " << regions[i].getCentroid().transpose() << " " << plane.v3center.transpose() << endl;
            //    cout << "D " << -(plane.v3normal.dot(plane.v3center)) << " " << plane.d << endl;

            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
            extract.setInputCloud ( frameRGBD_[sensor_id].getPointCloud() );
            //        extract.setInputCloud ( cloud );
            extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices[i]) );
            extract.setNegative (false);
            extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
            plane.inliers = inlier_indices[i].indices;
            //    cout << "Extract inliers\n";

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr contourPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);
            contourPtr->points = regions[i].getContour();

            plane.calcConvexHull(contourPtr);
            plane.computeMassCenterAndArea();

            plane.calcElongationAndPpalDir();

            plane.calcPlaneHistH();

            // Check whether this region correspond to the same plane as a previous one (this situation may happen when there exists a small discontinuity in the observation)
            bool isSamePlane = false;
            for (size_t j = 0; j < local_planes_[sensor_id].vPlanes.size(); j++)
                if( local_planes_[sensor_id].vPlanes[j].isSamePlane(plane, 0.998, 0.1, 0.4) ) // The planes are merged if they are the same
                {
                    //          cout << "Merge local region\n";
                    isSamePlane = true;
                    local_planes_[sensor_id].vPlanes[j].mergePlane(plane);

                    break;
                }
            if(!isSamePlane)
            {
                //          plane.calcMainColor();
                plane.id = local_planes_[sensor_id].vPlanes.size();
                local_planes_[sensor_id].vPlanes.push_back(plane);
            }
        }
        //      double extractPlanes_end = pcl::getTime();
        //    std::cout << local_planes_[sensor_id].vPlanes.size() << " planes. Extraction in " << sensor_id << " took " << double (extractPlanes_end - extractPlanes_start) << std::endl;

        //      segmentation_im_[sensor_id] = true;
    }


    /*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in the frame of reference of the omnidirectional camera
  */
    void getPlanesSensor(int sensor_id)
    {
        // Segment planes
        //    std::cout << "extractPlaneFeatures, size " << frameRGBD_[sensor_id].getPointCloud()->size() << "\n";
        double extractPlanes_start = pcl::getTime();
        assert(frameRGBD_[sensor_id].getPointCloud()->height > 1 && frameRGBD_[sensor_id].getPointCloud()->width > 1);

        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        //      ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);
        //      ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);
        ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
        //      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
        ne.setMaxDepthChangeFactor (0.02); // For VGA: 0.02f, 10.01
        ne.setNormalSmoothingSize (8.0f);
        ne.setDepthDependentSmoothing (true);

        pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
        //      mps.setMinInliers (std::max(uint32_t(40),frameRGBD_[sensor_id].getPointCloud()->height*2));
        mps.setMinInliers (80);
        mps.setAngularThreshold (0.039812); // (0.017453 * 2.0) // 3 degrees
        mps.setDistanceThreshold (0.02); //2cm
        //    cout << "PointCloud size " << frameRGBD_[sensor_id].getPointCloud()->size() << endl;

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud ( frameRGBD_[sensor_id].getPointCloud() );
        ne.compute (*normal_cloud);

        mps.setInputNormals (normal_cloud);
        mps.setInputCloud ( frameRGBD_[sensor_id].getPointCloud() );
        std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
        std::vector<pcl::ModelCoefficients> model_coefficients;
        std::vector<pcl::PointIndices> inlier_indices;
        pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
        std::vector<pcl::PointIndices> label_indices;
        std::vector<pcl::PointIndices> boundary_indices;
        mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

        // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
        unsigned single_cloud_size = frameRGBD_[sensor_id].getPointCloud()->size();
        Eigen::Matrix4f Rt = calib->getRt_id(sensor_id).cast<float>();
        for (size_t i = 0; i < regions.size (); i++)
        {
            mrpt::pbmap::Plane plane;

            plane.v3center = regions[i].getCentroid ();
            plane.v3normal = Eigen::Vector3f(model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);
            if( plane.v3normal.dot(plane.v3center) > 0)
            {
                plane.v3normal = -plane.v3normal;
                //          plane.d = -plane.d;
            }
            plane.curvature = regions[i].getCurvature ();
            //    cout << i << " getCurvature\n";

            //        if(plane.curvature > max_curvature_plane)
            //          continue;

            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
            extract.setInputCloud ( frameRGBD_[sensor_id].getPointCloud() );
            extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices[i]) );
            extract.setNegative (false);
            extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
            plane.inliers.resize(inlier_indices[i].indices.size());
            for(size_t j=0; j<inlier_indices[i].indices.size(); j++)
                plane.inliers[j] = inlier_indices[i].indices[j] + sensor_id*single_cloud_size;

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr contourPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);
            contourPtr->points = regions[i].getContour();

            //    cout << "Extract contour\n";
            if(contourPtr->size() != 0)
            {
                plane.calcConvexHull(contourPtr);
            }
            else
            {
                //        assert(false);
                std::cout << "HULL 000\n" << plane.planePointCloudPtr->size() << std::endl;
                static pcl::VoxelGrid<pcl::PointXYZRGBA> plane_grid;
                plane_grid.setLeafSize(0.05,0.05,0.05);
                plane_grid.setInputCloud (plane.planePointCloudPtr);
                plane_grid.filter (*contourPtr);
                plane.calcConvexHull(contourPtr);
            }

            //        assert(contourPtr->size() > 0);
            //        plane.calcConvexHull(contourPtr);
            //    cout << "calcConvexHull\n";
            plane.computeMassCenterAndArea();
            //    cout << "Extract convexHull\n";
            // Discard small planes
            if(plane.areaHull < min_area_plane)
                continue;

            plane.d = -plane.v3normal .dot( plane.v3center );

            plane.calcElongationAndPpalDir();
            // Discard narrow planes
            if(plane.elongation > max_elongation_plane)
                continue;

            //      double color_start = pcl::getTime();
            plane.calcPlaneHistH();
            plane.calcMainColor2();
            //      double color_end = pcl::getTime();
            //    std::cout << "color in " << (color_end - color_start)*1000 << " ms\n";

            //      color_start = pcl::getTime();
            plane.transform(Rt);
            //      color_end = pcl::getTime();
            //    std::cout << "transform in " << (color_end - color_start)*1000 << " ms\n";

            bool isSamePlane = false;
            if(plane.curvature < max_curvature_plane)
                for (size_t j = 0; j < local_planes_[sensor_id].vPlanes.size(); j++)
                    if( local_planes_[sensor_id].vPlanes[j].curvature < max_curvature_plane && local_planes_[sensor_id].vPlanes[j].isSamePlane(plane, 0.99, 0.05, 0.2) ) // The planes are merged if they are the same
                    {
                        //          cout << "Merge local region\n";
                        isSamePlane = true;
                        //            double time_start = pcl::getTime();
                        local_planes_[sensor_id].vPlanes[j].mergePlane2(plane);
                        //            double time_end = pcl::getTime();
                        //          std::cout << " mergePlane2 took " << double (time_start - time_end) << std::endl;

                        break;
                    }
            if(!isSamePlane)
            {
                //          plane.calcMainColor();
                plane.id = local_planes_[sensor_id].vPlanes.size();
                local_planes_[sensor_id].vPlanes.push_back(plane);
            }
        }
#if _DEBUG_MSG
        double extractPlanes_end = pcl::getTime();
        std::cout << "getPlanesInFrame in " << (extractPlanes_end - extractPlanes_start)*1000 << " ms\n";
#endif

    }

    /*! Undistort the depth image corresponding to the sensor 'sensor_id' */
    void undistortDepthSensor(int sensor_id)
    {
        //    double time_start = pcl::getTime();

        //    frameRGBD_[sensor_id].loadDepthEigen();
        calib->intrinsic_model_[sensor_id].undistort(&frameRGBD_[sensor_id].m_depthEigUndistort);

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << " undistort " << sensor_id << " took " << double (time_end - time_start)*10e3 << std::endl;
#endif

    }

    /*! Stitch both the RGB and the depth images corresponding to the sensor 'sensor_id' */
    void stitchImage(int sensor_id)
    {
        Eigen::Vector3d virtualPoint, pointFromCamera;
        const int size_w = frameRGBD_[sensor_id].getRGBImage().cols-1;
        const int size_h = frameRGBD_[sensor_id].getRGBImage().rows-1;
        const float offsetPhi = -sphereRGB.rows/2 + 0.5;
        const float offsetTheta = -frameRGBD_[sensor_id].getRGBImage().rows*15/2 + 0.5;
        const float angle_pixel = 8*frameRGBD_[sensor_id].getRGBImage().rows/(2*PI);

        for(int row_phi=0; row_phi < sphereRGB.rows; row_phi++)
        {
            //        int row_pixels = row_phi * frameRGBD_[sensor_id].getRGBImage().rows;
            float phi_i = (row_phi+offsetPhi) / angle_pixel;// + PI/2;
            virtualPoint(0) = -sin(phi_i);
            float cos_phi = cos(phi_i);

            // Stitch each image from each sensor
            //        for(unsigned cam=0; cam < 8; cam++)
            //      #pragma omp parallel for
            int init_col_sphere = (7-sensor_id)*frameRGBD_[sensor_id].getRGBImage().rows;
            int end_col_sphere = (8-sensor_id)*frameRGBD_[sensor_id].getRGBImage().rows;
            if(sensor_id != 0)
                end_col_sphere += 20;
            if(sensor_id != 7)
                init_col_sphere -= 20;
            for(int col_theta=init_col_sphere; col_theta < end_col_sphere; col_theta++)
            {
                float theta_i = (col_theta+offsetTheta) / angle_pixel; // + PI;
                virtualPoint(1) = cos_phi*sin(theta_i);
                virtualPoint(2) = cos_phi*cos(theta_i);

                //          pointFromCamera = calib->Rt_10.block(0,0,3,3) * virtualPoint + calib->Rt_10.block(0,3,3,1);
                pointFromCamera = calib->Rt_inv[sensor_id].block(0,0,3,3) * virtualPoint + calib->Rt_inv[sensor_id].block(0,3,3,1);
                float u = calib->cameraMatrix(0,0) * pointFromCamera(0) / pointFromCamera(2) + calib->cameraMatrix(0,2);
                float v = calib->cameraMatrix(1,1) * pointFromCamera(1) / pointFromCamera(2) + calib->cameraMatrix(1,2);

                //          #pragma omp critical
                if(u >= 0 && u < size_w && v >= 0 && v < size_h)
                {
                    sphereRGB.at<cv::Vec3b>(row_phi,col_theta) = frameRGBD_[sensor_id].getRGBImage().at<cv::Vec3b>(v,u);

                    if( pcl_isfinite(frameRGBD_[sensor_id].getDepthImage().at<float>(v,u)) )
                        sphereDepth.at<float>(row_phi,col_theta) =frameRGBD_[sensor_id].getDepthImage().at<float>(v,u) * sqrt(1 + pow((u-calib->cameraMatrix(0,2))/calib->cameraMatrix(0,0),2) + pow((v-calib->cameraMatrix(1,2))/calib->cameraMatrix(1,1),2));
                }
            }
        }
    }

};

#endif
