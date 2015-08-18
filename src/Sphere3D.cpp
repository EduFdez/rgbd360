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

#include <Sphere3D.h>
//#include <SphericalModel.h>
//#include <Miscellaneous.h>
#include <DownsampleRGBD.h>
#include <FilterPointCloud.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h> //Save global map as PCD file
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>

#ifdef _ENABLE_OPENMP
    #include <omp.h>
#endif

//typedef pcl::PointXYZRGBA PointT;

using namespace std;

/*! Constructor for the SphericalStereo sensor (outdoor sensor) */
Sphere3D::Sphere3D() :
    node(0),
    sphereCloud(new pcl::PointCloud<PointT>())
    //    bSphereCloudBuilt(false),
{

}

/*! Load a spherical point cloud */
void Sphere3D::loadCloud(const string &pointCloudPath)
{
    // Load pointCloud
    sphereCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PCDReader reader;
    reader.read (pointCloudPath, *sphereCloud);
}


/*! Save the PbMap from an omnidirectional RGB-D image */
void Sphere3D::savePlanes(string pathPbMap)
{
    mrpt::utils::CFileGZOutputStream serialize_planesLabeled(pathPbMap);
    serialize_planesLabeled << planes;
    serialize_planesLabeled.close();
}

/*! Save the pointCloud and PbMap from an omnidirectional RGB-D image */
void Sphere3D::save(string &path, unsigned &frame)
{
    assert(!sphereCloud->empty() && planes.vPlanes.size() > 0);

    string cloudPath = path + mrpt::format("/sphereCloud_%d.pcd",frame);
    pcl::io::savePCDFile(cloudPath, *sphereCloud);

    string pbmapPath = path + mrpt::format("/spherePlanes_%d.pbmap",frame);
    savePlanes(pbmapPath);
}

/*! Downsample and filter the individual point clouds from the different Asus XPL */
void Sphere3D::buildCloudsDownsampleAndFilter()
{
    double time_start = pcl::getTime();

#pragma omp parallel num_threads(NUM_ASUS_SENSORS)
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
    cout << "Build single clouds + downsample + filter took " << double (time_end - time_start) << endl;
}

/*! Fast version of the method 'buildPointCloud'. This one performs more poorly for plane segmentation. */
void Sphere3D::buildPointCloud_fast()
{
#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

#pragma omp parallel num_threads(NUM_ASUS_SENSORS) // I still don't understand why this doesn't work
    //      for(unsigned sensor_id=0; sensor_id < NUM_ASUS_SENSORS; ++sensor_id)
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
    for(unsigned sensor_id=1; sensor_id < NUM_ASUS_SENSORS; ++sensor_id)
        *sphereCloud += *cloud_[sensor_id];

    sphereCloud->height = cloud_[0]->width;
    sphereCloud->width = NUM_ASUS_SENSORS * cloud_[0]->height;
    sphereCloud->is_dense = false;

#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "buildPointCloud_fast took " << double (time_end - time_start) << endl;
#endif

}

/*! Build the spherical point cloud. The reference system is the one used by the INRIA SphericalStereo sensor. Z points forward, X points to the right and Y points downwards */
void Sphere3D::buildPointCloud()
{
//    if(bSphereCloudBuilt) // Avoid building twice the spherical point cloud
//      return;

#if _PRINT_PROFILING
    cout << " Sphere3D_stereo::buildPointCloud... " << endl;
    double time_start = pcl::getTime();
#endif

    sphereCloud->resize(sphereRGB.rows*sphereRGB.cols);
    sphereCloud->height = sphereRGB.rows;
    sphereCloud->width = sphereRGB.cols;
    sphereCloud->is_dense = false;
    const float min_depth = 0.4f;
    const float max_depth = 20.f;

    const float pixel_angle_ = 2*PI / sphereDepth.cols;
    const int half_width = sphereDepth.cols/2;

    //  Efficiency: store the values of the trigonometric functions
    Eigen::VectorXf v_sinTheta(sphereDepth.cols);
    Eigen::VectorXf v_cosTheta(sphereDepth.cols);
    float *sinTheta = &v_sinTheta[0];
    float *cosTheta = &v_cosTheta[0];
    for(int col_theta=-half_width; col_theta < half_width; ++col_theta)
    {
        float theta = (col_theta+0.5f)*pixel_angle_;
        *(sinTheta++) = sin(theta);
        *(cosTheta++) = cos(theta);
    }
    size_t start_row = (sphereRGB.cols-sphereRGB.rows) / 2;

    float *depth = sphereDepth.ptr<float>(0);
    cv::Vec3b *intensity = sphereRGB.ptr<cv::Vec3b>(0);
#if _ENABLE_OPENMP
    #pragma omp parallel for
#endif
    //for(int row_phi=0; row_phi < sphereDepth.rows; row_phi++)
    for(int row_phi=0, pixel_index=0; row_phi < sphereDepth.rows; row_phi++)//, row_phi += width_SphereImg)
    {
        float sin_phi = v_sinTheta[start_row+row_phi];
        float cos_phi = v_cosTheta[start_row+row_phi];

        int col_count = 0;
        for(int col_theta=-half_width; col_theta < half_width; ++col_theta, ++pixel_index, ++col_count)
        {
            if((*depth) > min_depth && (*depth) < max_depth)
            {
                //cout << min_depth << " depth " << *depth << " max_depth " << max_depth << endl;
                sphereCloud->points[pixel_index].x = (*depth) * cos_phi * v_sinTheta[col_count];
                sphereCloud->points[pixel_index].y = (*depth) * sin_phi;
                sphereCloud->points[pixel_index].z = (*depth) * cos_phi * v_cosTheta[col_count];
                sphereCloud->points[pixel_index].r = (*intensity)[2];
                sphereCloud->points[pixel_index].g = (*intensity)[1];
                sphereCloud->points[pixel_index].b = (*intensity)[0];
            }
            else
            {
                sphereCloud->points[pixel_index].x = numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].y = numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].z = numeric_limits<float>::quiet_NaN ();
            }
            ++depth;
            ++intensity;
        }
    }

#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "Sphere3D::buildPointCloud() took " << double (time_end - time_start)*1000 << " ms. \n";
#endif
}

///*! Build the spherical point cloud. The reference system is the one used by the INRIA SphericalStereo sensor. Z points forward, X points to the right and Y points downwards */
//void Sphere3D::buildPointCloud2()
//{
////    if(bSphereCloudBuilt) // Avoid building twice the spherical point cloud
////      return;

//#if _PRINT_PROFILING
//    cout << " Sphere3D_stereo::buildPointCloud2... " << endl;
//    double time_start = pcl::getTime();
//#endif

//    sphereCloud->resize(sphereRGB.rows*sphereRGB.cols);
//    sphereCloud->height = sphereRGB.rows;
//    sphereCloud->width = sphereRGB.cols;
//    sphereCloud->is_dense = false;
//    size_t img_size = sphereRGB.rows * sphereRGB.cols;

//    SphericalModel proj;
//    Eigen::MatrixXf xyz;
//    Eigen::VectorXi validPixels;
//    proj.reconstruct3D(sphereDepth, xyz, validPixels);

//    //float *depth = sphereDepth.ptr<float>(0);
//    cv::Vec3b *rgb = sphereRGB.ptr<cv::Vec3b>(0);

//#if _ENABLE_OPENMP
//    #pragma omp parallel for
//#endif
//    for(size_t i=0; i < img_size; i++)//, row_phi += width_SphereImg)
//    {
//        //if(validPixels(i) >= 0)
//        if(validPixels(i) != -1)
//        {
//            //cout << min_depth << " depth " << *depth << " max_depth " << max_depth << endl;
//            sphereCloud->points[i].x = xyz(i,0);
//            sphereCloud->points[i].y = xyz(i,1);
//            sphereCloud->points[i].z = xyz(i,2);
//            sphereCloud->points[i].r = (*rgb)[2];
//            sphereCloud->points[i].g = (*rgb)[1];
//            sphereCloud->points[i].b = (*rgb)[0];
//        }
//        else
//        {
//            sphereCloud->points[i].x = numeric_limits<float>::quiet_NaN ();
//            sphereCloud->points[i].y = numeric_limits<float>::quiet_NaN ();
//            sphereCloud->points[i].z = numeric_limits<float>::quiet_NaN ();
//        }
//        //++depth;
//        ++rgb;
//    }

//#if _PRINT_PROFILING
//    double time_end = pcl::getTime();
//    cout << "Sphere3D::buildPointCloud2() took " << double (time_end - time_start)*1000 << " ms. \n";
//#endif
//}

/*! Perform bilateral filtering on the point cloud
    */
void Sphere3D::filterCloudBilateral_stereo()
{
#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

    // Filter pointClouds
    pcl::FastBilateralFilter<PointT> filter;
    //      filter.setSigmaS(10.0);
    //      filter.setSigmaR(0.05);
    filter.setSigmaS(20.0);
    filter.setSigmaR(0.2);
    filter.setInputCloud(sphereCloud);
    //      filter.filter(*filteredCloud);
    filter.filter(*sphereCloud);

#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "filterCloudBilateral in " << (time_end - time_start)*1000 << " ms\n";
#endif
}

/*! Compute the normalMap from an organized cloud of normal vectors. */
void Sphere3D::computeNormalMap(const pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud, cv::Mat & normalMap, const bool display)
{
    normalMap.create(sphereCloud->height, sphereCloud->width, CV_8UC3);
    for(size_t r=0; r < sphereCloud->height; ++r)
        for(size_t c=0; c < sphereCloud->width; ++c)
        {
            int i = r*sphereCloud->width + c;
    //            if( normal_cloud->points[i].normal_x == numeric_limits<float>::quiet_NaN () )
            if( normal_cloud->points[i].normal_x < 2.f )
            {
                normalMap.at<cv::Vec3b>(r,c)[2] = 255*(0.5*normal_cloud->points[i].normal_x+0.5);
                normalMap.at<cv::Vec3b>(r,c)[1] = 255*(0.5*normal_cloud->points[i].normal_y+0.5);
                normalMap.at<cv::Vec3b>(r,c)[0] = 255*(0.5*normal_cloud->points[i].normal_z+0.5);
    //            // Create a RGB image file with the values of the normal (multiply by 1.5 to increase the saturation to 0.5 = 1.5*0.33. Notice that normal vectors can be directly interpreted as normalized rgb, which has constant saturation of 0.33).
    //            normalMap.at<cv::Vec3b>(r,c)[2] = max(255., 255*(1.5f*fabs(normal_cloud->points[i].normal_x) ) );
    //            normalMap.at<cv::Vec3b>(r,c)[1] = max(255., 255*(1.5f*fabs(normal_cloud->points[i].normal_y) ) );
    //            normalMap.at<cv::Vec3b>(r,c)[0] = max(255., 255*(1.5f*fabs(normal_cloud->points[i].normal_z) ) );
            }
            //else
                //cout << "normal " << normal_cloud->points[i].normal_x << " " << normal_cloud->points[i].normal_y << " " << normal_cloud->points[i].normal_z << endl;
//                cv::Vec3b white(255,255,255);
//                for(i=0; i < 20; i++)
//                    normalMap.at<cv::Vec3b>(i) = white;
        }
    if(display)
    {
        cv::imshow("normalMap",normalMap);
        cv::imwrite("/Data/Results_IROS15/normalMap.png",normalMap);
        cv::waitKey();
        cv::destroyWindow("normalMap");
    }
}
