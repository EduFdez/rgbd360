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

#include <Frame360.h>

#define ENABLE_OPENMP 1

//typedef pcl::PointXYZRGBA PointT;

/*! Constructor for the SphericalStereo sensor (outdoor sensor) */
Frame360::Frame360() :
    sphereCloud(new pcl::PointCloud<PointT>()),
    sensor_type(STEREO_OUTDOOR),
    //    bSphereCloudBuilt(false),
    node(0)
{
    double time_start = pcl::getTime();
}

/*! Constructor for the sensor RGBD360 (8 Asus XPL)*/
Frame360::Frame360(Calib360 *calib360) :
    calib(calib360),
    sphereCloud(new pcl::PointCloud<PointT>()),
    sensor_type(RGBD360_INDOOR),
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
float Frame360::getPlanarArea()
{
    float planarArea = 0;
    for(unsigned i = 0; i < planes.vPlanes.size(); i++)
        planarArea += planes.vPlanes[i].areaHull;

    return planarArea;
}

/*! Load a spherical point cloud */
void Frame360::loadCloud(const std::string &pointCloudPath)
{
    // Load pointCloud
    sphereCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PCDReader reader;
    reader.read (pointCloudPath, *sphereCloud);
}

/*! Load a spherical PbMap */
void Frame360::loadPbMap(std::string &pbmapPath)
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
void Frame360::load_PbMap_Cloud(std::string &pointCloudPath, std::string &pbmapPath)
{
    // Load pointCloud
    loadCloud(pointCloudPath);

    // Load planes
    loadPbMap(pbmapPath);
}

/*! Load a spherical frame from its point cloud and its PbMap files */
void Frame360::load_PbMap_Cloud(std::string &path, unsigned &index)
{
    std::string pointCloudPath = path + mrpt::format("/sphereCloud_%u.pcd", index);
    std::string pbmapPath = path + mrpt::format("/spherePlanes_%u.pbmap", index);
    load_PbMap_Cloud(pointCloudPath, pbmapPath);
}

/*! Load a spherical RGB-D image from the raw data stored in a binary file */
void Frame360::loadFrame(std::string &binaryFile)
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

/*! Undistort the omnidirectional depth image using the models acquired with CLAMS */
void Frame360::undistort()
{
    double time_start = pcl::getTime();
    //    std::cout << "Undistort Frame360\n";

#pragma omp parallel num_threads(8)
    {
        int sensor_id = omp_get_thread_num();
        undistortDepthSensor(sensor_id);
        //      cv::eigen2cv(frameRGBD_[sensor_id].m_depthEigUndistort, frameRGBD_[sensor_id].getDepthImage());
    }

    //#if _DEBUG_MSG
    double time_end = pcl::getTime();
    std::cout << "Undistort Frame360 took " << double (time_end - time_start) << std::endl;
    //#endif

}

/*! Save the PbMap from an omnidirectional RGB-D image */
void Frame360::savePlanes(std::string pathPbMap)
{
    mrpt::utils::CFileGZOutputStream serialize_planesLabeled(pathPbMap);
    serialize_planesLabeled << planes;
    serialize_planesLabeled.close();
}

/*! Save the pointCloud and PbMap from an omnidirectional RGB-D image */
void Frame360::save(std::string &path, unsigned &frame)
{
    assert(!sphereCloud->empty() && planes.vPlanes.size() > 0);

    std::string cloudPath = path + mrpt::format("/sphereCloud_%d.pcd",frame);
    pcl::io::savePCDFile(cloudPath, *sphereCloud);

    std::string pbmapPath = path + mrpt::format("/spherePlanes_%d.pbmap",frame);
    savePlanes(pbmapPath);
}

/*! Serialize the omnidirectional RGB-D image */
void Frame360::serialize(std::string &fileName)
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
void Frame360::fastStitchImage360() // Parallelized with OpenMP
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
void Frame360::stitchSphericalImage() // Parallelized with OpenMP
{
    //        cout << "stitchSphericalImage\n";
    double time_start = pcl::getTime();

    int width_SphereImg = frameRGBD_[0].getRGBImage().rows * 8;
    int height_SphereImg = width_SphereImg * 0.5 * 60.0/180; // Store only the part of the sphere which contains information
    sphereRGB = cv::Mat::zeros(height_SphereImg, width_SphereImg, CV_8UC3);
    sphereDepth = cv::Mat::zeros(height_SphereImg, width_SphereImg, CV_16UC1);
#pragma omp parallel num_threads(8)
    //        for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
    {
        int sensor_id = omp_get_thread_num();
        stitchImage(sensor_id);
        // cout << "Stitcher omp " << sensor_id << endl;
    }

    double time_end = pcl::getTime();
    std::cout << "stitchSphericalImage took " << double (time_end - time_start) << std::endl;
}

/*! Downsample and filter the individual point clouds from the different Asus XPL */
void Frame360::buildCloudsDownsampleAndFilter()
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
void Frame360::buildCloud_id(int sensor_id)
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

/*! Build the spherical point cloud by superimposing the 8 point clouds from the 8 Asus XPL*/
void Frame360::buildSphereCloud_rgbd360()
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
void Frame360::buildSphereCloud_fast()
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

/*! Build the spherical point cloud. The reference system is the one used by the INRIA SphericalStereo sensor. Z points forward, X points to the right and Y points downwards */
void Frame360::buildSphereCloud()
{
    //        std::cout << " Frame360_stereo::buildSphereCloud_rgbd360() " << std::endl;
    //    if(bSphereCloudBuilt) // Avoid building twice the spherical point cloud
    //      return;

    double time_start = pcl::getTime();

    sphereCloud->resize(sphereRGB.rows*sphereRGB.cols);
    sphereCloud->height = sphereRGB.rows;
    sphereCloud->width = sphereRGB.cols;
    sphereCloud->is_dense = false;
    const float min_depth = 0.4f;
    const float max_depth = 20.f;

    const float pixel_angle_ = 2*PI / sphereDepth.cols;
    const float step_theta = pixel_angle_;
    const float step_phi = pixel_angle_;
    //int phi_start_pixel_ = 166, end_phi = 166 + sphereDepth.rows; // For images of 665x2048
    const int phi_start_pixel_ = 174, end_phi = 174 + sphereDepth.rows; // For images of 640x2048
    //const float offset_phi = PI*31.5/180; // height_SphereImg((width_SphereImg/2) * 63.0/180), // RGBD360
    const int half_height = sphereDepth.rows/2;
    const int half_width = sphereDepth.cols/2;

//    std::cout << "half_height " << half_height << std::endl;
//    std::cout << "half_width " << half_width << std::endl;
//    std::cout << "ENABLE_OPENMP " << ENABLE_OPENMP << std::endl;

    //  Efficiency: store the values of the trigonometric functions
    Eigen::VectorXf v_sinTheta(sphereDepth.cols);
    Eigen::VectorXf v_cosTheta(sphereDepth.cols);
    float *sinTheta = &v_sinTheta[0];
    float *cosTheta = &v_cosTheta[0];
    for(int col_theta=-half_width; col_theta < half_width; ++col_theta)
    {
        float theta = col_theta*pixel_angle_;
        *(sinTheta++) = sin(theta);
        *(cosTheta++) = cos(theta);
    }

//time_start = pcl::getTime();
//for(unsigned i=0;i<1000;++i)
//{
#if ENABLE_OPENMP
    #pragma omp parallel for
    for(int row_phi=0; row_phi < sphereDepth.rows; row_phi++)
    {
        //float phi = offset_phi - row_phi*angle_pixel_inv;// + PI/2;   // RGBD360
        //const float phi = (row_phi-half_height)*step_phi;
        const float phi = (row_phi+phi_start_pixel_)*step_phi - PI/2;
        const float sin_phi = sin(phi);
        const float cos_phi = cos(phi);

        int index_row = row_phi*sphereDepth.cols;
        for(int col_theta=0, pixel_index=index_row; col_theta < sphereDepth.cols; ++col_theta, ++pixel_index)
        {
            float depth = sphereDepth.at<float> (row_phi, col_theta);
            if(depth > min_depth && depth < max_depth)
            {
//                float theta = col_theta*step_theta - PI;
//                sphereCloud->points[pixel_index].x = depth * cos_phi * sin(theta);
//                sphereCloud->points[pixel_index].y = -depth* sin_phi;
//                sphereCloud->points[pixel_index].z = depth * cos_phi * cos(theta);
                sphereCloud->points[pixel_index].x = depth * cos_phi * v_sinTheta[col_theta];
                sphereCloud->points[pixel_index].y = depth* sin_phi;
                sphereCloud->points[pixel_index].z = depth * cos_phi * v_cosTheta[col_theta];
                cv::Vec3b intensity = sphereRGB.at<cv::Vec3b>(row_phi, col_theta);
                sphereCloud->points[pixel_index].r = intensity[2];
                sphereCloud->points[pixel_index].g = intensity[1];
                sphereCloud->points[pixel_index].b = intensity[0];
            }
            else
            {
                sphereCloud->points[pixel_index].x = std::numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].y = std::numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].z = std::numeric_limits<float>::quiet_NaN ();
            }
        }
    }

//        int index_row = row_phi*sphereDepth.cols;
//        float *depth = sphereDepth.ptr<float>(index_row*sizeof(float));
//        cv::Vec3b *intensity = sphereRGB.ptr<cv::Vec3b>(index_row);
//        std::cout << " sphereCloud->points " << sphereCloud->points.size() << std::endl;
//        std::cout << " sphereDepth " << sphereDepth.rows << std::endl;
//        for(int col_theta=0, pixel_index=index_row; col_theta < sphereDepth.cols; ++col_theta, ++pixel_index)
//        {
//            std::cout << sphereDepth.size().area() << " pixel_index " << pixel_index << " row_phi " << row_phi << " col_theta " << col_theta << std::endl;
//            std::cout << " depth " << sphereDepth.at<float> (row_phi, col_theta) << std::endl;
//            std::cout << " depth " << (*depth) << std::endl;
//            if((*depth) > min_depth && (*depth) < max_depth)
//            {
//                std::cout << " depth " << sphereDepth.at<float> (row_phi, col_theta) << " max_depth " << max_depth << std::endl;
//                std::cout << min_depth << " depth " << *depth << " max_depth " << max_depth << std::endl;
//                sphereCloud->points[pixel_index].x = v_sinTheta[col_theta] *cos_phi * (*depth);
//                sphereCloud->points[pixel_index].y = -sin_phi * (*depth);
//                sphereCloud->points[pixel_index].z = v_cosTheta[col_theta] *cos_phi * (*depth);
//                sphereCloud->points[pixel_index].r = (*intensity)[2];
//                sphereCloud->points[pixel_index].g = (*intensity)[1];
//                sphereCloud->points[pixel_index].b = (*intensity)[0];
//            }
//            else
//            {
//                sphereCloud->points[pixel_index].x = std::numeric_limits<float>::quiet_NaN ();
//                sphereCloud->points[pixel_index].y = std::numeric_limits<float>::quiet_NaN ();
//                sphereCloud->points[pixel_index].z = std::numeric_limits<float>::quiet_NaN ();
//            }
//            ++depth;
//            ++intensity;
//        }

//    for(int row_phi=0; row_phi < sphereDepth.rows; row_phi++)//, row_phi += width_SphereImg)
//    {
//        //float phi = offset_phi - row_phi*angle_pixel_inv;// + PI/2;   // RGBD360
//        float phi = (half_height-row_phi)*step_phi;
//        float cos_phi = cos(phi);
//        float sin_phi = sin(phi);

//        int index_row = row_phi*sphereDepth.cols;
//        for(int col_theta=0, pixel_index=index_row; col_theta < sphereDepth.cols; ++col_theta, ++pixel_index)
//        {
//            float depth = sphereDepth.at<float> (row_phi, col_theta);
//            //std::cout << pixel_index << " " << depth << std::endl;

//            //                    // RGBD360
//            //float theta_i = col_theta*angle_pixel_inv; // - PI;
//            //float depth = 0.001f * sphereDepth.at<unsigned short>(row_phi,col_theta);

//            if(depth > min_depth && depth < max_depth)
//            {
//                //                    // RGBD360
//                //                    sphereCloud->points[row_pixels+col_theta].x = sin_phi * depth;
//                //                    sphereCloud->points[row_pixels+col_theta].y = -cos_phi * sin(theta_i) * depth;
//                //                    sphereCloud->points[row_pixels+col_theta].z = -cos_phi * cos(theta_i) * depth;

//                float theta = col_theta*step_theta - PI;
//                sphereCloud->points[pixel_index].x = sin(theta)*cos_phi * depth;
//                sphereCloud->points[pixel_index].y = -sin_phi * depth;
//                sphereCloud->points[pixel_index].z = cos(theta)*cos_phi * depth;
//                sphereCloud->points[pixel_index].r = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[2];
//                sphereCloud->points[pixel_index].g = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[1];
//                sphereCloud->points[pixel_index].b = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[0];
//            }
//            else
//            {
//                sphereCloud->points[pixel_index].x = std::numeric_limits<float>::quiet_NaN ();
//                sphereCloud->points[pixel_index].y = std::numeric_limits<float>::quiet_NaN ();
//                sphereCloud->points[pixel_index].z = std::numeric_limits<float>::quiet_NaN ();
//            }
//        }
//    }
#else
//    //  Efficiency: store the values of the trigonometric functions
//    Eigen::VectorXf v_sinTheta(sphereDepth.cols);
//    Eigen::VectorXf v_cosTheta(sphereDepth.cols);
//    float *sinTheta = &v_sinTheta[0];
//    float *cosTheta = &v_cosTheta[0];
//    for(int col_theta=-half_width; col_theta < half_width; ++col_theta)
//    {
//        float theta = col_theta*pixel_angle_;
//        *(sinTheta++) = sin(theta);
//        *(cosTheta++) = cos(theta);
//    }

    float *depth = sphereDepth.ptr<float>(0);
    cv::Vec3b *intensity = sphereRGB.ptr<cv::Vec3b>(0);
    const int minus_half_height = -half_height;
    for(int row_phi=half_height, pixel_index=0; row_phi > minus_half_height; --row_phi)//, row_phi += width_SphereImg)
    {
        //float phi = offset_phi - row_phi*angle_pixel_inv;// + PI/2;   // RGBD360
        float phi = (row_phi-half_height)*step_phi;
        float sin_phi = sin(phi);
        float cos_phi = cos(phi);

        int col_count = 0;
        for(int col_theta=-half_width; col_theta < half_width; ++col_theta, ++pixel_index, ++col_count)
        {
            if((*depth) > min_depth && (*depth) < max_depth)
            {
                //std::cout << min_depth << " depth " << *depth << " max_depth " << max_depth << std::endl;
                sphereCloud->points[pixel_index].x = (*depth) * cos_phi * v_sinTheta[col_count];
                sphereCloud->points[pixel_index].y = (*depth)* sin_phi;
                sphereCloud->points[pixel_index].z = (*depth) * cos_phi * v_cosTheta[col_count];
                sphereCloud->points[pixel_index].r = (*intensity)[2];
                sphereCloud->points[pixel_index].g = (*intensity)[1];
                sphereCloud->points[pixel_index].b = (*intensity)[0];
            }
            else
            {
                sphereCloud->points[pixel_index].x = std::numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].y = std::numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].z = std::numeric_limits<float>::quiet_NaN ();
            }
            ++depth;
            ++intensity;
        }
    }
#endif
//}
//double time_end = pcl::getTime();
//std::cout << "Construction took " << double (time_end - time_start) << std::endl;

    //    bSphereCloudBuilt = true;
    //std::cout << " Frame360::buildSphereCloud_rgbd360() finished" << std::endl;

//#if _DEBUG_MSG
    double time_end = pcl::getTime();
    std::cout << "Frame360::buildSphereCloud() took " << double (time_end - time_start) << std::endl;
//#endif
}

/*! Build the spherical point cloud. The reference system is the one used by the INRIA SphericalStereo sensor. Z points forward, X points to the right and Y points downwards */
void Frame360::buildSphereCloud_old()
{
    //    if(bSphereCloudBuilt) // Avoid building twice the spherical point cloud
    //      return;

    //        std::cout << " Frame360_stereo::buildSphereCloud_rgbd360() " << std::endl;

    double time_start = pcl::getTime();

    sphereCloud->resize(sphereRGB.rows*sphereRGB.cols);
    sphereCloud->height = sphereRGB.rows;
    sphereCloud->width = sphereRGB.cols;
    sphereCloud->is_dense = false;
    float min_depth = 0.4f;
    float max_depth = 20.f;

    pixel_angle_ = 2*PI / sphereDepth.cols;
    float step_theta = pixel_angle_;
    float step_phi = pixel_angle_;
    //int phi_start_pixel_ = 166, end_phi = 166 + sphereDepth.rows; // For images of 665x2048
    int phi_start_pixel_ = 174, end_phi = 174 + sphereDepth.rows; // For images of 640x2048
    //float offset_phi = PI*31.5/180; // height_SphereImg((width_SphereImg/2) * 63.0/180), // RGBD360
    const int half_height = sphereDepth.rows/2;

#if ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int row_phi=0; row_phi < sphereDepth.rows; row_phi++)//, row_phi += width_SphereImg)
    {
        //float phi = offset_phi - row_phi*angle_pixel_inv;// + PI/2;   // RGBD360
        float phi = (row_phi-half_height)*step_phi;
        float cos_phi = cos(phi);
        float sin_phi = sin(phi);

        int index_row = row_phi*sphereDepth.cols;
        for(int col_theta=0, pixel_index=index_row; col_theta < sphereDepth.cols; ++col_theta, ++pixel_index)
        {
            float depth = sphereDepth.at<float> (row_phi, col_theta);
            //std::cout << pixel_index << " " << depth << std::endl;

            //                    // RGBD360
            //float theta_i = col_theta*angle_pixel_inv; // - PI;
            //float depth = 0.001f * sphereDepth.at<unsigned short>(row_phi,col_theta);

            if(depth > min_depth && depth < max_depth)
            {
                //                    // RGBD360
                //                    sphereCloud->points[row_pixels+col_theta].x = sin_phi * depth;
                //                    sphereCloud->points[row_pixels+col_theta].y = -cos_phi * sin(theta_i) * depth;
                //                    sphereCloud->points[row_pixels+col_theta].z = -cos_phi * cos(theta_i) * depth;

                float theta = col_theta*step_theta - PI;
                sphereCloud->points[pixel_index].x = depth * cos_phi * sin(theta);
                sphereCloud->points[pixel_index].y = depth* sin_phi;
                sphereCloud->points[pixel_index].z = depth * cos_phi * cos(theta);
                sphereCloud->points[pixel_index].r = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[2];
                sphereCloud->points[pixel_index].g = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[1];
                sphereCloud->points[pixel_index].b = sphereRGB.at<cv::Vec3b>(row_phi,col_theta)[0];
            }
            else
            {
                sphereCloud->points[pixel_index].x = std::numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].y = std::numeric_limits<float>::quiet_NaN ();
                sphereCloud->points[pixel_index].z = std::numeric_limits<float>::quiet_NaN ();
            }
        }
    }

    //    bSphereCloudBuilt = true;
    //std::cout << " Frame360::buildSphereCloud_rgbd360() finished" << std::endl;

//#if _DEBUG_MSG
    double time_end = pcl::getTime();
    std::cout << "PointCloud sphere construction OLD took " << double (time_end - time_start) << std::endl;
//#endif
}

/*! Create the PbMap of the spherical point cloud */
void Frame360::getPlanes()
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
void Frame360::getLocalPlanes()
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
void Frame360::mergePlanes()
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
void Frame360::groupPlanes()
{
    //  cout << "groupPlanes...\n";
    double time_start = pcl::getTime();

    float maxDistHull = 0.5;
    float maxDistParallelHull = 0.09;

    //    Eigen::Matrix4f Rt = calib->getRt_id(0);
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

/*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in its local frame of reference
  */
void Frame360::getLocalPlanesInFrame(int sensor_id)
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
void Frame360::getPlanesSensor(int sensor_id)
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
void Frame360::undistortDepthSensor(int sensor_id)
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
void Frame360::stitchImage(int sensor_id)
{
    Eigen::Vector3f virtualPoint, pointFromCamera;
    const int size_w = frameRGBD_[sensor_id].getRGBImage().cols;
    const int size_h = frameRGBD_[sensor_id].getRGBImage().rows;
    const float offsetPhi = sphereRGB.rows/2 - 0.5;
    const float offsetTheta = -frameRGBD_[sensor_id].getRGBImage().rows*15/2 + 0.5;
    const float angle_pixel = 2*PI/sphereRGB.cols;
    const int marginStitching = 0;

    for(int row_phi=0; row_phi < sphereRGB.rows; row_phi++)
    {
        //        int row_pixels = row_phi * size_h;
        float phi_i = (offsetPhi-row_phi) * angle_pixel;// + PI/2;
        virtualPoint(0) = sin(phi_i);
        float cos_phi = cos(phi_i);

        // Stitch each image from each sensor
        //        for(unsigned cam=0; cam < 8; cam++)
        //      #pragma omp parallel for
        int init_col_sphere = (7-sensor_id)*size_h;
        int end_col_sphere = (8-sensor_id)*size_h;
        if(sensor_id != 0)
            end_col_sphere += marginStitching;
        if(sensor_id != 7)
            init_col_sphere -= marginStitching;
        for(int col_theta=init_col_sphere; col_theta < end_col_sphere; col_theta++)
        {
            float theta_i = (col_theta+offsetTheta) * angle_pixel; // + PI;
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
                //                    std::cout << " d " << frameRGBD_[sensor_id].getDepthImage().at<unsigned short>(v,u);
                if( pcl_isfinite(frameRGBD_[sensor_id].getDepthImage().at<unsigned short>(v,u)) ){
                    sphereDepth.at<unsigned short>(row_phi,col_theta) = frameRGBD_[sensor_id].getDepthImage().at<unsigned short>(v,u) * sqrt(1 + pow((u-calib->cameraMatrix(0,2))/calib->cameraMatrix(0,0),2) + pow((v-calib->cameraMatrix(1,2))/calib->cameraMatrix(1,1),2));
                    //                        std::cout << " dm " << sphereDepth.at<unsigned short>(row_phi,col_theta);
                }
            }
        }
    }
}

// Functions for SphereStereo images (outdoors)
/*! Load a spherical RGB-D image from the raw data stored in a binary file */
void Frame360::loadDepth (const std::string &binaryDepthFile, const cv::Mat * mask)
{
    double time_start = pcl::getTime();

    std::ifstream file (binaryDepthFile.c_str(), std::ios::in | std::ios::binary);
    if (file)
    {
        char *header_property = new char[2]; // Read height_ and width_
        file.seekg (0, ios::beg);
        file.read (header_property, 2);
        unsigned short *height = reinterpret_cast<unsigned short*> (header_property);
        height_ = *height;

        //file.seekg (2, ios::beg);
        file.read (header_property, 2);
        unsigned short *width = reinterpret_cast<unsigned short*> (header_property);
        width_ = *width;
        //std::cout << "height_ " << height_ << " width_ " << width_ << std::endl;

        cv::Mat sphereDepth_aux(width_, height_, CV_32FC1);
        char *mem_block = reinterpret_cast<char*>(sphereDepth_aux.data);
        std::streampos size = height_*width_*4; // file.tellg() - std::streampos(4); // Header is 4 bytes: 2 unsigned short for height and width
        //file.seekg (4, ios::beg);
        file.read (mem_block, size);

        //            cv::Mat sphereDepth_aux2(height_, width_, CV_32FC1);
        //            for(int i = 0; i < height_; i++)
        //                for(int j = 0; j < width_; j++)
        //                    sphereDepth_aux2.at<float>(i,j) = *(reinterpret_cast<float*>(mem_block + 4*(j*height_+i)));
        //            cv::imshow( "sphereDepth", sphereDepth_aux2 );
        //            cv::waitKey(0);

        //Close the binary bile
        file.close();
        //            sphereDepth.create(height_, width_, CV_32FC1);
        //            cv::transpose(sphereDepth_aux, sphereDepth);
        sphereDepth.create(640, width_, CV_32FC1);
        cv::Rect region_of_interest = cv::Rect(8, 0, 640, width_); // Select only a portion of the image with height = 640 to facilitate the pyramid constructions
        cv::transpose(sphereDepth_aux(region_of_interest), sphereDepth); // The savecd image is transposed wrt to the RGB img!

        if (mask && sphereDepth_aux.rows == mask->cols && sphereDepth_aux.cols == mask->rows){
            cv::Mat aux;
            sphereDepth.copyTo(aux, (*mask)(cv::Rect (0, 8, width_, 640) ));
            sphereDepth = aux;
        }
        // std::cout << "height_ " << sphereDepth.rows << " width_ " << sphereDepth.cols << std::endl;
    }
    else
        std::cerr << "File: " << binaryDepthFile << " does NOT EXIST.\n";

    //    bSphereCloudBuilt = false; // The spherical PointCloud of the frame just loaded is not built yet

#if _DEBUG_MSG
    double time_end = pcl::getTime();
    std::cout << "loadDepth took " << double (time_end - time_start) << std::endl;
#endif
}

/*! Load a spherical RGB-D image from the raw data stored in a binary file */
void Frame360::loadRGB(std::string &fileNamePNG)
{
    double time_start = pcl::getTime();

    std::ifstream file (fileNamePNG.c_str(), std::ios::in | std::ios::binary);
    if (file)
    {
        //            sphereRGB = cv::imread (fileNamePNG.c_str(), CV_LOAD_IMAGE_COLOR); // Full size 665x2048

        cv::Mat sphereRGB_aux = cv::imread (fileNamePNG.c_str(), CV_LOAD_IMAGE_COLOR);
        cv::Rect region_of_interest = cv::Rect(0, 8, width_, 640); // Select only a portion of the image with height = 640 to facilitate the pyramid constructions
        sphereRGB.create(640, width_, sphereRGB_aux.type () );
        sphereRGB = sphereRGB_aux (region_of_interest); // Size 640x2048
    }
    else
        std::cerr << "File: " << fileNamePNG << " does NOT EXIST.\n";

#if _DEBUG_MSG
    double time_end = pcl::getTime();
    std::cout << "loadRGB took " << double (time_end - time_start) << std::endl;
#endif
}

/*! Perform bilateral filtering on the point cloud
    */
void Frame360::filterCloudBilateral_stereo()
{
    double time_start = pcl::getTime();

    // Filter pointClouds
    pcl::FastBilateralFilter<PointT> filter;
    //      filter.setSigmaS(10.0);
    //      filter.setSigmaR(0.05);
    filter.setSigmaS(20.0);
    filter.setSigmaR(0.2);
    filter.setInputCloud(sphereCloud);
    //      filter.filter(*filteredCloud);
    filter.filter(*sphereCloud);

#if _DEBUG_MSG
    double time_end = pcl::getTime();
    std::cout << "filterCloudBilateral in " << (time_end - time_start)*1000 << " ms\n";
#endif
}


/*! This function segments planes from the point cloud
    */
void Frame360::segmentPlanesStereo()
{
    // Segment planes
    //    std::cout << "extractPlaneFeatures, size " << sphereCloud->size() << "\n";
    double extractPlanes_start = pcl::getTime();
    assert(sphereCloud->height > 1 && sphereCloud->width > 1);

    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    //      ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);
    //      ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    //      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor (0.1); // For VGA: 0.02f, 10.01
    ne.setNormalSmoothingSize (10.0f);
    ne.setDepthDependentSmoothing (true);

    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    //      mps.setMinInliers (std::max(uint32_t(40),sphereCloud->height*2));
    mps.setMinInliers (1000);
    mps.setAngularThreshold (0.07); // (0.017453 * 2.0 = 0.039812) // 3 degrees
    mps.setDistanceThreshold (0.1); //2cm
    //    cout << "PointCloud size " << sphereCloud->size() << endl;

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    ne.setInputCloud ( sphereCloud );
    ne.compute (*normal_cloud);

    // Visualize normal map in RGB
//    cv::Mat normalMap = cv::Mat::zeros(sphereCloud->height, sphereCloud->width, CV_8UC3);;
//    computeNormalMap(normal_cloud, normalMap, true);

    mps.setInputNormals (normal_cloud);
    mps.setInputCloud ( sphereCloud );
    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;
    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

    // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
    unsigned single_cloud_size = sphereCloud->size();
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
        extract.setInputCloud ( sphereCloud );
        extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices[i]) );
        extract.setNegative (false);
        extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
        plane.inliers.resize(inlier_indices[i].indices.size());
        for(size_t j=0; j<inlier_indices[i].indices.size(); j++)
            plane.inliers[j] = inlier_indices[i].indices[j];

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
        //plane.transform(Rt);
        //      color_end = pcl::getTime();
        //    std::cout << "transform in " << (color_end - color_start)*1000 << " ms\n";

        bool isSamePlane = false;
        if(plane.curvature < max_curvature_plane)
            for (size_t j = 0; j < planes.vPlanes.size(); j++)
                if( planes.vPlanes[j].curvature < max_curvature_plane && planes.vPlanes[j].isSamePlane(plane, 0.99, 0.05, 0.2) ) // The planes are merged if they are the same
                {
                    //          cout << "Merge local region\n";
                    isSamePlane = true;
                    //            double time_start = pcl::getTime();
                    planes.vPlanes[j].mergePlane2(plane);
                    //            double time_end = pcl::getTime();
                    //          std::cout << " mergePlane2 took " << double (time_start - time_end) << std::endl;

                    break;
                }
        if(!isSamePlane)
        {
            //          plane.calcMainColor();
            plane.id = planes.vPlanes.size();
            planes.vPlanes.push_back(plane);
        }
    }
#if _DEBUG_MSG
    double extractPlanes_end = pcl::getTime();
    std::cout << "getPlanesInFrame in " << (extractPlanes_end - extractPlanes_start)*1000 << " ms\n";
#endif
    std::cout << "Planes " << planes.vPlanes.size() << " \n";

}

/*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in the frame of reference of the omnidirectional camera
  */
void Frame360::segmentPlanesStereoRANSAC()
{
    // Segment planes
    //    std::cout << "extractPlaneFeatures, size " << sphereCloud->size() << "\n";
    double extractPlanes_start = pcl::getTime();

    pcl::PointCloud<PointT>::Ptr cloud_non_segmented (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud (*sphereCloud, *cloud_non_segmented);

    // Create the segmentation object
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients (true); // Optional
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.08);

    //        std::cout << "cloud_non_segmented " << cloud_non_segmented->size () << "\n";

    FilterPointCloud<PointT> filter(0.1);
    filter.filterVoxel(cloud_non_segmented);
    size_t min_cloud_segmentation = 0.2*cloud_non_segmented->size();

    // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
    unsigned single_cloud_size = sphereCloud->size();
    while (cloud_non_segmented->size() > min_cloud_segmentation )
    {
        std::cout << "cloud_non_segmented Pts " << cloud_non_segmented->size() << std::endl;

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

        seg.setInputCloud (cloud_non_segmented);
        seg.segment (*inliers, *coefficients);

        std::cout << "Inliers " << inliers->indices.size () << "\n";

        if (inliers->indices.size () < 1000)
            break;

        mrpt::pbmap::Plane plane;

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud ( cloud_non_segmented );
        extract.setIndices ( inliers );
        extract.setNegative (false);
        extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
        extract.setNegative (true);
        extract.filter (*cloud_non_segmented);    // Write the planar point cloud
        plane.inliers = inliers->indices; // TODO: only the first pass of inliers is good, the next ones need to be re-arranged
        double center_x=0, center_y=0, center_z=0;
        for(size_t j=0; j<plane.inliers.size(); j++)
        {
            center_x += plane.planePointCloudPtr->points[plane.inliers[j] ].x;
            center_y += plane.planePointCloudPtr->points[plane.inliers[j] ].y;
            center_z += plane.planePointCloudPtr->points[plane.inliers[j] ].z;
        }

        plane.v3center = Eigen::Vector3f(center_x/plane.inliers.size(), center_y/plane.inliers.size(), center_z/plane.inliers.size());
        plane.v3normal = Eigen::Vector3f(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if( plane.v3normal.dot(plane.v3center) > 0)
        {
            plane.v3normal = -plane.v3normal;
            //          plane.d = -plane.d;
        }
        plane.d = -plane.v3normal .dot( plane.v3center );

        //plane.curvature = regions[i].getCurvature ();
        //    cout << i << " getCurvature\n";


        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr contourPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);
        //            contourPtr->points = regions[i].getContour();

        //            //    cout << "Extract contour\n";
        static pcl::VoxelGrid<PointT> plane_grid;
        plane_grid.setLeafSize(0.05,0.05,0.05);
        plane_grid.setInputCloud (plane.planePointCloudPtr);
        plane_grid.filter (*contourPtr);
        plane.calcConvexHull(contourPtr);
        std::cout << "Inliers " << plane.planePointCloudPtr->size() << " hull " << plane.polygonContourPtr->size() << std::endl;

        //        assert(contourPtr->size() > 0);
        //        plane.calcConvexHull(contourPtr);
        //    cout << "calcConvexHull\n";
        plane.computeMassCenterAndArea();
        //    cout << "Extract convexHull\n";
        // Discard small planes
        if(plane.areaHull < min_area_plane)
            continue;

        plane.calcElongationAndPpalDir();
        // Discard narrow planes
        if(plane.elongation > max_elongation_plane)
            continue;

        //      double color_start = pcl::getTime();
        plane.calcPlaneHistH();
        plane.calcMainColor2();
        //          plane.calcMainColor();
        plane.id = planes.vPlanes.size();
        planes.vPlanes.push_back(plane);
    }
#if _DEBUG_MSG
    double extractPlanes_end = pcl::getTime();
    std::cout << "getPlanesRANSAC took " << (extractPlanes_end - extractPlanes_start)*1000 << " ms\n";
#endif
    std::cout << "Planes " << planes.vPlanes.size() << " \n";

}
