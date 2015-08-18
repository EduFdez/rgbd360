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

#include <Sphere3DFeat.h>

#ifdef _ENABLE_OPENMP
    #include <omp.h>
#endif

using namespace std;

/*! Save the PbMap from an omnidirectional RGB-D image */
void Sphere3DFeat::savePlanes(string pathPbMap)
{
    mrpt::utils::CFileGZOutputStream serialize_planesLabeled(pathPbMap);
    serialize_planesLabeled << planes;
    serialize_planesLabeled.close();
}

/*! Save the pointCloud and PbMap from an omnidirectional RGB-D image */
void Sphere3DFeat::save(string &path, unsigned &frame)
{
    assert(!sphereCloud->empty() && planes.vPlanes.size() > 0);

    string cloudPath = path + mrpt::format("/sphereCloud_%d.pcd",frame);
    pcl::io::savePCDFile(cloudPath, *sphereCloud);

    string pbmapPath = path + mrpt::format("/spherePlanes_%d.pbmap",frame);
    savePlanes(pbmapPath);
}

/*! Downsample and filter the individual point clouds from the different Asus XPL */
void Sphere3DFeat::buildCloudsDownsampleAndFilter()
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

/*! Create the PbMap of the spherical point cloud */
void Sphere3DFeat::segmentPlanes()
{
    cout << "Sphere3DFeat.segmentPlanes()\n";

#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

#if _ENABLE_OPENMP
    #pragma omp parallel num_threads(NUM_ASUS_SENSORS)
    {
        int sensor_id = omp_get_thread_num();
#else
    for(int sensor_id = 0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
    {
#endif
        segmentPlanesSensor(sensor_id);
    }

#if _PRINT_PROFILING
    double segmentation_end = pcl::getTime();
    cout << "Segmentation took " << double (segmentation_end - time_start)*1000 << " ms.\n";
#endif

    // Merge the big planes
    groupPlanes(); // Merge planes detected from adjacent sensors, and place them in "planes"
    mergePlanes(); // Merge big planes

#if _PRINT_PROFILING
    double extractPlanes_end = pcl::getTime();
    cout << planes.vPlanes.size() << " planes. Extraction took " << double (extractPlanes_end - time_start)*1000 << " ms.\n";
#endif

}

/*! Merge the planar patches that correspond to the same surface in the sphere */
void Sphere3DFeat::mergePlanes()
{
#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

    // Merge repeated planes
    for(size_t j = 0; j < planes.vPlanes.size(); j++) // numPrevPlanes
        if(planes.vPlanes[j].curvature < max_curvature_plane)
            for(size_t k = j+1; k < planes.vPlanes.size(); k++) // numPrevPlanes
                if(planes.vPlanes[k].curvature < max_curvature_plane)
                {
                    bool bSamePlane = false;
                    //        Eigen::Vector3f center_diff = planes.vPlanes[k].v3center - planes.vPlanes[j].v3center;
                    Eigen::Vector3f close_points_diff;
                    //float prev_dist = 1;
                    if( planes.vPlanes[j].v3normal.dot(planes.vPlanes[k].v3normal) > 0.99 )
                        if( fabs(planes.vPlanes[j].d - planes.vPlanes[k].d) < 0.45 )
                            //          if( BhattacharyyaDist_(plane1.hist_H, plane2.hist_H) > configLocaliser.hue_threshold )
                            //          if( fabs(planes.vPlanes[j].v3normal.dot(center_diff)) < max(0.07, 0.03*center_diff.norm() ) )
                        {
                            // Checking distances:
                            // a) Between an vertex and a vertex
                            // b) Between an edge and a vertex
                            // c) Between two edges (imagine two polygons on perpendicular planes)
                            for(unsigned i=1; i < planes.vPlanes[j].polygonContourPtr->size() && !bSamePlane; i++)
                                for(unsigned ii=1; ii < planes.vPlanes[k].polygonContourPtr->size(); ii++)
                                {
                                    close_points_diff = mrpt::pbmap::diffPoints(planes.vPlanes[j].polygonContourPtr->points[i], planes.vPlanes[k].polygonContourPtr->points[ii]);
                                    float dist = close_points_diff.norm();
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
                                        float dist = sqrt(mrpt::pbmap::dist3D_Segment_to_Segment2(mrpt::pbmap::Segment(planes.vPlanes[j].polygonContourPtr->points[i],planes.vPlanes[j].polygonContourPtr->points[i-1]), mrpt::pbmap::Segment(planes.vPlanes[k].polygonContourPtr->points[ii],planes.vPlanes[k].polygonContourPtr->points[ii-1])));
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
                        vector<mrpt::pbmap::Plane>::iterator itPlane = planes.vPlanes.begin();
                        for(size_t i = 0; i < k; i++)
                            itPlane++;
                        planes.vPlanes.erase(itPlane);

                        // Re-evaluate possible planes to merge
                        j--;
                        k = planes.vPlanes.size();
                    }
                }
#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "Merge planes took " << double (time_end - time_start) << endl;
#endif

}

/*! Group the planes segmented from each single sensor into the common PbMap 'planes' */
void Sphere3DFeat::groupPlanes()
{
    //  cout << "groupPlanes...\n";
    //double time_start = pcl::getTime();

    float maxDistHull = 0.5;
    float maxDistParallelHull = 0.09;

    //    Eigen::Matrix4f Rt = calib->getRt_id(0);
    //    planes.MergeWith(local_planes_[0], Rt);
    planes = local_planes_[0];
    set<unsigned> prev_planes, first_planes;
    for(size_t i=0; i < planes.vPlanes.size(); i++)
        first_planes.insert(planes.vPlanes[i].id);
    prev_planes = first_planes;

    for(unsigned sensor_id=1; sensor_id < NUM_ASUS_SENSORS; ++sensor_id)
    {
        size_t j;
        set<unsigned> next_prev_planes;
        for(size_t k = 0; k < local_planes_[sensor_id].vPlanes.size(); k++)
        {
            bool bSamePlane = false;
            if(local_planes_[sensor_id].vPlanes[k].areaHull > 0.5 || local_planes_[sensor_id].vPlanes[k].curvature < max_curvature_plane)
                for(set<unsigned>::iterator it = prev_planes.begin(); it != prev_planes.end() && !bSamePlane; it++) // numPrevPlanes
                {
                    j = *it;

                    if(planes.vPlanes[j].areaHull < 0.5 || planes.vPlanes[j].curvature > max_curvature_plane)
                        continue;

                    Eigen::Vector3f close_points_diff;
                    //float prev_dist = 1;
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
                                    float dist = close_points_diff.norm();
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
                                        float dist = sqrt(mrpt::pbmap::dist3D_Segment_to_Segment2(mrpt::pbmap::Segment(planes.vPlanes[j].polygonContourPtr->points[i],planes.vPlanes[j].polygonContourPtr->points[i-1]), mrpt::pbmap::Segment(local_planes_[sensor_id].vPlanes[k].polygonContourPtr->points[ii],local_planes_[sensor_id].vPlanes[k].polygonContourPtr->points[ii-1])));
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

// Functions for SphereStereo images (outdoors)
/*! Load a spherical RGB-D image from the raw data stored in a binary file */
void Sphere3DFeat::loadDepth (const string &binaryDepthFile, const cv::Mat * mask)
{
#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

    ifstream file (binaryDepthFile.c_str(), ios::in | ios::binary);
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
        //cout << "height_ " << height_ << " width_ " << width_ << endl;

        cv::Mat sphereDepth_aux(width_, height_, CV_32FC1);
        char *mem_block = reinterpret_cast<char*>(sphereDepth_aux.data);
        streampos size = height_*width_*4; // file.tellg() - streampos(4); // Header is 4 bytes: 2 unsigned short for height and width
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
        // sphereDepth.create(height_, width_, sphereDepth_aux.type());
        // cv::transpose(sphereDepth_aux, sphereDepth);
        //sphereDepth.create(640, width_, sphereDepth_aux.type());
        //cv::Rect region_of_interest = cv::Rect(8, 0, 640, width_); // Select only a portion of the image with height = 640 to facilitate the pyramid constructions
        sphereDepth.create(512, width_, sphereDepth_aux.type() );
        cv::Rect region_of_interest_transp = cv::Rect(90, 0, 512, width_); // Select only a portion of the image with height = width/4 (90 deg) with the FOV centered at the equator. This increases the performance of dense registration at the cost of losing some details from the upper/lower part of the images, which generally capture the sky and the floor.
        cv::transpose(sphereDepth_aux(region_of_interest_transp), sphereDepth); // The saved image is transposed wrt to the RGB img!

        //cv::imshow( "sphereDepth", sphereDepth );
        //cv::waitKey(0);

        if (mask && sphereDepth_aux.rows == mask->cols && sphereDepth_aux.cols == mask->rows){
            cv::Mat aux;
            cv::Rect region_of_interest = cv::Rect(0, 90, width_, 512); // This region of interest is the transposed of the above one (depth images are saved in disk as ColMajor)
            //cv::Rect region_of_interest = cv::Rect(0, 8, width_, 640);
            sphereDepth.copyTo(aux, (*mask)(region_of_interest) );
            sphereDepth = aux;
        }
        // cout << "height_ " << sphereDepth.rows << " width_ " << sphereDepth.cols << endl;
        //cv::imshow( "sphereDepth", sphereDepth );
        //cv::waitKey(0);
    }
    else
        cerr << "File: " << binaryDepthFile << " does NOT EXIST.\n";

    //    bSphereCloudBuilt = false; // The spherical PointCloud of the frame just loaded is not built yet

#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "loadDepth took " << double (time_end - time_start) << endl;
#endif
}

/*! Load a spherical RGB-D image from the raw data stored in a binary file */
void Sphere3DFeat::loadRGB(string &fileNamePNG)
{
#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

    ifstream file (fileNamePNG.c_str(), ios::in | ios::binary);
    if (file)
    {
        //            sphereRGB = cv::imread (fileNamePNG.c_str(), CV_LOAD_IMAGE_COLOR); // Full size 665x2048

        cv::Mat sphereRGB_aux = cv::imread (fileNamePNG.c_str(), CV_LOAD_IMAGE_COLOR);
        width_ = sphereRGB_aux.cols;
        sphereRGB.create(512, width_, sphereRGB_aux.type ());
        //sphereRGB.create(640, width_, sphereRGB_aux.type () );
        cv::Rect region_of_interest = cv::Rect(0,90, width_, 512); // Select only a portion of the image with height = width/4 (90 deg) with the FOV centered at the equator. This increases the performance of dense registration at the cost of losing some details from the upper/lower part of the images, which generally capture the sky and the floor.
        //cv::Rect region_of_interest = cv::Rect(0, 8, width_, 640); // Select only a portion of the image with height = 640 to facilitate the pyramid constructions
        sphereRGB = sphereRGB_aux (region_of_interest); // Size 640x2048
    }
    else
        cerr << "File: " << fileNamePNG << " does NOT EXIST.\n";

#if _PRINT_PROFILING
    double time_end = pcl::getTime();
    cout << "loadRGB took " << double (time_end - time_start) << endl;
#endif
}

/*! Perform bilateral filtering on the point cloud
    */
void Sphere3DFeat::filterCloudBilateral_stereo()
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


/*! This function segments planes from the point cloud
    */
void Sphere3DFeat::segmentPlanesStereo()
{
    // Segment planes
    //    cout << "extractPlaneFeatures, size " << sphereCloud->size() << "\n";

#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

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
    //      mps.setMinInliers (max(uint32_t(40),sphereCloud->height*2));
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
    vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    vector<pcl::ModelCoefficients> model_coefficients;
    vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    vector<pcl::PointIndices> label_indices;
    vector<pcl::PointIndices> boundary_indices;
    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

    // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
    //unsigned single_cloud_size = sphereCloud->size();
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
            cout << "HULL 000\n" << plane.planePointCloudPtr->size() << endl;
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
        //    cout << "color in " << (color_end - color_start)*1000 << " ms\n";

        //      color_start = pcl::getTime();
        //plane.transform(Rt);
        //      color_end = pcl::getTime();
        //    cout << "transform in " << (color_end - color_start)*1000 << " ms\n";

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
                    //          cout << " mergePlane2 took " << double (time_start - time_end) << endl;

                    break;
                }
        if(!isSamePlane)
        {
            //          plane.calcMainColor();
            plane.id = planes.vPlanes.size();
            planes.vPlanes.push_back(plane);
        }
    }
#if _PRINT_PROFILING
    double extractPlanes_end = pcl::getTime();
    cout << "segmentPlanesInFrame in " << (extractPlanes_end - time_start)*1000 << " ms\n";
#endif
    cout << "Planes " << planes.vPlanes.size() << " \n";

}

/*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in the frame of reference of the omnidirectional camera
  */
void Sphere3DFeat::segmentPlanesStereoRANSAC()
{
    // Segment planes
    //    cout << "extractPlaneFeatures, size " << sphereCloud->size() << "\n";

#if _PRINT_PROFILING
    double time_start = pcl::getTime();
#endif

    pcl::PointCloud<PointT>::Ptr cloud_non_segmented (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud (*sphereCloud, *cloud_non_segmented);

    // Create the segmentation object
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients (true); // Optional
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.08);

    //        cout << "cloud_non_segmented " << cloud_non_segmented->size () << "\n";

    FilterPointCloud<PointT> filter(0.1);
    filter.filterVoxel(cloud_non_segmented);
    size_t min_cloud_segmentation = 0.2*cloud_non_segmented->size();

    // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
    //unsigned single_cloud_size = sphereCloud->size();
    while (cloud_non_segmented->size() > min_cloud_segmentation )
    {
        cout << "cloud_non_segmented Pts " << cloud_non_segmented->size() << endl;

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

        seg.setInputCloud (cloud_non_segmented);
        seg.segment (*inliers, *coefficients);

        cout << "Inliers " << inliers->indices.size () << "\n";

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
        cout << "Inliers " << plane.planePointCloudPtr->size() << " hull " << plane.polygonContourPtr->size() << endl;

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
#if _PRINT_PROFILING
    double extractPlanes_end = pcl::getTime();
    cout << "segmentPlanesRANSAC took " << (extractPlanes_end - time_start)*1000 << " ms\n";
#endif
    cout << "Planes " << planes.vPlanes.size() << " \n";

}

/*! Compute the normalMap from an organized cloud of normal vectors. */
void Sphere3DFeat::computeNormalMap(const pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud, cv::Mat & normalMap, const bool display)
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
