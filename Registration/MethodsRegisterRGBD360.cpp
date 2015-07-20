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

#include <RegisterRGBD360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP
#include <pcl/registration/warp_point_rigid.h>

#include <mrpt/system/os.h>

#define VISUALIZE_POINT_CLOUD 1

using namespace std;
using namespace Eigen;

void print_help(char ** argv)
{
    cout << "\nThis program loads two raw omnidireactional RGB-D images and aligns them using PbMap-based registration.\n";
    cout << "usage: " << argv[0] << " [options] \n";
    cout << argv[0] << " -h | --help : shows this help" << endl;
    cout << argv[0] << " <frame360_1_1.bin> <frame360_1_2.bin>" << endl;
}

//pcl::PointCloud<PointT>::Ptr imgCloud;
//mrpt::pbmap::PbMap planes;

//void segmentPlanesLocalInFrame2()
//{
//    // Segment planes
//    //    cout << "extractPlaneFeatures\n";
//    //      double extractPlanes_start = pcl::getTime();

//    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
//    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
//    ne.setMaxDepthChangeFactor (0.02); // For VGA: 0.02f, 10.0f
//    ne.setNormalSmoothingSize (10.0f);
//    ne.setDepthDependentSmoothing (true);

//    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
//    mps.setMinInliers (200);
//    mps.setAngularThreshold (0.039812); // (0.017453 * 2.0) // 3 degrees
//    mps.setDistanceThreshold (0.02); //2cm

//    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
//    ne.setInputCloud ( imgCloud );
//    ne.compute (*normal_cloud);

//    mps.setInputNormals (normal_cloud);
//    mps.setInputCloud ( imgCloud );
//    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
//    std::vector<pcl::ModelCoefficients> model_coefficients;
//    std::vector<pcl::PointIndices> inlier_indices;
//    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
//    std::vector<pcl::PointIndices> label_indices;
//    std::vector<pcl::PointIndices> boundary_indices;
//    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);
//    //      mps.segment (model_coefficients, inlier_indices);
//    //    cout << regions.size() << " planes detected\n";

//    // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
//    for (size_t i = 0; i < regions.size (); i++)
//    {
//        //      std::cout << "curv " << regions[i].getCurvature() << std::endl;
//        if(regions[i].getCurvature() > max_curvature_plane)
//            continue;

//        mrpt::pbmap::Plane plane;

//        plane.v3center = regions[i].getCentroid ();
//        plane.v3normal = Eigen::Vector3f(model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);
//        plane.d = model_coefficients[i].values[3];
//        //        if( plane.v3normal.dot(plane.v3center) > 0)
//        if( model_coefficients[i].values[3] < 0)
//        {
//            plane.v3normal = -plane.v3normal;
//            plane.d = -plane.d;
//        }
//        plane.curvature = regions[i].getCurvature();
//        //      cout << "normal " << plane.v3normal.transpose() << " center " << regions[i].getCentroid().transpose() << " " << plane.v3center.transpose() << endl;
//        //    cout << "D " << -(plane.v3normal.dot(plane.v3center)) << " " << plane.d << endl;

//        // Extract the planar inliers from the input cloud
//        pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
//        extract.setInputCloud ( imgCloud );
//        //        extract.setInputCloud ( cloud );
//        extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices[i]) );
//        extract.setNegative (false);
//        extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
//        plane.inliers = inlier_indices[i].indices;
//        //    cout << "Extract inliers\n";

//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr contourPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);
//        contourPtr->points = regions[i].getContour();

//        plane.calcConvexHull(contourPtr);
//        plane.computeMassCenterAndArea();

//        plane.calcElongationAndPpalDir();

//        plane.calcPlaneHistH();

//        // Check whether this region correspond to the same plane as a previous one (this situation may happen when there exists a small discontinuity in the observation)
//        bool isSamePlane = false;
//        for (size_t j = 0; j < planes.vPlanes.size(); j++)
//            if( planes.vPlanes[j].isSamePlane(plane, 0.998, 0.1, 0.4) ) // The planes are merged if they are the same
//            {
//                //          cout << "Merge local region\n";
//                isSamePlane = true;
//                planes.vPlanes[j].mergePlane(plane);

//                break;
//            }
//        if(!isSamePlane)
//        {
//            //          plane.calcMainColor();
//            plane.id = planes.vPlanes.size();
//            planes.vPlanes.push_back(plane);
//        }
//    }
//    //      double extractPlanes_end = pcl::getTime();
//    //    std::cout << planes.vPlanes.size() << " planes. Extraction in " << sensor_id << " took " << double (extractPlanes_end - extractPlanes_start) << std::endl;

//    //      segmentation_im_[sensor_id] = true;
//}

///*! Visualization callback */
//void viz_cb(pcl::visualization::PCLVisualizer& viz)
//{

//  {
//    // Render the data
//    viz.removeAllPointClouds();
//    viz.removeAllShapes();
//    viz.setSize(800,600); // Set the window size

//    if (!viz.updatePointCloud (imgCloud, "sphereCloud"))
//      viz.addPointCloud (imgCloud, "sphereCloud");

//    char name[1024];

//    {
//      // Draw planes
//      for(size_t i=0; i < planes.vPlanes.size(); i++)
//      {
//        mrpt::pbmap::Plane &plane_i = planes.vPlanes[i];
//        sprintf (name, "normal_%u", static_cast<unsigned>(i));
//        pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
//        pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
//        pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
//                            plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
//                            plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
//        viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

//        {
//          sprintf (name, "n%u %s", static_cast<unsigned>(i), plane_i.label.c_str());
////            sprintf (name, "n%u %.1f %.2f", static_cast<unsigned>(i), plane_i.curvature*1000, plane_i.areaHull);
//          viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
//        }

//        sprintf (name, "approx_plane_%02d", int (i));
//        viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);

//        if(true)
//        {
//          sprintf (name, "plane_%02u", static_cast<unsigned>(i));
//          pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
//          viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
//          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);
//        }

//      }
//    }

//  }
//}

int main (int argc, char ** argv)
{
    if(argc != 3)
        print_help(argv);

    string file360_1 = static_cast<string>(argv[1]);
    string file360_2 = static_cast<string>(argv[2]);

    double time_start, time_end; // Measure performance of the different methods


    Calib360 calib;
    calib.loadExtrinsicCalibration();
    calib.loadIntrinsicCalibration();

    cout << "Create sphere 1\n";
    Frame360 frame360_1(&calib);
    frame360_1.loadFrame(file360_1);
    frame360_1.undistort();
    frame360_1.buildSphereCloud_rgbd360();

//    // Save images
//    {
//        int sensor_id = 5;
//        imgCloud = frame360_1.getCloud_id(sensor_id);
//        cv::Mat im_rgb = frame360_1.getFrameRGBD_id(sensor_id).getRGBImage();
//        cv::imwrite(mrpt::format("/home/edu/rgb_%01d.png",sensor_id), im_rgb);
//        segmentPlanesLocalInFrame2();

//        cv::Mat imgSegmentation = im_rgb.clone();
//        for(size_t i=0; i < planes.vPlanes.size(); i++)
//        {
//            mrpt::pbmap::Plane &plane_i = planes.vPlanes[i];
//            for(size_t j=0; j < plane_i.inliers.size(); j++)
//            {
//                int r = plane_i.inliers[j] / im_rgb.cols;
//                int c = plane_i.inliers[j] % im_rgb.cols;
//                imgSegmentation.at<cv::Vec3b>(r,c)[0] = blu[i%10];
//                imgSegmentation.at<cv::Vec3b>(r,c)[1] = grn[i%10];
//                imgSegmentation.at<cv::Vec3b>(r,c)[2] = red[i%10];
//            }
//        }

//        cv::imwrite(mrpt::format("/home/edu/segmented_%01d.png",sensor_id), imgSegmentation);

//        pcl::visualization::CloudViewer viewer("viz");
//        viewer.runOnVisualizationThread(viz_cb, "viz_cb");

//        mrpt::system::pause();
//    }


    cout << "Create sphere 2\n";
    Frame360 frame360_2(&calib);
    frame360_2.loadFrame(file360_2);
    frame360_2.undistort();
    frame360_2.buildSphereCloud_rgbd360();

    time_start = pcl::getTime();
    RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));
    frame360_1.segmentPlanes();
    frame360_2.segmentPlanes();
    registerer.RegisterPbMap(&frame360_1, &frame360_2, 25, RegisterRGBD360::PLANAR_3DoF);
    //  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::DEFAULT_6DoF);
    //  registerer.RegisterPbMap(&frame360_1, &frame360_2, 20, RegisterRGBD360::ODOMETRY_6DoF);
    Eigen::Matrix4f relPosePbMap = registerer.getPose();
    time_end = pcl::getTime();
    std::cout << "PbMap registration took " << double (time_end - time_start) << std::endl;

#if _DEBUG_MSG
    std::map<unsigned, unsigned> bestMatch = registerer.getMatchedPlanes();
    //  std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << " areaMatched " << registerer.getAreaMatched() << std::endl;
    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
        std::cout << it->first << " " << it->second << std::endl;

    cout << "Distance " << relPosePbMap.block(0,3,3,1).norm() << endl;
    cout << "PosePbMap \n" << relPosePbMap << endl;
#endif


      // ICP alignement
      double time_start_icp = pcl::getTime();
      pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;

      icp.setMaxCorrespondenceDistance (0.3);
      icp.setMaximumIterations (10);
      icp.setTransformationEpsilon (1e-6);
    //  icp.setEuclideanFitnessEpsilon (1);
      icp.setRANSACOutlierRejectionThreshold (0.1);

      // ICP
      // Filter the point clouds and remove nan points
      FilterPointCloud<PointT> filter(0.1);
      filter.filterVoxel(frame360_1.sphereCloud);
      filter.filterVoxel(frame360_2.sphereCloud);

      icp.setInputSource(frame360_2.sphereCloud);
      icp.setInputTarget(frame360_1.sphereCloud);
      pcl::PointCloud<PointT>::Ptr alignedICP(new pcl::PointCloud<PointT>);
      Matrix4f initRigidTransf = relPosePbMap;
//      Matrix4f initRigidTransf = Matrix4f::Identity();
      icp.align(*alignedICP, initRigidTransf);

      double time_end_icp = pcl::getTime();
      std::cout << "ICP took " << double (time_end_icp - time_start_icp) << std::endl;

//      std::cout << "has converged:" << icp.hasConverged() << " iterations " << icp.countIterations() << " score: " << icp.getFitnessScore() << std::endl;
      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      Matrix4f icpTransformation = icp.getFinalTransformation();
      cout << "ICP transformation:\n" << icpTransformation << endl;// << "PbMap-Registration\n" << registerer.getPose() << endl;


    // Test: align one image
    int sensorID = 0;
    const float img_width = frame360_1.frameRGBD_[sensorID].getRGBImage().cols;
    const float img_height = frame360_1.frameRGBD_[sensorID].getRGBImage().rows;
    const float res_factor_VGA = img_width / 640.0;
    const float focal_length = 525 * res_factor_VGA;
    const float inv_fx = 1.f/focal_length;
    const float inv_fy = 1.f/focal_length;
    const float ox = img_width/2 - 0.5;
    const float oy = img_height/2 - 0.5;
    Eigen::Matrix3f camIntrinsicMat; camIntrinsicMat << focal_length, 0, ox, 0, focal_length, oy, 0, 0, 1;
    DirectRegistration alignSingleFrames;
    alignSingleFrames.setCameraMatrix(camIntrinsicMat);

    std::cout << "\n\n SENSOR " << sensorID << std::endl;
    Eigen::Matrix4f sensorID_relPosePbMap = frame360_1.calib->Rt_[sensorID].inverse()*relPosePbMap*frame360_1.calib->Rt_[sensorID];
//    cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << endl;

    time_start = pcl::getTime();
    alignSingleFrames.setTargetFrame(frame360_1.frameRGBD_[sensorID].getRGBImage(), frame360_1.frameRGBD_[sensorID].getDepthImage());
    alignSingleFrames.setSourceFrame(frame360_2.frameRGBD_[sensorID].getRGBImage(), frame360_2.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.setVisualization(true);
    double time_start1 = pcl::getTime();
//    alignSingleFrames.alignFrames(Eigen::Matrix4f::Identity(), DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / PHOTO_DEPTH / DEPTH_CONSISTENCY
    alignSingleFrames.alignFrames(sensorID_relPosePbMap, DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / PHOTO_DEPTH / DEPTH_CONSISTENCY

    Eigen::Matrix4f sensorID_dense_relPose = alignSingleFrames.getOptimalPose();
    time_end = pcl::getTime();
    std::cout << "Dense alignment Single Frame took " << double (time_end - time_start) << std::endl;
//    std::cout << "alignment took " << double (time_end - time_start1) << std::endl;
//    std::cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << "\nsensorID_dense_relPose:\n" << sensorID_dense_relPose << std::endl;

//    // Use saliency
//    time_start = pcl::getTime();
//    alignSingleFrames.useSaliency(true);
//    alignSingleFrames.setTargetFrame(frame360_1.frameRGBD_[sensorID].getRGBImage(), frame360_1.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.setSourceFrame(frame360_2.frameRGBD_[sensorID].getRGBImage(), frame360_2.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.alignFrames(sensorID_relPosePbMap, DirectRegistration::DEPTH_CONSISTENCY); // PHOTO_CONSISTENCY
//    time_end = pcl::getTime();
//    std::cout << "Dense alignment Single Frame took " << double (time_end - time_start) << std::endl;
//    std::cout << "saliency:\n" << alignSingleFrames.getOptimalPose() << "\nsensorID_dense_relPose:\n" << sensorID_dense_relPose << std::endl;

    Eigen::Matrix4f dense_relPose = frame360_1.calib->Rt_[sensorID]*sensorID_dense_relPose*frame360_1.calib->Rt_[sensorID].inverse();
    std::cout << "dense_relPose:\n" << dense_relPose << std::endl;

//    // ICP preparation time
//    double time_start_icp = pcl::getTime();
//    pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>);
//    pcl::transformPointCloud(*frame360_2.getFrameRGBD_id(sensorID).getPointCloud(), *transformedCloud, sensorID_dense_relPose);
//    FilterPointCloud<PointT> filter(0.05);
//    filter.filterVoxel(transformedCloud);
//    double time_end_icp = pcl::getTime();
//    std::cout << "ICP preparation " << (time_end_icp - time_start_icp) << std::endl;

//    std::cout << "\n\n SENSOR " << sensorID << std::endl;
//    sensorID = 7;
//    sensorID_relPosePbMap = frame360_1.calib->Rt_[sensorID].inverse()*relPosePbMap*frame360_1.calib->Rt_[sensorID];
////    Eigen::Matrix4f sensorID_relPosePbMap = frame360_1.calib->Rt_[sensorID]*relPosePbMap.inverse()*frame360_1.calib->Rt_[sensorID].inverse();
//    cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << endl;

//    time_start = pcl::getTime();
//    alignSingleFrames.setTargetFrame(frame360_1.frameRGBD_[sensorID].getRGBImage(), frame360_1.frameRGBD_[sensorID].getDepthImage());
//    alignSingleFrames.setSourceFrame(frame360_2.frameRGBD_[sensorID].getRGBImage(), frame360_2.frameRGBD_[sensorID].getDepthImage());
////    alignSingleFrames.setVisualization(true);
//    alignSingleFrames.alignFrames(sensorID_relPosePbMap, DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / PHOTO_DEPTH / DEPTH_CONSISTENCY

//    sensorID_dense_relPose = alignSingleFrames.getOptimalPose();
//    time_end = pcl::getTime();
//    std::cout << "Dense alignment Single Frame took " << double (time_end - time_start) << std::endl;
//    std::cout << "sensorID_relPosePbMap:\n" << sensorID_relPosePbMap << "\nsensorID_dense_relPose:\n" << sensorID_dense_relPose << std::endl;

//    Eigen::Matrix4f dense_relPose2 = frame360_1.calib->Rt_[sensorID]*sensorID_dense_relPose*frame360_1.calib->Rt_[sensorID].inverse();
//    std::cout << "dense_relPose:\n" << dense_relPose2 << std::endl;



//    // Dense Photo/Depth consistency
//    time_start = pcl::getTime();
//    registerer.DenseRegistration(&frame360_1, &frame360_2, relPosePbMap, DirectRegistration::PHOTO_CONSISTENCY, RegisterRGBD360::DEFAULT_6DoF);
////    registerer.DenseRegistration_2(&frame360_1, &frame360_2, relPosePbMap, DirectRegistration::PHOTO_CONSISTENCY, RegisterRGBD360::DEFAULT_6DoF);
//    Matrix4f relPoseDense = registerer.getPose();
//    time_end = pcl::getTime();
//    std::cout << "Odometry dense alignment 360 took " << double (time_end - time_start) << std::endl;
//    cout << "DensePhotoICP transformation:\n" << relPoseDense << endl;
////mrpt::system::pause();


    // Dense Photo/Depth consistency spherical
    time_start = pcl::getTime();
    DirectRegistration align360;
    align360.setSensorType( ProjectionModel::RGBD360_INDOOR); // This is use to adapt some features/hacks for each type of image (see the implementation of DirectRegistration::register360 for more details)
    align360.setNumPyr(4);
    frame360_1.stitchSphericalImage();
    frame360_2.stitchSphericalImage();
    align360.useSaliency(false);
    align360.setTargetFrame(frame360_1.sphereRGB, frame360_1.sphereDepth);
    align360.setSourceFrame(frame360_2.sphereRGB, frame360_2.sphereDepth);

//    cv::Mat rgb_transposed, rgb_rotated, depth_transposed, depth_rotated;
//    cv::transpose(frame360_1.sphereRGB, rgb_transposed);
//    cv::flip(rgb_transposed, rgb_rotated, 0);
//    cv::transpose(frame360_1.sphereDepth, depth_transposed);
//    cv::flip(depth_transposed, depth_rotated, 0);

//    cv::imshow("rgb_rotated", rgb_rotated);
//    cv::waitKey(0);

//    align360.setTargetFrame(depth_rotated, depth_transposed);
//    cv::transpose(frame360_2.sphereRGB, rgb_transposed);
//    cv::flip(rgb_transposed, rgb_rotated, 0);
//    cv::transpose(frame360_2.sphereDepth, depth_transposed);
//    cv::flip(depth_transposed, depth_rotated, 0);
//    align360.setSourceFrame(depth_rotated, depth_transposed);

    // The reference of the spherical image and the point Clouds are not the same! I should always use the same coordinate system (TODO)
    float angleOffset = 157.5;
    Matrix4f rotOffset = Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);
    Matrix4f initDenseMatching = rotOffset * relPosePbMap * rotOffset.inverse();
//    cout << "initDenseMatching \n" << initDenseMatching << endl;

    time_start = pcl::getTime();
//    align360.setVisualization(true);
//    align360.setGrayVariance(4.f/255);
    align360.register360(initDenseMatching, DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//    align360.register360(relPosePbMap, DirectRegistration::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//    align360.register360(Matrix4f::Identity(), DirectRegistration::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    Matrix4f relPoseDenseSphere = align360.getOptimalPose();
    Matrix4f relPoseDenseSphere_ref = rotOffset.inverse() * relPoseDenseSphere * rotOffset;
    time_end = pcl::getTime();
    std::cout << "Spherical dense alignment took " << double (time_end - time_start) << std::endl;
    cout << "relPoseDenseSphere: " << align360.avResidual << " " << align360.avPhotoResidual << " " << align360.avDepthResidual << "\n" << relPoseDenseSphere_ref << endl;
    //cout << "relPoseDenseSphere_ref: " << align360.avPhotoResidual << " " << align360.avDepthResidual << "\n" << relPoseDenseSphere_ref << endl;

    time_start = pcl::getTime();
    align360.register360(initDenseMatching, DirectRegistration::PHOTO_DEPTH, 1); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    relPoseDenseSphere = align360.getOptimalPose();
    relPoseDenseSphere_ref = rotOffset.inverse() * relPoseDenseSphere * rotOffset;
    time_end = pcl::getTime();
    std::cout << "Spherical dense alignment took " << double (time_end - time_start) << std::endl;
    cout << "relPoseDenseSphere: " << align360.avResidual << " " << align360.avPhotoResidual << " " << align360.avDepthResidual << "\n" << relPoseDenseSphere_ref << endl;

    time_start = pcl::getTime();
    align360.setVisualization(true);
    align360.register360(initDenseMatching, DirectRegistration::PHOTO_DEPTH, 2); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
    relPoseDenseSphere = align360.getOptimalPose();
    relPoseDenseSphere_ref = rotOffset.inverse() * relPoseDenseSphere * rotOffset;
    time_end = pcl::getTime();
    std::cout << "Spherical dense alignment took " << double (time_end - time_start) << std::endl;
    cout << "relPoseDenseSphere: " << align360.avResidual << " " << align360.avPhotoResidual << " " << align360.avDepthResidual << "\n" << relPoseDenseSphere_ref << endl;


    //    cv::imwrite( mrpt::format("/home/edu/rgb.png"), frame360_1.sphereRGB);
    //    cv::imwrite( mrpt::format("/home/edu/depth.png"), frame360_1.sphereDepth);
    //    mrpt::system::pause();


    // Visualize
#if VISUALIZE_POINT_CLOUD
    // It runs PCL viewer in a different thread.
    //    cout << "Superimpose cloud\n";
    Map360 Map;
    Matrix4f pose = Matrix4f::Identity();
//    frame360_1.buildSphereCloud_fromImage();
//    frame360_2.buildSphereCloud_fromImage();
    Map.addKeyframe(&frame360_1, pose );
    Map.addKeyframe(&frame360_2, relPosePbMap );
    Map.vOptimizedPoses = Map.vTrajectoryPoses;
    //  Map.vOptimizedPoses[1] = icpTransformation;
//    Map.vOptimizedPoses[1] = dense_relPose;
//    Map.vOptimizedPoses[1] = relPoseDense;
    Map.vOptimizedPoses[1] = relPoseDenseSphere_ref;

    Map360_Visualizer Viewer(Map, 1);
    while (!Viewer.viewer.wasStopped() )
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
#endif

    cout << "EXIT\n";

    return (0);
}

