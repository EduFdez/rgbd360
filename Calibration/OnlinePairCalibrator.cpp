/*
 *  Copyright (c) 2013, Universidad de Málaga  - Grupo MAPIR
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
#include <Frame360_Visualizer.h>
#include <RegisterRGBD360.h>
#include <Calibrator.h>

#include <RGBDGrabber_OpenNI2.h>
#include <SerializeFrameRGBD.h> // For time-stamp conversion

#include <pcl/console/parse.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <signal.h>

#define USE_DEBUG_SEQUENCE 0
#define NUM_SENSORS 2
//const int NUM_SENSORS = 2;

#define SIGMA2 0.01

using namespace std;
using namespace mrpt::math;
using namespace Eigen;

#define RECORD_VIDEO 0
int numScreenshot = 0;

RGBDGrabber_OpenNI2 *grabber[NUM_SENSORS]; // This object is declared as global to be able to use it after capturing Ctrl-C interruptions

const int min_inliers = 1000;

/*! Catch interruptions like Ctrl-C */
void INThandler(int sig)
{
  char c;

  signal(sig, SIG_IGN);
  printf("\n  Do you really want to quit? [y/n] ");
  c = getchar();
  if (c == 'y' || c == 'Y')
  {
    for(unsigned sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
    {
      delete grabber[sensor_id]; // Turn off each Asus XPL sensor before exiting the program
    }
    exit(0);
  }
}

// Obtain the rigid transformation from 3 matched planes
CMatrixDouble getAlignment( const CMatrixDouble &matched_planes )
{
  assert(size(matched_planes,1) == 8 && size(matched_planes,2) == 3);

  //Calculate rotation
  Matrix3f normalCovariances = Matrix3f::Zero();
  normalCovariances(0,0) = 1;
  for(unsigned i=0; i<3; i++)
  {
    Vector3f n_i = Vector3f(matched_planes(0,i), matched_planes(1,i), matched_planes(2,i));
    Vector3f n_ii = Vector3f(matched_planes(4,i), matched_planes(5,i), matched_planes(6,i));
    normalCovariances += n_ii * n_i.transpose();
//    normalCovariances += matched_planes.block(i,0,1,3) * matched_planes.block(i,4,1,3).transpose();
  }

  JacobiSVD<MatrixXf> svd(normalCovariances, ComputeThinU | ComputeThinV);
  Matrix3f Rotation = svd.matrixV() * svd.matrixU().transpose();

//  float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
//  if(conditioning > 100)
//  {
//    cout << " ConsistencyTest::initPose -> Bad conditioning: " << conditioning << " -> Returning the identity\n";
//    return Eigen::Matrix4f::Identity();
//  }

  double det = Rotation.determinant();
  if(det != 1)
  {
    Eigen::Matrix3f aux;
    aux << 1, 0, 0, 0, 1, 0, 0, 0, det;
    Rotation = svd.matrixV() * aux * svd.matrixU().transpose();
  }


  // Calculate translation
  Vector3f translation;
  Matrix3f hessian = Matrix3f::Zero();
  Vector3f gradient = Vector3f::Zero();
  hessian(0,0) = 1;
  for(unsigned i=0; i<3; i++)
  {
    float trans_error = (matched_planes(3,i) - matched_planes(7,i)); //+n*t
//    hessian += matched_planes.block(i,0,1,3) * matched_planes.block(i,0,1,3).transpose();
//    gradient += matched_planes.block(i,0,1,3) * trans_error;
    Vector3f n_i = Vector3f(matched_planes(0,i), matched_planes(1,i), matched_planes(2,i));
    hessian += n_i * n_i.transpose();
    gradient += n_i * trans_error;
  }
  translation = -hessian.inverse() * gradient;
//cout << "Previous average translation error " << sumError / matched_planes.size() << endl;

//  // Form SE3 transformation matrix. This matrix maps the model into the current data reference frame
//  Eigen::Matrix4f rigidTransf;
//  rigidTransf.block(0,0,3,3) = Rotation;
//  rigidTransf.block(0,3,3,1) = translation;
//  rigidTransf.row(3) << 0,0,0,1;

  CMatrixDouble rigidTransf(4,4);
  rigidTransf(0,0) = Rotation(0,0);
  rigidTransf(0,1) = Rotation(0,1);
  rigidTransf(0,2) = Rotation(0,2);
  rigidTransf(1,0) = Rotation(1,0);
  rigidTransf(1,1) = Rotation(1,1);
  rigidTransf(1,2) = Rotation(1,2);
  rigidTransf(2,0) = Rotation(2,0);
  rigidTransf(2,1) = Rotation(2,1);
  rigidTransf(2,2) = Rotation(2,2);
  rigidTransf(0,3) = translation(0);
  rigidTransf(1,3) = translation(1);
  rigidTransf(2,3) = translation(2);
  rigidTransf(3,0) = 0;
  rigidTransf(3,1) = 0;
  rigidTransf(3,2) = 0;
  rigidTransf(3,3) = 1;

  return rigidTransf;
}

// Ransac functions to detect outliers in the plane matching
void ransacPlaneAlignment_fit(
        const CMatrixDouble &planeCorresp,
        const vector_size_t  &useIndices,
        vector< CMatrixDouble > &fitModels )
//        vector< Eigen::Matrix4f > &fitModels )
{
  ASSERT_(useIndices.size()==3);

  try
  {
    CMatrixDouble corresp(8,3);

//  cout << "Size planeCorresp: " << endl;
//  cout << "useIndices " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << endl;
    for(unsigned i=0; i<3; i++)
      corresp.col(i) = planeCorresp.col(useIndices[i]);

    fitModels.resize(1);
//    Eigen::Matrix4f &M = fitModels[0];
    CMatrixDouble &M = fitModels[0];
    M = getAlignment(corresp);
//  cout << "Ransac M\n" << M << endl;
  }
  catch(exception &)
  {
    fitModels.clear();
    return;
  }
}

void ransac3Dplane_distance(
        const CMatrixDouble &planeCorresp,
        const vector< CMatrixDouble > & testModels,
        const double distanceThreshold,
        unsigned int & out_bestModelIndex,
        vector_size_t & out_inlierIndices )
{
  ASSERT_( testModels.size()==1 )
  out_bestModelIndex = 0;
  const CMatrixDouble &M = testModels[0];

  Eigen::Matrix3f Rotation; Rotation << M(0,0), M(0,1), M(0,2), M(1,0), M(1,1), M(1,2), M(2,0), M(2,1), M(2,2);
  Eigen::Vector3f translation; translation << M(0,3), M(1,3), M(2,3);

	ASSERT_( size(M,1)==4 && size(M,2)==4 )

  const double angleThreshold = distanceThreshold / 3;

  const size_t N = size(planeCorresp,2);
  out_inlierIndices.clear();
  out_inlierIndices.reserve(100);
  for (size_t i=0;i<N;i++)
  {
    const Eigen::Vector3f n_i = Eigen::Vector3f(planeCorresp(0,i), planeCorresp(1,i), planeCorresp(2,i));
    const Eigen::Vector3f n_ii = Rotation * Eigen::Vector3f(planeCorresp(4,i), planeCorresp(5,i), planeCorresp(6,i));
    const float d_error = fabs((planeCorresp(7,i) - translation.dot(n_i)) - planeCorresp(3,i));
    const float angle_error = acos(n_i .dot (n_ii ));// * (180/PI);
//    const float angle_error = (n_i .cross (n_ii )).norm();
//    cout << "n_i " << n_i.transpose() << " n_ii " << n_ii.transpose() << "\n";
//    cout << "angle_error " << angle_error << " " << n_i .dot (n_ii ) << " d_error " << d_error << "\n";

    if (d_error < distanceThreshold)
     if (angle_error < 0.035){//cout << "new inlier\n";
      out_inlierIndices.push_back(i);}
  }
}

/** Return "true" if the selected points are a degenerate (invalid) case.
  */
bool ransac3Dplane_degenerate(
        const CMatrixDouble &planeCorresp,
        const mrpt::vector_size_t &useIndices )
{
  ASSERT_( useIndices.size()==3 )

  const Eigen::Vector3f n_1 = Eigen::Vector3f(planeCorresp(0,useIndices[0]), planeCorresp(1,useIndices[0]), planeCorresp(2,useIndices[0]));
  const Eigen::Vector3f n_2 = Eigen::Vector3f(planeCorresp(0,useIndices[1]), planeCorresp(1,useIndices[1]), planeCorresp(2,useIndices[1]));
  const Eigen::Vector3f n_3 = Eigen::Vector3f(planeCorresp(0,useIndices[2]), planeCorresp(1,useIndices[2]), planeCorresp(2,useIndices[2]));
//cout << "degenerate " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << " - " << n_1.transpose() << " v " << n_2.transpose() << " v " << n_3.transpose() << endl;

  Eigen::Matrix3f condTranslacion = n_1*n_1.transpose() + n_2*n_2.transpose() + n_3*n_3.transpose();
//cout << "degenerate " << condTranslacion.determinant() << endl;
  if( condTranslacion.determinant() < 0.01 )
//  if( fabs(n_1. dot( n_2. cross(n_3) ) ) < 0.9 )
    return true;

  return false;
}

/*! This class calibrates the extrinsic parameters of the omnidirectional RGB-D sensor. For that, the sensor is accessed
 *  and big planes are segmented and matched between different single sensors.
*/
class OnlinePairCalibration
{
  private:

    boost::mutex visualizationMutex;

    mrpt::pbmap::PbMap planes_i, planes_j;
    pcl::PointCloud<PointT>::Ptr cloud_i, cloud_j;
    Eigen::Matrix4f initOffset;

    bool bTakeSample;
    bool bDoCalibration;
    bool bFreezeFrame;
//    bool bTakeKeyframe;

    Calib360 calib;
//    Frame360 *frame_360;
//    pcl::PointCloud<PointT>::Ptr calibrated_cloud;
//    mrpt::pbmap::PbMap planes;

    Eigen::Matrix4f Rt_estimated[NUM_SENSORS];//, Rt_estimated_temp;
//    vector<mrpt::math::CMatrixDouble> correspondences;
    float weight_pair[NUM_SENSORS];
//    vector< vector< pair< Eigen::Vector4f, Eigen::Vector4f> > > correspondences;
    vector< pair< Eigen::Vector4f, Eigen::Vector4f> > correspondences;
//    vector< vector< pair< Eigen::Vector4f, Eigen::Vector4f> > > correspondences_2;
//    Eigen::Matrix3f covariances[NUM_SENSORS];
//    float conditioning[NUM_SENSORS];
//    unsigned conditioning_measCount[NUM_SENSORS];
    Eigen::Matrix3f covariance;
    float conditioning;
    unsigned conditioning_measCount;

    map<unsigned, unsigned> plane_corresp;

  public:
    OnlinePairCalibration() :
              cloud_i(new pcl::PointCloud<PointT>()),
              cloud_j(new pcl::PointCloud<PointT>()),
              bDoCalibration(false),
              bTakeSample(false),
              bFreezeFrame(false)
    {
//      correspondences.resize(8); // 8 pairs of RGBD sensors
//      correspondences_2.resize(8); // 8 pairs of RGBD sensors
//      std::fill(conditioning, conditioning+8, 9999.9);
//      std::fill(weight_pair, weight_pair+8, 0.0);
//      std::fill(conditioning_measCount, conditioning_measCount+8, 0);
//      std::fill(covariances, covariances+8, Eigen::Matrix3f::Zero());
//      calib.loadIntrinsicCalibration();
//      calib.loadExtrinsicCalibration();
      conditioning = 9999.9;
      conditioning_measCount = 0;
      covariance = Eigen::Matrix3f::Zero();
    }


  /*! This function segments planes from the point cloud */
    void segmentPlanesInFrame(CloudRGBD_Ext &cloudImg, mrpt::pbmap::PbMap &planes)
    {
      // Downsample and filter point cloud
//      DownsampleRGBD downsampler(2);
//      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampledCloud = downsampler.downsamplePointCloud(cloudImg.getPointCloud());
//      cloudImg.setPointCloud(downsampler.downsamplePointCloud(cloudImg.getPointCloud()));
      pcl::FastBilateralFilter<pcl::PointXYZRGBA> filter;
      filter.setSigmaS (10.0);
      filter.setSigmaR (0.05);
//      filter.setInputCloud(cloudImg.getDownsampledPointCloud(2));
      filter.setInputCloud(cloudImg.getPointCloud());
      filter.filter(*cloudImg.getPointCloud());

      // Segment planes
      std::cout << "extractPlaneFeatures, size " << cloudImg.getPointCloud()->size() << "\n";
      double extractPlanes_start = pcl::getTime();
    assert(cloudImg.getPointCloud()->height > 1 && cloudImg.getPointCloud()->width > 1);

      planes.vPlanes.clear();

      pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  //      ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);
  //      ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);
      ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
  //      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
      ne.setMaxDepthChangeFactor (0.02); // For VGA: 0.02f, 10.01
      ne.setNormalSmoothingSize (8.0f);
      ne.setDepthDependentSmoothing (true);

      pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
  //      mps.setMinInliers (std::max(uint32_t(40),cloudImg.getPointCloud()->height*2));
      mps.setMinInliers (min_inliers);
      mps.setAngularThreshold (0.039812); // (0.017453 * 2.0) // 3 degrees
      mps.setDistanceThreshold (0.02); //2cm
  //    cout << "PointCloud size " << cloudImg.getPointCloud()->size() << endl;

      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
      ne.setInputCloud ( cloudImg.getPointCloud() );
      ne.compute (*normal_cloud);

      mps.setInputNormals (normal_cloud);
      mps.setInputCloud ( cloudImg.getPointCloud() );
      std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
      std::vector<pcl::ModelCoefficients> model_coefficients;
      std::vector<pcl::PointIndices> inlier_indices;
      pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
      std::vector<pcl::PointIndices> label_indices;
      std::vector<pcl::PointIndices> boundary_indices;
      mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

      // Create a vector with the planes detected in this keyframe, and calculate their parameters (normal, center, pointclouds, etc.)
      unsigned single_cloud_size = cloudImg.getPointCloud()->size();
      for (size_t i = 0; i < regions.size (); i++)
      {
        mrpt::pbmap::Plane plane;

        plane.v3center = regions[i].getCentroid();
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
        extract.setInputCloud ( cloudImg.getPointCloud() );
        extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices[i]) );
        extract.setNegative (false);
        extract.filter (*plane.planePointCloudPtr);    // Write the planar point cloud
        plane.inliers = inlier_indices[i].indices;

        Eigen::Matrix3f cov_normal = Eigen::Matrix3f::Zero();
        Eigen::Vector3f cov_nd = Eigen::Vector3f::Zero();
        Eigen::Vector3f gravity_center = Eigen::Vector3f::Zero();
        for(size_t j=0; j < inlier_indices[i].indices.size(); j++)
        {
          Eigen::Vector3f pt; pt << plane.planePointCloudPtr->points[j].x, plane.planePointCloudPtr->points[j].y, plane.planePointCloudPtr->points[j].z;
          gravity_center += pt;
        }
        cov_nd = gravity_center;
        gravity_center /= plane.planePointCloudPtr->size();
//      cout << "gravity_center " << gravity_center.transpose() << "   " << plane.v3center.transpose() << endl;
//        Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
        for(size_t j=0; j < inlier_indices[i].indices.size(); j++)
        {
          Eigen::Vector3f pt; pt << plane.planePointCloudPtr->points[j].x, plane.planePointCloudPtr->points[j].y, plane.planePointCloudPtr->points[j].z;
          cov_normal += -pt*pt.transpose();// + (plane.v3normal.dot(pt-gravity_center))*(plane.v3normal.dot(pt))*Eigen::Matrix3f::Identity();
//          Eigen::Vector3f pt_rg = (pt-gravity_center);
//          M += pt_rg * pt_rg.transpose();
        }
//        Eigen::JacobiSVD<Eigen::Matrix3f> svdM(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
//      cout << "normalV " << plane.v3normal.transpose() << " covM \n" << svdM.matrixU() << endl;

        Eigen::Matrix4f fim;//, covariance;
        fim.block(0,0,3,3) = cov_normal;
        fim.block(0,3,3,1) = cov_nd;
        fim.block(3,0,1,3) = cov_nd.transpose();
        fim(3,3) = -plane.planePointCloudPtr->size();
        fim *= 1 / SIGMA2;
        Eigen::JacobiSVD<Eigen::Matrix4f> svd(fim, Eigen::ComputeFullU | Eigen::ComputeFullV);
        svd.pinv(plane.information);
//        std::cout << "covariance \n" << plane.information << std::endl;
        plane.information = -fim;
//
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr contourPtr(new pcl::PointCloud<pcl::PointXYZRGBA>);
        contourPtr->points = regions[i].getContour();
//        std::vector<size_t> indices_hull;

//      cout << "Extract contour\n";
        if(contourPtr->size() != 0)
        {
//      cout << "Extract contour2 " << contourPtr->size() << "\n";
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
//      cout << "calcConvexHull\n";
        plane.computeMassCenterAndArea();
  //    cout << "Extract convexHull\n";
        // Discard small planes
//        if(plane.areaHull < min_area_plane)
//          continue;

        plane.d = -plane.v3normal .dot( plane.v3center );

        plane.calcElongationAndPpalDir();
        // Discard narrow planes
//        if(plane.elongation > max_elongation_plane)
//          continue;

        double color_start = pcl::getTime();
        plane.calcPlaneHistH();
        plane.calcMainColor2();
        double color_end = pcl::getTime();
//      std::cout << "color in " << (color_end - color_start)*1000 << " ms\n";

  //      color_start = pcl::getTime();
//        plane.transform(initOffset);
  //      color_end = pcl::getTime();
  //    std::cout << "transform in " << (color_end - color_start)*1000 << " ms\n";

        bool isSamePlane = false;
        if(plane.curvature < max_curvature_plane)
          for (size_t j = 0; j < planes.vPlanes.size(); j++)
            if( planes.vPlanes[j].curvature < max_curvature_plane && planes.vPlanes[j].isSamePlane(plane, 0.99, 0.05, 0.2) ) // The planes are merged if they are the same
            {
//            cout << "Merge local region\n";
              isSamePlane = true;
  //            double time_start = pcl::getTime();
              planes.vPlanes[j].mergePlane2(plane);
  //            double time_end = pcl::getTime();
  //          std::cout << " mergePlane2 took " << double (time_start - time_end) << std::endl;

              break;
            }
        if(!isSamePlane)
        {
//          cout << "New plane\n";
  //          plane.calcMainColor();
          plane.id = planes.vPlanes.size();
          planes.vPlanes.push_back(plane);
        }
      }
        double extractPlanes_end = pcl::getTime();
      std::cout << "segmentPlanesInFrame in " << (extractPlanes_end - extractPlanes_start)*1000 << " ms\n";
    }


    void trimOutliersRANSAC(mrpt::math::CMatrixDouble &matched_planes, mrpt::math::CMatrixDouble &FIM_values)
    {
      cout << "trimOutliersRANSAC... " << endl;

    //  assert(matched_planes.size() >= 3);
    //  CTicTac tictac;

      if(matched_planes.getRowCount() <= 3)
      {
        cout << "Insuficient matched planes " << matched_planes.getRowCount() << endl;
//        return Eigen::Matrix4f::Identity();
        return;
      }

      CMatrixDouble planeCorresp(8, matched_planes.getRowCount());
      planeCorresp = matched_planes.block(0,0,matched_planes.getRowCount(),8).transpose();

      mrpt::vector_size_t inliers;
    //  Eigen::Matrix4f best_model;
      CMatrixDouble best_model;

      math::RANSAC::execute(planeCorresp,
                            ransacPlaneAlignment_fit,
                            ransac3Dplane_distance,
                            ransac3Dplane_degenerate,
                            0.05, // threshold
                            3,  // Minimum set of points
                            inliers,
                            best_model,
                            false,   // Verbose
                            0.51, // probGoodSample
                            1000 // maxIter
                            );

    //  cout << "Computation time: " << tictac.Tac()*1000.0/TIMES << " ms" << endl;

      cout << "Size planeCorresp: " << size(planeCorresp,2) << endl;
      cout << "RANSAC finished: " << inliers.size() << " from " << matched_planes.getRowCount() << ". \nBest model: \n" << best_model << endl;
    //        cout << "Best inliers: " << best_inliers << endl;

      mrpt::math::CMatrixDouble trimMatchedPlanes(inliers.size(), matched_planes.getColCount());
      mrpt::math::CMatrixDouble trimFIM_values(inliers.size(), FIM_values.getColCount());
      std::vector<double> row;
      for(unsigned i=0; i < inliers.size(); i++)
      {
        trimMatchedPlanes.row(i) = matched_planes.row(inliers[i]);
        trimFIM_values.row(i) = FIM_values.row(inliers[i]);
      }

      matched_planes = trimMatchedPlanes;
      FIM_values = trimFIM_values;
    }

    void run()
    {
        int rc = openni::OpenNI::initialize();
        printf("After initialization:\n %s\n", openni::OpenNI::getExtendedError());

        // Show devices list
        openni::Array<openni::DeviceInfo> deviceList;
        openni::OpenNI::enumerateDevices(&deviceList);
        printf("Get device list. %d devices connected\n", deviceList.getSize() );
        for (unsigned i=0; i < deviceList.getSize(); i++)
        {
          printf("Device %u: name=%s uri=%s vendor=%s \n", i+1 , deviceList[i].getName(), deviceList[i].getUri(), deviceList[i].getVendor());
        }
        if(deviceList.getSize() == 0)
        {
          cout << "No devices connected -> EXIT\n";
          return;
        }

    //    grabber->showDeviceList();
        int device_ref, device_trg; // = static_cast<int>(getchar());
        cout << "Choose the reference camera: ";
        cin >> device_ref;
        cout << "Choose the target camera: ";
        cin >> device_trg;

        cout << "Choose the resolution: 0 (640x480) or 1 (320x240): ";
        int resolution; // = static_cast<int>(getchar());
        cin >> resolution;
//        cout << "device_ref " << device_ref << " resolution " << resolution << endl;

        grabber[0] = new RGBDGrabber_OpenNI2(deviceList[device_ref-1].getUri(), resolution);
        grabber[1] = new RGBDGrabber_OpenNI2(deviceList[device_trg-1].getUri(), resolution);

        // Initialize the sensor
        for(int sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
  //      #pragma omp parallel num_threads(8)
        {
  //        int sensor_id = omp_get_thread_num();
          grabber[sensor_id]->init();
        }
        cout << "Grabber initialized\n";

        unsigned countMeas = 0;
        unsigned frame = 0;

        // Initialize visualizer
        pcl::visualization::CloudViewer viewer("PairCalibrator");
        viewer.runOnVisualizationThread (boost::bind(&OnlinePairCalibration::viz_cb, this, _1), "viz_cb");
        viewer.registerKeyboardCallback ( &OnlinePairCalibration::keyboardEventOccurred, *this );

        float angle_offset = 20;
//        float angle_offset = 180;
        initOffset = Eigen::Matrix4f::Identity();
        initOffset(1,1) = initOffset(2,2) = cos(angle_offset*3.14159/180);
        initOffset(1,2) = -sin(angle_offset*3.14159/180);
        initOffset(2,1) = -initOffset(1,2);

        PairCalibrator calibrator;
        // Get the plane correspondences
//        calibrator.correspondences = mrpt::math::CMatrixDouble(0,10);
        calibrator.correspondences = mrpt::math::CMatrixDouble(0,18);
        calibrator.setInitRt(initOffset);

        CMatrixDouble conditioningFIM(0,6);
        Eigen::Matrix3f FIMrot = Eigen::Matrix3f::Zero();
        Eigen::Matrix3f FIMtrans = Eigen::Matrix3f::Zero();

        while (!viewer.wasStopped() && !bDoCalibration)
//        while (true)
        {
cout << "run1\n";
          frame++;
          CloudRGBD_Ext frameRGBD_[NUM_SENSORS];
//          for(int sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
//            grabber[sensor_id]->grab(&frameRGBD_[sensor_id]);
          #pragma omp parallel num_threads(NUM_SENSORS)
          {
            int sensor_id = omp_get_thread_num();
            grabber[sensor_id]->grab(&frameRGBD_[sensor_id]);
          }

          if(frame < 2)
            continue;

//          if(bTakeSample == true)
//            continue;

          mrpt::pbmap::PbMap planes_j_orig;

          { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
          boost::mutex::scoped_lock updateLock(visualizationMutex);

//            cloud_i.reset(new pcl::PointCloud<PointT>());
            *cloud_i = *frameRGBD_[0].getPointCloud();
            frameRGBD_[0].getDownsampledPointCloud(2);
//            *cloud_i = *frameRGBD_[0].getDownsampledPointCloud(2);
//            pcl::transformPointCloud(*frameRGBD_[1].getDownsampledPointCloud(2), *cloud_j, calibrator.Rt_estimated);
            pcl::transformPointCloud(*frameRGBD_[1].getPointCloud(), *cloud_j, calibrator.Rt_estimated);
            frameRGBD_[1].getDownsampledPointCloud(2);

//          updateLock.unlock();
//          } // CS_visualize

            mrpt::pbmap::PbMap *planes_[2]; planes_[0] = &planes_i; planes_[1] = &planes_j;
            #pragma omp parallel num_threads(NUM_SENSORS)
            {
              int sensor_id = omp_get_thread_num();
              segmentPlanesInFrame(frameRGBD_[sensor_id], *planes_[sensor_id]);
            }

            planes_j_orig = planes_j;
            for(unsigned k=0; k < planes_j.vPlanes.size(); k++)
              planes_j.vPlanes[k].transform(calibrator.Rt_estimated);

            cout << "\tsegmentPlanesInFrame " << endl;

          updateLock.unlock();
          } // CS_visualize

          plane_corresp.clear();
          for(unsigned i=0; i < planes_i.vPlanes.size(); i++)
          {
            for(unsigned j=0; j < planes_j.vPlanes.size(); j++)
            {
              if( planes_i.vPlanes[i].inliers.size() > min_inliers && planes_j.vPlanes[j].inliers.size() > min_inliers &&
                  planes_i.vPlanes[i].elongation < 5 && planes_j.vPlanes[j].elongation < 5 &&
                  planes_i.vPlanes[i].v3normal .dot (planes_j.vPlanes[j].v3normal) > 0.99 &&
                  fabs(planes_i.vPlanes[i].d - planes_j.vPlanes[j].d) < 0.1 //&&
//                    planes_i.vPlanes[i].hasSimilarDominantColor(planes_j.vPlanes[j],0.06) &&
//                    planes_i.vPlanes[planes_counter_i+i].isPlaneNearby(planes_j.vPlanes[planes_counter_j+j], 0.5)
                )
              {
                bTakeSample = true;

              cout << "\tAssociate planes " << endl;
                Eigen::Vector4f pl1, pl2;
                pl1.head(3) = planes_i.vPlanes[i].v3normal; pl1[3] = planes_i.vPlanes[i].d;
                pl2.head(3) = planes_j_orig.vPlanes[j].v3normal; pl2[3] = planes_j_orig.vPlanes[j].d;
//              cout << "Corresp " << planes_i.vPlanes[i].v3normal.transpose() << " vs " << planes_j_orig.vPlanes[j].v3normal.transpose() << " = " << planes_j.vPlanes[j].v3normal.transpose() << endl;
////                        float factorDistInliers = std::min(planes_i.vPlanes[i].inliers.size(), planes_j.vPlanes[j].inliers.size()) / std::max(planes_i.vPlanes[i].v3center.norm(), planes_j.vPlanes[j].v3center.norm());
//                        float factorDistInliers = (planes_i.vPlanes[i].inliers.size() + planes_j.vPlanes[j].inliers.size()) / (planes_i.vPlanes[i].v3center.norm() * planes_j.vPlanes[j].v3center.norm());
//                        weight_pair[couple_id] += factorDistInliers;
//                        pl1 *= factorDistInliers;
//                        pl2 *= factorDistInliers;
//                ++weight_pair[couple_id];

                //Add constraints
//                  correspondences.push_back(pair<Eigen::Vector4f, Eigen::Vector4f>(pl1, pl2));
//                        correspondences[couple_id].push_back(pair<Eigen::Vector4f, Eigen::Vector4f>(pl1/planes_i.vPlanes[i].v3center.norm(), pl2/planes_j.vPlanes[j].v3center.norm());

                // Calculate conditioning
                ++conditioning_measCount;
                covariance += pl2.head(3) * pl1.head(3).transpose();
                Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
                conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
              cout << "conditioning " << conditioning << endl;

//          if(bTakeSample)

                unsigned prevSize = calibrator.correspondences.getRowCount();
                calibrator.correspondences.setSize(prevSize+1, calibrator.correspondences.getColCount());
                calibrator.correspondences(prevSize, 0) = pl1[0];
                calibrator.correspondences(prevSize, 1) = pl1[1];
                calibrator.correspondences(prevSize, 2) = pl1[2];
                calibrator.correspondences(prevSize, 3) = pl1[3];
                calibrator.correspondences(prevSize, 4) = pl2[0];
                calibrator.correspondences(prevSize, 5) = pl2[1];
                calibrator.correspondences(prevSize, 6) = pl2[2];
                calibrator.correspondences(prevSize, 7) = pl2[3];

                Eigen::Matrix4f informationFusion;
                Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
                tf.block(0,0,3,3) = calibrator.Rt_estimated.block(0,0,3,3);
                tf.block(3,0,1,3) = -calibrator.Rt_estimated.block(0,3,3,1).transpose();
                informationFusion = planes_i.vPlanes[i].information;
                informationFusion += tf * planes_j_orig.vPlanes[j].information * tf.inverse();
                Eigen::JacobiSVD<Eigen::Matrix4f> svd_cov(informationFusion, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Vector4f minEigenVector = svd_cov.matrixU().block(0,3,4,1);
              cout << "minEigenVector " << minEigenVector.transpose() << endl;
                informationFusion -= svd.singularValues().minCoeff() * minEigenVector * minEigenVector.transpose();
              cout << "informationFusion \n" << informationFusion << "\n minSV " << svd.singularValues().minCoeff() << endl;

                calibrator.correspondences(prevSize, 8) = informationFusion(0,0);
                calibrator.correspondences(prevSize, 9) = informationFusion(0,1);
                calibrator.correspondences(prevSize, 10) = informationFusion(0,2);
                calibrator.correspondences(prevSize, 11) = informationFusion(0,3);
                calibrator.correspondences(prevSize, 12) = informationFusion(1,1);
                calibrator.correspondences(prevSize, 13) = informationFusion(1,2);
                calibrator.correspondences(prevSize, 14) = informationFusion(1,3);
                calibrator.correspondences(prevSize, 15) = informationFusion(2,2);
                calibrator.correspondences(prevSize, 16) = informationFusion(2,3);
                calibrator.correspondences(prevSize, 17) = informationFusion(3,3);


              FIMrot += -skew(planes_j_orig.vPlanes[j].v3normal) * informationFusion.block(0,0,3,3) * skew(planes_j_orig.vPlanes[j].v3normal);
              FIMtrans += planes_i.vPlanes[i].v3normal * planes_i.vPlanes[i].v3normal.transpose() * informationFusion(3,3);

              Eigen::JacobiSVD<Eigen::Matrix3f> svd_rot(FIMrot, Eigen::ComputeFullU | Eigen::ComputeFullV);
              Eigen::JacobiSVD<Eigen::Matrix3f> svd_trans(FIMtrans, Eigen::ComputeFullU | Eigen::ComputeFullV);
//              float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
              conditioningFIM.setSize(prevSize+1, conditioningFIM.getColCount());
              conditioningFIM(prevSize, 0) = svd_rot.singularValues()[0];
              conditioningFIM(prevSize, 1) = svd_rot.singularValues()[1];
              conditioningFIM(prevSize, 2) = svd_rot.singularValues()[2];
              conditioningFIM(prevSize, 3) = svd_trans.singularValues()[0];
              conditioningFIM(prevSize, 4) = svd_trans.singularValues()[1];
              conditioningFIM(prevSize, 5) = svd_trans.singularValues()[2];

//              cout << "normalCovariance " << minEigenVector.transpose() << " covM \n" << svd_cov.matrixU() << endl;

//                calibrator.correspondences(prevSize, 8) = std::min(planes_i.vPlanes[i].inliers.size(), planes_j.vPlanes[j].inliers.size());
//
//                float dist_center1 = 0, dist_center2 = 0;
//                for(unsigned k=0; k < planes_i.vPlanes[i].inliers.size(); k++)
//                  dist_center1 += planes_i.vPlanes[i].inliers[k] / frameRGBD_[0].getPointCloud()->width + planes_i.vPlanes[i].inliers[k] % frameRGBD_[0].getPointCloud()->width;
////                      dist_center1 += (planes_i.vPlanes[i].inliers[k] / frame360.sphereCloud->width)*(planes_i.vPlanes[i].inliers[k] / frame360.sphereCloud->width) + (planes_i.vPlanes[i].inliers[k] % frame360.sphereCloud->width)+(planes_i.vPlanes[i].inliers[k] % frame360.sphereCloud->width);
//                dist_center1 /= planes_i.vPlanes[i].inliers.size();
//
//                for(unsigned k=0; k < planes_j.vPlanes[j].inliers.size(); k++)
//                  dist_center2 += planes_j.vPlanes[j].inliers[k] / frameRGBD_[0].getPointCloud()->width + planes_j.vPlanes[j].inliers[k] % frameRGBD_[0].getPointCloud()->width;
////                      dist_center2 += (planes_j.vPlanes[j].inliers[k] / frame360.sphereCloud->width)*(planes_j.vPlanes[j].inliers[k] / frame360.sphereCloud->width) + (planes_j.vPlanes[j].inliers[k] % frame360.sphereCloud->width)+(planes_j.vPlanes[j].inliers[k] % frame360.sphereCloud->width);
//                dist_center2 /= planes_j.vPlanes[j].inliers.size();
//                calibrator.correspondences(prevSize, 9) = std::max(dist_center1, dist_center2);

                { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
                boost::mutex::scoped_lock updateLock(visualizationMutex);

                  // For visualization
                  plane_corresp[i] = j;

                updateLock.unlock();
                } // CS_visualize

                break;
              }
            }
          }

//          calibrator.calcFisherInfMat();
          calibrator.CalibrateRotationManifold(1);
          calibrator.CalibrateTranslation(1);
//          calibrator.CalibrateRotation();

//          if(conditioning < 100 & conditioning_measCount > 3)
//            calibrator.CalibratePair();

//          if(conditioning < 50 & conditioning_measCount > 30)
//            bDoCalibration = true;

//cout << "run9\n";

//          while(bTakeSample == true)
//            boost::this_thread::sleep (boost::posix_time::milliseconds(10));
        }

        // Stop grabbing
        for(unsigned sensor_id = 0; sensor_id < NUM_SENSORS; sensor_id++)
          delete grabber[sensor_id];

        openni::OpenNI::shutdown();

        // Trim outliers
        trimOutliersRANSAC(calibrator.correspondences, conditioningFIM);


        float threshold_conditioning = 800.0;
        if(conditioning < threshold_conditioning)
        {
        cout << "\tSave CorrespMat\n";
          calibrator.correspondences.saveToTextFile( mrpt::format("%s/correspondences.txt", PROJECT_SOURCE_PATH) );
          conditioningFIM.saveToTextFile( mrpt::format("%s/conditioningFIM.txt", PROJECT_SOURCE_PATH) );

          calibrator.CalibratePair();

          calibrator.CalibrateRotationD();

          calibrator.setInitRt(initOffset);
          calibrator.CalibrateRotationManifold();
          calibrator.Rt_estimated.block(0,3,3,1) = calibrator.CalibrateTranslation();
          cout << "Rt_estimated \n" << calibrator.Rt_estimated << endl;
        }

    }

//    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
//    {
//      if ( event.keyDown () )
//      {
//        if(event.getKeySym () == "s" || event.getKeySym () == "S")
//          bDoCalibration = true;
//        else
//          bTakeSample = true;
//      }
//    }


    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
//    cout << "SphericalSequence::viz_cb(...)\n";
      if (cloud_i->empty() || bFreezeFrame)
      {
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
        return;
      }
//    cout << "   ::viz_cb(...)\n";

      viz.removeAllShapes();
      viz.removeAllPointClouds();

//      viz.setCameraPosition(0,0,-3,-1,0,0);
//      viz.setSize(640,480); // Set the window size
      viz.setSize(1280,960); // Set the window size
//      viz.setSize(800,800); // Set the window size
//      viz.setCameraPosition(0,0,-5,0,-0.707107,0.707107,1,0,0);

      { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
        boost::mutex::scoped_lock updateLock(visualizationMutex);

        if (!viz.updatePointCloud (cloud_i, "cloud_i"))
          viz.addPointCloud (cloud_i, "cloud_i");

        if (!viz.updatePointCloud (cloud_j, "sphereCloud"))
          viz.addPointCloud (cloud_j, "sphereCloud");

        // Draw camera system
        viz.removeCoordinateSystem();
        Eigen::Affine3f Rt;
        Rt.matrix() = initOffset;
        viz.addCoordinateSystem(0.05, Rt);
        viz.addCoordinateSystem(0.1, Eigen::Affine3f::Identity());

        char name[1024];

        sprintf (name, "%zu pts. Params ...", cloud_j->size());
        viz.addText (name, 20, 20, "params");

        // Draw planes
//        if(plane_corresp.size() > 0)
        {
//          bFreezeFrame = true;

          for(size_t i=0; i < planes_i.vPlanes.size(); i++)
          {
//            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
//              if(it->first == i)
//              {
            mrpt::pbmap::Plane &plane_i = planes_i.vPlanes[i];
            sprintf (name, "normal_%u", static_cast<unsigned>(i));
            pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
            pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
            pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
                                plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
                                plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
//            viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
            viz.addArrow (pt2, pt1, ared[0], agrn[0], ablu[0], false, name);

            sprintf (name, "approx_plane_%02d", int (i));
//            viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[5], 0.5 * grn[5], 0.5 * blu[5], name);
            viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[0], grn[0], blu[0], name);

//            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
//              if(it->first == i)
//              {
//                mrpt::pbmap::Plane &plane_i = planes_i.vPlanes[i];
//                sprintf (name, "normal_%u", static_cast<unsigned>(i));
//                pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
//                pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
//                pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
//                                    plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
//                                    plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
//                viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
//
//                sprintf (name, "approx_plane_%02d", int (i));
//                viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[0], grn[0], blu[0], name);
//
////                sprintf (name, "inliers_%02d", int (i));
////                pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[0], grn[0], blu[0]);
////                viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
//              }
          }
          for(size_t i=0; i < planes_j.vPlanes.size(); i++)
          {
//            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
//              if(it->second == i)
//              {
//    cout << "   planes_j " << i << "\n";

            mrpt::pbmap::Plane &plane_i = planes_j.vPlanes[i];
            sprintf (name, "normal_j_%u", static_cast<unsigned>(i));
            pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
            pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
            pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
                                plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
                                plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
//            viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
            viz.addArrow (pt2, pt1, ared[3], agrn[3], ablu[3], false, name);

            sprintf (name, "approx_plane_j_%02d", int (i));
//            viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[5], 0.5 * grn[5], 0.5 * blu[5], name);
            viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[3], grn[3], blu[3], name);

//            for(map<unsigned, unsigned>::iterator it=plane_corresp.begin(); it!=plane_corresp.end(); it++)
//              if(it->second == i)
//              {
//                mrpt::pbmap::Plane &plane_i = planes_j.vPlanes[i];
//                sprintf (name, "normal_j_%u", static_cast<unsigned>(i));
//                pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
//                pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
//                pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
//                                    plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
//                                    plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
//                viz.addArrow (pt2, pt1, ared[5], agrn[5], ablu[5], false, name);
//
//                sprintf (name, "approx_plane_j_%02d", int (i));
//                viz.addPolygon<PointT> (plane_i.polygonContourPtr, red[3], grn[3], blu[3], name);
//
////                sprintf (name, "inliers_j_%02d", int (i));
////                pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[3], grn[3], blu[3]);
////                viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
//              }
          }

          #if RECORD_VIDEO
            std::string screenshotFile = mrpt::format("im_%04u.png", ++numScreenshot);
            viz.saveScreenshot(screenshotFile);
          #endif
        }

      updateLock.unlock();
      }
    }

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
    {
      if ( event.keyDown () )
      {
        if(event.getKeySym () == "k" || event.getKeySym () == "K")
//          bTakeKeyframe = true;
          bDoCalibration = true;
        else if(event.getKeySym () == "l" || event.getKeySym () == "L"){
          bFreezeFrame = !bFreezeFrame;
          bTakeSample = !bTakeSample;
        }
      }
    }

};


void print_help(char ** argv)
{
  cout << "\nThis program calibrates the extrinsic parameters of the omnidirectional RGB-D device."
       << " This program opens the sensor, which has to be moved to take different"
       << " plane observations at different angles and distances. When enough information has been collected, a "
       << " Gauss-Newtown optimization is launched to obtain the extrinsic calibration. The user can decide whether to"
       << " save or not the new calibration after visualizing the calibrated image streaming from the sensor.\n";
  cout << "usage: " << argv[0] << " [options] \n";
  cout << argv[0] << " -h | --help : shows this help" << endl;
  cout << argv[0] << " -s | --save <pathToCalibrationFile>" << endl;
}


int main (int argc, char ** argv)
{
  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
    print_help(argv);

//  // Handle interruptions
//  signal(SIGINT, INThandler);

  cout << "Create OnlinePairCalibration object\n";
  OnlinePairCalibration calib_rgbd360;
  calib_rgbd360.run();

  cout << "EXIT\n";

  return (0);
}

