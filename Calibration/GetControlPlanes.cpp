/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR
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

#include <mrpt/utils/types_simple.h>
#include <mrpt/math/CMatrix.h>

#include <Calibrator.h>
#include <Frame360.h>
#include <Frame360_Visualizer.h>
//#include <RegisterRGBD360.h>

#include <pcl/console/parse.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define VISUALIZE_POINT_CLOUD 1
#define SAVE_CALIBRATION 1

using namespace std;
using namespace mrpt;
using namespace mrpt::math;
using namespace mrpt::utils;
using namespace Eigen;

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
    normalCovariances += n_i * n_ii.transpose();
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
//  rigidTransf.block(0,0,3,3) = Rotation;
//  rigidTransf.block(0,3,3,1) = translation;
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

  const double angleThreshold = distanceThreshold / 2;

  const size_t N = size(planeCorresp,2);
  out_inlierIndices.clear();
  out_inlierIndices.reserve(100);
  for (size_t i=0;i<N;i++)
  {
    const Eigen::Vector3f n_i = Eigen::Vector3f(planeCorresp(0,i), planeCorresp(1,i), planeCorresp(2,i));
    const Eigen::Vector3f n_ii = Rotation * Eigen::Vector3f(planeCorresp(4,i), planeCorresp(5,i), planeCorresp(6,i));
    const float d_error = fabs((planeCorresp(7,i) - translation.dot(n_i)) - planeCorresp(3,i));
    const float angle_error = (n_i .cross (n_ii )).norm();

    if (d_error < distanceThreshold)
     if (angle_error < angleThreshold)
      out_inlierIndices.push_back(i);
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
//cout << "degenerate " << useIndices[0] << " " << useIndices[1]  << " " << useIndices[2] << " - " << fabs(n_1. dot( n_2. cross(n_3) ) ) << endl;

  if( fabs(n_1. dot( n_2. cross(n_3) ) ) < 0.9 )
    return true;

  return false;
}

/*! This class is used to gather a set of control planes (they are analogous to the control points used to create panoramic images with a regular
    camera) from a sequence of RGBD360 observations. They permit to find the rigid transformations (extrinsic calibration) between the different
    Asus XPL sensors of the RGBD360 device.
*/
class GetControlPlanes
{
  private:

    mrpt::pbmap::PbMap planes;
    pcl::PointCloud<PointT>::Ptr sphere_cloud;

    unsigned match1, match2;

    char keyPressed;
    bool keyDown;
    bool drawMatch;

    pcl::visualization::CloudViewer viewer;
    boost::mutex visualizationMutex;

  public:

    GetControlPlanes() :
            sphere_cloud(new pcl::PointCloud<PointT>),
            viewer("RGBD360_calib")
    {
      keyPressed = 'b';
      keyDown = false;
      drawMatch = false;
    }

    void trimOutliersRANSAC(mrpt::math::CMatrixDouble &matched_planes)
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
                            0.2,
                            3,  // Minimum set of points
                            inliers,
                            best_model,
                            false,   // Verbose
                            0.99, // probGoodSample
                            5000 // maxIter
                            );

    //  cout << "Computation time: " << tictac.Tac()*1000.0/TIMES << " ms" << endl;

      cout << "Size planeCorresp: " << size(planeCorresp,2) << endl;
      cout << "RANSAC finished: " << inliers.size() << " from " << matched_planes.getRowCount() << ". \nBest model: \n" << best_model << endl;
    //        cout << "Best inliers: " << best_inliers << endl;

      mrpt::math::CMatrixDouble trimMatchedPlanes(inliers.size(), 10);
      std::vector<double> row;
      for(unsigned i=0; i < inliers.size(); i++)
        trimMatchedPlanes.row(i) = matched_planes.row(inliers[i]);

      matched_planes = trimMatchedPlanes;
    }

    void run(string &path_dataset, const int &selectSample)
    {
      Calib360 calib;
      calib.loadIntrinsicCalibration();
//      calib.loadExtrinsicCalibration();
      calib.loadExtrinsicCalibration(mrpt::format("%s/Calibration/Extrinsics/ConstructionSpecs", PROJECT_SOURCE_PATH));

//      // Initialize visualizer
//      Frame360_Visualizer visualizer;

      unsigned frame = 1;
      string fileName = mrpt::format("%s/sphere_images_%d.bin", path_dataset.c_str(), frame);
      cout << "fileName " << fileName << endl;

      #if VISUALIZE_POINT_CLOUD
//          pcl::visualization::CloudViewer viewer("RGBD360_calib");
        viewer.runOnVisualizationThread (boost::bind(&GetControlPlanes::viz_cb, this, _1), "viz_cb");
        viewer.registerKeyboardCallback(&GetControlPlanes::keyboardEventOccurred, *this);
      #endif

      Calibrator calibrator;
      ControlPlanes &matches = calibrator.matchedPlanes;
      for(unsigned sensor_id1=0; sensor_id1 < NUM_ASUS_SENSORS; sensor_id1++)
      {
        matches.mmCorrespondences[sensor_id1] = std::map<unsigned, mrpt::math::CMatrixDouble>();
        for(unsigned sensor_id2=sensor_id1+1; sensor_id2 < NUM_ASUS_SENSORS; sensor_id2++)
        {
          matches.mmCorrespondences[sensor_id1][sensor_id2] = mrpt::math::CMatrixDouble(0, 10);
        }
      }

      // Receive frame
//      while( frame < 520)
      while( fexists(fileName.c_str()) && keyPressed != 'a' )
      {
        // Grab frame
        Frame360 frame360(&calib);

//      cout << "Get new frame\n";
      cout << "fileName " << fileName << endl;
        frame360.loadFrame(fileName);
        frame360.undistort();
        frame += selectSample;
        fileName = mrpt::format("%s/sphere_images_%d.bin", path_dataset.c_str(), frame);

      cout << "Build point cloud\n";
        // Get spherical point cloud for visualization
        frame360.buildSphereCloud();

        #if VISUALIZE_POINT_CLOUD
          sphere_cloud = frame360.sphereCloud;
        #endif

      cout << "Segment planes\n";
        // Segment planes
        frame360.segmentPlanesLocal();

      cout << "Merge planes \n";
        planes.vPlanes.clear();
        vector<unsigned> planesSourceIdx(9, 0);
        for(int sensor_id = 0; sensor_id < 8; sensor_id++)
        {
          planes.MergeWith(frame360.local_planes_[sensor_id], calib.Rt_[sensor_id]);
          planesSourceIdx[sensor_id+1] = planesSourceIdx[sensor_id] + frame360.local_planes_[sensor_id].vPlanes.size();
//          cout << planesSourceIdx[sensor_id+1] << " ";
        }
//        cout << endl;

        for(unsigned sensor_id1=0; sensor_id1 < NUM_ASUS_SENSORS; sensor_id1++)
        {
//          matches.mmCorrespondences[sensor_id1] = std::map<unsigned, mrpt::math::CMatrixDouble>();
          for(unsigned sensor_id2=sensor_id1+1; sensor_id2 < NUM_ASUS_SENSORS; sensor_id2++)
//            if( sensor_id2 - sensor_id1 == 1 || sensor_id2 - sensor_id1 == 7)
          {
//            cout << " sensor_id1 " << sensor_id1 << " " << frame360.local_planes_[sensor_id1].vPlanes.size() << " sensor_id2 " << sensor_id2 << " " << frame360.local_planes_[sensor_id2].vPlanes.size() << endl;
//              matches.mmCorrespondences[sensor_id1][sensor_id2] = mrpt::math::CMatrixDouble(0, 10);

            for(unsigned i=0; i < frame360.local_planes_[sensor_id1].vPlanes.size(); i++)
            {
              unsigned planesIdx_i = planesSourceIdx[sensor_id1];
              for(unsigned j=0; j < frame360.local_planes_[sensor_id2].vPlanes.size(); j++)
              {
                unsigned planesIdx_j = planesSourceIdx[sensor_id2];

//                if(sensor_id1 == 0 && sensor_id2 == 2 && i == 0 && j == 0)
//                {
//                  cout << "Inliers " << planes.vPlanes[planesIdx_i+i].inliers.size() << " and " << planes.vPlanes[planesIdx_j+j].inliers.size() << endl;
//                  cout << "elongation " << planes.vPlanes[planesIdx_i+i].elongation << " and " << planes.vPlanes[planesIdx_j+j].elongation << endl;
//                  cout << "normal " << planes.vPlanes[planesIdx_i+i].v3normal.transpose() << " and " << planes.vPlanes[planesIdx_j+j].v3normal.transpose() << endl;
//                  cout << "d " << planes.vPlanes[planesIdx_i+i].d << " and " << planes.vPlanes[planesIdx_j+j].d << endl;
//                  cout << "color " << planes.vPlanes[planesIdx_i+i].hasSimilarDominantColor(planes.vPlanes[planesIdx_j+j],0.06) << endl;
//                  cout << "nearby " << planes.vPlanes[planesIdx_i+i].isPlaneNearby(planes.vPlanes[planesIdx_j+j], 0.5) << endl;
//                }

//                cout << "  Check planes " << planesIdx_i+i << " and " << planesIdx_j+j << endl;

                if( planes.vPlanes[planesIdx_i+i].inliers.size() > 1000 && planes.vPlanes[planesIdx_j+j].inliers.size() > 1000 &&
                    planes.vPlanes[planesIdx_i+i].elongation < 5 && planes.vPlanes[planesIdx_j+j].elongation < 5 &&
                    planes.vPlanes[planesIdx_i+i].v3normal .dot (planes.vPlanes[planesIdx_j+j].v3normal) > 0.99 &&
                    fabs(planes.vPlanes[planesIdx_i+i].d - planes.vPlanes[planesIdx_j+j].d) < 0.2 )//&&
//                    planes.vPlanes[planesIdx_i+i].hasSimilarDominantColor(planes.vPlanes[planesIdx_j+j],0.06) &&
//                    planes.vPlanes[planesIdx_i+i].isPlaneNearby(planes.vPlanes[planesIdx_j+j], 0.5) )
  //                      matches.inliersUpperFringe(planes.vPlanes[planesIdx_i+i], 0.2) > 0.2 &&
  //                      matches.inliersLowerFringe(planes.vPlanes[planesIdx_j+j], 0.2) > 0.2 ) // Assign correspondence
                  {
//                  cout << "\t   Associate planes " << planesIdx_i+i << " and " << planesIdx_j+j << endl;

                  #if VISUALIZE_POINT_CLOUD
                    // Visualize Control Planes
                    { boost::mutex::scoped_lock updateLock(visualizationMutex);
                      match1 = planesIdx_i+i;
                      match2 = planesIdx_j+j;
                      drawMatch = true;
                      keyDown = false;
                    updateLock.unlock();
                    }

//                    match1 = planesIdx_i+i;
//                    match2 = planesIdx_j+j;
//                    pcl::visualization::CloudViewer viewer("RGBD360_calib");
//                    viewer.runOnVisualizationThread (boost::bind(&GetControlPlanes::viz_cb, this, _1), "viz_cb");
//                    viewer.registerKeyboardCallback(&GetControlPlanes::keyboardEventOccurred, *this);

//                    cout << " keyDown " << keyDown << endl;
                    while(!keyDown)
                      boost::this_thread::sleep (boost::posix_time::milliseconds (10));
                  #endif

//                  cout << "\t   Record corresp " << endl;

                    unsigned prevSize = matches.mmCorrespondences[sensor_id1][sensor_id2].getRowCount();
                    matches.mmCorrespondences[sensor_id1][sensor_id2].setSize(prevSize+1, matches.mmCorrespondences[sensor_id1][sensor_id2].getColCount());
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 0) = frame360.local_planes_[sensor_id1].vPlanes[i].v3normal[0];
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 1) = frame360.local_planes_[sensor_id1].vPlanes[i].v3normal[1];
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 2) = frame360.local_planes_[sensor_id1].vPlanes[i].v3normal[2];
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 3) = frame360.local_planes_[sensor_id1].vPlanes[i].d;
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 4) = frame360.local_planes_[sensor_id2].vPlanes[j].v3normal[0];
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 5) = frame360.local_planes_[sensor_id2].vPlanes[j].v3normal[1];
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 6) = frame360.local_planes_[sensor_id2].vPlanes[j].v3normal[2];
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 7) = frame360.local_planes_[sensor_id2].vPlanes[j].d;
                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 8) = std::min(frame360.local_planes_[sensor_id1].vPlanes[i].inliers.size(), frame360.local_planes_[sensor_id2].vPlanes[j].inliers.size());

                    float dist_center1 = 0, dist_center2 = 0;
                    for(unsigned k=0; k < frame360.local_planes_[sensor_id1].vPlanes[i].inliers.size(); k++)
                      dist_center1 += frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] / frame360.frameRGBD_[sensor_id1].getPointCloud()->width + frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] % frame360.frameRGBD_[sensor_id1].getPointCloud()->width;
//                      dist_center1 += (frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] / frame360.sphereCloud->width)*(frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] / frame360.sphereCloud->width) + (frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] % frame360.sphereCloud->width)+(frame360.local_planes_[sensor_id1].vPlanes[i].inliers[k] % frame360.sphereCloud->width);
                    dist_center1 /= frame360.local_planes_[sensor_id1].vPlanes[i].inliers.size();

                    for(unsigned k=0; k < frame360.local_planes_[sensor_id2].vPlanes[j].inliers.size(); k++)
                      dist_center2 += frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] / frame360.frameRGBD_[sensor_id2].getPointCloud()->width + frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] % frame360.frameRGBD_[sensor_id2].getPointCloud()->width;
//                      dist_center2 += (frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] / frame360.sphereCloud->width)*(frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] / frame360.sphereCloud->width) + (frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] % frame360.sphereCloud->width)+(frame360.local_planes_[sensor_id2].vPlanes[j].inliers[k] % frame360.sphereCloud->width);
                    dist_center2 /= frame360.local_planes_[sensor_id2].vPlanes[j].inliers.size();

                    matches.mmCorrespondences[sensor_id1][sensor_id2](prevSize, 9) = std::max(dist_center1, dist_center2);
//                  cout << "\t Size " << matches.mmCorrespondences[sensor_id1][sensor_id2].getRowCount() << " x " << matches.mmCorrespondences[sensor_id1][sensor_id2].getColCount() << endl;

                    if( sensor_id2 - sensor_id1 == 1 ) // Calculate conditioning
                    {
//                      updateConditioning(couple_id, correspondences[couple_id].back());
                      matches.covariances[sensor_id1] += planes.vPlanes[planesIdx_i+i].v3normal * planes.vPlanes[planesIdx_j+j].v3normal.transpose();
                      matches.calcAdjacentConditioning(sensor_id1);
//                    cout << "Update " << sensor_id1 << endl;

  //                    // For visualization
  //                    plane_corresp[couple_id].push_back(pair<mrpt::pbmap::Plane*, mrpt::pbmap::Plane*>(&planes.vPlanes[planesIdx_i+i], &planes.vPlanes[planes_counter_j+j]));
                    }
                    else if(sensor_id2 - sensor_id1 == 7)
                    {
//                      updateConditioning(couple_id, correspondences[couple_id].back());
                      matches.covariances[sensor_id2] += planes.vPlanes[planesIdx_i+i].v3normal * planes.vPlanes[planesIdx_j+j].v3normal.transpose();
                      matches.calcAdjacentConditioning(sensor_id2);
//                    cout << "Update " << sensor_id2 << endl;
                    }

//                    break;
                  }
              }
            }
          }
        }

        #if VISUALIZE_POINT_CLOUD
          // Visualize Control Planes
          { boost::mutex::scoped_lock updateLock(visualizationMutex);
            drawMatch = false;
          updateLock.unlock();
          }
        #endif

        matches.printConditioning();

      } // End search of control planes

//      // Remove outliers with RANSAC
//      for(unsigned sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
//      {
//        if(sensor_id != 7)
//          trimOutliersRANSAC(matches.mmCorrespondences[sensor_id][sensor_id+1]);
//        else
//          trimOutliersRANSAC(matches.mmCorrespondences[0][7]);
//      }


      cout << "Conditioning " << *std::max_element(matches.conditioning, matches.conditioning+8) << " threshold " << calibrator.threshold_conditioning << endl;
      if(*std::max_element(matches.conditioning, matches.conditioning+8) < calibrator.threshold_conditioning)
      {
//          printConditioning();
      cout << "\tSave CorrespMat\n";
        // Save correspondence matrices
        matches.savePlaneCorrespondences(mrpt::format("%s/Calibration/ControlPlanes", PROJECT_SOURCE_PATH));

        calibrator.loadConstructionSpecs();
        calibrator.CalibrateRotation();
//        calibrator.CalibrateTranslation();

        // Save calibration matrices
        #if SAVE_CALIBRATION
        cout << "   SAVE_CALIBRATION \n";
        ofstream calibFile;
        for(unsigned sensor_id=0; sensor_id < 8; sensor_id++)
        {
          string calibFileName = mrpt::format("%s/Calibration/Rt_0%i.txt", PROJECT_SOURCE_PATH, sensor_id+1);
          calibFile.open(calibFileName.c_str());
          if (calibFile.is_open())
          {
            calibFile << calibrator.Rt_estimated[sensor_id];
            calibFile.close();
          }
          else
            cout << "Unable to open file " << calibFileName << endl;
        }
        #endif

        cout << "   CalibrateTranslation \n";
        calibrator.CalibrateTranslation();

      }

    }

  private:

    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
//    cout << "SphericalSequence::viz_cb(...)\n";
      if (sphere_cloud == NULL || sphere_cloud->empty() )
      {
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
        return;
      }
//    cout << "   ::viz_cb(...)\n";

      viz.removeAllShapes();
      viz.removeAllPointClouds();

      { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
        boost::mutex::scoped_lock updateLock(visualizationMutex);

        if (!viz.updatePointCloud (sphere_cloud, "sphereCloud"))
          viz.addPointCloud (sphere_cloud, "sphereCloud");

        // Draw matched planes
        char name[1024];
//
//        for(size_t i=0; i < planes.vPlanes.size(); i++)
        if(drawMatch)
        {
          mrpt::pbmap::Plane &plane_i = planes.vPlanes[match1];
          sprintf (name, "normal_%u", static_cast<unsigned>(plane_i.id));
          pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
          pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
          pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
                              plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
                              plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
          viz.addArrow (pt2, pt1, ared[0], agrn[0], ablu[0], false, name);

//          {
//            sprintf (name, "n%u", static_cast<unsigned>(i));
////            sprintf (name, "n%u_%u", static_cast<unsigned>(i), static_cast<unsigned>(plane_i.semanticGroup));
//            viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
//          }

          sprintf (name, "plane_%02u", static_cast<unsigned>(plane_i.id));
          pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[0], grn[0], blu[0]);
          viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, name);

          sprintf (name, "approx_plane_%02d", int (plane_i.id));
          viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[0], 0.5 * grn[0], 0.5 * blu[0], name);

          sprintf (name, "C ", int (plane_i.id));
          viz.addText (name, 20, 200, "params");
        }
        if(drawMatch)
        {
          mrpt::pbmap::Plane &plane_i = planes.vPlanes[match2];
          sprintf (name, "normal_%u", static_cast<unsigned>(plane_i.id));
          pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
          pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
          pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.3f * plane_i.v3normal[0]),
                              plane_i.v3center[1] + (0.3f * plane_i.v3normal[1]),
                              plane_i.v3center[2] + (0.3f * plane_i.v3normal[2]));
          viz.addArrow (pt2, pt1, ared[1], agrn[1], ablu[1], false, name);

//          {
//            sprintf (name, "n%u", static_cast<unsigned>(i));
////            sprintf (name, "n%u_%u", static_cast<unsigned>(i), static_cast<unsigned>(plane_i.semanticGroup));
//            viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
//          }

          sprintf (name, "plane_%02u", static_cast<unsigned>(plane_i.id));
          pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[1], grn[1], blu[1]);
          viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
          viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, name);

          sprintf (name, "approx_plane_%02d", int (plane_i.id));
          viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[1], 0.5 * grn[1], 0.5 * blu[1], name);
        }

      updateLock.unlock();
      }
    }

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
    {
      if ( event.keyDown () )
      {
        keyDown = true;
        keyPressed = event.getKeySym()[0];
//        if(event.getKeySym () == "k" || event.getKeySym () == "K")
//          bTakeKeyframe = true;
//        else if(event.getKeySym () == "l" || event.getKeySym () == "L")
//          bFreezeFrame = !bFreezeFrame;
      }
    }

};


void print_help(char ** argv)
{
  cout << "\nThis class is used to gather a set of control planes (they are analogous to the control points used to create panoramic images with a regular"
       << " camera) from a sequence of RGBD360 observations. They permit to find the rigid transformations (extrinsic calibration) between the different"
       << " Asus XPL sensors of the RGBD360 device\n";

  cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> <sampleStream> " << endl;
  cout << "    <pathToRawRGBDImagesDir> is the directory containing the data stream as a set of '.bin' files" << endl;
  cout << "    <sampleStream> is the sampling step used in the dataset (e.g. 1: all the frames are used " << endl;
  cout << "  - Press 'a' to stop grabbing control planes " << endl;

  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}


int main (int argc, char ** argv)
{
  if(argc != 3 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  string path_dataset = static_cast<string>(argv[1]);
  int selectSample = atoi(argv[2]);

  cout << "Create GetControlPlanes object\n";
  GetControlPlanes calib_rgbd360;
  calib_rgbd360.run(path_dataset, selectSample);

  cout << "EXIT\n";

  return (0);
}

