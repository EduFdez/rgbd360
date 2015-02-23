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

#ifndef MISCELLANEOUS_H
#define MISCELLANEOUS_H

#include <mrpt/pbmap.h>
#include <mrpt/utils/CStream.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <fstream>

#ifndef PI
    #define PI 3.14159265359
#endif

/*! This file contains a set of generic attributes used along the 'RGBD360' project.
 */

/*! Maximum number of planes to match when registering a pair of Spheres */
static float max_match_planes = 25;

/*! Maximum curvature to consider the region as planar */
static float max_curvature_plane = 0.0013;

/*! Minimum area to consider the planar patch */
static float min_area_plane = 0.12;

/*! Maximum elongation to consider the planar patch */
static float max_elongation_plane = 6;

/*! Minimum number of matched planes to consider a good registration */
static float min_planes_registration = 4;

/*! Minimum distance between keyframes */
static float min_dist_keyframes = 0.2;

/*! Maximum distance between two consecutive frames of a RGBD360 video sequence */
static float max_translation_odometry = 1.8;

/*! Maximum rotation between two consecutive frames of a RGBD360 video sequence */
static float max_rotation_odometry = 1.2;

/*! Maximum conditioning to resolve the calibration equation system. This parameter
    represent the ratio between the maximum and the minimum eigenvalue of the system */
static float threshold_conditioning = 8000.0;

static unsigned char red [10] = {255,   0,   0, 255, 255,   0, 255, 204,   0, 255};
static unsigned char grn [10] = {  0, 255,   0, 255,   0, 255, 160,  51, 128, 222};
static unsigned char blu [10] = {  0,   0, 255,   0, 255, 255, 0  , 204,   0, 173};

static double ared [10] = {1.0,   0,   0, 1.0, 1.0,   0, 1.0, 0.8,   0, 1.0};
static double agrn [10] = {  0, 1.0,   0, 1.0,   0, 1.0, 0.6, 0.2, 0.5, 0.9};
static double ablu [10] = {  0,   0, 1.0,   0, 1.0, 1.0,   0, 0.8,   0, 0.7};

/*! Generate a skew-symmetric matrix from a 3D vector */
template<typename dataType> inline
Eigen::Matrix<dataType,3,3> skew(const Eigen::Matrix<dataType,3,1> vec)
{
  Eigen::Matrix<dataType,3,3> skew_matrix = Eigen::Matrix<dataType,3,3>::Zero();
  skew_matrix(0,1) = -vec(2);
  skew_matrix(1,0) = vec(2);
  skew_matrix(0,2) = vec(1);
  skew_matrix(2,0) = -vec(1);
  skew_matrix(1,2) = -vec(0);
  skew_matrix(2,1) = vec(0);
  return skew_matrix;
}

/*! Return the translation vector of the input pose */
template<typename dataType> inline
Eigen::Matrix<dataType,3,1> getPoseTranslation(Eigen::Matrix<dataType,4,4> pose)
{
  Eigen::Matrix<dataType,3,1> translation = pose.block(0,3,3,1);
  return  translation;
}

/*! Return the rotation vector of the input pose */
template<typename dataType> inline
Eigen::Matrix<dataType,3,1> getPoseRotation(Eigen::Matrix<dataType,4,4> pose)
{
  mrpt::math::CMatrixDouble44 mat_mrpt(pose);
  mrpt::poses::CPose3D pose_mrpt(mat_mrpt);
  mrpt::math::CArrayDouble<3> vRot_mrpt = pose_mrpt.ln_rotation();
  Eigen::Matrix<dataType,3,1> rotation = Eigen::Matrix<dataType,3,1>(vRot_mrpt[0],vRot_mrpt[1],vRot_mrpt[2]);
  return  rotation;
}

/*! Check if the path 'filename' corresponds to a file */
inline bool fexists(const char *filename)
{
  std::ifstream ifile(filename);
  return ifile;
}

/*! Calculate the rotation difference between the two poses */
inline float diffRotation(Eigen::Matrix4f &pose1, Eigen::Matrix4f &pose2)
{
    // Eigen::Matrix3f relativeRotation = pose1.block(0,0,3,3).transpose() * pose2.block(0,0,3,3);
    //    Eigen::Isometry3d cam; // camera pose
    Eigen::Matrix3f m_rot1 = pose1.block(0,0,3,3);
    Eigen::Quaternionf _pose1(m_rot1);
    Eigen::Matrix3f m_rot2 = pose2.block(0,0,3,3);
    Eigen::Quaternionf _pose2(m_rot2);

    float anglePoses = _pose1.angularDistance(_pose2); // in radians
//    std::cout << "  anglePoses " << anglePoses << std::endl;

    return mrpt::utils::RAD2DEG(anglePoses);
//    return RAD2DEG(anglePoses);
}

/*! Calculate the rotation difference between the two poses */
inline float difTranslation(Eigen::Matrix4f &pose1, Eigen::Matrix4f &pose2)
{
    Eigen::Matrix4f relativePose = pose1.inverse() * pose2;
    std::cout << "  distPoses " << relativePose.block(0,3,3,1).norm() << std::endl;
    return relativePose.block(0,3,3,1).norm(); // in meters
//    Eigen::Vector3f diffTrans = pose1.block(0,3,3,1) -
}

// Return a diagonal matrix where the values of the diagonal are assigned from the input vector
template<typename typedata, int nRows, int nCols> inline
Eigen::Matrix<typedata,nRows,nCols> getDiagonalMatrix(const Eigen::Matrix<typedata,nRows,nCols> &matrix_generic)
{
    assert(nRows == nCols);

    Eigen::Matrix<typedata,nRows,nCols> m_diag = Eigen::Matrix<typedata,nRows,nCols>::Zero();
    for(size_t i=0; i < nRows; i++)
        m_diag(i,i) = matrix_generic(i,i);

    return m_diag;
}

/*! Compute the mean and standard deviation from a std::vector of float/double values.*/
template<typename dataType> inline
void calcMeanAndStDev(std::vector<dataType> &v, dataType &mean, dataType &stdev)
{
    dataType sum = std::accumulate(v.begin(), v.end(), 0.0);
    mean =  sum / v.size();

    dataType accum = 0.0;
    for(unsigned i=0; i<v.size(); i++)
        accum += (v[i] - mean) * (v[i] - mean);
    stdev = sqrt(accum / (v.size()-1));

//    dataType accum = 0.0;
//    std::for_each (v.begin(), v.end(), [&](const dataType d) {
//        accum += (d - mean) * (d - mean);
//    });
//    stdev = sqrt(accum / (v.size()-1));

//    std::vector<dataType> diff(v.size());
//    std::transform(v.begin(), v.end(), diff.begin(),
//                   std::bind2nd(std::minus<double>(), mean));
//    dataType sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//    stdev = std::sqrt(sq_sum / v.size()-1);
}

///* Transform pose from Tawsif reference system to the one of RGBD360 */
//Eigen::Matrix4f T_axis, T_rot_offset, T, T_edu_tawsif;
//void loadTransf_edu_tawsif()
//{
//  T_axis << 0, -1, 0, 0,
//            1, 0, 0, 0,
//            0, 0, 1, 0,
//            0, 0, 0, 1;
//  float angle_offset = PI*7/8;
//  T_rot_offset << 1, 0, 0, 0,
//                  0, cos(angle_offset), -sin(angle_offset), 0,
//                  0, sin(angle_offset), cos(angle_offset), 0,
//                  0, 0, 0, 1;

////  T_edu_tawsif = (T_axis.inverse() * T_rot_offset.inverse());
//  T_edu_tawsif = T_rot_offset.inverse() * T_axis;

//  std::cout << "T_edu_tawsif \n" << T_edu_tawsif << std::endl;
//}

///* Load pose from Tawsif and convert it to the RGBD360 reference system */
//Eigen::Matrix4f loadPoseTawsifRef(std::string poseName)
//{
//  Eigen::Matrix4f poseTawsif;
//  poseTawsif.loadFromTextFile(poseName);
////  Eigen::Matrix4f poseEdu = T_edu_tawsif.inverse() * poseTawsif * T_edu_tawsif;
//  Eigen::Matrix4f poseEdu = T_edu_tawsif * poseTawsif * T_edu_tawsif.inverse();

//  return poseEdu;
//}

/*! Return a vector of pairs of 2D-points (x1,y1,x2,y2) defining segments which correspond to vertical planes */
inline std::vector<Eigen::Vector4f> getVerticalPlanes(mrpt::pbmap::PbMap &planes)
{
  std::vector<Eigen::Vector4f> wall_planes2D;
  for(unsigned i=0; i<planes.vPlanes.size(); i++)
  {
    mrpt::pbmap::Plane &plane_i = planes.vPlanes[i];

    if(plane_i.v3normal(0) < 0.98) // Check that the plane is vertical
      continue;

    if(plane_i.areaHull < 2.0) // Discard small planes
      continue;

    Eigen::Vector2f normal2D = Eigen::Vector2f(plane_i.v3normal(1),plane_i.v3normal(2));
    Eigen::Vector2f center2D = Eigen::Vector2f(plane_i.v3center(1),plane_i.v3center(2));
    normal2D /= normal2D.norm();
    Eigen::Vector2f extremPointRight, extremPointLeft;
    float distRight=0, distLeft=0;

    unsigned dirPpal;
    if(fabs(normal2D(0)) > 0.1)
      dirPpal = 1;
    else
      dirPpal = 0;

    for(unsigned j=0; j<plane_i.polygonContourPtr->size(); j++)
    {
      Eigen::Vector2f vertex2D = Eigen::Vector2f(plane_i.polygonContourPtr->points[j].y, plane_i.polygonContourPtr->points[j].z);
      if(vertex2D(1) > center2D(1))
      {
        if((vertex2D - center2D).norm() > distRight)
        {
          extremPointRight = vertex2D;
          distRight = (vertex2D - center2D).norm();
        }
      }
      else
      {
        if((vertex2D - center2D).norm() > distLeft)
        {
          extremPointLeft = vertex2D;
          distLeft = (vertex2D - center2D).norm();
        }
      }
    }
    wall_planes2D.push_back(Eigen::Vector4f(extremPointRight(0),extremPointRight(1),extremPointLeft(0),extremPointLeft(1)));
  }

  return wall_planes2D;
}

template<typename T> inline
T median(std::vector<T> &v)
{
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

#endif
