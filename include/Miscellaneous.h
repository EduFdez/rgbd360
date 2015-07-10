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

#pragma once

#ifndef MISCELLANEOUS_H
#define MISCELLANEOUS_H

#ifndef PI
    #define PI 3.14159265359
#endif

#include <mrpt/pbmap.h>
#include <mrpt/utils/CStream.h>

#include <opencv2/core/eigen.hpp>

#include <iterator>
#include <algorithm>
//#include <vector>

//#include <Eigen/Core>
//#include <Eigen/SVD>
//#include <iostream>
//#include <fstream>

/*! Generate a skew-symmetric matrix from a 3D vector */
template<typename dataType> inline Eigen::Matrix<dataType,3,3> skew(const Eigen::Matrix<dataType,3,1> &vec)
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
Eigen::Matrix<dataType,3,1> getPoseTranslation(const Eigen::Matrix<dataType,4,4> pose)
{
  Eigen::Matrix<dataType,3,1> translation = pose.block(0,3,3,1);
  return  translation;
}

/*! Return the rotation vector of the input pose */
template<typename dataType> inline
Eigen::Matrix<dataType,3,1> getPoseRotation(const Eigen::Matrix<dataType,4,4> pose)
{
  mrpt::math::CMatrixDouble44 mat_mrpt(pose);
  mrpt::poses::CPose3D pose_mrpt(mat_mrpt);
  mrpt::math::CArrayDouble<3> vRot_mrpt = pose_mrpt.ln_rotation();
  Eigen::Matrix<dataType,3,1> rotation = Eigen::Matrix<dataType,3,1>(vRot_mrpt[0],vRot_mrpt[1],vRot_mrpt[2]);
  return  rotation;
}

/*! Return the rotation vector of the input pose */
inline Eigen::Matrix4f getPoseEigenMatrix(const mrpt::poses::CPose3D & pose)
{
  Eigen::Matrix4f pose_mat;
  mrpt::math::CMatrixDouble44 pose_mat_mrpt;
  pose.getHomogeneousMatrix(pose_mat_mrpt);
  pose_mat << pose_mat_mrpt(0,0), pose_mat_mrpt(0,1), pose_mat_mrpt(0,2), pose_mat_mrpt(0,3),
              pose_mat_mrpt(1,0), pose_mat_mrpt(1,1), pose_mat_mrpt(1,2), pose_mat_mrpt(1,3),
              pose_mat_mrpt(2,0), pose_mat_mrpt(2,1), pose_mat_mrpt(2,2), pose_mat_mrpt(2,3),
              pose_mat_mrpt(3,0), pose_mat_mrpt(3,1), pose_mat_mrpt(3,2), pose_mat_mrpt(3,3) ;
  return  pose_mat;
}

/*! Check if the path 'filename' corresponds to a file */
inline bool fexists(const char *filename)
{
  std::ifstream ifile(filename);
  return ifile;
}

/*! Calculate the rotation difference between the two poses */
inline float diffRotation(const Eigen::Matrix4f & pose1, const Eigen::Matrix4f & pose2)
{
    // Eigen::Matrix3f relativeRotation = pose1.block(0,0,3,3).transpose() * pose2.block(0,0,3,3);
    //    Eigen::Isometry3d cam; // camera pose
    Eigen::Matrix3f m_rot1 = pose1.block(0,0,3,3);
    Eigen::Quaternionf _pose1(m_rot1);
    Eigen::Matrix3f m_rot2 = pose2.block(0,0,3,3);
    Eigen::Quaternionf _pose2(m_rot2);

    float anglePoses = _pose1.angularDistance(_pose2); // in radians
//    std::cout << "  anglePoses " << anglePoses << std::endl;

    //return mrpt::utils::RAD2DEG(anglePoses);
    return anglePoses;
}

/*! Calculate the rotation difference between the two poses */
inline float difTranslation(const Eigen::Matrix4f & pose1, const Eigen::Matrix4f & pose2)
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
void calcMeanAndStDev(const std::vector<dataType> & vec, dataType mean, dataType stdev)
{
    dataType sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean =  sum / vec.size();

    dataType accum = 0.0;
    for(unsigned i=0; i<vec.size(); i++)
        accum += (vec[i] - mean) * (vec[i] - mean);
    stdev = sqrt(accum / (vec.size()-1));

//    dataType accum = 0.0;
//    std::for_each (vec.begin(), vec.end(), [&](const dataType d) {
//        accum += (d - mean) * (d - mean);
//    });
//    stdev = sqrt(accum / (vec.size()-1));

//    std::vector<dataType> diff(vec.size());
//    std::transform(vec.begin(), vec.end(), diff.begin(),
//                   std::bind2nd(std::minus<double>(), mean));
//    dataType sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//    stdev = std::sqrt(sq_sum / vec.size()-1);
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
inline std::vector<Eigen::Vector4f> getVerticalPlanes(mrpt::pbmap::PbMap & planes)
{
  std::vector<Eigen::Vector4f> wall_planes2D;
  for(unsigned i=0; i < planes.vPlanes.size(); i++)
  {
    mrpt::pbmap::Plane & plane_i = planes.vPlanes[i];

    if(plane_i.v3normal(0) < 0.98) // Check that the plane is vertical
      continue;

    if(plane_i.areaHull < 2.0) // Discard small planes
      continue;

    Eigen::Vector2f normal2D = Eigen::Vector2f(plane_i.v3normal(1),plane_i.v3normal(2));
    Eigen::Vector2f center2D = Eigen::Vector2f(plane_i.v3center(1),plane_i.v3center(2));
    normal2D /= normal2D.norm();
    Eigen::Vector2f extremPointRight, extremPointLeft;
    float distRight=0, distLeft=0;

//    unsigned dirPpal;
//    if(fabs(normal2D(0)) > 0.1)
//      dirPpal = 1;
//    else
//      dirPpal = 0;

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

///*! Sort a vector and retrieve the indexes of teh sorted values.*/ // This is already in PbMap
//std::vector<size_t> sort_indexes__(const std::vector<float> & v)
//{
//  // initialize original index locations
//  std::vector<size_t> idx(v.size());
//  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

//  // sort indexes based on comparing values in v
//  std::sort( idx.begin(), idx.end(), [&v](size_t i1, size_t i2) -> bool {return (v[i1]) > (v[i2]);} );

//  return idx;
//}

//template <typename T>
//std::vector<size_t> sort_vector__(std::vector<T> & v)
//{
//  // initialize original index locations
//  std::vector<size_t> idx = sort_indexes(v);

//  std::vector<T> sorted_vector(v.size());
//  for (size_t i = 0; i != idx.size(); ++i)
//    sorted_vector[i] = v[idx[i]];
//  v = sorted_vector;

//  return idx;
//}

/*! Sort a vector and retrieve the indexes of teh sorted values.*/
template <typename T>
std::vector<size_t> sort_indexes_(const Eigen::Matrix<T, Eigen::Dynamic, 1> & v)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), static_cast<size_t>(0));

  // sort indexes based on comparing values in v
  std::sort( idx.begin(), idx.end(), [&v](const size_t &i1, const size_t &i2) {return fabs(v[i1]) > fabs(v[i2]);} );

  return idx;
}

template <typename T>
std::vector<size_t> sort_vector_(Eigen::Matrix<T, Eigen::Dynamic, 1> & v)
{
  // initialize original index locations
  std::vector<size_t> idx = sort_indexes_(v);

  Eigen::Matrix<T, Eigen::Dynamic, 1> sorted_vector(v.rows());
  for (size_t i = 0; i != idx.size(); ++i)
    sorted_vector[i] = v[idx[i]];
  v = sorted_vector;

  return idx;
}

inline void convertRange_mrpt2cvMat(const mrpt::math::CMatrix &range_mrpt, cv::Mat & depthImage)
{
    //Eigen::MatrixXf range_eigen(range_mrpt.getMatrix());
    Eigen::MatrixXf range_eigen(range_mrpt.eval());
    cv::eigen2cv(range_eigen, depthImage);
}

#endif
