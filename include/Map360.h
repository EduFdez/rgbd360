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
 *  Author: Eduardo Fernandez-Moral
 */

#ifndef MAP360_H
#define MAP360_H

#include "Frame360.h"

/*! This class defines a pose-graph map of spherical RGB-D keyframes. This map can be arranged in topological areas
 *  defining places with respect to any criteria (e.g. mutual visibility or the description of the image content).
 *  This map is structured in a hybrid represetation, as the spherical RGB-D keyframes are annotated with metric,
 *  topological and semantic attributes.
 */
struct Map360
{
 public:

  /*! Vector of spherical keyframes (created from omnidirectional RGB-D images) */
  std::vector<Frame360*> vpSpheres;

  /*! Vector containing the global SE3 poses of vpSpheres (odometry) */
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > vTrajectoryPoses;

  /*! Vector of the global SE3 poses optimized with graphSLAM (or pose-graph SLAM)*/
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > vOptimizedPoses;

  /*! Vector storing the euclidean distance between of each stored keyframe with respect to the previous one (the first element of the vector is neglected) */
  std::vector<float> vTrajectoryIncrements;

  /*! Double map storing the connections between keyframes. Each connection stores the SE3 pose and a 6x6 covariance matrix */
  std::map<unsigned, std::map<unsigned, std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > > > mmConnectionKFs;

  /*! Topological area where the camera was last localized */
  unsigned currentArea;

  /*! std::map defining the topological areas, where they key is the topological reference and the value contains the keyframe indices */
  std::vector< std::set<unsigned> > vsAreas;

  /*! Vector storing the indices of neighboring topologial nodes */
  std::vector< std::set<unsigned> > vsNeighborAreas;

  /*! Selected keyframes (they correspond to the most representative (most highly connected) sphere of each area) */
  std::vector<unsigned> vSelectedKFs;
//  std::map<unsigned, unsigned> msSelectedKFs;

  /*! Local reference system for each topological area */
  std::vector<Eigen::Matrix4f> vRef;

  /*! Mutex to syncrhronize eventual changes in the map */
  boost::mutex mapMutex;

  Map360() //:
//    currentArea(0)
  {
//    std::set<unsigned> firstArea;
//    firstArea.reserve(100);
//    vsAreas.push_back(firstArea);
  }

  /*! Add a new keyframe (sphere+pose) */
  void addKeyframe(Frame360* sphere, Eigen::Matrix4f &pose)
  {
    vpSpheres.push_back(sphere);
    vTrajectoryPoses.push_back(pose);
  }
};

#endif
