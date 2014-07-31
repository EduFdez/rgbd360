/*
 *  Kinect-RGBD-GraphSLAM6D
 *  Simultaneous Localization and Mapping with 3 or 6 degrees of freedom using the Kinect sensor
 *  Copyright (c) 2011-2012, Miguel Algaba Borrego.
 *  http://code.google.com/p/kinect-rgbd-graphslam6d/
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
 */

#include "GraphOptimizer.h"

#include <mrpt/base.h>
#include <mrpt/graphs.h>
#include <mrpt/graphslam.h>
#include <mrpt/poses/CPose3DPDFGaussianInf.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/graphs/CNetworkOfPoses.h>

/*!This class encapsulates the functionality of the MRPT graph-slam module to perform 3D/6D graph optimization.*/
class GraphOptimizer_MRPT : public GraphOptimizer
{
private:
  int vertexIdx;
    mrpt::graphs::CNetworkOfPoses3DInf graph3D; //Graph for 3D poses (6DoF)
    mrpt::graphs::CNetworkOfPoses2DInf graph2D; //Graph for 2D poses (3DoF)

public:
  GraphOptimizer_MRPT();
  /*!Adds a new vertex to the graph. The provided 4x4 matrix will be considered as the pose of the new added vertex. It returns the index of the added vertex.*/
  int addVertex(Eigen::Matrix4d& pose);
  /*!Adds an edge that defines a spatial constraint between the vertices "fromIdx" and "toIdx" with information matrix that determines the weight of the added edge.*/
  void addEdge(const int fromIdx,const int toIdx,Eigen::Matrix4d& relPose,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& infMatrix);
  /*!Calls the graph optimization process to determine the pose configuration that best satisfies the constraints defined by the edges.*/
  void optimizeGraph();
  /*!Returns a vector with all the optimized poses of the graph.*/
  void getPoses(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d > >&);
  /*!Saves the graph to file.*/
  void saveGraph(std::string fileName);
};
