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

#ifndef GRAPHOPTIMIZER_H
#define GRAPHOPTIMIZER_H

#include <mrpt/base.h>
#include <mrpt/graphs.h>
#include <mrpt/graphslam.h>
#include <mrpt/poses/CPose3DPDFGaussianInf.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/graphs/CNetworkOfPoses.h>

/*! This class performs pose-graph optimization. This class is based on the previous work of
 *  Miguel Algaba Borrego, in "http://code.google.com/p/kinect-rgbd-graphslam6d/".
 */
class GraphOptimizer
{
 private:

  /*! Vertex counter */
  int vertexIdx;

  /*! Graph containing the 3D poses (6DoF) */
  mrpt::graphs::CNetworkOfPoses3DInf graph3D;

  /*! Graph containing the 2D poses (3DoF) */
  mrpt::graphs::CNetworkOfPoses2DInf graph2D;

 public:

  /*! Different optimization modes for pairs of PbMaps from spherical RGB-D observations */
  enum RigidTransformationType
  {
    SixDegreesOfFreedom,
    ThreeDegreesOfFreedom,
  } rigidTransformationType;

  /*! Constructor */
  GraphOptimizer()
  {
      //Set the vertex index to 0
      vertexIdx=0;
      rigidTransformationType = GraphOptimizer::SixDegreesOfFreedom;
  }

  /*!Sets the rigid transformation type to the graph optimizer (6DoF or 3DoF).*/
  inline void setRigidTransformationType(RigidTransformationType rttype)
  {
       rigidTransformationType = rttype;
  }

  /*! Add a new vertex (node) to the pose-graph */
  int addVertex(Eigen::Matrix4f& vertexPose)
  {
  //  std::cout << "Add vertex\n";
      //Transform the vertex pose from Eigen::Matrix4f to MRPT CPose3D
      mrpt::math::CMatrixDouble44 vertexPoseMRPT;

      for(int i=0;i<4;i++)
      {
          for(int j=0;j<4;j++)
          {
              vertexPoseMRPT(i,j)=vertexPose(i,j);
          }
      }

      if(rigidTransformationType==GraphOptimizer::SixDegreesOfFreedom) //Rigid transformation 6DoF
      {
          mrpt::poses::CPose3D p(vertexPoseMRPT);

          //Add the vertex to the graph
          graph3D.nodes[vertexIdx] = p; vertexIdx++;
      }
      else //Rigid transformation 3DoF
      {
          mrpt::poses::CPose3D p3D(vertexPoseMRPT);
          mrpt::poses::CPose2D p(p3D); //Construct a 2D pose from a 3D pose (x,y,phi):=(x',y',yaw')

          //Add the vertex to the graph
          graph2D.nodes[vertexIdx] = p; vertexIdx++;
      }

      //Return the vertex index
      return vertexIdx-1;
  }

  /*! Add a new edge (connection) to the pose-graph */
  void addEdge(const int fromIdx,
                                    const int toIdx,
                                    Eigen::Matrix4f& relativePose,
                                    Eigen::Matrix<float,6,6>& informationMatrix)
  //                                  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& informationMatrix)
  {
  //  std::cout << "Add edge\n";
      //Transform the relative pose from Eigen::Matrix4f to MRPT CPose3D
      mrpt::math::CMatrixDouble44 relativePoseMRPT;

      for(int i=0;i<4;i++)
      {
          for(int j=0;j<4;j++)
          {
              relativePoseMRPT(i,j)=relativePose(i,j);
          }
      }

      if(rigidTransformationType==GraphOptimizer::SixDegreesOfFreedom) //Rigid transformation 6DoF
      {
          mrpt::poses::CPose3D rp(relativePoseMRPT);

          //Convert Eigen::Matrix<double,6,6> to mrpt::math::CMatrixDouble66
          mrpt::math::CMatrixDouble66 infMat;
          for(int row=0;row<6;row++)
          {
              for(int col=0;col<6;col++)
              {
                  infMat(row,col)=informationMatrix(row,col);
              }
          }

          //Create a 6D pose with uncertainty
          mrpt::poses::CPose3DPDFGaussianInf rpInf = mrpt::poses::CPose3DPDFGaussianInf(rp,infMat);

          //Add the edge to the graph
          /*typename*/ mrpt::graphs::CNetworkOfPoses3DInf::edge_t RelativePose(rpInf);

          graph3D.insertEdge(mrpt::utils::TNodeID(fromIdx),mrpt::utils::TNodeID(toIdx),RelativePose);

      }
      else //Rigid transformation 3DoF
      {
          mrpt::poses::CPose3D rp3D(relativePoseMRPT);
          mrpt::poses::CPose2D rp(rp3D); //Construct a 2D pose from a 3D pose (x,y,phi):=(x',y',yaw')

          //Convert Eigen::Matrix<double,3,3> to mrpt::math::CMatrixDouble33
          mrpt::math::CMatrixDouble33 infMat;
          for(int row=0;row<3;row++)
          {
              for(int col=0;col<3;col++)
              {
                  infMat(row,col)=informationMatrix(row,col);
              }
          }

          //Create a 2D pose with uncertainty
          mrpt::poses::CPosePDFGaussianInf rpInf = mrpt::poses::CPosePDFGaussianInf(rp,infMat);

          //Add the edge to the graph
          /*typename*/ mrpt::graphs::CNetworkOfPoses2DInf::edge_t RelativePose(rpInf);

          graph2D.insertEdge(mrpt::utils::TNodeID(fromIdx),mrpt::utils::TNodeID(toIdx),RelativePose);
      }
  }

  /*! Optimize the pose-graph using the Levenberg-Marquardt algorithm */
  void optimizeGraph()
  {
  //  std::cout << "Optimize Graph \n";
      mrpt::utils::TParametersDouble  params;
      params["verbose"]  = 0;
      params["profiler"] = 0;
      params["max_iterations"] = 500;
      params["initial_lambda"] = 0.1;
      params["scale_hessian"] = 0.1;
      params["tau"] = 1e-3;

      mrpt::graphslam::TResultInfoSpaLevMarq  levmarq_info;

      if(rigidTransformationType==GraphOptimizer::SixDegreesOfFreedom) //Rigid transformation 6DoF
      {
          // The root node (the origin of coordinates):
          graph3D.root = mrpt::utils::TNodeID(0);

          // Do the optimization
          mrpt::graphslam::optimize_graph_spa_levmarq(graph3D,
                                                      levmarq_info,
                                                      NULL,  // List of nodes to optimize. NULL -> all but the root node.
                                                      params);
      }
      else //Rigid transformation 3DoF
      {
          // The root node (the origin of coordinates):
          graph2D.root = mrpt::utils::TNodeID(0);

          // Do the optimization
          mrpt::graphslam::optimize_graph_spa_levmarq(graph2D,
                                                      levmarq_info,
                                                      NULL,  // List of nodes to optimize. NULL -> all but the root node.
                                                      params);
      }
  }

  /*! Return the global poses of the graph nodes */
  void getPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f > >& poses)
  {
      poses.clear();
      if(rigidTransformationType==GraphOptimizer::SixDegreesOfFreedom) //Rigid transformation 6DoF
      {
          poses.resize(graph3D.nodes.size());

          for(int poseIdx=0;poseIdx<graph3D.nodes.size();poseIdx++)
          {
              //Transform the vertex pose from MRPT CPose3D to Eigen::Matrix4f
              mrpt::math::CMatrixDouble44 optimizedPoseMRPT;
              graph3D.nodes[poseIdx].getHomogeneousMatrix(optimizedPoseMRPT);

              Eigen::Matrix4f optimizedPose;
              for(int i=0;i<4;i++)
              {
                  for(int j=0;j<4;j++)
                  {
                      optimizedPose(i,j)=optimizedPoseMRPT(i,j);
                  }
              }

              //Set the optimized pose to the vector of poses
              poses[poseIdx]=optimizedPose;
          }
      }
      else //Rigid transformation 3DoF
      {
          poses.resize(graph2D.nodes.size());

          for(int poseIdx=0;poseIdx<graph2D.nodes.size();poseIdx++)
          {
              //Transform the vertex pose from MRPT CPose2D to Eigen::Matrix4f
              mrpt::poses::CPose2D optimizedPoseMRPT;
              optimizedPoseMRPT = graph2D.nodes[poseIdx];

              Eigen::Matrix4f optimizedPose = Eigen::Matrix4f::Identity();
              optimizedPose(0,0)=cos(optimizedPoseMRPT.phi());
              optimizedPose(0,1)=-sin(optimizedPoseMRPT.phi());
              optimizedPose(1,0)=sin(optimizedPoseMRPT.phi());
              optimizedPose(1,1)=cos(optimizedPoseMRPT.phi());
              optimizedPose(0,3)=optimizedPoseMRPT.x();
              optimizedPose(1,3)=optimizedPoseMRPT.y();

              //Set the optimized pose to the vector of poses
              poses[poseIdx]=optimizedPose;
          }
      }
  }

};
#endif
