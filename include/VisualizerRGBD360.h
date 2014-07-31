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
 * Author: Eduardo Fernandez-Moral
 */

#ifndef VISUALIZER_RGBD360_H
#define VISUALIZER_RGBD360_H

//#include <mrpt/base.h>
//#include <mrpt/pbmap.h>

#include "Frame360.h"

//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <boost/thread/thread.hpp>

#include <Eigen/Core>

#include <map>
#include <string>

#define MAX_MATCH_PLANES 20

typedef pcl::PointXYZRGBA PointT;

mrpt::pbmap::config_heuristics configLocaliser;

class VisualizerRGBD360
{
 public:

  Eigen::Matrix4f rigidTransf; // Pose of pTrg360 as seen from pRef360

  pcl::visualization::CloudViewer viewer;

  VisualizerRGBD360(Frame360 *frame1, Frame360 *frame2) :
    viewer("RGBD360")
  {
  }

  void Register()
  {
//    double time_start1 = pcl::getTime();

////  std::cout << "Registration\n";
//    // Register point clouds
//    mrpt::pbmap::Subgraph refGraph;
//    refGraph.pPBM = &pRef360->planes;
//    for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++){
////     if(pRef360->planes.vPlanes[i].curvature < 0.002)
////      std::cout << i << " " << pRef360->planes.vPlanes[i].id << std::endl;
//      refGraph.subgraphPlanesIdx.insert(pRef360->planes.vPlanes[i].id);}
//
//    mrpt::pbmap::Subgraph trgGraph;
//    trgGraph.pPBM = &pTrg360->planes;
//    for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
////     if(pTrg360->planes.vPlanes[i].curvature < 0.002)
//      trgGraph.subgraphPlanesIdx.insert(pTrg360->planes.vPlanes[i].id);

    // Register point clouds
    mrpt::pbmap::Subgraph refGraph;
    refGraph.pPBM = &pRef360->planes;
  //cout << "Size planes1 " << pRef360->planes.vPlanes.size() << endl;
    if(pRef360->planes.vPlanes.size() > MAX_MATCH_PLANES)
    {
      std::vector<float> planeAreas(pRef360->planes.vPlanes.size());
      for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++)
        planeAreas[i] = pRef360->planes.vPlanes[i].areaHull;

      std::sort(planeAreas.begin(),planeAreas.end());
      float areaThreshold = planeAreas[pRef360->planes.vPlanes.size() - MAX_MATCH_PLANES];

      for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++)
//            std::cout << i << " " << planeAreas[i] << std::endl;
       if(pRef360->planes.vPlanes[i].areaHull >= areaThreshold)
        refGraph.subgraphPlanesIdx.insert(pRef360->planes.vPlanes[i].id);
    }
    else
    {
      for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++){//cout << "id " << pRef360->planes.vPlanes[i].id << endl;
//         if(pRef360->planes.vPlanes[i].curvature < 0.001)
        refGraph.subgraphPlanesIdx.insert(pRef360->planes.vPlanes[i].id);}
    }

    mrpt::pbmap::Subgraph trgGraph;
    trgGraph.pPBM = &pTrg360->planes;
    if(pTrg360->planes.vPlanes.size() > MAX_MATCH_PLANES)
    {
      std::vector<float> planeAreas(pTrg360->planes.vPlanes.size());
      for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
        planeAreas[i] = pTrg360->planes.vPlanes[i].areaHull;

      std::sort(planeAreas.begin(),planeAreas.end());
      float areaThreshold = planeAreas[pTrg360->planes.vPlanes.size() - MAX_MATCH_PLANES];

      for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
//            std::cout << i << " " << planeAreas[i] << std::endl;
       if(pTrg360->planes.vPlanes[i].areaHull >= areaThreshold)
        trgGraph.subgraphPlanesIdx.insert(pTrg360->planes.vPlanes[i].id);
    }
    else
    {
      for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
//         if(pTrg360->planes.vPlanes[i].curvature < 0.001)
        trgGraph.subgraphPlanesIdx.insert(pTrg360->planes.vPlanes[i].id);
    }
    cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << endl;

//    std::cout << "Number of planes in Ref " << pRef360->planes.vPlanes.size() << " Trg " << pTrg360->planes.vPlanes.size() << std::endl;
//    std::cout << "Number of planes in Ref " << pRef360->planes.vPlanes.size() << " Trg " << &pTrg360->planes.vPlanes.size() << std::endl;
//    std::cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << std::endl;

    configLocaliser.load_params("../../../Dropbox/Doctorado/Projects/mrpt/share/mrpt/config_files/pbmap/configLocaliser_spherical.ini");
//    configLocaliser.load_params("../../../Dropbox/Doctorado/Projects/mrpt/share/mrpt/config_files/pbmap/configLocaliser_sphericalOdometry.ini");
  std::cout << "hue_threshold " << configLocaliser.color_threshold << " cosAngle " << configLocaliser.angle << std::endl;

    mrpt::pbmap::SubgraphMatcher matcher;
//    std::map<unsigned, unsigned> bestMatch = matcher.compareSubgraphs(trgGraph,refGraph);

//  double time_end1 = pcl::getTime();
//  std::cout << "preparation took " << double (time_end1 - time_start1) << std::endl;

    matcher.totalUnary = 0;
    matcher.semanticPair = 0;
    matcher.rejectSemantic = 0;
    double time_start = pcl::getTime();

    std::map<unsigned, unsigned> bestMatch = matcher.compareSubgraphs(refGraph, trgGraph);
//    std::map<unsigned, unsigned> bestMatch = matcher.compareSubgraphsOdometry(refGraph, trgGraph);

  double time_end = pcl::getTime();
  std::cout << "compareSubgraphs took " << double (time_end - time_start)*1000 << " ms\n";
//  std::cout << "stats " << matcher.totalUnary << " " << matcher.semanticPair << " " << matcher.rejectSemantic << std::endl;
//  std::cout << "planes " << pRef360->planes.vPlanes.size() << " " << pTrg360->planes.vPlanes.size() << std::endl;

    std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << std::endl;
    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
      std::cout << it->first << " " << it->second << std::endl;

    if(bestMatch.size() < 3)
    {
      cout << "\n\tInsuficient matching\n\n";
      return;
    }

//    std::map<unsigned, unsigned> manualMatch_15_16; manualMatch_15_16[0] = 0; manualMatch_15_16[18] = 21; manualMatch_15_16[19] = 18;// manualMatch_15_16[0] = 7; manualMatch_15_16[13] = 5; manualMatch_15_16[8] = 2;

//    time_start = pcl::getTime();

    // Superimpose model
//    mrpt::pbmap::ConsistencyTest fitModel(pTrg360->planes, pRef360->planes);
    mrpt::pbmap::ConsistencyTest fitModel(pRef360->planes, pTrg360->planes);
//    mrpt::pbmap::ConsistencyTest fitModel(pbmap2, pbmap1);
//    rigidTransf = fitModel.initPose2D(manualMatch_12_13);
    rigidTransf = fitModel.initPose(bestMatch);
//    rigidTransf = fitModel.estimatePose(bestMatch);
//    Eigen::Matrix4f rigidTransf2 = fitModel.estimatePoseRANSAC(bestMatch);
//    std::cout << "VisualizerRGBD360 Rigid transformation:\n" << rigidTransf << std::endl;
//    std::cout << "Rigid transformation:\n" << rigidTransf.block(0,0,3,3) << std::endl;
//    std::cout << "Rigid transformation:\n" << rigidTransf.block(0,3,3,1) << std::endl;

//  time_end = pcl::getTime();
//  std::cout << "ConsistencyTest took " << double (time_end - time_start) << std::endl;

//    // Generate exact data
//    std::vector<Eigen::Vector3f> n_(bestMatch.size());
//    std::vector<Eigen::Vector3f> c_(bestMatch.size());
//    std::vector<float> d_(bestMatch.size());
//    unsigned i=0;
//    Eigen::Vector3f translation = rigidTransf.block(0,3,3,1);
////    Eigen::Vector3f translation = -(rigidTransf.block(0,0,3,3).transpose() * rigidTransf.block(0,3,3,1));
////    Eigen::Vector3f translation = -(rigidTransf.block(0,0,3,3) * rigidTransf.block(0,3,3,1));
//  std::cout << "Translation G " << translation.transpose() << std::endl;
//    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++, i++)
//    {
////    std::cout << "d1 " << pRef360->planes.vPlanes[it->first].d << " " << -(pRef360->planes.vPlanes[it->first].v3normal. dot(pRef360->planes.vPlanes[it->first].v3center) ) << std::endl;
////    std::cout << "d2 " << pTrg360->planes.vPlanes[it->second].d << " " << -(pTrg360->planes.vPlanes[it->second].v3normal. dot(pTrg360->planes.vPlanes[it->second].v3center) ) << std::endl;
//      n_[i] = rigidTransf.block(0,0,3,3) * pTrg360->planes.vPlanes[it->second].v3normal;
//      c_[i] = rigidTransf.block(0,0,3,3) * pTrg360->planes.vPlanes[it->second].v3center + rigidTransf.block(0,3,3,1);
//      d_[i] = -(pTrg360->planes.vPlanes[it->second].v3normal .dot(pTrg360->planes.vPlanes[it->second].v3center)) - n_[i] .dot ( translation );
////      d_[i] = -(pTrg360->planes.vPlanes[it->second].v3normal .dot(pTrg360->planes.vPlanes[it->second].v3center)) - pTrg360->planes.vPlanes[it->second].v3normal .dot ( translation );
//    std::cout << "d2 " << pTrg360->planes.vPlanes[it->second].d << " d2 " << -(pTrg360->planes.vPlanes[it->second].v3normal .dot(pTrg360->planes.vPlanes[it->second].v3center)) << std::endl;
//    std::cout << "error " << d_[i] +(n_[i].dot(c_[i])) << " d' " << d_[i] << " " << -(n_[i].dot(c_[i])) << std::endl;
//    }
//
////    //Calculate rotation
////    i=0;
////    Eigen::Matrix3f normalCovariances = Eigen::Matrix3f::Zero();
////    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++, i++)
////      normalCovariances += pTrg360->planes.vPlanes[it->second].v3normal * n_[i].transpose();
//////      normalCovariances += pTrg360->planes.vPlanes[it->second].v3normal * pRef360->planes.vPlanes[it->first].v3normal.transpose();
////
////    Eigen::JacobiSVD<Eigen::MatrixXf> svd(normalCovariances, Eigen::ComputeThinU | Eigen::ComputeThinV);
////    Eigen::Matrix3f Rotation = svd.matrixV() * svd.matrixU().transpose();
////
////    float conditioning = svd.singularValues().maxCoeff()/svd.singularValues().minCoeff();
////    if(conditioning > 100)
////    {
////      cout << " Bad conditioning: " << conditioning << " -> Returning the identity\n";
////    }
////
////    double det = Rotation.determinant();
////    if(det != 1)
////    {
////      Eigen::Matrix3f aux;
////      aux << 1, 0, 0, 0, 1, 0, 0, 0, det;
////      Rotation = svd.matrixV() * aux * svd.matrixU().transpose();
////    }
////    std::cout << "Rotation\n" << Rotation << std::endl;
//
//    // Calculate translation
//    Eigen::Matrix3f hessian = Eigen::Matrix3f::Zero();
//    Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
//    i=0;
//    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++, i++)
//    {
//      float trans_error = (d_[i] + (pTrg360->planes.vPlanes[it->second].v3normal .dot(pTrg360->planes.vPlanes[it->second].v3center)));
//        hessian += n_[i] * n_[i].transpose();
//        gradient += n_[i] * trans_error;
//    }
//    translation = -hessian.inverse() * gradient;
//  std::cout << "Translation " << translation.transpose() << std::endl;
//
//
//    if(rigidTransf == Eigen::Matrix4f::Identity())
//    {
//      cout << "\n\tInsuficient matching (bad conditioning).\n\n";
//      return;
//    }

//    ofstream saveRt;
//    saveRt.open("Rt.txt");
//    saveRt << rigidTransf;
//    saveRt.close();
  }

  void visualize()
  {
    viewer.runOnVisualizationThread (boost::bind(&VisualizerRGBD360::viz_cb, this, _1), "viz_cb");
    //    viewer.registerKeyboardCallback(&RegisterPairRGBD360::keyboardEventOccurred, *this);
  }

  void viz_cb (pcl::visualization::PCLVisualizer& viz)
  {
    if (pRef360->sphereCloud->empty())
    {
      mrpt::system::sleep(10);
      return;
    }

    {
  //  mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);

      // Render the data
      viz.removeAllPointClouds();
      viz.removeAllShapes();

      if (!viz.updatePointCloud (pRef360->sphereCloud, "pRef360->sphereCloud"))
        viz.addPointCloud (pRef360->sphereCloud, "pRef360->sphereCloud");

      if (!viz.updatePointCloud (pTrg360->sphereCloud, "pTrg360->sphereCloud"))
        viz.addPointCloud (pTrg360->sphereCloud, "pTrg360->sphereCloud");

      // Draw pbmap1
      char name[1024];

      for(size_t i=0; i < pRef360->planes.vPlanes.size(); i++)
      {
        mrpt::pbmap::Plane &plane_i = pRef360->planes.vPlanes[i];
        sprintf (name, "normal_%u", static_cast<unsigned>(i));
        pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
        pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
        pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
                            plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
                            plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
        viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

        {
          sprintf (name, "n%u %s", static_cast<unsigned>(i), plane_i.label.c_str());
    //            sprintf (name, "n%u_%u", static_cast<unsigned>(i), static_cast<unsigned>(plane_i.semanticGroup));
          viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
        }

  //      sprintf (name, "plane_%02u", static_cast<unsigned>(i));
  //      pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
  //      viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
  //      viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);

        sprintf (name, "approx_plane_%02d", int (i));
        viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);
      }

      for(size_t i=0; i < pTrg360->planes.vPlanes.size(); i++)
      {
        mrpt::pbmap::Plane &plane_i = pTrg360->planes.vPlanes[i];
        sprintf (name, "normal_%u_2", static_cast<unsigned>(i));
        pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
        pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
        pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
                            plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
                            plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
        viz.addArrow (pt2, pt1, ared[i%10], agrn[i%10], ablu[i%10], false, name);

        {
          sprintf (name, "n%u %s _2", static_cast<unsigned>(i), plane_i.label.c_str());
    //            sprintf (name, "n%u_%u", static_cast<unsigned>(i), static_cast<unsigned>(plane_i.semanticGroup));
          viz.addText3D (name, pt2, 0.1, ared[i%10], agrn[i%10], ablu[i%10], name);
        }

  //      sprintf (name, "plane_%02u", static_cast<unsigned>(i));
  //      pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[i%10], grn[i%10], blu[i%10]);
  //      viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
  //      viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);

        sprintf (name, "approx_plane_%02d_2", int (i));
        viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[i%10], 0.5 * grn[i%10], 0.5 * blu[i%10], name);
      }
    }
  }
};
#endif
