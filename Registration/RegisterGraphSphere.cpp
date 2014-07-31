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

#include <mrpt/base.h>
#include <mrpt/pbmap.h>
#include <mrpt/utils/CStream.h>
//#include <mrpt/synch/CCriticalSection.h>

#include <Calib360.h>
#include <Frame360.h>
#include <Miscellaneous.h>

//#include <FrameRGBD.h>
//#include <RGBDGrabber.h>
//#include <RGBDGrabberOpenNI_PCL.h>
//#include <SerializeFrameRGBD.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h> //Save global map as PCD file
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/openni_camera/openni_driver.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cvmat_serialization.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Graph optimization
#include <GraphOptimizer/include/GraphOptimizer_MRPT.h>

// Topological partitioning
#include <mrpt/graphs/CGraphPartitioner.h>

#include <Eigen/Core>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define MAX_MATCH_PLANES 20
#define ENABLE_OPENMP_MULTITHREADING 0
#define SAVE_TRAJECTORY 0
#define VISUALIZE_POINT_CLOUD 0
#define RECORD_VIDEO 0

using namespace std;

typedef pcl::PointXYZRGBA PointT;

int frame;

mrpt::pbmap::config_heuristics configLocaliser;

void mrptMat2Eigen(mrpt::math::CMatrixDouble44 &relativePoseMRPT, Eigen::Matrix4f &relativePose)
{
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
//            relativePoseMRPT(i,j)=relativePose(i,j);
            relativePose(i,j)=relativePoseMRPT(i,j);
        }
    }
}
#include<mrpt/random/RandomGenerators.h>
using namespace mrpt::random;

class RegisterGraphSphere
{
  public:

//    void viz_cb2 (pcl::visualization::PCLVisualizer& viz)
//    {
//      viz.removeAllShapes();
//
//      for(unsigned i=0; i < vPoses.size(); i++)
//        viz.removeCoordinateSystem();
//
//      {
//
//        if(!bGraphSLAM)
//        {
////    cout << "   ::viz_cb(...)\n";
//
//          for(unsigned i=0; i < vPoses.size(); i++)
//          {
//            Eigen::Affine3f Rt;
//            Rt.matrix() = vPoses[i];
//            viz.addCoordinateSystem(0.1, Rt);
//          }
//
//        }
//        else
//        {
////    cout << "   ::viz_cb(...)\n";
//          for(unsigned i=0; i < vOptimizedPoses.size(); i++)
//          {
//            Eigen::Affine3f Rt;
//            Rt.matrix() = vOptimizedPoses[i];
//            viz.addCoordinateSystem(0.1, Rt);
//          }
//        }
//
//      }
//    }
//
//    void keyboardEventOccurred2 (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
//    {
//      if ( event.keyDown () )
//      {
//        if(event.getKeySym () == "l" || event.getKeySym () == "L"){cout << " Press O\n";
//          bGraphSLAM = !bGraphSLAM;
//          bFreezeFrame = false;
//          }
//      }
//    }
//    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > vOptimizedPoses;
//    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > vPoses;

    RegisterGraphSphere()
        :
        globalMap(new pcl::PointCloud<PointT>),
        bFreezeFrame(false),
        bTakeKeyframe(false),
        bDrawPlanes(false),
        bGraphSLAM(false),
        frameRecNum(0)
    {

//      // Silly test graph SLAM
//      mrpt::random::CRandomGenerator randomizer;
//      GraphOptimizer_MRPT optimizer;
//      optimizer.setRigidTransformationType(GraphOptimizer::SixDegreesOfFreedom);
////      Eigen::Matrix4d pose_eigen;
//      mrpt::math::CArrayNumeric<double, 6> pose_se3, noise;
//      pose_se3[0] = 1;
//      pose_se3[1] = 0;
//      pose_se3[2] = 0;
//      pose_se3[3] = 0;
//      pose_se3[4] = 0;
//      pose_se3[5] = 0.5236; // turn 30 deg around the Z axis
//      mrpt::math::CMatrixDouble44 pose_mrpt, pose_simulated, pose_noise;
//      mrpt::poses::CPose3D::exp(pose_se3).getHomogeneousMatrix(pose_mrpt);
//      Eigen::Matrix4f relativePose, currentPose = Eigen::Matrix4f::Identity();
//      vPoses.push_back(currentPose);
//      std::cout<< "First vertex: "<< optimizer.addVertex(currentPose) << std::endl;
////      cout << "pose_mrpt \n" << pose_mrpt << endl << endl;
////      cout << "currentPose \n" << currentPose << endl;
////      std::vector<Eigen::Matrix4f> absolute_poses(12);
//      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = Eigen::Matrix<double,6,6>::Identity();
//      double uniformNoise = 0.05;
//      for(unsigned i=1; i < 12; i++)
//      {
//        noise[0] = randomizer.drawUniform(0,uniformNoise);
//        noise[1] = randomizer.drawUniform(0,uniformNoise);
//        noise[2] = randomizer.drawUniform(0,uniformNoise);
//        noise[3] = randomizer.drawUniform(0,uniformNoise);
//        noise[4] = randomizer.drawUniform(0,uniformNoise);
//        noise[5] = randomizer.drawUniform(0,uniformNoise);
//        mrpt::poses::CPose3D::exp(noise).getHomogeneousMatrix(pose_noise);
//        pose_simulated = pose_noise * pose_mrpt;
////      cout << "pose_mrpt \n" << pose_mrpt << endl << "pose_noise \n" << pose_noise << endl << "pose_simulated \n" << pose_simulated << endl;
//
//        mrptMat2Eigen(pose_simulated, relativePose);
//        currentPose = currentPose * relativePose;
//        vPoses.push_back(currentPose);
////      cout << "relativePose \n" << relativePose << endl;
//      cout << "currentPose \n" << currentPose << endl;
//        std::cout<< "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;
//        optimizer.addEdge(i-1,i,relativePose,informationMatrix);
//      }
//    // Add the loop closure constraint
//      noise[0] = randomizer.drawUniform(0,uniformNoise);
//      noise[1] = randomizer.drawUniform(0,uniformNoise);
//      noise[2] = randomizer.drawUniform(0,uniformNoise);
//      noise[3] = randomizer.drawUniform(0,uniformNoise);
//      noise[4] = randomizer.drawUniform(0,uniformNoise);
//      noise[5] = randomizer.drawUniform(0,uniformNoise);
//      mrpt::poses::CPose3D::exp(noise).getHomogeneousMatrix(pose_noise);
//      pose_simulated = pose_noise * pose_mrpt;
//      mrptMat2Eigen(pose_simulated, relativePose);
//      optimizer.addEdge(11,0,relativePose,informationMatrix);
//
//      optimizer.optimizeGraph();
//      optimizer.getPoses(vOptimizedPoses);
//
//      for(unsigned i=0; i < vPoses.size(); i++)
//      {
//        cout << "Pose " << i << "\n" << vPoses[i] << "\n optimizedPose \n" << vOptimizedPoses[i] << endl;
//      }
//      mrptMat2Eigen(pose_mrpt, relativePose);
//      cout << "Pose 12 " << "\n" << vPoses.back() * relativePose << "\n optimizedPose \n" << vOptimizedPoses.back() * relativePose << endl;
//
//
//      pcl::visualization::CloudViewer viewer("RGBD360");
//      viewer.runOnVisualizationThread (boost::bind(&RegisterGraphSphere::viz_cb2, this, _1), "viz_cb"); //PCL viewer. It runs in a different thread.
//      viewer.registerKeyboardCallback(&RegisterGraphSphere::keyboardEventOccurred2, *this);
//
//      while (!viewer.wasStopped() )
//        boost::this_thread::sleep (boost::posix_time::milliseconds (10));

      // Get the calibration matrices for the different sensors
      calib.loadExtrinsicCalibration();
      calib.loadIntrinsicCalibration();

      configLocaliser.load_params("../../../Dropbox/Doctorado/Projects/mrpt/share/mrpt/config_files/pbmap/configLocaliser_spherical.ini");
  //  std::cout << "color_threshold " << configLocaliser.hue_threshold << std::endl;
    }

    ~RegisterGraphSphere()
    {
      // Clean memory
      for(unsigned i=0; i < spheres.size(); i++)
        delete spheres[i];
    }

    Eigen::Matrix4f Register(Frame360 *frame360_1, Frame360 *frame360, unsigned &numMatched, float &areaMatched_src)//std::map<unsigned, unsigned> &bestMatch, float &score)
    {
  //  std::cout << "Register...\n";
      // Register point clouds
      Eigen::Matrix4f rigidTransf;
      mrpt::pbmap::Subgraph refGraph;
      refGraph.pPBM = &frame360_1->planes;
      for(unsigned i=0; i < frame360_1->planes.vPlanes.size(); i++){
  //      std::cout << i << " " << frame360_1->planes.vPlanes[i].id << std::endl;
        refGraph.subgraphPlanesIdx.insert(frame360_1->planes.vPlanes[i].id);}

      mrpt::pbmap::Subgraph trgGraph;
      trgGraph.pPBM = &frame360->planes;
      for(unsigned i=0; i < frame360->planes.vPlanes.size(); i++)
        trgGraph.subgraphPlanesIdx.insert(frame360->planes.vPlanes[i].id);

//      std::cout << "Number of planes in Ref " << frame360_1->planes.vPlanes.size() << " Trg " << frame360->planes.vPlanes.size() << std::endl;
//      std::cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << std::endl;

      mrpt::pbmap::SubgraphMatcher matcher;
      std::map<unsigned, unsigned> bestMatch = matcher.compareSubgraphs(refGraph, trgGraph);
      areaMatched_src = matcher.calcAreaMatched(bestMatch);
      numMatched = bestMatch.size();

//      std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << std::endl;
//      for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
//        std::cout << it->first << " " << it->second << std::endl;

  //    std::map<unsigned, unsigned> manualMatch_15_16; manualMatch_15_16[0] = 0; manualMatch_15_16[18] = 21; manualMatch_15_16[19] = 18;// manualMatch_15_16[0] = 7; manualMatch_15_16[13] = 5; manualMatch_15_16[8] = 2;
  //    std::map<unsigned, unsigned> manualMatch_62_63; manualMatch_62_63[10] = 3; manualMatch_62_63[2] = 4; manualMatch_62_63[3] = 0; manualMatch_62_63[0] = 7; manualMatch_62_63[13] = 5; manualMatch_62_63[8] = 2;

      if(bestMatch.size() < 3)
      {
        cout << "\n\tInsuficient matching\n\n";
        return Eigen::Matrix4f::Identity();
      }

      // Superimpose model
  //    mrpt::pbmap::ConsistencyTest fitModel(frame360->planes, frame360_1->planes);
      mrpt::pbmap::ConsistencyTest fitModel(frame360_1->planes, frame360->planes);
  //    mrpt::pbmap::ConsistencyTest fitModel(pbmap2, pbmap1);
  //    rigidTransf = fitModel.initPose2D(manualMatch_12_13);
//      rigidTransf = fitModel.initPose(bestMatch);
      rigidTransf = fitModel.estimatePose(bestMatch);
  //    std::cout << "Rigid transformation:\n" << rigidTransf << std::endl;
      Eigen::Matrix4f rigidTransfInv = mrpt::pbmap::inverse(rigidTransf);
//      std::cout << "Rigid transformation:\n" << rigidTransf << std::endl << "Inverse\n" << rigidTransfInv << std::endl;
  //
  //
  //    std::cout << "Number of planes in Ref " << frame360_1->planes.vPlanes.size() << " Trg " << frame360->planes.vPlanes.size() << std::endl;
  //    std::cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << std::endl;
  //
  //    bestMatch = matcher.compareSubgraphs(trgGraph,refGraph);
  //    std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << std::endl;
  //    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
  //      std::cout << it->first << " " << it->second << std::endl;
  //
  //    mrpt::pbmap::ConsistencyTest fitModel2(frame360->planes, frame360_1->planes);
  //    rigidTransf = fitModel2.initPose(bestMatch);
  //    std::cout << "Rigid transformation:\n" << rigidTransf << std::endl;
  //
  //    if(rigidTransf == Eigen::Matrix4f::Identity())
  //      cout << "\n\tInsuficient matching (bad conditioning).\n\n";

      return rigidTransf;
    }

    Eigen::Matrix4f RegisterOdometry(Frame360 *frame360_1, Frame360 *frame360_2, float &areaMatched_src)//std::map<unsigned, unsigned> &bestMatch, float &score)
    {
  //  std::cout << "Register...\n";
      // Register point clouds
        // Register point clouds
        mrpt::pbmap::Subgraph refGraph;
        refGraph.pPBM = &frame360_1->planes;
      //cout << "Size planes1 " << frame360_1->planes.vPlanes.size() << endl;
        if(frame360_1->planes.vPlanes.size() > MAX_MATCH_PLANES)
        {
          vector<float> planeAreas(frame360_1->planes.vPlanes.size());
          for(unsigned i=0; i < frame360_1->planes.vPlanes.size(); i++)
            planeAreas[i] = frame360_1->planes.vPlanes[i].areaHull;

          sort(planeAreas.begin(),planeAreas.end());
          float areaThreshold = planeAreas[frame360_1->planes.vPlanes.size() - MAX_MATCH_PLANES];

          for(unsigned i=0; i < frame360_1->planes.vPlanes.size(); i++)
//            std::cout << i << " " << planeAreas[i] << std::endl;
            if(frame360_1->planes.vPlanes[i].areaHull >= areaThreshold)
              refGraph.subgraphPlanesIdx.insert(frame360_1->planes.vPlanes[i].id);
        }
        else
        {
          for(unsigned i=0; i < frame360_1->planes.vPlanes.size(); i++){//cout << "id " << frame360_1->planes.vPlanes[i].id << endl;
  //         if(frame360_1->planes.vPlanes[i].curvature < 0.001)
            refGraph.subgraphPlanesIdx.insert(frame360_1->planes.vPlanes[i].id);}
        }

      double sort_start = pcl::getTime();
        mrpt::pbmap::Subgraph trgGraph;
        trgGraph.pPBM = &frame360_2->planes;
        if(frame360_2->planes.vPlanes.size() > MAX_MATCH_PLANES)
        {
          vector<float> planeAreas(frame360_2->planes.vPlanes.size());
          for(unsigned i=0; i < frame360_2->planes.vPlanes.size(); i++)
            planeAreas[i] = frame360_2->planes.vPlanes[i].areaHull;

          sort(planeAreas.begin(),planeAreas.end());
          float areaThreshold = planeAreas[frame360_2->planes.vPlanes.size() - MAX_MATCH_PLANES];

          for(unsigned i=0; i < frame360_2->planes.vPlanes.size(); i++)
//            std::cout << i << " " << planeAreas[i] << std::endl;
            if(frame360_2->planes.vPlanes[i].areaHull >= areaThreshold)
              trgGraph.subgraphPlanesIdx.insert(frame360_2->planes.vPlanes[i].id);
        }
        else
        {
          for(unsigned i=0; i < frame360_2->planes.vPlanes.size(); i++)
  //         if(frame360_2->planes.vPlanes[i].curvature < 0.001)
            trgGraph.subgraphPlanesIdx.insert(frame360_2->planes.vPlanes[i].id);
        }
//      std::cout << "Sort took " << double (pcl::getTime() - sort_start)*1000 << " ms\n";
        cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << endl;

        double graph_matching_start = pcl::getTime();
        mrpt::pbmap::SubgraphMatcher matcher;
//        map<unsigned, unsigned> bestMatch = matcher.compareSubgraphs(refGraph, trgGraph);
        map<unsigned, unsigned> bestMatch = matcher.compareSubgraphs(refGraph, trgGraph, 2);
      std::cout << "Graph-Matching took " << double (pcl::getTime() - graph_matching_start)*1000 << " ms\n";

        cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << endl;
        for(map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
          cout << it->first << " " << it->second << endl;

        Eigen::Matrix4f rigidTransf = Eigen::Matrix4f::Identity();    // Pose of map as from current model

        if(bestMatch.size() >= 3)
        {
          // Superimpose model
          mrpt::pbmap::ConsistencyTest fitModel(frame360_1->planes, frame360_2->planes);
      //    mrpt::pbmap::ConsistencyTest fitModel(frame360_2->planes, frame360_1->planes);
  //    //    rigidTransf = fitModel.initPose2D(manualMatch_12_13);
//          rigidTransf = fitModel.initPose(bestMatch);
          rigidTransf = fitModel.estimatePose(bestMatch);
  //    //    rigidTransf = fitModel.getRTwithModel(manualMatch_12_13);
  //        rigidTransfInv = mrpt::pbmap::inverse(rigidTransf);
  //        cout << "Rigid transformation:\n" << rigidTransf << endl;
          cout << "Distance " << rigidTransf.block(0,3,3,1).norm() << endl;
        }

      areaMatched_src = matcher.calcAreaMatched(bestMatch);

      return rigidTransf;
    }

    // Evaluate sensed-space-overlap (SSO) between KF1 and KF2
    float calculateSSO(Frame360* frame_src, float& areaMatched_src)
    {
      float area_src = 0;
      for(unsigned i = 0; i < frame_src->planes.vPlanes.size(); i++)
        area_src += frame_src->planes.vPlanes[i].areaHull;

      float sso = areaMatched_src / area_src;

      return sso;
    }

//    //CALCULATE GRAPH CONECTION WEIGHTS
//    //Reckon Sensed Space Overlap (SSO) between new keyframe (nKF) and previous keyframes of the neighboring areas.
//    void MapMaker::UpdateSSO(KeyFrame &nKF)
//    {
//    //  mrpt::utils::CTicTac time;
//    //  time.Tic();
//      unsigned gMapSize = gMapMM->vpKeyFrames.size(); // Size of the new matrix
//      unsigned lastKFpos = gMapSize - 1;
//      gMapMM->vpSSO.setSize( gMapSize, gMapSize );		// Add a column and a row for the newKF
//      gMapMM->vpSSO(lastKFpos,lastKFpos) = 0.0;
//      gMapMM->KFdistributionSSO.push_back( nKF.idKF ); // Update KFdistributionSSO
//
//      // Update gMapMM's SSO
//      for( unsigned *itFrame=0; *itFrame < lastKFpos; *itFrame++)
//        gMapMM->vpSSO(*itFrame,lastKFpos) = gMapMM->vpSSO(lastKFpos,*itFrame) = calculateSSO(nKF, *(gMapMM->vpKeyFrames[*itFrame]));
//      // The SSO off-diagonal matrices of gMapMM's neighbors also have to be rearranged to insert the new row or column
//      for( set<unsigned>::iterator neig = gMapMM->neighborMaps.begin(); neig != gMapMM->neighborMaps.end(); neig++ )
//      {
//        //f << "Entra\n";
//        Map &neigMap = *mvpMaps[*neig];
//        if( gMapMM->mnMapNum > neigMap.mnMapNum )
//        {
//          neigMap.vpNeigSSO[gMapMM->mnMapNum].setSize(neigMap.vpKeyFrames.size(), gMapSize);
//          for( unsigned *itFrame=0; *itFrame < neigMap.vpKeyFrames.size(); *itFrame++)			//SSO with keyframes of this neighbor map
//            neigMap.vpNeigSSO[gMapMM->mnMapNum](*itFrame,lastKFpos) = calculateSSO(nKF, *(neigMap.vpKeyFrames[*itFrame]));
//        }
//        else{
//          gMapMM->vpNeigSSO[neigMap.mnMapNum].setSize(gMapSize, neigMap.vpKeyFrames.size() );
//          for( unsigned *itFrame=0; *itFrame < neigMap.vpKeyFrames.size(); *itFrame++)			//SSO with keyframes of this neighbor map
//            gMapMM->vpNeigSSO[neigMap.mnMapNum](lastKFpos,*itFrame) = calculateSSO(nKF, *(neigMap.vpKeyFrames[*itFrame]));
//        }
//      }
//    //mrpt::math::CMatrix KF_SSO(1,gMapSize);
//    //gMapMM->vpSSO.extractMatrix(gMapSize,0,KF_SSO);
//    //cout << "KF SSO:\n" << KF_SSO << endl;
//
//    //  for( unsigned counterNeig = 0; counterNeig < gMapMM->neighborMaps.size()-1; counterNeig++ )
//    //      cout << "SSO update neig " << gMapMM->neighborMaps[counterNeig] << endl << (*mvpMaps[ gMapMM->neighborMaps[counterNeig] ]->vpSSO) << endl;
//
//    //  cout << "SSO update in " << time.Tac()*1000 << " ms." << endl;
//    //  if (configSystem.verbose) cout << "SSO Matrix" << endl << *(gMapMM->vpSSO) << endl;	// Print SSO Weight matrix
//    }

    //// Arrange KFdistribution of map. Needed when swinging from neigbor nodes
    //void MapMaker::ArrangeKFdistributionSSO(Map *map)
    //{
    //  map->KFdistributionSSO.clear();
    //  for( unsigned int KF = 0; KF < map->vpKeyFrames.size(); KF++)
    //    map->KFdistributionSSO.push_back( map->vpKeyFrames[KF]->idKF );
    //}
    //
    //// Arrange KFdistribution of gMap and its neighbors
    //void MapMaker::ArrangeKFdistributionSSO(const set<unsigned int> &neighbors)
    //{
    //  for( set<unsigned int>::iterator Neigh = neighbors.begin(); Neigh != neighbors.end(); Neigh++ )
    //  {
    //    Map & neighborMap1 = *mvpMaps[ *Neigh ];                // neighbor1: Neighbor of first order
    //
    //    neighborMap1.prevKFdistributionSSO.clear();        // ¿esto esta bien? Y si viene de estar vacio?
    //    neighborMap1.prevKFdistributionSSO = neighborMap1.KFdistributionSSO;
    //    ArrangeKFdistributionSSO(&neighborMap1);
    //  }
    //  cout << "Finish ArrangeKFdistributionSSO\n";
    //}

    // Return a symmetric matrix with SSO coeficients of sensed-space-overlap (SSO) between the frames in the vicinity
    mrpt::math::CMatrix getVicinitySSO(set<unsigned> &vicinityCurrentNode)
    {
//    cout << "getVicinitySSO...\m";
      // Calculate size
//      vicinityCurrentNode.insert(currentNode);

      unsigned vicinitySize = 0;
      map<unsigned, unsigned> vicinitySSO_pos;    // Keeps the vicinitySSO position of each gMapMM's neighbor sub-map
      for( set<unsigned>::iterator node = vicinityCurrentNode.begin(); node != vicinityCurrentNode.end(); node++ )
      {
        vicinitySSO_pos[*node] = vicinitySize;
        vicinitySize += vsFramesInPlace[*node].size();
      }

      mrpt::math::CMatrix vicinitySSO(vicinitySize,vicinitySize);
      vicinitySSO.zeros();    // Is this necessary?
//    cout << "SSO_0\n" << vicinitySSO << endl << endl;
      for( set<unsigned>::iterator node = vicinityCurrentNode.begin(); node != vicinityCurrentNode.end(); node++ )
      {
//    cout << "SSO_\n" << vSSO[*node] << endl << endl;
        vicinitySSO.insertMatrix( vicinitySSO_pos[*node], vicinitySSO_pos[*node], vSSO[*node] ); // Intruduce diagonal block SSO
    //cout << "SSO_1\n" << vicinitySSO << endl << endl;
        // Intruduce off-diagonal SSO
        for( set<unsigned>::iterator neig = vicinityCurrentNode.begin(); (neig != vicinityCurrentNode.end() && *neig < *node); neig++ )
        {
//    cout << "SSO_COUPLE\n" << mmNeigSSO[*neig][*node] << endl << endl;
            vicinitySSO.insertMatrix( vicinitySSO_pos[*neig], vicinitySSO_pos[*node], mmNeigSSO[*neig][*node] ); // Insert up-diagonal block
            vicinitySSO.insertMatrixTranspose( vicinitySSO_pos[*node], vicinitySSO_pos[*neig], mmNeigSSO[*neig][*node] ); // Insert down-diagonal block
        }
//    cout << "SSO_2\n" << vicinitySSO << endl << endl;
      }
//    ASSERT_( isZeroDiag(vicinitySSO) );
//    ASSERT_( isSymmetrical(vicinitySSO) );
//    cout << "SSO\n" << vicinitySSO << endl;
      return vicinitySSO;
    }

    // Update vicinity information
    void ArrangeGraphSSO(const vector<mrpt::vector_uint> &parts,
                         mrpt::math::CMatrix &SSO,
                         const set<unsigned> &newNodes,
                         const set<unsigned> &previousNodes_,
                         map<unsigned, unsigned> &oldNodeStart)	// Update Map's vicinity & SSO matrices
    {
//    cout << "ArrangeGraphSSO...\n";
//
      set<unsigned> previousNodes = previousNodes_;
      previousNodes.erase(currentNode);

//      Eigen::MatrixXd SSO_reordered;
      mrpt::math::CMatrix SSO_reordered;
      std::vector<size_t> new_ordenation(0);
      // QUITAR
//      cout << "Old and new ordination of kfs in SSO";
      for( unsigned count = 0; count < parts.size(); count++ )
      {
        cout << "\nGroup: ";
        for( unsigned count2 = 0; count2 < parts[count].size(); count2++ )
          cout << parts[count][count2] << "\t";
        new_ordenation.insert( new_ordenation.end(), parts[count].begin(), parts[count].end() );
      }
      cout << "\nNew order: ";
      for( unsigned count = 0; count < new_ordenation.size(); count++ )
        cout << new_ordenation[count] << "\t";
      cout << endl;

      SSO.extractSubmatrixSymmetrical( new_ordenation, SSO_reordered ); // Update the own SSO matrices


      unsigned posNode1SSO = 0;
      // Update SSO of newNodes: a) Update internal SSO of nodes (diagonal blocks), b) Update connections of the newNodes
      for( set<unsigned>::iterator node1 = newNodes.begin(); node1 != newNodes.end(); node1++ )
      {
        cout << "New node " << *node1 << endl;
        vsNeighborPlaces[*node1].insert( *node1 );

      // a) Update SSO of neighbors
        vSSO[*node1].setSize( vsFramesInPlace[*node1].size(), vsFramesInPlace[*node1].size() );
        SSO_reordered.extractMatrix( posNode1SSO, posNode1SSO, vSSO[*node1] ); // Update the own SSO matrices

        // Get the most representative keyframe
        float sum_sso, highest_sum_sso = 0;
        unsigned most_representativeKF;
        set<unsigned>::iterator row_id = vsFramesInPlace[*node1].begin();
        for(unsigned row=0; row < vSSO[*node1].getRowCount(); row++, row_id++)
        {
          sum_sso = 0;
          for(unsigned col=0; col < vSSO[*node1].getRowCount(); col++)
            sum_sso += vSSO[*node1](row,col);
          if(sum_sso > highest_sum_sso)
          {
            most_representativeKF = *row_id;
            highest_sum_sso = sum_sso;
          }
        }
        selectedKeyframes[*node1] = most_representativeKF;

//        cout << "Extracted SSO\n" << vSSO[*node1] << endl;

        unsigned posNode2SSO = posNode1SSO + vsFramesInPlace[*node1].size();
        set<unsigned>::iterator node2 = node1;
        node2++;
        for( ; node2 != newNodes.end(); node2++ )
        {
          bool isNeighbor = false;
          for( unsigned frameN1 = 0; frameN1 < vsFramesInPlace[*node1].size(); frameN1++ )
            for( unsigned frameN2 = 0; frameN2 < vsFramesInPlace[*node2].size(); frameN2++ )
              if( SSO_reordered(posNode1SSO + frameN1, posNode2SSO + frameN2) > 0 ) // Graph connected
              {
              cout << "Neighbor submaps" << vsFramesInPlace[*node1].size() << " " << vsFramesInPlace[*node2].size() << "\n";
                vsNeighborPlaces[*node1].insert( *node2 );
                vsNeighborPlaces[*node2].insert( *node1 );

                mrpt::math::CMatrix SSOconnectNeigs( vsFramesInPlace[*node1].size(), vsFramesInPlace[*node2].size() );
                SSO_reordered.extractMatrix( posNode1SSO, posNode2SSO, SSOconnectNeigs ); // Update the own SSO matrices
              cout << "SSOconnectNeigs\n" << SSOconnectNeigs << endl;

                mmNeigSSO[*node1][*node2] = SSOconnectNeigs;

                isNeighbor = true;
                frameN1 = vsFramesInPlace[*node1].size();    // Force exit of nested for loop
                break;
              }
          if(!isNeighbor)
            if(vsNeighborPlaces[*node1].count(*node2) != 0 )
            {
    //        cout << "Entra con " << *node1 << " y " << *node2 << endl;
              vsNeighborPlaces[*node1].erase(*node2);
              mmNeigSSO[*node1].erase(*node2);
              vsNeighborPlaces[*node2].erase(*node1);
            }
          posNode2SSO += vsFramesInPlace[*node2].size();
        }
        posNode1SSO += vsFramesInPlace[*node1].size();
      }

      // Update the newNodes submaps' KFdistribution
//      ArrangeKFdistributionSSO( newNodes );

      // Create list of 2nd order neighbors and make copy of the interSSO that are going to change
      map<unsigned, map<unsigned, mrpt::math::CMatrix> > prevNeigSSO;
      for( set<unsigned>::iterator node = previousNodes.begin(); node != previousNodes.end(); node++ )
        for( map<unsigned, mrpt::math::CMatrix>::iterator neigNeig = mmNeigSSO[*node].begin(); neigNeig != mmNeigSSO[*node].end(); neigNeig++ )
          if( newNodes.count(neigNeig->first) == 0 ) // If neigNeig is not in the vicinity
          {
            if(neigNeig->first < *node)
            {
              prevNeigSSO[neigNeig->first][*node] = mmNeigSSO[neigNeig->first][*node];
              mmNeigSSO[neigNeig->first].erase(*node);
            }
            else
            {
              prevNeigSSO[neigNeig->first][*node] = mmNeigSSO[neigNeig->first][*node].transpose();
              mmNeigSSO[*node].erase(neigNeig->first);
            }
            vsNeighborPlaces[neigNeig->first].erase(*node);
            vsNeighborPlaces[*node].erase(neigNeig->first);
          cout << "Copy 2nd order relation between 2nd " << neigNeig->first << " and " << *node << " whose size is " << vsFramesInPlace[*node].size() << endl;
          }

    cout << "Update the SSO interconnections with 2nd order neighbors\n";
      // Update the SSO interconnections with 2nd order neighbors (the 2nd order neighbors is referred by the map's first element, and the 1st order is referred in its second element)
    //  map<unsigned, map<unsigned, mrpt::math::CMatrix> > newInterSSO;
      for( map<unsigned, map<unsigned, mrpt::math::CMatrix> >::iterator neig2nd = prevNeigSSO.begin(); neig2nd != prevNeigSSO.end(); neig2nd++ )
      {
      cout << "Analyse neig " << neig2nd->first << endl;
        map<unsigned, mrpt::math::CMatrix> &interSSO2nd = neig2nd->second;
        mrpt::math::CMatrix subCol( vsFramesInPlace[neig2nd->first].size(), 1 );

        // Search for all relations of current neighbors with previous neighbors
        for( map<unsigned, mrpt::math::CMatrix>::iterator neig1st = interSSO2nd.begin(); neig1st != interSSO2nd.end(); neig1st++ )
        {
      cout << " with neighbor " << neig1st->first << endl;
          // Check for non-zero elements
          for( unsigned KF_pos1st = 0; KF_pos1st < vsFramesInPlace[neig1st->first].size(); KF_pos1st++ )
          {
            for( unsigned KF_pos2nd = 0; KF_pos2nd < vsFramesInPlace[neig2nd->first].size(); KF_pos2nd++ )
            {
              if( KF_pos2nd >= neig1st->second.getRowCount() || KF_pos1st >= neig1st->second.getColCount() )
                assert(false);
//                cout << "Matrix dimensions " << neig1st->second.getRowCount() << "x" << neig1st->second.getColCount() << " trying to access " << KF_pos2nd << " " << KF_pos1st << endl;
              if( neig1st->second(KF_pos2nd,KF_pos1st) > 0 )
              {
                // Extract subCol
                neig1st->second.extractMatrix(0,KF_pos1st,subCol);

                // Search where is now that KF to insert the extracted column
                unsigned KF_oldPos = oldNodeStart[neig1st->first] + KF_pos1st;
                bool found = false;
                for( unsigned new_node = 0; new_node < parts.size(); new_node++ ) // Search first in the same node
                 for( unsigned KF_newPos = 0; KF_newPos < parts[new_node].size(); KF_newPos++ ) // Search first in the same node
                 {
                  if( KF_oldPos == KF_newPos )
                  {
                    vsNeighborPlaces[neig2nd->first].insert(neig1st->first);
                    vsNeighborPlaces[neig1st->first].insert(neig2nd->first);

                    if( neig2nd->first < neig1st->first )
                    {
                      if( mmNeigSSO[neig2nd->first].count(neig1st->first) == 0 )
                        mmNeigSSO[neig2nd->first][neig1st->first].zeros( vsFramesInPlace[neig2nd->first].size(), vsFramesInPlace[neig1st->first].size() );

                      mmNeigSSO[neig2nd->first][neig1st->first].insertMatrix( 0, KF_newPos, subCol );
                    }
                    else
                    {
                      if( mmNeigSSO[neig1st->first].count(neig2nd->first) == 0 )
                        mmNeigSSO[neig1st->first][neig2nd->first].zeros( vsFramesInPlace[neig1st->first].size(), vsFramesInPlace[neig2nd->first].size() );

                      mmNeigSSO[neig1st->first][neig2nd->first].insertMatrix( KF_newPos, 0, subCol.transpose() );
                    }
    //                interSSO2nd[neig1st->first].insertMatrix( 0, KF_newPos, subCol );
                    new_node = parts.size();
                    found = true;
                    break;
                  }
                 }
                 assert(found);
                break;
              }
            }
          }
        }
      cout << "InterSSO calculated\n";

      }

      cout << "Finish ArrangeGraphSSO\n";
    }

    void RearrangePartition( vector<mrpt::vector_uint> &parts )	// Arrange the parts given by the partition
    {
    cout << "RearrangePartition...\n";
      vector<mrpt::vector_uint> parts_swap;
      parts_swap.resize( parts.size() );
      for( unsigned int countPart1 = 0; countPart1 < parts_swap.size(); countPart1++ )	// First, we arrange the parts regarding its front element (for efficiency in the next step)
      {
        unsigned int smallestKF = 0;
        for( unsigned int countPart2 = 1; countPart2 < parts.size(); countPart2++ )
          if( parts[smallestKF].front() > parts[countPart2].front() )
            smallestKF = countPart2;
        parts_swap[countPart1] = parts[smallestKF];
        parts.erase( parts.begin() + smallestKF );
      }
      parts = parts_swap;
    }

    // Calculate Partitions of the current map(s). Arrange KFs and points of updated partition
    void Partitioner()
    {
    cout << "Partitioner...\n";
      mrpt::utils::CTicTac time;	// Clock to measure performance times
      time.Tic();

      unsigned numPrevNeigNodes = vsNeighborPlaces[currentNode].size();
      vector<mrpt::vector_uint> parts;  // Vector of vectors to keep the KFs index of the different partitions (submaps)
      parts.reserve( numPrevNeigNodes+5 );	// We reserve enough memory for the eventual creation of more than one new map	//Preguntar a JL -> Resize o reserve? // Quitar // Necesario?
//    cout << "getVicinitySSO(vsNeighborPlaces[currentNode])...\n";
//    cout << "currentNode " << currentNode << ". Neigs:";
//    for( set<unsigned>::iterator node = vsNeighborPlaces[currentNode].begin(); node != vsNeighborPlaces[currentNode].end(); node++ )
//      cout << " " << *node;
//    cout << endl;

      mrpt::math::CMatrix SSO = getVicinitySSO(vsNeighborPlaces[currentNode]);
//      cout << "SSO\n" << SSO << endl;
      if(SSO.getRowCount() < 3)
        return;

      mrpt::graphs::CGraphPartitioner<mrpt::math::CMatrix>::RecursiveSpectralPartition(SSO, parts, 0.2, false, true, true, 3);
    cout << "Time RecursiveSpectralPartition " << time.Tac()*1000 << "ms" << endl;

      int numberOfNewMaps = parts.size() - numPrevNeigNodes;
    cout <<"numPrevNeigNodes " << numPrevNeigNodes << " numberOfNewMaps " << numberOfNewMaps << endl;

      if( numberOfNewMaps == 0 ) //|| !ClosingLoop ) // Restructure the map only if there isn't a loop closing right now
        return;

      RearrangePartition( parts );	// Arrange parts ordination to reduce computation in next steps

//    { mrpt::synch::CCriticalSectionLocker csl(&CS_RM_T); // CRITICAL SECTION Tracker
  cout << "Rearrange map CS\n";

//    cout << "Traza1.b\n";
      if( numberOfNewMaps > 0 )
      {
    //	// Show previous KF* ordination
        cout << "\nParts:\n";
        for( unsigned counter_part=0; counter_part < parts.size(); counter_part++ )	// Search for the new location of the KF
        {
          for( unsigned counter_KFpart=0; counter_KFpart < parts[counter_part].size(); counter_KFpart++)
            cout << parts[counter_part][counter_KFpart] << " ";
        }

//    cout << "Traza1.2\n";

        std::vector<unsigned> prevDistributionSSO;
        std::map<unsigned, unsigned> oldNodeStart;
        std::map<unsigned, set<unsigned> > vsFramesInPlace_prev;
        unsigned posInSSO = 0;
        for( set<unsigned>::iterator node = vsNeighborPlaces[currentNode].begin(); node != vsNeighborPlaces[currentNode].end(); node++ )
        {
          vsFramesInPlace_prev[*node] = vsFramesInPlace[*node];
          oldNodeStart[*node] = posInSSO;
          posInSSO += vsFramesInPlace[*node].size();
          for( set<unsigned>::iterator frame = vsFramesInPlace[*node].begin(); frame != vsFramesInPlace[*node].end(); frame++ )
            prevDistributionSSO.push_back(*frame);
        }

    //    vector<unsigned> new_DistributionSSO;

        set<unsigned> previousNodes = vsNeighborPlaces[currentNode];
        set<unsigned> newNodes = previousNodes;

        for( int NumNewMap = 0; NumNewMap < numberOfNewMaps; NumNewMap++ )	// Should new maps be created?
        {
          newNodes.insert(vsFramesInPlace.size());
          mmNeigSSO[vsFramesInPlace.size()] = std::map<unsigned, mrpt::math::CMatrix>();
          vsNeighborPlaces.push_back( std::set<unsigned>() );
          vsFramesInPlace.push_back( std::set<unsigned>() );
          mrpt::math::CMatrix sso(parts[previousNodes.size()+NumNewMap].size(),parts[previousNodes.size()+NumNewMap].size());
          vSSO.push_back(sso);

          // Set the local reference system of this new node
    //      unsigned size_append = 0;
        }

      cout << "Traza1.3a\n";

//        cout << "previousNodes: ";
//        for( set<unsigned>::iterator node = previousNodes.begin(); node != previousNodes.end(); node++ )
//          cout << *node << " ";
//        cout << "vsNeighborPlaces[currentNode]: ";
//        for( set<unsigned>::iterator node = vsNeighborPlaces[currentNode].begin(); node != vsNeighborPlaces[currentNode].end(); node++ )
//          cout << *node << " ";
//        cout << endl;

//      cout << "Traza1.4\n";
        // Rearrange the KF in the maps acording to the new ordination
        unsigned counter_SSO_KF = 0;
        unsigned actualFrame = prevDistributionSSO[counter_SSO_KF];
//        vector<unsigned> insertedKFs(parts.size(), 0);
        unsigned counterNode = 0;
        for( set<unsigned>::iterator node = previousNodes.begin(); node != previousNodes.end(); node++ )// For all the maps of the previous step
        {
//      cout << "Traza1.5a\n";
          cout << "NumNode: " << *node << " previous KFs: " << vsFramesInPlace[*node].size() << endl;
    //      unsigned counter_KFsamePart = insertedKFs[counterNode];??
          unsigned counter_KFsamePart = 0;
          for( set<unsigned>::iterator itFrame = vsFramesInPlace_prev[*node].begin(); itFrame != vsFramesInPlace_prev[*node].end(); itFrame++ )	//For all the keyframes of this map (except for the those transferred in this step
          {
            cout << "Check KF " << *itFrame << "th of map " << *node << " which has KFs " << vsFramesInPlace[*node].size() << endl;
            while ( counter_KFsamePart < parts[counterNode].size() && counter_SSO_KF > parts[counterNode][counter_KFsamePart] )
              ++counter_KFsamePart;	// It takes counter_KFpart to the position where it can find its corresponding KF index (increase lightly the efficiency)

            if( counter_KFsamePart < parts[counterNode].size() && counter_SSO_KF == parts[counterNode][counter_KFsamePart] )	//Check if the KF has switched to another map (For efficiency)
            {
    //          new_DistributionSSO.push_back(actualFrame);
              ++counter_KFsamePart;
              cout << "KF " << counter_SSO_KF << endl;
            }
            else	// KF switches
            {
              bool found = false;
              set<unsigned>::iterator neigNode = newNodes.begin();
              for( unsigned counter_part=0; counter_part < parts.size(); counter_part++, neigNode++ )	// Search for the new location of the KF
              {
                cout << "counter_part " << counter_part << " neigNode " << *neigNode << endl;
                if( counter_part == counterNode )
                  { continue;}	// Do not search in the same map
                for( unsigned counter_KFpart = 0; counter_KFpart < parts[counter_part].size(); counter_KFpart++)
                {
                  if( counter_SSO_KF < parts[counter_part][counter_KFpart] )  // Skip part when the parts[counter_part][counter_KFpart] is higher than counter_SSO_KF
                    break;
                  if( counter_SSO_KF == parts[counter_part][counter_KFpart] )	// Then transfer KF* and its points to the corresponding map
                  {
      cout << "aqui1\n";
                    found = true;

    //                SE3<> se3NewFromOld = foundMap.se3MapfromW * thisMap.se3MapfromW.inverse();
                    vsFramesInPlace[*node].erase(actualFrame);
                    vsFramesInPlace[*neigNode].insert(actualFrame);
                    spheres[actualFrame]->node = *neigNode;
    //                spheres[actualFrame]->node = *neigNode;

                    counter_part = parts.size();	// This forces to exit both for loops
                    break;
                  }// End if
                }// End for
              }// End for
              if(!found)
              {
                cout << "\n\nWarning1: KF lost\n Looking for KF " << counter_SSO_KF << "\nParts:\n";
                for( unsigned counter_part=0; counter_part < parts.size(); counter_part++ )	// Search for the new location of the KF
                {
                  for( unsigned counter_KFpart=0; counter_KFpart < parts[counter_part].size(); counter_KFpart++)
                    cout << parts[counter_part][counter_KFpart] << " ";
//                  cout << " insertedKFs " << insertedKFs[counter_part] << endl;
                }
                assert(false);
              }
            } // End else
            ++counter_SSO_KF;
            actualFrame = prevDistributionSSO[counter_SSO_KF];
          } // End for
          ++counterNode;
        } // End for

        for( set<unsigned>::iterator node = newNodes.begin(); node != newNodes.end(); node++ )// For all the maps of the previous step
          cout << "Node " << *node << "numKFs " << vsFramesInPlace[*node].size() << endl;

        ArrangeGraphSSO( parts, SSO, newNodes, previousNodes, oldNodeStart );	// Update neighbors and SSO matrices of the surrounding maps

        // Search for the part containing the last KF to set it as gMap
        unsigned nextCurrentNode, highestKF = 0;
        int part_count=0;
        for( set<unsigned>::iterator node = newNodes.begin(); node != newNodes.end(); node++, part_count++ )
        {
          cout << "Last of " << *node << " is " << parts[part_count].back() << endl;
          if( parts[part_count].back() > highestKF )
          {
            nextCurrentNode = *node;
            highestKF = parts[part_count].back();
          }
        }
        cout << "nextCurrentNode " << nextCurrentNode << endl;
        currentNode = nextCurrentNode;

        // Update current pose


      cout << "critical section " << time.Tac()*1000 << " ms\n";

      } // End if( numberOfNewMaps > 0 )
  //    cout << "Map Rearranged CS\n";

    }

    void run(string path)
    {
      std::cout << "RegisterGraphSphere::run... " << std::endl;

//      unsigned frame = 0;
      frame = 0;
      int frameOrder = 0;
      int numCheckRegistration = 5; // Check 20 frames backwards registration
      unsigned noAssoc_threshold = 40; // Limit the number of correspondences that are checked backwards when noAssoc_threshold are not associated

      // Partition
      currentNode = 0;
      vsFramesInPlace.push_back( std::set<unsigned>() );
      vsFramesInPlace[currentNode].insert(frameOrder);
//      std::set<unsigned> neigCurrent;
//      neigCurrent.insert(currentNode);
      vsNeighborPlaces.push_back( std::set<unsigned>() );	// Vector with numbers of neighbor topologial nodes
      vsNeighborPlaces[currentNode].insert(currentNode);
      vSSO.push_back( mrpt::math::CMatrix(1,1) );
      vSSO[currentNode](frameOrder,frameOrder) = 0.0;

      string fileName = path + mrpt::format("/sphere_images_%d.bin",frame); //436

      // Load first frame
      Frame360* frame360 = new Frame360(&calib);
      frame360->loadFrame(fileName);
      frame360->undistort();
//        frame360->stitchSphericalImage();
      frame360->buildSphereCloud();
      frame360->getPlanes();
      frame360->id = frameOrder;
      frame360->node = currentNode;
      spheres.push_back(frame360);
      selectedKeyframes[0] = 0;
      trajectory_increment.push_back(0);

      ++frameOrder;
      frame += 1;
      fileName = path + mrpt::format("/sphere_images_%d.bin",frame); //436


      pcl::PassThrough<PointT> filter_pass_x, filter_pass_y, filter_pass_z;
      filter_pass_x.setFilterFieldName ("x");
      filter_pass_x.setFilterLimits (-2.0, 1.0);
      filter_pass_z.setFilterFieldName ("z");
      filter_pass_z.setFilterLimits (-4.0, 4.0);
      filter_pass_y.setFilterFieldName ("y");
      filter_pass_y.setFilterLimits (-4.0, 4.0);
      pcl::VoxelGrid<pcl::PointXYZRGBA> filter_voxel;
      filter_voxel.setLeafSize(0.05,0.05,0.05);

      // Initialize the global map with the first observation
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filteredCloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);
      filter_pass_y.setInputCloud (frame360->sphereCloud);
      filter_pass_y.filter (*filteredCloud);
      filter_pass_z.setInputCloud (filteredCloud);
      filter_pass_z.filter (*filteredCloud2);
      filter_pass_x.setInputCloud (filteredCloud2);
      filter_pass_x.filter (*frame360->sphereCloud);

//      vector<Eigen::Vector2f> trajectory2D(1, Eigen::Vector2f(300,300) );
//      cv::Mat canvas_trajectory = cv::Mat::zeros(600,600,CV_8U);
//      cv::Scalar white(255), grey(185);
//      cv::Point center(trajectory2D.back()(0),trajectory2D.back()(1));
//      cv::circle(canvas_trajectory, center, 4, white);
//      vector<Eigen::Vector4f> wallsOnGround = getVerticalPlanes(frame360->planes);
//      for(unsigned i=0; i<wallsOnGround.size(); i++)
//        cv::line(canvas_trajectory, cv::Point(wallsOnGround[i](0)*30+300,wallsOnGround[i](1)*30+300),
//                                    cv::Point(wallsOnGround[i](2)*30+300,wallsOnGround[i](3)*30+300), grey);
////      cv::imwrite("../../test.png", canvas_trajectory);

//      pcl::visualization::PCLVisualizer viewer("RGBD360");
      pcl::visualization::CloudViewer viewer("RGBD360");
      viewer.runOnVisualizationThread (boost::bind(&RegisterGraphSphere::viz_cb, this, _1), "viz_cb"); //PCL viewer. It runs in a different thread.
      viewer.registerKeyboardCallback(&RegisterGraphSphere::keyboardEventOccurred, *this);

  //  boost::this_thread::sleep (boost::posix_time::milliseconds (1000));
      float trajectory_length = 0;
      Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();
      trajectory_poses.push_back(currentPose);
      trajectory_pose_increments.push_back(currentPose);

      // Graph-SLAM
      float areaMatched_src;
      GraphOptimizer_MRPT optimizer;
      optimizer.setRigidTransformationType(GraphOptimizer::SixDegreesOfFreedom);
      std::cout << "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;

      while( fexists(fileName.c_str()) )
//      while( frame < 171 )
      {
      cout << "Frame " << fileName << endl;

        // Load pointCloud
        Frame360* frame360 = new Frame360(&calib);
        frame360->loadFrame(fileName);
        frame360->undistort();
//        frame360->stitchSphericalImage();
        frame360->buildSphereCloud();
        frame360->getPlanes();
        frame360->id = frameOrder;
        frame360->node = currentNode;

//        spheres.push_back(frame360);
//        vsFramesInPlace[currentNode].insert(frameOrder);
        int newLocalFrameID = vsFramesInPlace[currentNode].size();
        int newSizeLocalSSO = newLocalFrameID+1;
        vSSO[currentNode].setSize(newSizeLocalSSO,newSizeLocalSSO);
        vSSO[currentNode](newLocalFrameID,newLocalFrameID) = 0.0;

        // Search for connections in the previous frames (detect inconsistencies)
        int compareLocalIdx = newLocalFrameID-1;
        unsigned noAssoc= 0;

        connections[frameOrder] = map<unsigned, pair<unsigned,Eigen::Matrix4f> >();
        bool frameRegistered = false;
//        while(compareLocalIdx >= 0 && (compareLocalIdx >= newLocalFrameID-numCheckRegistration) && noAssoc < noAssoc_threshold)
        set<unsigned>::reverse_iterator compareSphereId = vsFramesInPlace[currentNode].rbegin();
        while(compareLocalIdx >= 0 && (compareLocalIdx >= newLocalFrameID-numCheckRegistration) && noAssoc < noAssoc_threshold)
        {
          Eigen::Matrix4f rigidTransf = RegisterOdometry(spheres[*compareSphereId], frame360, areaMatched_src);
        cout << "areaMatched_src " << areaMatched_src << endl;
          if(rigidTransf != Eigen::Matrix4f::Identity())
          {
            if(!frameRegistered)
            {
              currentPose = currentPose * rigidTransf;
//              Eigen::Matrix4f currentPoseInv = currentPose.inverse();
              std::cout<< "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;
//              std::cout<< "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;
//              trajectory_poses.push_back(currentPose);
              trajectory_increment.push_back(rigidTransf.block(0,3,3,1).norm());
              trajectory_length += rigidTransf.block(0,3,3,1).norm();
  //        Eigen::Vector3f translation_ = rigidTransf.block(0,3,3,1);
  //        trajectory_length += translation_.norm();
              frameRegistered = true;
//            }

//            // Compare pose with currentPose (which is assumed to be good) to detect wrong registration
//            Eigen::Matrix4f currentPoseLink = trajectory_poses[*compareSphereId] * rigidTransf;
//            float euclidean_dist = (currentPoseLink.block(0,3,3,1) - currentPose.block(0,3,3,1)).norm();
//            if(euclidean_dist > 0.12)
//            {
//              --compareLocalIdx;
//              ++compareSphereId;
//              continue;
//            }
//
//            mrpt::math::CMatrixDouble44 cmat_current(currentPose);
//            mrpt::math::CMatrixDouble44 cmat_currentLink(currentPoseLink);
//            mrpt::poses::CPose3D pose_current(cmat_current);
//            mrpt::poses::CPose3D pose_currentLink(cmat_currentLink);
//            mrpt::math::CArrayDouble<3> rot_diff = pose_current.ln_rotation() - pose_currentLink.ln_rotation();
//            float angle_diff = mrpt::math::norm(rot_diff);
//            if(angle_diff > 0.12)
//            {
//              --compareLocalIdx;
//              ++compareSphereId;
//              continue;
//            }
////            cout << "euclidean_dist " << euclidean_dist << " angle_diff " << angle_diff << endl;

            trajectory_pose_increments.push_back(rigidTransf);

            connections[frameOrder][*compareSphereId] = pair<unsigned,Eigen::Matrix4f>(areaMatched_src, rigidTransf);
            vSSO[currentNode](newLocalFrameID,compareLocalIdx) = vSSO[currentNode](compareLocalIdx,newLocalFrameID) = calculateSSO(spheres[compareLocalIdx], areaMatched_src);
//            cout << "SSO is " << vSSO[currentNode](newLocalFrameID,compareLocalIdx) << endl;
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = areaMatched_src * Eigen::Matrix<double,6,6>::Identity();
//          std::cout << "informationMatrix \n" << informationMatrix << endl;
            optimizer.addEdge(compareLocalIdx,newLocalFrameID,rigidTransf,informationMatrix);
            std::cout << "Add edge between " << compareLocalIdx << " and " << newLocalFrameID << "\n";

            break;
            }
          }
          else
            ++noAssoc;

          --compareLocalIdx;
          ++compareSphereId; // Increase the reversed iterator
        }
        if(!frameRegistered) //
        {
          cout << "No registration available for " << fileName << endl;

          vSSO[currentNode].setSize(newLocalFrameID,newLocalFrameID);

          fileName = path + mrpt::format("/sphere_images_%d.bin",++frame);

          continue;
//          assert(false);
        }
        // Calculate distance of furthest registration (approximated, we do not know if the last frame compared is actually the furthest)
        map<unsigned, pair<unsigned,Eigen::Matrix4f> >::iterator itFurthest = connections[frameOrder].begin(); // Aproximated
//        Eigen::Vector3f translation_ = itFurthest->second.block(0,3,3,1);
        cout << "Furthest distance is " << itFurthest->second.second.block(0,3,3,1).norm() << " diffFrameNum " << newLocalFrameID - itFurthest->first << endl;

        // Filter cloud
        filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
        filteredCloud2.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
        filter_pass_y.setInputCloud (frame360->sphereCloud);
        filter_pass_y.filter (*filteredCloud);
        filter_pass_z.setInputCloud (filteredCloud);
        filter_pass_z.filter (*filteredCloud2);
        filter_pass_x.setInputCloud (filteredCloud2);
        filter_pass_x.filter (*frame360->sphereCloud);

        spheres.push_back(frame360);
        vsFramesInPlace[currentNode].insert(frameOrder);


        filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
        filter_voxel.setInputCloud (frame360->sphereCloud);
        filter_voxel.filter (*filteredCloud);
        frame360->sphereCloud = filteredCloud;

        {boost::mutex::scoped_lock updateLock(visualizationMutex);
            trajectory_poses.push_back(currentPose);
        }


        // Search for loop closure constraints in nearby frames (excluding the inmediate previous)
        compareLocalIdx -= 40;
        for(; compareLocalIdx >= 0; compareLocalIdx-=3)
        {
//        cout << "Check loop closure\n";
          float distance = (currentPose.block(0,3,3,1) - trajectory_poses[compareLocalIdx].block(0,3,3,1)).norm();
          if( distance < 0.2*(trajectory_length - trajectory_increment[compareLocalIdx]) )
          {
          cout << "Check loop closure between " << frameOrder << " " << compareLocalIdx << "\n";
            areaMatched_src = 0;
            unsigned numMatched = 0;
            Eigen::Matrix4f rigidTransf = Register(spheres[compareLocalIdx], frame360, numMatched, areaMatched_src);
            if(numMatched > 8 && areaMatched_src > 20 && rigidTransf != Eigen::Matrix4f::Identity())
            {
              cout << "Loop closed between " << newLocalFrameID << " and " << compareLocalIdx << " matchedArea " << areaMatched_src << endl;
//            mrpt::system::pause();
              connections[newLocalFrameID][compareLocalIdx] = pair<unsigned,Eigen::Matrix4f>(areaMatched_src, rigidTransf);
              vSSO[currentNode](newLocalFrameID,compareLocalIdx) = vSSO[currentNode](newLocalFrameID,compareLocalIdx) = calculateSSO(spheres[compareLocalIdx], areaMatched_src);
            cout << "SSO is " << vSSO[currentNode](newLocalFrameID,compareLocalIdx) << endl;
              Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = areaMatched_src * Eigen::Matrix<double,6,6>::Identity();
//            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = Eigen::Matrix<double,6,6>::Identity() / areaMatched_src;
//          std::cout << "informationMatrix \n" << informationMatrix << endl;
              optimizer.addEdge(compareLocalIdx,newLocalFrameID,rigidTransf,informationMatrix);
            std::cout << "Add edge between " << compareLocalIdx << " and " << newLocalFrameID << "\n";
              break;
//              assert(false); // Check if OK
            }
          }
        }

        // Optimize graphSLAM g2o
        if( connections[newLocalFrameID].size() > 1 )
        {
        cout << "Optimize graph SLAM \n";
          //Optimize the graph
          optimizer.optimizeGraph();
          //Update the optimized poses (for loopClosure detection & visualization)
          {
            cout << "Get optimized poses \n";
            boost::mutex::scoped_lock updateLock(visualizationMutex);
            optimizer.getPoses(optimized_poses);
            cout << "Get optimized poses " << optimized_poses.size() << "\n";
//            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > vOptimizedRelativePoses;
//            optimizer.getPoses(vOptimizedRelativePoses);
//            cout << "First pose optimized \n" << vOptimizedRelativePoses[0] << endl << trajectory_pose_increments[0] << endl;
//            cout << "First pose optimized \n" << vOptimizedRelativePoses[0] << endl << trajectory_poses[0] << endl;
////            Eigen::Matrix4f poseComposition = vOptimizedRelativePoses[0];
//            optimized_poses.resize(vOptimizedRelativePoses.size());
//            optimized_poses[0] = vOptimizedRelativePoses[0];
//            for(unsigned i=1; i < vOptimizedRelativePoses.size(); i++)
//            {
//              cout << "Distances " << vOptimizedRelativePoses[i].block(0,3,3,1).norm() << " " << trajectory_poses[i].block(0,3,3,1).norm() << " " << trajectory_increment[i] << endl;
//              optimized_poses[i] = optimized_poses[i-1] * vOptimizedRelativePoses[i];
//            }
          }
//          for(unsigned i=0; i < optimized_poses.size(); i++)
//            cout << "Pose " << i << endl << optimized_poses[i] << endl << trajectory_poses[i] << endl;

          //Update the optimized pose
//          currentPose = optimized_poses.back();
//            mrpt::system::pause();
        }

//        // Visibility divission (Normalized-cut)
//        if(newLocalFrameID % 5 == 0)
//        {
//          cout << "Eval partitions\n";
//          Partitioner();
//
//          cout << "\tPlaces\n";
//          for( unsigned node = 0; node < vsNeighborPlaces.size(); node++ )
//          {
//            cout << "\t" << node << ":";
//            for( set<unsigned>::iterator it = vsNeighborPlaces[node].begin(); it != vsNeighborPlaces[node].end(); it++ )
//              cout << " " << *it;
//            cout << endl;
//          }
//        }

        bFreezeFrame = false;

        ++frameOrder;
        frame += 1;
        fileName = path + mrpt::format("/sphere_images_%d.bin",frame);
      }

      while (!viewer.wasStopped() )
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));

      cout << "Path length " << trajectory_length << endl;
    }

  private:

    boost::mutex visualizationMutex;

    Calib360 calib;
    std::vector<Frame360*> spheres;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > trajectory_poses;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > trajectory_pose_increments;
    std::vector<float> trajectory_increment;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > optimized_poses;
    map<unsigned, map<unsigned, pair<unsigned,Eigen::Matrix4f> > > connections;

//    mrpt::pbmap::config_heuristics configLocaliser;

    pcl::PointCloud<PointT>::Ptr globalMap;

    map<unsigned,unsigned> selectedKeyframes;

    pcl::PointCloud<PointT>::Ptr registrationClouds[2];

    bool bFreezeFrame;
    unsigned frameRecNum;

    unsigned currentNode;
    std::vector<mrpt::math::CMatrix> vSSO;
    std::vector<std::set<unsigned> > vsFramesInPlace;	// Vector with index of Frames in each topologial node
    std::vector<std::set<unsigned> > vsNeighborPlaces;	// Vector with index of neighbor topologial nodes
    std::map<unsigned, std::map<unsigned, mrpt::math::CMatrix> > mmNeigSSO;    // A sub-map also contains the SSO matrix w.r.t. its neighbors (with bigger index)

    std::vector<pcl::PointXYZ> spheres_centers;

    void viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
//    cout << "SphericalSequence::viz_cb(...)\n";
      if (spheres[0]->sphereCloud->empty() || bFreezeFrame)
      {
        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
        return;
      }
//    cout << "   ::viz_cb(...)\n";

//      viz.setFullscreen(true);
      viz.removeAllShapes();
      viz.removeAllPointClouds();
      viz.removeCoordinateSystem();

      { //mrpt::synch::CCriticalSectionLocker csl(&CS_visualize);
        boost::mutex::scoped_lock updateLock(visualizationMutex);

//        if (!viz.updatePointCloud (globalMap, "sphereCloud"))
//          viz.addPointCloud (globalMap, "sphereCloud");

//        if (!viz.updatePointCloud (registrationClouds[0], "sphereCloud0"))
//          viz.addPointCloud (registrationClouds[0], "sphereCloud0");
//
//        if (!viz.updatePointCloud (registrationClouds[1], "sphereCloud1"))
//          viz.addPointCloud (registrationClouds[1], "sphereCloud1");

        char name[1024];

        sprintf (name, "Frame %u. Graph-SLAM %d", frame, bGraphSLAM ? 1 : 0);
        viz.addText (name, 20, 20, "params");

        // Draw map and trajectory
        pcl::PointXYZ pt_center;//, pt_center_prev;
        if(!bGraphSLAM)
        {
//    cout << "   ::viz_cb(...)\n";

//          unsigned lastFrame = trajectory_poses.size()-1;
          spheres_centers.resize(trajectory_poses.size());
  //        for(unsigned i=0; i < spheres.size(); i++)
          for(unsigned i=0; i < trajectory_poses.size(); i++)
          {
            sprintf (name, "cloud%u", i);
            if (!viz.updatePointCloud (spheres[i]->sphereCloud, name))
              viz.addPointCloud (spheres[i]->sphereCloud, name);
            Eigen::Affine3f Rt;
            Rt.matrix() = trajectory_poses[i];
            viz.updatePointCloudPose(name, Rt);

            pt_center = pcl::PointXYZ(trajectory_poses[i](0,3), trajectory_poses[i](1,3), trajectory_poses[i](2,3));
            spheres_centers[i] = pt_center;
            sprintf (name, "pose%u", i);
//            viz.addSphere (pt_center, 0.04, 1.0, 0.0, 0.0, name);
            viz.addSphere (pt_center, 0.04, ared[spheres[i]->node%10], agrn[spheres[i]->node%10], ablu[spheres[i]->node%10], name);

            // Draw links
            for(map<unsigned, pair<unsigned,Eigen::Matrix4f> >::iterator it=connections[i].begin(); it != connections[i].end(); it++)
            {
              sprintf (name, "link%u_%u", i, it->first);
              viz.addLine (pt_center, spheres_centers[it->first], name);
            }
//            if(i > 0)
//            {
//              map<unsigned, pair<unsigned,Eigen::Matrix4f > >::reverse_iterator it = connections[i].rbegin();
////              cout << "frame " << trajectory_poses.size()-1 << " link " << it->first << endl;
//              sprintf (name, "link%u_%u", i, it->first);
//              viz.addLine (pt_center, spheres_centers[it->first], name);
//            }

            sprintf (name, "%u", i);
            pt_center.x += 0.05;
            viz.addText3D (name, pt_center, 0.05, ared[spheres[i]->node%10], agrn[spheres[i]->node%10], ablu[spheres[i]->node%10], name);
          }

//          for(map<unsigned,unsigned>::iterator node=selectedKeyframes.begin(); node != selectedKeyframes.end(); node++)
//          {
//            pt_center = pcl::PointXYZ(trajectory_poses[node->second](0,3), trajectory_poses[node->second](1,3), trajectory_poses[node->second](2,3));
//            sprintf (name, "poseKF%u", node->first);
////            viz.addSphere (pt_center, 0.04, 1.0, 0.0, 0.0, name);
//            viz.addSphere (pt_center, 0.1, ared[node->first%10], agrn[node->first%10], ablu[node->first%10], name);
//
//            sprintf (name, "KF%u", node->second);
//            if (!viz.updatePointCloud (spheres[node->second]->sphereCloud, name))
//              viz.addPointCloud (spheres[node->second]->sphereCloud, name);
//            Eigen::Affine3f Rt;
//            Rt.matrix() = trajectory_poses[node->second];
//            viz.updatePointCloudPose(name, Rt);
//          }

        }
        else
        {
//    cout << "   ::viz_cb(...)\n";
          spheres_centers.resize(optimized_poses.size());
          for(unsigned i=0; i < optimized_poses.size(); i++)
          {
            sprintf (name, "cloud%u", i);
            if (!viz.updatePointCloud (spheres[i]->sphereCloud, name))
              viz.addPointCloud (spheres[i]->sphereCloud, name);
            Eigen::Affine3f Rt;
//            Rt.matrix() = optimized_poses[i].inv();
            Rt.matrix() = optimized_poses[i];
            viz.updatePointCloudPose(name, Rt);

            pt_center = pcl::PointXYZ(optimized_poses[i](0,3), optimized_poses[i](1,3), optimized_poses[i](2,3));
            spheres_centers[i] = pt_center;
            sprintf (name, "pose%u", i);
            viz.addSphere (pt_center, 0.04, ared[spheres[i]->node%10], agrn[spheres[i]->node%10], ablu[spheres[i]->node%10], name);

            // Draw links
            for(map<unsigned, pair<unsigned,Eigen::Matrix4f> >::iterator it=connections[i].begin(); it != connections[i].end(); it++)
            {
              sprintf (name, "link%u_%u", i, it->first);
              viz.addLine (pt_center, spheres_centers[it->first], name);
            }

            sprintf (name, "%u", i);
            pt_center.x += 0.05;
            viz.addText3D (name, pt_center, 0.05, ared[spheres[i]->node%10], agrn[spheres[i]->node%10], ablu[spheres[i]->node%10], name);
          }
        }

        bFreezeFrame = true;

        #if RECORD_VIDEO
          string frameRecord = mrpt::format("/home/edu/Videos/im_%04u.png",frameRecNum);
          viz.saveScreenshot (frameRecord);
          ++frameRecNum;
        #endif

      updateLock.unlock();
      }
    }

    bool bTakeKeyframe;
    bool bDrawPlanes;
    bool bGraphSLAM;

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
    {
      if ( event.keyDown () )
      {
//      cout << "Key pressed " << event.getKeySym () << endl;
        if(event.getKeySym () == "k" || event.getKeySym () == "K")
          bTakeKeyframe = true;
//        else if(event.getKeySym () == "l" || event.getKeySym () == "L")//{cout << " Press L\n";
//          bDrawPlanes = !bDrawPlanes;//}
        else if(event.getKeySym () == "l" || event.getKeySym () == "L"){cout << " Press O\n";
          bGraphSLAM = !bGraphSLAM;
          bFreezeFrame = false;
          }
      }
    }
};


void print_help(char ** argv)
{
  cout << "\nThis program loads the pointCloud and the frame360_1->planes segmented from an observation RGBD360\n";
  cout << "usage: " << argv[0] << " [options] \n";
  cout << argv[0] << " -h | --help : shows this help" << endl;
//  cout << argv[0] << " <pathToPointCloud> <pathToPbMap>" << endl;
}

int main (int argc, char ** argv)
{
//  if(argc != 1)
//    print_help(argv);

  string path_data = static_cast<string>(argv[1]);

cout << "Create RegisterGraphSphere object\n";
  RegisterGraphSphere rgbd360_reg_seq;
  rgbd360_reg_seq.run(path_data);

cout << " EXIT\n";
  return (0);
}
