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

#ifndef LOOPCLOSURE360_H
#define LOOPCLOSURE360_H

#include "Map360.h"
#include "RegisterRGBD360.h"
#include "GraphOptimizer.h" // Pose-Graph optimization
//#include <GraphOptimizer.h> // Pose-Graph optimization

/*! This class manages the Map360's loop closure in a separate thread. As new keyframes are added to the map, they are compared
 *  with previous ones which are close-by, and if a good registration is found, the pose-graph is updated and the map is optimized
 *  using the loop closure information.
 */
class LoopClosure360
{
private:
    /*! Reference to the Map */
    Map360 &Map;

    /*! Keyframe registration object */
    RegisterRGBD360 registerer;

    /*! Pose-graph optimization object */
    GraphOptimizer optimizer;

    /*! Number of keyframes to optimize (the total number of keyframes - 1) */
    int numKFsLC;

    /*! Vector of queued Keyframes waiting for loop closure checking */
    std::vector<std::pair<unsigned,unsigned> > vQueueNewKFs;

    /*! Do not check for LC with every KF, but each certain amount 'sampleSearchLC' */
    int sampleSearchLC;

    /*! Boolean to control the stop of the infinite loop */
    bool bShouldStop;

    /*! Thread handler*/
    mrpt::system::TThreadHandle thread_hd_;


public:

    /*! Connections found with loop closure. The matched planar area is saved here */
    std::map<unsigned, std::map<unsigned, float> > connectionsLC;

    //  /*! Vector of queued Keyframes waiting for loop closure checking */
    //  std::vector<unsigned> vQueueNewKFs;
    //  std::vector<unsigned> vQueueNewKFs_match; // The match from vQueueNewKFs -> Change to a pair!!!

    /*! This constructor starts the loop closure checking in a new thread */
    LoopClosure360(Map360 &map) :
        Map(map),
        registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH)),
        numKFsLC(1),
        bShouldStop(false)
    {
        optimizer.setRigidTransformationType(GraphOptimizer::SixDegreesOfFreedom);
        Eigen::Matrix4d firstPose = Eigen::Matrix4d::Identity();
        optimizer.addVertex(firstPose); // Set the first pose to the Identity

        thread_hd_ = mrpt::system::createThreadFromObjectMethod(this, &LoopClosure360::run);
    }

    /*! This destructor finishes the loop closure thread */
    ~LoopClosure360()
    {
        bShouldStop = true;
        mrpt::system::joinThread(thread_hd_);
        thread_hd_.clear();
    }


private:

    /*! Execute the loop closure functionality continuously in an infinite loop */
    void run()
    {
        std::cout << "Start loop closure thread... \n";

        sampleSearchLC = 1; // Do not check for LC with every KF, but each certain amount 'sampleSearchLC'
        float factorDistance = 0.2; // This factor multiplies the trajectory length to obtain the maximum radius where loop closures are searched for
        int minMatchesThreshold = 5; // Thresholds for the loop closure to reduce the number of inconsistent matches
        float areaThreshold = 15.0;

        int newFrameID = 1;

        // Dense RGB-D alignment
        RegisterPhotoICP align360;
        align360.setNumPyr(5);
        align360.useSaliency(false);
        //align360.setVisualization(true);
        align360.setGrayVariance(3.f/255);
        float angleOffset = 157.5;
        Eigen::Matrix4f rotOffset = Eigen::Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);

        // The first keyframe should always correspond with the frame of reference (Identity). WARNING: if this is not like that, then some changes have to be done!!!
        while(!bShouldStop) // Infinite loop
        {
//            //      std::cout << "Check map \n";
//            // Add the new vertex provided by the tracker
//            while(Map.mmConnectionKFs.size() > numKFsLC)
//            {
//                //      std::cout << "\t LC " << Map.mmConnectionKFs.size() << " " << numKFsLC << " \n";
//                optimizer.addVertex(Map.vTrajectoryPoses[++numKFsLC].cast<double>());
//                std::map<unsigned, std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > >::iterator itLink = Map.mmConnectionKFs[numKFsLC].begin();
//                optimizer.addEdge(itLink->first, numKFsLC, itLink->second.first.cast<double>(), itLink->second.second.cast<double>());

//                if(vQueueNewKFs.size() < 2)
//                {
//                    //          vQueueNewKFs.push_back(frame360->id);
//                    vQueueNewKFs.push_back(std::pair<unsigned,unsigned>(numKFsLC, itLink->first));
//                    //          vQueueNewKFs_match.push_back(*compareSphereId);
//                }
//            }

//            // Check for loop closures
//            if(!vQueueNewKFs.empty())
//            {
//                // std::cout << "\t LC search connections " << vQueueNewKFs[0].first << " \n";

//                // int newFrameID = vQueueNewKFs[0];
//                int newFrameID = vQueueNewKFs[0].first;
//                int compareLocalIdx = vQueueNewKFs[0].second;
//                vQueueNewKFs.erase(vQueueNewKFs.begin());
//                Frame360 *newKF = Map.vpSpheres[newFrameID];
//                //        Eigen::Matrix4d &newPose = Map.vTrajectoryPoses[newFrameID];
//                Eigen::Matrix4f &newPose = Map.vOptimizedPoses[newFrameID];
//                //        int compareLocalIdx = vQueueNewKFs_match[0]-1;
//                //        vQueueNewKFs_match.erase(vQueueNewKFs_match.begin());

//                // Search for loop closure constraints in the current local map
//                std::vector<unsigned> vCompareWith;//(std::distance(Map.vsAreas[newKF->node].begin(), Map.vsAreas[newKF->node].find(compareLocalIdx)));
//                for(std::set<unsigned>::iterator itKF = Map.vsAreas[newKF->node].begin(); itKF != Map.vsAreas[newKF->node].end() && *itKF < compareLocalIdx; itKF++) // Set the iterator to the previous KFs not matched yet
//                    vCompareWith.push_back(*itKF);
//                std::cout << "\t LC search connections " << vQueueNewKFs[0].first << " vCompareWith size " << vCompareWith.size() << " \n";
//                for(int compId=vCompareWith.size()-1; compId >= 0; compId-=sampleSearchLC)
//                {
//                    compareLocalIdx = vCompareWith[compId];

//                    // Do not check for loop closure with the previous close-by frames
//                    if(Map.vTrajectoryIncrements[newFrameID] - Map.vTrajectoryIncrements[compareLocalIdx] < 6)
//                        continue;

//                    //      std::cout << "Check loop closure with " << compareLocalIdx << " \n";
//                    float distance = (newPose.block(0,3,3,1) - Map.vTrajectoryPoses[compareLocalIdx].block(0,3,3,1)).norm();
//                    float Threshold_distance = std::max(3.f, factorDistance*(Map.vTrajectoryIncrements[newFrameID] - Map.vTrajectoryIncrements[compareLocalIdx]) );
//                    if( distance < Threshold_distance )
//                    {
//                        std::cout << "Check loop closure between " << newFrameID << " " << compareLocalIdx << "\n";
//                        bool bGoodRegistration = registerer.RegisterPbMap(Map.vpSpheres[compareLocalIdx], newKF, 0, RegisterRGBD360::RegisterRGBD360::RegisterRGBD360::PLANAR_3DoF);
//                        if(bGoodRegistration && registerer.getMatchedPlanes().size() > minMatchesThreshold && registerer.getAreaMatched() > areaThreshold)
//                        {
//                            std::cout << "Loop closed between " << newFrameID << " and " << compareLocalIdx << " matchedArea " << registerer.getAreaMatched() << std::endl;
//                            Eigen::Matrix4f relativePose = registerer.getPose();
//                            Eigen::Matrix<float,6,6> informationMatrix = registerer.getInfoMat();

//                            // Refine registration
//                            RegisterPhotoICP align360;
//                            Map.vpSpheres[compareLocalIdx]->stitchSphericalImage();
//                            newKF->stitchSphericalImage();
//                            align360.setSourceFrame(Map.vpSpheres[compareLocalIdx]->sphereRGB, Map.vpSpheres[compareLocalIdx]->sphereDepth); // The reference keyframe
//                            align360.setTargetFrame(newKF->sphereRGB, newKF->sphereDepth);
//                            Eigen::Matrix4f initTransf_dense = rotOffset * relativePose * rotOffset.inverse();
//                            align360.alignFrames360(initTransf_dense, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
//                            relativePose = rotOffset.inverse() * align360.getOptimalPose() * rotOffset;
//                            informationMatrix = align360.getHessian();

//                            if(align360.avDepthResidual < 2.0)
//                            {
//                                optimizer.addEdge(compareLocalIdx, newFrameID, relativePose.cast<double>(), informationMatrix.cast<double>());
//                                Map.mmConnectionKFs[newFrameID][compareLocalIdx] = std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> >(relativePose, informationMatrix);
//                                connectionsLC[newFrameID][compareLocalIdx] = registerer.getAreaMatched() / registerer.areaSource;
//                                std::cout << "\n\t LC Add edge between " << compareLocalIdx << " and " << newFrameID << "\n";
//                            }
//                            //              break;
//                            //              assert(false); // Check if OK
//                        }
//                    }
//                }

//                // Optimize graphSLAM (like with g2o)
//                if( Map.mmConnectionKFs[newFrameID].size() > 1 )
//                {
//                    //        std::cout << "Optimize graph SLAM \n";
//                    double time_start = pcl::getTime();
//                    //Optimize the graph
//                    optimizer.optimizeGraph();
//                    double time_end = pcl::getTime();
//                    std::cout << " Optimize graph SLAM " << numKFsLC << " took " << double (time_end - time_start) << " s.\n";

//                    //Update the optimized poses (for loopClosure detection & visualization)
//                    {
//                        boost::mutex::scoped_lock updateLock(Map.mapMutex);
//                        optimizer.getPoses(Map.vOptimizedPoses);
//                        //Update the optimized pose
//                        Eigen::Matrix4f &newPose = Map.vOptimizedPoses[newFrameID];
//                        //            std::cout << "First pose optimized \n" << Map.vOptimizedPoses[0] << std::endl;
//                    }
//                }

                //        // Look for loop closures in the neighbor areas
                ////      std::cout << "\t Look for loop closures in the neighbor areas \n";
                //        for(std::set<unsigned>::iterator neigArea = Map.vsNeighborAreas[newKF->node].begin();
                //            neigArea != Map.vsNeighborAreas[newKF->node].end();
                //            neigArea++)
                //        {
                //          std::cout << "Check neigArea " << *neigArea << " \n";
                //          if(*neigArea == newKF->node)
                //            continue;
                //
                //          for(std::set<unsigned>::iterator itKF = Map.vsAreas[*neigArea].begin(); itKF != Map.vsAreas[*neigArea].end(); itKF++) //itKF+=sampleSearchLC
                //          {
                //            compareLocalIdx = *itKF;
                ////            compareLocalIdx = vSelectedKFs[*neigArea];
                //  //        std::cout << "Check loop closure\n";
                //            float distance = (newPose.block(0,3,3,1) - Map.vTrajectoryPoses[compareLocalIdx].block(0,3,3,1)).norm();
                //            if( distance < factorDistance*(Map.vTrajectoryIncrements[newFrameID] - Map.vTrajectoryIncrements[compareLocalIdx]) )
                //            {
                //            std::cout << "Check loop closure between " << newFrameID << " " << compareLocalIdx << "\n";
                //              unsigned numMatched = 0;
                //              bool bGoodRegistration = registerer.Register(Map.vpSpheres[compareLocalIdx], newKF, 0, RegisterRGBD360::RegisterRGBD360::PLANAR_3DoF);
                //              if(bGoodRegistration && registerer.getMatchedPlanes().size() > minMatchesThreshold && registerer.getAreaMatched() > areaThreshold)
                //              {
                //                std::cout << "Loop closed between " << newFrameID << " and " << compareLocalIdx << " matchedArea " << registerer.getAreaMatched() << std::endl;
                //                Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix2 = registerer.getInfoMat().cast<double>();
                //                Eigen::Matrix4d relativePose = registerer.getPose();
                //                Eigen::Matrix<double,6,6> informationMatrix = registerer.getInfoMat();
                //                optimizer.addEdge(compareLocalIdx, newFrameID, relativePose, informationMatrix);
                //                Map.mmConnectionKFs[newFrameID][compareLocalIdx] = std::pair<Eigen::Matrix4d, Eigen::Matrix<double,6,6> >(registerer.getPose(), registerer.getInfoMat());
                //                connectionsLC[newFrameID][compareLocalIdx] = registerer.getAreaMatched();
                //              std::cout << "Add edge between " << compareLocalIdx << " and " << newFrameID << "\n";
                //  //              assert(false); // Check if OK
                //              }
                //            }
                //          }
                //        }
                //

            // Look for loop closures in the other areas further away
            std::cout << "\t other areas further away \n";
            while(Map.vpSpheres.size() > newFrameID)
            {
                Frame360* newKF = Map.vpSpheres[++newFrameID];
                int compareLocalIdx;
                bool bFoundSelectedKFs = false;
                unsigned areaID;
                for( areaID=0; areaID < Map.vsAreas.size(); areaID++)
                {
                    if(Map.vsNeighborAreas[newKF->node].count(areaID) > 0)
                        continue;

                    assert(areaID != newKF->node);

                    //          for(std::set<unsigned>::iterator itKF = Map.vsAreas[areaID].begin(); itKF != Map.vsAreas[areaID].end(); itKF+=sampleSearchLC)
                    //          {
                    //            compareLocalIdx = *itKF;
                    compareLocalIdx = Map.vSelectedKFs[areaID];
                    //        std::cout << "Check loop closure\n";
                    Eigen::Matrix4f relativePose = Map.vpSpheres[compareLocalIdx]->pose.inverse() * newKF->pose;
                    float distance = relativePose.block(0,3,3,1).norm();
//                    if( distance < factorDistance*(Map.vTrajectoryIncrements[compareLocalIdx] - Map.vTrajectoryIncrements[newFrameID]) )
                    if( distance < 5.f )
                    {
                        std::cout << "Check loop closure between " << newFrameID << " " << compareLocalIdx << "\n";
                        bool bGoodRegistration = registerer.RegisterPbMap(Map.vpSpheres[compareLocalIdx], newKF, 25, RegisterRGBD360::RegisterRGBD360::PLANAR_3DoF);
                        if(bGoodRegistration && registerer.getMatchedPlanes().size() > minMatchesThreshold && registerer.getAreaMatched() > areaThreshold)
                        {
                            std::cout << "Loop closed between " << newFrameID << " and " << compareLocalIdx << " matchedArea " << registerer.getAreaMatched() << std::endl;
                            //                Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = registerer.getAreaMatched() * registerer.getInfoMat().cast<double>();
                            Eigen::Matrix4f relativePose = registerer.getPose();
//                            Eigen::Matrix<double,6,6> informationMatrix = registerer.getInfoMat();

                            // Refine registration
                            RegisterPhotoICP align360;
//                            Map.vpSpheres[compareLocalIdx]->stitchSphericalImage();
//                            newKF->stitchSphericalImage();
                            align360.setSourceFrame(Map.vpSpheres[compareLocalIdx]->sphereRGB, Map.vpSpheres[compareLocalIdx]->sphereDepth); // The reference keyframe
                            align360.setTargetFrame(newKF->sphereRGB, newKF->sphereDepth);
                            Eigen::Matrix4f initTransf_dense = rotOffset * relativePose * rotOffset.inverse();
                            align360.alignFrames360(initTransf_dense, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
                            relativePose = rotOffset.inverse() * align360.getOptimalPose() * rotOffset;
                            Eigen::Matrix<float,6,6> informationMatrix = align360.getHessian();

                            if(align360.avDepthResidual < 2.0)
                            {
                                optimizer.addEdge(compareLocalIdx, newFrameID, relativePose.cast<double>(), informationMatrix.cast<double>());
                                Map.mmConnectionKFs[newFrameID][compareLocalIdx] = std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> >(relativePose, informationMatrix);
//                                connectionsLC[newFrameID][compareLocalIdx] = registerer.getAreaMatched() / registerer.areaSource;
                                connectionsLC[newFrameID][compareLocalIdx] = align360.SSO;
                                std::cout << "\n\t LC Add edge between " << compareLocalIdx << " and " << newFrameID << "\n";
                            }

                            bFoundSelectedKFs = true;
                            break; // If a loop closure is found in one area, stop searching other areas
                            //              assert(false); // Check if OK
                        }
                    }
                }

                if(bFoundSelectedKFs) // Look for more loop closures in this submap
                {
                    for(unsigned kf=0; kf < Map.vpSpheres.size(); kf++)
                        if((Map.vpSpheres[kf]->node == areaID && kf != Map.vSelectedKFs[areaID]))// || isInNeighbourSubmap(areaID,kf)) && kf != nearestKF)
                        {
                            Eigen::Matrix4f relativePose = Map.vpSpheres[kf]->pose.inverse() * newKF->pose;
                            if(relativePose.block(0,3,3,1).norm() < 5.f) // If the KFs are closeby
                            {
                                std::cout << "Check loop closure between " << newFrameID << " " << compareLocalIdx << "\n";
                                bool bGoodRegistration = registerer.RegisterPbMap(Map.vpSpheres[compareLocalIdx], newKF, 25, RegisterRGBD360::RegisterRGBD360::PLANAR_3DoF);
                                if(bGoodRegistration && registerer.getMatchedPlanes().size() > minMatchesThreshold && registerer.getAreaMatched() > areaThreshold)
                                {
                                    std::cout << "Loop closed between " << newFrameID << " and " << compareLocalIdx << " matchedArea " << registerer.getAreaMatched() << std::endl;
                                    //                Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = registerer.getAreaMatched() * registerer.getInfoMat().cast<double>();
                                    Eigen::Matrix4f relativePose = registerer.getPose();

                                    align360.setTargetFrame(Map.vpSpheres[kf]->sphereRGB, Map.vpSpheres[kf]->sphereDepth);
                                    align360.setSourceFrame(newKF->sphereRGB, newKF->sphereDepth);
                                    Eigen::Matrix4f initRigidTransfDense = rotOffset * relativePose * rotOffset.inverse();
                                    align360.alignFrames360(initRigidTransfDense, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
                                    relativePose = rotOffset.inverse() * align360.getOptimalPose() * rotOffset;
                                    Eigen::Matrix<float,6,6> informationMatrix = align360.getHessian();

                                    if(align360.avDepthResidual < 2.0)
                                    {
                                        optimizer.addEdge(compareLocalIdx, newFrameID, relativePose.cast<double>(), informationMatrix.cast<double>());
                                        Map.mmConnectionKFs[newFrameID][compareLocalIdx] = std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> >(relativePose, informationMatrix);
        //                                connectionsLC[newFrameID][compareLocalIdx] = registerer.getAreaMatched() / registerer.areaSource;
                                        connectionsLC[newFrameID][compareLocalIdx] = align360.SSO;
                                        std::cout << "\n\t LC Add edge between " << compareLocalIdx << " and " << newFrameID << "\n";
                                    }
                                }
                            }
                        }
                }
//                else // Look for LC using all the KFs in the map
//                {

//                }
            }

//            }
//            else
//                boost::this_thread::sleep (boost::posix_time::microseconds(10));

        }
    }
};

#endif
