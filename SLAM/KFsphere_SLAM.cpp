/*
 *  Copyright (c) 2012, Universidad de Mรกlaga - Grupo MAPIR and
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

#include <mrpt/utils/CStream.h>

#include <Map360.h>
#include <Map360_Visualizer.h>
#include <TopologicalMap360.h>
#include <Relocalizer360.h>
#include <LoopClosure360.h>
#include <FilterPointCloud.h>
#include <GraphOptimizer.h> // Pose-Graph optimization

//#include <pcl/registration/icp.h>
//#include <pcl/registration/icp_nl.h> //ICP LM
//#include <pcl/registration/gicp.h> //GICP

#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>
#include <mrpt/system/os.h>

#define MAX_MATCH_PLANES 25

using namespace std;
using namespace Eigen;

/*! This class contains the functionality to perform hybrid (metric-topological-semantic) SLAM.
 *  The input data is a stream of omnidirectional RGB-D images (from a recorded sequence). Tracking is tackled
 *  by registering planes extracted from the RGB-D frames. Semantic information is inferred online, which increases
 *  the robustness and efficiency of registration. A pose-graph of spherical keyframes is optimized continuously in
 *  a back-end process which uses the information given by a separate back-end process to detect loop closures. The
 *  map is organized in a set of topological nodes according to a criteria of common observability.
 */
class SphereGraphSLAM
{
private:
    Map360 Map;

    // Create the topological arrangement object
    TopologicalMap360 topologicalMap;

    // Get the calibration matrices for the different sensors
    Calib360 calib;

    // Create registration object
    RegisterRGBD360 registerer;

    // Create relocalization object
    Relocalizer360 relocalizer;

    // Nearest KF index
    unsigned nearestKF;

    struct connection
    {
        unsigned KF_id;
        std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > geomConnection;
        float sso;
    };

public:
    SphereGraphSLAM() :
        registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH)),
        relocalizer(Map),
        topologicalMap(Map)
    {
        Map.currentArea = 0;
        calib.loadExtrinsicCalibration();
        calib.loadIntrinsicCalibration();
    }

    ~SphereGraphSLAM()
    {
        // Clean memory
        for(unsigned i=0; i < Map.vpSpheres.size(); i++)
            delete Map.vpSpheres[i];
    }

    // This function decides to select a new keyframe when the proposed keyframe does not have keyframes too near/related
    bool isInNeighbourSubmap(unsigned submap, unsigned kf)
    {
        for(std::set<unsigned>::iterator it=Map.vsNeighborAreas[submap].begin(); it != Map.vsNeighborAreas[submap].end(); it++)
            if( Map.vsAreas[*it].count(kf) )
                return true;

        return false;
    }

    bool isOdometryContinuousMotion(Eigen::Matrix4f &prevPose, Eigen::Matrix4f &currPose, float thresDist = 0.1)
    {
        Eigen::Matrix4f relativePose = prevPose.inverse() * currPose;
        if( relativePose.block(0,3,3,1).norm() > thresDist )
            return false;

        return true;
    }

    // This function decides to select a new keyframe when the proposed keyframe does not have keyframes too near/related
    bool shouldSelectKeyframe(Frame360 *candidateKF, std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > candidateKF_connection, Frame360* currentFrame)
    {
        cout << "shouldSelectKeyframe ...\n";
        double time_start = pcl::getTime();

        RegisterPhotoICP align360; // Dense RGB-D alignment
        align360.setNumPyr(5);
        align360.useSaliency(false);
        align360.setVisualization(true);
        align360.setGrayVariance(3.f/255);

        // The reference of the spherical image and the point Clouds are not the same! I should always use the same coordinate system (TODO)
        float angleOffset = 157.5;
        Eigen::Matrix4f rotOffset = Eigen::Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);

        cout << "Align Sphere " << endl;
        double time_start_dense = pcl::getTime();
        align360.setTargetFrame(Map.vpSpheres[nearestKF]->sphereRGB, Map.vpSpheres[nearestKF]->sphereDepth);
        align360.setSourceFrame(candidateKF->sphereRGB, candidateKF->sphereDepth);
        //                align360.alignFrames360(Eigen::Matrix4f::Identity(), RegisterPhotoICP::PHOTO_CONSISTENCY); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
        align360.alignFrames360(rotOffset * candidateKF_connection.first * rotOffset.inverse(), RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
        Eigen::Matrix4f rigidTransf_dense = rotOffset.inverse() * align360.getOptimalPose() * rotOffset;
        double time_end_dense = pcl::getTime();
        std::cout << "Dense " << (time_end_dense - time_start_dense) << std::endl;
        cout << "Dense regist \n" << rigidTransf_dense << " \nRegist-PbMap \n" << candidateKF_connection.first << endl;

        //        math::mrpt::CPose3D
        if(!rigidTransf_dense.isApprox(candidateKF_connection.first,1e-1))
        {
            std::cout << "shouldSelectKeyframe INVALID KF\n";
            return false;
        }
        else
        {
            candidateKF_connection.first = rigidTransf_dense;
        }

        double dist_candidateKF = candidateKF_connection.first.block(0,3,3,1).norm();
        Eigen::Matrix4f candidateKF_pose = Map.vTrajectoryPoses[nearestKF] * candidateKF_connection.first;
        map<float,unsigned> nearest_KFs;
        for(std::set<unsigned>::iterator itKF = Map.vsAreas[Map.currentArea].begin(); itKF != Map.vsAreas[Map.currentArea].end(); itKF++) // Set the iterator to the previous KFs not matched yet
        {
            if(*itKF == nearestKF)
                continue;

            double dist_to_prev_KF = (candidateKF_pose.block(0,3,3,1) - Map.vTrajectoryPoses[*itKF].block(0,3,3,1)).norm();
            cout << *itKF << " dist_to_prev_KF " << dist_to_prev_KF << " dist_candidateKF " << dist_candidateKF << endl;
            if( dist_to_prev_KF < dist_candidateKF ) // Check if there are even nearer keyframes
                nearest_KFs[dist_to_prev_KF] = *itKF;
        }
        for(map<float,unsigned>::iterator itNearKF = nearest_KFs.begin(); itNearKF != nearest_KFs.end(); itNearKF++) // Set the iterator to the previous KFs not matched yet
        {
            cout << "itNearKF " << itNearKF->first << " " << itNearKF->second << endl;
            bool bGoodRegistration = registerer.RegisterPbMap(Map.vpSpheres[itNearKF->second], candidateKF, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
            if(bGoodRegistration && registerer.getMatchedPlanes().size() >= min_planes_registration && registerer.getAreaMatched() > 6 )
            {
                nearestKF = itNearKF->second;
                double time_end = pcl::getTime();
                std::cout << "shouldSelectKeyframe INVALID KF. took " << double (time_end - time_start) << std::endl;
                return false;
            }
        }
        double time_end = pcl::getTime();
        std::cout << "shouldSelectKeyframe VALID KF. took " << double (time_end - time_start) << std::endl;
        //      cout << "shouldSelectKeyframe VALID KF\n";
        return true;
    }

    void run(string path, const int &selectSample)
    {
        std::cout << "SphereGraphSLAM::run... " << std::endl;

        //      // Create the topological arrangement object
        //      TopologicalMap360 topologicalMap(Map);
        //      Map.currentArea = 0;

        //      // Get the calibration matrices for the different sensors
        //      Calib360 calib;
        //      calib.loadExtrinsicCalibration();
        //      calib.loadIntrinsicCalibration();

        //      // Create registration object
        //      RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));

        // Filter the point clouds before visualization
        FilterPointCloud<PointT> filter;

        int frame = 282; // Skip always the first frame, which sometimes contains errors
        int frameOrder = 0;

        Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();

        string fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

        // Load first frame
        Frame360* frame360 = new Frame360(&calib);
        frame360->loadFrame(fileName);
        frame360->undistort();
        frame360->stitchSphericalImage();
        frame360->buildSphereCloud_rgbd360();
        frame360->getPlanes();
        //frame360->id = frame;
        frame360->id = frameOrder;
        frame360->node = Map.currentArea;

        nearestKF = frameOrder;

        filter.filterEuclidean(frame360->sphereCloud);

        Map.addKeyframe(frame360,currentPose);
//        Map.vOptimizedPoses.push_back( currentPose );
        Map.vSelectedKFs.push_back(0);
        Map.vTrajectoryIncrements.push_back(0);

        // Topological Partitioning
        Map.vsAreas.push_back( std::set<unsigned>() );
        Map.vsAreas[Map.currentArea].insert(frameOrder);
        Map.vsNeighborAreas.push_back( std::set<unsigned>() );	// Vector with numbers of neighbor areas (topological nodes)
        Map.vsNeighborAreas[Map.currentArea].insert(Map.currentArea);
        topologicalMap.vSSO.push_back( mrpt::math::CMatrix(1,1) );
        topologicalMap.vSSO[Map.currentArea](frameOrder,frameOrder) = 0.0;

        frame += selectSample;
        fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

        // Start visualizer
        Map360_Visualizer Viewer(Map,1);

        // Graph-SLAM
        //      std::cout << "\t  mmConnectionKFs " << Map.mmConnectionKFs.size() << " \n";
        // LoopClosure360 loopCloser(Map);
        //        float areaMatched, currentPlanarArea;
        //      Frame360 *candidateKF = new Frame360(&calib); // A reference to the last registered frame to add as a keyframe when needed
        //      std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > candidateKF_connection; // The register information of the above
        GraphOptimizer optimizer;
        optimizer.setRigidTransformationType(GraphOptimizer::SixDegreesOfFreedom);
        std::cout << "Added vertex: "<< optimizer.addVertex(currentPose.cast<double>()) << std::endl;
        bool bHasNewLC = false;

        Frame360 *candidateKF = NULL; // A reference to the last registered frame to add as a keyframe when needed
        std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > candidateKF_connection; // The register information of nearestKF
        float candidateKF_sso;

        //      while( true )
        //        boost::this_thread::sleep (boost::posix_time::milliseconds (10));

        bool bPrevKFSelected = true, bGoodTracking = true;
        int lastTrackedKF = 0; // Count the number of not tracked frames after the last registered one

        // Dense RGB-D alignment
        RegisterPhotoICP align360;
        align360.setNumPyr(5);
        align360.useSaliency(false);
        //        align360.setVisualization(true);
        align360.setGrayVariance(3.f/255);
        double selectKF_ICPdist = 0.9;
//        double selectKF_ICPdist = 1.1;
        double thresholdConnections = 2.5; // Distance in meters to evaluate possible connections
        Eigen::Matrix4f rigidTransf_dense = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f rigidTransf_dense_ref = rigidTransf_dense;
        // The reference of the spherical image and the point Clouds are not the same! I should always use the same coordinate system (TODO)
        float angleOffset = 157.5;
        Eigen::Matrix4f rotOffset = Eigen::Matrix4f::Identity(); rotOffset(1,1) = rotOffset(2,2) = cos(angleOffset*PI/180); rotOffset(1,2) = sin(angleOffset*PI/180); rotOffset(2,1) = -rotOffset(1,2);

        while( fexists(fileName.c_str()) )
        {
            // Load pointCloud
            //        if(!bPrevKFSelected)
            {
                cout << "Frame " << fileName << endl;
                frame360 = new Frame360(&calib);
                frame360->loadFrame(fileName);
                frame360->undistort();
                frame360->stitchSphericalImage();
                frame360->buildSphereCloud_rgbd360();
                frame360->getPlanes();
                frame360->id = frameOrder+1;
            }
            //            cout << " Load Keyframe \n" << endl;

            // The next frame to process
            frame += selectSample;
            fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

            // Register the pair of frames
            bGoodTracking = registerer.RegisterPbMap(Map.vpSpheres[nearestKF], frame360, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
            Matrix4f trackedPosePbMap = registerer.getPose();
            cout << "bGoodTracking " << bGoodTracking << " with " << nearestKF << ". Distance "
                 << registerer.getPose().block(0,3,3,1).norm() << " entropy " << registerer.calcEntropy() << " matches " << registerer.getMatchedPlanes().size() << " area " << registerer.getAreaMatched() << endl;

            //            // If the registration is good, do not evaluate this KF and go to the next
            //            if( bGoodTracking  && registerer.getMatchedPlanes().size() >= min_planes_registration && registerer.getAreaMatched() > 12 )//&& registerer.getPose().block(0,3,3,1).norm() < 1.5 )
            //                //        if( bGoodTracking && registerer.getMatchedPlanes().size() > 6 && registerer.getPose().block(0,3,3,1).norm() < 1.0 )
            //            {
            //                float dist_odometry = 0; // diff_roatation
            //                if(candidateKF)
            //                    dist_odometry = (candidateKF_connection.first.block(0,3,3,1) - trackedPosePbMap.block(0,3,3,1)).norm();
            //                //          else
            //                //            dist_odometry = trackedPosePbMap.norm();
            //                if( dist_odometry < max_translation_odometry ) // Check that the registration is coherent with the camera movement (i.e. in odometry the registered frames must lay nearby)
            //                {
            //                    if(trackedPosePbMap.block(0,3,3,1).norm() > min_dist_keyframes) // Minimum distance to select a possible KF
            //                    {
            //                        cout << "Set candidateKF \n";
            //                        if(candidateKF)
            //                            delete candidateKF; // Delete previous candidateKF

            //                        candidateKF = frame360;
            //                        candidateKF_connection.first = trackedPosePbMap;
            //                        candidateKF_connection.second = registerer.getInfoMat();
            //                        candidateKF_sso = registerer.getAreaMatched() / registerer.areaSource;
            //                    }

            //                    { boost::mutex::scoped_lock updateLockVisualizer(Viewer.visualizationMutex);
            //                        Viewer.currentLocation.matrix() = (currentPose * trackedPosePbMap);
            //                        Viewer.bDrawCurrentLocation = true;
            //                    }

            //                    bPrevKFSelected = false;
            //                    lastTrackedKF = 0;
            //                }

            //                continue; // Eval next frame
            //            }

            if(bGoodTracking && registerer.getMatchedPlanes().size() >= 6 && registerer.getAreaMatched() > 12)
            {
                delete frame360;
                rigidTransf_dense_ref = rotOffset * trackedPosePbMap * rotOffset.inverse();

                { boost::mutex::scoped_lock updateLockVisualizer(Viewer.visualizationMutex);
                    Viewer.currentLocation.matrix() = (currentPose * rigidTransf_dense);
                    Viewer.bDrawCurrentLocation = true;
                }

                cout << " skip frame PbMap " << endl;
                continue;
            }

            cout << "Align Spheres " << frame360->id << " and " << Map.vpSpheres[nearestKF]->id << endl;
            double time_start_dense = pcl::getTime();
            align360.setTargetFrame(Map.vpSpheres[nearestKF]->sphereRGB, Map.vpSpheres[nearestKF]->sphereDepth);
            align360.setSourceFrame(frame360->sphereRGB, frame360->sphereDepth);
//            rigidTransf_dense_ref = rotOffset * registerer.getPose() * rotOffset.inverse();
            align360.alignFrames360(rigidTransf_dense_ref, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
            Eigen::Matrix4f rigidTransf_dense_ref_prev = rigidTransf_dense_ref;
            rigidTransf_dense_ref = align360.getOptimalPose();
            rigidTransf_dense = rotOffset.inverse() * rigidTransf_dense_ref * rotOffset;
            double time_end_dense = pcl::getTime();
            //std::cout << "Dense " << (time_end_dense - time_start_dense) << std::endl;
            double depth_residual = align360.avDepthResidual;
            cout << " Residuals: " << align360.avPhotoResidual << " " << align360.avDepthResidual;
            //cout << " regist \n" << rigidTransf_dense << endl;

            { boost::mutex::scoped_lock updateLockVisualizer(Viewer.visualizationMutex);
                Viewer.currentLocation.matrix() = (currentPose * rigidTransf_dense);
                Viewer.bDrawCurrentLocation = true;
            }

            if(align360.avDepthResidual < selectKF_ICPdist && isOdometryContinuousMotion(rigidTransf_dense_ref_prev, rigidTransf_dense_ref, 0.2))
            {
//                assert(align360.avDepthResidual < 1.5);

                delete frame360;
                //                bPrevKFSelected = false;
                cout << " skip frame " << endl;
                continue;
            }

            // The last frame is a candidate for a KF
            candidateKF = frame360;
            candidateKF_connection.first = rigidTransf_dense;
            candidateKF_connection.second = align360.getHessian();
            candidateKF_sso = align360.SSO;

            // Check registration with nearby keyframes
            vector<connection> vConnections;
            unsigned kf;
            for( kf=0; kf < Map.vpSpheres.size(); kf++)
                //                if(Map.vpSpheres[kf]->node == Map.currentArea && kf != nearestKF) // || Map.vsAreas[Map.currentArea].count(kf))
                if((Map.vpSpheres[kf]->node == Map.currentArea || isInNeighbourSubmap(Map.currentArea,kf)) && kf != nearestKF)
                {
                    Eigen::Matrix4f relativePose = Map.vpSpheres[kf]->pose.inverse() * currentPose * rigidTransf_dense;
                    if(relativePose.block(0,3,3,1).norm() < thresholdConnections) // If the KFs are closeby
                    {
//                        std::cout << "Check loop closure between " << kf << " the current frame " << "\n";
//                        bool bGoodRegistration = registerer.RegisterPbMap(Map.vpSpheres[kf], frame360, 25, RegisterRGBD360::RegisterRGBD360::PLANAR_3DoF);
//                        if(bGoodRegistration && registerer.getMatchedPlanes().size() > 5 && registerer.getAreaMatched() > 15.0)
                        {
//                            std::cout << "Loop closed between " << newFrameID << " and " << compareLocalIdx << " matchedArea " << registerer.getAreaMatched() << std::endl;
                            //                Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = registerer.getAreaMatched() * registerer.getInfoMat().cast<double>();
//                            relativePose = registerer.getPose();
//                            Eigen::Matrix4f initRigidTransfDense = rotOffset * relativePose * rotOffset.inverse();

                            align360.setTargetFrame(Map.vpSpheres[kf]->sphereRGB, Map.vpSpheres[kf]->sphereDepth);
                            align360.setSourceFrame(frame360->sphereRGB, frame360->sphereDepth);
                            Eigen::Matrix4f initRigidTransfDense = rotOffset * relativePose * rotOffset.inverse();
                            align360.alignFrames360(initRigidTransfDense, RegisterPhotoICP::PHOTO_DEPTH); // PHOTO_CONSISTENCY / DEPTH_CONSISTENCY / PHOTO_DEPTH  Matrix4f relPoseDense = registerer.getPose();
                            rigidTransf_dense_ref_prev = rigidTransf_dense_ref;
                            rigidTransf_dense_ref = align360.getOptimalPose();
                            if(isOdometryContinuousMotion(rigidTransf_dense_ref_prev, rigidTransf_dense_ref, 0.2))
                                continue;
                            Eigen::Matrix4f rigidTransf_denseKF = rotOffset.inverse() * rigidTransf_dense_ref * rotOffset;

                            cout << "CHECK WITH " << kf << " nearestKF " << nearestKF << " avDepthResidual " << align360.avDepthResidual << " sso " << align360.SSO << endl;
                            //                        cout << "dist " << (Map.vpSpheres[kf]->pose.block(0,3,3,1)-frame360->pose.block(0,3,3,1)).norm() << " " << (initRigidTransfDense.block(0,3,3,1)).norm() << endl;

                            if(align360.avDepthResidual < selectKF_ICPdist)
                                break;
                            else if(align360.avDepthResidual < 1.8) // Keep connection
                            {
                                connection Connection;
                                Connection.KF_id = kf;
                                Connection.geomConnection = pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> >(rigidTransf_denseKF, align360.getHessian());
                                Connection.sso = align360.SSO;
                                vConnections.push_back(Connection);
                            }
                            //else
                            {
                                //std::cout << "Check loop closure between " << kf << " the current frame " << "\n";
                                bool bGoodRegistration = registerer.RegisterPbMap(Map.vpSpheres[kf], frame360, 25, RegisterRGBD360::RegisterRGBD360::PLANAR_3DoF);
                                if(bGoodRegistration && registerer.getMatchedPlanes().size() > 5 && registerer.getAreaMatched() > 25.0)
                                {
                                    connection Connection;
                                    Connection.KF_id = kf;
                                    Connection.geomConnection = pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> >(registerer.getPose(), registerer.getAreaMatched() * registerer.getInfoMat());
                                    Connection.sso = align360.SSO;
                                    vConnections.push_back(Connection);
                                }
                            }
                        }
                    }
                }
            if(align360.avDepthResidual < selectKF_ICPdist)
            {
                cout << "\t align360.avDepthResidual " << align360.avDepthResidual << " " << selectKF_ICPdist << " " << kf << endl;
                if(align360.avDepthResidual < depth_residual)
                {
                    currentPose = Map.vpSpheres[kf]->pose;
                    nearestKF = kf;
                    Viewer.currentSphere = nearestKF;
                    cout << "\t nearestKF " << nearestKF << endl;
                }
                delete frame360;
                cout << "skip frame 2 " << endl;
                continue;
            }


            cout << "SSO " << align360.SSO << endl;
            cout << "Select keyframe " << frame360->id << endl;
            cout << "Information \n" << candidateKF_connection.second << endl;
            rigidTransf_dense_ref = Eigen::Matrix4f::Identity();

            ++lastTrackedKF;

            //            if(bPrevKFSelected) // If the previous frame was a keyframe, just try to register the next frame as usually
            //            {
            //                std::cout << "Tracking lost " << lastTrackedKF << "\n";
            //                //          bGoodTracking = true; // Read a new frame in the next iteration
            //                bPrevKFSelected = false;
            //                continue;
            //            }

            //            if(lastTrackedKF > 3)
            //            {
            //                std::cout << "\n  Launch Relocalization \n";
            //                double time_start = pcl::getTime();

            //                if(relocalizer.relocalize(frame360) != -1)
            //                {
            //                    double time_end = pcl::getTime();
            //                    std::cout << "Relocalization in " << Map.vpSpheres.size() << " KFs map took " << double (time_end - time_start) << std::endl;
            //                    lastTrackedKF = 0;
            //                    candidateKF = frame360;
            //                    candidateKF_connection.first = relocalizer.registerer.getPose();
            //                    candidateKF_connection.second = relocalizer.registerer.getInfoMat();
            //                    nearestKF = relocalizer.relocKF;
            //                    currentPose = Map.vTrajectoryPoses[nearestKF]; //Map.vOptimizedPoses
            //                    Viewer.currentSphere = nearestKF;
            //                    std::cout << "Relocalized with " << nearestKF << " \n\n\n";

            //                    continue;
            //                }
            //            }

            //            if(!candidateKF)
            //            {
            //                std::cout << "_Tracking lost " << lastTrackedKF << "\n";
            //                continue;
            //            }

            //            if( !shouldSelectKeyframe(candidateKF, candidateKF_connection, frame360) )
            //            {
            //                cout << "  Do not add a keyframe yet. Now the camera is around KF " << nearestKF << endl;
            //                candidateKF = NULL;
            //                currentPose = Map.vTrajectoryPoses[nearestKF]; //Map.vOptimizedPoses
            //                Viewer.currentSphere = nearestKF;

            //                frame -= selectSample;
            //                fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

            //                //          lastTrackedKF = 0;
            //                //          bGoodTracking = true; // Read a new frame in the next iteration
            //                //          bPrevKFSelected = true; // This avoids selecting a KF for which there is no registration
            //                continue;
            //            }
            //
            //            bPrevKFSelected = true;
            //            frame -= selectSample;
            //            fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

            currentPose = currentPose * candidateKF_connection.first;
            std::cout<< "Added vertex: "<< optimizer.addVertex(currentPose.cast<double>()) << std::endl;
            std::cout<< "Added addEdge:" << nearestKF << " " << Map.vpSpheres.size() << "\n" << candidateKF_connection.first.cast<double>() << "\n";
            optimizer.addEdge(nearestKF, Map.vpSpheres.size(), candidateKF_connection.first.cast<double>(), candidateKF_connection.second.cast<double>());

            std::cout<< " Edge PbMap: " << bGoodTracking << " " << registerer.getMatchedPlanes().size() << " " << registerer.getAreaMatched() << " " << diffRotation(trackedPosePbMap, rigidTransf_dense) << " " << difTranslation(trackedPosePbMap, rigidTransf_dense) << "\n";
            if(bGoodTracking && registerer.getMatchedPlanes().size() >= 4 && registerer.getAreaMatched() > 6)
                if(diffRotation(trackedPosePbMap, rigidTransf_dense) < 5 && difTranslation(trackedPosePbMap, rigidTransf_dense) < 0.1) // If the difference is less than 5º and 10 cm, add the PbMap connection
                {
                    optimizer.addEdge(nearestKF, Map.vpSpheres.size(), trackedPosePbMap.cast<double>(), registerer.getInfoMat().cast<double>());
                    std::cout<< "Added addEdge PbMap\n";
                    std::cout<< "Edge PbMap\n" << registerer.getInfoMat() << endl;
                    std::cout<< "Edge Dense" << candidateKF_connection.second << endl;
                }

            Map.vTrajectoryIncrements.push_back(Map.vTrajectoryIncrements.back() + candidateKF_connection.first.block(0,3,3,1).norm());
            // Filter cloud
            filter.filterEuclidean(candidateKF->sphereCloud);
        std::cout<< "filterEuclidean:\n";

            cout << "\tGet previous frame as a keyframe " << frameOrder+1 << " dist " << candidateKF_connection.first.norm() << " candidateKF_sso " << candidateKF_sso << endl;
            // Add new keyframe
            candidateKF->id = ++frameOrder;
            candidateKF->node = Map.currentArea;

            // Update topologicalMap
            int newLocalFrameID = Map.vsAreas[Map.currentArea].size();
            topologicalMap.addKeyframe(Map.currentArea);
            std::cout<< "newSizeLocalSSO "<< newLocalFrameID+1 << std::endl;
//            int newSizeLocalSSO = newLocalFrameID+1;
//            topologicalMap.vSSO[Map.currentArea].setSize(newSizeLocalSSO,newSizeLocalSSO);
//            topologicalMap.vSSO[Map.currentArea](newLocalFrameID,newLocalFrameID) = 0.0;
//            // Re-adjust size of adjacency matrices
//            for(set<unsigned>::iterator it=Map.vsNeighborAreas[Map.currentArea].begin(); it != Map.vsNeighborAreas[Map.currentArea].end(); it++)
//            {
//                assert(*it != Map.currentArea);

//                if(*it < Map.currentArea)
//                {
//                    cout << " sizeSSO neig " << topologicalMap.mmNeigSSO[*it][Map.currentArea].getRowCount() << endl;
//                    topologicalMap.mmNeigSSO[*it][Map.currentArea].setSize(5, newSizeLocalSSO);
//                }
//                else
//                {
//                    topologicalMap.mmNeigSSO[Map.currentArea][*it].setSize(newSizeLocalSSO, topologicalMap.mmNeigSSO[*it][Map.currentArea].getRowCount());
//                }
//            }

            // Track the current position. TODO: (detect inconsistencies)
            int compareLocalIdx = newLocalFrameID-1;
            std::cout<< "compareLocalIdx "<< compareLocalIdx << std::endl;
            // Lock map to add a new keyframe
            {boost::mutex::scoped_lock updateLock(Map.mapMutex);
                Map.addKeyframe(candidateKF, currentPose);
                //Map.vOptimizedPoses.push_back( Map.vOptimizedPoses[nearestKF] * candidateKF_connection.first );
                std::cout << "\t  mmConnectionKFs loop " << frameOrder << " " << nearestKF << " \n";
                Map.mmConnectionKFs[frameOrder] = std::map<unsigned, std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > >();
                Map.mmConnectionKFs[frameOrder][nearestKF] = std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> >(candidateKF_connection.first, candidateKF_connection.second);
                Map.vsAreas[Map.currentArea].insert(frameOrder);
            }

            // Update candidateKF
            cout << "Update candidateKF " << endl;
            candidateKF = NULL;
            nearestKF = frameOrder;
            Viewer.currentSphere = nearestKF;
//            Viewer.bFreezeFrame = false;

            cout << "Add tracking SSO " << endl;
            if((topologicalMap.vSSO[Map.currentArea].getRowCount() <= newLocalFrameID) && (topologicalMap.vSSO[Map.currentArea].getRowCount() <= compareLocalIdx))
                cout << "\n\n\t ERROR vSSO\n";

            topologicalMap.vSSO[Map.currentArea](newLocalFrameID,compareLocalIdx) =
                    topologicalMap.vSSO[Map.currentArea](compareLocalIdx,newLocalFrameID) = candidateKF_sso;
            cout << "SSO is " << topologicalMap.vSSO[Map.currentArea](newLocalFrameID,compareLocalIdx) << endl;

            // Add for Map.mmConnectionKFs in the previous frames.
            if(vConnections.size() > 0)// && frameOrder !=3  && frameOrder !=4)
            {
                bHasNewLC = true;
                cout << "Add vConnections " << vConnections.size() << endl;
                for(unsigned link=0; link < vConnections.size(); link++)
                {
//                    cout << "Add vConnections between " << frameOrder << " " << vConnections[link].KF_id << "\n";
//                    cout << vConnections[link].KF_id << " sso " << vConnections[link].sso << "\n";
                    Map.mmConnectionKFs[frameOrder][vConnections[link].KF_id] = vConnections[link].geomConnection;
//                    cout << "Add vConnections_ A\n";
                    topologicalMap.addConnection(unsigned(frameOrder), vConnections[link].KF_id, vConnections[link].sso);
//                    cout << "Add vConnections_ B\n";
                    optimizer.addEdge(vConnections[link].KF_id, unsigned(frameOrder), vConnections[link].geomConnection.first.cast<double>(), vConnections[link].geomConnection.second.cast<double>());
//                cout << "frameOrder " << frameOrder << " " << Map.vpSpheres.size() << endl;
                }
            }

            //        // Calculate distance of furthest registration (approximated, we do not know if the last frame compared is actually the furthest)
            //        std::map<unsigned, pair<unsigned,Eigen::Matrix4f> >::iterator itFurthest = Map.mmConnectionKFs[frameOrder].begin(); // Aproximated
            //        cout << "Furthest distance is " << itFurthest->second.second.block(0,3,3,1).norm() << " diffFrameNum " << newLocalFrameID - itFurthest->first << endl;


            //            // Synchronize topologicalMap with loopCloser
            //            while( !loopCloser.connectionsLC.empty() )
            //            {
            //              std::map<unsigned, std::map<unsigned, float> >::iterator it_connectLC1 = loopCloser.connectionsLC.begin();
            //              while( !it_connectLC1->second.empty() )
            //              {
            //                std::map<unsigned, float>::iterator it_connectLC2 = it_connectLC1->second.begin();
            //                if(Map.vpSpheres[it_connectLC1->first]->node != Map.currentArea || Map.vpSpheres[it_connectLC2->first]->node != Map.currentArea)
            //                {
            //                  loopCloser.connectionsLC.erase(it_connectLC1);
            //                  continue;
            //                }
            //                int localID1 = std::distance(Map.vsAreas[Map.currentArea].begin(), Map.vsAreas[Map.currentArea].find(it_connectLC1->first));
            //                int localID2 = std::distance(Map.vsAreas[Map.currentArea].begin(), Map.vsAreas[Map.currentArea].find(it_connectLC2->first));
            //              cout << "Add LC SSO " << topologicalMap.vSSO[Map.currentArea].getRowCount() << " " << localID1 << " " << localID2 << endl;
            //                topologicalMap.vSSO[Map.currentArea](localID1,localID2) = topologicalMap.vSSO[Map.currentArea](localID2,localID1) =
            //                                                                          it_connectLC2->second;
            //                it_connectLC1->second.erase(it_connectLC2);
            //              }
            //              loopCloser.connectionsLC.erase(it_connectLC1);
            //            }

            // Optimize graphSLAM with g2o
            if(bHasNewLC)
            {
                std::cout << "Optimize graph SLAM \n";
                double time_start = pcl::getTime();
                //Optimize the graph
//                for(int j=0; j < Map.vpSpheres.size(); j++)
//                {
////                    std::cout << "optimizedPose " << poseIdx << " submap " << Map.vpSpheres[j]->node << " same? " << Map.vOptimizedPoses[j]==Map.vTrajectoryPoses[j] ? 1 : 0 << std::endl;
//                    g2o::OptimizableGraph::Vertex *vertex_j = optimizer.optimizer.vertex(j);
//                    if(Map.vpSpheres[j]->node == Map.currentArea && j != Map.vSelectedKFs[Map.currentArea])
//                        vertex_j->setFixed(false);
//                    else
//                        vertex_j->setFixed(true);
//                }
                //VertexContainer vertices = optimizer.activeVertices.activeVertices();

                optimizer.optimizeGraph();
                double time_end = pcl::getTime();
                std::cout << " Optimize graph SLAM " << Map.vpSpheres.size() << " took " << double (time_end - time_start) << " s.\n";

//                for(int j=0; j < Map.vpSpheres.size(); j++)
//                    std::cout << "optimizedPose " << j << endl << Map.vOptimizedPoses[j] << endl;

                //Update the optimized poses (for loopClosure detection & visualization)
                {
                    boost::mutex::scoped_lock updateLock(Map.mapMutex);
                    optimizer.getPoses(Map.vOptimizedPoses);
                    //Viewer.currentLocation.matrix() = Map.vOptimizedPoses.back();
                }

//                for(int j=0; j < Map.vpSpheres.size(); j++)
//                    std::cout << "optimizedPose " << j << endl << Map.vOptimizedPoses[j] << endl;

                //Update the optimized pose
                //Eigen::Matrix4f &newPose = Map.vOptimizedPoses.back();
//                std::cout << "First pose optimized \n" << Map.vOptimizedPoses[0] << std::endl;

//                assert(Map.vpSpheres.size() == Map.vOptimizedPoses.size());
                for(int j=0; j < Map.vpSpheres.size(); j++)
                    std::cout << "optimizedPose " << j << " " << optimizer.optimizer.vertex(j)->fixed() << " submap " << Map.vpSpheres[j]->node << std::endl; //<< " same? " << Map.vOptimizedPoses[j]==Map.vTrajectoryPoses[j] ? 1 : 0 << std::endl;
//                std::cout << "newPose \n" << Map.vOptimizedPoses.back() << "\n previous \n" << currentPose << std::endl;

                currentPose = Map.vOptimizedPoses.back();
                bHasNewLC = false;
            }

            // Visibility divission (Normalized-cut)
            if(newLocalFrameID % 4 == 0)
            {
                //                    for(unsigned i=0; i < Map.vTrajectoryPoses.size(); i++)
                //                        cout << "pose " << i << "\n" << Map.vTrajectoryPoses[i] << endl;

                double time_start = pcl::getTime();
                //          cout << "Eval partitions\n";
                topologicalMap.Partitioner();

//                // Set only the current submap's keyframes as optimizable vertex in Graph-SLAM
//                for(int j=0; j < Map.vpSpheres.size(); j++)
//                {
////                    std::cout << "optimizedPose " << poseIdx << " submap " << Map.vpSpheres[j]->node << " same? " << Map.vOptimizedPoses[j]==Map.vTrajectoryPoses[j] ? 1 : 0 << std::endl;
//                    g2o::OptimizableGraph::Vertex *vertex_j = optimizer.optimizer.vertex(j);
//                    if(Map.vpSpheres[j]->node == Map.currentArea && j != Map.vSelectedKFs[Map.currentArea])
//                        vertex_j->setFixed(false);
//                    else
//                        vertex_j->setFixed(true);
//                }

                //                      for(unsigned i=0; i < Map.vTrajectoryPoses.size(); i++)
                //                          cout << "pose " << i << "\n" << Map.vTrajectoryPoses[i] << endl;

                cout << "\tPlaces\n";
                for( unsigned node = 0; node < Map.vsNeighborAreas.size(); node++ )
                {
                    cout << "\t" << node << ":";
                    for( set<unsigned>::iterator it = Map.vsNeighborAreas[node].begin(); it != Map.vsNeighborAreas[node].end(); it++ )
                        cout << " " << *it;
                    cout << endl;
                }

                double time_end = pcl::getTime();
                std::cout << " Eval partitions took " << double (time_end - time_start)*10e3 << "ms" << std::endl;
            }

            //if(Map.vsNeighborAreas.size() > 1)
                //mrpt::system::pause();

//            Viewer.bFreezeFrame = false;
        }

        //      while (!Viewer.viewer.wasStopped() )
        //        boost::this_thread::sleep (boost::posix_time::milliseconds (10));

        cout << "Path length " << Map.vTrajectoryIncrements.back() << endl;
    }

};


void print_help(char ** argv)
{
    cout << "\nThis program performs metric-topological SLAM from the data stream recorded by an omnidirectional RGB-D sensor."
         << " The directory containing the input raw omnidireactional RGB-D images (.frame360 files) has to be specified\n";
    //  cout << "usage: " << argv[0] << " [options] \n";
    //  cout << argv[0] << " -h | --help : shows this help" << endl;
    cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> <selectSample=1> " << endl;
}

int main (int argc, char ** argv)
{
    if(argc < 2 || argc > 3 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
    {
        print_help(argv);
        return 0;
    }

    // Set the path to the directory containing the omnidirectional RGB-D images
    string path_dataset = static_cast<string>(argv[1]);

    // Set the sampling for the recorded frames (e.g. 1 -> all the frames are evaluated; 2-> each other frame is evaluated, etc.)
    int selectSample = 1;
    if(argc == 3)
        selectSample = atoi(argv[2]);

    cout << "Create SphereGraphSLAM object\n";
    SphereGraphSLAM rgbd360_reg_seq;
    rgbd360_reg_seq.run(path_dataset, selectSample);

    cout << " EXIT\n";

    return (0);
}
