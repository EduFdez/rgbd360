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

//#include <pcl/registration/icp.h>
//#include <pcl/registration/icp_nl.h> //ICP LM
//#include <pcl/registration/gicp.h> //GICP

#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>

#define MAX_MATCH_PLANES 25

using namespace std;

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

    // Keyframe object pointer
    Frame360 *candidateKF; // A reference to the last registered frame to add as a keyframe when needed

    // Nearest KF index
    unsigned nearestKF;

    std::pair<Eigen::Matrix4d, Eigen::Matrix<double,6,6> > candidateKF_connection; // The register information of the above
    float candidateKF_sso;

  public:
    SphereGraphSLAM() :
      registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH)),
      candidateKF(NULL),
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
    bool shouldSelectKeyframe(Frame360* currentFrame)
    {
      cout << "shouldSelectKeyframe ...\n";
      double time_start = pcl::getTime();

      double dist_candidateKF = candidateKF_connection.first.block(0,3,3,1).norm();
      Eigen::Matrix4d candidateKF_pose = Map.vTrajectoryPoses[nearestKF] * candidateKF_connection.first;
      map<float,unsigned> nearest_KFs;
      for(std::set<unsigned>::iterator itKF = Map.vsAreas[Map.currentArea].begin(); itKF != Map.vsAreas[Map.currentArea].end(); itKF++) // Set the iterator to the previous KFs not matched yet
      {
        if(*itKF == nearestKF)
          continue;

        double dist_to_prev_KF = (candidateKF_pose.block(0,3,3,1) - Map.vTrajectoryPoses[*itKF].block(0,3,3,1)).norm();
        cout << *itKF << " dist_to_prev_KF " << dist_to_prev_KF << " dist_candidateKF " << dist_candidateKF << endl;
        if( dist_to_prev_KF < dist_candidateKF )
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
      FilterPointCloud filter;

      int frame = 2; // Skip always the first frame, which sometimes contains errors
      int frameOrder = 0;

      Eigen::Matrix4d currentPose = Eigen::Matrix4d::Identity();

      string fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

      // Load first frame
      Frame360* frame360 = new Frame360(&calib);
      frame360->loadFrame(fileName);
      frame360->undistort();
//        frame360->stitchSphericalImage();
      frame360->buildSphereCloud();
      frame360->getPlanes();
//      frame360->id = frame;
      frame360->id = frameOrder;
      frame360->node = Map.currentArea;

      nearestKF = frameOrder;

      filter.filterEuclidean(frame360->sphereCloud);

      Map.addKeyframe(frame360,currentPose);
      Map.vOptimizedPoses.push_back( currentPose );
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
      Map360_Visualizer Viewer(Map);

      // Graph-SLAM
//      std::cout << "\t  mmConnectionKFs " << Map.mmConnectionKFs.size() << " \n";
//      LoopClosure360 loopCloser(Map);
      float areaMatched, currentPlanarArea;
//      Frame360 *candidateKF = new Frame360(&calib); // A reference to the last registered frame to add as a keyframe when needed
//      std::pair<Eigen::Matrix4f, Eigen::Matrix<float,6,6> > candidateKF_connection; // The register information of the above
//      float candidateKF_sso;
//      GraphOptimizer_MRPT optimizer;
//      optimizer.setRigidTransformationType(GraphOptimizer::SixDegreesOfFreedom);
//      std::cout << "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;

//      while( true )
//        boost::this_thread::sleep (boost::posix_time::milliseconds (10));

      bool bPrevKFSelected = true, bGoodTracking = true;
      int lastTrackedKF = 0; // Count the number of not tracked frames after the last registered one

      while( fexists(fileName.c_str()) )
      {

        // Load pointCloud
//        if(!bPrevKFSelected)
        {
        cout << "Frame " << fileName << endl;
          frame360 = new Frame360(&calib);
          frame360->loadFrame(fileName);
          frame360->undistort();
  //        frame360->stitchSphericalImage();
          frame360->buildSphereCloud();
          frame360->getPlanes();
  //        frame360->id = ++frameOrder;
        }

        frame += selectSample;
        fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

        // Register the pair of frames
        bGoodTracking = registerer.RegisterPbMap(Map.vpSpheres[nearestKF], frame360, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
        Matrix4d trackedRelPose = registerer.getPose().cast<double>();
        cout << "bGoodTracking " << bGoodTracking << " with " << nearestKF << ". Distance " << registerer.getPose().block(0,3,3,1).norm() << " entropy " << registerer.calcEntropy() << endl;

        // If the registration is good, do not evaluate this KF and go to the next
//        if( bGoodTracking )
        if( bGoodTracking  && registerer.getMatchedPlanes().size() >= min_planes_registration && registerer.getAreaMatched() > 6 )//&& registerer.getPose().block(0,3,3,1).norm() < 1.5 )
//        if( bGoodTracking && registerer.getMatchedPlanes().size() > 6 && registerer.getPose().block(0,3,3,1).norm() < 1.0 )
        {
          float dist_odometry = 0; // diff_roatation
          if(candidateKF)
            dist_odometry = (candidateKF_connection.first.block(0,3,3,1) - trackedRelPose.block(0,3,3,1)).norm();
//          else
//            dist_odometry = trackedRelPose.norm();
          if( dist_odometry < max_translation_odometry ) // Check that the registration is coherent with the camera movement (i.e. in odometry the registered frames must lay closeby)
          {
            if(trackedRelPose.block(0,3,3,1).norm() > min_dist_keyframes) // Minimum distance to select a possible KF
            {
              if(candidateKF)
                delete candidateKF; // Delete previous candidateKF

              candidateKF = frame360;
              candidateKF_connection.first = trackedRelPose;
              candidateKF_connection.second = registerer.getInfoMat().cast<double>();
              candidateKF_sso = registerer.getAreaMatched() / registerer.areaSource;
            }

            { boost::mutex::scoped_lock updateLockVisualizer(Viewer.visualizationMutex);
              Viewer.currentLocation.matrix() = (currentPose * trackedRelPose).cast<float>();
              Viewer.bDrawCurrentLocation = true;
            }

            bPrevKFSelected = false;
            lastTrackedKF = 0;
          }

          continue; // Eval next frame
        }

        ++lastTrackedKF;
        Viewer.bDrawCurrentLocation = false;

        if(bPrevKFSelected) // If the previous frame was a keyframe, just try to register the next frame as usually
        {
          std::cout << "Tracking lost " << lastTrackedKF << "\n";
//          bGoodTracking = true; // Read a new frame in the next iteration
          bPrevKFSelected = false;
          continue;
        }

        if(lastTrackedKF > 3)
        {
          std::cout << "\n  Launch Relocalization \n";
          double time_start = pcl::getTime();

          if(relocalizer.relocalize(frame360) != -1)
          {
            double time_end = pcl::getTime();
            std::cout << "Relocalization in " << Map.vpSpheres.size() << " KFs map took " << double (time_end - time_start) << std::endl;
            {
              boost::mutex::scoped_lock updateLockVisualizer(Viewer.visualizationMutex);
              Viewer.currentLocation.matrix() = (currentPose * trackedRelPose).cast<float>();
              Viewer.bDrawCurrentLocation = true;
            }

            lastTrackedKF = 0;
            candidateKF = frame360;
            candidateKF_connection.first = relocalizer.registerer.getPose().cast<double>();
            candidateKF_connection.second = relocalizer.registerer.getInfoMat().cast<double>();
            nearestKF = relocalizer.relocKF;
            currentPose = Map.vTrajectoryPoses[nearestKF]; //Map.vOptimizedPoses
            Viewer.currentSphere = nearestKF;
            std::cout << "Relocalized with " << nearestKF << " \n\n\n";

            continue;
          }
        }

        if(!candidateKF)
        {
          std::cout << "_Tracking lost " << lastTrackedKF << "\n";
          continue;
        }

        if( !shouldSelectKeyframe(frame360) )
        {
        cout << "  Do not add a keyframe yet. Now the camera is around KF " << nearestKF << endl;
          candidateKF = NULL;
          currentPose = Map.vTrajectoryPoses[nearestKF]; //Map.vOptimizedPoses
          Viewer.currentSphere = nearestKF;

          frame -= selectSample;
          fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

//          lastTrackedKF = 0;
//          bGoodTracking = true; // Read a new frame in the next iteration
//          bPrevKFSelected = true; // This avoids selecting a KF for which there is no registration
          continue;
        }

        bPrevKFSelected = true;
        frame -= selectSample;
        fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

        cout << "\tGet previous frame as a keyframe " << frameOrder+1 << endl;
        // Add new keyframe
        candidateKF->id = ++frameOrder;
        candidateKF->node = Map.currentArea;

        currentPose = currentPose * candidateKF_connection.first;
//              std::cout<< "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;

        Map.vTrajectoryIncrements.push_back(Map.vTrajectoryIncrements.back() + candidateKF_connection.first.block(0,3,3,1).norm());

        // Filter cloud
        filter.filterEuclidean(candidateKF->sphereCloud);

        // Update topologicalMap
        int newLocalFrameID = Map.vsAreas[Map.currentArea].size();
        int newSizeLocalSSO = newLocalFrameID+1;
        topologicalMap.vSSO[Map.currentArea].setSize(newSizeLocalSSO,newSizeLocalSSO);
        topologicalMap.vSSO[Map.currentArea](newLocalFrameID,newLocalFrameID) = 0.0;

        // Track the current position. Search for Map.mmConnectionKFs in the previous frames TODO: (detect inconsistencies)
        int compareLocalIdx = newLocalFrameID-1;
        // Lock map to add a new keyframe
        {boost::mutex::scoped_lock updateLock(Map.mapMutex);
          Map.addKeyframe(candidateKF, currentPose);
          Map.vOptimizedPoses.push_back( Map.vOptimizedPoses.back() * candidateKF_connection.first );
          std::cout << "\t  mmConnectionKFs loop " << frameOrder << " " << nearestKF << " \n";
          Map.mmConnectionKFs[frameOrder] = std::map<unsigned, std::pair<Eigen::Matrix4d, Eigen::Matrix<double,6,6> > >();
          Map.mmConnectionKFs[frameOrder][nearestKF] = std::pair<Eigen::Matrix4d, Eigen::Matrix<double,6,6> >(candidateKF_connection.first, candidateKF_connection.second);
          Map.vsAreas[Map.currentArea].insert(frameOrder);
        }

        // Update candidateKF
        candidateKF = NULL;
        nearestKF = frameOrder;
        Viewer.currentSphere = nearestKF;

//          cout << "Add tracking SSO " << topologicalMap.vSSO[Map.currentArea].getRowCount() << " " << newLocalFrameID << " " << compareLocalIdx << endl;
        topologicalMap.vSSO[Map.currentArea](newLocalFrameID,compareLocalIdx) = topologicalMap.vSSO[Map.currentArea](compareLocalIdx,newLocalFrameID)
                                                                              = candidateKF_sso;
//            cout << "SSO is " << topologicalMap.vSSO[Map.currentArea](newLocalFrameID,compareLocalIdx) << endl;

//        // Calculate distance of furthest registration (approximated, we do not know if the last frame compared is actually the furthest)
//        std::map<unsigned, pair<unsigned,Eigen::Matrix4f> >::iterator itFurthest = Map.mmConnectionKFs[frameOrder].begin(); // Aproximated
//        cout << "Furthest distance is " << itFurthest->second.second.block(0,3,3,1).norm() << " diffFrameNum " << newLocalFrameID - itFurthest->first << endl;


//        // Synchronize topologicalMap with loopCloser
//        while( !loopCloser.connectionsLC.empty() )
//        {
//          std::map<unsigned, std::map<unsigned, float> >::iterator it_connectLC1 = loopCloser.connectionsLC.begin();
//          while( !it_connectLC1->second.empty() )
//          {
//            std::map<unsigned, float>::iterator it_connectLC2 = it_connectLC1->second.begin();
//            if(Map.vpSpheres[it_connectLC1->first]->node != Map.currentArea || Map.vpSpheres[it_connectLC2->first]->node != Map.currentArea)
//            {
//              loopCloser.connectionsLC.erase(it_connectLC1);
//              continue;
//            }
//            int localID1 = std::distance(Map.vsAreas[Map.currentArea].begin(), Map.vsAreas[Map.currentArea].find(it_connectLC1->first));
//            int localID2 = std::distance(Map.vsAreas[Map.currentArea].begin(), Map.vsAreas[Map.currentArea].find(it_connectLC2->first));
//          cout << "Add LC SSO " << topologicalMap.vSSO[Map.currentArea].getRowCount() << " " << localID1 << " " << localID2 << endl;
//            topologicalMap.vSSO[Map.currentArea](localID1,localID2) = topologicalMap.vSSO[Map.currentArea](localID2,localID1) =
//                                                                      it_connectLC2->second;
//            it_connectLC1->second.erase(it_connectLC2);
//          }
//          loopCloser.connectionsLC.erase(it_connectLC1);
//        }

//        // Visibility divission (Normalized-cut)
//        if(newLocalFrameID % 5 == 0)
//        {
//          double time_start = pcl::getTime();
////          cout << "Eval partitions\n";
//          topologicalMap.Partitioner();
//
//          cout << "\tPlaces\n";
//          for( unsigned node = 0; node < Map.vsNeighborAreas.size(); node++ )
//          {
//            cout << "\t" << node << ":";
//            for( set<unsigned>::iterator it = Map.vsNeighborAreas[node].begin(); it != Map.vsNeighborAreas[node].end(); it++ )
//              cout << " " << *it;
//            cout << endl;
//          }
//
//          double time_end = pcl::getTime();
//          std::cout << " Eval partitions took " << double (time_end - time_start)*10e3 << std::endl;
//        }

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
