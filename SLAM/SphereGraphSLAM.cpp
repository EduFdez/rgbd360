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
 * Author: Eduardo Fernandez-Moral
 */

#include <mrpt/utils/CStream.h>

#include <Map360.h>
#include <Map360_Visualizer.h>
#include <TopologicalMap360.h>
#include <LoopClosure360.h>
#include <FilterPointCloud.h>
#include <FilterPointCloud.h>
#include <GraphOptimizer.h>
//#include <GraphOptimizer_G2O.h>

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

  public:
    SphereGraphSLAM()
    {

    }

    ~SphereGraphSLAM()
    {
      // Clean memory
      for(unsigned i=0; i < Map.vpSpheres.size(); i++)
        delete Map.vpSpheres[i];
    }

    void run(string path, const int &selectSample)
    {
      std::cout << "SphereGraphSLAM::run... " << std::endl;

      // Create the topological arrangement object
      TopologicalMap360 topologicalMap(Map);
      Map.currentArea = 0;

      // Get the calibration matrices for the different sensors
      Calib360 calib;
      calib.loadExtrinsicCalibration();
      calib.loadIntrinsicCalibration();

      // Create registration object
      RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));

      // Filter the point clouds before visualization
      FilterPointCloud filter;

      int frame = 2; // Skip always the first frame, which sometimes contains errors
      int frameOrder = 0;
      int numCheckRegistration = 5; // Check 'numCheckRegistration' frames backwards registration
      unsigned noAssoc_threshold = 40; // Limit the number of correspondences that are checked backwards when noAssoc_threshold are not associated

      float trajectory_length = 0;
      Eigen::Matrix4d currentPose = Eigen::Matrix4d::Identity();

      string fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

      // Load first frame
      Frame360* frame360 = new Frame360(&calib);
      frame360->loadFrame(fileName);
      frame360->undistort();
//        frame360->stitchSphericalImage();
      frame360->buildSphereCloud();
      frame360->getPlanes();
      frame360->id = frameOrder;
      frame360->node = Map.currentArea;

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
      LoopClosure360 loopCloser(Map);
      float areaMatched, currentPlanarArea;
//      GraphOptimizer_MRPT optimizer;
//      optimizer.setRigidTransformationType(GraphOptimizer::SixDegreesOfFreedom);
//      std::cout << "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;

//      while( true )
//        boost::this_thread::sleep (boost::posix_time::milliseconds (10));
      while( fexists(fileName.c_str()) )
      {
      cout << "Frame " << fileName << endl;

        // Load pointCloud
        frame360 = new Frame360(&calib);
        frame360->loadFrame(fileName);
        frame360->undistort();
//        frame360->stitchSphericalImage();
        frame360->buildSphereCloud();
        frame360->getPlanes();
        frame360->id = ++frameOrder;
        frame360->node = Map.currentArea;
        int newLocalFrameID = Map.vsAreas[Map.currentArea].size();
        int newSizeLocalSSO = newLocalFrameID+1;
        topologicalMap.vSSO[Map.currentArea].setSize(newSizeLocalSSO,newSizeLocalSSO);
        topologicalMap.vSSO[Map.currentArea](newLocalFrameID,newLocalFrameID) = 0.0;
//        currentPlanarArea = frame360->getPlanarArea();

        // Track the current position. Search for Map.mmConnectionKFs in the previous frames TODO: (detect inconsistencies)
        int compareLocalIdx = newLocalFrameID-1;
        unsigned noAssoc= 0;
        bool frameRegistered = false;
        set<unsigned>::reverse_iterator compareSphereId = Map.vsAreas[Map.currentArea].rbegin();
//      cout << "compareSphereId " << *compareSphereId << endl;
        while(compareLocalIdx >= 0 && (compareLocalIdx >= newLocalFrameID-numCheckRegistration) && noAssoc < noAssoc_threshold)
        {
          std::cout<< "Register  " << frame360->id << " with " << compareLocalIdx << std::endl;

          // Register the pair of frames
          if( registerer.RegisterPbMap(Map.vpSpheres[*compareSphereId], frame360, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_ODOMETRY_3DoF) )
          {
//            if(registerer.getMatchedPlanes().size() >= 8) //Do not select as a keyframe
//              break;
            cout << "Good TRACKING between " << frameOrder << " " << *compareSphereId << endl;

            Matrix4d trackedRelPose = registerer.getPose().cast<double>();
            currentPose = currentPose * trackedRelPose;
//              std::cout<< "Added vertex: "<< optimizer.addVertex(currentPose) << std::endl;


            Map.vTrajectoryIncrements.push_back(Map.vTrajectoryIncrements.back() + trackedRelPose.block(0,3,3,1).norm());
            trajectory_length += trackedRelPose.block(0,3,3,1).norm();

            // Filter cloud
            filter.filterEuclidean(frame360->sphereCloud);

            // Lock map to add a new keyframe
            {boost::mutex::scoped_lock updateLock(Map.mapMutex);
              Map.addKeyframe(frame360, currentPose);
              Map.vOptimizedPoses.push_back( Map.vOptimizedPoses.back() * trackedRelPose );
              std::cout << "\t  mmConnectionKFs loop " << frameOrder << " " << *compareSphereId << " \n";
              Map.mmConnectionKFs[frameOrder] = std::map<unsigned, std::pair<Eigen::Matrix4d, Eigen::Matrix<double,6,6> > >();
              Map.mmConnectionKFs[frameOrder][*compareSphereId] = std::pair<Eigen::Matrix4d, Eigen::Matrix<double,6,6> >(trackedRelPose, registerer.getInfoMat().cast<double>());
              Map.vsAreas[Map.currentArea].insert(frameOrder);
            }

            frameRegistered = true;

          cout << "Add tracking SSO " << topologicalMap.vSSO[Map.currentArea].getRowCount() << " " << newLocalFrameID << " " << compareLocalIdx << endl;
            topologicalMap.vSSO[Map.currentArea](newLocalFrameID,compareLocalIdx) = topologicalMap.vSSO[Map.currentArea](compareLocalIdx,newLocalFrameID)
                                                                                  = registerer.getAreaMatched() / registerer.areaSource;
//            cout << "SSO is " << topologicalMap.vSSO[Map.currentArea](newLocalFrameID,compareLocalIdx) << endl;
//            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationMatrix = registerer.getAreaMatched() * Eigen::Matrix<double,6,6>::Identity();
//            Eigen::Matrix4f relativePose = registerer.getPose();
//            Eigen::Matrix<double,6,6> informationMatrix = registerer.getInfoMat();
//            optimizer.addEdge(compareLocalIdx, newLocalFrameID, relativePose, informationMatrix);
//            std::cout << "Add edge between " << compareLocalIdx << " and " << newLocalFrameID << "\n";

            break; // Stop the loop when there is a valid registration
          }
          else
            ++noAssoc;

          --compareLocalIdx;
          ++compareSphereId;
        }
        if(!frameRegistered) //
        {
          cout << "No registration available for " << fileName << endl;

          topologicalMap.vSSO[Map.currentArea].setSize(newLocalFrameID,newLocalFrameID);

          frame += selectSample;
          --frameOrder;
          fileName = path + mrpt::format("/sphere_images_%d.bin",frame);

          continue;
//          assert(false);
        }
//        // Calculate distance of furthest registration (approximated, we do not know if the last frame compared is actually the furthest)
//        std::map<unsigned, pair<unsigned,Eigen::Matrix4f> >::iterator itFurthest = Map.mmConnectionKFs[frameOrder].begin(); // Aproximated
//        cout << "Furthest distance is " << itFurthest->second.second.block(0,3,3,1).norm() << " diffFrameNum " << newLocalFrameID - itFurthest->first << endl;


        // Synchronize topologicalMap with loopCloser
        while( !loopCloser.connectionsLC.empty() )
        {
          std::map<unsigned, std::map<unsigned, float> >::iterator it_connectLC1 = loopCloser.connectionsLC.begin();
//          currentPlanarArea = Map.vpSpheres[it_connectLC1->first]->getPlanarArea();
          while( !it_connectLC1->second.empty() )
          {
            std::map<unsigned, float>::iterator it_connectLC2 = it_connectLC1->second.begin();
            if(Map.vpSpheres[it_connectLC1->first]->node != Map.currentArea || Map.vpSpheres[it_connectLC2->first]->node != Map.currentArea)
            {
              loopCloser.connectionsLC.erase(it_connectLC1);
              continue;
            }
            int localID1 = std::distance(Map.vsAreas[Map.currentArea].begin(), Map.vsAreas[Map.currentArea].find(it_connectLC1->first));
            int localID2 = std::distance(Map.vsAreas[Map.currentArea].begin(), Map.vsAreas[Map.currentArea].find(it_connectLC2->first));
          cout << "Add LC SSO " << topologicalMap.vSSO[Map.currentArea].getRowCount() << " " << localID1 << " " << localID2 << endl;
            topologicalMap.vSSO[Map.currentArea](localID1,localID2) = topologicalMap.vSSO[Map.currentArea](localID2,localID1) =
                                                                      it_connectLC2->second;
            it_connectLC1->second.erase(it_connectLC2);
          }
          loopCloser.connectionsLC.erase(it_connectLC1);
        }

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

//        ++frameOrder;
        frame += selectSample;
        fileName = path + mrpt::format("/sphere_images_%d.bin",frame);
      }

//      while (!Viewer.viewer.wasStopped() )
//        boost::this_thread::sleep (boost::posix_time::milliseconds (10));

      cout << "Path length " << trajectory_length << endl;
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
