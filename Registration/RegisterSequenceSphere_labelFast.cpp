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
 *  Author: Eduardo Fernandez-Moral
 */

#include <Map360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>
#include <RegisterRGBD360.h>

#include <pcl/console/parse.h>

#define MAX_MATCH_PLANES 25
#define VISUALIZE_POINT_CLOUD 1

using namespace std;

unsigned frame;

class RegisterSequenceSphere
{
  private:
    Map360 Map;

    RegisterRGBD360 registerer;

    Calib360 calib;

    Frame360 *frame360_1, *frame360_2;

  public:
    RegisterSequenceSphere() :
        registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH))
    {
      // Get the calibration matrices for the different sensors
      calib.loadExtrinsicCalibration();
      calib.loadIntrinsicCalibration();
    }


    void run(string pathClouds, string pathPbMaps)
    {
//      unsigned frame = 0;
      frame = 0;

      string fileCloud, filePbMap;

      // Skip the first non-labelized frames
      int count_labels = 0;
      while(count_labels == 0)
      {
        fileCloud = mrpt::format("%s/sphereCloud_%d.pcd", pathClouds.c_str(), frame);
        filePbMap = mrpt::format("%s/spherePlanes_%d.pbmap", pathPbMaps.c_str(), frame);
        frame360_2 = new Frame360(&calib);
//        frame360_2->loadPbMap(filePbMap);
        frame360_2->load_PbMap_Cloud(fileCloud, filePbMap);
        for(unsigned i=0; i < frame360_2->planes.vPlanes.size(); i++)
          if(!frame360_2->planes.vPlanes[i].label.empty())
            ++count_labels;
        ++frame;
      }


      // Initialize filters (for visualization)
      FilterPointCloud filter;
      filter.filterEuclidean(frame360_2->sphereCloud);

//    cout << "Size global map " << globalMap->size() << endl;

      bool bGoodRegistration = true;
      Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();

    #if VISUALIZE_POINT_CLOUD
      Map360_Visualizer viewer(Map);

      // Add the first observation to the map
      {boost::mutex::scoped_lock updateLock(Map.mapMutex);

        Map.addKeyframe(frame360_2, currentPose);
//        *viewer.globalMap += *frame360_2->sphereCloud;
//        filter.filterVoxel(viewer.globalMap);
      }
    #endif

//  boost::this_thread::sleep (boost::posix_time::milliseconds (1000));
    float trajectory_length = 0;
//      while( frame < 300 && !bTakeKeyframe)

//      ++frame;
      filePbMap = mrpt::format("%s/spherePlanes_%d.pbmap", pathPbMaps.c_str(), frame);
      fileCloud = mrpt::format("%s/sphereCloud_%d.pcd", pathClouds.c_str(), frame);

      int labelized = 0, unlabelized = 0;
      double time_matching = 0, avLabels = 0;
      while( fexists(filePbMap.c_str()) )
      {
      cout << "Frame " << frame << endl << endl;

//        if(bGoodRegistration)
        {
          frame360_1 = frame360_2;
////          {
////            boost::mutex::scoped_lock updateLock(visualizationMutex);
////            registrationClouds[0] = registrationClouds[1];
////          }
        }

        // Load pointCLoud
        frame360_2 = new Frame360(&calib);
//        frame360_2->loadFrame(filePbMap);
//        frame360_2->undistort();
////        frame360_2->stitchSphericalImage();
//        frame360_2->buildSphereCloud();
//        frame360_2->getPlanes();

//        frame360_2->load_PbMap_Cloud(path, frame);
        frame360_2->load_PbMap_Cloud(fileCloud, filePbMap);
        count_labels = 0;
        for(unsigned i=0; i < frame360_2->planes.vPlanes.size(); i++)
          if(!frame360_2->planes.vPlanes[i].label.empty())
            ++count_labels;

        ++frame;
        filePbMap = mrpt::format("%s/spherePlanes_%d.pbmap", pathPbMaps.c_str(), frame);
        fileCloud = mrpt::format("%s/sphereCloud_%d.pcd", pathClouds.c_str(), frame);

        while(count_labels == 0)
        {
      cout << "NO LABELS\n";

//          frame360_2->load_PbMap_Cloud(path, frame);
          fileCloud = mrpt::format("%s/sphereCloud_%d.pcd", pathClouds.c_str(), frame);
          filePbMap = mrpt::format("%s/spherePlanes_%d.pbmap", pathPbMaps.c_str(), frame);
          for(unsigned i=0; i < frame360_2->planes.vPlanes.size(); i++)
            if(!frame360_2->planes.vPlanes[i].label.empty())
              ++count_labels;
          ++frame;

          ++unlabelized;

          continue;
        }

        ++labelized;
        avLabels += count_labels;

    double time_start = pcl::getTime();

        registerer.Register(frame360_1, frame360_2, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);

    double time_end = pcl::getTime();
    time_matching += (time_end - time_start)*1000;
    std::cout << "registerer took " << double (time_end - time_start) << std::endl;

        currentPose = currentPose * registerer.getPose();

        filter.filterEuclidean(frame360_2->sphereCloud);

      #if VISUALIZE_POINT_CLOUD
        // Add the first observation to the map
        {boost::mutex::scoped_lock updateLock(Map.mapMutex);

          Map.addKeyframe(frame360_2, currentPose);
//          *viewer.globalMap += *frame360_2->sphereCloud;
//          filter.filterVoxel(viewer.globalMap);
        }
      #endif

        bGoodRegistration = true;
        frame360_1->sphereCloud.reset(new pcl::PointCloud<PointT>());
//        delete frame360_1;

      std::cout << "Stats: avTime " << time_matching/labelized << " avLabels " << avLabels/labelized << " \n";
      std::cout << " labelized " << labelized << " unlabelized " << unlabelized << " frame " << frame << " T " << (time_end - time_start)*1000 << " \n";

//        boost::this_thread::sleep (boost::posix_time::milliseconds (500));
    //    mrpt::system::pause();
      }
//      boost::this_thread::sleep (boost::posix_time::milliseconds (2000));
        time_matching /= labelized;
        avLabels /= labelized;
      std::cout << "Stats: avTime " << time_matching << " avLabels " << avLabels << " \n";
      std::cout << " labelized " << labelized << " unlabelized " << unlabelized << " \n";

      cout << "Path length " << trajectory_length << endl;
      delete frame360_2;
    }

};


//void print_help(char ** argv)
//{
//  cout << "\nThis program loads the pointCloud and the frame360_1->planes segmented from an observation RGBD360\n";
//  cout << "usage: " << argv[0] << " [options] \n";
//  cout << argv[0] << " -h | --help : shows this help" << endl;
////  cout << argv[0] << " <pathToPointCloud> <pathToPbMap>" << endl;
//}

int main (int argc, char ** argv)
{
//  if(argc != 1)
//    print_help(argv);

  string pathClouds = static_cast<string>(argv[1]);
  string pathPbMaps = static_cast<string>(argv[2]);

cout << "Create RegisterSequenceSphere object\n";
  RegisterSequenceSphere rgbd360_reg_seq;
  rgbd360_reg_seq.run(pathClouds, pathPbMaps);

cout << " EXIT\n";
  return (0);
}
