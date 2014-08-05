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

#define RECORD_VIDEO 0

#include <Map360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>
#include <RegisterRGBD360.h>

//#include <pcl/registration/icp.h>
//#include <pcl/registration/icp_nl.h> //ICP LM
//#include <pcl/registration/gicp.h> //GICP

#include <pcl/console/parse.h>

#define MAX_MATCH_PLANES 25
#define SAVE_TRAJECTORY 0
#define SAVE_IMAGES 0
#define VISUALIZE_POINT_CLOUD 1

using namespace std;

/*! This class' main function 'run' performs PbMap-based odometry with the input data of a stream of
 *  omnidirectional RGB-D images (from a recorded sequence).
 */
class Odometry360
{
  private:
    Map360 Map;

    RegisterRGBD360 registerer;

    Calib360 calib;

    Frame360 *frame360_1, *frame360_2;

  public:
    Odometry360() :
      registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH))
    {
      // Get the calibration matrices for the different sensors
      calib.loadExtrinsicCalibration();
      calib.loadIntrinsicCalibration();
    }

    void run(string &path_dataset, string &path_results, const int &selectSample)
    {
      unsigned frame = 1;
      unsigned frameOrder = frame+1;

      string fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
      cout << "Frame " << fileName << endl;

      frame360_2 = new Frame360(&calib);
      frame360_2->loadFrame(fileName);
      frame360_2->undistort();
//      frame360_2->stitchSphericalImage();
      frame360_2->buildSphereCloud();
      frame360_2->getPlanes();
//    cout << "regsitrationCloud has " << registrationClouds[1]->size() << " Pts\n";

      bool bGoodRegistration = true;
      Eigen::Matrix4d currentPose = Eigen::Matrix4d::Identity();

//      // ICP parameters
////        pcl::IterativeClosestPoint<PointT,PointT> icp;
//      pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
////        pcl::IterativeClosestPointNonLinear<PointT,PointT> icp;
//
//      icp.setMaxCorrespondenceDistance (0.3);
//      icp.setMaximumIterations (20);
//      icp.setTransformationEpsilon (1e-5);
////        icp.setEuclideanFitnessEpsilon (1);
//      icp.setRANSACOutlierRejectionThreshold (0.1);
//
////       boost::shared_ptr<pcl::WarpPointRigid3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA> > warp_fcn
////            (new pcl::WarpPointRigid3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA>);
////
////       // Create a TransformationEstimationLM object, and set the warp to it
////       boost::shared_ptr<pcl::registration::TransformationEstimationLM<pcl::PointXYZRGBA, pcl::PointXYZRGBA> > te (new pcl::registration::TransformationEstimationLM<pcl::PointXYZRGBA, pcl::PointXYZRGBA>);
////       te->setWarpFunction (warp_fcn);
////
////       // Pass the TransformationEstimation objec to the ICP algorithm
////       icp->setTransformationEstimation (te);


    // Filter the spherical point cloud
    FilterPointCloud filter(0.2); // Initialize filters (for visualization)
    filter.filterEuclidean(frame360_2->sphereCloud);
    filter.filterVoxel(frame360_2->sphereCloud);

    // Visualize
    #if VISUALIZE_POINT_CLOUD
      Map360_Visualizer viewer(Map);

      // Add the first observation to the map
      {boost::mutex::scoped_lock updateLock(Map.mapMutex);

        Map.addKeyframe(frame360_2, currentPose);
        *viewer.globalMap += *frame360_2->sphereCloud;
      }

    #endif

      frame += selectSample;
      fileName = mrpt::format("%s/sphere_images_%d.bin", path_dataset.c_str(), frame);
//      fileName = mrpt::format("%s/sphere_images_%d.bin", path_dataset.c_str(), ++frame);

      while( fexists(fileName.c_str()) )
      {
      cout << "Frame " << fileName << endl;

        if(bGoodRegistration)
        {
          frame360_1 = frame360_2;
        }

        // Load pointCLoud
        frame360_2 = new Frame360(&calib);
        frame360_2->loadFrame(fileName);
        frame360_2->undistort();
//        frame360_2->stitchSphericalImage();
        frame360_2->buildSphereCloud();
        frame360_2->getPlanes();

        #if SAVE_IMAGES
          frame360_2->stitchSphericalImage();
          cv::imwrite(path_results + mrpt::format("/rgb_%03d.png",frameOrder), frame360_2->sphereRGB);
          frame360_2->sphereDepth.convertTo( frame360_2->sphereDepth, CV_16U, 1000 ); //CV_16UC1
          cv::imwrite(path_results + mrpt::format("/depth_%03d.png",frameOrder), frame360_2->sphereDepth);
        #endif

        // Align the two frames
        bGoodRegistration = registerer.RegisterPbMap(frame360_1, frame360_2, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
//        bGoodRegistration = registerer.RegisterPbMap(frame360_1, frame360_2, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_ODOMETRY_3DoF);
        Eigen::Matrix4d rigidTransf = registerer.getPose().cast<double>();

        if(!bGoodRegistration)
        {
          cout << "\tBad registration\n\n";
          fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",++frame);
          bGoodRegistration = false;
          delete frame360_2;
          continue;
        }

        // Filter the spherical point cloud
        filter.filterEuclidean(frame360_2->sphereCloud);
//        filter.filterVoxel(frame360_2->sphereCloud);
//
//        // ICP
//        icp.setInputSource(frame360_1->sphereCloud);
//        icp.setInputTarget(frame360_2->sphereCloud);
//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr alignedICP(new pcl::PointCloud<pcl::PointXYZRGBA>);
////        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
////        Eigen::Matrix4f transformation = rigidTransf;
//        icp.align (*alignedICP, rigidTransf);
//      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
////        transformation = icp.getFinalTransformation ();
//      cout << "ICP transformation:\n" << icp.getFinalTransformation() << endl << "PbMap-Registration\n" << rigidTransf << endl;

        // Update currentPose
        currentPose = currentPose * rigidTransf;

        #if VISUALIZE_POINT_CLOUD
        {boost::mutex::scoped_lock updateLock(Map.mapMutex);

          Map.addKeyframe(frame360_2, currentPose); // Add keyframe

          *viewer.globalMap += *frame360_2->sphereCloud;
          filter.filterVoxel(viewer.globalMap);
        }
//        #elif
//          delete frame360_1;
        #endif

        #if SAVE_TRAJECTORY
          rigidTransf.saveToTextFile(mrpt::format("%s/Rt_%d_%d.txt", path_results.c_str(), frameOrder-1, frameOrder).c_str());
//          ofstream saveRt;
//          saveRt.open(mrpt::format("%s/poses/Rt_%d_%d.txt", path_dataset.c_str(), frame-1, frame).c_str());
//          saveRt << rigidTransf;
//          saveRt.close();
        #endif

        ++frameOrder;
        frame += selectSample;
//        fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",++frame);
        fileName = path_dataset + mrpt::format("/sphere_images_%d.bin",frame);
      cout << "fileName " << fileName << endl;

//        boost::this_thread::sleep (boost::posix_time::milliseconds (500));
    //    mrpt::system::pause();
      }

//      delete frame360_2;
    }

};


void print_help(char ** argv)
{
  cout << "\nThis program performs PbMap-based Odometry from the data stream recorded by an omnidirectional RGB-D sensor.\n\n";

  cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> <pathToResults> <sampleStream> " << endl;
  cout << "    <pathToRawRGBDImagesDir> is the directory containing the data stream as a set of '.bin' files" << endl;
  cout << "    <pathToResults> is the directory where results (PbMap, Spherical PNG images, Poses files, etc.) should be saved" << endl;
  cout << "    <sampleStream> is the sampling step used in the dataset (e.g. 1: all the frames are used " << endl;
  cout << "                                                                  2: each other frame is used " << endl;
  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}

int main (int argc, char ** argv)
{
  if(argc != 4 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  string path_dataset = static_cast<string>(argv[1]);
  string path_results = static_cast<string>(argv[2]);
  int selectSample = atoi(argv[3]);

  cout << "Create Odometry360 object\n";
  Odometry360 odometry360;
  odometry360.run(path_dataset, path_results, selectSample);

  cout << " EXIT\n";

  return (0);
}
