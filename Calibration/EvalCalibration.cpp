/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR
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

#include <Frame360.h>
#include <Map360_Visualizer.h>
#include <FilterPointCloud.h>
#include <RegisterRGBD360.h>
//#include <Calibrator.h>

#include <pcl/console/parse.h>
//#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h> //ICP LM
#include <pcl/registration/gicp.h> //GICP

#define VISUALIZE_POINT_CLOUD 0
#define MAX_MATCH_PLANES 25

using namespace std;

void print_help(char ** argv)
{
  cout << "\nThis program evaluates the extrinsic calibration of the RGBD360 sensor using the residual error from ICP alignment of several nearby frames "
       << "(so that the non-overlapping content is ngligible)\n";

  cout << "  usage: " << argv[0] << " <pathToRawRGBDImagesDir> <path_extrinsicCalib> <useIntrinsicModel> <first_frame> <last_frame> " << endl;
  cout << "    <pathToRawRGBDImagesDir> is the directory containing the data stream as a set of '.bin' files" << endl;
  cout << "    <path_extrinsicCalib> is the directory containing the extrinsic matrices" << endl;
  cout << "    <useIntrinsicModel> uses the intrinsic calibration when set to '1'" << endl;
  cout << "    <first_frame> first frame of the sequence to be compared" << endl;
  cout << "    <last_frame> last frame of the sequence to be compared" << endl;
  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}

// Tested initially in sequence Khan0_10fps, between the frames 290-350 and 655-745

int main (int argc, char ** argv)
{
  if(argc != 6 || pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
  {
    print_help(argv);
    return 0;
  }

  string path_dataset = static_cast<string>(argv[1]);
  string path_extrinsicCalib = static_cast<string>(argv[2]);
  int useIntrinsicModel = atoi(argv[3]);
  int first_frame = atoi(argv[4]);
  int last_frame = atoi(argv[5]);
  assert(last_frame > first_frame);

  int sampleDataset = 3;

  cout << "Evaluate Calibration RGBD360 multisensor\n";

  Calib360 calib;
  if(useIntrinsicModel == 1)
    calib.loadIntrinsicCalibration();
  calib.loadExtrinsicCalibration(path_extrinsicCalib);

  RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));

  FilterPointCloud filter(0.025); // Create filter

  // ICP parameters
  pcl::GeneralizedIterativeClosestPoint<PointT,PointT> icp;
//        pcl::IterativeClosestPointNonLinear<PointT,PointT> icp;
  icp.setMaxCorrespondenceDistance(0.4);
  icp.setMaximumIterations(10);
  icp.setTransformationEpsilon(1e-7);
//      icp.setRotationEpsilon (1e-5);
//        icp.setEuclideanFitnessEpsilon (1);
//  icp.setRANSACOutlierRejectionThreshold(0.1);

//       boost::shared_ptr<pcl::WarpPointRigid3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA> > warp_fcn
//            (new pcl::WarpPointRigid3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA>);
//
//       // Create a TransformationEstimationLM object, and set the warp to it
//       boost::shared_ptr<pcl::registration::TransformationEstimationLM<pcl::PointXYZRGBA, pcl::PointXYZRGBA> > te (new pcl::registration::TransformationEstimationLM<pcl::PointXYZRGBA, pcl::PointXYZRGBA>);
//       te->setWarpFunction (warp_fcn);
//
//       // Pass the TransformationEstimation objec to the ICP algorithm
//       icp->setTransformationEstimation (te);


  vector<Frame360> vFrames360;
  vector<Eigen::Matrix4f> vPoses;
  Eigen::Matrix4f currentPose = Eigen::Matrix4f::Identity();
  vPoses.push_back(currentPose);
  float avScoreFitness = 0;
  unsigned numPairs = 0;
  for(int frame2=first_frame; frame2 < last_frame; frame2+=sampleDataset)
  {
    string fileName = mrpt::format("%s/sphere_images_%d.bin", path_dataset.c_str(), frame2);
    cout << "Frame " << fileName << endl;

    if( !fexists(fileName.c_str()) )
    {
      cout << "Frame " << fileName << " does not exist\n";
      return 0;
    }

    // Load frame
    Frame360 frame360(&calib);
    frame360.loadFrame(fileName);
    if(useIntrinsicModel == 1)
      frame360.undistort();
    frame360.buildSphereCloud();
    frame360.getPlanes();

    // Filter the spherical point cloud
    filter.filterVoxel(frame360.sphereCloud);
//    std::vector<int> indices;
//    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr aux(new pcl::PointCloud<pcl::PointXYZRGBA>);
//    pcl::removeNaNFromPointCloud(*frame360.sphereCloud, *aux, indices);
//    cout << "Filtered 153600 -> " << aux->points.size() << endl;
//    frame360.sphereCloud = aux;

    vFrames360.push_back(frame360);

//    unsigned frame1 = frame2 - sampleDataset;
//    if(frame1 >= first_frame)
    for(int frame1=first_frame; frame1 < frame2; frame1+=sampleDataset)
    {
      // Align the two frames
      bool bGoodRegistration = registerer.Register(&vFrames360[(frame1-first_frame)/sampleDataset], &frame360, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_3DoF);
      Eigen::Matrix4f rigidTransf = registerer.getPose();

      if(!bGoodRegistration)
      {
        cout << "\tBad registration\n\n";
        continue;
      }
      cout << "\tPbMap Matched between " << frame1 << " " << frame2 << endl;

      // ICP
    double time_start = pcl::getTime();
      icp.setInputSource(frame360.sphereCloud);
      icp.setInputTarget(vFrames360[(frame1-first_frame)/sampleDataset].sphereCloud);
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr alignedICP(new pcl::PointCloud<pcl::PointXYZRGBA>);
    cout << "Sizes " << frame360.sphereCloud->points.size() << " and " << vFrames360[(frame1-first_frame)/sampleDataset].sphereCloud->points.size() << endl;
//        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
      icp.align (*alignedICP, rigidTransf);
    double time_end = pcl::getTime();
    std::cout << "ICP took " << double (time_end - time_start) << std::endl;
      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      rigidTransf = icp.getFinalTransformation();
      cout << "ICP transformation:\n" << icp.getFinalTransformation() << endl << "PbMap-Registration\n" << rigidTransf << endl;

      avScoreFitness += icp.getFitnessScore();
      numPairs++;

      // Update currentPose
      if(frame1 == frame2 - sampleDataset)
      {
        currentPose = currentPose * rigidTransf;
        vPoses.push_back(currentPose);
      }
    }
  }

  avScoreFitness /= numPairs;
  cout << "avScoreFitness " << avScoreFitness << endl;

  #if VISUALIZE_POINT_CLOUD
    Map360 Map;
    {boost::mutex::scoped_lock updateLock(Map.mapMutex);

      for(int frame=0; frame < vFrames360.size(); frame++)
        Map.addKeyframe(&vFrames360[frame], vPoses[frame]); // Add keyframe

//        *viewer.globalMap += *frame360.sphereCloud;
//        filter.filterVoxel(viewer.globalMap);
    }
    Map360_Visualizer viewer(Map);
    while (!viewer.viewer.wasStopped() )
      boost::this_thread::sleep (boost::posix_time::milliseconds (10));
  #endif

  cout << "EXIT\n";

  return (0);
}
