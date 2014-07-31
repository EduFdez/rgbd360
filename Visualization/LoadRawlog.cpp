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
 *  Author: efernand Fernandez-Moral
 */

#include <mrpt/gui.h>
#include <mrpt/opengl.h>
#include <mrpt/utils/CFileGZInputStream.h>
#include <mrpt/slam/CObservation3DRangeScan.h>
#include <mrpt/slam/CObservationRGBD360.h>
//#include <mrpt/slam/CObservationIMU.h>
#include <mrpt/slam/CRawlog.h>
//#include <mrpt/slam/CActionCollection.h>
//#include <mrpt/slam/CSensoryFrame.h>

#include <Frame360.h>
#include <Frame360_Visualizer.h>

#ifndef PI
  #define PI 3.14159265
#endif

using namespace std;
using namespace mrpt;
using namespace mrpt::slam;
using namespace mrpt::utils;
//using namespace mrpt::hwdrivers;

// This simple demo opens a rawlog dataset containing the data recorded
// by several rgbd sensors and build the calibrated rgbd images from it

int main ( int argc, char** argv )
{
  try
  {
    if (argc != 2)
    {
      cerr << "Usage: " << argv[0] << " <path_to_rawlog_dataset\n";
      return 1;
    }

    const string RAWLOG_FILENAME = string( argv[1] );
    const unsigned num_sensors = 4;

//    unsigned SensorArrangement[] = {1,3,2,0};
    unsigned SensorArrangement[] = {3,0,2,1,3,0,2,1};

    // Set the sensor poses (Extrinsic calibration)
    mrpt::poses::CPose3D sensorPoses[num_sensors];
    math::CMatrixDouble44 pose_sensor_mat[num_sensors];
    Eigen::Matrix4d pose_sensor_mat0 = Eigen::Matrix4d::Identity(); pose_sensor_mat0.block(0,3,3,1) << 0.055, 0, 0;
    pose_sensor_mat[0] = math::CMatrixDouble44(pose_sensor_mat0);

    //TODO: Load proper calibration of the ominidirectional RGBD device
    Eigen::Matrix4d Rt_45 = Eigen::Matrix4d::Identity();
    Rt_45(0,0) = Rt_45(2,2) = cos(45*PI/180);
    Rt_45(0,2) = sin(45*PI/180);
    Rt_45(2,0) = -Rt_45(0,2);

    for(unsigned i=1; i < num_sensors; i++){
      pose_sensor_mat[i] = math::CMatrixDouble44(Rt_45) * pose_sensor_mat[i-1];
//      cout << "Sensor pose \n" << pose_sensor_mat[i].getEigenBase() << endl;
    }

    for(unsigned i=0; i < num_sensors; i++)
      sensorPoses[i] = mrpt::poses::CPose3D(pose_sensor_mat[SensorArrangement[i]]);

    CFileGZInputStream rawlogFile(RAWLOG_FILENAME);
    CActionCollectionPtr action;
    CSensoryFramePtr observations;
    CObservationPtr observation;
    size_t rawlogEntry=0;
    //bool end = false;

    CObservation3DRangeScanPtr obsRGBD[4];  // Pointers to the 4 images that compose an observation
    bool rgbd1 = false, rgbd2 = false, rgbd3 = false, rgbd4 = false;
    CObservation2DRangeScanPtr laserObs;    // Pointer to the laser observation
    const int decimation = 1;
    int num_observations = 0, num_rgbd360_obs = 0;

    // Create window and prepare OpenGL object in the scene:
    // --------------------------------------------------------
    bool bVisualize = false;
    mrpt::gui::CDisplayWindow3D  win3D("OpenNI2 3D view",800,600);

    win3D.setCameraAzimuthDeg(140);
    win3D.setCameraElevationDeg(20);
    win3D.setCameraZoom(8.0);
    win3D.setFOV(90);
    win3D.setCameraPointingToPoint(2.5,0,0);

//    mrpt::opengl::CPointCloudColouredPtr gl_points = mrpt::opengl::CPointCloudColoured::Create();
//    gl_points->setPointSize(2.5);
    mrpt::opengl::CPointCloudColouredPtr gl_points[num_sensors];
    for(unsigned i=0; i < num_sensors; i++)
    {
      gl_points[i] = mrpt::opengl::CPointCloudColoured::Create();
      gl_points[i]->setPointSize(2.5);
    }

    opengl::COpenGLViewportPtr viewInt; // Extra viewports for the RGB images.
    {
      mrpt::opengl::COpenGLScenePtr &scene = win3D.get3DSceneAndLock();

      // Create the Opengl object for the point cloud:
//      scene->insert( gl_points );
      for(unsigned i=0; i < num_sensors; i++)
        scene->insert( gl_points[i] );

      scene->insert( mrpt::opengl::CGridPlaneXY::Create() );
      scene->insert( mrpt::opengl::stock_objects::CornerXYZ() );

      const double aspect_ratio =  480.0 / 640.0;
      const int VW_WIDTH = 400;	// Size of the viewport into the window, in pixel units.
      const int VW_HEIGHT = aspect_ratio*VW_WIDTH;

      // Create an extra opengl viewport for the RGB image:
//      viewInt = scene->createViewport("view2d_int");
//      viewInt->setViewportPosition(5, 30, VW_WIDTH,VW_HEIGHT );
      win3D.addTextMessage(10, 30+VW_HEIGHT+10,"Intensity data",TColorf(1,1,1), 2, MRPT_GLUT_BITMAP_HELVETICA_12 );

      win3D.addTextMessage(5,5,
        format("'o'/'i'-zoom out/in, ESC: quit"),
          TColorf(0,0,1), 110, MRPT_GLUT_BITMAP_HELVETICA_18 );

      win3D.unlockAccess3DScene();
      win3D.repaint();
    }

    // Create rgbd360 structures
    Calib360 calib;
    calib.loadIntrinsicCalibration();
    calib.loadExtrinsicCalibration();
//    for(unsigned i=0; i < num_sensors; i++)
//      cout << calib.getRt_id(i) << endl << endl;

//    Frame360 frame360(&calib);
    // For PointCloud visualization
    Frame360 *frame360 = new Frame360(&calib);
    Frame360_Visualizer *SphereViewer;

    unsigned mode = 3;
    unsigned frame = 0;
    string path_results = "/home/edu";
    if(mode == 1 || mode == 2)
    {
      cv::namedWindow( "sphereRGB" );
      cv::namedWindow( "sphereDepth", CV_WINDOW_AUTOSIZE );
    }
    else if(mode == 3 || mode == 5)
    {
      SphereViewer = new Frame360_Visualizer(frame360);
      SphereViewer->frameIdx = frame;
    }

    while ( CRawlog::getActionObservationPairOrObservation(
                                                 rawlogFile,      // Input file
                                                 action,            // Possible out var: action of a pair action/obs
                                                 observations,  // Possible out var: obs's of a pair action/obs
                                                 observation,    // Possible out var: a single obs.
                                                 rawlogEntry    // Just an I/O counter
                                                 ) )
    {
      // Process action & observations
      if (observation)
      {
        // assert(IS_CLASS(observation, CObservation2DRangeScan) || IS_CLASS(observation, CObservation3DRangeScan));
        ++num_observations;
//        cout << "Observation " << num_observations++ << " timestamp " << observation->timestamp << endl;

        // TODO: Get closest frames in time (more tight synchronization)

        if(observation->sensorLabel == "RGBD1")
        {
          obsRGBD[0] = CObservation3DRangeScanPtr(observation);
          rgbd1 = true;
        }
        if(observation->sensorLabel == "RGBD2")
        {
          obsRGBD[1] = CObservation3DRangeScanPtr(observation);
          rgbd2 = true;
        }
        if(observation->sensorLabel == "RGBD3")
        {
          obsRGBD[2] = CObservation3DRangeScanPtr(observation);
          rgbd3 = true;
        }
        if(observation->sensorLabel == "RGBD4")
        {
          obsRGBD[3] = CObservation3DRangeScanPtr(observation);
          rgbd4 = true;
        }
        else if(observation->sensorLabel == "LASER")
        {
          laserObs = CObservation2DRangeScanPtr(observation);
        }
      }
      else
      {
      // action, observations should contain a pair of valid data (Format #1 rawlog file)
        THROW_EXCEPTION("Not a valid observation");
      }

      if(!(rgbd1 && rgbd2 && rgbd3 && rgbd4))
        continue;

      rgbd1 = rgbd2 = rgbd3 = rgbd4 = false; // Reset the counter of simultaneous observations

      // Apply decimation
      num_rgbd360_obs++;
      if(num_rgbd360_obs%decimation != 0)
        continue;

    cout << "Observation " << num_observations << " timestamp " << observation->timestamp << endl;

      // Fill the frame180 structure
      CObservationRGBD360 obs360;
//      for(unsigned i=0; i<obs360.NUM_SENSORS; i++)
      for(unsigned i=0; i < 8; i++)
      {
        obs360.rgbd[i] = *obsRGBD[SensorArrangement[i]];
        obs360.rgbd[i].sensorPose = sensorPoses[SensorArrangement[i]];
cout << "Load img " << i << endl;
      }

      // Load Frame360 structure
cout << "Load Frame360\n" << fflush;
//      frame360->loadFrame(fileName);
//      for(unsigned sensor_id=0; sensor_id < num_sensors; ++sensor_id)
      for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
      {
        mrpt::utils::CImage intensityImage;
//        cv::Mat rgb = cv::cvarrToMat( obs360.rgbd[sensor_id].intensityImage.getAs<IplImage>(), false /* dont copy buffers */ );
        cv::Mat rgb( obs360.rgbd[sensor_id].intensityImage.getAs<IplImage>() );

//        cv::imshow( "test", rgb );
//        cv::waitKey(1);

//        mrpt::math::CMatrix obs360.rgbd[sensor_id].rangeImage;
//        cv::Mat depth(obs360.rgbd[sensor_id].rangeImage.rows(), obs360.rgbd[sensor_id].rangeImage.cols(), CV_16UC1);
        cv::Mat depth(obs360.rgbd[sensor_id].rangeImage.rows(), obs360.rgbd[sensor_id].rangeImage.cols(), CV_32FC1);
        for(unsigned r=0; r < obs360.rgbd[sensor_id].rangeImage.rows(); r++)
          for(unsigned c=0; c < obs360.rgbd[sensor_id].rangeImage.cols(); c++)
            depth.at<float>(r,c) = obs360.rgbd[sensor_id].rangeImage(r,c);
        cv::Mat depthInMeters = cv::Mat(depth.rows, depth.cols, CV_16UC1);
        depth.convertTo( depthInMeters, CV_16UC1, 1000 ); //CV_16UC1

//        cv::Mat depthViz;
//        depth.convertTo( depthViz, CV_8U, 25 ); //CV_16UC1
//        cv::imshow( "test2", depthViz );
//        cv::waitKey(1);

        frame360->frameRGBD_[sensor_id].getRGBImage() = rgb.clone();
//        frame360->frameRGBD_[sensor_id].setRGBImage(rgb);
//        frameRGBD_[sensor_id].getDepthImage(depth);
//        frame360->frameRGBD_[sensor_id].setDepthImage(depthInMeters);
        frame360->frameRGBD_[sensor_id].getDepthImage() = depthInMeters.clone();
      }

cout << "Load frame\n";

      #pragma omp parallel num_threads(8)
      {
        int sensor_id = omp_get_thread_num();
        frame360->frameRGBD_[sensor_id].loadDepthEigen();
      }
cout << "Get eigen depth\n";


    //  frame360->undistort();
//      frame360->buildSphereCloud();
//      frame360->getPlanes();
//cout << "Get eigen depth\n";

      if(!bVisualize)
      {
        if(mode == 1 || mode == 2)
        {
//cout << "frame360 stitchSphericalImage\n";
          frame360->stitchSphericalImage();
//          frame360->fastStitchImage360();

          cv::imshow( "sphereRGB", frame360->sphereRGB );

          cv::Mat sphDepthVis;
          frame360->sphereDepth.convertTo( sphDepthVis, CV_8U, 25 ); //CV_16UC1
          cv::imshow( "sphereDepth", sphDepthVis );
          cv::waitKey(1);

          if(mode == 2)
          {
            frame360->sphereDepth.convertTo( frame360->sphereDepth, CV_8U, 25 ); //CV_16UC1
//            frame360->sphereDepth.convertTo( frame360->sphereDepth, CV_16U, 1000 ); //CV_16UC1
            cv::imwrite(path_results + mrpt::format("/rgb_%04d.png",frame), frame360->sphereRGB);
            cv::imwrite(path_results + mrpt::format("/depth_%04d.png",frame), frame360->sphereDepth);
          }
        }
        else if(mode == 3 || mode == 4)
        {
          frame360->undistort();
cout << "frame360->buildSphereCloud\n";
          frame360->buildSphereCloud();
//          frame360->getPlanes();

          if(mode == 3)
          {
            boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
            delete SphereViewer->frame360;
            SphereViewer->frame360 = frame360;
          }
          else
          {
            frame360->save(path_results, frame);
          }

        }
        else if(mode == 5)
        {
          frame360->buildSphereCloud_fast();
          boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
          delete SphereViewer->frame360;
          SphereViewer->frame360 = frame360;
        }
        mrpt::system::pause();

        if(mode != 3 && mode != 5)
          delete frame360;
        else
          SphereViewer->frameIdx = frame;

        frame+=1;

        mrpt::system::pause();
      }
//      {
////        frame360->buildSphereCloud_fast();
//        boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
//        delete SphereViewer->frame360;
//        SphereViewer->frame360 = frame360;
//      }
//
//      mrpt::system::pause();
//
//      {
//        boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
//        delete SphereViewer;
//        delete frame360;
//      }

      // Segment surfaces (planes and curve regions)
//      obs360.getPlanes();

      // Visualize the data
      if(bVisualize)
      {
        // It IS a new observation:
        mrpt::system::TTimeStamp last_obs_tim = observation->timestamp;

        // Update visualization ---------------------------------------

        win3D.get3DSceneAndLock();

        // Estimated grabbing rate:
        win3D.addTextMessage(-350,-13, format("Timestamp: %s", mrpt::system::dateTimeLocalToString(last_obs_tim).c_str()), TColorf(0.6,0.6,0.6),"mono",10,mrpt::opengl::FILL, 100);

//        // Show intensity image:
//        if (obsRGBD[0]->hasIntensityImage )
//        {
//          viewInt->setImageView(obsRGBD[0]->intensityImage); // This is not "_fast" since the intensity image may be needed later on.
//        }
        win3D.unlockAccess3DScene();

        // -------------------------------------------------------
        //           Create 3D points from RGB+D data
        //
        // There are several methods to do this.
        //  Switch the #if's to select among the options:
        // See also: http://www.mrpt.org/Generating_3D_point_clouds_from_RGB_D_observations
        // -------------------------------------------------------
        {
          win3D.get3DSceneAndLock();
//            obsRGBD[0]->project3DPointsFromDepthImageInto(*gl_points, false /* without obs.sensorPose */);
            for(unsigned i=0; i < num_sensors; i++)
              obs360.rgbd[i].project3DPointsFromDepthImageInto(*gl_points[i], true);
//              obsRGBD[i]->project3DPointsFromDepthImageInto(*gl_points[i], true, &sensorPoses[i]);

          win3D.unlockAccess3DScene();
        }

        win3D.repaint();

        mrpt::system::pause();
      }

    };

    // Free memory
    if(mode == 3)
    {
      boost::mutex::scoped_lock updateLock(SphereViewer->visualizationMutex);
      delete SphereViewer;
      delete frame360;
    }

    if(mode == 1 || mode == 2)
    {
      cv::destroyWindow("sphereRGB");
      cv::destroyWindow("sphereDepth");
    }

    cout << "\n ... END rgbd360-visualizer ...\n";

    return 0;

	} catch (std::exception &e)
	{
		std::cout << "MRPT exception caught: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		printf("Untyped exception!!");
		return -1;
	}
}
