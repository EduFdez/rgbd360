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

#ifndef CLOUDGRABBER_H
#define CLOUDGRABBER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h> //Save global map as PCD file
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/openni_camera/openni_driver.h>

#include <boost/thread/thread.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cvmat_serialization.h>

#include <FrameRGBD.h>
#include <RGBDGrabber.h>
#include <RGBDGrabberOpenNI_PCL.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Eigen/Core>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

typedef pcl::PointXYZRGBA PointT;

/*! This class is used to grab frames from an RGB-D sensor (e.g. Asus XPL or Kinect) using OpenNI-1.X
 */
class CloudGrabber
{
  public:

    /*!Conastructor.*/
    CloudGrabber(const std::string device_id = "",
                 const pcl::OpenNIGrabber::Mode mode = pcl::OpenNIGrabber::OpenNI_QVGA_30Hz) //pcl::OpenNIGrabber::OpenNI_Default_Mode;
    {
      interface = new pcl::OpenNIGrabber(device_id, mode, mode);

      pointCloudPtr_aux.reset(new pcl::PointCloud<PointT>());

      boost::function<void(const pcl::PointCloud<PointT>::ConstPtr&)> h = boost::bind (&CloudGrabber::cloud_cb_, this, _1);

      cloud_connection = interface->registerCallback(h);

     	interface->start ();

     	boost::this_thread::sleep(boost::posix_time::milliseconds(1500)); // Let some time to initialize the sensor properly
    cout << "CloudGrabber " << device_id << " initialized\n";
    }

    /*!Get the current frame.*/
    void grab(pcl::PointCloud<PointT>::Ptr& currentPointCloudPtr)
    {
      //lock while we set the point cloud;
      boost::mutex::scoped_lock lock (mtx_grabbing);

      //Grab the current point cloud
      currentPointCloudPtr.reset(new pcl::PointCloud<PointT>());
      pcl::copyPointCloud(*pointCloudPtr_aux,*currentPointCloudPtr);
    }

    /*!Stop grabing RGBD frames.*/
    inline void stop()
    {
      cloud_connection.disconnect();
      interface->stop();
    }

  private:

    boost::mutex mtx_grabbing;

    void cloud_cb_ (const pcl::PointCloud<PointT>::ConstPtr &cloud)
    {
      //lock while we set our cloud;
      boost::mutex::scoped_lock lock (mtx_grabbing);
      pointCloudPtr_aux = cloud;
    }

    pcl::OpenNIGrabber* interface;

    pcl::PointCloud<PointT>::ConstPtr pointCloudPtr_aux;

    boost::signals2::connection cloud_connection;

};

#endif
