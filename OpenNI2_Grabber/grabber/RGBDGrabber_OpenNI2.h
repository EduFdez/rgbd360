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

#ifndef RGBDGRABBER_OPENNI2
#define RGBDGRABBER_OPENNI2

#include <OpenNI.h>
#include "RGBDGrabber.h"

/*! This class captures RGBD frames from an OpenNI2 compatible sensor.
 *  It grabs the intensity and depth images and stores them in a 'FrameRGBD' object.
 */
class RGBDGrabber_OpenNI2 : public RGBDGrabber
{
 private:

    /*! OpenNI2 status signal */
    openni::Status rc;

    /*! Device information */
    openni::Device device;

    /*! Grabbing mode: resulution and framerate */
    openni::VideoMode	video_mode;

//    /*! Resolution mode of the device (default is QVGA = 320x240) */
//    enum Resolution
//    {
//      VGA = 1,
//      QVGA = 2,
//      QQVGA = 4,
//    } resolution;

    /*! Image properties */
    openni::VideoFrameRef depth_frame, color_frame;

    /*! Depth stream object */
    openni::VideoStream depth_stream;

    /*! RGB stream object */
    openni::VideoStream rgb_stream;

    /*! RGB stream object */
    int exposure_; //(miliseconds)

//    int gain_; // In percentage (%)

    /*! Device id */
    const char* deviceURI;

//    /*! Connected devices information */
//    openni::Array<openni::DeviceInfo> deviceList;

 public:
    /*!Constructor. Creates a RGBDGrabber_OpenNI2 instance that grabs RGBD frames from an OpenNI compatible sensor.*/
    RGBDGrabber_OpenNI2(const char* uri = openni::ANY_DEVICE, const int mode = 1, const int exposure = 10)
      : deviceURI(uri), exposure_(exposure)
    {
      printf("RGBDGrabber_OpenNI2...\n");

      if(mode == 0 || mode == 1)
        setResolution(mode);
      else
        setResolution(1);
    }

    ~RGBDGrabber_OpenNI2()
    {
      stop();
    }

//    /*!Shows a list of connected sensors*/
//    void showDeviceList()
//    {
////    	openni::Array<openni::DeviceInfo> deviceList;
//      openni::OpenNI::enumerateDevices(&deviceList);
//      printf("Get device list. %d devices connected\n", deviceList.getSize() );
//      for (unsigned i=0; i < deviceList.getSize(); i++)
//      {
//        printf("Device %u: name=%s uri=%s vendor=%s \n", i , deviceList[i].getName(), deviceList[i].getUri(), deviceList[i].getVendor());
//      }
//    }

//    /*!Set the sensor URI.*/
//    void setSensor(int sensor_id)
//    {
//      if(sensor_id >= 0 && sensor_id < deviceList.getSize())
//        deviceURI = deviceList[sensor_id].getUri();
//      else
//        printf("The sensor index %i is not found. Using the first available device\n", sensor_id);
//    }

    /*!Shows a list of the available modes for the chosen*/
    void showAvailableModes()
    {
//      const openni::SensorInfo *sensor_info = device.getSensorInfo(openni::SENSOR_COLOR);
      const openni::Array<openni::VideoMode>& modes_list = device.getSensorInfo(openni::SENSOR_COLOR)->getSupportedVideoModes();
      for (unsigned i=0; i < modes_list.getSize(); i++)
      {
        printf("Mode %u. Resolution=%dx%d FPS=%d pixelFormat=%d\n",i , modes_list[i].getResolutionX(), modes_list[i].getResolutionY(), modes_list[i].getFps(), modes_list[i].getPixelFormat() );
      }
    }

    /*!Set the resolution.*/
    void setResolution(int res)
    {
      // Set the size of the RGB and depth images.
      if(res == 0)
      {
        height = 480;
        width = 640;
      }
      else if(res == 1)
      {
        height = 240;
        width = 320;
      }
      else // To do: QQVGA modes
      {
        printf("\nError: OpenNI2 mode not valid! -> Previous value left\n\n");
      }
    }

    /*!Set the exposure value (in milliseconds).*/
    void setShutter(int exposure)
    {
//      exposure_ = exposure;
      rc = rgb_stream.getCameraSettings()->setExposure(exposure);
      if (rc != openni::STATUS_OK)
      {
        printf("RGBDGrabber_OpenNI2-> Couldn't change exposure: %s\n", openni::OpenNI::getExtendedError());
        return;
      }
//      printf("New exposure %d - %d \n", exposure, rgb_stream.getCameraSettings()->getExposure() );
    }

    /*!Return the current Exposure value (in milliseconds).*/
    int getShutter()
    {
//      return exposure_;
        return rgb_stream.getCameraSettings()->getExposure();
    }

    /*!Set the exposure value (in milliseconds).*/
    void setGain(int gain)
    {
      int gain_ = gain; //(100 = default)
      rc = rgb_stream.getCameraSettings()->setGain(gain);
      if (rc != openni::STATUS_OK)
      {
        printf("RGBDGrabber_OpenNI2-> Couldn't change gain: %s\n", openni::OpenNI::getExtendedError());
        return;
      }
    }

    /*!Return the current Gain value (in percentage 100%).*/
    int getGain()
    {
//      return gain_;
        return rgb_stream.getCameraSettings()->getGain();
    }

    /*!Initializes the grabber object*/
    inline void init()
    {
      // start receiving data
      rc = openni::STATUS_OK;
      rc = openni::OpenNI::initialize();
      printf("After initialization:\n %s\n", openni::OpenNI::getExtendedError());//      deviceURI = openni::ANY_DEVICE;
      rc = device.open(deviceURI);

      if (rc != openni::STATUS_OK)
      {
        printf("RGBDGrabber_OpenNI2: Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
        openni::OpenNI::shutdown();
        return;// 1;
      }

      //								Create RGB and Depth channels
      //========================================================================================
      rc = depth_stream.create(device, openni::SENSOR_DEPTH);
      if (rc == openni::STATUS_OK)
      {
        rc = depth_stream.start();
        if (rc != openni::STATUS_OK)
        {
          printf("RGBDGrabber_OpenNI2: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
          depth_stream.destroy();
        }
      }
      else
      {
        printf("RGBDGrabber_OpenNI2: Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
      }

      rc = rgb_stream.create(device, openni::SENSOR_COLOR);
      if (rc == openni::STATUS_OK)
      {
        rc = rgb_stream.start();
        if (rc != openni::STATUS_OK)
        {
          printf("RGBDGrabber_OpenNI2: Couldn't start rgb stream:\n%s\n", openni::OpenNI::getExtendedError());
          rgb_stream.destroy();
        }
      }
      else
      {
        printf("RGBDGrabber_OpenNI2: Couldn't find rgb stream:\n%s\n", openni::OpenNI::getExtendedError());
      }

      // This sleep is needed before changing the properties of the captured stream
      sleep(1);
//      usleep(1000000);
//      boost::this_thread::sleep (boost::posix_time::milliseconds (300));
//      std::this_thread::sleep_for(std::chrono::milliseconds(200));

      // Turn off auto shutter:
      bool auto_exposure = false;
//      rc = rgb_stream.setProperty(openni::STREAM_PROPERTY_AUTO_EXPOSURE, 0);
      rc = rgb_stream.getCameraSettings()->setAutoExposureEnabled(auto_exposure);
      if (rc != openni::STATUS_OK)
      {
        printf("RGBDGrabber_OpenNI2-> Cannot turn-off Auto exposure: %s\n", openni::OpenNI::getExtendedError());
        return;
      }
      // Turn off auto WB
      rc = rgb_stream.setProperty(openni::STREAM_PROPERTY_AUTO_WHITE_BALANCE, 0);
//      rc = rgb_stream.getCameraSettings()->setAutoWhiteBalanceEnabled(auto_exposure);
      if (rc != openni::STATUS_OK)
      {
        printf("RGBDGrabber_OpenNI2-> Cannot turn-off Auto White-Balance: %s\n", openni::OpenNI::getExtendedError());
        return;
      }

      if (!depth_stream.isValid() || !rgb_stream.isValid())
      {
        printf("RGBDGrabber_OpenNI2: No valid streams. Exiting\n");
        openni::OpenNI::shutdown();
        return;// 2;
      }

      if (rc != openni::STATUS_OK)
      {
        openni::OpenNI::shutdown();
        return;// 3;
      }

      //						Configure some properties (resolution)
      //========================================================================================

//        unsigned width = 320, height = 240;
      rc = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
      //rc = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_OFF);

      video_mode = rgb_stream.getVideoMode();
//      printf("\nInitial resolution RGB (%d, %d)", video_mode.getResolutionX(), video_mode.getResolutionY());
      video_mode.setResolution(width,height);
      rc = rgb_stream.setVideoMode(video_mode);
      rc = rgb_stream.setMirroringEnabled(false);

      video_mode = depth_stream.getVideoMode();
//      printf("\nInitial resolution Depth(%d, %d)", video_mode.getResolutionX(), video_mode.getResolutionY());
      video_mode.setResolution(width,height);
      rc = depth_stream.setVideoMode(video_mode);
      rc = depth_stream.setMirroringEnabled(false);

      // Set to manual exposure
//      sleep(1);
      setShutter(exposure_);

      // Grabber info
      printf("Sensor %s resolution %d x %i exposure %i gain %i \n", deviceURI, video_mode.getResolutionX(), video_mode.getResolutionY(), rgb_stream.getCameraSettings()->getExposure(), rgb_stream.getCameraSettings()->getGain() );
    }

    /*!Copy the current RGBD frame into 'framePtr'.*/
    void grab(FrameRGBD* framePtr)
    {
//    std::cout << "RGBDGrabber_OpenNI2::grab...\n";
      depth_stream.readFrame(&depth_frame);
      rgb_stream.readFrame(&color_frame);
      assert((depth_frame.getWidth() == color_frame.getWidth()) || (depth_frame.getHeight() == color_frame.getHeight()));

//    // Check exposure
//    std::cout << "exposure set to " << rgb_stream.getCameraSettings()->getExposure() << std::endl;

      // Fill the color image
      const openni::RGB888Pixel* imageBuffer = (const openni::RGB888Pixel*)color_frame.getData();
      framePtr->getRGBImage().create(color_frame.getHeight(), color_frame.getWidth(), CV_8UC3);
      memcpy( framePtr->getRGBImage().data, imageBuffer, 3*color_frame.getHeight()*color_frame.getWidth()*sizeof(uint8_t) );
      cv::cvtColor(framePtr->getRGBImage(),framePtr->getRGBImage(),CV_BGR2RGB); //this will put colors right
//    std::cout << "RGBDGrabber_OpenNI2::grab - fillColor\n";

      // Fill the depth image
      const openni::DepthPixel* depthImgRaw = (openni::DepthPixel*)depth_frame.getData();
      framePtr->getDepthImage().create(depth_frame.getHeight(), depth_frame.getWidth(), CV_16U);
      memcpy( framePtr->getDepthImage().data, depthImgRaw, depth_frame.getHeight()*depth_frame.getWidth()*sizeof(uint16_t) );
//    std::cout << "RGBDGrabber_OpenNI2::grab - fillDepth\n";
    }

    /*!Stop grabing RGBD frames.*/
    inline void stop()
    {
      printf("RGBDGrabber_OpenNI2: Stop device %s\n", deviceURI);

      depth_stream.destroy();
      rgb_stream.destroy();
      openni::OpenNI::shutdown();
    }

};
#endif
