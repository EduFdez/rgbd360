/*
 *  Copyright (c) 2013, Universidad de Málaga - Grupo MAPIR and
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

#pragma once

#include "definitions.h"

#define USE_BILATERAL_FILTER 1
#define DOWNSAMPLE_160 1

#include <mrpt/pbmap.h>
#include <mrpt/pbmap/Miscellaneous.h>

#include "Sphere3D.h"

#include <CloudRGBD_Ext.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cvmat_serialization.h>

#include <opencv/cv.h>
//#include <opencv2/opencv.hpp> // which one should I use?



/*! This class defines the omnidirectional RGB-D frame 'Spieye360'. It contains a serie of attributes and methods to
 *  produce the omnidirectional images, and to obtain the spherical point cloud and a the planar representation for it
 */
class SphereFeat : public virtual Sphere3D
{
  protected:

    //    /*! Points */
    //    PointFeatures;

    //    /*! Lines */
    //    LSD_lines;

    /*! PbMap of the spherical frame */
    mrpt::pbmap::PbMap planes;        

  public:

    /*! Constructor for the SphericalStereo sensor (outdoor sensor) */
    Sphere3DFeat()
    {};

    /*! Return the total area of the planar patches from this frame */
    float getPlanarArea();

    /*! Load a spherical PbMap */
    void loadPbMap(std::string &pbmapPath);

    /*! Load a spherical frame from its point cloud and its PbMap files */
    void load_PbMap_Cloud(std::string &pointCloudPath, std::string &pbmapPath);

    /*! Load a spherical frame from its point cloud and its PbMap files */
    void load_PbMap_Cloud(std::string &path, unsigned &index);

    /*! Save the PbMap from an omnidirectional RGB-D image */
    void savePlanes(std::string pathPbMap);

    /*! Save the pointCloud and PbMap from an omnidirectional RGB-D image */
    void save(std::string &path, unsigned &frame);

    /*! Create the PbMap of the spherical point cloud */
    virtual void segmentPlanes();

//    /*! Merge the planar patches that correspond to the same surface in the sphere */
//    virtual void mergePlanes();

//    /*! Group the planes segmented from each single sensor into the common PbMap 'planes' */
//    void groupPlanes();

    /*! Perform bilateral filtering on the point cloud   */
    void filterCloudBilateral_stereo();

    /*! This function segments planes from the point cloud    */
    void segmentPlanesStereo();

    /*! This function segments planes from the point cloud corresponding to the sensor 'sensor_id',
      in the frame of reference of the omnidirectional camera   */
    void segmentPlanesStereoRANSAC();

};
