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

#ifndef RELOCALIZER360_H
#define RELOCALIZER360_H

#include "Map360.h"
#include "RegisterRGBD360.h"

/*! This class is used to relocalize a given Frame360 in the map of spherical keyframes. This is useful when the odometry
 *  track has been lost. The core of this functionality performs PbMap matching, what makes it highly efficient for
 *  real-time applications like SLAM.
 */
class Relocalizer360
{
 private:
  /*! Reference to the Map */
  Map360 &Map;

 public:

  /*! Keyframe registration object */
  RegisterRGBD360 registerer;

  /*! Index of the keyframe with respect to which the system was relocalized */
  unsigned relocKF;

  /*! Pose of the relocalized frame wrt 'relocKF'*/
  Eigen::Matrix4d rigidTransf;

  /*! 6x6 covariance matrix of the relocalization */
  Eigen::Matrix<double,6,6> informationM;

  /*! Constructor */
  Relocalizer360(Map360 &map) :
    Map(map),
    registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH))
  {

  }

//  /*! Destructor */
//  ~Relocalizer360()
//  {
//
//  }

  /*! Relocalize the input Frame360 'currentFrame'. It returns the frame wrt the system has relocalized or -1 otherwise */
  int relocalize(Frame360 *currentFrame)
  {
    for(int i=Map.vpSpheres.size()-1; i >= 0; i--)
    {
      bool bGoodRegistration = registerer.Register(Map.vpSpheres[i], currentFrame, max_match_planes, RegisterRGBD360::PLANAR_3DoF);
      if(bGoodRegistration && registerer.getMatchedPlanes().size() >= 5 && registerer.getAreaMatched() > 10 )
      {
        relocKF = i;
        rigidTransf = registerer.getPose().cast<double>();
        informationM = registerer.getInfoMat().cast<double>();
        return relocKF;
      }
    }

    return -1;
  }
};

#endif
