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

#ifndef PARAMS_ODOMETRY_H
#define PARAMS_ODOMETRY_H

/*! This file contains a set of heuristic thresholds used for PbMap registration and for visual/range odometry within the 'RGBD360' project.
 */

/*! Maximum number of planes to match when registering a pair of Spheres */
static float max_match_planes = 25;

/*! Minimum number of matched planes to consider a good registration */
static float min_planes_registration = 4;

/*! Minimum distance between keyframes */
static float min_dist_keyframes = 0.2;

/*! Maximum distance between two consecutive frames of a RGBD360 video sequence */
static float max_translation_odometry = 1.8;

/*! Maximum rotation between two consecutive frames of a RGBD360 video sequence */
static float max_rotation_odometry = 1.2;

/*! Maximum conditioning to resolve the calibration equation system. This parameter
    represent the ratio between the maximum and the minimum eigenvalue of the system */
static float threshold_conditioning = 8000.0;


#endif //PARAMS_ODOMETRY_H
