/*
 *  Copyright (c) 2015,   INRIA Sophia Antipolis - LAGADIC Team
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

#define NUM_ASUS_SENSORS 4

#include <mrpt/poses/CPose3D.h>
#include <mrpt/utils.h>
#include <mrpt/system/os.h>
#include <iostream>

using namespace std;
using namespace mrpt::utils;

/*! Return the rotation vector of the input pose */
inline Eigen::Matrix4f getPoseEigenMatrix(const mrpt::poses::CPose3D & pose)
{
  Eigen::Matrix4f pose_mat;
  mrpt::math::CMatrixDouble44 pose_mat_mrpt;
  pose.getHomogeneousMatrix(pose_mat_mrpt);
  pose_mat << pose_mat_mrpt(0,0), pose_mat_mrpt(0,1), pose_mat_mrpt(0,2), pose_mat_mrpt(0,3),
              pose_mat_mrpt(1,0), pose_mat_mrpt(1,1), pose_mat_mrpt(1,2), pose_mat_mrpt(1,3),
              pose_mat_mrpt(2,0), pose_mat_mrpt(2,1), pose_mat_mrpt(2,2), pose_mat_mrpt(2,3),
              pose_mat_mrpt(3,0), pose_mat_mrpt(3,1), pose_mat_mrpt(3,2), pose_mat_mrpt(3,3) ;
  return  pose_mat;
}

int main (int argc, char ** argv)
{
    assert(argc > 1);

  string path_calibration;
  if(argc > 1)
    path_calibration = static_cast<string>(argv[1]);

    cout << "Change the reference system of the calibrattion of RGBD360 multisensor\n";
    float angle_offset = 22.5; //45
    Eigen::Matrix4f rot_offset = Eigen::Matrix4f::Identity(); rot_offset(1,1) = rot_offset(2,2) = cos(DEG2RAD(angle_offset)); rot_offset(1,2) = -sin(DEG2RAD(angle_offset)); rot_offset(2,1) = -rot_offset(1,2);
    // Load initial calibration
    cout << "Load initial calibration\n";
    mrpt::poses::CPose3D pose[NUM_ASUS_SENSORS];
    pose[0] = mrpt::poses::CPose3D(0.285, 0, 1.015, DEG2RAD(0), DEG2RAD(1.3), DEG2RAD(-90));
    pose[1] = mrpt::poses::CPose3D(0.271, -0.031, 1.015, DEG2RAD(-45), DEG2RAD(0), DEG2RAD(-90));
    pose[2] = mrpt::poses::CPose3D(0.271, 0.031, 1.125, DEG2RAD(45), DEG2RAD(2), DEG2RAD(-89));
    pose[3] = mrpt::poses::CPose3D(0.24, -0.045, 0.975, DEG2RAD(-90), DEG2RAD(1.5), DEG2RAD(-90));
//    int rgbd180_arrangement[4] = {1,8,2,7};

    Eigen::Matrix4f Rt_[4];
    Eigen::Matrix4f Rt_raul[4];
    Eigen::Matrix4f relative_edu[4];
    Eigen::Matrix4f relative_raul[4];
    Eigen::Matrix4f Rt_raul_new[4];

    Eigen::Matrix4f change_ref = Eigen::Matrix4f::Zero();
    change_ref(0,2) = 1.f;
    change_ref(1,0) = -1.f;
    change_ref(2,1) = -1.f;
    change_ref(3,3) = 1.f;

    for(size_t sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
    {
        Rt_[sensor_id].loadFromTextFile( mrpt::format("%s/Rt_0%i.txt", path_calibration.c_str(), sensor_id+1) );
        Rt_raul[sensor_id] = getPoseEigenMatrix( pose[sensor_id] ); //.inverse();
        //cout << " Rt_raul \n" << pose[sensor_id].getHomogeneousMatrixVal() << endl << Rt_raul[sensor_id] << endl;

        if(sensor_id > 0)
        {
            //relative_edu[sensor_id] = Rt_[0].inverse() * Rt_[sensor_id];
            relative_raul[sensor_id] = change_ref * Rt_[0].inverse() * Rt_[sensor_id] * change_ref.inverse();

            Rt_raul_new[sensor_id] = Rt_raul[0] * relative_raul[sensor_id];

            //cout << " rel edu \n" << relative_edu[sensor_id] << endl;
            cout << " rel edu to raul \n" << relative_raul[sensor_id] << endl;
            cout << " rel raul \n" << Rt_raul[0].inverse() * Rt_raul[sensor_id] << endl;
            cout << " raul \n" << Rt_raul[sensor_id] << endl;
            cout << " new \n" << Rt_raul_new[sensor_id] << endl;
            mrpt::system::pause();            
        }
        else
            Rt_raul_new[0] = Rt_raul[0];

        ofstream calibFile;
        string calibFileName = mrpt::format("%s/ref_raul/Rt_0%u.txt", path_calibration.c_str(), sensor_id+1);
        calibFile.open(calibFileName.c_str());
        if (calibFile.is_open())
        {
            calibFile << Rt_raul_new[sensor_id];
            calibFile.close();
        }
        else
            cout << "Unable to open file " << calibFileName << endl;
    }

  cout << "EXIT\n";

  return (0);
}
