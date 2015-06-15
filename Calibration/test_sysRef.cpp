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

#include <definitions.h>

#include <mrpt/poses/CPose3D.h>
#include <mrpt/utils.h>
#include <mrpt/system/os.h>
#include <iostream>

#include <Miscellaneous.h>

using namespace std;
using namespace mrpt::utils;

int main (int argc, char ** argv)
{

    cout << "Change the reference system of the calibrattion of RGBD360 multisensor\n";
    float angle_offset = 22.5; //45
    Eigen::Matrix4f rot_offset = Eigen::Matrix4f::Identity(); rot_offset(1,1) = rot_offset(2,2) = cos(angle_offset*PI/180); rot_offset(1,2) = -sin(angle_offset*PI/180); rot_offset(2,1) = -rot_offset(1,2);
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
    Eigen::Matrix4f rel_edu[4];
    Eigen::Matrix4f rel_raul[4];
    Eigen::Matrix4f Rt_raul_new[4];

    Eigen::Matrix4f change_ref = Eigen::Matrix4f::Zero();
//    change_ref(0,1) = 1.f;
//    change_ref(1,2) = 1.f;
//    change_ref(2,0) = 1.f;
//    change_ref(3,3) = 1.f;

    change_ref(1,1) = -1.f;
    change_ref(0,2) = 1.f;
    change_ref(2,0) = 1.f;
    change_ref(3,3) = 1.f;

    for(size_t sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
    {
        Rt_[sensor_id].loadFromTextFile( mrpt::format("/home/efernand/Libraries/rgbd360/Calibration/test/Rt_0%i.txt",sensor_id+1) );
        Rt_raul[sensor_id] = getPoseEigenMatrix( pose[sensor_id] ); //.inverse();

        if(sensor_id > 0)
        {
            rel_edu[sensor_id] = Rt_[0].inverse() * Rt_[sensor_id];
//            rel_edu[sensor_id] = Rt_[sensor_id] * Rt_[0].inverse();
//            rel_edu[sensor_id] = Rt_[sensor_id].inverse() * Rt_[0];
            rel_raul[sensor_id] = change_ref.transpose() * rel_edu[sensor_id] * change_ref;
//            rel_raul[sensor_id] = change_ref * rel_edu[sensor_id] * change_ref.transpose();

//            rel_raul[sensor_id] = rel_raul[sensor_id].inverse();

//            Rt_raul_new[sensor_id] = Rt_raul[0] * rel_raul[sensor_id];
            Rt_raul_new[sensor_id] = rel_raul[sensor_id] * Rt_raul[0];

            //cout << " rel edu \n" << rel_edu[sensor_id] << endl;
            cout << " rel edu change \n" << rel_raul[sensor_id] << endl;
//            cout << " rel edu change inv \n" << rel_raul[sensor_id].inverse() << endl;
            cout << " rel raul \n" << Rt_raul[sensor_id] * Rt_raul[0].inverse() << endl;
            cout << " raul \n" << Rt_raul[sensor_id] << endl;
            cout << " new \n" << Rt_raul_new[sensor_id] << endl;
            mrpt::system::pause();

            ofstream calibFile;
              string calibFileName = mrpt::format("%s/Calibration/test/ref_Rt_0%u.txt", PROJECT_SOURCE_PATH, sensor_id);
              calibFile.open(calibFileName.c_str());
              if (calibFile.is_open())
              {
                calibFile << Rt_raul_new[sensor_id];
                calibFile.close();
              }
              else
                cout << "Unable to open file " << calibFileName << endl;
        }
    }
    Rt_raul_new[0] = Rt_raul[0];

//  // Ask the user if he wants to save the calibration matrices
//  string input;
//  cout << "Do you want to save the calibrated extrinsic matrices? (y/n)" << endl;
//  getline(cin, input);
//  if(input == "y" || input == "Y")
  {
    ofstream calibFile;
    for(unsigned sensor_id=0; sensor_id < NUM_ASUS_SENSORS; sensor_id++)
    {
      string calibFileName = mrpt::format("%s/Calibration/test/ref_Rt_0%u.txt", PROJECT_SOURCE_PATH, sensor_id+1);
      calibFile.open(calibFileName.c_str());
      if (calibFile.is_open())
      {
        calibFile << Rt_raul_new[sensor_id];
        calibFile.close();
      }
      else
        cout << "Unable to open file " << calibFileName << endl;
    }
  }

  cout << "EXIT\n";

  return (0);
}
