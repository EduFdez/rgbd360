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

#include <Calibrator.h>

#include <pcl/console/parse.h>

using namespace std;


void print_help(char ** argv)
{
  cout << "\nThis program evaluates the calibration of a pair of cameras for a given set of plane correspondences.\n\n";

  cout << "  usage: " << argv[0] << " <extrinsicCalibFile1> <extrinsicCalibFile2> <extrinsicCalibFile3> <matchedPlanesFile>\n";
  cout << "    <extrinsicCalibFile1> is the file containing the extrinsic matrix" << endl;
  cout << "    <extrinsicCalibFile2> is the file containing the extrinsic matrix" << endl;
  cout << "    <extrinsicCalibFile3> is the file containing the extrinsic matrix" << endl;
  cout << "    <matchedPlanesFile> is the directory containing the plane correspondences" << endl << endl;
  cout << "         " << argv[0] << " -h | --help : shows this help" << endl;
}

int main (int argc, char ** argv)
{
  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help"))
    print_help(argv);

  string extrinsicCalibFile1 = static_cast<string>(argv[1]);
  string extrinsicCalibFile2 = static_cast<string>(argv[2]);
  string extrinsicCalibFile3 = static_cast<string>(argv[3]);

  string matchedPlanesFile = static_cast<string>(argv[4]);

  cout << "Calibrate Pair\n";
  PairCalibrator calibrator1, calibrator2, calibrator3;

  calibrator1.setInitRt(extrinsicCalibFile1);
  calibrator2.setInitRt(extrinsicCalibFile2);
  calibrator3.setInitRt(extrinsicCalibFile3);

  cout << "calibrator1 \n" << calibrator1.Rt_estimated << endl;
  cout << "calibrator2 \n" << calibrator1.Rt_estimated << endl;
  cout << "calibrator3 \n" << calibrator1.Rt_estimated << endl;

  // Get the plane correspondences
  calibrator1.loadPlaneCorrespondences(matchedPlanesFile);
  calibrator2.loadPlaneCorrespondences(matchedPlanesFile);
  calibrator3.loadPlaneCorrespondences(matchedPlanesFile);

  cout << "Rotation errors " << calibrator1.calcCorrespRotError() << " " << calibrator2.calcCorrespRotError() << " " << calibrator3.calcCorrespRotError() << " \n";

  cout << "EXIT\n";

  return (0);
}
