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
 * Author: Eduardo Fernandez-Moral
 */

#include <Frame360_Visualizer.h>

using namespace std;

void print_help(char ** argv)
{
  cout << "\nThis program loads the pointCloud and the PbMap from a RGBD360 observation. \n";
  cout << "  usage: " << argv[0] << " <pathToImagesDir> <frameID>" << endl;
  cout << "  usage: " << argv[0] << " <pathToPointCloud> <pathToPbMap>" << endl;
}

int main (int argc, char ** argv)
{
  if(argc != 3)
  {
    print_help(argv);
    return 0;
  }

  Calib360 *calib; // This pointer points to nothing because its content isn't really needed here, since the spheres (PointCloud+PbMap) have already been built.
  Frame360 frame360(calib);

  // Load pointCloud and PbMap. Both files have to be in the same directory, and be named
  // sphereCloud_X.pcd and spherePlanes_X.pbmap, with X the being the frame idx
  string firstPath = static_cast<string>(argv[1]);
  string pointCloudLabel = ".pcd";
  if( pointCloudLabel.compare( firstPath.substr(firstPath.length()-4) ) == 0 ) // If the first string correspond to a pointCloud path
  {
    string pathToPointCloud = firstPath;
    string pathToPbMap = static_cast<string>(argv[2]);
    frame360.load_PbMap_Cloud(pathToPointCloud, pathToPbMap);
  }
  else
  {
    string pathToImagesDir = firstPath;
    unsigned frameID = atoi(argv[2]);
    frame360.load_PbMap_Cloud(pathToImagesDir, frameID);
  }

  Frame360_Visualizer sphereViewer(&frame360);
  cout << "\n  Press 'q' to close the program\n";

  while (!sphereViewer.viewer.wasStopped() )
    boost::this_thread::sleep (boost::posix_time::milliseconds (10));

  cout << "EXIT\n\n";

  return (0);
}

