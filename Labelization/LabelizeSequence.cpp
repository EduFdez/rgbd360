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

#include <Frame360.h>
#include <FilterPointCloud.h>
#include <RegisterRGBD360.h>
#include <Map360_Visualizer.h>

#define SUPERVISE_REGISTRATION 0

using namespace std;

/*! This class' main function "run" loads a stream of Frame360 files partially labelized,
 *  and expands the labels by doing consecutive frame registration. */
class LabelizeSequence
{
  private:
    /*! Sequence of matched planes. It is used for backwards landmark propagation */
    std::vector<std::map<unsigned, unsigned> > matched_planes;

  public:

    void run(string &path)
    {
      unsigned int frame = 1;

      Calib360 calib;
      calib.loadIntrinsicCalibration();
      calib.loadExtrinsicCalibration();

      Frame360 *frame360_1, *frame360_2;
      frame360_2 = new Frame360(&calib);
      frame360_2->load_PbMap_Cloud(path, frame); // Load pointCloud and PbMap

      // Create registration object
      RegisterRGBD360 registerer(mrpt::format("%s/config_files/configLocaliser_sphericalOdometry.ini", PROJECT_SOURCE_PATH));

      // Filter the point clouds before visualization
      FilterPointCloud filter;

      bool bGoodRegistration = true;
      map<unsigned, unsigned> emptyMatch;

      const int MAX_MATCH_PLANES = 30; // Avoid matching subgraphs that are too big.

      string fileName = path + mrpt::format("/spherePlanes_%d.pbmap",++frame);
      while( fexists(fileName.c_str()) )
      {
      cout << "Frame " << fileName << endl;

//        if(bGoodRegistration)
        {
          frame360_1 = frame360_2;
//        cout << "Number of planes " << frame360_1->planes.vPlanes.size() << endl;
        }

        // Load pointCloud and PbMap
        frame360_2 = new Frame360(&calib);
        #if SUPERVISE_REGISTRATION
          frame360_2->load_PbMap_Cloud(path, frame); // Load pointCloud and PbMap
        #else
          frame360_2->loadPbMap(fileName); // Load pointCloud and PbMap
        #endif

        fileName = path + mrpt::format("/spherePlanes_%d.pbmap", ++frame);

        bGoodRegistration = registerer.Register(frame360_1, frame360_2, MAX_MATCH_PLANES, RegisterRGBD360::PLANAR_ODOMETRY_3DoF);
//        cout << "bGoodRegistration" << bGoodRegistration << " " << registerer.getMatchedPlanes().size() << " \n";
        if( !bGoodRegistration || registerer.getMatchedPlanes().size() < min_planes_registration)
        {
          matched_planes.push_back(emptyMatch);
          continue;
        }

        // Visualize. Superimpose the last two frames
        #if SUPERVISE_REGISTRATION
          Map360 Map;
          Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
          Map.addKeyframe(frame360_1, pose );
          pose = registerer.getPose();
          Map.addKeyframe(frame360_2, pose );
          Map360_Visualizer Viewer(Map);
//        cout << "cloud size " << frame360_1->sphereCloud->size() << " " << frame360_2->sphereCloud->size() << " " << registrationClouds[0]->size() << " " << registrationClouds[1]->size() << endl;
          string input;
          cout << "Do you accept this registration? (y/n)" << endl;
//          int key_pressed = mrpt::system::os::getch();
          getline(cin, input);
          if(input == "n" || input == "N")
          {
            matched_planes.push_back(emptyMatch);
            continue;
          }
        #endif

        // Expand Labelization in the registered frame
        map<unsigned, unsigned> bestMatch = registerer.getMatchedPlanes();
        matched_planes.push_back(bestMatch);
        for(map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
        {
          // Plane label
          if(frame360_1->planes.vPlanes[it->first].label == frame360_2->planes.vPlanes[it->second].label)
            continue;
          else {
            if(frame360_1->planes.vPlanes[it->first].label != "")
              frame360_2->planes.vPlanes[it->second].label = frame360_1->planes.vPlanes[it->first].label;
            else if(frame360_2->planes.vPlanes[it->second].label != "")
              frame360_1->planes.vPlanes[it->first].label = frame360_2->planes.vPlanes[it->second].label;
          }

          // Object label
          if(frame360_1->planes.vPlanes[it->first].label_object == frame360_2->planes.vPlanes[it->second].label_object)
            continue;
          else {
            if(frame360_1->planes.vPlanes[it->first].label_object != "")
              frame360_2->planes.vPlanes[it->second].label_object = frame360_1->planes.vPlanes[it->first].label_object;
            else if(frame360_2->planes.vPlanes[it->second].label_object != "")
              frame360_1->planes.vPlanes[it->first].label_object = frame360_2->planes.vPlanes[it->second].label_object;
          }

          // Context label
          if(frame360_1->planes.vPlanes[it->first].label_context == frame360_2->planes.vPlanes[it->second].label_context)
            continue;
          else {
            if(frame360_1->planes.vPlanes[it->first].label_context != "")
              frame360_2->planes.vPlanes[it->second].label_context = frame360_1->planes.vPlanes[it->first].label_context;
            else if(frame360_2->planes.vPlanes[it->second].label_context != "")
              frame360_1->planes.vPlanes[it->first].label_context = frame360_2->planes.vPlanes[it->second].label_context;
          }
        }

        // Save Labelized frames
        frame360_1->savePlanes( mrpt::format("%s/spherePlanes_%d.pbmap", path.c_str(), frame-2) );
        frame360_2->savePlanes( mrpt::format("%s/spherePlanes_%d.pbmap", path.c_str(), frame-1) );
//        cout << "Save labelized frame " << frame-1 << endl;

        delete frame360_1;
      }
//      boost::this_thread::sleep (boost::posix_time::milliseconds (2000));

      // Reverse registration order -> Expand labels backwards
      cout << "\n\nReverse order in labelized sequence" << endl;
      frame -= 1;
//      std::vector<std::map<unsigned, unsigned> >::iterator it_match = matched_planes.end();
      while( !matched_planes.empty() )
      {
        fileName = path + mrpt::format("/spherePlanes_%d.pbmap",--frame);
      cout << "Frame " << fileName << endl;

//        if(bGoodRegistration)
        {
          frame360_1 = frame360_2;
        }

        // Load pointCloud and PbMap
        frame360_2 = new Frame360(&calib);
        frame360_2->loadPbMap(fileName); // Load pointCloud and PbMap

        std::map<unsigned, unsigned> matchedPlanes = matched_planes.back();
        matched_planes.pop_back();
        if( matchedPlanes.empty() )
          continue;

        // Expand Labelization in the registered frame
        for(map<unsigned, unsigned>::iterator it=matchedPlanes.begin(); it != matchedPlanes.end(); it++)
        {
          // Plane label
          if(frame360_1->planes.vPlanes[it->second].label == frame360_2->planes.vPlanes[it->first].label)
            continue;
          else {
            if(frame360_1->planes.vPlanes[it->second].label != "")
              frame360_2->planes.vPlanes[it->first].label = frame360_1->planes.vPlanes[it->second].label;
            else if(frame360_2->planes.vPlanes[it->first].label != "")
              frame360_1->planes.vPlanes[it->second].label = frame360_2->planes.vPlanes[it->first].label;
          }

          // Object label
          if(frame360_1->planes.vPlanes[it->second].label_object == frame360_2->planes.vPlanes[it->first].label_object)
            continue;
          else {
            if(frame360_1->planes.vPlanes[it->second].label_object != "")
              frame360_2->planes.vPlanes[it->first].label_object = frame360_1->planes.vPlanes[it->second].label_object;
            else if(frame360_2->planes.vPlanes[it->first].label_object != "")
              frame360_1->planes.vPlanes[it->second].label_object = frame360_2->planes.vPlanes[it->first].label_object;
          }

          // Context label
          if(frame360_1->planes.vPlanes[it->second].label_context == frame360_2->planes.vPlanes[it->first].label_context)
            continue;
          else {
            if(frame360_1->planes.vPlanes[it->second].label_context != "")
              frame360_2->planes.vPlanes[it->first].label_context = frame360_1->planes.vPlanes[it->second].label_context;
            else if(frame360_2->planes.vPlanes[it->first].label_context != "")
              frame360_1->planes.vPlanes[it->second].label_context = frame360_2->planes.vPlanes[it->first].label_context;
          }
        }

        // Save Labelized frames
        frame360_1->savePlanes(mrpt::format("%s/spherePlanes_%d.pbmap", path.c_str(), frame+1));
        frame360_2->savePlanes(mrpt::format("%s/spherePlanes_%d.pbmap", path.c_str(), frame));
//        cout << "Save labelized frames " << frame << " and " << frame+1 << endl;

        delete frame360_1;
      }

      delete frame360_2;
      cout << "Finish labelization " << endl;
    }

};


void print_help(char ** argv)
{
  cout << "\nThis program loads a stream of previously built spheres (PbMap+PointCloud) partially labelized,"
       << " and expands the labels by doing consecutive frame registration.\n";
  cout << "  usage: " << argv[0] << " <pathToFolderWithSpheres> \n";
}

int main (int argc, char ** argv)
{
  if(argc != 2)
  {
    print_help(argv);
    return 0;
  }

  string path = static_cast<string>(argv[1]);

  cout << "Create LabelizeSequence object\n";
  LabelizeSequence rgbd360_expandLabels;
  rgbd360_expandLabels.run(path);

  cout << " EXIT\n";

  return 0;
}

