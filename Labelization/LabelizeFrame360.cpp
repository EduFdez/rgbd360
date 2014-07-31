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
 *   Author: efernand Fernandez-Moral
 */

#include <Frame360.h>
//#include <Frame360_Visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

size_t plane_idx; // This variable is used to communicate the viewer with the main program, so that only the plane being labelized is highlighted

/*! This class' is used to visualize the manual labelization. The plane being labelized is highlighted to facilitate this task */
struct PbMapVisualizer
{
  Frame360 *frame360;

  pcl::visualization::CloudViewer cloudViewer;

  bool bShowPlanes;

  PbMapVisualizer(Frame360 *frame) :
      frame360(frame),
      cloudViewer("PbMap-Labelization"),
      bShowPlanes(true)
  {
  }

  void viz_cb (pcl::visualization::PCLVisualizer& viz)
  {
    if (frame360->sphereCloud->empty())
    {
      mrpt::system::sleep(10);
      return;
    }

    {
      // Render the data
      viz.removeAllPointClouds();
      viz.removeAllShapes();

      if (!viz.updatePointCloud (frame360->sphereCloud, "sphereCloud"))
        viz.addPointCloud (frame360->sphereCloud, "sphereCloud");

      // Draw planes
      char name[1024];

  //  if(bShowPlanes)
  //    for(size_t plane_idx=0; plane_idx < frame360->planes.vPlanes.size(); plane_idx++)
      if(plane_idx < frame360->planes.vPlanes.size())
      {
        mrpt::pbmap::Plane &plane_i = frame360->planes.vPlanes[plane_idx];
        sprintf (name, "normal_%u", static_cast<unsigned>(plane_idx));
        pcl::PointXYZ pt1, pt2; // Begin and end points of normal's arrow for visualization
        pt1 = pcl::PointXYZ(plane_i.v3center[0], plane_i.v3center[1], plane_i.v3center[2]);
        pt2 = pcl::PointXYZ(plane_i.v3center[0] + (0.5f * plane_i.v3normal[0]),
                            plane_i.v3center[1] + (0.5f * plane_i.v3normal[1]),
                            plane_i.v3center[2] + (0.5f * plane_i.v3normal[2]));
        viz.addArrow (pt2, pt1, ared[plane_idx%10], agrn[plane_idx%10], ablu[plane_idx%10], false, name);

        {
          sprintf (name, "n%u %s %s % s", static_cast<unsigned>(plane_idx), plane_i.label.c_str(), plane_i.label_object.c_str(), plane_i.label_context.c_str());
  //        sprintf (name, "n%u_%u", static_cast<unsigned>(plane_idx), static_cast<unsigned>(plane_i.semanticGroup));
          viz.addText3D (name, pt2, 0.1, ared[plane_idx%10], agrn[plane_idx%10], ablu[plane_idx%10], name);
        }

        sprintf (name, "plane_%02u", static_cast<unsigned>(plane_idx));
        pcl::visualization::PointCloudColorHandlerCustom <PointT> color (plane_i.planePointCloudPtr, red[plane_idx%10], grn[plane_idx%10], blu[plane_idx%10]);
        viz.addPointCloud (plane_i.planePointCloudPtr, color, name);
        viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);

        sprintf (name, "approx_plane_%02d", int (plane_idx));
        viz.addPolygon<PointT> (plane_i.polygonContourPtr, 0.5 * red[plane_idx%10], 0.5 * grn[plane_idx%10], 0.5 * blu[plane_idx%10], name);
      }
    }
  }

  void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
  {
    if ( event.keyDown () )
    {
      if(event.getKeySym () == "l" || event.getKeySym () == "L")
        bShowPlanes = !bShowPlanes;
    }
  }

  void run()
  {
    cloudViewer.runOnVisualizationThread (boost::bind(&PbMapVisualizer::viz_cb, this, _1), "viz_cb");
    cloudViewer.registerKeyboardCallback ( &PbMapVisualizer::keyboardEventOccurred, *this );
  }

};


void print_help(char ** argv)
{
  cout << "\nThis program loads a PbMap (a its corresponding point cloud) and asks the user to labelize the planes"
       << " before saving the annotated PbMap to disk\n";
//  cout << "  usage: " << argv[0] << " <pathToPointCloud> <pathToPbMap>" << endl;
//  cout << "  usage: " << argv[0] << " <pathToImagesDir> <frameID>" << endl;
  cout << "  usage: " << argv[0] << " <pathToPbMap>" << endl;
}

int main (int argc, char ** argv)
{
//  if(argc != 3)
  if(argc != 2)
  {
    print_help(argv);
    return 0;
  }

  string pathToPbMap = static_cast<string>(argv[1]);
  if( !fexists(pathToPbMap.c_str()) )
  {
    cout << "The path to the PbMap does not exist. Provide a valid PbMap as input, please. \n\t EXIT \n";
    return 0;
  }

  size_t lastSlashPos = pathToPbMap.find_last_of('/')+1;
  size_t indexBegin = pathToPbMap.find_last_of('_')+1;
  int frameID = atoi( pathToPbMap.substr(indexBegin, pathToPbMap.find_last_of('.')-indexBegin).c_str() );
  string pathToPointCloud = pathToPbMap.substr(0,lastSlashPos).append(mrpt::format("sphereCloud_%i.pcd", frameID));
  cout << "pathCloud: " << pathToPointCloud << endl;

//  string pathToImagesDir = static_cast<string>(argv[1]);
//  unsigned frameID = atoi(argv[2]);

  Calib360 *calib;
  Frame360 *frame360 = new Frame360(calib);

//  // Load pointCloud and PbMap. Both files have to be in the same directory, and be named
//  // sphereCloud_X.pcd and spherePlanes_X.pbmap, with X the being the frame idx
//  frame360->load_PbMap_Cloud(pathToImagesDir, frameID);
  frame360->load_PbMap_Cloud(pathToPointCloud, pathToPbMap);
  cout << "sphereCloud size  " << frame360->sphereCloud->size() << endl;

//  Frame360_Visualizer sphereViewer(frame360);
  plane_idx = 9999;
  PbMapVisualizer LabelizerViewer(frame360);
  LabelizerViewer.run();

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
  extract.setInputCloud (frame360->sphereCloud);
  for(size_t i=0; i < frame360->planes.vPlanes.size(); i++)
  {
    mrpt::pbmap::Plane &plane_i = frame360->planes.vPlanes[i];
//    cout << "plane "<< i << " normal " << plane_i.v3normal[0] << " " << plane_i.v3normal[1] << " " << plane_i.v3normal[2] << " center " << plane_i.v3center[0] << " " << plane_i.v3center[1] << " " << plane_i.v3center[2] << endl;

    pcl::PointIndices inlier_indices;
    inlier_indices.indices.resize(plane_i.inliers.size());
    for(size_t j=0; j < plane_i.inliers.size(); j++)
      inlier_indices.indices[j] = plane_i.inliers[j];
    extract.setIndices ( boost::make_shared<const pcl::PointIndices> (inlier_indices) );
    extract.setNegative (false);
    extract.filter (*plane_i.planePointCloudPtr);    // Write the planar point cloud
//  cout << "planePointCloudPtr size " << plane_i.planePointCloudPtr->size() << endl;
  }

  string input_label = "";

  cout << "Labelize the Frame360-pbmap or 'Enter' to skip it" << endl;
  getline(cin, input_label);
  frame360->planes.label = input_label;

  // Label the planes
  for(plane_idx=0; plane_idx < frame360->planes.vPlanes.size(); plane_idx++)
  {
    cout << "Enter label for plane " << plane_idx << " or 'q' to exit and save: ";
    getline(cin, input_label);
    if(input_label == "q")
      break;
    frame360->planes.vPlanes[plane_idx].label = input_label;
//    cout << "Plane label " << frame360->planes.vPlanes[plane_idx].label << endl;

    cout << "Enter object label: ";
    getline(cin, input_label);
    if(input_label == "q")
      break;
    frame360->planes.vPlanes[plane_idx].label_object = input_label;

    cout << "Enter context label: ";
    getline(cin, input_label);
    if(input_label == "q")
      break;
    frame360->planes.vPlanes[plane_idx].label_context = input_label;
  }
  cout << "\n Manual labelization terminated\n";
  cout << "Close viewer to EXIT\n";

//  std::string pbmapPath = pathToImagesDir + mrpt::format("/spherePlanes_%u.pbmap", frameID); // Warning: if the standard name 'spherePlanes_%u.pbmap' changes, it can be a mess. TODO: rewrite this program to accpect a single input specifying the pbmap
  frame360->savePlanes(pathToPbMap);

  while (!LabelizerViewer.cloudViewer.wasStopped() )
    boost::this_thread::sleep (boost::posix_time::milliseconds (10));

  cout << "EXIT\n";

  return (0);
}
