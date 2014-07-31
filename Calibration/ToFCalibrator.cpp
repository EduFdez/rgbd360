#include <mrpt/base.h>
#include <mrpt/gui.h>
#include <mrpt/opengl.h>
#include <mrpt/slam.h>
#include <mrpt/utils.h>
#include <mrpt/obs.h>

//#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

//#include <pcl/io/openni_grabber.h>
//#include <pcl/common/time.h>
//#include <pcl/visualization/cloud_viewer.h>

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace mrpt;
using namespace mrpt::gui;
using namespace mrpt::opengl;
using namespace mrpt::math;



class DatasetParser{
private:
	mrpt::slam::CRawlog m_dataset;
	ifstream m_fgt;
	bool m_last_groundtruth_ok;
	bool m_groundtruth_ok;
	//bool m_first_frame = true;

	double m_last_groundtruth;
	CPose3D m_pose;
	CColouredPointsMap m_pntsMap;
	CImage  m_depthimg, m_colorimg;
	pcl::PointCloud<pcl::PointXYZRGB> m_pclCloud;


public:
	int m_count;
	DatasetParser(string path_rawlog){//,string path_groundtruth){

//		m_last_groundtruth_ok = true;
//		m_groundtruth_ok = true;
		m_count = 0;
//		m_last_groundtruth = 0;

		//TODO error checking
		m_dataset.loadFromRawLogFile(path_rawlog);
//		m_fgt.open(path_groundtruth.c_str());

//		path_rawlog.replace(path_rawlog.find(".rawlog"),7,"_Images/");
//		// Set external images directory:
//		CImage::IMAGES_PATH_BASE = path_rawlog;

//		char aux[100];
//		m_fgt.getline(aux, 100);
//		m_fgt.getline(aux, 100);
//		m_fgt.getline(aux, 100);
//		m_fgt >> m_last_groundtruth;
//		m_last_groundtruth_ok = true;
	}

	//LoadNextFrame returns true if the next frame has correct ground truth.
	bool LoadNextFrame(void){
		CObservationPtr obs;
//#include <mrpt/slam/CSensoryFrame.h>
//		mrpt::slam::CSensoryFrame obs;

		do
		{
			if (m_dataset.size() <= m_count)
				return false;
			obs = m_dataset.getAsObservation(m_count);

//  cout << "Dataset size " << m_dataset.size() << endl;
//  if (IS_CLASS(obs, CObservation3DRangeScan))
//    cout << "LoadNextFrame CObservation3DRangeScan" << endl;
//  //if (IS_CLASS(obs, mrpt::slam::CSensoryFrame))
//  else
//    cout << "LoadNextFrame CSensoryFrame" << endl;

			m_count++;
		}
		while (!IS_CLASS(obs, CObservation3DRangeScan));

		CObservation3DRangeScanPtr obsRGBD = CObservation3DRangeScanPtr(obs);
		obsRGBD->load();


		//Calculate the 3D cloud from RGBD and include in the observation.
		//obsRGBD->project3DPointsFromDepthImage();
		obsRGBD->project3DPointsFromDepthImageInto(*obsRGBD,false,NULL,true);
		//obsRGBD->project3DPointsFromDepthImageInto(m_pclCloud,false,NULL,true);

		cout << "obsRGBD->points3D_x.size(): " << obsRGBD->points3D_x.size() << endl;


//		//Copy to the CColouredPointsMap
//		m_pntsMap.clear();
//		m_pntsMap.colorScheme.scheme = CColouredPointsMap::cmFromIntensityImage;
//		m_pntsMap.insertionOptions.minDistBetweenLaserPoints = 0; // don't drop any point
//		m_pntsMap.insertionOptions.disableDeletion = true;
//		m_pntsMap.insertionOptions.fuseWithExisting = false;
//		m_pntsMap.insertionOptions.insertInvalidPoints = true;
//		m_pntsMap.insertObservation(obs.pointer());
//
//		cout << "m_pntsMap.size():" << m_pntsMap.size() << endl;

		//Copy rgb and depth images.
		m_depthimg.setFromMatrix(obsRGBD->rangeImage);
		m_colorimg = obsRGBD->intensityImage;

		cout << "depth img width: " << m_depthimg.getWidth()<<endl;
		cout << "depth img height: " << m_depthimg.getHeight()<<endl;


		double timestamp_gt;
		double timestamp_obs = timestampTotime_t(obsRGBD->timestamp);

		obsRGBD->unload();

//		return m_groundtruth_ok;
		return true;
	}


	CPose3D getGroundTruthPose(){
		return m_pose;
	}

	CColouredPointsMap getPointCloud(){
		return m_pntsMap;
	}

	pcl::PointCloud<pcl::PointXYZRGB> getpclCloud(){
		return m_pclCloud;
	}

	CImage getRGB(){
		return m_colorimg;
	}

	CImage getD(){
		return m_depthimg;
	}

};

int main ()
{
	CTicTac  tictac;

	size_t frame=0;

  CRawlog dataset;
//  dataset.loadFromRawLogFile("/media/Data/Datasets360/ToF+Kinect/varevalo/dataset_2014-01-24_14h29m42s.rawlog");
  dataset.loadFromRawLogFile("/media/Data/Datasets360/ToF+Kinect/varevalo/dataset_2014-01-24_14h34m55s.rawlog");
  cout << dataset.size() << " entries loaded." << endl;

  cout << "Read dataset\n";

  mrpt::math::CMatrixDouble correspMat(0,8);

  while(frame < dataset.size())
  {
    CSensoryFramePtr observations = dataset.getAsObservations(frame);

    bool bObsKinect = false, bObsToF = false;
    CObservation3DRangeScanPtr obsKinect;
    CObservation3DRangeScanPtr obsToF;

//    cout << "Read frame CSensoryFrame\n";

    // Process action & observations
    // action, observations should contain a pair of valid data (Format #1 rawlog file)
    for(CSensoryFrame::iterator it=observations->begin(); it != observations->end(); ++it)
    {
      if(IS_CLASS(*it, CObservation3DRangeScan))
        if((*it)->sensorLabel == "KINECT")
        {
//          cout << "Read KINECT\n";
          obsKinect = CObservation3DRangeScanPtr(*it);
          bObsKinect = true;
        }
        else if((*it)->sensorLabel == "CAM3D")
        {
//          cout << "Read CAM3D\n";
          obsToF = CObservation3DRangeScanPtr(*it);
//          obsToF->load();
//          obsToF->project3DPointsFromDepthImageInto(*obsToF,false,NULL,true);
          bObsToF = true;
        }
    }
    if(bObsKinect && bObsToF)
    {
      cout << "Valid observation in CSensoryFrame " << frame << endl;

      //Copy to the CColouredPointsMap
      CColouredPointsMap m_pntsMap;
//      CPointsMap m_pntsMap;

//  		m_pntsMap.clear();
      m_pntsMap.colorScheme.scheme = CColouredPointsMap::cmFromIntensityImage;
      m_pntsMap.insertionOptions.minDistBetweenLaserPoints = 0; // don't drop any point
      m_pntsMap.insertionOptions.disableDeletion = true;
      m_pntsMap.insertionOptions.fuseWithExisting = false;
      m_pntsMap.insertionOptions.insertInvalidPoints = true;
      m_pntsMap.insertObservation(obsKinect.pointer());
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudKinect(new pcl::PointCloud<pcl::PointXYZRGB>);
      m_pntsMap.getPCLPointCloud(*cloudKinect);

      cout << "Cloud-Kinect pts " << cloudKinect->points.size() << endl;

      //Extract a plane with RANSAC
      Eigen::VectorXf modelcoeff_Kinect(4);
      vector<int> inliers;
      pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloudKinect));
      pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_p);
      ransac.setDistanceThreshold (.03);
      tictac.Tic();
      ransac.computeModel();
      ransac.getModelCoefficients(modelcoeff_Kinect);
      if(modelcoeff_Kinect[3] < 0) modelcoeff_Kinect *= -1;
//      modelcoeff_Kinect *= (modelcoeff_Kinect[3]/fabs(modelcoeff_Kinect[3]));
    cout << "RANSAC (pcl) computation time: " << tictac.Tac()*1000.0 << " ms " << modelcoeff_Kinect.transpose() << endl;
      ransac.getInliers(inliers);

      // Segment plane in the point cloud from the ToF
  		m_pntsMap.clear();
      m_pntsMap.colorScheme.scheme = CColouredPointsMap::cmFromIntensityImage;
      m_pntsMap.insertionOptions.minDistBetweenLaserPoints = 0; // don't drop any point
      m_pntsMap.insertionOptions.disableDeletion = true;
      m_pntsMap.insertionOptions.fuseWithExisting = false;
      m_pntsMap.insertionOptions.insertInvalidPoints = true;
      m_pntsMap.insertObservation(obsToF.pointer());
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudToF(new pcl::PointCloud<pcl::PointXYZRGB>);
      m_pntsMap.getPCLPointCloud(*cloudToF);

      cout << "Cloud-ToF pts " << cloudToF->points.size() << endl;

      //Extract a plane with RANSAC
      Eigen::VectorXf modelcoeff_ToF(4);
      vector<int> inliers_ToF;
      pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_pl(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloudToF));
      pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac_ToF (model_pl);
      ransac_ToF.setDistanceThreshold (.03);
      tictac.Tic();
      ransac_ToF.computeModel();
      ransac_ToF.getModelCoefficients(modelcoeff_ToF);
      if(modelcoeff_ToF[3] < 0) modelcoeff_ToF *= -1;
    cout << "RANSAC (pcl) computation time: " << tictac.Tac()*1000.0 << " ms " << modelcoeff_ToF.transpose() << endl;
      ransac_ToF.getInliers(inliers_ToF);

      unsigned prevSize = correspMat.getRowCount();
      correspMat.setSize(prevSize+1, correspMat.getColCount());
      correspMat(prevSize, 0) = modelcoeff_Kinect[0];
      correspMat(prevSize, 1) = modelcoeff_Kinect[1];
      correspMat(prevSize, 2) = modelcoeff_Kinect[2];
      correspMat(prevSize, 3) = modelcoeff_Kinect[3];
      correspMat(prevSize, 4) = modelcoeff_ToF[0];
      correspMat(prevSize, 5) = modelcoeff_ToF[1];
      correspMat(prevSize, 6) = modelcoeff_ToF[2];
      correspMat(prevSize, 7) = modelcoeff_ToF[3];
    }

    frame++;
  }

  cout << "\tSave CorrespMat\n";
  correspMat.saveToTextFile( mrpt::format("%s/correspondences.txt", PROJECT_SOURCE_PATH) );

//  CFileGZInputStream   rawlogFile("/media/Data/Datasets360/ToF+Kinect/varevalo/dataset_2014-01-24_14h29m42s.rawlog");   // "file.rawlog"
//  CActionCollectionPtr action;
//  CSensoryFramePtr     observations;
//  CObservationPtr      observation;
//  size_t              rawlogEntry=0;
//  bool                end = false;
//
//  cout << "Read CRawlog Generic\n";
//
//  // Read from the rawlog:
//  while ( CRawlog::getActionObservationPairOrObservation(
//                                               rawlogFile,      // Input file
//                                               action,            // Possible out var: action of a pair action/obs
//                                               observations,  // Possible out var: obs's of a pair action/obs
//                                               observation,    // Possible out var: a single obs.
//                                               rawlogEntry    // Just an I/O counter
//                                               ) )
//  {
//    bool bObsKinect = false, bObsToF = false;
//    CObservation3DRangeScanPtr obsKinect;
//    CObservation3DRangeScanPtr obsToF;
//
//    cout << "Read frame\n";
//
//    // Process action & observations
//    if (observation)
//    {
//      // Read a single observation from the rawlog (Format #2 rawlog file)
//      cout << "This rawlog contains observations\n";
//    }
//    else if(observations)
//    {
//      cout << "Read CSensoryFrame\n";
//      // action, observations should contain a pair of valid data (Format #1 rawlog file)
//      for(CSensoryFrame::iterator it=observations->begin(); it != observations->end(); ++it)
//      {
//        if(IS_CLASS(*it, CObservation3DRangeScan))
//          if((*it)->sensorLabel == "KINECT")
//          {
//            cout << "Read KINECT\n";
//            obsKinect = CObservation3DRangeScanPtr(*it);
//            bObsKinect = true;
//          }
//          else if((*it)->sensorLabel == "CAM3D")
//          {
//            cout << "Read CAM3D\n";
//            obsToF = CObservation3DRangeScanPtr(*it);
//            bObsToF = true;
//          }
//      }
//      if(bObsKinect && bObsToF)
//      {
//        cout << "Valid observation in CSensoryFrame " << frame << endl;
//      }
//
//    }
//
//    frame++;
//  };
//  // Smart pointers will be deleted automatically.

//	DatasetParser p("/media/Data/Datasets360/ToF+Kinect/varevalo/dataset_2014-01-24_14h29m42s.rawlog");
//cout << "DatasetParser " << endl;
//
//	CColouredPointsMap displaycloud;
//	CImage displayimg,displayimg2;
//	cv::Mat dimg,dimg2;
//	CPointCloudColouredPtr viscloud = CPointCloudColoured::Create();
//
//
//	viscloud->setLocation(0,0,0);
//	viscloud->setPointSize(2.0);
//	viscloud->enablePointSmooth();
//	//viscloud->enableColorFromY();
//
////	int b = 0;
////
////	vector<TColor> colorlist;
////	colorlist.clear();
////	colorlist.push_back(TColor(141,211,199));
////	colorlist.push_back(TColor(255,255,179));
////	colorlist.push_back(TColor(190,186,218));
////	colorlist.push_back(TColor(251,128,114));
////	colorlist.push_back(TColor(128,177,211));
////	colorlist.push_back(TColor(253,180,98));
////	colorlist.push_back(TColor(179,222,105));
////	colorlist.push_back(TColor(252,205,229));
////	colorlist.push_back(TColor(217,217,217));
////	colorlist.push_back(TColor(188,128,189));
////	colorlist.push_back(TColor(204,235,197));
////	colorlist.push_back(TColor(255,237,111));
//
//	//while (p.LoadNextFrame())
//cout << "Load observation " << endl;
//
//	p.LoadNextFrame();
//
//cout << "Load observation1 " << endl;
//
//	//        cout << i++ << endl;
//	//        if (b++ < 10) continue; //take 1 of every 10 frames
//	//        b = 0;
//
////	CColouredPointsMap a = p.getPointCloud();
//
//	//displaycloud.insertAnotherMap(&a,p.getGroundTruthPose()); //Adds the map to the other one, adding a transformation (rot+translation)
//	//a.changeCoordinatesReference(p.getGroundTruthPose()); // Rotate and translate
//	//float fusedist = 0.02;
//	//displaycloud.fuseWith(&a); //Fuses the map (points very close to each other are fused instead of added) -Doesn't include rotation or translation.
//
//
//
//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
//	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);
//	//std::vector<int> inliers;
//
//
//	*cloud = p.getpclCloud();
////	a.getPCLPointCloud(*cloud);
//	cloud->width = p.getD().getWidth();
//	cloud->height = p.getD().getHeight();
//	cout << "Cloud size:" << cloud->points.size() << endl;
////	float R,G,B;
////	for (size_t i=0; i < cloud->size(); ++i) {
////		a.getPointColor(i,R,G,B);
////		cloud->points[i].r = R*255;
////		cloud->points[i].g = G*255;
////		cloud->points[i].b = B*255;
////	}
//
//	//		if(0) //Organized multiplane segmentation
//	//		{
//	//			// Estimate point normals
//	//			tictac.Tic();
//	//			pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
//	//			pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
//	//			pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
//	//			ne.setSearchMethod (tree);
//	//			ne.setInputCloud (cloud);
//	//			ne.setKSearch (50);
//	//			ne.compute (*normal_cloud);
//	//			cout << "Normals estimation time: " << tictac.Tac()*1000.0 << " ms" << endl;
//
//
//	//			//ORGANIZED MULTIPLANE SEGMENTATION
//	//			tictac.Tic();
//	//			pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGB, pcl::Normal, pcl::Label > mps;
//	//			mps.setMinInliers (10000);
//	//			mps.setAngularThreshold (0.017453 * 2.0); // 2 degrees
//	//			mps.setDistanceThreshold (0.02); // 2cm
//	//			mps.setInputNormals (normal_cloud);
//	//			mps.setInputCloud (cloud);
//	//			std::vector< pcl::PlanarRegion< pcl::PointXYZRGB >, Eigen::aligned_allocator< pcl::PlanarRegion< pcl::PointXYZRGB > > > regions;
//	//			mps.segment(regions);
//	//			cout << "Multiplane segmentation time: " << tictac.Tac()*1000.0 << " ms" << endl;
//	//		}
//
//	Eigen::VectorXf modelcoeff(4);
//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//	float x,y,z;
//	int imax,imin;
//	vector<int> inliers;
//	if(1) //Extract a plane with RANSAC
//	{
//		pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr
//				model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloud));
//
//		pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_p);
//		ransac.setDistanceThreshold (.03);
//
//		tictac.Tic();
//		ransac.computeModel();
//		cout << "RANSAC (pcl) computation time: " << tictac.Tac()*1000.0 << " ms" << endl;
//
//
//		ransac.getModelCoefficients(modelcoeff);
//		//cout << modelcoeff << endl;
//
//
//		ransac.getInliers(inliers);
//		//cout << "inliers size:" << inliers.size() << endl;
//
//
//		//displayimg.resize(p.getRGB().getWidth(),
//		//				  p.getRGB().getHeight(),
//		//				  3,
//		//				  true);
//		displayimg2.resize(p.getRGB().getWidth(),
//						   p.getRGB().getHeight(),
//						   3,
//						   true);
//		displayimg=(p.getRGB());
//		//displayimg2=p.getRGB();
//
//		for (int i=0;i<inliers.size();i++)
//		{
//			int j=inliers[i];
//			int r = j/cloud->width;
//			int c = j%cloud->width;
//
//			size_t color = (size_t)((*displayimg(c,r,2)<<16)+
//									(*displayimg(c,r,1)<<8 )+
//									(*displayimg(c,r,0)));
//			displayimg2.setPixel(c,r,color);
//			//displayimg.setPixel(c,r,colorlist[5]);
//		}
//
//		//dimg = displayimg.getAs<IplImage>();
//		dimg2 = displayimg2.getAs<IplImage>();
//
//		//		cv::Mat M(3,3,CV_32F,cvScalar(0));
//		//		M.at<float>(0,0) = .1;
//		//		M.at<float>(1,1) = .5;
//		//		M.at<float>(2,2) = 1;
//
//		//M = cv::getPerspectiveTransform();
//
//		//cv::Size sz(640,480);
//		//cv::warpPerspective(dimg2,dimg2,M,sz);
//
//
//		if(1)
//		{
//			// copies all inliers of the model computed to another PointCloud
//			pcl::copyPointCloud<pcl::PointXYZRGB>
//					(*cloud, inliers, *segmented_cloud);
//
//		}
//
//		//cout << "modelcoeff: " << modelcoeff << endl;
//		x = -modelcoeff(0)*modelcoeff(3);
//		y = -modelcoeff(1)*modelcoeff(3);
//		z = -modelcoeff(2)*modelcoeff(3);
//
//
//
//		Eigen::Vector3f y_dir,z_axis,origin;
//		Eigen::Affine3f tform;
//		z_axis[0] = x;
//		z_axis[1] = y;
//		z_axis[2] = z;
//		origin=z_axis;
//
//		y_dir[0]=0;
//		y_dir[1]=-1;
//		y_dir[2]=0;
//
//		//ortogonal al eje y y al vector normal del plano
//		y_dir = z_axis.cross(y_dir).eval();
//
//		pcl::getTransformationFromTwoUnitVectorsAndOrigin(
//					y_dir,
//					z_axis,
//					origin,
//					tform);
//
//		pcl::transformPointCloud
//				(*segmented_cloud,
//				 *segmented_cloud,
//				 tform);
//
//		pcl::transformPointCloud
//				(*cloud,
//				 *cloud,
//				 tform);
//
//		pcl::PointXYZRGB pMin,pMax;
//		//pcl::getMinMax3D(segmented_cloud,pMin,pMax);
//		pMin = cloud->at(inliers[0]);
//		pMax = cloud->at(inliers[0]);
//
//		for (int i = 1; i < inliers.size() ; i++){
//			pcl::PointXYZRGB pt = cloud->at(inliers[i]);
//			if (pt.x > pMax.x || pt.y > pMax.y){
//				pMax = pt;
//				imax = inliers[i];
//			}
//			if (pt.x < pMin.x || pt.y < pMin.y){
//				pMin = pt;
//				imin = inliers[i];
//			}
//		}
//
//		cout << "imin:"<<imin<<". imax:"<<imax<<'.'<<endl;
//
//	}
//
//
////	cv::warpPerspective(dimg2,
////						dimg,
////						tform,
////						sz
////						);
//
////	if(1) //Displays
////	{
////
////		if(1){
////			namedWindow( "fig", cv::WINDOW_AUTOSIZE );// Create a window for display.
////			imshow( "fig", dimg );                   // Show our image inside it.
////			namedWindow( "fig2", cv::WINDOW_AUTOSIZE );// Create a window for display.
////			imshow( "fig2", dimg2 );                   // Show our image inside it.
////			cv::waitKey(0);
////		}
////
////		//TODO SLOW. FIX!
////		if(1)//Copy inliers to the displaycloud for visualization.
////		{
////			for(size_t i=0;i<segmented_cloud->size();i++)
////				displaycloud.insertPoint(segmented_cloud->points[i].x,
////										 segmented_cloud->points[i].y,
////										 segmented_cloud->points[i].z,
////										 segmented_cloud->points[i].r/255.0,
////										 segmented_cloud->points[i].g/255.0,
////										 segmented_cloud->points[i].b/255.0);
////		}
////		else//Copy all the cloud
////		{
////			for(size_t i=0;i<cloud->size();i++)
////				displaycloud.insertPoint(cloud->points[i].x,
////										 cloud->points[i].y,
////										 cloud->points[i].z,
////										 cloud->points[i].r/255.0,
////										 cloud->points[i].g/255.0,
////										 cloud->points[i].b/255.0);
////		}
////
////		Visualizer v;
////		v.win.get3DSceneAndLock();
////		v.theScene->insert(viscloud);
////		viscloud->loadFromPointsMap(&displaycloud);
////
////		if(1)
////		{
////			opengl::CSimpleLinePtr glLine = opengl::CSimpleLine::Create();
////			glLine->setLineCoords(0,0,0,x,y,z);
////			v.theScene->insert(glLine);
////		}
////
////
////		if(1) //Include the plane model into the 3D scene.
////		{
////
////			TPlane plane(modelcoeff(0),modelcoeff(1),modelcoeff(2),modelcoeff(3));
////			opengl::CTexturedPlanePtr glPlane = opengl::CTexturedPlane::Create(-3,3,-3,3);
////			CPose3D glPlanePose;
////			plane.getAsPose3D(glPlanePose);
////			glPlane->setPose(glPlanePose);
////			glPlane->setColor(colorlist[4].R/255.0,colorlist[4].G/255.0,colorlist[4].B/255.0, 0.5);
////			v.theScene->insert( glPlane );
////
////			//opengl::CTexturedPlanePtr glPlane2 = opengl::CTexturedPlane::Create(-2,2,-2,2);
////			//glPlane2->setPose(glPlanePose);
////			//glPlane2->setColor(colorlist[1].R/255.0,colorlist[1].G/255.0,colorlist[1].B/255.0, 0.5);
////			//v.theScene->insert( glPlane2 );
////		}
////
////		v.win.unlockAccess3DScene();
////
////		while (v.win.isOpen())
////		{
////			v.win.repaint();
////			mrpt::system::sleep(10);
////		}
////
////	}

	return (0);
}
