/*!\mainpage Documentation Overview
 *
 * \section intro_sec Introduction
 *
 * This is the documentation of the RGBD360 Project. This project integrates the functionality to do image acquisition, localization and mapping using an omnidirectional RGB-D sensor developed in INRIA Sophia-Antipolis by the team LAGADIC, and with the collaboration of the University of Malaga. This functionality comprises: reading and serializing the data streaming from the omnidirectional RGB-D sensor; registering frames based on a compact planar description of the scene (http://www.mrpt.org/pbmap); loop closure detection; performing human-guided semi-automatic labelization of the scene; PbMap-based hybrid SLAM (i.e. using metric-topological-semantic information) with the omnidirectional RGB-D sensor moving freely with 6 DoF, or in planar movement with 3 DoF. Also, some visualization tools are provided to show the results from the above applications.
 * \internal TO DO: create a graph with the classes relations \endinternal
 * \image html  device+robot.png "Omnidirectional RGB-D sensor and robot set-up." width=10cm
 * \image html  omnidirectional_rgbd_image2.png "Omnidirectional RGB and Depth images."  \n
 *
 * \section project_sec Project tree
 *
 * This project contains a previous one named 'OpenNI2_Grabber', which is used to access and read OpenNI2 sensors like Asus Xtion Pro Live (Asus XPL). Only the header files are used by 'RGBD360', thus it does not require to compile 'OpenNI2_Grabber'. However, this project can be compiled independently to use its test applications. The documentation of this project can be browsed in 'OpenNI2_Grabber/doc/html/index.html'. \n
 *
 * The main functionality provided by this project is implemented in a set of header files that are found in the directory 'include/'
 *	- Calib360.h
 *	- Calibrator.h
 *	- Frame360.h
 *	- Frame360_Visualizer.h
 *	- RegisterRGBD360.h
 *	- FilterPointCloud.h
 *	- Map360.h
 *	- Map360_Visualizer.h
 *	- TopologicalMap360.h
 *	- LoopClosure360.h
 *	- GraphOptimizer.h
 *	- Miscellaneous.h
 *
 * The project applications are structured in different sections depending on its utility:
 *	- Calibration/ \n
 * 	    -> Contains the applications to calibrate the extrinsic parameters of the sensor.
 *	- Grabber/ \n
 *          -> Grab and serialize the omnidirectional RGB-D image stream.
 *	- Labelization/ \n
 *          -> Write semantic labels on the images.
 *	- Registration/ \n
 *          -> Register (align) pairs of spherical RGB-D images. Odometry applications.
 *	- SLAM/ \n
 *          -> Hybrid pose-graph SLAM using metric-topological-semantic information.
 *	- Visualization/ \n
 *          -> Load a serialized image stream. Load and build the spheres.
 *
 * \section dependencies_sec Dependencies
 * This project depends on several open-source libraries to build the whole solution. The main dependencies are:
 *    - OpenCV: http://opencv.org/ (Installing the binaries is recommended. This project has also been tested with the version 2.4.5 compiled from sources.)
 *    - PCL: http://pointclouds.org/ (This project uses the version of Eduardo Fernandez, which can be downloaded from https://github.com/EduFdez/pcl.git)
 *    - MRPT: http://www.mrpt.org/ (This project uses the version of Eduardo Fernandez, which can be downloaded from https://github.com/EduFdez/mrpt.git)
 *    - Eigen: http://eigen.tuxfamily.org (Install the current version for your system)
 *
 * This project also contains some third party code and dependencies which are provided here to facilitate compilation and to avoid possible compatibility issues in the future. They are found in the directory 'OpenNI2_Grabber/third_party/'. These dependencies are:
 *    - OpenNI2: http://www.openni.org/ (This library is needed to open and read the sensor, it is not required to work with data already recorded in datasets.)
 *    - CLAMS: http://www.alexteichman.com/octo/clams/ (This project performs the intrinsic calibration of RGB-D sensors (see "paper"). It is used here to undistort the depth images captured with our device.)
 *
 *
 * \section install_sec Installation
 * This project has been implemented and tested in Ubuntu 12.04 and 13.04. This project contains a CMalelists.txt file to facilitate the integration of the different dependencies. Thus, the software CMake is required to produce the configuration file 'Makefile' for compilation. To compile the source code the above dependencies must be installed first (make sure that their dependencies are also installed, for that, follow the instructions given in the website of each library). MRPT and PCL must be compiled using the sources referred above. A version of OpenNI2 (downloaded on November 8th, 2013) is provided within this project to avoid possible compatibility problems in the future. After that, the following steps will guide you to compile the project.
\verbatim
cd yourPathTo/RGBD360
\endverbatim
 *
 *    - Generate the Makefile with CMake.
 *         -# Open CMake (the following instructions are for cmake-gui).
 *         -# Set the source directory to RGBD360 and the build directory to RGBD360/build.
 *         -# Set OpenCV_DIR, PCL_DIR and MRPT_DIR to the directories containing the built packages from OpenCV, PCL and MRPT respectively.
 *         -# Set the application packages to build (Grabber, Visualizer, etc.). To reckon which packages you need, go to the next section to find out a more detailed description of each package's applications.
 *         -# Configure.
 *         -# Generate.
 *
 *    - Compile the RGBD360 project.
 *         -# Go to the directory RGBD360/build/
 *         -# Compile with 'make'.
 *
 * Important notes:
 * 	   - the library Boost is a dependency of PCL, the version 1.46 (or newer) of Boost is required here.
 * 	   - the executable files that access the omnidirectional RGB-D sensor link dynamically against the library OpenNI2, this requires that those executables and the OpenNI2 library files are found in the same directory. Thus, if you do not build the project in the default 'build/' directory, then you must copy the original content of the 'build/' directory to the directory containing your executables (e.g. RGBD30_Grabber, or OnlineCalibratorRGBD360).
 * 	   - compilation errors occur when different versions of the dependencies (OpenCV, PCL or MRPT) are installed in the machine and they are not correctly specified in CMake (so that include and lib files of different versions are mixed). To solve this problem, make sure that the paths given in CMake-GUI refer to the correct libraries.
 *
 * \section usage_sec Software usage
 *
 * After compiling this project, a number of directories containing the different application packages will be created. The applications of these packages are described below (<b>a brief description of each application and its syntaxis is shown on executing ./application -h'</b>):
 * \subsection Calibration
 * This package contains the applications to calibrate the extrinsic parameters of the sensor
 *
\verbatim
./Calibrator
\endverbatim
 *
 * This program calibrates the extrinsic parameters of the omnidirectional RGB-D device. The key idea is to match planar observations, assuming that the dominant planes (e.g. walls, ceiling or floor) can be observed at the same time by several contiguous sensors (take into account that the overlapping between the different sensors is negligible). The planes are segmented from the depth images using a region growing approach (thus, color information is not used for calibration). These planes are matched automatically according to the device construction specifications, which are refined by this program. This program opens the sensor, which has to be moved to take different plane observations at different angles and distances. When enough information has been collected, a Gauss-Newtown optimization is launched to obtain the extrinsic calibration, which can be saved if the user demands it after visual validation of the calibrated images.
 *
 * \subsection Grabber
 * Read the omnidirectional RGB-D image stream from the RGBD360 sensor.
 *
\verbatim
./RGBD360_Grabber <pathToSaveToDisk>
\endverbatim
 *
 * \t This program accesses the omnidirectional RGB-D sensor, and reads the image streaming it captures. The image stream is recorded in the path specified by the user.
 * \subsection Labelization
 * This package contains applications to annotate semantic labels on the spherical RGB-D images and on the planes extracted from them.
 *
\verbatim
./LabelizeFrame <pathToPbMap>
\endverbatim
 *
 * This program loads a PbMap (a its corresponding point cloud) and asks the user to labelize the planes before saving the annotated PbMap to disk.
 *
\verbatim
./LabelizeSequence <pathToFolderWithSpheres>
\endverbatim
 *
 * This program loads a stream of previously built spheres (PbMap+PointCloud) partially labelized, and expands the labels by doing consecutive frame registration.
 *
 * \subsection Registration
 * This package contains applications based on the registration of pairs of RGB-D images through the matching and alignment of the PbMaps extracted from them.
 *
\verbatim
./OdometryRGBD360 <pathToRawRGBDImagesDir> <pathToResults> <sampleStream>
\endverbatim
 *
 * This program performs PbMap-based Odometry from the data stream recorded by an omnidirectional RGB-D sensor.
 *
\verbatim
./RegisterPairRGBD360 <frame360_1_1.bin> <frame360_1_2.bin>
\endverbatim
 *
 * This program loads two raw omnidireactional RGB-D images and aligns them using PbMap-based registration.
 *
 * \subsection SLAM
 * This package is dedicated to SLAM applications.
 *
\verbatim
./SphereGraphSLAM <pathToRawRGBDImagesDir>
\endverbatim
 *
 * This program performs metric-topological SLAM from the data stream recorded by an omnidirectional RGB-D sensor. The directory containing the input raw omnidireactional RGB-D images (.frame360 files) has to be specified.
 *
 * \subsection Visualization
 * This package contains applications to read and visualize the spherical RGB-D images, and the point clouds and PbMaps built from them.
 *
\verbatim
./LoadFrame360 <pathToFrame360.bin>
\endverbatim
 *
 * This program loads a Frame360.bin (an omnidirectional RGB-D image in raw binary format). It builds the pointCloud and creates a PbMap from it. The spherical frame is shown: the keys 'k' and 'l' are used to switch between visualization modes.
 *
\verbatim
./LoadSequence <mode> <pathToData> <pathToResultsFolder>
\endverbatim
 *
 * This program loads a sequence of observations 'RGBD360.bin' and visualizes and/or saves the spherical images, the pointCloud or the PbMap extracted from it according to the command options. \n
 * mode = 1 -> Show the reconstructed spherical images \n
 * mode = 2 -> Show and save the reconstructed spherical images \n
 * mode = 3 -> Show the reconstructed PointCloud and the PbMap \n
 * mode = 4 -> Save the reconstructed PointCloud and the PbMap \n
 * mode = 5 -> Show a video streaming of the reconstructed PointCloud \n
 *
\verbatim
./LoadSphere <pathToPointCloud> <pathToPbMap>
\endverbatim
 *
 * \t \t This program loads the pointCloud and the PbMap from a RGBD360 observation.
 * \n \n
 *
 *
 *
 * \author	Eduardo Fernandez-Moral
 */
