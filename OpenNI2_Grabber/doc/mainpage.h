/*!\mainpage Documentation Overview
 *
 * \section intro_sec Introduction
 *
 * This project contains the basic functionality to acquire images from a RGB-D sensor using OpenNI2, and to perform some basic operations. Functions to serialize (load/save) such images, to undistort the depth images, or building point clouds are also provided. Two applications are provided.
 *
 * \section project_sec Project tree
 *
 * The main functionality of this project is implemented in a set of header files that are found in the directories: \n 'frameRGBD/'
 *	- FrameRGBD.h
 *	- CloudRGBD.h
 *	- CloudVisualization.h
 * and 'grabber/'
 *	- RGBDGrabber.h
 * The project applications are found in the directory 'apps/'
 *	- Calibration/ \n
 * 	    -> Contains the applications to calibrate the extrinsic parameters of the sensor
 *	- Grabber/ \n
 *          -> Grab and serialize the omnidirectional RGB-D image 
 *
 *
 * \section dependencies_sec Dependencies
 * This project integrates several open-source libraries to build the whole solution. The main dependencies are:
 *    - OpenCV: http://opencv.org/
 *    - PCL: http://pointclouds.org/
 *    - OpenNI2: http://www.openni.org/ (This library is needed to open and read the sensor)    
 *
 * \section install_sec Installation
 * This project has been implemented and tested in Ubuntu 12.04 and 13.04. This project contains a CMalelists.txt file to facilitate the integration of the different dependencies. Thus, the program CMake is required to produce the Makefile configuration file for compilation. To compile the source code the above dependencies must be installed first. After that, the following steps will guide you to compile the project.
\verbatim
cd yourPathTo/RGBD360
\endverbatim
 *
 *    - Generate the Makefile with CMake.
 *         -# Open CMake (the following instructions are for cmake-gui).
 *         -# Set the source directory to RGBD360 and the build directory to RGBD360/build.
 *         -# Set OpenCV_DIR, PCL_DIR and MRPT_DIR to the OpenCV, PCL and MRPT build directories respectively.
 *         -# Set the application packages to build (Grabber, Visualizer, etc.). To reckon which packages you need, go to the next section to find out a more detailed description of each package's applications.
 *         -# Configure.
 *         -# Generate.
 *
 *    - Compile the RGBD360 project.
 *         -# Go to the directory RGBD360/build/
 *         -# Compile with 'make'.
 *
 *
 *
 * \section usage_sec Software usage
 *
 * After compiling the project, a number of directories containing the different application packages will be created. The applications of these packages are described below (a brief description of each application and its syntaxis is shown on executing ./application -h'):
 * \subsection Calibration
 * This package contains the applications to calibrate the extrinsic parameters of the sensor
 *
 * \author	Eduardo Fernandez-Moral
 */
