PROJECT(OpenNI2_Grabber)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

# Set the output directory for the build executables
#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Set the OPENNIGRABBER_SOURCE_DIR variable to the current source directory
set(OPENNIGRABBER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#Project dependencies
FIND_PACKAGE(OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )
link_directories(${OpenCV_LIBS_DIR})

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Link the OpenNI2 dynamic libs. This libraries must be downloaded or compiled (http://www.openni.org/openni-sdk/), and copied to the execution path of this project (by default the binary dir) 
LINK_DIRECTORIES(${CMAKE_BINARY_DIR})
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/OpenNI2/Drivers)

# Include FrameRGBD files (types and basic functionality)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/FrameRGBD)

# Include grabber files (OpenNI2 wrapper)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/grabber)

# Include third party files
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/OpenNI2/Include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/CLAMS/include)

# Select the "apps" to be built
#SET( BUILD_CLOUDGRABBER ON CACHE BOOL "Build CloudGrabber application")

#ADD_SUBDIRECTORY(third_party/CLAMS)
#ADD_SUBDIRECTORY(apps)

# Declare the target (an executable)
ADD_EXECUTABLE(grabber ./apps/OpenNIGrabber_App.cpp
./FrameRGBD/FrameRGBD.h
./grabber/RGBDGrabber_OpenNI2.h	
#./grabber/SerializeFrameRGBD.h
#./grabber/DownsampleRGBD.h	#./grabber/DownsampleRGBD.cpp
)

TARGET_LINK_LIBRARIES(grabber ${OpenCV_LIBS} libOpenNI2.so)

OPTION(BUILD_CLOUD "Build the pointCloud functionality. This requires PCL" OFF)
IF(BUILD_CLOUD)

	FIND_PACKAGE(PCL REQUIRED)
	INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
	LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
	ADD_DEFINITIONS(${PCL_DEFINITIONS})

	ADD_EXECUTABLE(CloudGrabber ./apps/CloudGrabber.cpp
	./FrameRGBD/FrameRGBD.h
	./FrameRGBD/CloudRGBD.h
	./FrameRGBD/CloudVisualizer.h
	./grabber/RGBDGrabber_OpenNI2.h	
	)
	TARGET_LINK_LIBRARIES(CloudGrabber ${OpenCV_LIBS} ${PCL_LIBRARIES} libOpenNI2.so)

ENDIF(BUILD_CLOUD)

# Project options
OPTION(BUILD_SERIALIZATION "Build the serialization functionality. This requires Boost and MRPT" OFF)
IF(BUILD_SERIALIZATION)
	#FIND_PACKAGE(MRPT REQUIRED hwdrivers maps graphslam)
	FIND_PACKAGE(MRPT REQUIRED base hwdrivers slam maps graphslam)

	FIND_PACKAGE( Boost 1.46.0 REQUIRED COMPONENTS serialization )
	INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
ENDIF(BUILD_SERIALIZATION)


OPTION(BUILD_UNDISTORT_LIB "Build the undistorting library from the third_party CLAMS. An aplication to gather frames with OpenNI2, and undistort them is generated as an example." OFF)
IF(BUILD_UNDISTORT_LIB)
	INCLUDE_DIRECTORIES ( /usr/include/eigen3 )
	#INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )

	#FIND_PACKAGE( Boost 1.46.0 REQUIRED COMPONENTS system )
	#INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
	#find_package(Boost REQUIRED COMPONENTS system thread)
	#include_directories(${Boost_INCLUDE_DIR})
	#link_directories(${Boost_LIBRARY_DIR})
	set(Boost_INCLUDE_DIR /usr/include)
	set(Boost_INCLUDE_DIR /usr/include/system)
	set(Boost_INCLUDE_DIR /usr/include/thread)
	set(Boost_LIBRARY_DIR /usr/lib)

	# Create the undistort library
	ADD_LIBRARY(Undistort
			./third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
			./third_party/CLAMS/discrete_depth_distortion_model.cpp
	)
	TARGET_LINK_LIBRARIES(Undistort libboost_system.so libboost_thread.so)

	# Tell CMake that the linker language is C++
	SET_TARGET_PROPERTIES(Undistort PROPERTIES LINKER_LANGUAGE CXX)

	#OPTION(BUILD_TEST "Build the pointCloud functionality. This requires PCL" OFF)

ENDIF(BUILD_UNDISTORT_LIB)


OPTION(BUILD_CLOUD_UNDISTORT2 "Build the undistorting library from the third_party CLAMS. An aplication to gather frames with OpenNI2, and undistort them is generated as an example." OFF)
IF(BUILD_CLOUD_UNDISTORT2)
	INCLUDE_DIRECTORIES ( /usr/include/eigen3 )

	FIND_PACKAGE(PCL REQUIRED)
	INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
	LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
	ADD_DEFINITIONS(${PCL_DEFINITIONS})

	# Declare the target (an executable)
	ADD_EXECUTABLE(CloudGrabberUndist ./apps/CloudGrabber_undistort.cpp
		./FrameRGBD/CloudRGBD.h
		./FrameRGBD/CloudVisualizer.h
		./grabber/RGBDGrabber_OpenNI2.h	

		./third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
		#./third_party/CLAMS/discrete_depth_distortion_model.cpp

	#./grabber/SerializeFrameRGBD.h
	#./grabber/DownsampleRGBD.h	#./grabber/DownsampleRGBD.cpp
	)
#${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
	TARGET_LINK_LIBRARIES(CloudGrabber ${OpenCV_LIBS} ${PCL_LIBRARIES} libboost_system.so libboost_thread.so libOpenNI2.so libUndistort.a)

ENDIF(BUILD_CLOUD_UNDISTORT2)


OPTION(BUILD_CLOUD_UNDISTORT "Build the undistorting library from the third_party CLAMS. An aplication to gather frames with OpenNI2, and undistort them is generated as an example." OFF)
IF(BUILD_CLOUD_UNDISTORT)
	#INCLUDE_DIRECTORIES ( /usr/include/eigen3 )

	#FIND_PACKAGE( Boost 1.46.0 REQUIRED COMPONENTS system )
	#INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
set(Boost_INCLUDE_DIR /usr/include)
set(Boost_INCLUDE_DIR /usr/include/system)
set(Boost_INCLUDE_DIR /usr/include/thread)
set(Boost_LIBRARY_DIR /usr/lib)
#find_package(Boost REQUIRED COMPONENTS system thread)
#include_directories(${Boost_INCLUDE_DIR})
#link_directories(${Boost_LIBRARY_DIR})

	# Declare the target (an executable)
	ADD_EXECUTABLE(CloudGrabber ./apps/CloudGrabber_undistort.cpp
	#./FrameRGBD/FrameRGBD.h
	./FrameRGBD/FrameRGBD_undistort.h
	./grabber/RGBDGrabber.h
	./grabber/RGBDGrabber_OpenNI2.h	
	#./grabber/SerializeFrameRGBD.h
	#./grabber/DownsampleRGBD.h	#./grabber/DownsampleRGBD.cpp
	)
#${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
	TARGET_LINK_LIBRARIES(CloudGrabber ${OpenCV_LIBS} libboost_system.so libboost_thread.so libOpenNI2.so libUndistort.a)

ENDIF(BUILD_CLOUD_UNDISTORT)


# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
