PROJECT(RGBD360)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

# Set the same build directory for all executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR} CACHE PATH "Build directory" FORCE)

include(cmakemodules/script_SIMD.cmake REQUIRED)        # SSE2/SSE3/... optimization options

# Now SSE is added automatically below
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2")


# The script below is an adaptation of cmakemodules/script_SIMD.cmake from "MRPT"
# ----------------------------------------------------------------------------
# 			Set the maximum level of warnings:
# ----------------------------------------------------------------------------
SET( EXTRA_WARNINGS OFF CACHE BOOL "Enable extra warnings apart from -Wall")
MARK_AS_ADVANCED(EXTRA_WARNINGS)

# Should be set to true for development
SET( WARNINGS_ARE_ERRORS OFF CACHE BOOL "Treat warnings as errors")
MARK_AS_ADVANCED(WARNINGS_ARE_ERRORS)

# Even more warnings for different compilers
IF(EXTRA_WARNINGS)
   #TODO - Add options for MSVC
   IF (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wreturn-type -Wextra") # -Wold-style-cast  -Woverloaded-virtual
   ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
        ENDIF()
ENDIF(EXTRA_WARNINGS)

# Whole program optimization?
SET( WHOLE_PROGRAM_OPTIMIZATION OFF CACHE BOOL "Flags for whole program optimization.")
MARK_AS_ADVANCED(WHOLE_PROGRAM_OPTIMIZATION)

# Enable profiling?
SET(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling (add -g -pg in GCC/CLANG, /PROFILE in Visual C++)")

# GNU GCC options ================================
IF(CMAKE_COMPILER_IS_GNUCXX)
        # Wall & pedantic?
        IF(APPLE)
                SET(BUILD_GCC_PEDANTIC_DEFAULT "OFF")
        ELSE(APPLE)
                SET(BUILD_GCC_PEDANTIC_DEFAULT "ON")
        ENDIF(APPLE)

        SET( BUILD_GCC_PEDANTIC ${BUILD_GCC_PEDANTIC_DEFAULT} CACHE BOOL "Enable pedantic error detection (with GCC only)")
        MARK_AS_ADVANCED(BUILD_GCC_PEDANTIC)

        # High level of warnings.
        # The -Wno-long-long is required in 64bit systems when including sytem headers.
        # The -Wno-variadic-macros was needed for Eigen3, StdVector.h
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CONFIGURE_CFLAGS} -Wall -Wno-long-long -Wno-variadic-macros")
  IF(NOT APPLE)
    # This causes the option "-Wnorite-strings" to be set on gcc-4.9 on OS X
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")
  ENDIF(NOT APPLE)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CONFIGURE_CFLAGS}")

        # Use "modern" C99 ! ;-)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")


        IF(BUILD_GCC_PEDANTIC)
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
                SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
        ENDIF(BUILD_GCC_PEDANTIC)

        # Enable C++11 features in GCC 4.7+
        IF (NOT CMAKE_GCC_VERSION LESS 47)
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        ENDIF (NOT CMAKE_GCC_VERSION LESS 47)

        # Parallel stdlibc++?
        IF(ENABLE_LIBSTD_PARALLEL_MODE)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -D_GLIBCXX_PARALLEL -fopenmp")
        ENDIF(ENABLE_LIBSTD_PARALLEL_MODE)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# Shared options between GCC and CLANG:
# ======================================
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX)
        IF(WARNINGS_ARE_ERRORS)
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
                SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
        ENDIF(WARNINGS_ARE_ERRORS)

        # We need pthread's
        IF (UNIX AND NOT APPLE)
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
        ENDIF(UNIX AND NOT APPLE)

        IF(CMAKE_BUILD_TYPE MATCHES "Debug")
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -g")
                ADD_DEFINITIONS( -D_DEBUG)
                ADD_DEFINITIONS( -DDEBUG)
        ENDIF(CMAKE_BUILD_TYPE MATCHES "Debug")

        # Profiling?
        IF(ENABLE_PROFILING)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -pg -g")
        ELSEIF(ENABLE_PROFILING)
                # Remove unreferenced functions: function level linking
        # Remove unreferenced functions: function level linking
        if(NOT APPLE)
                        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections")
        endif(NOT APPLE)
        ENDIF(ENABLE_PROFILING)

        # Whole program optimization
        IF(WHOLE_PROGRAM_OPTIMIZATION)
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --combine")
                SET(EXES_CXX_FLAGS "${EXES_CXX_FLAGS} -fwhole-program --combine")
        ENDIF(WHOLE_PROGRAM_OPTIMIZATION)


        # "-mtune=native" generates code optimized for the detected current processor.
        if (WIN32 OR "${CMAKE_ARCH}" MATCHES ".*86" OR "${CMAKE_ARCH}" MATCHES "amd64")
                set(opt_native_def ON)
        endif (WIN32 OR "${CMAKE_ARCH}" MATCHES ".*86" OR "${CMAKE_ARCH}" MATCHES "amd64")

        SET( OPTIMIZE_NATIVE ${opt_native_def} CACHE BOOL "GCC/clang optimizations for current processor (-mtune=native). Requires GCC 4.2+")

        IF(CMAKE_BUILD_TYPE STREQUAL "Debug")  # Not in debug!
                SET(OPTIMIZE_NATIVE OFF)
        ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

        # "-ffast-math"
        SET( OPTIMIZE_FFAST-MATH OFF CACHE BOOL "GCC/clang optimization for floating math (-ffast-math).")
        IF(OPTIMIZE_FFAST-MATH)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -ffast-math")
        ENDIF(OPTIMIZE_FFAST-MATH)

        # SSE2?
        IF (CMAKE_HAS_SSE2)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -msse2 -funroll-loops")
                IF (CMAKE_COMPILER_IS_GNUCXX)
                        SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -mfpmath=sse")
                ENDIF(CMAKE_COMPILER_IS_GNUCXX)
        ENDIF(CMAKE_HAS_SSE2)

        # SSE3?
        IF (CMAKE_HAS_SSE3)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -msse3 -mssse3")
        ENDIF(CMAKE_HAS_SSE3)

        # SSE4*?
        IF (CMAKE_HAS_SSE4_1)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -msse4.1")
        ENDIF(CMAKE_HAS_SSE4_1)
        IF (CMAKE_HAS_SSE4_2)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -msse4.2")
        ENDIF(CMAKE_HAS_SSE4_2)
        IF (CMAKE_HAS_SSE4_A)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -msse4a")
        ENDIF(CMAKE_HAS_SSE4_A)

        # AVX?
        IF (CMAKE_HAS_AVX)
                SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -mavx")
        ENDIF(CMAKE_HAS_AVX)

        # AVX2?
        #IF (CMAKE_HAS_SSE3)
        #        SET(EXTRA_CPP_FLAGS "${EXTRA_CPP_FLAGS} -mavx2")
        #ENDIF(CMAKE_HAS_SSE3)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CPP_FLAGS}")

endif ()

FIND_PACKAGE( Boost 1.46.0 REQUIRED serialization )

# Set the RGBD360_SOURCE_DIR variable to the current source directory
set(RGBD360_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#Add here your project dependencies
#FIND_PACKAGE(MRPT REQUIRED hwdrivers maps graphslam)
FIND_PACKAGE(MRPT REQUIRED base hwdrivers obs slam maps graphslam pbmap)
FIND_PACKAGE(PCL REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED )

INCLUDE_DIRECTORIES(${MRPT_INCLUDE_DIRS})

LINK_DIRECTORIES(${OpenCV_LIBS_DIR})

INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
ADD_DEFINITIONS(${PCL_DEFINITIONS})

#FIND_PACKAGE(G2O REQUIRED)
#INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR})

# OpenMP library
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

SET( CMAKE_TEST_SIMD OFF CACHE BOOL "Enable/disable SIMD tests (SSE, AVX): check that the SIMD functions produce the same result than the standard functions")
IF (CMAKE_TEST_SIMD)
	add_definitions(-D_TEST_SIMD=1)
ELSE (CMAKE_TEST_SIMD)
	add_definitions(-D_TEST_SIMD=0)
ENDIF (CMAKE_TEST_SIMD)

SET( CMAKE_ENABLE_OPENMP OFF CACHE BOOL "Enable/disable OpenMP")
IF (CMAKE_ENABLE_OPENMP)
	add_definitions(-D_ENABLE_OPENMP=1)
ELSE (CMAKE_ENABLE_OPENMP)
	add_definitions(-D_ENABLE_OPENMP=0)
ENDIF (CMAKE_ENABLE_OPENMP)

SET( CMAKE_PRINT_PROFILING ON CACHE BOOL "Enable/disable Profiling: print the time costs of the functions")
IF (CMAKE_PRINT_PROFILING)
	add_definitions(-D_PRINT_PROFILING=1)
ELSE (CMAKE_PRINT_PROFILING)
	add_definitions(-D_PRINT_PROFILING=0)
ENDIF (CMAKE_PRINT_PROFILING)
#configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_SOURCE_DIR}/include/config.h) 

# Create the DR library (Direct Registration)
ADD_LIBRARY(DR
            ${PROJECT_SOURCE_DIR}/include/DirectRegistration.h
            ${PROJECT_SOURCE_DIR}/src/DirectRegistration.cpp
            ${PROJECT_SOURCE_DIR}/include/MEstimator.h
            ${PROJECT_SOURCE_DIR}/include/ProjectionModel.h
            ${PROJECT_SOURCE_DIR}/include/PinholeModel.h
            ${PROJECT_SOURCE_DIR}/src/PinholeModel.cpp
            ${PROJECT_SOURCE_DIR}/include/SphericalModel.h
            ${PROJECT_SOURCE_DIR}/src/SphericalModel.cpp
            ${PROJECT_SOURCE_DIR}/include/Pyramid.h
            ${PROJECT_SOURCE_DIR}/src/Pyramid.cpp
            ${PROJECT_SOURCE_DIR}/include/transformPts3D.h
            ${PROJECT_SOURCE_DIR}/src/transformPts3D.cpp
            ${PROJECT_SOURCE_DIR}/include/Miscellaneous.h
)

TARGET_LINK_LIBRARIES(DR
                        ${MRPT_LIBS}
                        ${OpenCV_LIBS}
                        ${PCL_LIBRARIES}
                        ${Boost_SERIALIZATION_LIBRARY}
)

# Tell CMake that the linker language is C++
SET_TARGET_PROPERTIES(DR PROPERTIES LINKER_LANGUAGE CXX)


# Include the project 'OpenNI2_Grabber'
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenNI2_Grabber/FrameRGBD)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenNI2_Grabber/grabber)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenNI2_Grabber/third_party/OpenNI2/Include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenNI2_Grabber/third_party/CLAMS/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenNI2_Grabber/third_party/cvSerialization)

# Set the path to this project' sources
ADD_DEFINITIONS(-DPROJECT_SOURCE_PATH="${PROJECT_SOURCE_DIR}")
SET( NUM_ASUS_SENSORS 8 CACHE INTERNAL "Number of ASUS sensors in the rig")
add_definitions(-D_NUM_ASUS_SENSORS=${NUM_ASUS_SENSORS})
SET( USE_INTRINSIC_CALIB ON CACHE BOOL "Use the intrinsic calibration of the sensor to denoise depth data")
add_definitions(-D_USE_INTRINSIC_CALIB=${USE_INTRINSIC_CALIB})

#OpenNI2
LINK_DIRECTORIES(${CMAKE_BINARY_DIR})
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/OpenNI2/Drivers)

# Select the "apps" to be built
SET( BUILD_CALIBRATION OFF CACHE BOOL "Build Calibration tools")
ADD_SUBDIRECTORY(Calibration)

SET( BUILD_GRABBER ON CACHE BOOL "Build Grabber tools")
ADD_SUBDIRECTORY(Grabber)

SET( BUILD_VISUALIZATION ON CACHE BOOL "Build Visualization tools")
ADD_SUBDIRECTORY(Visualization)

SET( BUILD_LABELIZATION ON CACHE BOOL "Build Labelization tools")
ADD_SUBDIRECTORY(Labelization)

SET( BUILD_REGISTRATION ON CACHE BOOL "Build Registration tools")
ADD_SUBDIRECTORY(Registration)

SET( BUILD_SLAM ON CACHE BOOL "Build SLAM Applications")
ADD_SUBDIRECTORY(SLAM)

#option(BUILD_LC   "Build LoopClosure"     ON)
SET( BUILD_LC ON CACHE BOOL "Loop Closure tests")
ADD_SUBDIRECTORY(LoopClosure)

# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
