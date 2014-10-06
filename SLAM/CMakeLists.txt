IF(BUILD_SLAM)

	# G2O library
	#find_package(G2O)
        #INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR})
	#LINK_DIRECTORIES(${G2O_LIBRARIES})

	# Set up the top-level include directories
	#SET(G2O_INCLUDE_DIR ${RGBD360_SOURCE_DIR}/../g2o/trunk CACHE PATH "Directory of G2O")
	SET(G2O_INCLUDE_DIR /home/edu/Libraries/g2o/trunk CACHE PATH "Directory of G2O")
	#SET(G2O_INCLUDE_DIR /home/edu/Libraries/g2o_ylatif/trunk CACHE PATH "Directory of G2O")
        INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR})
        LINK_DIRECTORIES(${G2O_INCLUDE_DIR}/lib)

	#Generate config.h 
	#configure_file(${RGBD360_SOURCE_DIR}/../g2o/trunk/config.h.in ${RGBD360_SOURCE_DIR}/../g2o/trunk/g2o/config.h)
	#INSTALL(FILES ${RGBD360_SOURCE_DIR}/../g2o/trunk/g2o/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/g2o)
	configure_file(/home/edu/Libraries/g2o/trunk/config.h.in /home/edu/Libraries/g2o/trunk/g2o/config.h)
	INSTALL(FILES /home/edu/Libraries/g2o/trunk/g2o/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/g2o)
	#configure_file(/home/edu/Libraries/g2o_ylatif/trunk/config.h.in /home/edu/Libraries/g2o_ylatif/trunk/g2o/config.h)
	#INSTALL(FILES /home/edu/Libraries/g2o_ylatif/trunk/g2o/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/g2o)

	# Include the subdirectories
	#ADD_SUBDIRECTORY(g2o/trunk)

	INCLUDE_DIRECTORIES(${CSPARSE_INCLUDE_DIR}) #You can use CPARSE or CHOLMOD
	INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDE_DIR}) 

	#set(G2O_DIR ${RGBD360_SOURCE_DIR}/../g2o/trunk)
	#include_directories(${G2O_DIR})
	#link_directories(${G2O_DIR}/lib)

	#SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_slam2d g2o_viewer g2o_ext_csparse g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse ${CHOLMOD_LIBRARIES})
	SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_slam2d g2o_viewer g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse ${CHOLMOD_LIBRARIES})

	#INCLUDE_DIRECTORIES(${RGBD360_SOURCE_DIR}/GraphOptimization)

        # Add rrr lib dir
        SET(RRR_INCLUDE /home/edu/Libraries/rrr CACHE PATH "Directory of RRR")
        INCLUDE_DIRECTORIES(${RRR_INCLUDE})

	# Declare the target (an executable)
	ADD_EXECUTABLE(SphereGraphSLAM SphereGraphSLAM.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
						../include/FilterPointCloud.h
						../include/Map360.h
						../include/TopologicalMap360.h
						../include/Map360_Visualizer.h
						../include/LoopClosure360.h
						../include/GraphOptimizer.h

						#../GraphOptimization/GraphOptimizer.h
						#../GraphOptimization/GraphOptimizer_G2O.h
						#../GraphOptimization/GraphOptimizer_G2O.cpp
						#../GraphOptimization/GraphOptimizer_MRPT.h
						#../GraphOptimization/GraphOptimizer_MRPT.cpp
	)

	TARGET_LINK_LIBRARIES(SphereGraphSLAM
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					${G2O_LIBS}
				   	boost_serialization
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(KFsphere_SLAM KFsphere_SLAM.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
						../include/FilterPointCloud.h
						../include/Map360.h
						../include/TopologicalMap360.h
						../include/Map360_Visualizer.h
						../include/Relocalizer360.h
						../include/LoopClosure360.h
						../include/GraphOptimizer.h
#					../third_party/GraphOptimizer/include/GraphOptimizer.h
#					../third_party/GraphOptimizer/include/GraphOptimizer_MRPT.h
#					../third_party/GraphOptimizer/src/GraphOptimizer_MRPT.cpp
	)

	TARGET_LINK_LIBRARIES(KFsphere_SLAM
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					${G2O_LIBS}
				   boost_serialization
					   )
ENDIF(BUILD_SLAM)
