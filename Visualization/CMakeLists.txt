IF(BUILD_VISUALIZATION)

	# Declare the target (an executable)
	ADD_EXECUTABLE(LoadSphere LoadSphere.cpp)

	TARGET_LINK_LIBRARIES(LoadSphere
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
	)

	# Declare the target (an executable)
	ADD_EXECUTABLE(LoadFrame360 LoadFrame360.cpp
					../include/Calib360.h
					../include/Frame360.h
					../include/Frame360_Visualizer.h
					../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
					../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
	)

	TARGET_LINK_LIBRARIES(LoadFrame360
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					${MRPT_LIBS}
	)

	# Declare the target (an executable)
	ADD_EXECUTABLE(LoadSequence LoadSequence.cpp
					../include/Calib360.h
					../include/Frame360.h
					../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
					../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
	)
	TARGET_LINK_LIBRARIES(LoadSequence
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
	)

	# Declare the target (an executable)
	ADD_EXECUTABLE(LoadRawlog LoadRawlog.cpp
					../include/Calib360.h
					../include/Frame360.h
					../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
					../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
	)
	TARGET_LINK_LIBRARIES(LoadRawlog
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
	)

ENDIF(BUILD_VISUALIZATION)
