IF(BUILD_LABELIZATION)

	# Declare the target (an executable)
	ADD_EXECUTABLE(LabelizeFrame360 LabelizeFrame360.cpp
					../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
					../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
					../include/Miscellaneous.h
					../include/Calib360.h
					../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(LabelizeFrame360
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

	ADD_EXECUTABLE(LabelizeSequence LabelizeSequence.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(LabelizeSequence
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

ENDIF(BUILD_LABELIZATION)
