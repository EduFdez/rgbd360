IF(BUILD_GRABBER)

	# Declare the target (an executable)
	ADD_EXECUTABLE(RGBD360_Grabber RGBD360_Grabber.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(RGBD360_Grabber
						${MRPT_LIBS}
						${OpenCV_LIBS}
						${PCL_LIBRARIES}
						${Boost_SERIALIZATION_LIBRARY}
						libOpenNI2.so
						libOniFile.so
						libPS1080.so
						#../build/libOpenNI2.so
						#../build/OpenNI2/Drivers/libOniFile.so
						#../build/OpenNI2/Drivers/libPS1080.so
					   )

ENDIF(BUILD_GRABBER)
