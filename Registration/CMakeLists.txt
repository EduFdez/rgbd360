IF(BUILD_REGISTRATION)

	INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

        ADD_EXECUTABLE(test test.cpp)

        ADD_EXECUTABLE(test_opencv test_opencv.cpp)
        #TARGET_LINK_LIBRARIES( ${OpenCV_LIBS} )

	# Declare the target (an executable)
	ADD_EXECUTABLE(RegisterPairRGBD360 RegisterPairRGBD360.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
                                                ../src/Frame360.cpp
                                                ../include/Map360_Visualizer.h
						../include/RegisterRGBD360.h
	)

	TARGET_LINK_LIBRARIES(RegisterPairRGBD360 DR
							${MRPT_LIBS}
							${OpenCV_LIBS}
							${PCL_LIBRARIES}
							${Boost_SERIALIZATION_LIBRARY}
		   		)

	# Declare the target (an executable)
	ADD_EXECUTABLE(Register360_indoor_outdoor Register360_indoor_outdoor.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
                                                ../src/Frame360.cpp
                                                ../include/Map360_Visualizer.h
						../include/RegisterRGBD360.h
	)

	TARGET_LINK_LIBRARIES(Register360_indoor_outdoor DR
							${MRPT_LIBS}
							${OpenCV_LIBS}
							${PCL_LIBRARIES}
							${Boost_SERIALIZATION_LIBRARY}
		   		)


	# Declare the target (an executable)
	ADD_EXECUTABLE(RegisterPairRGBD360_PbMap RegisterPairRGBD360_PbMap.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
                                                ../src/Frame360.cpp
                                                ../include/Map360_Visualizer.h
						../include/RegisterRGBD360.h
	)

	TARGET_LINK_LIBRARIES(RegisterPairRGBD360_PbMap DR
							${MRPT_LIBS}
							${OpenCV_LIBS}
							${PCL_LIBRARIES}
							${Boost_SERIALIZATION_LIBRARY}
			   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(RegisterPairRGBD360_stereo RegisterPairRGBD360_stereo.cpp
                                                ../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
                                                ../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
                                                ../src/Frame360.cpp
						../include/Map360_Visualizer.h
						../include/RegisterRGBD360.h
	)

        TARGET_LINK_LIBRARIES(RegisterPairRGBD360_stereo DR
							${MRPT_LIBS}
							${OpenCV_LIBS}
							${PCL_LIBRARIES}
							${Boost_SERIALIZATION_LIBRARY}
			   )

        # Declare the target (an executable)
        ADD_EXECUTABLE(OdometryRGBD_rawlog OdometryRGBD_rawlog.cpp
                                            ../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
                                            ../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
        )

        TARGET_LINK_LIBRARIES(OdometryRGBD_rawlog   DR
                                                    ${MRPT_LIBS}
                                                    ${OpenCV_LIBS}
                                                    ${PCL_LIBRARIES}
                                                    ${Boost_SERIALIZATION_LIBRARY}
                                )

        # Declare the target (an executable)
        ADD_EXECUTABLE(OdometryRGBD_rawlog2sphere OdometryRGBD_rawlog2sphere.cpp
                                            ../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
                                            ../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
        )

        TARGET_LINK_LIBRARIES(OdometryRGBD_rawlog2sphere   
							DR
							${MRPT_LIBS}
							${OpenCV_LIBS}
							${PCL_LIBRARIES}
							${Boost_SERIALIZATION_LIBRARY}
                                )

        # Declare the target (an executable)
        ADD_EXECUTABLE(OdometryRGBD360_stereo OdometryRGBD360_stereo.cpp
                                                ../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
                                                ../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
                                                ../include/Miscellaneous.h
                                                ../include/Calib360.h
                                                ../include/Frame360.h
                                                ../include/SequenceStereo360.h
                                                ../src/Frame360.cpp
                                                ../include/Map360_Visualizer.h
                                                ../include/RegisterRGBD360.h
        )

        TARGET_LINK_LIBRARIES(OdometryRGBD360_stereo 	DR
				                        ${MRPT_LIBS}
				                        ${OpenCV_LIBS}
				                        ${PCL_LIBRARIES}
				                        ${Boost_SERIALIZATION_LIBRARY}
                           )

#	# Declare the target (an executable)
#	ADD_EXECUTABLE(RegisterRGBD_dataset RegisterRGBD_dataset.cpp
#					../include/Miscellaneous.h
#					../third_party/cvShowManyImages.h
#	)
#
#	TARGET_LINK_LIBRARIES(RegisterRGBD_dataset DR
#					${MRPT_LIBS}
#					${OpenCV_LIBS}
#					${PCL_LIBRARIES}
#					${Boost_SERIALIZATION_LIBRARY}
#					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(MethodsRegisterRGBD360 MethodsRegisterRGBD360.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
                                                ../src/Frame360.cpp
						../include/Map360_Visualizer.h
						../include/RegisterRGBD360.h
	)

	TARGET_LINK_LIBRARIES(MethodsRegisterRGBD360 	DR
							${MRPT_LIBS}
							${OpenCV_LIBS}
							${PCL_LIBRARIES}
							${Boost_SERIALIZATION_LIBRARY}
			   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(OdometryRGBD360 OdometryRGBD360.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
                                                ../src/Frame360.cpp
						../include/Map360.h
						../include/RegisterRGBD360.h
	)

	TARGET_LINK_LIBRARIES(OdometryRGBD360	DR
						${MRPT_LIBS}
						${OpenCV_LIBS}
						${PCL_LIBRARIES}
						${Boost_SERIALIZATION_LIBRARY}
			   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(OdometryKeyFrame360 OdometryKeyFrame360.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Frame360.h
                                                ../src/Frame360.cpp
						../include/Map360.h
						../include/RegisterRGBD360.h
	)

	TARGET_LINK_LIBRARIES(OdometryKeyFrame360	DR
							${MRPT_LIBS}
							${OpenCV_LIBS}
							${PCL_LIBRARIES}
							${Boost_SERIALIZATION_LIBRARY}
			   )



#	# Declare the target (an executable)
#	ADD_EXECUTABLE(RegisterSequenceLabel RegisterSequenceSphere_labelFast.cpp
#						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
#						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
#						../include/Miscellaneous.h
#						../include/Calib360.h
#						../include/Frame360.h
#	)
#
#	TARGET_LINK_LIBRARIES(RegisterSequenceLabel
#					${MRPT_LIBS}
#					${OpenCV_LIBS}
#					${PCL_LIBRARIES}
#					${Boost_SERIALIZATION_LIBRARY}
#					   )
#
#	# Declare the target (an executable)
#	ADD_EXECUTABLE(OnlineOdometryRGBD360 OnlineOdometryRGBD360.cpp
#						../include/CloudGrabber.h
#						../include/Miscellaneous.h
#						../include/Calib360.h
#						../include/Frame360.h
                                                #../src/Frame360.cpp
#						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
#						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
#	)
#
#	TARGET_LINK_LIBRARIES(OnlineOdometryRGBD360
#						${MRPT_LIBS}
#						${OpenCV_LIBS}
#						${PCL_LIBRARIES}
#						${Boost_SERIALIZATION_LIBRARY}
#					   )

ENDIF(BUILD_REGISTRATION)
