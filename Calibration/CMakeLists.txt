IF(BUILD_CALIBRATION)

	# Declare the target (an executable)
        ADD_EXECUTABLE(Calibrator   Calibrator.cpp
                                    ../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
                                    ../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
                                    ../include/Miscellaneous.h
                                    ../include/Calib360.h
                                    ../include/Calibrator.h
                                    ../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(Calibrator
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

        # Declare the target (an executable)
        ADD_EXECUTABLE(RawlogCalibrator RawlogCalibrator.cpp
                                        ../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
                                        ../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
                                        ../include/Miscellaneous.h
                                        ../include/Calib360.h
                                        ../include/Calibrator.h
                                        ../include/Frame360.h
                                        ../include/Frame360.cpp
        )

        TARGET_LINK_LIBRARIES(RawlogCalibrator
                                        ${MRPT_LIBS}
                                        ${OpenCV_LIBS}
                                        ${PCL_LIBRARIES}
                                        ${Boost_SERIALIZATION_LIBRARY}
                                           )

	# Declare the target (an executable)
	ADD_EXECUTABLE(OnlineCalibration OnlineCalibration.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
#	 					../include/CloudGrabber.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Calibrator.h
						../include/Frame360.h
        )

	TARGET_LINK_LIBRARIES(OnlineCalibration
						${MRPT_LIBS}
						${OpenCV_LIBS}
						${PCL_LIBRARIES}
						${Boost_SERIALIZATION_LIBRARY}
						libOpenNI2.so
						libOniFile.so
						libPS1080.so
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(EvalCalibration EvalCalibration.cpp
							../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
							../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
							../include/Miscellaneous.h
							../include/Calib360.h
							../include/Calibrator.h
							../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(EvalCalibration
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(VisualizeCalibration VisualizeCalibration.cpp
							../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
							../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
							../include/Miscellaneous.h
							../include/Calib360.h
							../include/Calibrator.h
							../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(VisualizeCalibration
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(GetControlPlanes GetControlPlanes.cpp
							../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
							../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
							../include/Miscellaneous.h
							../include/Calib360.h
							../include/Calibrator.h
							../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(GetControlPlanes
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(PairCalibrator PairCalibrator.cpp
							../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
							../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
							../include/Miscellaneous.h
							../include/Calib360.h
							../include/Calibrator.h
							../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(PairCalibrator
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(EvalPairCalibration EvalPairCalibration.cpp
							../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
							../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
							../include/Miscellaneous.h
							../include/Calib360.h
							../include/Calibrator.h
							../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(EvalPairCalibration
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(OnlinePairCalibrator OnlinePairCalibrator.cpp
						../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
						../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
#	 					../include/CloudGrabber.h
						../include/Miscellaneous.h
						../include/Calib360.h
						../include/Calibrator.h
						../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(OnlinePairCalibrator
						${MRPT_LIBS}
						${OpenCV_LIBS}
						${PCL_LIBRARIES}
						${Boost_SERIALIZATION_LIBRARY}
						libOpenNI2.so
						libOniFile.so
						libPS1080.so
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(ToFCalibrator ToFCalibrator.cpp
							#../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
							#../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
							../include/Miscellaneous.h
							../include/Calib360.h
							../include/Calibrator.h
							../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(ToFCalibrator
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					   )

	# Declare the target (an executable)
	ADD_EXECUTABLE(LaserCalibrator LaserCalibrator.cpp
							../OpenNI2_Grabber/third_party/CLAMS/discrete_depth_distortion_model.cpp
							../OpenNI2_Grabber/third_party/CLAMS/include/clams/discrete_depth_distortion_model.h
							../include/Miscellaneous.h
							../include/Calib360.h
							../include/Calibrator.h
							../include/Frame360.h
	)

	TARGET_LINK_LIBRARIES(LaserCalibrator
					${MRPT_LIBS}
					${OpenCV_LIBS}
					${PCL_LIBRARIES}
					${Boost_SERIALIZATION_LIBRARY}
					libOpenNI2.so
					libOniFile.so
					libPS1080.so
					   )

ENDIF(BUILD_CALIBRATION)
