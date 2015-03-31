/*
 *  Copyright (c) 2012, Universidad de Málaga - Grupo MAPIR
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the holder(s) nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *  Author: Eduardo Fernandez-Moral
 */

#ifndef REGISTER_RGBD360_H
#define REGISTER_RGBD360_H

#include "Frame360.h"
#include "RegisterDense.h"

#ifndef _DEBUG_MSG
    #define _DEBUG_MSG 1
#endif

#define DOF 6 // Degrees of freedom for the registration


/*! This class implements the functionality to register the PbMaps from two spherical RGB-D frames
 */
class RegisterRGBD360
{
private:

    //  /*! Maximum curvature of a patch to be considered by the matching algorithm*/
    //  float max_curvature_plane;

    /*! Reference frame for registration */
    Frame360 *pRef360;

    /*! Target frame for registration */
    Frame360 *pTrg360;

    /*! Subgraph of 'matchable' planes from the reference frame */
    mrpt::pbmap::Subgraph refGraph;

    /*! Subgraph of 'matchable' planes from the target frame */
    mrpt::pbmap::Subgraph trgGraph;

    /*! SubgraphMatcher object */
    mrpt::pbmap::SubgraphMatcher matcher;

    /*! Pose of pTrg360 as seen from pRef360 */
    Eigen::Matrix4f rigidTransf;

    /*! 6x6 covariance matrix of the pose of pTrg360 as seen from pRef360 (the inverse of information matrix) */
    Eigen::Matrix<float,6,6> covarianceM;

    /*! 6x6 information matrix of the pose of pTrg360 as seen from pRef360 (the inverse of covariance matrix) */
    //  Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> informationM;
    Eigen::Matrix<float,6,6> informationM;

    /*! The matched planar patches */
    std::map<unsigned, unsigned> bestMatch;

    /*! The sum of areas of the matched planar patches in the reference frame */
    float areaMatched;

    /*! has the registration already been done? */
    bool bRegistrationDone;

public:

    /*! The sum of areas of the registration source planes */
    float areaSource;

    /*! The sum of areas of the registration target planes */
    float areaTarget;

    /*! Constructor */
    RegisterRGBD360(const std::string &configFile) :
        bRegistrationDone(false)
    {
        matcher.configLocaliser.load_params(configFile);
        rigidTransf = Eigen::Matrix4f::Identity();
#if _DEBUG_MSG
        matcher.configLocaliser.print_params();
#endif
    }

    /*! Set the reference frame for registration. If the parameter 'max_match_planes' is set, only the number
      'max_match_planes' planes with the largest area are selected for registration. If this parameter is 0 (default)
      all the planes are used.
  */
    void setReference(Frame360 *ref, const size_t max_match_planes = 0)
    {
        pRef360 = ref;
        refGraph.pPBM = &pRef360->planes;
        refGraph.subgraphPlanesIdx.clear();
        //  std::cout << "Size planes1 " << pRef360->planes.vPlanes.size() << " max_match_planes " << max_match_planes << std::endl;
        if(max_match_planes > 0 && pRef360->planes.vPlanes.size() > max_match_planes) // Select the largest 'max_match_planes' planes
        {
            //      // Add all the planes that have been labelized
            //      for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++)
            //       if(pRef360->planes.vPlanes[i].curvature < max_curvature_plane && pRef360->planes.vPlanes[i].label != "")
            //        refGraph.subgraphPlanesIdx.push_back(pRef360->planes.vPlanes[i].id);

            std::vector<float> planeAreas(pRef360->planes.vPlanes.size(), 0);
            for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++)
                if(pRef360->planes.vPlanes[i].curvature < max_curvature_plane)
                {
                    if(pRef360->planes.vPlanes[i].label != "") // If the plane has a label, make sure it is used in registration
                        planeAreas[i] = 10; // We tweak this parameter to make sure that the plane is used in registration
                    else
                        planeAreas[i] = pRef360->planes.vPlanes[i].areaHull;
                }

            std::vector<float> sortedAreas = planeAreas;
            std::sort(sortedAreas.begin(),sortedAreas.end());
            float areaThreshold = sortedAreas[pRef360->planes.vPlanes.size() - max_match_planes - 1];
            //      for(unsigned i=0; i < planeAreas.size(); i++)
            //        std::cout << "Area " << planeAreas[i] << std::endl;
            //      std::cout << "areaThreshold " << areaThreshold << std::endl;

            for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++)
                //            std::cout << i << " " << planeAreas[i] << std::endl;
                if(planeAreas[i] > areaThreshold)
                    refGraph.subgraphPlanesIdx.push_back(pRef360->planes.vPlanes[i].id);
        }
        else
        {
            for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++){//std::cout << "id " << pRef360->planes.vPlanes[i].id << endl;
                if(pRef360->planes.vPlanes[i].curvature < max_curvature_plane)
                    refGraph.subgraphPlanesIdx.push_back(pRef360->planes.vPlanes[i].id);}
        }
        //    std::cout << "Subgraph planes in Ref " << refGraph.subgraphPlanesIdx.size() << max_match_planes << endl;

        bRegistrationDone = false; // Set to false as a new reference frame is set
    }

    /*! Set the target frame for registration. If the parameter 'max_match_planes' is set, only the number
      'max_match_planes' planes with the largest area are selected for registration. If this parameter is 0 (default)
      all the planes are used.
  */
    void setTarget(Frame360 *trg, const size_t max_match_planes = 0)
    {
        pTrg360 = trg;
        trgGraph.pPBM = &pTrg360->planes;
        trgGraph.subgraphPlanesIdx.clear();
        if(max_match_planes > 0 && pTrg360->planes.vPlanes.size() > max_match_planes)
        {
            std::vector<float> planeAreas(pTrg360->planes.vPlanes.size(), 0);
            for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
                if(pTrg360->planes.vPlanes[i].curvature < max_curvature_plane)
                {
                    if(pTrg360->planes.vPlanes[i].label != "") // If the plane has a label, make sure it is used in registration
                        planeAreas[i] = 10; // We tweak this parameter to make sure that the plane is used in registration
                    else
                        planeAreas[i] = pTrg360->planes.vPlanes[i].areaHull;
                }

            std::vector<float> sortedAreas = planeAreas;
            std::sort(sortedAreas.begin(),sortedAreas.end());
            float areaThreshold = sortedAreas[pTrg360->planes.vPlanes.size() - max_match_planes - 1];

            for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
                //            std::cout << i << " " << planeAreas[i] << std::endl;
                if(planeAreas[i] > areaThreshold)
                    trgGraph.subgraphPlanesIdx.push_back(pTrg360->planes.vPlanes[i].id);
        }
        else
        {
            for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
                if(pTrg360->planes.vPlanes[i].curvature < max_curvature_plane)
                    trgGraph.subgraphPlanesIdx.push_back(pTrg360->planes.vPlanes[i].id);
        }
        //    std::cout << "Subgraph planes in Trg " << trgGraph.subgraphPlanesIdx.size() << max_match_planes << endl;

        bRegistrationDone = false; // Set to false as a new target frame is set
    }

    /*! Return the pose of pTrg360 as seen from pRef360 */
    Eigen::Matrix4f getPose()
    {
        if(!bRegistrationDone)
            RegisterPbMap();

        return rigidTransf;
    }

    /*! Return the 6x6 covariance matrix of the pose of pTrg360 as seen from pRef360 */
    Eigen::Matrix<float,6,6> getCovMat()
    {
        if(!bRegistrationDone)
            RegisterPbMap();

        covarianceM = informationM.inverse();

        return covarianceM;
    }

    /*! Return the 6x6 information matrix of the pose of pTrg360 as seen from pRef360 */
    Eigen::Matrix<float,6,6> & getInfoMat()
    {
        if(!bRegistrationDone)
            RegisterPbMap();

        return informationM;
    }

    /*! Calculate entropy of the planar matching. This is the differential entropy of a multivariate normal distribution given
      by the matched planes. This is calculated with the formula used in the paper ["Dense Visual SLAM for RGB-D Cameras",
      by C. Kerl et al., in IROS 2013]. */
    float calcEntropy()
    {
        if(!bRegistrationDone)
            RegisterPbMap();

        covarianceM = informationM.inverse();
        float entropy = 0.5 * ( DOF * (1 + log(2*PI)) + log(covarianceM.determinant()) );

        return entropy;
    }

    /*! Get matched planes */
    std::map<unsigned, unsigned> getMatchedPlanes()
    {
        if(!bRegistrationDone)
            RegisterPbMap();

        return bestMatch;
    }

    /*! Return the total area of the matched planes measured in the source frame */
    float getAreaMatched()
    {
        if(!bRegistrationDone)
            RegisterPbMap();

        return areaMatched;
    }

    /*! Different registration modes for pairs of PbMaps from spherical RGB-D observations */
    enum registrationType
    {
        DEFAULT_6DoF,
        PLANAR_3DoF,
        ODOMETRY_6DoF,
        PLANAR_ODOMETRY_3DoF,
    };

    /*! PbMap registration. Set the target frame and source frames for registration if specified. If the parameter 'max_match_planes' is set,
      only the number 'max_match_planes' planes with the largest area are selected for registration. If this parameter is 0 (default)
      all the planes are used.
      The parameter 'registMode' is used constraint the plane matching as
        0(DEFAULT_6DoF): unconstrained movement between frames,
        1(PLANAR_3DoF): planar movement (the camera is fixed in height)
        2(ODOMETRY_6DoF): odometry (small displacement between the two frames/places),
        3(PLANAR_ODOMETRY_3DoF): odometry + planar movement (small displacementes + the camera is fixed in height) */
    bool RegisterPbMap(Frame360 *frame1 = NULL, Frame360 *frame2 = NULL, const size_t max_match_planes = 0, registrationType registMode = DEFAULT_6DoF)
    {
        std::cout << "RegisterPbMap..." << _DEBUG_MSG << "\n";
//        double time_start = pcl::getTime();

        if(frame1) // Create the subgraphs corresponding to input frames for plane matching
            setReference(frame1, max_match_planes);

        if(frame2) // Create the subgraphs corresponding to input frames for plane matching
            setTarget(frame2, max_match_planes);

#if _DEBUG_MSG
        double time_start = pcl::getTime();
        std::cout << "set source and target subgraphs \n";
        std::cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << " limit " << max_match_planes << endl;
#endif

        bRegistrationDone = true;
        //    bestMatch = matcher.compareSubgraphs(refGraph, trgGraph);
        bestMatch = matcher.compareSubgraphs(refGraph, trgGraph, registMode);
        areaMatched = matcher.calcAreaMatched(bestMatch);

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "compareSubgraphs took " << double (time_end - time_start)*1000 << " ms\n";

        std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << " areaMatched " << areaMatched << " score " << matcher.score_best_match << std::endl;
        //    for(std::map<unsigned, unsigned>::iterator it=bestMatch.begin(); it != bestMatch.end(); it++)
        //      std::cout << it->first << " " << it->second << std::endl;
#endif

        if(bestMatch.size() < 3)
        {
            std::cout << "\tInsuficient matching\n\n";
            return false;
        }

        //    time_start = pcl::getTime();

        // Align the two frames
//        Eigen::Matrix4f rigidTransf_float;
//        Eigen::Matrix<float,6,6> infMat;
        mrpt::pbmap::ConsistencyTest fitModel(pRef360->planes, pTrg360->planes);
        //    rigidTransf = fitModel.initPose(bestMatch);
        //    Eigen::Matrix4f rigidTransf = fitModel.estimatePose(bestMatch);
        bool goodAlignment = fitModel.estimatePoseWithCovariance(bestMatch, rigidTransf, informationM);
        //    Eigen::Matrix4f rigidTransf2 = fitModel.estimatePoseRANSAC(bestMatch);
//        rigidTransf = rigidTransf_float.cast<double>();
//        informationM = infMat_float.cast<double>();

        if(goodAlignment) // Get the maximum matchable areas
        {
            areaSource = 0;
            for(std::vector<unsigned>::iterator it=refGraph.subgraphPlanesIdx.begin(); it != refGraph.subgraphPlanesIdx.end(); it++)
                areaSource += pRef360->planes.vPlanes[*it].areaHull;

            areaTarget = 0;
            for(std::vector<unsigned>::iterator it=trgGraph.subgraphPlanesIdx.begin(); it != trgGraph.subgraphPlanesIdx.end(); it++)
                areaTarget += pTrg360->planes.vPlanes[*it].areaHull;
        }


        return goodAlignment;
    }

//    /*! Dense (direct) registration. It performs registration for each separated sensor, and consequently is designed for small rotations/transformations
//     * Set the target frame and source frames for registration if specified.
//      The parameter 'registMode' is used constraint the plane matching as
//        0(DEFAULT_6DoF): unconstrained movement between frames,
//        1(PLANAR_3DoF): planar movement (the camera is fixed in height) */
//    bool DenseRegistration(Frame360 *frame1, Frame360 *frame2,
//                           Eigen::Matrix4f pose_estim = Eigen::Matrix4f::Identity(),
//                           RegisterDense::costFuncType method = RegisterDense::PHOTO_CONSISTENCY,
//                           registrationType registMode = DEFAULT_6DoF)
//    {
//        assert(frame1 && frame2);
//        //    std::cout << "RegisterDense...\n";
////        double time_start = pcl::getTime();

//        // Steps:
//        // 1) Compute the image pyramids
//        // 2) Get Hessian and Gradient
//        // 3) Iterate

//        Eigen::Matrix4f pose_estim_temp;

//        const float img_width = frame1->frameRGBD_[0].getRGBImage().cols;
//        const float img_height = frame1->frameRGBD_[0].getRGBImage().rows;
//        const float res_factor_VGA = img_width / 640.0;
//        const float focal_length = 525 * res_factor_VGA;
//        const float inv_fx = 1.f/focal_length;
//        const float inv_fy = 1.f/focal_length;
//        const float ox = img_width/2 - 0.5;
//        const float oy = img_height/2 - 0.5;
//        Eigen::Matrix3f camIntrinsicMat; camIntrinsicMat << focal_length, 0, ox, 0, focal_length, oy, 0, 0, 1;

//        std::vector<RegisterDense> alignSensorID(NUM_ASUS_SENSORS);

//        //    #pragma omp parallel num_threads(8)
//        for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
//        {
//            //      int sensor_id = omp_get_thread_num();
//            alignSensorID[sensor_id].setCameraMatrix(camIntrinsicMat);
//            alignSensorID[sensor_id].setSourceFrame(frame2->frameRGBD_[sensor_id].getRGBImage(), frame2->frameRGBD_[sensor_id].getDepthImage());
//            alignSensorID[sensor_id].setTargetFrame(frame1->frameRGBD_[sensor_id].getRGBImage(), frame1->frameRGBD_[sensor_id].getDepthImage());
//        }
////std::cout << "Set the frames." << std::endl;

//        Eigen::Matrix<float,6,6> Hessian;
//        Eigen::Matrix<float,6,1> Gradient;

//        for(int pyramidLevel = alignSensorID[0].nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
//        {
//            int nRows = alignSensorID[0].graySrcPyr[pyramidLevel].rows;
//            int nCols = alignSensorID[0].graySrcPyr[pyramidLevel].cols;

//            double lambda = 0.001; // Levenberg-Marquardt (LM) lambda
//            double step = 10; // Update step
//            unsigned LM_maxIters = 1;

//            int it = 0, maxIters = 10;
//            double tol_residual = pow(10,-1);
//            double tol_update = pow(10,-6);
//            Eigen::Matrix<float,6,1> update_pose; update_pose << 1, 1, 1, 1, 1, 1;
//            double error = 0.0;
//            std::vector<double> error2_SensorID(NUM_ASUS_SENSORS);

//            omp_set_num_threads(8);
////            #pragma omp parallel num_threads(8)
//            #pragma omp parallel for reduction (+:error)
//            for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
//            {
//                error += alignSensorID[sensor_id].calcDenseError_robot(pyramidLevel, pose_estim, frame1->calib->Rt_[sensor_id], method);
////                int sensor_id = omp_get_thread_num();
////                error2_SensorID[sensor_id] = alignSensorID[sensor_id].calcDenseError_robot(pyramidLevel, pose_estim, frame1->calib->Rt_[sensor_id], method);
////            std::cout << "error2_SensorID[sensor_id] \n" << error2_SensorID[sensor_id] << std::endl;
//            }
////            for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
////                error += error2_SensorID[sensor_id];

//            double diff_error = error;

//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//            std::cout << "Level_ " << pyramidLevel << " error " << error << std::endl;
//#endif

//#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
//cv::TickMeter tm;tm.start();
//#endif
//            while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
//            {
//                Hessian = Eigen::Matrix<float,6,6>::Zero();
//                Gradient = Eigen::Matrix<float,6,1>::Zero();

//                #pragma omp parallel num_threads(8)
////                for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
//                {
//                    int sensor_id = omp_get_thread_num();
//                    alignSensorID[sensor_id].calcHessianGradient_robot(pyramidLevel, pose_estim, frame1->calib->Rt_[sensor_id], method);
////                    std::cout << "hessian \n" << alignSensorID[sensor_id].getHessian() << std::endl;
//                }
//                for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
//                {
////                    std::cout << "hessian \n" << alignSensorID[sensor_id].getHessian() << std::endl;
//                    Hessian += alignSensorID[sensor_id].getHessian();
//                    Gradient += alignSensorID[sensor_id].getGradient();
//                }

////                assert(Hessian.rank() == 6); // Make sure that the problem is observable
//                if((Hessian + lambda*getDiagonalMatrix(Hessian)).rank() != 6)
//                {
//                    std::cout << "\t The problem is ILL-POSED \n";
////                    std::cout << "Hessian \n" << Hessian << std::endl;
////                    std::cout << "Gradient \n" << Gradient << std::endl;
//                    rigidTransf = pose_estim;
//                    return false;
//                }

//                // Compute the pose update
//                update_pose = -(Hessian + lambda*getDiagonalMatrix(Hessian)).inverse() * Gradient;
//                Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
//                pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
////            cout << "update_pose \n" << update_pose.transpose() << endl;
////            cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;

//                double new_error = 0.0;
//                #pragma omp parallel for reduction (+:new_error)
//                for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
//                    new_error += alignSensorID[sensor_id].calcDenseError_robot(pyramidLevel, pose_estim, frame1->calib->Rt_[sensor_id], method);

//                diff_error = error - new_error;
////            cout << "diff_error \n" << diff_error << endl;
//                if(diff_error > 0)
//                {
//                    lambda /= step;
//                    pose_estim = pose_estim_temp;
//                    error = new_error;
//                    it = it+1;
//                }
//                else
//                {
//                    unsigned LM_it = 0;
//                    while(LM_it < LM_maxIters && diff_error < 0)
//                    {
//                        lambda = lambda * step;

//                        update_pose = -(Hessian + lambda*getDiagonalMatrix(Hessian)).inverse() * Gradient;
//                        Eigen::Matrix<double,6,1> update_pose_d = update_pose.cast<double>();
//                        pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose_d)).getHomogeneousMatrixVal().cast<float>() * pose_estim;
////                    cout << "update_pose LM \n" << update_pose.transpose() << endl;

//                        new_error = 0.0;
//                        #pragma omp parallel for reduction (+:new_error)
//                        for(unsigned sensor_id=0; sensor_id < 8; ++sensor_id)
//                            new_error += alignSensorID[sensor_id].calcDenseError_robot(pyramidLevel, pose_estim, frame1->calib->Rt_[sensor_id], method);
//                        diff_error = error - new_error;

////                        cout << "diff_error LM \n" << diff_error << endl;
//                        if(diff_error > 0)
//                        {
//                            pose_estim = pose_estim_temp;
//                            error = new_error;
//                            it = it+1;
//                        }
//                        LM_it = LM_it + 1;
//                    }
//                }
//            }

////            tm.stop(); std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;

//        }

//        bRegistrationDone = true;

////        if(Hessian.det() > 1000)
//        {
//            std::cout << "Hessian.det() = " << Hessian.det() << std::endl;
//            rigidTransf = pose_estim;
//            informationM = Hessian;
//            return true;
//        }
////        else
////            return false;

////        return true;
//    }

    enum trackingQuality{GOOD=0, WEAK, BAD};

    /*! Evaluate the quality of the PbMap registration. The ratio of PbMap's matched area and the Euclidean distance are taken into account.
        TODO: the score should be done from the covariance of the registration (least eigenvalue). */
    int trackingScore(float &score)
    {
        // Euclidean distance
        float dist = getPose().block(0,3,3,1).norm();

        // Area 70 -30
        score = getAreaMatched() / areaSource;

        // Assign the tracking quality
        if(score >= 0.7) // GOOD registration
            return 0;
        if(score >= 0.3) // WEAK registration
            return 1;
        return 2;
    }

};
#endif
