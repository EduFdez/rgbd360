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
    //  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> informationM;
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
            //        refGraph.subgraphPlanesIdx.insert(pRef360->planes.vPlanes[i].id);

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
                    refGraph.subgraphPlanesIdx.insert(pRef360->planes.vPlanes[i].id);
        }
        else
        {
            for(unsigned i=0; i < pRef360->planes.vPlanes.size(); i++){//std::cout << "id " << pRef360->planes.vPlanes[i].id << endl;
                if(pRef360->planes.vPlanes[i].curvature < max_curvature_plane)
                    refGraph.subgraphPlanesIdx.insert(pRef360->planes.vPlanes[i].id);}
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
                    trgGraph.subgraphPlanesIdx.insert(pTrg360->planes.vPlanes[i].id);
        }
        else
        {
            for(unsigned i=0; i < pTrg360->planes.vPlanes.size(); i++)
                if(pTrg360->planes.vPlanes[i].curvature < max_curvature_plane)
                    trgGraph.subgraphPlanesIdx.insert(pTrg360->planes.vPlanes[i].id);
        }
        //    std::cout << "Subgraph planes in Trg " << trgGraph.subgraphPlanesIdx.size() << max_match_planes << endl;

        bRegistrationDone = false; // Set to false as a new target frame is set
    }

    /*! Return the pose of pTrg360 as seen from pRef360 */
    Eigen::Matrix4f getPose()
    {
        if(!bRegistrationDone)
            Register();

        return rigidTransf;
    }

    /*! Return the 6x6 covariance matrix of the pose of pTrg360 as seen from pRef360 */
    Eigen::Matrix<float,6,6> getCovMat()
    {
        if(!bRegistrationDone)
            Register();

        covarianceM = informationM.inverse();

        return covarianceM;
    }

    /*! Return the 6x6 information matrix of the pose of pTrg360 as seen from pRef360 */
    Eigen::Matrix<float,6,6> getInfoMat()
    {
        if(!bRegistrationDone)
            Register();

        return informationM;
    }

    /*! Calculate entropy of the planar matching. This is the differential entropy of a multivariate normal distribution given
      by the matched planes. This is calculated with the formula used in the paper ["Dense Visual SLAM for RGB-D Cameras",
      by C. Kerl et al., in IROS 2013]. */
    float calcEntropy()
    {
        if(!bRegistrationDone)
            Register();

        covarianceM = informationM.inverse();
        float entropy = 0.5 * ( DOF * (1 + log(2*PI)) + log(covarianceM.determinant()) );

        return entropy;
    }

    /*! Get matched planes */
    std::map<unsigned, unsigned> getMatchedPlanes()
    {
        if(!bRegistrationDone)
            Register();

        return bestMatch;
    }

    /*! Return the total area of the matched planes measured in the source frame */
    float getAreaMatched()
    {
        if(!bRegistrationDone)
            Register();

        return areaMatched;
    }

    /*! Different registration modes for pairs of PbMaps from spherical RGB-D observations */
    enum registrationType
    {
        DEFAULT_6DoF,
        ODOMETRY_6DoF,
        PLANAR_3DoF,
        PLANAR_ODOMETRY_3DoF,
    };

    /*! Set the target frame and source frames for registration if specified. If the parameter 'max_match_planes' is set,
      only the number 'max_match_planes' planes with the largest area are selected for registration. If this parameter is 0 (default)
      all the planes are used.
      The parameter 'registMode' is used constraint the plane matching as
        0(DEFAULT_6DoF): unconstrained movement between frames,
        1(ODOMETRY_6DoF): odometry (small displacement between the two frames/places),
        2(PLANAR_3DoF): planar movement (the camera is fixed in height)
        3(PLANAR_ODOMETRY_3DoF): odometry + planar movement (small displacementes + the camera is fixed in height) */
    bool Register(Frame360 *frame1 = NULL, Frame360 *frame2 = NULL, const size_t max_match_planes = 0, registrationType registMode = DEFAULT_6DoF)
    {
        //    std::cout << "Register...\n";
//        double time_start = pcl::getTime();

        if(frame1) // Create the subgraphs corresponding to input frames for plane matching
            setReference(frame1, max_match_planes);

        if(frame2) // Create the subgraphs corresponding to input frames for plane matching
            setTarget(frame2, max_match_planes);

#if _DEBUG_MSG
        double time_start = pcl::getTime();
        std::cout << "Number of planes in Ref " << refGraph.subgraphPlanesIdx.size() << " Trg " << trgGraph.subgraphPlanesIdx.size() << " limit " << max_match_planes << endl;
#endif

        bRegistrationDone = true;
        //    bestMatch = matcher.compareSubgraphs(refGraph, trgGraph);
        bestMatch = matcher.compareSubgraphs(refGraph, trgGraph, registMode);
        areaMatched = matcher.calcAreaMatched(bestMatch);

#if _DEBUG_MSG
        double time_end = pcl::getTime();
        std::cout << "compareSubgraphs took " << double (time_end - time_start)*1000 << " ms\n";

        std::cout << "NUMBER OF MATCHED PLANES " << bestMatch.size() << " areaMatched " << areaMatched << std::endl;
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
        mrpt::pbmap::ConsistencyTest fitModel(pRef360->planes, pTrg360->planes);
        //    rigidTransf = fitModel.initPose(bestMatch);
        //    Eigen::Matrix4f rigidTransf = fitModel.estimatePose(bestMatch);
        bool goodAlignment = fitModel.estimatePoseWithCovariance(bestMatch, rigidTransf, informationM);
        //    Eigen::Matrix4f rigidTransf2 = fitModel.estimatePoseRANSAC(bestMatch);

        if(goodAlignment) // Get the maximum matchable areas
        {
            areaSource = 0;
            for(std::set<unsigned>::iterator it=refGraph.subgraphPlanesIdx.begin(); it != refGraph.subgraphPlanesIdx.end(); it++)
                areaSource += pRef360->planes.vPlanes[*it].areaHull;

            areaTarget = 0;
            for(std::set<unsigned>::iterator it=trgGraph.subgraphPlanesIdx.begin(); it != trgGraph.subgraphPlanesIdx.end(); it++)
                areaTarget += pTrg360->planes.vPlanes[*it].areaHull;
        }


        return goodAlignment;
    }

};
#endif
