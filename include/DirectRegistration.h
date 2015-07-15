/*
 *  Copyright (c) 2015,   INRIA Sophia Antipolis - LAGADIC Team
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
 * Author: Eduardo Fernandez-Moral
 */

#pragma once

#include "Miscellaneous.h"
#include "ProjectionModel.h"
#include "PinholeModel.h"
#include "SphericalModel.h"
#include "Pyramid.h"
#include "MEstimator.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Core>


/*! This class performs dense registration (also called direct registration) minimizing a cost function based on either intensity, depth or both.
 *  Refer to the last chapter of my thesis dissertation for more details:
 * "Contributions to metric-topological localization and mapping in mobile robotics" 2014. Eduardo Fernández-Moral.
 */
class DirectRegistration : public Pyramid, MEstimator //ProjectionModel
{
//public:

//    PinholeModel pinhole_model;
//    SphericalModel spherical_model;
    ProjectionModel *ProjModel;

    /*! The reference intensity image */
    cv::Mat graySrc;
    cv::Mat rgbSrc;

    /*! The target intensity image */
    cv::Mat grayTrg;

    /*! The reference depth image */
    cv::Mat depthSrc;

    /*! The target depth image */
    cv::Mat depthTrg;

    /*! The relative pose between source and target RGB-D images */
    Eigen::Matrix4f registered_pose_;

    /*! The Hessian matrix of the optimization problem. At the solution it represents the inverse of the covariance matrix of the relative pose */
    Eigen::Matrix<float,6,6> hessian;

    /*! The Gradient vector of the optimization problem. At the solution it should be zero */
    Eigen::Matrix<float,6,1> gradient;

    Eigen::Matrix<float,3,3> hessian_rot;
    Eigen::Matrix<float,3,1> gradient_rot;

    /*! Optimization parameters. */
    int max_iters_;
    float tol_update_;
    float tol_update_rot_;
    float tol_update_trans_;
    double tol_residual_;

    /*! Minimum depth difference to filter the outliers.*/
    float min_depth_Outliers;

    /*! Threshold depth difference to filter the outliers.*/
    float thresDepthOutliers;

    /*! Maximum depth difference to filter the outliers.*/
    float max_depth_Outliers;

    /*! Variance of intensity measurements. */
    float varPhoto;
    float stdDevPhoto;

    /*! Variance of intensity measurements. */
    float varDepth;
    float stdDevDepth;

    /*! Depth component gain. This variable is used to scale the depth values so that depth components are similar to intensity values.*/
    float depthComponentGain;

    /*! If set to true, only the pixels with high gradient in the gray image are used (for both photo and depth minimization) */
    bool use_salient_pixels_;

    /*! If set to true, only the pixels with high gradient in the gray image are used (for both photo and depth minimization) */
    bool compute_MAD_stdDev_;

    /*! Optimize using bilinear interpolation in the last pyramid.*/
    bool use_bilinear_;

    /*! A threshold to select salient pixels on the intensity and depth images.*/
    float thresSaliency;
    float thres_saliency_gray_;
    float thres_saliency_depth_;
    float _max_depth_grad;

//    /*! Sensed-Space-Overlap of the registered frames. This is the relation between the co-visible pixels and the total number of pixels in the image.*/
//    float SSO;

    /*! Enable the visualization of the optimization process (only for debug).*/
    bool visualize_;

    /*! Warped intensity image. It is used in the optimization and also for visualization purposes.*/
    cv::Mat warped_gray;
    cv::Mat warped_gray_gradX, warped_gray_gradY;

    /*! Warped intensity depth image. It is used in the optimization and also for visualization purposes.*/
    cv::Mat warped_depth;
    cv::Mat warped_depth_gradX, warped_depth_gradY;

    /*! Vector containing the indices of the pixels that move (forward-backward) or are occluded in the target image to the source image.*/
    cv::Mat mask_dynamic_occlusion;

//    /*! LUT of 3D points of the spherical image.*/
//    std::vector<Eigen::Vector3f> unit_sphere_;

    /*! LUT of 3D points of the source image.*/
    Eigen::MatrixXf LUT_xyz_source;
    Eigen::MatrixXf LUT_xyz_target;
    Eigen::MatrixXf xyz_src_transf;
    Eigen::MatrixXf xyz_trg_transf;

    std::vector<size_t> salient_pixels_;
    std::vector<size_t> salient_pixels_photo;
    std::vector<size_t> salient_pixels_depth;

    /*! Store a copy of the residuals and the weights to speed-up the registration. (Before they were computed twice: in the error function and the Jacobian)*/
    Eigen::MatrixXf warp_img_src;
    Eigen::VectorXi warp_pixels_src;
    Eigen::VectorXf residualsPhoto_src;
    Eigen::VectorXf residualsDepth_src;
    Eigen::MatrixXf residualsDepth3D_src;
    Eigen::MatrixXf jacobiansPhoto;
    Eigen::MatrixXf jacobiansDepth;
    Eigen::VectorXf stdDevError_inv_src;
    Eigen::VectorXf wEstimPhoto_src;
    Eigen::VectorXf wEstimDepth_src;

    /*! If set to true, only the pixels with high gradient in the gray image are used (for both photo and depth minimization) */    
    Eigen::VectorXi validPixels_src;
    //Eigen::VectorXi visible_pixels_src;
    Eigen::VectorXi validPixelsPhoto_src;
    Eigen::VectorXi validPixelsDepth_src;

    Eigen::MatrixXf warp_img_trg;
    Eigen::VectorXi warp_pixels_trg;
    Eigen::VectorXf residualsPhoto_trg;
    Eigen::VectorXf residualsDepth_trg;
    Eigen::VectorXf stdDevError_inv_trg;
    Eigen::VectorXf wEstimPhoto_trg;
    Eigen::VectorXf wEstimDepth_trg;

    Eigen::VectorXi validPixels_trg;
    Eigen::VectorXi validPixelsPhoto_trg;
    Eigen::VectorXi validPixelsDepth_trg;

    /*! Number of iterations in each pyramid level.*/
    std::vector<int> num_iters;

//    /*! The average residual per pixel of photo/depth consistency.*/
//    float avResidual;

//    /*! The average residual per pixel of photo consistency.*/
//    double avPhotoResidual;

//    /*! The average residual per pixel of depth consistency.*/
//    double avDepthResidual;

    /*! Number of pyramid levels.*/
    int nPyrLevels;

public:

    /*! Sensor */
    enum sensorType
    {
        RGBD360_INDOOR = 0,
        STEREO_OUTDOOR,
        KINECT // Same for Asus XPL
    } sensor_type;

    /*! Optimization method (cost function). The options are: 0=Photoconsistency, 1=Depth consistency (ICP like), 2= A combination of photo-depth consistency */
    enum costFuncType
    {
        PHOTO_DEPTH,
        PHOTO_CONSISTENCY,
        DEPTH_CONSISTENCY,
        DEPTH_ICP
    } method;

    /*! Sensed-Space-Overlap of the registered frames. This is the relation between the co-visible pixels and the total number of pixels in the image.*/
    float SSO;

    /*! Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numpyramidLevels' levels.*/
    std::vector<cv::Mat> graySrcPyr, grayTrgPyr, depthSrcPyr, depthTrgPyr;
    std::vector<cv::Mat> grayTrgGradXPyr, grayTrgGradYPyr, depthTrgGradXPyr, depthTrgGradYPyr;
    std::vector<cv::Mat> graySrcGradXPyr, graySrcGradYPyr, depthSrcGradXPyr, depthSrcGradYPyr;
    std::vector<cv::Mat> colorSrcPyr;

    DirectRegistration(sensorType sensor = KINECT);

    ~DirectRegistration();

    /*! Set the number of pyramid levels.*/
    inline void setSensorType(const sensorType sensor)
    {
        sensor_type = sensor;

        if(sensor_type == KINECT)
            ProjModel = new PinholeModel;
        else //RGBD360_INDOOR, STEREO_OUTDOOR
            ProjModel = new SphericalModel;
    };

    /*! Set the number of pyramid levels.*/
    inline void setNumPyr(const int Npyr)
    {
        assert(Npyr >= 0);
        nPyrLevels = Npyr;
    };

    /*! Set the bilinear interpolation for the last pyramid.*/
    inline void setBilinearInterp(const bool use_bilinear)
    {
        use_bilinear_ = use_bilinear;
    };

    /*! Set the variance of the intensity.*/
    void setGrayVariance(const float stdDev)
    {
        stdDevPhoto = stdDev;
    };

    /*! Set the variance of the depth.*/
    void setDepthVariance(const float stdDev)
    {
        stdDevDepth = stdDev;
    };

    /*! Set the saliency threshold of the intensity. */
    void setSaliencyThreshodIntensity(const float thres)
    {
        thres_saliency_gray_ = thres;
    };

    /*! Set the saliency threshold of the depth. */
    void setSaliencyThreshodDepth(const float thres)
    {
        thres_saliency_depth_ = thres;
    };

    /*! Set optimization tolerance parameter. */
    void setMaxIterations(const int max_iters)
    {
        max_iters_ = max_iters;
    };

    /*! Set optimization tolerance parameter. */
    void setToleranceUpdate(const float tol_update)
    {
        tol_update_ = tol_update;
    };

    /*! Set optimization tolerance parameter. */
    void setToleranceUpdateRot(const float tol_update_rot)
    {
        tol_update_rot_ = tol_update_rot;
    };

    /*! Set optimization tolerance parameter. */
    void setToleranceUpdateTrans(const float tol_update_trans)
    {
        tol_update_trans_ = tol_update_trans;
    };

    /*! Set optimization tolerance parameter. */
    void setToleranceRes(const double tol_residual)
    {
        tol_residual_ = tol_residual;
    };

    /*! Set the visualization of the optimization progress.*/
    inline void setVisualization(const bool viz)
    {
        visualize_ = viz;
    };

    /*! Set the a variable to indicate whether pixel saliency is used.*/
    void useSaliency(const bool use_salient_pixels__)
    {
        use_salient_pixels_ = use_salient_pixels__;
    };

    /*! Set the minimum depth distance (m) to consider a certain pixel valid.*/
    inline void setMinDepth(const float minD)
    {
        ProjModel->setMinDepth(minD);
    };

    /*! Set the maximum depth distance (m) to consider a certain pixel valid.*/
    inline void setMaxDepth(const float maxD)
    {
        ProjModel->setMaxDepth(maxD);
    };

    /*! Returns the optimal SE(3) rigid transformation matrix between the source and target frame.
     * This method has to be called after calling the regist() method.*/
    inline Eigen::Matrix4f getOptimalPose()
    {
        return registered_pose_;
    }

    /*! Returns the Hessian (the information matrix).*/
    inline Eigen::Matrix<float,6,6> getHessian()
    {
        return hessian;
    }

    /*! Returns the Gradient (the information matrix).*/
    inline Eigen::Matrix<float,6,1> getGradient()
    {
        return gradient;
    }

    /*! Sets the source (Intensity+Depth) frame.*/
    void setSourceFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth);

    /*! Sets the source (Intensity+Depth) frame. Depth image is ignored*/
    void setTargetFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth);

    /*! Build the pyramid levels from the intensity images.*/
    inline void buildGrayPyramids()
    {
        buildPyramid(graySrc, graySrcPyr, nPyrLevels);
        buildPyramid(grayTrg, graySrcPyr, nPyrLevels);
    };

    /*! Swap the source and target images */
    void swapSourceTarget();

    /*! Get a list of salient points from a list of Jacobians corresponding to a set of 3D points */
    void getSalientPts(const Eigen::MatrixXf & jacobians, std::vector<size_t> & salient_pixels, const float r_salient = 0.05f );

    void trimValidPoints(Eigen::MatrixXf & LUT_xyz, Eigen::VectorXi & validPixels, Eigen::MatrixXf & xyz_transf,
                         Eigen::VectorXi & validPixelsPhoto, Eigen::VectorXi & validPixelsDepth,
                         costFuncType method,
                         std::vector<size_t> &salient_pixels, std::vector<size_t> &salient_pixels_photo, std::vector<size_t> &salient_pixels_depth);

    /*! Re-project the source image onto the target one according to the input relative pose 'poseGuess' to compute the error.
     *  If the parameter 'direction' is -1, then the reprojection is computed from the target to the source images. */
    double computeReprojError_perspective( const int pyrLevel, const Eigen::Matrix4f & poseGuess, const costFuncType method = PHOTO_CONSISTENCY, const int direction = 1);//, const bool use_bilinear = false);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double computeError( const int pyrLevel, const Eigen::Matrix4f & poseGuess, const costFuncType method = PHOTO_CONSISTENCY);//, const bool use_bilinear = false);
    double computeError2( const int pyrLevel, const Eigen::Matrix4f & poseGuess, const costFuncType method = PHOTO_CONSISTENCY);//, const bool use_bilinear = false);

    double computeError_IC( const int pyrLevel, const Eigen::Matrix4f & poseGuess, const costFuncType method = PHOTO_CONSISTENCY);//, const bool use_bilinear = false);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double computeError_inv( const int pyrLevel, const Eigen::Matrix4f & poseGuess, const costFuncType method = PHOTO_CONSISTENCY);//, const bool use_bilinear = false);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad( const int pyrLevel,
                       const costFuncType method = PHOTO_CONSISTENCY );

    double calcHessGrad_IC(const int pyrLevel,
                           const Eigen::Matrix4f & poseGuess,
                           const costFuncType method = PHOTO_CONSISTENCY );

//    /* Compute the Jacobian matrices which are used to select the salient pixels. */
//    void computeJacobian(int pyrLevel,
//                        const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
//                        const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    /*! Compute the averaged squared error of the salient points. */
    double computeErrorHessGrad_salient(std::vector<size_t> & salient_pixels, costFuncType method );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad_inv( const int pyrLevel,
                           const Eigen::Matrix4f & poseGuess,
                           const costFuncType method = PHOTO_CONSISTENCY );

    /*! Re-project the source image onto the target one according to the input relative pose 'poseGuess' to compute the error.
     *  If the parameter 'direction' is -1, then the reprojection is computed from the target to the source images. */
    double computeReprojError_spherical(int pyrLevel,
                                        const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                        const costFuncType method = PHOTO_CONSISTENCY,
                                        const int direction = 1 ) ;//,const bool use_bilinear = false );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
    double computeError_sphere (  const int pyrLevel,
                                const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    double computeErrorWarp_sphere (  int pyrLevel,
                                    const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                    const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    /*! This function do the same as 'computeError_sphere'. But applying inverse compositional, which only affects the depth error */
    double computeErrorIC_sphere( const int pyrLevel,
                                const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    double computeErrorInv_sphere(const int pyrLevel,
                                const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                const costFuncType method = PHOTO_CONSISTENCY); //,const bool use_bilinear = false );

    double computeError_sphere_bidirectional (const int pyrLevel,
                                            const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                            const costFuncType method = PHOTO_CONSISTENCY); //,const bool use_bilinear = false );

    double computeJacobian_sphere ( const int pyrLevel,
                                    const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                    const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad_sphere (  const int pyrLevel,
                                const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    void calcHessGrad_warp_sphere ( const int pyrLevel,
                                    const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                    const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    void calcHessGrad_side_sphere ( const int pyrLevel,
                                    const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                    const costFuncType method = PHOTO_CONSISTENCY,
                                    const int side = 0 ); // side is an aproximation parameter, 0 -> optimization starts at the identity and gradients are computed at the source; 1 at the target

    void calcHessGradRot_sphere(const int pyrLevel,
                                const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    double calcHessGradIC_sphere (  const int pyrLevel,
                                    const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                    const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    void calcHessGradInv_sphere(const int pyrLevel,
                                const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    void calcHessGrad_sphere_bidirectional (const int pyrLevel,
                                            const Eigen::Matrix4f & poseGuess, // The relative pose of the robot between the two frames
                                            const costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        Occlusions are taken into account by a Z-buffer. */
//    double computeError_sphereOcc1(int pyrLevel,
//                                    const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );
//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        This function takes into account the occlusions by storing a Z-buffer */
//    void calcHessGrad_sphereOcc1( int pyrLevel,
//                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );
//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        Occlusions are taken into account by a Z-buffer. */
//    double computeError_sphereOcc2(int pyrLevel,
//                                    const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );

//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        This function takes into account the occlusions and the moving pixels by applying a filter on the maximum depth error */
//    void calcHessGrad_sphereOcc2( int pyrLevel,
//                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );

    /*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
      * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
      * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution. */
    void regist( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(), costFuncType method = PHOTO_CONSISTENCY, const int occlusion = 0);

    void register_InvDepth ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                                 costFuncType method = PHOTO_CONSISTENCY,
                                 const int occlusion = 0);

    void register_IC(const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                         costFuncType method = PHOTO_CONSISTENCY,
                         const int occlusion = 0);

    void register_inv ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                            costFuncType method = PHOTO_CONSISTENCY,
                            const int occlusion = 0);

    void register_bidirectional ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                            costFuncType method = PHOTO_CONSISTENCY,
                            const int occlusion = 0);

    /*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
      * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
      * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
      * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
    */
    void register360 ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                        costFuncType method = PHOTO_CONSISTENCY,
                        const int occlusion = 0);

    void register360_warp ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                            costFuncType method = PHOTO_CONSISTENCY,
                            const int occlusion = 0);

    void register360_side ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                            costFuncType method = PHOTO_CONSISTENCY,
                            const int occlusion = 0);

    void register360_rot ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                        costFuncType method = PHOTO_CONSISTENCY,
                        const int occlusion = 0);

    void register360_salientJ ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                                costFuncType method = PHOTO_CONSISTENCY,
                                const int occlusion = 0);

    void register360_IC(const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                        costFuncType method = PHOTO_CONSISTENCY,
                        const int occlusion = 0);

    void register360_depthPyr(const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                                costFuncType method = PHOTO_CONSISTENCY,
                                const int occlusion = 0);

    void register360_inv(const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                            costFuncType method = PHOTO_CONSISTENCY,
                            const int occlusion = 0);

    void register360_bidirectional ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                                        costFuncType method = PHOTO_CONSISTENCY,
                                        const int occlusion = 0);

    /*! Calculate entropy of the planar matching. This is the differential entropy of a multivariate normal distribution given
      by the matched planes. This is calculated with the formula used in the paper ["Dense Visual SLAM for RGB-D Cameras",
      by C. Kerl et al., in IROS 2013]. */
    inline float calcEntropy()
    {
        Eigen::Matrix<float,6,6> covarianceM = hessian.inverse();
        std::cout << covarianceM.determinant() << " " << hessian.determinant() << " " << log(2*PI) << endl;
        float DOF = 6;
        float entropy = 0.5 * ( DOF * (1 + log(2*PI)) + log(covarianceM.determinant()) );

        return entropy;
    }

    /*! Align depth frames applying ICP in different pyramid scales. */
    double alignPyramidICP(Eigen::Matrix4f poseGuess);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double calcDenseError_rgbd360_singlesensor(int pyrLevel,
                                   const Eigen::Matrix4f poseGuess,
                                   const Eigen::Matrix4f &poseCamRobot,
                                   costFuncType method = PHOTO_CONSISTENCY);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad_rgbd360_singlesensor( int pyrLevel,
                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                    const Eigen::Matrix4f &poseCamRobot, // The pose of the camera wrt to the Robot (fixed beforehand through calibration) // Maybe calibration can be computed at the same time
                                    costFuncType method = PHOTO_CONSISTENCY);

    /*! Update the Hessian and the Gradient from a list of jacobians and residuals. */
    void updateHessianAndGradient(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi & valid_pixels);
    void updateHessianAndGradient(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXf & pixel_weights, const Eigen::MatrixXi & valid_pixels);
    void updateHessianAndGradient3D(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals_3D, const Eigen::MatrixXi & valid_pixels);
    void updateHessianAndGradient3D(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals_3D, const Eigen::MatrixXf & pixel_weights, const Eigen::MatrixXi & valid_pixels);

//    void updateHessianAndGradient(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi &warp_pixels);

    void updateGrad(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi & valid_pixels);

};
