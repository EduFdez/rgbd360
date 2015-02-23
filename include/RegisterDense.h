/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include <Miscellaneous.h>

#include <opencv2/opencv.hpp>
//#include <Eigen/Core>

#ifndef REGISTER_PHOTO_ICP_H
#define REGISTER_PHOTO_ICP_H

/*! This class performs dense registration (also called direct registration) minimizing a cost function based on either intensity, depth or both.
 *  Refer to the last chapter of my thesis dissertation for more details:
 * "Contributions to metric-topological localization and mapping in mobile robotics" 2014. Eduardo Fernández-Moral.
 */
class RegisterDense
{
//public:
    /*! Camera matrix (intrinsic parameters). This is only required for the sesor RGBD360*/
    Eigen::Matrix3f cameraMatrix;

//    /* Vertical field of view in the sphere (in) .*/
//    float phi_FoV;

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
    Eigen::Matrix4f relPose;

    /*! The Hessian matrix of the optimization problem. At the solution it represents the inverse of the covariance matrix of the relative pose */
    Eigen::Matrix<float,6,6> hessian;

    /*! The Gradient vector of the optimization problem. At the solution it should be zero */
    Eigen::Matrix<float,6,1> gradient;

    //    /* Current iteration at the current optimization level.*/
    //    int iter;

    /*! Minimum allowed depth to consider a depth pixel valid.*/
    float minDepth;

    /*! Maximum allowed depth to consider a depth pixel valid.*/
    float maxDepth;

    /*! Minimum depth difference to filter the outliers.*/
    float minDepthOutliers;

    /*! Threshold depth difference to filter the outliers.*/
    float thresDepthOutliers;

    /*! Maximum depth difference to filter the outliers.*/
    float maxDepthOutliers;

    /*! Variance of intensity measurements. */
    float varPhoto;
    float stdDevPhoto;

    /*! Variance of intensity measurements. */
    float varDepth;
    float stdDevDepth;

    /*! Depth component gain. This variable is used to scale the depth values so that depth components are similar to intensity values.*/
    float depthComponentGain;

    /*! If set to true, only the pixels with high gradient in the gray image are used (for both photo and depth minimization) */
    bool bUseSalientPixels;

    /*! A threshold to select salient pixels on the intensity and depth images.*/
    float thresSaliency;
    float thresSaliencyIntensity;
    float thresSaliencyDepth;

//    /*! Sensed-Space-Overlap of the registered frames. This is the relation between the co-visible pixels and the total number of pixels in the image.*/
//    float SSO;

    /*! If set to true, only the pixels with high gradient in the gray image are used (for both photo and depth minimization) */
    std::vector<std::vector<int> > vSalientPixels;

    /*! Enable the visualization of the optimization process (only for debug).*/
    bool visualizeIterations;

    /*! Warped intensity image for visualization purposes.*/
    cv::Mat warped_source_grayImage;

    /*! Warped intensity image for visualization purposes.*/
    cv::Mat warped_source_depthImage;

    /*! Vector containing the indices of the pixels that move (forward-backward) or are occluded in the target image to the source image.*/
    cv::Mat mask_dynamic_occlusion;

    /*! LUT of 3D points of the spherical image.*/
    std::vector<Eigen::Vector3f> unit_sphere_;

    /*! LUT of 3D points of the source image.*/
    std::vector<Eigen::Vector3f>  LUT_xyz_source;

    /*! Number of iterations in each pyramid level.*/
    std::vector<int> num_iterations;

public:

    /*! Sensed-Space-Overlap of the registered frames. This is the relation between the co-visible pixels and the total number of pixels in the image.*/
    float SSO;

    /*! The average residual per pixel of photo/depth consistency.*/
    float avResidual;

    /*! The average residual per pixel of photo consistency.*/
    double avPhotoResidual;

    /*! The average residual per pixel of depth consistency.*/
    double avDepthResidual;

    /*! Number of pyramid levels.*/
    int nPyrLevels;

    /*! Optimization method (cost function). The options are: 0=Photoconsistency, 1=Depth consistency (ICP like), 2= A combination of photo-depth consistency */
    enum costFuncType {PHOTO_CONSISTENCY, DEPTH_CONSISTENCY, PHOTO_DEPTH} method;

    /*! Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numpyramidLevels' levels.*/
    std::vector<cv::Mat> graySrcPyr, grayTrgPyr, depthSrcPyr, depthTrgPyr, grayTrgGradXPyr, grayTrgGradYPyr, depthTrgGradXPyr, depthTrgGradYPyr;
    std::vector<cv::Mat> colorSrcPyr;

    RegisterDense();

    /*! Set the the number of pyramid levels.*/
    inline void setNumPyr(const int Npyr)
    {
        nPyrLevels = Npyr;
    };

    /*! Set the minimum depth distance (m) to consider a certain pixel valid.*/
    inline void setMinDepth(const float minD)
    {
        minDepth = minD;
    };

    /*! Set the maximum depth distance (m) to consider a certain pixel valid.*/
    inline void setMaxDepth(const float maxD)
    {
        maxDepth = maxD;
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

    /*! Set the 3x3 matrix of (pinhole) camera intrinsic parameters used to obtain the 3D colored point cloud from the RGB and depth images.*/
    inline void setCameraMatrix(const Eigen::Matrix3f & camMat)
    {
        cameraMatrix = camMat;
    };

    /*! Set the visualization of the optimization progress.*/
    inline void setVisualization(const bool viz)
    {
        visualizeIterations = viz;
    };

    /*! Set the a variable to indicate whether pixel saliency is used.*/
    void useSaliency(const bool bUseSalientPixels_)
    {
        bUseSalientPixels = bUseSalientPixels_;
    };

    /*! Returns the optimal SE(3) rigid transformation matrix between the source and target frame.
     * This method has to be called after calling the alignFrames() method.*/
    inline Eigen::Matrix4f getOptimalPose()
    {
        return relPose;
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

    /*! Build a pyramid of nLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
    void buildPyramid(cv::Mat &img, std::vector<cv::Mat> &pyramid, const int nLevels);

    /*! Build a pyramid of nLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
    void buildPyramidRange(cv::Mat &img, std::vector<cv::Mat> &pyramid, const int nLevels);

    /*! Build the pyramid levels from the intensity images.*/
    inline void buildGrayPyramids()
    {
        buildPyramid(graySrc, graySrcPyr, nPyrLevels);
        buildPyramid(grayTrg, graySrcPyr, nPyrLevels);
    };

    /*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
    void calcGradientXY(cv::Mat &src, cv::Mat &gradX, cv::Mat &gradY);

    /*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
    void calcGradientXY_saliency(cv::Mat &src, cv::Mat &gradX, cv::Mat &gradY, std::vector<int> &vSalientPixels_);

    /*! Compute the gradient images for each pyramid level. */
    //    void buildGradientPyramids(std::vector<cv::Mat>& graySrcPyr,std::vector<cv::Mat>& grayTrgGradXPyr,std::vector<cv::Mat>& grayTrgGradYPyr)
    void buildGradientPyramids();

    /*! Sets the source (Intensity+Depth) frame.*/
    void setSourceFrame(cv::Mat &imgRGB,cv::Mat &imgDepth);

    /*! Sets the source (Intensity+Depth) frame. Depth image is ignored*/
    void setTargetFrame(cv::Mat &imgRGB,cv::Mat &imgDepth);

    /*! Project pixel spherical */
    inline void projectSphere(int &nCols, int &nRows, float &phi_FoV, int &c, int &r, float &depth, Eigen::Matrix4f &poseGuess,
                              int &transformed_r_int, int &transformed_c_int) // output parameters
    {
        float phi = r*(2*phi_FoV/nRows);
        float theta = c*(2*PI/nCols);

        //Compute the 3D coordinates of the pij of the source frame
        Eigen::Vector4f point3D;
        point3D(0) = depth*sin(phi);
        point3D(1) = -depth*cos(phi)*sin(theta);
        point3D(2) = -depth*cos(phi)*cos(theta);
        point3D(3) = 1;

        //Transform the 3D point using the transformation matrix Rt
        Eigen::Vector4f transformedPoint3D = poseGuess*point3D;
        float depth_trg = sqrt(transformedPoint3D(0)*transformedPoint3D(0) + transformedPoint3D(1)*transformedPoint3D(1) + transformedPoint3D(2)*transformedPoint3D(2));

        //Project the 3D point to the S2 sphere
        float phi_trg = asin(transformedPoint3D(2)/depth_trg);
        float theta_trg = atan2(-transformedPoint3D(1),-transformedPoint3D(2));
        transformed_r_int = round(phi_trg*nRows/phi_FoV + nRows/2);
        transformed_c_int = round(theta_trg*nCols/(2*PI));
    };

    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightHuber(const T &error)//, const T &scale)
    {
        //        assert(!std::isnan(error) && !std::isnan(scale))
        const T scale = 1.345;
        T error_abs = fabs(error);
        if(error_abs < scale)
            return 1.;

        T weight = scale / error_abs;
        return weight;
        //MAD/0.6745,
    };

    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightTukey(const T &error)//, const T &scale)
    {
        const T scale = 4.685;
        T error_abs = fabs(error);
        if(error_abs > scale)
            return 0.;

        T error_scale = error_abs/scale;
        T w_aux = 1 - error_scale*error_scale;
        T weight = w_aux*w_aux;
        return weight;
    };

    template<typename T>
    inline T weightMEstimator(const T &error)
    {
        weightHuber(error);
//        weightTukey(error);
    };

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double errorDense(const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method = PHOTO_CONSISTENCY);//, const bool use_bilinear = false);
    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad(const int &pyramidLevel,
                               const Eigen::Matrix4f poseGuess,
                               costFuncType method = PHOTO_CONSISTENCY );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method. */
    double errorDense_Occ1(const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method = PHOTO_CONSISTENCY);
    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient. */
    void calcHessGrad_Occ1(const int &pyramidLevel,
                           const Eigen::Matrix4f poseGuess,
                           costFuncType method = PHOTO_CONSISTENCY);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method. */
    double errorDense_Occ2(const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method = PHOTO_CONSISTENCY);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient. */
    void calcHessGrad_Occ2(const int &pyramidLevel,
                           const Eigen::Matrix4f poseGuess,
                           costFuncType method = PHOTO_CONSISTENCY);


    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double calcDense_SqError (  const int &pyramidLevel,
                                const Eigen::Matrix4f poseGuess,
                                double varPhoto = pow(3./255,2),
                                double varDepth = 0.0001,
                                costFuncType method = PHOTO_CONSISTENCY);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessianAndGradient(const int &pyramidLevel,
                               const Eigen::Matrix4f poseGuess,
                               costFuncType method = PHOTO_CONSISTENCY);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
    double errorDense_sphere (  const int &pyramidLevel,
                                const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                costFuncType method = PHOTO_CONSISTENCY,
                                const bool use_bilinear = false );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad_sphere(   const int &pyramidLevel,
                                const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                costFuncType method = PHOTO_CONSISTENCY,
                                const bool use_bilinear = false );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        Occlusions are taken into account by a Z-buffer. */
    double errorDense_sphereOcc1(const int &pyramidLevel,
                                    const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                    costFuncType method = PHOTO_CONSISTENCY );
    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This function takes into account the occlusions by storing a Z-buffer */
    void calcHessGrad_sphereOcc1( const int &pyramidLevel,
                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                    costFuncType method = PHOTO_CONSISTENCY );
    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        Occlusions are taken into account by a Z-buffer. */
    double errorDense_sphereOcc2(const int &pyramidLevel,
                                    const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                    costFuncType method = PHOTO_CONSISTENCY );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This function takes into account the occlusions and the moving pixels by applying a filter on the maximum depth error */
    void calcHessGrad_sphereOcc2( const int &pyramidLevel,
                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                    costFuncType method = PHOTO_CONSISTENCY );

    /*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
      * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
      * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution. */
    void alignFrames(const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                     costFuncType method = PHOTO_CONSISTENCY,
                     const int occlusion = 0);

    /*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
      * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
      * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution.
      * The input parameter occlusion stands for: (0->Regular dense registration, 1->Occlusion1, 2->Occlusion2)
    */
    void alignFrames360(const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                        costFuncType method = PHOTO_CONSISTENCY,
                        const int occlusion = 0);

    void computeUnitSphere();

    void alignFrames360_unity (  const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
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
    double calcDenseError_robot(const int &pyramidLevel,
                                   const Eigen::Matrix4f poseGuess,
                                   const Eigen::Matrix4f &poseCamRobot,
                                   costFuncType method = PHOTO_CONSISTENCY);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessianGradient_robot( const int &pyramidLevel,
                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                    const Eigen::Matrix4f &poseCamRobot, // The pose of the camera wrt to the Robot (fixed beforehand through calibration) // Maybe calibration can be computed at the same time
                                    costFuncType method = PHOTO_CONSISTENCY);

    ///*! Return the value of the bilinear interpolation on the image 'img' given by the floating point indices 'x' and 'y' */
    //inline float
    //bilinearInterp(const cv::Mat & img, cv::Point2f pt)

    /*! Return the value of the bilinear interpolation on the image 'img' given by the floating point indices 'x' and 'y' */
//    inline cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2f pt)
    inline float getColorSubpix(const cv::Mat& img, cv::Point2f pt)
    {
        cv::Mat patch;
        cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
        return patch.at<float>(0,0);
    }

    /*! Return the value of the bilinear interpolation on the image 'img' given by the floating point indices 'x' and 'y'.
     * It takes into account NaN pixels and (<= 0 && > maxDepth) values to rule them out of the interpolation
     */
    inline float getDepthSubpix(const cv::Mat& img, cv::Point2f pt)
    {
        assert(!img.empty());

    //    cv::Mat patch;
    //    cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
    //    return patch.at<float>(0,0);

        int x = (int)pt.x;
        int y = (int)pt.y;

        int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
        int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
        int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
        int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

        float a = pt.x - (float)x;
        float b = 1.f - a;
        float c = pt.y - (float)y;
        float d = 1.f - c;

        float pt_y0;
        if( img.at<float>(y0, x0) < maxDepth && img.at<float>(y0, x1) < maxDepth && img.at<float>(y0, x0) >= 0 && img.at<float>(y0, x1) >= 0 )
            pt_y0 = img.at<float>(y0, x0) * b + img.at<float>(y0, x1) * a;
        else if (img.at<float>(y0, x0) < maxDepth && img.at<float>(y0, x0) >= 0 )
            pt_y0 = img.at<float>(y0, x0);
        else //if(img.at<float>(y0, x0) < maxDepth)
            pt_y0 = img.at<float>(y0, x1);
        // The NaN/OutOfDepth case (img.at<float>(y0, x0) > maxDepth && img.at<float>(y0, x1) > maxDepth) is automatically assumed

        float pt_y1;
        if( img.at<float>(y1, x0) < maxDepth && img.at<float>(y1, x1) < maxDepth && img.at<float>(y1, x0) >= 0 && img.at<float>(y1, x1) >= 0 )
            pt_y1 = img.at<float>(y1, x0) * b + img.at<float>(y1, x1) * a;
        else if (img.at<float>(y1, x0) < maxDepth && img.at<float>(y1, x0) >= 0)
            pt_y1 = img.at<float>(y1, x0);
        else //if(img.at<float>(y1, x0) < maxDepth)
            pt_y1 = img.at<float>(y1, x1);

        float interpDepth;
        if( pt_y0 < maxDepth && img.at<float>(y1, x1) < maxDepth )
            interpDepth = pt_y0 * d + pt_y1 * c;
        else if (img.at<float>(y1, x0) < maxDepth)
            interpDepth = pt_y0;
        else
            interpDepth = pt_y1;

        return interpDepth;
    }

};

#endif
