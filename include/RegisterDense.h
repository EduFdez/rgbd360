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
    /*! Camera matrix (intrinsic parameters). This is only required for pinhole perspective sensors */
    Eigen::Matrix3f cameraMatrix;

    /*! Camera intrinsic parameters */
    float fx, fy, ox, oy;

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
    Eigen::Matrix4f registered_pose_;

    /*! The Hessian matrix of the optimization problem. At the solution it represents the inverse of the covariance matrix of the relative pose */
    Eigen::Matrix<float,6,6> hessian;

    /*! The Gradient vector of the optimization problem. At the solution it should be zero */
    Eigen::Matrix<float,6,1> gradient;

    //    /* Current iteration at the current optimization level.*/
    //    int iter;

    /*! Minimum allowed depth to consider a depth pixel valid.*/
    float min_depth_;

    /*! Maximum allowed depth to consider a depth pixel valid.*/
    float max_depth_;

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

//    /*! Sensed-Space-Overlap of the registered frames. This is the relation between the co-visible pixels and the total number of pixels in the image.*/
//    float SSO;

    /*! Enable the visualization of the optimization process (only for debug).*/
    bool visualizeIterations;

    /*! Warped intensity image for visualization purposes.*/
    cv::Mat warped_source_grayImage;

    /*! Warped intensity image for visualization purposes.*/
    cv::Mat warped_source_depthImage;

    /*! Vector containing the indices of the pixels that move (forward-backward) or are occluded in the target image to the source image.*/
    cv::Mat mask_dynamic_occlusion;

//    /*! LUT of 3D points of the spherical image.*/
//    std::vector<Eigen::Vector3f> unit_sphere_;

    /*! LUT of 3D points of the source image.*/
    Eigen::MatrixXf LUT_xyz_source;
    Eigen::MatrixXf LUT_xyz_target;
    Eigen::MatrixXf pts_src_transformed;
    Eigen::MatrixXf pts_trg_transformed;

    /*! Store a copy of the residuals and the weights to speed-up the registration. (Before they were computed twice: in the error function and the Jacobian)*/
    Eigen::MatrixXi warp_img_src;
    Eigen::VectorXi warp_pixels_src;
    Eigen::VectorXf residualsPhoto_src;
    Eigen::VectorXf residualsDepth_src;
    Eigen::MatrixXf jacobiansPhoto;
    Eigen::MatrixXf jacobiansDepth;
    Eigen::VectorXf stdDevError_inv_src;
    Eigen::VectorXf wEstimPhoto_src;
    Eigen::VectorXf wEstimDepth_src;

    /*! If set to true, only the pixels with high gradient in the gray image are used (for both photo and depth minimization) */
    Eigen::VectorXi validPixels_src;
    Eigen::VectorXi visible_pixels_src;
    Eigen::VectorXi validPixelsPhoto_src;
    Eigen::VectorXi validPixelsDepth_src;

    Eigen::MatrixXi warp_img_trg;
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
    std::vector<int> num_iterations;

public:

    enum sensorType
    {
        RGBD360_INDOOR = 0,
        STEREO_OUTDOOR,
    } sensor_type;

    /*! Sensed-Space-Overlap of the registered frames. This is the relation between the co-visible pixels and the total number of pixels in the image.*/
    float SSO;

//    /*! The average residual per pixel of photo/depth consistency.*/
//    float avResidual;

//    /*! The average residual per pixel of photo consistency.*/
//    double avPhotoResidual;

//    /*! The average residual per pixel of depth consistency.*/
//    double avDepthResidual;

    /*! Number of pyramid levels.*/
    int nPyrLevels;

    /*! Optimization method (cost function). The options are: 0=Photoconsistency, 1=Depth consistency (ICP like), 2= A combination of photo-depth consistency */
    enum costFuncType {PHOTO_CONSISTENCY, DEPTH_CONSISTENCY, PHOTO_DEPTH} method;

    /*! Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numpyramidLevels' levels.*/
    std::vector<cv::Mat> graySrcPyr, grayTrgPyr, depthSrcPyr, depthTrgPyr;
    std::vector<cv::Mat> grayTrgGradXPyr, grayTrgGradYPyr, depthTrgGradXPyr, depthTrgGradYPyr;
    std::vector<cv::Mat> graySrcGradXPyr, graySrcGradYPyr, depthSrcGradXPyr, depthSrcGradYPyr;
    std::vector<cv::Mat> colorSrcPyr;

    RegisterDense();

    /*! Set the number of pyramid levels.*/
    inline void setSensorType(const sensorType sensor)
    {
        sensor_type = sensor;
    };

    /*! Set the number of pyramid levels.*/
    inline void setNumPyr(const int Npyr)
    {
        nPyrLevels = Npyr;
    };

    /*! Set the bilinear interpolation for the last pyramid.*/
    inline void setBilinearInterp(const bool use_bilinear)
    {
        use_bilinear_ = use_bilinear;
    };

    /*! Set the minimum depth distance (m) to consider a certain pixel valid.*/
    inline void setMinDepth(const float minD)
    {
        min_depth_ = minD;
    };

    /*! Set the maximum depth distance (m) to consider a certain pixel valid.*/
    inline void setMaxDepth(const float maxD)
    {
        max_depth_ = maxD;
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
    void useSaliency(const bool use_salient_pixels__)
    {
        use_salient_pixels_ = use_salient_pixels__;
    };

    /*! Returns the optimal SE(3) rigid transformation matrix between the source and target frame.
     * This method has to be called after calling the register() method.*/
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

    /*! Build a pyramid of nLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
    void buildPyramid( const cv::Mat & img, std::vector<cv::Mat> & pyramid, const int nLevels);

    /*! Build a pyramid of nLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
    void buildPyramidRange( const cv::Mat & img, std::vector<cv::Mat> & pyramid, const int nLevels);

    /*! Build the pyramid levels from the intensity images.*/
    inline void buildGrayPyramids()
    {
        buildPyramid(graySrc, graySrcPyr, nPyrLevels);
        buildPyramid(grayTrg, graySrcPyr, nPyrLevels);
    };

    /*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
    void calcGradientXY( const cv::Mat & src, cv::Mat & gradX, cv::Mat & gradY);

    /*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
    void calcGradientXY_saliency( const cv::Mat & src, cv::Mat & gradX, cv::Mat & gradY, std::vector<int> & vSalientPixels_);

    /*! Compute the gradient images for each pyramid level. */
    void buildGradientPyramids( const std::vector<cv::Mat> & grayPyr, std::vector<cv::Mat> & grayGradXPyr, std::vector<cv::Mat> & grayGradYPyr,
                                const std::vector<cv::Mat> & depthPyr, std::vector<cv::Mat> & depthGradXPyr, std::vector<cv::Mat> & depthGradYPyr);

    /*! Sets the source (Intensity+Depth) frame.*/
    void setSourceFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth);

    /*! Sets the source (Intensity+Depth) frame. Depth image is ignored*/
    void setTargetFrame(const cv::Mat & imgRGB, cv::Mat & imgDepth);

    /*! Swap the source and target images */
    inline void swapSourceTarget()
    {
        double time_start = pcl::getTime();

        cv::Mat graySrc_tmp = graySrc;
        cv::Mat depthSrc_tmp = depthSrc;
        std::vector<cv::Mat> graySrcGradXPyr_tmp = graySrcGradXPyr;
        std::vector<cv::Mat> graySrcGradYPyr_tmp = graySrcGradYPyr;
        std::vector<cv::Mat> depthSrcGradXPyr_tmp = depthSrcGradXPyr;
        std::vector<cv::Mat> depthSrcGradYPyr_tmp = depthSrcGradYPyr;

        graySrc = grayTrg;
        depthSrc = depthTrg;
        graySrcGradXPyr = grayTrgGradXPyr;
        graySrcGradYPyr = grayTrgGradYPyr;
        depthSrcGradXPyr = depthTrgGradXPyr;
        depthSrcGradYPyr = depthTrgGradYPyr;

        grayTrg = graySrc_tmp;
        depthTrg = depthSrc_tmp;
        grayTrgGradXPyr = graySrcGradXPyr_tmp;
        grayTrgGradYPyr = graySrcGradYPyr_tmp;
        depthTrgGradXPyr = depthSrcGradXPyr_tmp;
        depthTrgGradYPyr = depthSrcGradYPyr_tmp;

        double time_end = pcl::getTime();
        std::cout << "RegisterDense::swapSourceTarget took " << (time_end - time_start) << std::endl;
    };

    /*! Compute the 3D points XYZ according to the pinhole camera model. */
    void computePinholeXYZ(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels);

    /*! Compute the 3D points XYZ according to the pinhole camera model. */
    void computePinholeXYZ_sse(const cv::Mat & depth_img, Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels);

    /*! Compute the 3D points XYZ by multiplying the unit sphere by the spherical depth image. */
    void computeSphereXYZ(const cv::Mat & depth_img, Eigen::MatrixXf & sphere_xyz, Eigen::VectorXi & validPixels);

    /*! Compute the 3D points XYZ by multiplying the unit sphere by the spherical depth image. */
    void computeSphereXYZ_sse(const cv::Mat & depth_img, Eigen::MatrixXf & sphere_xyz, Eigen::VectorXi & validPixels);

    /*! Get a list of salient points (pixels with hugh gradient) and compute their 3D position xyz */
    void getSalientPoints_sphere(Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                 const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                 const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY
                                ); // TODO extend this function to employ only depth

    void getSalientPoints_sphere_sse(Eigen::MatrixXf & xyz, Eigen::VectorXi & validPixels,
                                     const cv::Mat & depth_img, const cv::Mat & depth_gradX, const cv::Mat & depth_gradY,
                                     const cv::Mat & intensity_img, const cv::Mat & intensity_gradX, const cv::Mat & intensity_gradY
                                    ); // TODO extend this function to employ only depth

    /*! Transform 'input_pts', a set of 3D points according to the given rigid transformation 'Rt'. The output set of points is 'output_pts' */
    void transformPts3D(const Eigen::MatrixXf & input_pts, const Eigen::Matrix4f & Rt, Eigen::MatrixXf & output_pts);

    /*! Transform 'input_pts', a set of 3D points according to the given rigid transformation 'Rt'. The output set of points is 'output_pts' */
    void transformPts3D_sse(const Eigen::MatrixXf & input_pts, const Eigen::Matrix4f & Rt, Eigen::MatrixXf & output_pts);

    /*! Project pixel spherical */
    inline void projectSphere(const int & nCols, const int & nRows, const float & phi_FoV, const int & c, const int & r, const float & depth, const Eigen::Matrix4f & poseGuess,
                              int & transformed_r_int, int & transformed_c_int) // output parameters
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
        Eigen::Vector4f xyz = poseGuess*point3D;
        float depth_trg = sqrt(xyz(0)*xyz(0) + xyz(1)*xyz(1) + xyz(2)*xyz(2));

        //Project the 3D point to the S2 sphere
        float phi_trg = asin(xyz(2)/depth_trg);
        float theta_trg = atan2(-xyz(1),-xyz(2));
        transformed_r_int = round(phi_trg*nRows/phi_FoV + nRows/2);
        transformed_c_int = round(theta_trg*nCols/(2*PI));
    };

    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightHuber(const T & error)//, const T &scale)
    {
        //        assert(!std::isnan(error) && !std::isnan(scale))
        T weight = (T)1;
        const T scale = 1.345;
        T error_abs = fabs(error);
        if(error_abs < scale){//std::cout << "weight One\n";
            return weight;}

        weight = scale / error_abs;
        //std::cout << "weight " << weight << "\n";
        return weight;
    };

    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightHuber_sqrt(const T & error)//, const T &scale)
    {
        //        assert(!std::isnan(error) && !std::isnan(scale))
        T weight = (T)1;
        const T scale = 1.345;
        T error_abs = fabs(error);
        if(error_abs < scale){//std::cout << "weight One\n";
            return weight;}

        weight = sqrt(scale / error_abs);
        //std::cout << "weight " << weight << "\n";
        return weight;
    };

    /*! Tukey weight for robust estimation. */
    template<typename T>
    inline T weightTukey(const T & error)//, const T &scale)
    {
        T weight = (T)0.;
        const T scale = 4.685;
        T error_abs = fabs(error);
        if(error_abs > scale)
            return weight;

        T error_scale = error_abs/scale;
        T w_aux = 1 - error_scale*error_scale;
        weight = w_aux*w_aux;
        return weight;
    };

    /*! T-distribution weight for robust estimation. This is computed following the paper: "Robust Odometry Estimation for RGB-D Cameras" Kerl et al. ICRA 2013 */
    template<typename T>
    inline T weightTDist(const T & error, const T & stdDev, const T & nu)//, const T &scale)
    {
        T err_std = error / stdDev;
        T weight = nu+1 / (nu + err_std*err_std);
        return weight;
    };

    /*! Compute the standard deviation of the T-distribution following the paper: "Robust Odometry Estimation for RGB-D Cameras" Kerl et al. ICRA 2013 */
    template<typename T>
    inline T stdDev_TDist(const std::vector<T> & v_error, const T & stdDev, const T & nu)//, const T &scale)
    {
        std::vector<T> &v_error2( v_error.size() );
        for(size_t i=0; i < v_error.size(); ++i)
            v_error2[i] = v_error[i]*v_error[i];

        int it = 0;
        int max_iterations = 5;
        T diff_convergence = 1e-3;
        T diff_var = 100;
        T variance_prev = 10000;
        while (diff_var > diff_convergence && it < max_iterations)
        {
            T variance = 0.f;
            for(size_t i=0; i < v_error.size(); ++i)
                variance += v_error2[i]*weightTDist(v_error[i]);
            variance /= v_error.size();
            diff_var = fabs(variance_prev - variance);
            variance_prev = variance;
            ++it;
        }
        return sqrt(variance_prev);
    };

    template<typename T>
    inline T weightMEstimator(const T & error_scaled)
    {
        //std::cout << " error_scaled " << error_scaled << "weightHuber(error_scaled) " << weightHuber(error_scaled) << "\n";
        return weightHuber(error_scaled);
        //return weightTukey(error_scaled);
        //return weightTDist(error_scaled,dev,5);
    };

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double errorDense(const int & pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method = PHOTO_CONSISTENCY);//, const bool use_bilinear = false);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double errorDense_inv(const int & pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method = PHOTO_CONSISTENCY);//, const bool use_bilinear = false);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad( const int & pyramidLevel,
                       const Eigen::Matrix4f poseGuess,
                       costFuncType method = PHOTO_CONSISTENCY );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad_inv( const int & pyramidLevel,
                           const Eigen::Matrix4f poseGuess,
                           costFuncType method = PHOTO_CONSISTENCY );

//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method. */
//    double errorDense_Occ1(const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method = PHOTO_CONSISTENCY);
//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient. */
//    void calcHessGrad_Occ1(const int &pyramidLevel,
//                           const Eigen::Matrix4f poseGuess,
//                           costFuncType method = PHOTO_CONSISTENCY);

//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method. */
//    double errorDense_Occ2(const int &pyramidLevel, const Eigen::Matrix4f poseGuess, costFuncType method = PHOTO_CONSISTENCY);

//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient. */
//    void calcHessGrad_Occ2(const int &pyramidLevel,
//                           const Eigen::Matrix4f poseGuess,
//                           costFuncType method = PHOTO_CONSISTENCY);


//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        This is done following the work in:
//        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
//        in Computer Vision Workshops (ICCV Workshops), 2011. */
//    void calcHessianAndGradient(const int &pyramidLevel,
//                               const Eigen::Matrix4f poseGuess,
//                               costFuncType method = PHOTO_CONSISTENCY);

    void warpImage ( const int &pyramidLevel, const Eigen::Matrix4f &poseGuess, costFuncType method );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
    double errorDense_sphere (  const int &pyramidLevel,
                                const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    double errorDenseInv_sphere(const int &pyramidLevel,
                                const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                costFuncType method = PHOTO_CONSISTENCY); //,const bool use_bilinear = false );

    double errorDense_sphere_bidirectional (const int &pyramidLevel,
                                            const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
                                            costFuncType method = PHOTO_CONSISTENCY); //,const bool use_bilinear = false );

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
    This is done following the work in:
    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
    in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad_sphere(   const int &pyramidLevel,
                                const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    void calcHessGradInv_sphere(   const int &pyramidLevel,
                                const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

    void calcHessGrad_sphere_bidirectional (const int &pyramidLevel,
                                            const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                            costFuncType method = PHOTO_CONSISTENCY );//,const bool use_bilinear = false );

//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        Occlusions are taken into account by a Z-buffer. */
//    double errorDense_sphereOcc1(const int &pyramidLevel,
//                                    const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );
//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        This function takes into account the occlusions by storing a Z-buffer */
//    void calcHessGrad_sphereOcc1( const int &pyramidLevel,
//                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );
//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        Occlusions are taken into account by a Z-buffer. */
//    double errorDense_sphereOcc2(const int &pyramidLevel,
//                                    const Eigen::Matrix4f &poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );

//    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
//        This function takes into account the occlusions and the moving pixels by applying a filter on the maximum depth error */
//    void calcHessGrad_sphereOcc2( const int &pyramidLevel,
//                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
//                                    costFuncType method = PHOTO_CONSISTENCY );

    /*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
      * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
      * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution. */
    void registerRGBD ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                         costFuncType method = PHOTO_CONSISTENCY,
                         const int occlusion = 0);

    void registerRGBD_inv ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
                            costFuncType method = PHOTO_CONSISTENCY,
                            const int occlusion = 0);

    void registerRGBD_bidirectional ( const Eigen::Matrix4f pose_guess = Eigen::Matrix4f::Identity(),
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

    /*! Compute the unit sphere for the given spherical image dimmensions. This serves as a LUT to speed-up calculations. */
    void computeUnitSphere();

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
    double calcDenseError_rgbd360_singlesensor(const int &pyramidLevel,
                                   const Eigen::Matrix4f poseGuess,
                                   const Eigen::Matrix4f &poseCamRobot,
                                   costFuncType method = PHOTO_CONSISTENCY);

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessGrad_rgbd360_singlesensor( const int &pyramidLevel,
                                    const Eigen::Matrix4f poseGuess, // The relative pose of the robot between the two frames
                                    const Eigen::Matrix4f &poseCamRobot, // The pose of the camera wrt to the Robot (fixed beforehand through calibration) // Maybe calibration can be computed at the same time
                                    costFuncType method = PHOTO_CONSISTENCY);

    /*! Update the Hessian and the Gradient from a list of jacobians and residuals. */
    void updateHessianAndGradient(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi & valid_pixels);

//    void updateHessianAndGradient(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi &warp_pixels);

    void updateGrad(const Eigen::MatrixXf & pixel_jacobians, const Eigen::MatrixXf & pixel_residuals, const Eigen::MatrixXi & valid_pixels);

    /*! Return the value of the bilinear interpolation on the image 'img' given by the floating point indices 'x' and 'y' */
//    inline cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2f pt)
    //template <typename T>
    inline float bilinearInterp(const cv::Mat & img, cv::Point2f pt)
    {
        assert( img.type() == CV_32FC1 && !img.empty() );
        cv::Mat patch;
        cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
        return patch.at<float>(0,0);
    }

    /*! Return the value of the bilinear interpolation on the image 'img' given by the floating point indices 'x' and 'y'.
     * It takes into account NaN pixels and (<= 0 && > max_depth_) values to rule them out of the interpolation
     */
    inline float bilinearInterp_depth(const cv::Mat& img, const cv::Point2f &pt)
    {
        assert( img.type() == CV_32FC1 && !img.empty() );

        float *_img = reinterpret_cast<float*>(img.data);

        int x = (int)pt.x;
        int y = (int)pt.y;

        size_t x0_y0 = y * img.cols + x;
        size_t x1_y0 = x0_y0 + 1;
        size_t x0_y1 = x0_y0 + img.cols;
        size_t x1_y1 = x0_y1 + 1;

        float a = pt.x - (float)x;
        float b = 1.f - a;
        float c = pt.y - (float)y;
        float d = 1.f - c;

        float pt_y0;
        if( _img[x0_y0] < max_depth_ && _img[x1_y0] < max_depth_ && _img[x0_y0] >= 0 && _img[x1_y0] >= 0 )
            pt_y0 = _img[x0_y0] * b + _img[x1_y0] * a;
        else if (_img[x0_y0] < max_depth_ && _img[x0_y0] >= 0 )
            pt_y0 = _img[x0_y0];
        else //if(_img[x0_y0] < max_depth_)
            pt_y0 = _img[x1_y0];
        // The NaN/OutOfDepth case (_img[x0_y0] > max_depth_ && _img[x1_y0] > max_depth_) is automatically assumed

        float pt_y1;
        if( _img[x0_y1] < max_depth_ && _img[x1_y1] < max_depth_ && _img[x0_y1] >= 0 && _img[x1_y1] >= 0 )
            pt_y1 = _img[x0_y1] * b + _img[x1_y1] * a;
        else if (_img[x0_y1] < max_depth_ && _img[x0_y1] >= 0)
            pt_y1 = _img[x0_y1];
        else //if(_img[x0_y1] < max_depth_)
            pt_y1 = _img[x1_y1];

        float interpDepth;
        if( pt_y0 < max_depth_ && _img[x1_y1] < max_depth_ )
            interpDepth = pt_y0 * d + pt_y1 * c;
        else if (_img[x0_y1] < max_depth_)
            interpDepth = pt_y0;
        else
            interpDepth = pt_y1;

//        int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
//        int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
//        int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
//        int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

//        float pt_y0;
//        if( img.at<float>(y0, x0) < max_depth_ && img.at<float>(y0, x1) < max_depth_ && img.at<float>(y0, x0) >= 0 && img.at<float>(y0, x1) >= 0 )
//            pt_y0 = img.at<float>(y0, x0) * b + img.at<float>(y0, x1) * a;
//        else if (img.at<float>(y0, x0) < max_depth_ && img.at<float>(y0, x0) >= 0 )
//            pt_y0 = img.at<float>(y0, x0);
//        else //if(img.at<float>(y0, x0) < max_depth_)
//            pt_y0 = img.at<float>(y0, x1);
//        // The NaN/OutOfDepth case (img.at<float>(y0, x0) > max_depth_ && img.at<float>(y0, x1) > max_depth_) is automatically assumed

//        float pt_y1;
//        if( img.at<float>(y1, x0) < max_depth_ && img.at<float>(y1, x1) < max_depth_ && img.at<float>(y1, x0) >= 0 && img.at<float>(y1, x1) >= 0 )
//            pt_y1 = img.at<float>(y1, x0) * b + img.at<float>(y1, x1) * a;
//        else if (img.at<float>(y1, x0) < max_depth_ && img.at<float>(y1, x0) >= 0)
//            pt_y1 = img.at<float>(y1, x0);
//        else //if(img.at<float>(y1, x0) < max_depth_)
//            pt_y1 = img.at<float>(y1, x1);

//        float interpDepth;
//        if( pt_y0 < max_depth_ && img.at<float>(y1, x1) < max_depth_ )
//            interpDepth = pt_y0 * d + pt_y1 * c;
//        else if (img.at<float>(y1, x0) < max_depth_)
//            interpDepth = pt_y0;
//        else
//            interpDepth = pt_y1;

        return interpDepth;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_sphere(const Eigen::Vector3f & xyz, const float & dist, const float & pixel_angle_inv, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz(1)*xyz(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );

        jacobianWarpRt(0,0) = commonDer_c * xyz(2);
        jacobianWarpRt(0,1) = 0.f;
        jacobianWarpRt(0,2) = -commonDer_c * xyz(0);
//        jacobianWarpRt(1,0) = commonDer_r * xyz(0) * xyz(1);
        jacobianWarpRt(1,1) =-commonDer_r * x2_z2;
//        jacobianWarpRt(1,2) = commonDer_r * xyz(2) * xyz(1);
        float commonDer_r_y = commonDer_r * xyz(1);
        jacobianWarpRt(1,0) = commonDer_r_y * xyz(0);
        jacobianWarpRt(1,2) = commonDer_r_y * xyz(2);

        jacobianWarpRt(0,3) = jacobianWarpRt(0,2) * xyz(1);
        jacobianWarpRt(0,4) = jacobianWarpRt(0,0) * xyz(2) - jacobianWarpRt(0,2) * xyz(0);
        jacobianWarpRt(0,5) =-jacobianWarpRt(0,0) * xyz(1);
        jacobianWarpRt(1,3) =-jacobianWarpRt(1,1) * xyz(2) + jacobianWarpRt(1,2) * xyz(1);
        jacobianWarpRt(1,4) = jacobianWarpRt(1,0) * xyz(2) - jacobianWarpRt(1,2) * xyz(0);
        jacobianWarpRt(1,5) =-jacobianWarpRt(1,0) * xyz(1) + jacobianWarpRt(1,1) * xyz(0);

        //return jacobianWarpRt;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF of the inverse transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_sphere_inv(const Eigen::Vector3f & xyz, const Eigen::Vector3f & xyz_orig, const Eigen::Matrix3f & rotation, const float & dist, const float & pixel_angle_inv, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        // The Jacobian of the spherical projection
        Eigen::Matrix<float,2,3> jacobianProj23;
        float dist2 = dist * dist;
        float x2_z2 = dist2 - xyz(1)*xyz(1);
        float x2_z2_sqrt = sqrt(x2_z2);
        float commonDer_c = pixel_angle_inv / x2_z2;
        float commonDer_r = -pixel_angle_inv / ( dist2 * x2_z2_sqrt );
        jacobianProj23(0,0) = commonDer_c * xyz(2);
        jacobianProj23(0,1) = 0;
        jacobianProj23(0,2) =-commonDer_c * xyz(0);
//        jacobianProj23(1,0) = commonDer_r * xyz(0) * xyz(1);
        jacobianProj23(1,1) =-commonDer_r * x2_z2;
//        jacobianProj23(1,2) = commonDer_r * xyz(2) * xyz(1);
        float commonDer_r_y = commonDer_r * xyz(1);
        jacobianProj23(1,0) = commonDer_r_y * xyz(0);
        jacobianProj23(1,2) = commonDer_r_y * xyz(2);

        // !!! NOTICE that the 3D points involved are those from the target frame, which are projected throught the inverse transformation into the reference frame!!!
        Eigen::Matrix<float,3,6> jacobianT36_inv;
        jacobianT36_inv.block(0,0,3,3) = -rotation.transpose();
        jacobianT36_inv.block(0,3,3,1) = xyz_orig(2)*rotation.block(0,1,3,1) - xyz_orig(1)*rotation.block(0,2,3,1);
        jacobianT36_inv.block(0,4,3,1) = xyz_orig(0)*rotation.block(0,2,3,1) - xyz_orig(2)*rotation.block(0,0,3,1);
        jacobianT36_inv.block(0,5,3,1) = xyz_orig(1)*rotation.block(0,0,3,1) - xyz_orig(0)*rotation.block(0,1,3,1);

        std::cout << "jacobianProj23 \n" << jacobianProj23 << "\n jacobianT36_inv \n" << jacobianT36_inv << std::endl;

        jacobianWarpRt = jacobianProj23 * jacobianT36_inv;

        //return jacobianWarpRt;
    }

    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
    inline void
    //Eigen::Matrix<float,2,6>
    computeJacobian26_wT_pinhole(const Eigen::Vector3f & xyz, Eigen::Matrix<float,2,6> &jacobianWarpRt)
    {
        //Eigen::Matrix<float,2,6> jacobianWarpRt;

        float inv_transf_z = 1.0/xyz(2);

        //Derivative with respect to x
        jacobianWarpRt(0,0)=fx*inv_transf_z;
        jacobianWarpRt(1,0)=0.f;

        //Derivative with respect to y
        jacobianWarpRt(0,1)=0.f;
        jacobianWarpRt(1,1)=fy*inv_transf_z;

        //Derivative with respect to z
        float inv_transf_z_2 = inv_transf_z*inv_transf_z;
        jacobianWarpRt(0,2)=-fx*xyz(0)*inv_transf_z_2;
        jacobianWarpRt(1,2)=-fy*xyz(1)*inv_transf_z_2;

        //Derivative with respect to \w_x
        jacobianWarpRt(0,3)=-fx*xyz(1)*xyz(0)*inv_transf_z_2;
        jacobianWarpRt(1,3)=-fy*(1+xyz(1)*xyz(1)*inv_transf_z_2);

        //Derivative with respect to \w_y
        jacobianWarpRt(0,4)= fx*(1+xyz(0)*xyz(0)*inv_transf_z_2);
        jacobianWarpRt(1,4)= fy*xyz(0)*xyz(1)*inv_transf_z_2;

        //Derivative with respect to \w_z
        jacobianWarpRt(0,5)=-fx*xyz(1)*inv_transf_z;
        jacobianWarpRt(1,5)= fy*xyz(0)*inv_transf_z;

        //return jacobianWarpRt;
    }

//    /*! Compute the Jacobian composition of the warping + 3D transformation wrt to the 6DoF transformation */
//    inline void
//    computeJacobian16_depth(const Eigen::Vector3f & xyz, const float & dist_inv, Eigen::Matrix<float,1,6> &jacobian_depthT)
//    {
//        //Eigen::Matrix<float,2,6> jacobianWarpRt;
//        jacobian_depthT.block(0,0,1,3) = dist_inv * xyz.transpose();

//        //return jacobian_depthT;
//    }
};

#endif
