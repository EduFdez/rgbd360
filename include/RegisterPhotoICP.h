/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#ifndef REGISTER_PHOTO_ICP_H
#define REGISTER_PHOTO_ICP_H

#define ENABLE_OPENMP 0
#define ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS 1

#include "Miscellaneous.h"

#include <mrpt/slam/CSimplePointsMap.h>
#include <mrpt/slam/CObservation2DRangeScan.h>
#include <mrpt/slam/CICP.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/poses/CPosePDF.h>
#include <mrpt/poses/CPosePDFGaussian.h>
//#include <mrpt/gui.h>
#include <mrpt/math/utils.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp> //TickMeter
#include <iostream>
#include <fstream>

// Return a diagonal matrix where the values of the diagonal are assigned from the input vector
template<typename typedata, int nRows, int nCols>
Eigen::Matrix<typedata,nRows,nCols> getDiagonalMatrix(const Eigen::Matrix<typedata,nRows,nCols> &matrix_generic)
{
    assert(nRows == nCols);

    Eigen::Matrix<typedata,nRows,nCols> m_diag = Eigen::Matrix<typedata,nRows,nCols>::Zero();
    for(size_t i=0; i < nRows; i++)
        m_diag(i,i) = matrix_generic(i,i);

    return m_diag;
}

class RegisterPhotoICP
{
    /*!Camera matrix (intrinsic parameters).*/
    Eigen::Matrix3f cameraMatrix;

    /*! The reference intensity image */
    cv::Mat graySrc;

    /*! The target intensity image */
    cv::Mat grayTrg;

    /*! The reference depth image */
    cv::Mat depthSrc;

    /*! The target depth image */
    cv::Mat depthTrg;

    /*! The relative pose between source and target RGB-D images */
    Eigen::Matrix4d relPose;

    /*! The Hessian matrix of the optimization problem. At the solution it represents the inverse of the covariance matrix of the relative pose */
    Eigen::Matrix<double,6,6> hessian;

    /*! The Gradient vector of the optimization problem. At the solution it should be zero */
    Eigen::Matrix<double,6,1> gradient;

    //    /* Current iteration at the current optimization level.*/
    //    int iter;

    /*! Minimum allowed depth to consider a depth pixel valid.*/
    float minDepth;

    /*! Maximum allowed depth to consider a depth pixel valid.*/
    float maxDepth;

    /*! Variance of intensity measurements. */
    double varPhoto;
    double stdDevPhoto;

    /*! Variance of intensity measurements. */
    double varDepth;
    double stdDevDepth;


    /*! Depth component gain. This variable is used to scale the depth values so that depth components are similar to intensity values.*/
    float depthComponentGain;

    /*! Enable the visualization of the optimization process (only for debug).*/
    bool visualizeIterations;

    /*! Warped intensity image for visualization purposes.*/
    cv::Mat warped_source_grayImage;

    /*! Warped intensity image for visualization purposes.*/
    cv::Mat warped_source_depthImage;

public:

    /*! Number of pyramid levels.*/
    int nPyrLevels;

    /*! Optimization method (cost function). The options are: 0=Photoconsistency, 1=Depth consistency (ICP like), 2= A combination of photo-depth consistency */
    enum costFuncType {PHOTO_CONSISTENCY, DEPTH_CONSISTENCY, PHOTO_DEPTH} method;

    /*! Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numpyramidLevels' levels.*/
    std::vector<cv::Mat> graySrcPyr, grayTrgPyr, depthSrcPyr, depthTrgPyr, grayTrgGradXPyr, grayTrgGradYPyr, depthTrgGradXPyr, depthTrgGradYPyr;

    RegisterPhotoICP() :
        minDepth(0.3),
        maxDepth(8.0),
        nPyrLevels(3),
        visualizeIterations(false)
    {
        stdDevPhoto = 3./255;
        varPhoto = stdDevPhoto*stdDevPhoto;

        stdDevDepth = 0.01;
        varDepth = stdDevDepth*stdDevDepth;
    };

    /*! Set the minimum depth distance (m) to consider a certain pixel valid.*/
    void setMinDepth(float minD)
    {
        minDepth = minD;
    };

    /*! Set the maximum depth distance (m) to consider a certain pixel valid.*/
    void setMaxDepth(float maxD)
    {
        maxDepth = maxD;
    };

    /*! Set the variance of the intensity.*/
    void setGrayVariance(double var)
    {
        varPhoto = var;
    };

    /*! Set the variance of the depth.*/
    void setDepthVariance(double var)
    {
        varDepth = var;
    };

    /*! Set the 3x3 matrix of (pinhole) camera intrinsic parameters used to obtain the 3D colored point cloud from the RGB and depth images.*/
    void setCameraMatrix(Eigen::Matrix3f & camMat)
    {
        cameraMatrix = camMat;
    };

    /*! Set the visualization of the optimization progress.*/
    void setVisualization(bool viz)
    {
        visualizeIterations = viz;
    };

    /*! Returns the optimal SE(3) rigid transformation matrix between the source and target frame.
     * This method has to be called after calling the alignFrames() method.*/
    Eigen::Matrix4d getOptimalPose()
    {
        return relPose;
    }

    /*! Returns the Hessian (the information matrix).*/
    Eigen::Matrix<double,6,6> getHessian()
    {
        return hessian;
    }

    /*! Returns the Gradient (the information matrix).*/
    Eigen::Matrix<double,6,1> getGradient()
    {
        return gradient;
    }

    /*! Build a pyramid of nLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
    void buildPyramid(cv::Mat &img, std::vector<cv::Mat> &pyramid, const int nLevels)
    {
        //Create space for all the images // ??
        pyramid.resize(nLevels);
        pyramid[0] = img;
        //        pyramid[0] = img - mean(img);

        for(int level=1; level < nLevels; level++)
        {
            //Create an auxiliar image of factor times the size of the original image
            cv::Mat imgAux;
            pyrDown( pyramid[level-1], imgAux, cv::Size( pyramid[level-1].cols/2, pyramid[level-1].rows/2 ) );

            //Assign the resized image to the current level of the pyramid
            pyramid[level]=imgAux;
        }
    };

    /*! Build a pyramid of nLevels of image resolutions from the input image.
     * The resolution of each layer is 2x2 times the resolution of its image above.*/
    void buildPyramidRange(cv::Mat &img, std::vector<cv::Mat> &pyramid, const int nLevels)
    {
        //Create space for all the images // ??
        pyramid.resize(nLevels);
        img.convertTo(pyramid[0], CV_32FC1, 0.001 );
//    std::cout << "types " << pyramid[0].type() << " " << img.type() << std::endl;

        for(int level=1; level < nLevels; level++)
        {
            //Create an auxiliar image of factor times the size of the original image
            cv::Mat imgAux(cv::Size( pyramid[level-1].cols/2, pyramid[level-1].rows/2 ), pyramid[0].type() );
            for(unsigned r=0; r < pyramid[level-1].rows; r+=2)
                for(unsigned c=0; c < pyramid[level-1].cols; c+=2)
                {
                    float avDepth = 0.f;
                    unsigned nValidPixels = 0;
                    for(unsigned i=0; i < 2; i++)
                        for(unsigned j=0; j < 2; j++)
                        {
                            float z = pyramid[level-1].at<float>(r+i,c+j);
//                        cout << "z " << z << endl;
                            if(z > minDepth && z < maxDepth)
                            {
                                avDepth += z;
                                ++nValidPixels;
                            }
                        }
                    //pyramid[level](r/2,c/2) = avDepth / nValidPixels;
                    imgAux.at<float>(r/2,c/2) = avDepth / nValidPixels;
                }

            //Assign the resized image to the current level of the pyramid
            pyramid[level] = imgAux;
        }
    };

    /*! Build the pyramid levels from the intensity images.*/
    void buildGrayPyramids()
    {
        buildPyramid(graySrc, graySrcPyr, nPyrLevels);
        buildPyramid(grayTrg, graySrcPyr, nPyrLevels);
    };

    /*! Calculate the image gradients in X and Y. This gradientes are calculated through weighted first order approximation (as adviced by Mariano Jaimez). */
    void calcGradientXY(cv::Mat &src, cv::Mat &gradX, cv::Mat &gradY)
    {
        int dataType = src.type();

        gradX = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );
        gradY = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type() );

        cout << "calcGradientXY \n";
        for(unsigned r=1; r < src.rows-1; r++)
            for(unsigned c=1; c < src.cols-1; c++)
            {
                if( (src.at<float>(r,c) > src.at<float>(r,c+1) && src.at<float>(r,c) < src.at<float>(r,c-1) ) ||
                    (src.at<float>(r,c) < src.at<float>(r,c+1) && src.at<float>(r,c) > src.at<float>(r,c-1) )   )
                    gradX.at<float>(r,c) = 2.f / (1/(src.at<float>(r,c+1)-src.at<float>(r,c)) + 1/(src.at<float>(r,c)-src.at<float>(r,c-1)));
//                else
//                    gradX.at<float>(r,c) = 0;

                if( (src.at<float>(r,c) > src.at<float>(r+1,c) && src.at<float>(r,c) < src.at<float>(r-1,c) ) ||
                    (src.at<float>(r,c) < src.at<float>(r+1,c) && src.at<float>(r,c) > src.at<float>(r-1,c) )   )
                    gradY.at<float>(r,c) = 2.f / (1/(src.at<float>(r+1,c)-src.at<float>(r,c)) + 1/(src.at<float>(r,c)-src.at<float>(r-1,c)));
            }

    };

    /*! Compute the gradient images for each pyramid level. */
    //    void buildGradientPyramids(std::vector<cv::Mat>& graySrcPyr,std::vector<cv::Mat>& grayTrgGradXPyr,std::vector<cv::Mat>& grayTrgGradYPyr)
    void buildGradientPyramids()
    {
        //Compute image gradients
//        double scale = 1./255;
//        double delta = 0;
//        int dataType = CV_32FC1; // grayTrgPyr[level].type();

        //Create space for all the derivatives images
        grayTrgGradXPyr.resize(grayTrgPyr.size());
        grayTrgGradYPyr.resize(grayTrgPyr.size());

        depthTrgGradXPyr.resize(grayTrgPyr.size());
        depthTrgGradYPyr.resize(grayTrgPyr.size());

        for(unsigned level=0; level < nPyrLevels; level++)
        {
            calcGradientXY(grayTrgPyr[level], grayTrgGradXPyr[level], grayTrgGradYPyr[level]);
            calcGradientXY(depthTrgPyr[level], depthTrgGradXPyr[level], depthTrgGradYPyr[level]);

            // Compute the gradient in x
//            grayTrgGradXPyr[level] = cv::Mat(cv::Size( grayTrgPyr[level].cols, grayTrgPyr[level].rows), grayTrgPyr[level].type() );
//            cv::Scharr( grayTrgPyr[level], grayTrgGradXPyr[level], dataType, 1, 0, scale, delta, cv::BORDER_DEFAULT );
//            cv::Sobel( grayTrgPyr[level], grayTrgGradXPyr[level], dataType, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );

            // Compute the gradient in y
//            grayTrgGradYPyr[level] = cv::Mat(cv::Size( grayTrgPyr[level].cols, grayTrgPyr[level].rows), grayTrgPyr[level].type() );
//            cv::Scharr( grayTrgPyr[level], grayTrgGradYPyr[level], dataType, 0, 1, scale, delta, cv::BORDER_DEFAULT );
//            cv::Sobel( grayTrgPyr[level], grayTrgGradYPyr[level], dataType, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

//            cv::Mat imgNormalizedDepth;
//            imagePyramid[level].convertTo(imgNormalizedDepth, CV_32FC1,1./maxDepth);

            // Compute the gradient in x
//            cv::Scharr( depthTrgPyr[level], depthTrgGradXPyr[level], dataType, 1, 0, scale, delta, cv::BORDER_DEFAULT );

            // Compute the gradient in y
//            cv::Scharr( depthTrgPyr[level], depthTrgGradYPyr[level], dataType, 0, 1, scale, delta, cv::BORDER_DEFAULT );

//            cv::imshow("DerX", grayTrgGradXPyr[level]);
//            cv::imshow("DerY", grayTrgGradYPyr[level]);
//            cv::waitKey(0);
//            cv::imwrite(mrpt::format("/home/edu/gradX_%d.png",level), grayTrgGradXPyr[level]);
//            cv::imwrite(mrpt::format("/home/edu/gray_%d.png",level), grayTrgPyr[level]);
        }
    };

    /*! Sets the source (Intensity+Depth) frame.*/
    void setSourceFrame(cv::Mat &imgRGB,cv::Mat &imgDepth)
    {
        //Create a float auxialiary image from the imput image
        //        cv::Mat imgGrayFloat;
        cv::cvtColor(imgRGB, graySrc, CV_RGB2GRAY);
        graySrc.convertTo(graySrc, CV_32FC1, 1./255 );

        //Compute image pyramids for the grayscale and depth images
        buildPyramid(graySrc, graySrcPyr, nPyrLevels);
        buildPyramidRange(imgDepth, depthSrcPyr, nPyrLevels);
        //        buildPyramid(imgGrayFloat,gray0Pyr,numnLevels,true);
        //        buildPyramid(imgDepth,depth0Pyr,numnLevels,false);
    };

    /*! Sets the source (Intensity+Depth) frame. Depth image is ignored*/
    void setTargetFrame(cv::Mat &imgRGB,cv::Mat &imgDepth)
    {
        //Create a float auxialiary image from the imput image
        //        cv::Mat imgGrayFloat;
        cv::cvtColor(imgRGB, grayTrg, CV_RGB2GRAY);
        grayTrg.convertTo(grayTrg, CV_32FC1, 1./255 );

        //Compute image pyramids for the grayscale and depth images
        //        buildPyramid(imgGrayFloat,gray1Pyr,numnLevels,true);
        buildPyramid(grayTrg, grayTrgPyr, nPyrLevels);
        buildPyramidRange(imgDepth, depthTrgPyr, nPyrLevels);

        //Compute image pyramids for the gradients images
        buildGradientPyramids();

//        cv::imshow("GradX ", grayTrgGradXPyr[0]);
//        cv::imshow("GradX_d ", depthTrgGradXPyr[0]);
//        cv::waitKey(0);
    };

    /*! Huber weight for robust estimation. */
    inline double weightHuber(const double &error, const double &regularization)
    {
        double error_abs = abs(error);
        if(error_abs < regularization)
            return 1.;
        else
            return sqrt(2*regularization*error_abs-regularization*regularization) / error_abs;
    };

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double computePhotoICP_SqError(const int &pyramidLevel, const Eigen::Matrix4d &poseGuess, double varPhoto = pow(3./255,2), double varDepth = 0.0001, costFuncType method = PHOTO_CONSISTENCY)
    {
        double error2 = 0.0; // Squared error

        int nRows = graySrcPyr[pyramidLevel].rows;
        int nCols = graySrcPyr[pyramidLevel].cols;

        double scaleFactor = 1.0/pow(2,pyramidLevel);
        double fx = cameraMatrix(0,0)*scaleFactor;
        double fy = cameraMatrix(1,1)*scaleFactor;
        double ox = cameraMatrix(0,2)*scaleFactor;
        double oy = cameraMatrix(1,2)*scaleFactor;
        double inv_fx = 1./fx;
        double inv_fy = 1./fy;

        double varianceRegularization = 1; // 63%
        double stdDevReg = sqrt(varianceRegularization);
        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance
        double stdDevPhoto_inv = 1./stdDevPhoto;
        double stdDevDepth_inv = 1./stdDevDepth;

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int r=0;r<nRows;r++)
        {
            for (int c=0;c<nCols;c++)
            {
                //                int i = nCols*r+c; //vector index

                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector4d point3D;
                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
                if(minDepth < point3D(2) && point3D(2) < maxDepth) //Compute the jacobian only for the valid points
                {
                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
                    point3D(1)=(r - oy) * point3D(2) * inv_fy;
                    point3D(3)=1;

                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector4d transformedPoint3D = poseGuess*point3D;
                    //                    Eigen::Vector3f  transformedPoint3D = poseGuess.block(0,0,3,3)*point3D;

                    //Project the 3D point to the 2D plane
                    double inv_transformedPz = 1.0/transformedPoint3D(2);
                    double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
                    transformed_c = (transformedPoint3D(0) * fx)*inv_transformedPz + ox; //transformed x (2D)
                    transformed_r = (transformedPoint3D(1) * fy)*inv_transformedPz + oy; //transformed y (2D)
                    int transformed_r_int = round(transformed_r);
                    int transformed_c_int = round(transformed_c);

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
                        (transformed_c_int>=0 && transformed_c_int < nCols) )
                    {
                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            float pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            float pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            double photoDiff = pixel2 - pixel1;
                            weight_photo = weightHuber(photoDiff,stdDevPhoto)*stdDevPhoto_inv;
                            double weightedErrorPhoto = weight_photo * photoDiff;
                            // Apply M-estimator weighting
//                            if(photoDiff2 > varianceRegularization)
//                            {
////                                float photoDiff2_norm = sqrt(photoDiff2);
////                                double weight = sqrt(2*stdDevReg*photoDiff2_norm-varianceRegularization) / photoDiff2_norm;
//                                photoDiff2 = 2*stdDevReg*sqrt(photoDiff2)-varianceRegularization;
//                            }
                            error2 += weightedErrorPhoto*weightedErrorPhoto;

                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            float depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            float depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            double depthDiff = depth2 - depth1;
                            weight_depth = weightHuber(depthDiff,stdDevDepth)*stdDevPhoto_inv;
                            double weightedErrorDepth = weight_depth * depthDiff;
                            error2 += weightedErrorDepth*weightedErrorDepth;
                        }
                    }
                }
            }
        }

        return error2;
    }

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessianAndGradient(const int &pyramidLevel,
                               Eigen::Matrix4d &poseGuess,
                               costFuncType method = PHOTO_CONSISTENCY)
    {
        int nRows = graySrcPyr[pyramidLevel].rows;
        int nCols = graySrcPyr[pyramidLevel].cols;

        double scaleFactor = 1.0/pow(2,pyramidLevel);
        double fx = cameraMatrix(0,0)*scaleFactor;
        double fy = cameraMatrix(1,1)*scaleFactor;
        double ox = cameraMatrix(0,2)*scaleFactor;
        double oy = cameraMatrix(1,2)*scaleFactor;
        double inv_fx = 1./fx;
        double inv_fy = 1./fy;

        hessian = Eigen::Matrix<double,6,6>::Zero();
        gradient = Eigen::Matrix<double,6,1>::Zero();

        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance
//        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

//        double varianceRegularization = 1; // 63%
//        double stdDevReg = sqrt(varianceRegularization);
        double stdDevPhoto_inv = 1./stdDevPhoto;
        double stdDevDepth_inv = 1./stdDevDepth;

        if(visualizeIterations)
        {
            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                warped_source_grayImage = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                warped_source_depthImage = cv::Mat::zeros(nRows,nCols,depthSrcPyr[pyramidLevel].type());
        }

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int r=0;r<nRows;r++)
        {
            for (int c=0;c<nCols;c++)
            {
                //                int i = nCols*r+c; //vector index

                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector4d point3D;
                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
//            cout << "z " << depthSrcPyr[pyramidLevel].at<float>(r,c) << endl;
                if(minDepth < point3D(2) && point3D(2) < maxDepth) //Compute the jacobian only for the valid points
                {
                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
                    point3D(1)=(r - oy) * point3D(2) * inv_fy;
                    point3D(3)=1;

                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector4d transformedPoint3D = poseGuess*point3D;
                    //                    Eigen::Vector3f  transformedPoint3D = poseGuess.block(0,0,3,3)*point3D;

                    //Project the 3D point to the 2D plane
                    double inv_transformedPz = 1.0/transformedPoint3D(2);
                    double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
                    transformed_c = (transformedPoint3D(0) * fx)*inv_transformedPz + ox; //transformed x (2D)
                    transformed_r = (transformedPoint3D(1) * fy)*inv_transformedPz + oy; //transformed y (2D)
                    int transformed_r_int = round(transformed_r);
                    int transformed_c_int = round(transformed_c);

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
                        (transformed_c_int>=0 && transformed_c_int < nCols) )
                    {
                        //Compute the pixel jacobian
                        Eigen::Matrix<double,2,6> jacobianWarpRt;

                        //Derivative with respect to x
                        jacobianWarpRt(0,0)=fx*inv_transformedPz;
                        jacobianWarpRt(1,0)=0;

                        //Derivative with respect to y
                        jacobianWarpRt(0,1)=0;
                        jacobianWarpRt(1,1)=fy*inv_transformedPz;

                        //Derivative with respect to z
                        jacobianWarpRt(0,2)=-fx*transformedPoint3D(0)*inv_transformedPz*inv_transformedPz;
                        jacobianWarpRt(1,2)=-fy*transformedPoint3D(1)*inv_transformedPz*inv_transformedPz;

                        Eigen::Vector3d rotatedPoint3D = transformedPoint3D.block(0,0,3,1) - poseGuess.block(0,3,3,1); // TODO: make more efficient

                        //Derivative with respect to \lambda_x
                        jacobianWarpRt(0,3)=-fx*rotatedPoint3D(1)*transformedPoint3D(0)*inv_transformedPz*inv_transformedPz;
                        jacobianWarpRt(1,3)=-fy*(rotatedPoint3D(2)+rotatedPoint3D(1)*transformedPoint3D(1)*inv_transformedPz)*inv_transformedPz;

                        //Derivative with respect to \lambda_y
                        jacobianWarpRt(0,4)= fx*(rotatedPoint3D(2)+rotatedPoint3D(0)*transformedPoint3D(0)*inv_transformedPz)*inv_transformedPz;
                        jacobianWarpRt(1,4)= fy*rotatedPoint3D(0)*transformedPoint3D(1)*inv_transformedPz*inv_transformedPz;

                        //Derivative with respect to \lambda_z
                        jacobianWarpRt(0,5)=-fx*rotatedPoint3D(1)*inv_transformedPz;
                        jacobianWarpRt(1,5)= fy*rotatedPoint3D(0)*inv_transformedPz;

//                        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance
//                        float photoDiff2, depthDiff2;
                        float pixel1, pixel2, depth1, depth2;
                        double weightedErrorPhoto, weightedErrorDepth;
                        Eigen::Matrix<double,1,6> jacobianPhoto, jacobianDepth;

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
//                        cout << "pixel1 " << pixel1 << " pixel2 " << pixel2 << endl;
                            double photoDiff = pixel2 - pixel1;
                            weight_photo = weightHuber(photoDiff,stdDevPhoto)*stdDevPhoto_inv;
//                            if(photoDiff2 > varianceRegularization)
//                                weight_photo = sqrt(2*stdDevReg*abs(photoDiff2)-varianceRegularization) / photoDiff2_norm;
                            weightedErrorPhoto = weight_photo * photoDiff;

                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<double,1,2> target_imgGradient;
                            target_imgGradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
                            target_imgGradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            jacobianPhoto = weight_photo * target_imgGradient*jacobianWarpRt;
//                        cout << "target_imgGradient " << target_imgGradient << endl;

                        if(visualizeIterations)
                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;

                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            if(minDepth < depth2 && depth2 < maxDepth) // Make sure this point has depth (not a NaN)
                            {
                                depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                                double depthDiff = depth2 - depth1;
                                weight_depth = weightHuber(depthDiff,stdDevDepth)*stdDevPhoto_inv;
                                weightedErrorDepth = weight_depth * depthDiff;

                                //Depth jacobian:
                                //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                                Eigen::Matrix<double,1,2> target_depthGradient;
                                target_depthGradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                                target_depthGradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);

                                Eigen::Matrix<double,1,6> jacobianRt_z;
                                jacobianRt_z << 0,0,1,transformedPoint3D(1),-transformedPoint3D(0),0;
                                jacobianDepth = weight_depth * (target_depthGradient*jacobianWarpRt-jacobianRt_z);
//                            cout << "jacobianDepth " << jacobianDepth << " weightedErrorDepth " << weightedErrorDepth << endl;

                                if(visualizeIterations)
                                    warped_source_depthImage.at<float>(transformed_r_int,transformed_c_int) = depth1;
                            }
                        }

                        //Assign the pixel residual and jacobian to its corresponding row
#if ENABLE_OPENMP
#pragma omp critical
#endif
                        {
                            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                            {
                                // Photometric component
//                                hessian += jacobianPhoto.transpose()*jacobianPhoto / varPhoto;
                                hessian += jacobianPhoto.transpose()*jacobianPhoto;
                                gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
                            }
                            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                                if(minDepth < depth2 && depth2 < maxDepth) // Make sure this point has depth (not a NaN)
                            {
                                // Depth component (Plane ICL like)
                                hessian += jacobianDepth.transpose()*jacobianDepth;
                                gradient += jacobianDepth.transpose()*weightedErrorDepth;
                            }
                        }

                    }
                }
            }
        }
    }


    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double calcPhotoICPError_robot(const int &pyramidLevel, const Eigen::Matrix4d &poseGuess, const Eigen::Matrix4d &poseCamRobot, costFuncType method = PHOTO_CONSISTENCY)
    {
        double error2 = 0.0; // Squared error

        int nRows = graySrcPyr[pyramidLevel].rows;
        int nCols = graySrcPyr[pyramidLevel].cols;

        double scaleFactor = 1.0/pow(2,pyramidLevel);
        double fx = cameraMatrix(0,0)*scaleFactor;
        double fy = cameraMatrix(1,1)*scaleFactor;
        double ox = cameraMatrix(0,2)*scaleFactor;
        double oy = cameraMatrix(1,2)*scaleFactor;
        double inv_fx = 1./fx;
        double inv_fy = 1./fy;

        Eigen::Matrix4d poseCamRobot_inv = poseCamRobot.inverse();

        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance

        for (int r=0;r<nRows;r++)
        {
            for (int c=0;c<nCols;c++)
            {
                //                int i = nCols*r+c; //vector index

                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector4d point3D;
                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
                if(minDepth < point3D(2) && point3D(2) < maxDepth) //Compute the jacobian only for the valid points
                {
                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
                    point3D(1)=(r - oy) * point3D(2) * inv_fy;
                    point3D(3)=1;

                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector4d transformedPoint3D = poseCamRobot_inv*poseGuess*poseCamRobot*point3D;
                    //                    Eigen::Vector3f  transformedPoint3D = poseGuess.block(0,0,3,3)*point3D;

                    //Project the 3D point to the 2D plane
                    double inv_transformedPz = 1.0/transformedPoint3D(2);
                    double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
                    transformed_c = (transformedPoint3D(0) * fx)*inv_transformedPz + ox; //transformed x (2D)
                    transformed_r = (transformedPoint3D(1) * fy)*inv_transformedPz + oy; //transformed y (2D)
                    int transformed_r_int = round(transformed_r);
                    int transformed_c_int = round(transformed_c);

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
                        (transformed_c_int>=0 && transformed_c_int < nCols) )
                    {
                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            double pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            double pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float photoDiff = pixel2 - pixel1;
                            double weightedErrorPhoto = weight_photo * photoDiff;

                            error2 += weightedErrorPhoto*weightedErrorPhoto;
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            double depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            double depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float depthDiff = depth2 - depth1;
                            double weightedErrorDepth = weight_depth * depthDiff;

                            error2 += weightedErrorDepth*weightedErrorDepth;
                        }
                    }
                }
            }
        }

        return error2;
    }


    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessianGradient_robot( const int &pyramidLevel,
                                    Eigen::Matrix4d &poseGuess, // The relative pose of the robot between the two frames
                                    const Eigen::Matrix4d &poseCamRobot, // The pose of the camera wrt to the Robot (fixed beforehand through calibration) // Maybe calibration can be computed at the same time
                                    costFuncType method = PHOTO_CONSISTENCY)
    {
        int nRows = graySrcPyr[pyramidLevel].rows;
        int nCols = graySrcPyr[pyramidLevel].cols;

        double scaleFactor = 1.0/pow(2,pyramidLevel);
        double fx = cameraMatrix(0,0)*scaleFactor;
        double fy = cameraMatrix(1,1)*scaleFactor;
        double ox = cameraMatrix(0,2)*scaleFactor;
        double oy = cameraMatrix(1,2)*scaleFactor;
        double inv_fx = 1./fx;
        double inv_fy = 1./fy;

        hessian = Eigen::Matrix<double,6,6>::Zero();
        gradient = Eigen::Matrix<double,6,1>::Zero();

        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance
        //        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

        Eigen::Matrix4d poseCamRobot_inv = poseCamRobot.inverse();
//    std::cout << "poseCamRobot \n" << poseCamRobot << std::endl;

        for (int r=0;r<nRows;r++)
        {
            for (int c=0;c<nCols;c++)
            {
                //                int i = nCols*r+c; //vector index

                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector4d point3D;
                point3D(2) = depthSrcPyr[pyramidLevel].at<float>(r,c);
                if(minDepth < point3D(2) && point3D(2) < maxDepth) //Compute the jacobian only for the valid points
                {
                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
                    point3D(1)=(r - oy) * point3D(2) * inv_fy;
                    point3D(3)=1;

                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector4d point3D_robot = poseCamRobot*point3D;
                    Eigen::Vector4d transformedPoint3D = poseCamRobot_inv*poseGuess*point3D_robot;
                    //                    Eigen::Vector3f  transformedPoint3D = poseGuess.block(0,0,3,3)*point3D;
//                std::cout << "transformedPoint3D " << transformedPoint3D.transpose() << std::endl;

                    //Project the 3D point to the 2D plane
                    double inv_transformedPz = 1.0/transformedPoint3D(2);
                    double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
                    transformed_c = (transformedPoint3D(0) * fx)*inv_transformedPz + ox; //transformed x (2D)
                    transformed_r = (transformedPoint3D(1) * fy)*inv_transformedPz + oy; //transformed y (2D)
                    int transformed_r_int = round(transformed_r);
                    int transformed_c_int = round(transformed_c);
//                std::cout << "transformed_r_int " << transformed_r_int << " transformed_c_int " << transformed_c_int << std::endl;

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( (transformed_r_int>=0 && transformed_r_int < nRows) &&
                        (transformed_c_int>=0 && transformed_c_int < nCols) )
                    {
                        //Compute the pixel jacobian
                        Eigen::Matrix<double,3,6> jacobianT36;
                        jacobianT36.block(0,0,3,3) = Eigen::Matrix<double,3,3>::Identity();
                        Eigen::Vector3d rotatedPoint3D = transformedPoint3D.block(0,0,3,1) - poseGuess.block(0,3,3,1); // TODO: make more efficient
                        jacobianT36.block(0,3,3,3) = -skew( rotatedPoint3D );
                        jacobianT36 = poseCamRobot_inv.block(0,0,3,3) * jacobianT36;

                        Eigen::Matrix<double,2,3> jacobianProj23;
                        //Derivative with respect to x
                        jacobianProj23(0,0)=fx*inv_transformedPz;
                        jacobianProj23(1,0)=0;
                        //Derivative with respect to y
                        jacobianProj23(0,1)=0;
                        jacobianProj23(1,1)=fy*inv_transformedPz;
                        //Derivative with respect to z
                        jacobianProj23(0,2)=-fx*transformedPoint3D(0)*inv_transformedPz*inv_transformedPz;
                        jacobianProj23(1,2)=-fy*transformedPoint3D(1)*inv_transformedPz*inv_transformedPz;

                        Eigen::Matrix<double,2,6> jacobianWarpRt = jacobianProj23 * jacobianT36;

                        double weightedErrorPhoto, weightedErrorDepth;
                        float pixel1, pixel2, depth1, depth2;
                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float photoDiff = pixel2 - pixel1;
                            weightedErrorPhoto = weight_photo * photoDiff;

                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<double,1,2> target_imgGradient;
                            target_imgGradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);//(i);
                            target_imgGradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            Eigen::Matrix<double,1,6> jacobianPhoto = weight_photo*target_imgGradient*jacobianWarpRt;

                            // Photometric component
                            hessian += jacobianPhoto.transpose()*jacobianPhoto;
                            gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float depthDiff = depth2 - depth1;
                            weightedErrorDepth = weight_depth * depthDiff;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<double,1,2> target_depthGradient;
                            target_depthGradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(r,c);
                            target_depthGradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(r,c);
                            Eigen::Matrix<double,1,6> jacobianRt_z;
                            jacobianRt_z << 0,0,1,transformedPoint3D(1),-transformedPoint3D(0),0;
                            Eigen::Matrix<double,1,6> jacobianDepth = weight_depth*(target_depthGradient*jacobianWarpRt-jacobianRt_z);

                            // Depth component (Plane ICL like)
                            hessian += jacobianDepth.transpose()*jacobianDepth;
                            gradient += jacobianDepth.transpose()*weightedErrorDepth;
                        }

//                        if(visualizeIterations)
//                        {
//                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
////                            warped_source_depthImage.at<float>(transformed_r_int,transformed_c_int) = depth1;
//                        }
                    }
                }
            }
        }
    }

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double calcPhotoICPError_sphere(const int &pyramidLevel,
                                        Eigen::Matrix4d &poseGuess, // The relative pose of the robot between the two frames
                                        costFuncType method = PHOTO_CONSISTENCY )
    {
        double error2 = 0.0;

        int phi_res = graySrcPyr[pyramidLevel].rows;
        int theta_res = graySrcPyr[pyramidLevel].cols;
        double phi_fov = 2*cameraMatrix(2,0); // Half of the vertical FOV in radians

        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance
//        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(int r=0;r<phi_res;r++)
        {
            double phi = r*(2*phi_fov/phi_res);
            for(int c=0;c<theta_res;c++)
            {
                double theta = c*(2*PI/theta_res);

                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector4d point3D;
                double depth = depthSrcPyr[pyramidLevel].at<float>(r,c);
                if(minDepth < depth && depth < maxDepth) //Compute the jacobian only for the valid points
                {
                    point3D(0) = depth*sin(phi);
                    point3D(1) = -depth*cos(phi)*sin(theta);
                    point3D(2) = -depth*cos(phi)*cos(theta);
                    point3D(3) = 1;

                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector4d transformedPoint3D = poseGuess*point3D;

                    //Project the 3D point to the S2 sphere
                    double phi_trg = asin(point3D(2)/depth);
                    double theta_trg = atan2(-point3D(1),-point3D(2));
                    int transformed_r_int = round(phi_trg*phi_res/phi_fov + phi_res/2);
                    int transformed_c_int = round(theta_trg*theta_res/(2*PI));

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( (transformed_r_int>=0 && transformed_r_int < phi_res) )
                    {
                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            double pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            double pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float photoDiff = pixel2 - pixel1;
                            double weightedErrorPhoto = weight_photo * photoDiff;
                            error2 += weightedErrorPhoto*weightedErrorPhoto;
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            double depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            double depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float depthDiff = depth2 - depth1;
                            double weightedErrorDepth = weight_depth * depthDiff;
                            error2 += weightedErrorDepth*weightedErrorDepth;
                        }
                    }
                }
            }
        }

        return error2;
    }

    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method to build the Hessian and Gradient.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    void calcHessianAndGradient_sphere(const int &pyramidLevel,
                                            Eigen::Matrix4d &poseGuess, // The relative pose of the robot between the two frames
                                            costFuncType method
                                            )
    {
        int phi_res = graySrcPyr[pyramidLevel].rows;
        int theta_res = graySrcPyr[pyramidLevel].cols;
        double phi_fov = 2*cameraMatrix(2,0); // Half of the vertical FOV in radians

        hessian = Eigen::Matrix<double,6,6>::Zero();
        gradient = Eigen::Matrix<double,6,1>::Zero();

        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance

//        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for(int r=0;r<phi_res;r++)
        {
            double phi = r*(2*phi_fov/phi_res);
            for(int c=0;c<theta_res;c++)
            {
                double theta = c*(2*PI/theta_res);

                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector4d point3D;
                double depth = depthSrcPyr[pyramidLevel].at<float>(r,c);
                if(minDepth < depth && depth < maxDepth) //Compute the jacobian only for the valid points
                {
                    point3D(0) = depth*sin(phi);
                    point3D(1) = -depth*cos(phi)*sin(theta);
                    point3D(2) = -depth*cos(phi)*cos(theta);
                    point3D(3) = 1;

                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector4d transformedPoint3D = poseGuess*point3D;

                    //Project the 3D point to the S2 sphere
                    double phi_trg = asin(point3D(2)/depth);
                    double theta_trg = atan2(-point3D(1),-point3D(2));
                    int transformed_r_int = round(phi_trg*phi_res/phi_fov + phi_res/2);
                    int transformed_c_int = round(theta_trg*theta_res/(2*PI));

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
                    if( (transformed_r_int>=0 && transformed_r_int < phi_res) )
                    {
                        //Compute the pixel jacobian
                        Eigen::Matrix<double,3,6> jacobianT36;
                        jacobianT36.block(0,0,3,3) = Eigen::Matrix<double,3,3>::Identity();
                        Eigen::Vector3d rotatedPoint3D = transformedPoint3D.block(0,0,3,1) - poseGuess.block(0,3,3,1); // TODO: make more efficient
                        jacobianT36.block(0,3,3,3) = -skew( rotatedPoint3D );

                        Eigen::Matrix<double,2,3> jacobianProj23;
                        //Derivative with respect to x
                        jacobianProj23(0,0) = 0;
                        jacobianProj23(1,0) = 1 / (depth*sqrt(1 - pow(point3D(0)/depth,2)));
                        //Derivative with respect to y
                        jacobianProj23(0,1) = 1 / (point3D(2)*sqrt(1 + pow(point3D(1)/point3D(2),2)));
                        jacobianProj23(1,1) = 0;
                        //Derivative with respect to z
                        jacobianProj23(0,2) = -point3D(1) / (point3D(2)*point3D(2)*sqrt(1 + pow(point3D(1)/point3D(2),2)));
                        jacobianProj23(1,2) = 0;

                        Eigen::Matrix<double,2,6> jacobianWarpRt;

                        jacobianWarpRt = jacobianProj23 * jacobianT36;

//                        double weight_photo = 1./stdDevPhoto, weight_depth = 1./stdDevDepth; // The weights should include the Huber estimator plus the information of the covariance
                        double weightedErrorPhoto, weightedErrorDepth;
                        Eigen::Matrix<double,1,6> jacobianPhoto, jacobianDepth;

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            double pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            double pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            float photoDiff = pixel2 - pixel1;
                            weightedErrorPhoto = weight_photo * photoDiff;

                            //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<double,1,2> target_imgGradient;
                            target_imgGradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(r,c);//(i);
                            target_imgGradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(r,c);
                            jacobianPhoto = weight_photo*target_imgGradient*jacobianWarpRt;
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            double depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            double depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            float depthDiff = depth2 - depth1;
                            weightedErrorDepth = weight_depth * depthDiff;

                            //Depth jacobian:
                            //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                            Eigen::Matrix<double,1,2> target_depthGradient;
                            target_depthGradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(r,c);
                            target_depthGradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(r,c);
                            Eigen::Matrix<double,1,6> jacobianRt_z;
                            jacobianRt_z << 0,0,1,transformedPoint3D(1),-transformedPoint3D(0),0;
                            jacobianDepth = weight_depth*(target_depthGradient*jacobianWarpRt-jacobianRt_z);
                        }

                        //Assign the pixel residual and jacobian to its corresponding row
#if ENABLE_OPENMP
#pragma omp critical
#endif
                        {
                            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                            {
                                // Photometric component
                                hessian += jacobianPhoto.transpose()*jacobianPhoto;
                                gradient += jacobianPhoto.transpose()*weightedErrorPhoto;
                            }
                            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                            {
                                // Depth component (Plane ICL like)
                                hessian += jacobianDepth.transpose()*jacobianDepth;
                                gradient += jacobianDepth.transpose()*weightedErrorDepth;
                            }
                            //                              if(visualizeIterations)
                            //                                warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
                        }

//                        if(visualizeIterations)
//                        {
//                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
//                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = intensity1;
//                        }
                    }
                }
            }
        }
    }

    /*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
      * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
      * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution. */
    void alignFrames(const Eigen::Matrix4d pose_guess = Eigen::Matrix4d::Identity(), costFuncType method = PHOTO_CONSISTENCY)
    {
        Eigen::Matrix4d pose_estim_temp, pose_estim = pose_guess;
        for(int pyramidLevel = nPyrLevels-1; pyramidLevel >= 0; pyramidLevel--)
        {
            int nRows = graySrcPyr[pyramidLevel].rows;
            int nCols = graySrcPyr[pyramidLevel].cols;

            double lambda = 0.001; // Levenberg-Marquardt (LM) lambda
            double step = 10; // Update step
            unsigned LM_maxIters = 3;

            int it = 0, maxIters = 10;
            double tol_residual = pow(10,-10);
            double tol_update = pow(10,-10);
            Eigen::Matrix<double,6,1> update_pose; update_pose(0,0) = 1;
            double error = computePhotoICP_SqError(pyramidLevel, pose_estim, method);
            double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            std::cout << "Level " << pyramidLevel << " error " << error << std::endl;
#endif
            while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
            {
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                cv::TickMeter tm;tm.start();
#endif

                calcHessianAndGradient(pyramidLevel, pose_estim, method);
                assert(hessian.rank() == 6); // Make sure that the problem is observable

                // Compute the pose update
                update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
                pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose)).getHomogeneousMatrixVal() * pose_estim;
            cout << "hessian \n" << hessian << endl;
            cout << "gradient \n" << gradient.transpose() << endl;
            cout << "update_pose \n" << update_pose.transpose() << endl;

                double new_error = computePhotoICP_SqError(pyramidLevel, pose_estim_temp, method);
                diff_error = error - new_error;
                cout << "diff_error \n" << diff_error << endl;
                if(diff_error > 0)
                {
                cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
                    lambda /= step;
                    pose_estim = pose_estim_temp;
                    error = new_error;
                    it = it+1;
                }
                else
                {
                    unsigned LM_it = 0;
                    while(LM_it < LM_maxIters && diff_error < 0)
                    {
                        lambda = lambda * step;

                        update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
                        pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose)).getHomogeneousMatrixVal() * pose_estim;
                        double new_error = computePhotoICP_SqError(pyramidLevel, pose_estim_temp, method);
                        diff_error = error - new_error;
                    cout << "diff_error LM \n" << diff_error << endl;

                        if(diff_error > 0)
                        {
                        cout << "pose_estim \n" << pose_estim << "\n pose_estim_temp \n" << pose_estim_temp << endl;
                            pose_estim = pose_estim_temp;
                            error = new_error;
                            it = it+1;
                        }
                        LM_it = LM_it + 1;
                    }
                }

#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                tm.stop(); std::cout << "Iterations " << it << " time = " << tm.getTimeSec() << " sec." << std::endl;
#endif

                cout << "visualizeIterations \n" << endl;

                if(visualizeIterations)
                {
                    cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,grayTrgPyr[pyramidLevel].type());
                    cv::absdiff(grayTrgPyr[pyramidLevel], warped_source_grayImage, imgDiff);
//                std::cout << "imgDiff " << imgDiff.at<float>(20,20) << " " << grayTrgPyr[pyramidLevel].at<float>(20,20) << " " << warped_source_grayImage.at<float>(20,20) << std::endl;
                    cv::imshow("optimize::imgDiff", imgDiff);

                    cv::imshow("orig", grayTrgPyr[pyramidLevel]);
                    cv::imshow("warp", warped_source_grayImage);

                    cv::Mat depthDiff = cv::Mat::zeros(nRows,nCols,depthTrgPyr[pyramidLevel].type());
                    cv::absdiff(depthTrgPyr[pyramidLevel], warped_source_depthImage, depthDiff);
                    cv::imshow("depthDiff", depthDiff);

                    cv::waitKey(0);
                }
            }
        }

        relPose = pose_estim;
    }

};

#endif
