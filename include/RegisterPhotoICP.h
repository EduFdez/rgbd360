/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#define ENABLE_OPENMP 0
#define ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS 1

#include <mrpt/slam/CSimplePointsMap.h>
#include <mrpt/slam/CObservation2DRangeScan.h>
#include <mrpt/slam/CICP.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/poses/CPosePDF.h>
#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/gui.h>
#include <mrpt/math/utils.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp> //TickMeter
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace Eigen;

// Return a diagonal matrix where the values of the diagonal are assigned from the input vector
template<typename typedata, int nRows, int nCols>
Matrix<typedata,nRows,nCols> getDiagonalMatrix(const Matrix<typedata,nRows,nCols> &matrix_generic)
{
    assert(nRows == nCols);

    Matrix<typedata,nRows,nCols> m_diag = Matrix<typedata,nRows,nCols>::Zero();
    for(size_t i=0; i < nRows; i++)
        m_diag(i,i) = matrix_generic(i,i);

    return m_diag;
}

class RegisterPhotoICP_RGBD
{
    /*!Camera matrix (intrinsic parameters).*/
    Eigen::Matrix3f cameraMatrix;

    /*! The reference intensity image */
    Mat graySrc;

    /*! The target intensity image */
    Mat grayTrg;

    /*! The reference depth image */
    Mat depthSrc;

    /*! The target depth image */
    Mat depthTrg;

    /*! The relative pose between source and target RGB-D images */
    Matrix4d relPose;

    /*! The Hessian matrix of the optimization problem. At the solution it represents the inverse of the covariance matrix of the relative pose */
    Matrix<double,6,6> hessian;

    /*! The Gradient vector of the optimization problem. At the solution it should be zero */
    Matrix<double,6,1> gradient;

//    /* Current iteration at the current optimization level.*/
//    int iter;

    /*! Minimum allowed depth to consider a depth pixel valid.*/
    float minDepth;

    /*! Maximum allowed depth to consider a depth pixel valid.*/
    float maxDepth;

    /*! Depth component gain. This variable is used to scale the depth values so that depth components are similar to intensity values.*/
    float depthComponentGain;

    /*! Number of pyramid levels.*/
    int nPyrLevels;

    /*! Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numpyramidLevels' levels.*/
    std::vector<cv::Mat> graySrcPyr, grayTrgPyr, depthSrcPyr, depthTrgPyr, grayTrgGradXPyr, grayTrgGradYPyr, depthTrgGradXPyr, depthTrgGradYPyr;

    /*! Enable the visualization of the optimization process (only for debug).*/
    bool visualizeIterations;

    /*! Optimization method (cost function). The options are: 0=Photoconsistency, 1=Depth consistency (ICP like), 2= A combination of photo-depth consistency */
    enum costFuncType {PHOTO_CONSISTENCY, DEPTH_CONSISTENCY, PHOTO_DEPTH} method;

  public:

    denseRGBDAlignment() :
        minDepth(0.3),
        maxDepth(8.0),
        nPyrLevels(4)
    {
    };

    /*! Sets the minimum depth distance (m) to consider a certain pixel valid.*/
    void setMinDepth(float minD)
    {
        minDepth = minD;
    };

    /*! Sets the maximum depth distance (m) to consider a certain pixel valid.*/
    void setMaxDepth(float maxD)
    {
        maxDepth = maxD;
    };

    /*! Sets the 3x3 matrix of (pinhole) camera intrinsic parameters used to obtain the 3D colored point cloud from the RGB and depth images.*/
    void setCameraMatrix(Eigen::Matrix3f & camMat)
    {
        cameraMatrix = camMat;
    };

    /*! Returns the optimal SE(3) rigid transformation matrix between the source and target frame.
     * This method has to be called after calling the alignFrames() method.*/
    Matrix4d getOptimalPose()
    {
        return relPose;
    }

    /*! Returns the Hessian (the information matrix).*/
    Matrix<double,6,6> getHessian()
    {
        return Hessian;
    }

    /*! Returns the Gradient (the information matrix).*/
    Matrix<double,6,1> getGradient()
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
            pyrDown( img, imgAux, Size( img.cols/2, img.rows/2 ) );

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
        pyramid[0] = img;

        for(int level=1; level < nLevels; level++)
        {
            //Create an auxiliar image of factor times the size of the original image
            cv::Mat imgAux(Size( img.cols/2, img.rows/2 ), CV_MAT_TYPE(img.type));
            for(unsigned r=0; r < img.rows; r+=2)
                for(unsigned c=0; c < img.cols; c+=2)
                {
                    float avDepth = 0.f;
                    unsigned nValidPixels = 0;
                    for(unsigned i=0; i < 2; i++)
                        for(unsigned j=0; j < 2; j++)
                        {
                            float z = pyramid[level-1].at<float>(r+i,c+j);
                            if(z > minDepth && z < maxDepth)
                            {
                                avDepth += z;
                                ++nValidPixels;
                            }
                        }
                    //pyramid[level](r/2,c/2) = avDepth / nValidPixels;
                    imgAux(r/2,c/2) = avDepth / nValidPixels;
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

    /*! Compute the gradient images for each pyramid level. */
//    void buildGradientPyramids(std::vector<cv::Mat>& graySrcPyr,std::vector<cv::Mat>& grayTrgGradXPyr,std::vector<cv::Mat>& grayTrgGradYPyr)
    void buildGradientPyramids()
    {
        //Compute image gradients
        double scale = 1;
        double delta = 0;
        int dataType = CV_32FC1;

        //Create space for all the derivatives images
        grayTrgGradXPyr.resize(graySrcPyr.size());
        grayTrgGradYPyr.resize(graySrcPyr.size());

        depthTrgGradXPyr.resize(graySrcPyr.size());
        depthTrgGradYPyr.resize(graySrcPyr.size());

        for(unsigned level=0;level < graySrcPyr.size();level++)
        {
            // Compute the gradient in x
            cv::Scharr( grayTrgPyr[level], grayTrgGradXPyr[level], dataType, 1, 0, scale, delta, cv::BORDER_DEFAULT );

            // Compute the gradient in y
            cv::Scharr( grayTrgPyr[level], grayTrgGradYPyr[level], dataType, 0, 1, scale, delta, cv::BORDER_DEFAULT );

//            cv::Mat imgNormalizedDepth;
//            imagePyramid[level].convertTo(imgNormalizedDepth, CV_32FC1,1./maxDepth);

            // Compute the gradient in x
            cv::Scharr( depthTrgPyr[level], depthTrgGradXPyr[level], dataType, 1, 0, scale, delta, cv::BORDER_DEFAULT );

            // Compute the gradient in y
            cv::Scharr( depthTrgPyr[level], depthTrgGradYPyr[level], dataType, 0, 1, scale, delta, cv::BORDER_DEFAULT );
        }
    };

    /*! Sets the source (Intensity+Depth) frame.*/
    void setSourceFrame(cv::Mat & imgRGB,cv::Mat & imgDepth)
    {
        //Create a float auxialiary image from the imput image
//        cv::Mat imgGrayFloat;
        imgRGB.convertTo(graySrc, CV_32FC1, 1./255 );

        //Compute image pyramids for the grayscale and depth images
        buildPyramid(graySrc, graySrcPyr, nPyrLevels);
        buildPyramid(imgDepth, depthSrcPyr, nPyrLevels);
//        buildPyramid(imgGrayFloat,gray0Pyr,numnLevels,true);
//        buildPyramid(imgDepth,depth0Pyr,numnLevels,false);
    };

    /*! Sets the source (Intensity+Depth) frame. Depth image is ignored*/
    void setTargetFrame(cv::Mat & imgRGB,cv::Mat & imgDepth)
    {
        //Create a float auxialiary image from the imput image
//        cv::Mat imgGrayFloat;
        imgRGB.convertTo(grayTrg, CV_32FC1, 1./255 );

        //Compute image pyramids for the grayscale and depth images
//        buildPyramid(imgGrayFloat,gray1Pyr,numnLevels,true);
        buildPyramid(grayTrg, grayTrgPyr, nPyrLevels);
        buildPyramid(imgDepth, depthTrgPyr, nPyrLevels);

        //Compute image pyramids for the gradients images
        buildGradientPyramids();
    };


    /*! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
        This is done following the work in:
        Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
        in Computer Vision Workshops (ICCV Workshops), 2011. */
    double computePhotoICP_SqError(const int &pyramidLevel, const Eigen::Matrix4d &poseGuess)
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

        hessian = Matrix<double,6,6>::Zero();
        gradient = Matrix<double,6,1>::Zero();

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
                    if((transformed_r_int>=0 && transformed_r_int < nRows) &
                       (transformed_c_int>=0 && transformed_c_int < nCols))
                    {
                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            double pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            double pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            double photoDiff = pixel2 - pixel1;
                            error2 += photoDiff*photoDiff;

                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            double depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            double depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            double depthDiff = pixel2 - pixel1;
                            error2 += depthDiff*depthDiff;
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
    void computeHessianAndGradient(const int &pyramidLevel,
                                      Eigen::Matrix4d &poseGuess,
                                        //costFuncType method,
                                      cv::Mat & warped_source_grayImage)
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

        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

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
                    if((transformed_r_int>=0 && transformed_r_int < nRows) &
                       (transformed_c_int>=0 && transformed_c_int < nCols))
                    {
                        Eigen::Vector4d rotatedPoint3D = transformedPoint3D - poseGuess.block(0,3,3,1); // TODO: make more efficient

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

                            //Derivative with respect to \lambda_x
                            jacobianWarpRt(0,3)=fx*transformedPoint3D(0)*rotatedPoint3D(1)*inv_transformedPz;
                            jacobianWarpRt(1,3)=fy*(rotatedPoint3D(2)+rotatedPoint3D(0)*transformedPoint3D(0)*inv_transformedPz)*inv_transformedPz;

                            //Derivative with respect to \lambda_y
                            jacobianWarpRt(0,4)=-fx*(rotatedPoint3D(2)+rotatedPoint3D(1)*transformedPoint3D(1)*inv_transformedPz)*inv_transformedPz;
                            jacobianWarpRt(1,4)=fy*transformedPoint3D(1)*rotatedPoint3D(0)*inv_transformedPz;

                            //Derivative with respect to \lambda_z
                            jacobianWarpRt(0,5)=-fx*rotatedPoint3D(1)*inv_transformedPz;
                            jacobianWarpRt(1,5)=fy*rotatedPoint3D(0)*inv_transformedPz;

                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            double pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            double pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame

                          //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                          Eigen::Matrix<double,1,2> target_imgGradient;
                          target_imgGradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(r,c);//(i);
                          target_imgGradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(r,c);
                          Eigen::Matrix<double,1,6> jacobian = target_imgGradient*jacobianWarpRt;
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            double depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            double depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);

                          //Depth jacobian:
                          //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                          Eigen::Matrix<double,1,2> target_depthGradient;
                          target_depthGradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(i);
                          target_depthGradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(i);
                          Eigen::Matrix<double,1,6> jacobianRt_z;
                          jacobianRt_z << 0,0,1,transformedPoint3D(1),-transformedPoint3D(0),0;
                          Eigen::Matrix<double,1,6> jacobianDepth = depthComponentGain*(target_depthGradient*jacobianWarpRt-jacobianRt_z);
                        }

                          //Assign the pixel residual and jacobian to its corresponding row
                          #if ENABLE_OPENMP
                          #pragma omp critical
                          #endif
                          {
                            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                            {
                              // Photometric component
                              hessian += jacobian.transpose()*jacobian;
                              gradient += jacobian.transpose()*(pixel2 - pixel1);
                            }
                            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                            {
                                // Depth component (Plane ICL like)
                              hessian += jacobianDepth.transpose()*jacobianDepth;
                              gradient += jacobianDepth.transpose()*depthComponentGain*(depth2 - depth1);
                            }
//                              if(visualizeIterations)
//                                warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
                          }

                          if(visualizeIterations)
                          {
                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = intensity1;
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
    double computePhotoICP_SqError_wrtRobot(const int &pyramidLevel, const Eigen::Matrix4d &poseGuess, const Eigen::Matrix4d &poseCamRobot)
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
                    if((transformed_r_int>=0 && transformed_r_int < nRows) &
                       (transformed_c_int>=0 && transformed_c_int < nCols))
                    {
                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            double pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            double pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame
                            double photoDiff = pixel2 - pixel1;
                            error2 += photoDiff*photoDiff;

                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            double depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            double depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);
                            double depthDiff = pixel2 - pixel1;
                            error2 += depthDiff*depthDiff;
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
    void computeHessianAndGradient_wrtRobot(const int &pyramidLevel,
                                            Eigen::Matrix4d &poseGuess, // The relative pose of the robot between the two frames
                                            const Eigen::Matrix4d &poseCamRobot // The pose of the camera wrt to the Robot (fixed beforehand through calibration) // Maybe calibration can be computed at the same time
                                            // costFuncType method)
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

        hessian = Matrix<double,6,6>::Zero();
        gradient = Matrix<double,6,1>::Zero();

        poseCamRobot_inv = poseCamRobot.inverse();

        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

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
                    Eigen::Vector4d point3D_robot = poseCamRobot*point3D;
                    Eigen::Vector4d transformedPoint3D = poseCamRobot_inv*poseGuess*point3D_robot;
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
                    if((transformed_r_int>=0 && transformed_r_int < nRows) &
                       (transformed_c_int>=0 && transformed_c_int < nCols))
                    {
                        //Eigen::Vector4d rotatedPoint3D = transformedPoint3D - poseGuess.block(0,3,3,1); // TODO: make more efficient

                        //Compute the pixel jacobian
                        Eigen::Matrix<double,3,6> jacobianT36;
                        jacobianT.block(0,0,3,3) = Eigen::Matrix<double,3,3>::Identity();
                        jacobianT.block(0,3,3,3) = -skew( poseGuess.block(0,0,3,3) * point3D_robot.block(0,0,3,1) );
                        jacobianT = poseCamRobot_inv.block(0,3,3,3) * jacobianT;


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

                        Eigen::Matrix<double,2,6> jacobianWarpRt;

jacobianWarpRt = jacobianProj23 * jacobianT36;


                        if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the pixel values that will be used to compute the pixel residual
                            double pixel1 = graySrcPyr[pyramidLevel].at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
                            double pixel2 = grayTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame

                          //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                          Eigen::Matrix<double,1,2> target_imgGradient;
                          target_imgGradient(0,0) = grayTrgGradXPyr[pyramidLevel].at<float>(r,c);//(i);
                          target_imgGradient(0,1) = grayTrgGradYPyr[pyramidLevel].at<float>(r,c);
                          Eigen::Matrix<double,1,6> jacobian = target_imgGradient*jacobianWarpRt;
                        }
                        if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                        {
                            //Obtain the depth values that will be used to the compute the depth residual
                            double depth1 = depthSrcPyr[pyramidLevel].at<float>(r,c); // Depth value of the pixel(r,c) of the warped frame 1 (target)
                            double depth2 = depthTrgPyr[pyramidLevel].at<float>(transformed_r_int,transformed_c_int);

                          //Depth jacobian:
                          //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                          Eigen::Matrix<double,1,2> target_depthGradient;
                          target_depthGradient(0,0) = depthTrgGradXPyr[pyramidLevel].at<float>(i);
                          target_depthGradient(0,1) = depthTrgGradYPyr[pyramidLevel].at<float>(i);
                          Eigen::Matrix<double,1,6> jacobianRt_z;
                          jacobianRt_z << 0,0,1,transformedPoint3D(1),-transformedPoint3D(0),0;
                          Eigen::Matrix<double,1,6> jacobianDepth = depthComponentGain*(target_depthGradient*jacobianWarpRt-jacobianRt_z);
                        }

                          //Assign the pixel residual and jacobian to its corresponding row
                          #if ENABLE_OPENMP
                          #pragma omp critical
                          #endif
                          {
                            if(method == PHOTO_CONSISTENCY || method == PHOTO_DEPTH)
                            {
                              // Photometric component
                              hessian += jacobian.transpose()*jacobian;
                              gradient += jacobian.transpose()*(pixel2 - pixel1);
                            }
                            if(method == DEPTH_CONSISTENCY || method == PHOTO_DEPTH)
                            {
                                // Depth component (Plane ICL like)
                              hessian += jacobianDepth.transpose()*jacobianDepth;
                              gradient += jacobianDepth.transpose()*depthComponentGain*(depth2 - depth1);
                            }
//                              if(visualizeIterations)
//                                warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
                          }

                          if(visualizeIterations)
                          {
                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = intensity1;
                          }
                    }
                }
            }
        }
    }

    /*! Search for the best alignment of a pair of RGB-D frames based on photoconsistency and depthICP.
     * This pose is obtained from an optimization process using Levenberg-Marquardt which is maximizes the photoconsistency and depthCOnsistency
     * between the source and target frames. This process is performed sequentially on a pyramid of image with increasing resolution. */
    void alignFrames(const Matrix4d pose_guess = Matrix4d::Identity())
    {
        Matrix4d pose_estim_temp, pose_estim = pose_guess;
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
            Matrix<double,6,1> update_pose; update_pose(0,0) = 1;
            double error = computePhotoICP_SqError(pyramidLevel, pose_estim);
            double diff_error = error;
#if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            cout << "Level " << pyramidLevel << " error " << error << endl;
#endif
            while(it < maxIters && update_pose.norm() > tol_update && diff_error > tol_residual) // The LM optimization stops either when the max iterations is reached, or when the alignment converges (error or pose do not change)
            {
                #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                cv::TickMeter tm;tm.start();
                #endif
                cv::Mat warped_source_grayImage;
                if(visualizeIterations)
                    warped_source_grayImage = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());

                computeHessianAndGradient(pyramidLevel, pose_estim, warped_source_grayImage);
                assert(hessian.rank() == 6); // Make sure that the problem is observable

                // Compute the pose update
                update_pose = -(hessian + lambda*getDiagonalMatrix(hessian)).inverse() * gradient;
                pose_estim_temp = mrpt::poses::CPose3D::exp(mrpt::math::CArrayNumeric<double,6>(update_pose)).getHomogeneousMatrixVal() * pose_estim;

                double new_error = computePhotoICP_SqError(pyramidLevel, pose_estim_temp);
                diff_error = error - new_error;
                if(diff_error > 0)
                {
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
                        double new_error = computePhotoICP_SqError(pyramidLevel, pose_estim_temp);
                        diff_error = error - new_error;

                        if(diff_error > 0)
                        {
                            pose_estim = pose_estim_temp;
                            error = new_error;
                            it = it+1;
                        }
                        LM_it = LM_it + 1;
                    }
                }

                #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                tm.stop(); std::cout << "Iteration time = " << tm.getTimeSec() << " sec." << std::endl;
                #endif

                if(visualizeIterations)
                {
                    cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,graySrcPyr[pyramidLevel].type());
                    cv::absdiff(graySrcPyr[pyramidLevel], warped_source_grayImage, imgDiff);
                    cv::imshow("optimize::imgDiff", imgDiff);
                    cv::waitKey(0);
                }
            }
        }
    }

};


///! Compute the residuals and the jacobians for each iteration of the dense alignemnt method.
//    This is done following the work in:
//    Direct iterative closest point for real-time visual odometry. Tykkala, Tommi and Audras, Cédric and Comport, Andrew I.
//    in Computer Vision Workshops (ICCV Workshops), 2011. */
//void computeHessianAndGradient(cv::Mat &source_grayImg,
//                                  cv::Mat &source_depthImg,
//                                  cv::Mat &target_grayImg,
//                                  cv::Mat &source_gradXImg,
//                                  cv::Mat &source_gradYImg,
//                                  const int &pyramidLevel,
//                                  Eigen::Matrix4d &poseGuess,
////                                      Eigen::Matrix<double,Eigen::Dynamic,1> &residuals,
////                                      Eigen::Matrix<double,Eigen::Dynamic,6> &jacobians,
//                                  cv::Mat & warped_source_grayImage)
//{
//    int nRows = source_grayImg.rows;
//    int nCols = source_grayImg.cols;

//    double scaleFactor = 1.0/pow(2,pyramidLevel);
//    double fx = cameraMatrix(0,0)*scaleFactor;
//    double fy = cameraMatrix(1,1)*scaleFactor;
//    double ox = cameraMatrix(0,2)*scaleFactor;
//    double oy = cameraMatrix(1,2)*scaleFactor;
//    double inv_fx = 1./fx;
//    double inv_fy = 1./fy;

//    hessian = Matrix<double,6,6>::Zero();
//    gradient = Matrix<double,6,1>::Zero();

//    #if ENABLE_OPENMP
//    #pragma omp parallel for
//    #endif
//    for (int r=0;r<nRows;r++)
//    {
//        for (int c=0;c<nCols;c++)
//        {
////                int i = nCols*r+c; //vector index

//            //Compute the 3D coordinates of the pij of the source frame
//            Eigen::Vector4d point3D;
//            point3D(2)=source_depthImg.at<float>(r,c);
//            if(minDepth < point3D(2) && point3D(2) < maxDepth) //Compute the jacobian only for the valid points
//            {
//                point3D(0)=(c - ox) * point3D(2) * inv_fx;
//                point3D(1)=(r - oy) * point3D(2) * inv_fy;
//                point3D(3)=1;

//                //Transform the 3D point using the transformation matrix Rt
//                Eigen::Vector4d transformedPoint3D = poseGuess*point3D;
////                    Eigen::Vector3f  transformedPoint3D = poseGuess.block(0,0,3,3)*point3D;

//                //Project the 3D point to the 2D plane
//                double inv_transformedPz = 1.0/transformedPoint3D(2);
//                double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
//                transformed_c = (transformedPoint3D(0) * fx)*inv_transformedPz + ox; //transformed x (2D)
//                transformed_r = (transformedPoint3D(1) * fy)*inv_transformedPz + oy; //transformed y (2D)
//                int transformed_r_int = round(transformed_r);
//                int transformed_c_int = round(transformed_c);

//                //Asign the intensity value to the warped image and compute the difference between the transformed
//                //pixel of the source frame and the corresponding pixel of target frame. Compute the error function
//                if((transformed_r_int>=0 && transformed_r_int < nRows) &
//                   (transformed_c_int>=0 && transformed_c_int < nCols))
//                {
//                    //Obtain the pixel values that will be used to compute the pixel residual
//                    double pixel1 = source_grayImg.at<float>(r,c); // Intensity value of the pixel(r,c) of source frame
//                    double pixel2 = target_grayImg.at<float>(transformed_r_int,transformed_c_int); //Intensity value of the pixel(r,c) of the warped target (reference) frame

//                    Eigen::Vector4d rotatedPoint3D = transformedPoint3D - poseGuess.block(0,3,3,1);


//                    //Compute the pixel jacobian
//                    Eigen::Matrix<double,2,6> jacobianWarpRt;

//                        //Derivative with respect to x
//                        jacobianWarpRt(0,0)=fx*inv_transformedPz;
//                        jacobianWarpRt(1,0)=0;

//                        //Derivative with respect to y
//                        jacobianWarpRt(0,1)=0;
//                        jacobianWarpRt(1,1)=fy*inv_transformedPz;

//                        //Derivative with respect to z
//                        jacobianWarpRt(0,2)=-fx*transformedPoint3D(0)*inv_transformedPz*inv_transformedPz;
//                        jacobianWarpRt(1,2)=-fy*transformedPoint3D(1)*inv_transformedPz*inv_transformedPz;

//                        //Derivative with respect to \lambda_x
//                        jacobianWarpRt(0,3)=fx*transformedPoint3D(0)*rotatedPoint3D(1)*inv_transformedPz;
//                        jacobianWarpRt(1,3)=fy*(rotatedPoint3D(2)+rotatedPoint3D(0)*transformedPoint3D(0)*inv_transformedPz)*inv_transformedPz;

//                        //Derivative with respect to \lambda_y
//                        jacobianWarpRt(0,4)=-fx*(rotatedPoint3D(2)+rotatedPoint3D(1)*transformedPoint3D(1)*inv_transformedPz)*inv_transformedPz;
//                        jacobianWarpRt(1,4)=fy*transformedPoint3D(1)*rotatedPoint3D(0)*inv_transformedPz;

//                        //Derivative with respect to \lambda_z
//                        jacobianWarpRt(0,5)=-fx*rotatedPoint3D(1)*inv_transformedPz;
//                        jacobianWarpRt(1,5)=fy*rotatedPoint3D(0)*inv_transformedPz;

//                      //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
//                      Eigen::Matrix<double,1,2> target_imgGradient;
//                      target_imgGradient(0,0) = source_gradXImg.at<float>(r,c);//(i);
//                      target_imgGradient(0,1) = source_gradYImg.at<float>(r,c);
//                      Eigen::Matrix<double,1,6> jacobian = target_imgGradient*jacobianWarpRt;

//                      //Assign the pixel residual and jacobian to its corresponding row
//                      #if ENABLE_OPENMP
//                      #pragma omp critical
//                      #endif
//                      {
//                          hessian += jacobian.transpose()*jacobian;
//                          gradient += jacobian.transpose()*(pixel2 - pixel1);

////                              jacobians(i,0)=jacobian(0,0);
////                              jacobians(i,1)=jacobian(0,1);
////                              jacobians(i,2)=jacobian(0,2);
////                              jacobians(i,3)=jacobian(0,3);
////                              jacobians(i,4)=jacobian(0,4);
////                              jacobians(i,5)=jacobian(0,5);

////                              residuals(nCols*transformed_r_int+transformed_c_int,0) = pixel2 - pixel1;

//                          if(visualizeIterations)
//                            warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = pixel1;
//                      }
//                }
//            }
//        }
//    }
//}




/*
// ------------------------------------------------------
//				TestDenseRGBD_alignment
// ------------------------------------------------------
void TestDenseRGBD_alignment()
{
	CSimplePointsMap		m1,m2;
	float					runningTime;
	CICP::TReturnInfo		info;
	CICP					ICP;

	// Load scans:
	CObservation2DRangeScan	scan1;
	scan1.aperture = M_PIf;
	scan1.rightToLeft = true;
	scan1.validRange.resize( SCANS_SIZE );
	scan1.scan.resize(SCANS_SIZE);
	ASSERT_( sizeof(SCAN_RANGES_1) == sizeof(float)*SCANS_SIZE );

	memcpy( &scan1.scan[0], SCAN_RANGES_1, sizeof(SCAN_RANGES_1) );
	memcpy( &scan1.validRange[0], SCAN_VALID_1, sizeof(SCAN_VALID_1) );

	CObservation2DRangeScan	scan2 = scan1;
	memcpy( &scan2.scan[0], SCAN_RANGES_2, sizeof(SCAN_RANGES_2) );
	memcpy( &scan2.validRange[0], SCAN_VALID_2, sizeof(SCAN_VALID_2) );

	// Build the points maps from the scans:
	m1.insertObservation( &scan1 );
	m2.insertObservation( &scan2 );

#if MRPT_HAS_PCL
	cout << "Saving map1.pcd and map2.pcd in PCL format...\n";
	m1.savePCDFile("map1.pcd", false);
	m2.savePCDFile("map2.pcd", false);
#endif

	// -----------------------------------------------------
//	ICP.options.ICP_algorithm = icpLevenbergMarquardt;
//	ICP.options.ICP_algorithm = icpClassic;
	ICP.options.ICP_algorithm = (TICPAlgorithm)ICP_method;

	ICP.options.maxIterations			= 100;
	ICP.options.thresholdAng			= DEG2RAD(10.0f);
	ICP.options.thresholdDist			= 0.75f;
	ICP.options.ALFA					= 0.5f;
	ICP.options.smallestThresholdDist	= 0.05f;
	ICP.options.doRANSAC = false;

	ICP.options.dumpToConsole();
	// -----------------------------------------------------

	CPose2D		initialPose(0.8f,0.0f,(float)DEG2RAD(0.0f));

	CPosePDFPtr pdf = ICP.Align(
		&m1,
		&m2,
		initialPose,
		&runningTime,
		(void*)&info);

	printf("ICP run in %.02fms, %d iterations (%.02fms/iter), %.01f%% goodness\n -> ",
			runningTime*1000,
			info.nIterations,
			runningTime*1000.0f/info.nIterations,
			info.goodness*100 );

	cout << "Mean of estimation: " << pdf->getMeanVal() << endl<< endl;

	CPosePDFGaussian  gPdf;
	gPdf.copyFrom(*pdf);

	cout << "Covariance of estimation: " << endl << gPdf.cov << endl;

	cout << " std(x): " << sqrt( gPdf.cov(0,0) ) << endl;
	cout << " std(y): " << sqrt( gPdf.cov(1,1) ) << endl;
	cout << " std(phi): " << RAD2DEG(sqrt( gPdf.cov(2,2) )) << " (deg)" << endl;

	//cout << "Covariance of estimation (MATLAB format): " << endl << gPdf.cov.inMatlabFormat()  << endl;

	cout << "-> Saving reference map as scan1.txt" << endl;
	m1.save2D_to_text_file("scan1.txt");

	cout << "-> Saving map to align as scan2.txt" << endl;
	m2.save2D_to_text_file("scan2.txt");

	cout << "-> Saving transformed map to align as scan2_trans.txt" << endl;
	CSimplePointsMap m2_trans = m2;
	m2_trans.changeCoordinatesReference( gPdf.mean );
	m2_trans.save2D_to_text_file("scan2_trans.txt");


	cout << "-> Saving MATLAB script for drawing 2D ellipsoid as view_ellip.m" << endl;
	CMatrixFloat COV22 =  CMatrixFloat( CMatrixDouble( gPdf.cov ));
	COV22.setSize(2,2);
	CVectorFloat MEAN2D(2);
	MEAN2D[0] = gPdf.mean.x();
	MEAN2D[1] = gPdf.mean.y();
	{
		ofstream f("view_ellip.m");
		f << math::MATLAB_plotCovariance2D( COV22, MEAN2D, 3.0f);
	}


	// If we have 2D windows, use'em:
#if MRPT_HAS_WXWIDGETS
	if (!skip_window)
	{
		gui::CDisplayWindowPlots	win("ICP results");

		// Reference map:
		vector<float>   map1_xs, map1_ys, map1_zs;
		m1.getAllPoints(map1_xs,map1_ys,map1_zs);
		win.plot( map1_xs, map1_ys, "b.3", "map1");

		// Translated map:
		vector<float>   map2_xs, map2_ys, map2_zs;
		m2_trans.getAllPoints(map2_xs,map2_ys,map2_zs);
		win.plot( map2_xs, map2_ys, "r.3", "map2");

		// Uncertainty
		win.plotEllipse(MEAN2D[0],MEAN2D[1],COV22,3.0,"b2", "cov");

		win.axis(-1,10,-6,6);
		win.axis_equal();

		cout << "Close the window to exit" << endl;
		win.waitForKey();
	}
#endif


}*/
