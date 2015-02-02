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
 * Author: Eduardo Fernandez-Moral
 */

#ifndef DOWNSAMPLERGBD_H
#define DOWNSAMPLERGBD_H

#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//#include <pcl/filters/filter.h>
//#include <pcl/filters/fast_bilateral.h>
//#include <pcl/filters/bilateral.h>

#define ENABLE_OPENMP_MULTITHREADING 0

class DownsampleRGBD
{
  public:
//    DownsampleRGBD(int step = 2);
//
//    virtual ~DownsampleRGBD();

DownsampleRGBD(int step = 2) :
    downsamplingStep(step),
    minDepth(0.3),
    maxDepth(5.0)
{
//std::cout << "DownsampleRGBD::Ctor...\n";
  pointCloudAvailable=false;
  pointCloudAvailable=false;
  intensityImageAvailable=false;
  rgbImageAvailable=false;
  depthImageAvailable=false;
}

    inline void setDownsamplingStep(const int step){downsamplingStep = step;};

    inline void setMinMaxDepth(const int minD, const int maxD){minDepth = minD; maxDepth = maxD;};


cv::Mat downsampleDepth(cv::Mat &source)
{
  if(!depthImageAvailable)
  {
    assert(source.cols % downsamplingStep == 0 && source.rows % downsamplingStep);

    IplImage aux(source);
    IplImage *source_ = &aux;

  // declare a destination IplImage object with correct size, depth and channels
    IplImage *destination = cvCreateImage( cvSize((int)(source_->width/downsamplingStep), (int)(source_->height/downsamplingStep) ), source_->depth, source_->nChannels );

    //use cvResize to resize source_ to a destination image
    cvResize(source_, destination);

    //m_depthImage = cv::Mat(destination);
    m_depthImage = cv::cvarrToMat(destination);

    depthImageAvailable = true;
  }

  return m_depthImage;
}

cv::Mat downsampleIntensity(cv::Mat &source)
{
  if(!intensityImageAvailable)
  {
    assert(source.cols % downsamplingStep == 0 && source.rows % downsamplingStep);

    IplImage aux(source);
    IplImage *source_ = &aux;

  // declare a destination IplImage object with correct size, depth and channels
    IplImage *destination = cvCreateImage( cvSize((int)(source_->width/downsamplingStep), (int)(source_->height/downsamplingStep) ), source_->depth, source_->nChannels );

    //use cvResize to resize source_ to a destination image
    cvResize(source_, destination);

    //m_intensityImage = cv::Mat(destination);
    m_depthImage = cv::cvarrToMat(destination);
  }

  return m_intensityImage;
}

inline cv::Mat downsampleRGB(cv::Mat &source)
//cv::Mat DownsampleRGBD::buildPyramid(cv::Mat & img,std::vector<cv::Mat>& pyramid,int levels,bool applyBlur)
{
  if(!rgbImageAvailable)
  {
    assert(source.cols % downsamplingStep == 0 && source.rows % downsamplingStep);

    IplImage aux(source);
    IplImage *source_ = &aux;

  // declare a destination IplImage object with correct size, depth and channels
    IplImage *destination = cvCreateImage( cvSize((int)(source_->width/downsamplingStep), (int)(source_->height/downsamplingStep) ), source_->depth, source_->nChannels );

    //use cvResize to resize source_ to a destination image
    cvResize(source_, destination);

    //m_rgbImage = cv::Mat(destination);
    m_rgbImage = cv::cvarrToMat(destination);

//    pyrDown(InputArray src, OutputArray dst, const Size& dstsize=Size(), int borderType=BORDER_DEFAULT )
//
//    //Create space for all the images
//    pyramid.resize(levels);
//
//    double factor = 1;
//    for(int level=0;level<levels;level++)
//    {
//        //Create an auxiliar image of factor times the size of the original image
//        cv::Mat imgAux;
//        if(level!=0)
//        {
//            cv::resize(img,imgAux,cv::Size(0,0),factor,factor);
//        }
//        else
//        {
//            imgAux = img;
//        }
//
//        //Blur the resized image with different filter size depending on the current pyramid level
//        if(applyBlur)
//        {
//            #if ENABLE_GAUSSIAN_BLUR
//            if(blurFilterSize[level]>0)
//            {
//                cv::GaussianBlur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]),3);
//                cv::GaussianBlur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]),3);
//            }
//            #elif ENABLE_BOX_FILTER_BLUR
//            if(blurFilterSize[level]>0)
//            {
//                cv::blur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]));
//                cv::blur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]));
//            }
//            #endif
//        }
//
//        //Assign the resized image to the current level of the pyramid
//        pyramid[level]=imgAux;
//
//        factor = factor/2;
//    }
    rgbImageAvailable = true;
  }

  return m_rgbImage;
}

//DownsampleRGBD::DownsampleRGBD(const int width_, const int height_, const int step) :
//  m_DownsampledPointCloudPtr(new pcl::PointCloud<pcl::PointXYZRGBA>)
//{
//    width = width_;
//    height = height_;
//    downsamplingStep = step;
//    m_DownsampledPointCloudPtr->points.resize(width*height/(downsamplingStep*downsamplingStep));
//    m_DownsampledPointCloudPtr->width = width/downsamplingStep;
//    m_DownsampledPointCloudPtr->height = height/downsamplingStep;
//    m_DownsampledPointCloudPtr->is_dense = false;
//}

//DownsampleRGBD::~DownsampleRGBD(){}

//void DownsampleRGBD::setDownsamplingStep(const int step)
//{
//  assert()
//    width = width_;
//    height = height_;
//    downsamplingStep=step;
//    m_DownsampledPointCloudPtr->points.resize(width*height/(downsamplingStep*downsamplingStep));
//    m_DownsampledPointCloudPtr->width = width/downsamplingStep;
//    m_DownsampledPointCloudPtr->height = height/downsamplingStep;
//    m_DownsampledPointCloudPtr->is_dense = false;
//}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsamplePointCloud( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloudPtr)
{
//std::cout << "DownsampleRGBD::downsamplePointCloud...\n";

  if(!pointCloudAvailable)
  {
//  std::cout << "Downsampling\n";
    m_DownsampledPointCloudPtr.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
    m_DownsampledPointCloudPtr->points.resize(pointCloudPtr->size()/(downsamplingStep*downsamplingStep));
    m_DownsampledPointCloudPtr->width = pointCloudPtr->width/downsamplingStep;
    m_DownsampledPointCloudPtr->height = pointCloudPtr->height/downsamplingStep;
    m_DownsampledPointCloudPtr->is_dense = false;

    static int j;j=0;
    std::vector<double> xV;
    std::vector<double> yV;
    std::vector<double> zV;
//  std::cout << "Downsampling1\n";

    #if ENABLE_OPENMP_MULTITHREADING
    #pragma omp parallel for
    #endif
    for(unsigned r=0;r<pointCloudPtr->height;r=r+downsamplingStep)
    {
        for(unsigned c=0;c<pointCloudPtr->width;c=c+downsamplingStep)
        {
            unsigned nPoints=0;
            xV.resize(downsamplingStep*downsamplingStep);
            yV.resize(downsamplingStep*downsamplingStep);
            zV.resize(downsamplingStep*downsamplingStep);

            unsigned centerPatch = (r+downsamplingStep/2)*pointCloudPtr->width+c+downsamplingStep/2;

            for(unsigned r2=r;r2<r+downsamplingStep;r2++)
            {
                for(unsigned c2=c;c2<c+downsamplingStep;c2++)
                {
                    //Check if the point has valid data
                    if(pcl_isfinite (pointCloudPtr->points[r2*pointCloudPtr->width+c2].x) &&
//                       pcl_isfinite (pointCloudPtr->points[r2*pointCloudPtr->width+c2].y) &&
//                       pcl_isfinite (pointCloudPtr->points[r2*pointCloudPtr->width+c2].z) &&
                       minDepth < pointCloudPtr->points[r2*pointCloudPtr->width+c2].z &&
                       pointCloudPtr->points[r2*pointCloudPtr->width+c2].z < maxDepth)
                    {
                        //Create a vector with the x, y and z coordinates of the square region
                        xV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].x;
                        yV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].y;
                        zV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].z;

                        nPoints++;
                    }
                }
            }

            //Check if there are points in the region
            if(nPoints>0)
            {
                xV.resize(nPoints);
                yV.resize(nPoints);
                zV.resize(nPoints);

                //Compute the mean 3D point
                std::sort(xV.begin(),xV.end());
                std::sort(yV.begin(),yV.end());
                std::sort(zV.begin(),zV.end());

                pcl::PointXYZRGBA point;
                point.x=xV[nPoints/2];
                point.y=yV[nPoints/2];
                point.z=zV[nPoints/2];

                point.r = pointCloudPtr->points[centerPatch].r;
                point.g = pointCloudPtr->points[centerPatch].g;
                point.b = pointCloudPtr->points[centerPatch].b;

                //Set the mean point as the representative point of the region
                #if ENABLE_OPENMP_MULTITHREADING
                #pragma omp critical
                #endif
                {
                  m_DownsampledPointCloudPtr->points[j]=point;
                  j++;
                }
            }
            else
            {
                //Set a nan point to keep the m_DownsampledPointCloudPtr organised
                #if ENABLE_OPENMP_MULTITHREADING
                #pragma omp critical
                #endif
                {
                  m_DownsampledPointCloudPtr->points[j] = pointCloudPtr->points[centerPatch];
                  j++;
                }
            }
        }
    }
//  std::cout << "Downsampling2\n";

    pointCloudAvailable = true;
  }
  return m_DownsampledPointCloudPtr;
}


//void DownsampleRGBD::downsamplePointCloudColor( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloudPtr)
//{
//    static int j;j=0;
//    std::vector<double> xV;
//    std::vector<double> yV;
//    std::vector<double> zV;
////    std::vector<double> rV;
////    std::vector<double> gV;
////    std::vector<double> bV;
//
//    #if ENABLE_OPENMP_MULTITHREADING
//    #pragma omp parallel for
//    #endif
//    for(int r=0;r<pointCloudPtr->height;r=r+downsamplingStep)
//    {
//        for(int c=0;c<pointCloudPtr->width;c=c+downsamplingStep)
//        {
//            int nPoints=0;
//            xV.resize(downsamplingStep*downsamplingStep);
//            yV.resize(downsamplingStep*downsamplingStep);
//            zV.resize(downsamplingStep*downsamplingStep);
////            rV.resize(downsamplingStep*downsamplingStep);
////            gV.resize(downsamplingStep*downsamplingStep);
////            bV.resize(downsamplingStep*downsamplingStep);
//
//            for(int r2=r;r2<r+downsamplingStep;r2++)
//            {
//                for(int c2=c;c2<c+downsamplingStep;c2++)
//                {
//                    //Check if the point has valid data
//                    if(pcl_isfinite (pointCloudPtr->points[r2*pointCloudPtr->width+c2].x) &&
////                       pcl_isfinite (pointCloudPtr->points[r2*pointCloudPtr->width+c2].y) &&
////                       pcl_isfinite (pointCloudPtr->points[r2*pointCloudPtr->width+c2].z) &&
//                       minDepth < pointCloudPtr->points[r2*pointCloudPtr->width+c2].z &&
//                       pointCloudPtr->points[r2*pointCloudPtr->width+c2].z < maxDepth)
//                    {
//                        //Create a vector with the x, y and z coordinates of the square region and RGB info
//                        xV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].x;
//                        yV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].y;
//                        zV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].z;
////                        rV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].r;
////                        gV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].g;
////                        bV[nPoints]=pointCloudPtr->points[r2*pointCloudPtr->width+c2].b;
//
//                        nPoints++;
//                    }
//                }
//            }
//
//            int cell_center_idx = (r+downsamplingStep/2)*width+c+downsamplingStep/2;
//
//            if(nPoints>0)
//            {
//                xV.resize(nPoints);
//                yV.resize(nPoints);
//                zV.resize(nPoints);
////                rV.resize(nPoints);
////                gV.resize(nPoints);
////                bV.resize(nPoints);
//
//                //Compute the mean 3D point and mean RGB value
//                std::sort(xV.begin(),xV.end());
//                std::sort(yV.begin(),yV.end());
//                std::sort(zV.begin(),zV.end());
////                std::sort(rV.begin(),rV.end());
////                std::sort(gV.begin(),gV.end());
////                std::sort(bV.begin(),bV.end());
//
//                pcl::PointXYZRGBA point;
//                point.x=xV[nPoints/2];
//                point.y=yV[nPoints/2];
//                point.z=zV[nPoints/2];
////                point.r=rV[nPoints/2];
////                point.g=gV[nPoints/2];
////                point.b=bV[nPoints/2];
//                point.r=pointCloudPtr->points[cell_center_idx].r;
//                point.g=pointCloudPtr->points[cell_center_idx].g;
//                point.b=pointCloudPtr->points[cell_center_idx].b;
//
//                //Set the mean point as the representative point of the region
//                #if ENABLE_OPENMP_MULTITHREADING
//                #pragma omp critical
//                #endif
//                {
//                  m_DownsampledPointCloudPtr->points[j]=point;
//                  j++;
//                }
//            }
//            else
//            {
//                //Set a nan point to keep the m_DownsampledPointCloudPtr organised
//                #if ENABLE_OPENMP_MULTITHREADING
//                #pragma omp critical
//                #endif
//                {
//                  m_DownsampledPointCloudPtr->points[j]=pointCloudPtr->points[cell_center_idx]; // std::numeric_limits<float>::quiet_NaN ();
//                  j++;
//                }
//            }
//        }
//    }
////  std::cout << "m_DownsampledPointCloudPtr size " << m_DownsampledPointCloudPtr->size() << std::endl;
//}


//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr DownsampleRGBD::getFilteredPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& pointCloudPtr)
//{
//  if(!filteredCloudAvailable)
//  {
//    pcl::FastBilateralFilter<pcl::PointXYZRGBA> filter;
//    //filter.setSigmaS (sigma_s);
//    //filter.setSigmaR (sigma_r);
//    //filter.setEarlyDivision (early_division);
//    filter.setInputCloud(pointCloudPtr);
//    m_FilteredPointCloudPtr.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
//    filter.filter(*m_FilteredPointCloudPtr);
//  }
//  return m_FilteredPointCloudPtr;
//}

//    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getFilteredPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& pointCloudPtr);

//    cv::Mat downsampleDepth(cv::Mat &source);
//
//    cv::Mat downsampleIntensity(cv::Mat &source);
//
//    cv::Mat downsampleRGB(cv::Mat &source);

  protected:

  private:

    int downsamplingStep;
    double minDepth, maxDepth;
    bool pointCloudAvailable;
    bool filteredCloudAvailable;
    bool intensityImageAvailable;
    bool rgbImageAvailable;
    bool depthImageAvailable;

    /*!RGB image*/
    cv::Mat m_rgbImage;

    /*!Intensity image (grayscale version of the RGB image)*/
    cv::Mat m_intensityImage;

    /*!Depth image*/
    cv::Mat m_depthImage;

    /*!Downsampled point cloud*/
//    pcl::PointCloud<pcl::PointXYZ>::Ptr mDownsampledPointCloudPtr;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_DownsampledPointCloudPtr;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_FilteredPointCloudPtr;

//
//    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsamplePointCloud( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloudPtr);

};

#endif // DOWNSAMPLERGBD_H
