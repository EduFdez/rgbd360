#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <eigen_extensions/eigen_extensions.h>

#define MAX_MULT 1.3
#define MIN_MULT 0.7

namespace clams
{
  typedef Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic> DepthMat;  
  typedef boost::shared_ptr<DepthMat> DepthMatPtr;
  typedef boost::shared_ptr<const DepthMat> DepthMatConstPtr;
}

#endif // TYPEDEFS_H
