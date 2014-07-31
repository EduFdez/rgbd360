#ifndef DISCRETE_DEPTH_DISTORTION_MODEL_H
#define DISCRETE_DEPTH_DISTORTION_MODEL_H

#include <assert.h>
#include <vector>
#include <boost/thread/shared_mutex.hpp>
#include "../eigen_extensions/eigen_extensions.h"
#include <stream_sequence/typedefs.h>

namespace clams
{

/* SharedLockable is based on the uncopyable boost::shared_mutex.
   This presents a dilemma when assigning or copy constructing.
   Right now, the state of the mutex in the other SharedLockable
   does not get copied to the target SharedLockable.
   I'm not sure yet if this is the desired behavior.
*/
  class SharedLockable
  {
  public:
    SharedLockable() {}
    //! Copy constructor will make a new shared_mutex that is unlocked.
    SharedLockable(const SharedLockable& other) {}
    //! Assignment operator will *not* copy the shared_mutex_ or the state of shared_mutex_ from other.
    SharedLockable& operator=(const SharedLockable& other) { return *this; }

    void lockWrite() { shared_mutex_.lock(); }
    void unlockWrite() { shared_mutex_.unlock(); }
    bool trylockWrite() { return shared_mutex_.try_lock(); }

    void lockRead() { shared_mutex_.lock_shared(); }
    void unlockRead() { shared_mutex_.unlock_shared(); }
    bool trylockRead() { return shared_mutex_.try_lock_shared(); }

  protected:
    //! For the first time ever, I'm tempted to make this mutable.
    //! It'd make user methods still be able to be const even if they are locking.
    boost::shared_mutex shared_mutex_;
  };

  class DiscreteFrustum : public SharedLockable
  {
  public:
    DiscreteFrustum(int smoothing = 1, double bin_depth = 1.0);
    //! z value, not distance to origin.
    //! thread-safe.
    void addExample(double ground_truth, double measurement);
    int index(float z) const;
    void undistort(float* z) const;
    void interpolatedUndistort(float* z) const;
    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

  protected:
    double max_dist_;
    int num_bins_;
    double bin_depth_;
    Eigen::VectorXf counts_;
    Eigen::VectorXf total_numerators_;
    Eigen::VectorXf total_denominators_;
    Eigen::VectorXf multipliers_;

    friend class DiscreteDepthDistortionModel;
  };

  class DiscreteDepthDistortionModel
  {
  public:
    typedef boost::shared_ptr<DiscreteDepthDistortionModel> Ptr;
    typedef boost::shared_ptr<const DiscreteDepthDistortionModel> ConstPtr;
    DiscreteDepthDistortionModel() {}
    ~DiscreteDepthDistortionModel();
    DiscreteDepthDistortionModel(int width, int height, int bin_width = 8, int bin_height = 6, double bin_depth = 2.0, int smoothing = 1);
    DiscreteDepthDistortionModel(const DiscreteDepthDistortionModel& other);
    DiscreteDepthDistortionModel& operator=(const DiscreteDepthDistortionModel& other);
//    void undistort(DepthMat* depth) const;
//    void undistort(Eigen::Matrix<double, 240, 320>* depth) const;
    void undistort(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>* depth) const;

    //! Returns the number of training examples it used from this pair.
    //! Thread-safe.
    size_t accumulate(const DepthMat& ground_truth, const DepthMat& measurement);
    void addExample(int v, int u, double ground_truth, double measurement);
    void save(const std::string& path) const;
    void load(const std::string& path);
    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);
    //! Saves images to the directory found at path.
    //! If path doesn't exist, it will be created.
    void visualize(const std::string& path) const;
    std::string status(const std::string& prefix = "") const;

    void downsampleParams(const int& downsamplingStep);

  protected:
    //! Image width.
    int width_;
    //! Image height.
    int height_;
    //! Width of each bin in pixels.
    int bin_width_;
    //! Height of each bin in pixels.
    int bin_height_;
    //! Depth of each bin in meters.
    double bin_depth_;
    int num_bins_x_;
    int num_bins_y_;
    //! frustums_[y][x]
    std::vector< std::vector<DiscreteFrustum*> > frustums_;

    void deleteFrustums();
    DiscreteFrustum& frustum(int y, int x);
    const DiscreteFrustum& frustum(int y, int x) const;
  };

}  // namespace clams

#endif // DISCRETE_DEPTH_DISTORTION_MODEL_H
