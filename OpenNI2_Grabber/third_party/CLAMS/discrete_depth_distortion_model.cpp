#include "include/clams/discrete_depth_distortion_model.h"

using namespace std;
using namespace Eigen;
namespace bfs = boost::filesystem;

namespace clams
{

  DiscreteFrustum::DiscreteFrustum(int smoothing, double bin_depth) :
    max_dist_(10),
    bin_depth_(bin_depth)
  {
    num_bins_ = ceil(max_dist_ / bin_depth_);
    counts_ = VectorXf::Ones(num_bins_) * smoothing;
    total_numerators_ = VectorXf::Ones(num_bins_) * smoothing;
    total_denominators_ = VectorXf::Ones(num_bins_) * smoothing;
    multipliers_ = VectorXf::Ones(num_bins_);
  }

  void DiscreteFrustum::addExample(double ground_truth, double measurement)
  {
    boost::unique_lock<boost::shared_mutex> ul(shared_mutex_);

    double mult = ground_truth / measurement;
    if(mult > MAX_MULT || mult < MIN_MULT)
      return;

    int idx = min(num_bins_ - 1, (int)floor(measurement / bin_depth_));
    assert(idx >= 0);

    total_numerators_(idx) += ground_truth * ground_truth;
    total_denominators_(idx) += ground_truth * measurement;
    ++counts_(idx);
    multipliers_(idx) = total_numerators_(idx) / total_denominators_(idx);
  }

  inline int DiscreteFrustum::index(float z) const
  {
    return min(num_bins_ - 1, (int)floor(z / bin_depth_));
  }

  inline void DiscreteFrustum::undistort(float* z) const
  {
    *z *= multipliers_.coeffRef(index(*z));
  }

  void DiscreteFrustum::interpolatedUndistort(float* z) const
  {
//  std::cout << "DiscreteFrustum bin_depth_ " << bin_depth_ << " " << multipliers_.size() << std::endl;
    int idx = index(*z);
    float start = bin_depth_ * idx;
    int idx1;
    if(*z - start < bin_depth_ / 2)
      idx1 = idx;
    else
      idx1 = idx + 1;
    int idx0 = idx1 - 1;
    if(idx0 < 0 || idx1 >= num_bins_ || counts_(idx0) < 50 || counts_(idx1) < 50) {
      undistort(z);
      return;
    }

    double z0 = (idx0 + 1) * bin_depth_ - bin_depth_ * 0.5;
    double coeff1 = (*z - z0) / bin_depth_;
    double coeff0 = 1.0 - coeff1;
    double mult = coeff0 * multipliers_.coeffRef(idx0) + coeff1 * multipliers_.coeffRef(idx1);
    *z *= mult;
  }

  void DiscreteFrustum::serialize(std::ostream& out) const
  {
    eigen_extensions::serializeScalar(max_dist_, out);
    eigen_extensions::serializeScalar(num_bins_, out);
    eigen_extensions::serializeScalar(bin_depth_, out);
    eigen_extensions::serialize(counts_, out);
    eigen_extensions::serialize(total_numerators_, out);
    eigen_extensions::serialize(total_denominators_, out);
    eigen_extensions::serialize(multipliers_, out);
  }

  void DiscreteFrustum::deserialize(std::istream& in)
  {
    eigen_extensions::deserializeScalar(in, &max_dist_);
    eigen_extensions::deserializeScalar(in, &num_bins_);
    eigen_extensions::deserializeScalar(in, &bin_depth_);
    eigen_extensions::deserialize(in, &counts_);
    eigen_extensions::deserialize(in, &total_numerators_);
    eigen_extensions::deserialize(in, &total_denominators_);
    eigen_extensions::deserialize(in, &multipliers_);
  }

  DiscreteDepthDistortionModel::DiscreteDepthDistortionModel(const DiscreteDepthDistortionModel& other)
  {
    *this = other;
  }

  DiscreteDepthDistortionModel& DiscreteDepthDistortionModel::operator=(const DiscreteDepthDistortionModel& other)
  {
    width_ = other.width_;
    height_ = other.height_;
    bin_width_ = other.bin_width_;
    bin_height_ = other.bin_height_;
    bin_depth_ = other.bin_depth_;
    num_bins_x_ = other.num_bins_x_;
    num_bins_y_ = other.num_bins_y_;

    frustums_ = other.frustums_;
    for(size_t i = 0; i < frustums_.size(); ++i)
      for(size_t j = 0; j < frustums_[i].size(); ++j)
        frustums_[i][j] = new DiscreteFrustum(*other.frustums_[i][j]);

    return *this;
  }

  DiscreteDepthDistortionModel::DiscreteDepthDistortionModel(int width, int height,
                                                             int bin_width, int bin_height,
                                                             double bin_depth,
                                                             int smoothing)
  :
    width_(width),
    height_(height),
    bin_width_(bin_width),
    bin_height_(bin_height),
    bin_depth_(bin_depth)
  {
    assert(width_ % bin_width_ == 0);
    assert(height_ % bin_height_ == 0);

    num_bins_x_ = width_ / bin_width_;
    num_bins_y_ = height_ / bin_height_;

    frustums_.resize(num_bins_y_);
    for(size_t i = 0; i < frustums_.size(); ++i) {
      frustums_[i].resize(num_bins_x_, NULL);
      for(size_t j = 0; j < frustums_[i].size(); ++j)
        frustums_[i][j] = new DiscreteFrustum(smoothing, bin_depth);
    }
  }

  void DiscreteDepthDistortionModel::deleteFrustums()
  {
    for(size_t y = 0; y < frustums_.size(); ++y)
      for(size_t x = 0; x < frustums_[y].size(); ++x)
        if(frustums_[y][x])
          delete frustums_[y][x];
  }

  DiscreteDepthDistortionModel::~DiscreteDepthDistortionModel()
  {
    deleteFrustums();
  }

//  void DiscreteDepthDistortionModel::undistort(DepthMat* depth) const
//  {
//    assert(width_ == depth->cols());
//    assert(height_ ==depth->rows());
////    assert(width_ % depth->cols() == 0);
////    assert(height_ % depth->rows() == 0);
//
//    #pragma omp parallel for
//    for(int v = 0; v < height_; ++v) {
//      for(int u = 0; u < width_; ++u) {
//        if(depth->coeffRef(v, u) == 0)
//          continue;
//
//        double z = depth->coeffRef(v, u) * 0.001;
//        frustum(v, u).interpolatedUndistort(&z);
//        depth->coeffRef(v, u) = z * 1000;
//      }
//    }
//  }

//  void DiscreteDepthDistortionModel::undistort(Eigen::Matrix<double, 240, 320>* depth) const
  void DiscreteDepthDistortionModel::undistort(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>* depth) const
  {
//    #pragma omp parallel for schedule (static, 30)
    for(int v = 0; v < 240; ++v) {
      for(int u = 0; u < 320; ++u) {
        if(depth->coeffRef(v, u) == 0)
          continue;

        frustum(v, u).interpolatedUndistort(&depth->coeffRef(v, u));
      }
    }
  }

  void DiscreteDepthDistortionModel::addExample(int v, int u, double ground_truth, double measurement)
  {
    frustum(v, u).addExample(ground_truth, measurement);
  }

  size_t DiscreteDepthDistortionModel::accumulate(const DepthMat& ground_truth,
                                                  const DepthMat& measurement)
  {
    assert(width_ == ground_truth.cols());
    assert(height_ == ground_truth.rows());
    assert(width_ == measurement.cols());
    assert(height_ == measurement.rows());

    size_t num_training_examples = 0;
    for(int v = 0; v < height_; ++v) {
      for(int u = 0; u < width_; ++u) {
        if(ground_truth.coeffRef(v, u) == 0)
          continue;
        if(measurement.coeffRef(v, u) == 0)
          continue;

        double gt = ground_truth.coeffRef(v, u) * 0.001;
        double meas = measurement.coeffRef(v, u) * 0.001;
        frustum(v, u).addExample(gt, meas);
        ++num_training_examples;
      }
    }

    return num_training_examples;
  }

  void DiscreteDepthDistortionModel::load(const std::string& path)
  {
    ifstream f;
    f.open(path.c_str());
    if(!f.is_open()) {
      cerr << "Failed to open " << path << endl;
      assert(f.is_open());
    }
    deserialize(f);
    f.close();
  }

  void DiscreteDepthDistortionModel::save(const std::string& path) const
  {
    ofstream f;
    f.open(path.c_str());
    if(!f.is_open()) {
      cerr << "Failed to open " << path << endl;
      assert(f.is_open());
    }
    serialize(f);
    f.close();
  }

  void DiscreteDepthDistortionModel::serialize(std::ostream& out) const
  {
    out << "DiscreteDepthDistortionModel v01" << endl;
    eigen_extensions::serializeScalar(width_, out);
    eigen_extensions::serializeScalar(height_, out);
    eigen_extensions::serializeScalar(bin_width_, out);
    eigen_extensions::serializeScalar(bin_height_, out);
    eigen_extensions::serializeScalar(bin_depth_, out);
    eigen_extensions::serializeScalar(num_bins_x_, out);
    eigen_extensions::serializeScalar(num_bins_y_, out);

    for(int y = 0; y < num_bins_y_; ++y)
      for(int x = 0; x < num_bins_x_; ++x)
        frustums_[y][x]->serialize(out);
  }

  void DiscreteDepthDistortionModel::deserialize(std::istream& in)
  {
    string buf;
    getline(in, buf);
    assert(buf == "DiscreteDepthDistortionModel v01");
    eigen_extensions::deserializeScalar(in, &width_);
    eigen_extensions::deserializeScalar(in, &height_);
    eigen_extensions::deserializeScalar(in, &bin_width_);
    eigen_extensions::deserializeScalar(in, &bin_height_);
    eigen_extensions::deserializeScalar(in, &bin_depth_);
    eigen_extensions::deserializeScalar(in, &num_bins_x_);
    eigen_extensions::deserializeScalar(in, &num_bins_y_);

    deleteFrustums();
    frustums_.resize(num_bins_y_);
    for(size_t y = 0; y < frustums_.size(); ++y) {
      frustums_[y].resize(num_bins_x_, NULL);
      for(size_t x = 0; x < frustums_[y].size(); ++x) {
        frustums_[y][x] = new DiscreteFrustum;
        frustums_[y][x]->deserialize(in);
      }
    }
  }

  DiscreteFrustum& DiscreteDepthDistortionModel::frustum(int y, int x)
  {
//  std::cout << "DiscreteDepthDistortionModel bin_depth_ " << bin_depth_ << std::endl;
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    int xidx = x / bin_width_;
    int yidx = y / bin_height_;
    return (*frustums_[yidx][xidx]);
  }

  const DiscreteFrustum& DiscreteDepthDistortionModel::frustum(int y, int x) const
  {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    int xidx = x / bin_width_;
    int yidx = y / bin_height_;
    return (*frustums_[yidx][xidx]);
  }

  std::string DiscreteDepthDistortionModel::status(const std::string& prefix) const
  {
    ostringstream oss;
    oss << prefix << "Image width (pixels): " << width_ << endl;
    oss << prefix << "Image height (pixels): " << height_ << endl;
    oss << prefix << "Bin width (pixels): " << bin_width_ << endl;
    oss << prefix << "Bin height (pixels): " << bin_height_ << endl;
    oss << prefix << "Bin depth (m): " << bin_depth_ << endl;
    return oss.str();
  }

  void DiscreteDepthDistortionModel::downsampleParams(const int& downsamplingStep)
  {
    assert(bin_width_ % downsamplingStep == 0 && bin_height_ % downsamplingStep == 0);
    width_ /= downsamplingStep;
    height_ /= downsamplingStep;
    bin_width_ /= downsamplingStep;
    bin_height_ /= downsamplingStep;
  }

}  // namespace clams
