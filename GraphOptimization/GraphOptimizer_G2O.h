/*! This class performs pose-graph optimization. This class is based on the previous work of
 *  Miguel Algaba Borrego, in "http://code.google.com/p/kinect-rgbd-graphslam6d/".
 */

#include "GraphOptimizer.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
//#include <g2o/core/factory.h>
//#include <g2o/core/optimization_algorithm_factory.h>
//#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

//Graph optimization with 6DoF
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

//Graph optimization with 3DoF
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/types/slam2d/edge_se2.h>

/*!This class encapsulates the functionality of the G2O library to perform 3D/6D graph optimization.*/
class GraphOptimizer_G2O : public GraphOptimizer
{
private:
    int vertexIdx;
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    g2o::BlockSolverX * solver_ptr;

public:
  GraphOptimizer_G2O();
  /*!Adds a new vertex to the graph. The provided 4x4 matrix will be considered as the pose of the new added vertex. It returns the index of the added vertex.*/
  int addVertex(Eigen::Matrix4d& pose);
  /*!Adds an edge that defines a spatial constraint between the vertices "fromIdx" and "toIdx" with information matrix that determines the weight of the added edge.*/
  void addEdge(const int fromIdx,const int toIdx,Eigen::Matrix4d& relPose,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& infMatrix);
  /*!Calls the graph optimization process to determine the pose configuration that best satisfies the constraints defined by the edges.*/
  void optimizeGraph();
  /*!Returns a vector with all the optimized poses of the graph.*/
  void getPoses(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d > >&);
  /*!Saves the graph to file.*/
  void saveGraph(std::string fileName);
};
