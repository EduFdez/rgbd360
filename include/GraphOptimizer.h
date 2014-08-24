/*! This class performs pose-graph optimization. This class is based on the previous work of
 *  Miguel Algaba Borrego, in "http://code.google.com/p/kinect-rgbd-graphslam6d/".
 */

#ifndef GRAPH_OPTIMIZER
#define GRAPH_OPTIMIZER

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

#include <vector>
#include <Eigen/Core>
//#include <Eigen/Quaternion.h>
#include <string>

typedef float datatype;

/*!Abstract class that defines the mandatory methods that a 3D/6D GraphSLAM optimizer must implement.*/
//template<class datatype>
class GraphOptimizer
{
private:
    int vertexIdx;
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    g2o::BlockSolverX * solver_ptr;

public:

    /*! The index of the next vertex to be introduced in the graph (it corresponds to the graph size). */
    int vertex_id;

    enum RigidTransformationType {SixDegreesOfFreedom = 0,ThreeDegreesOfFreedom = 1} rigidTransformationType;

    GraphOptimizer()
    {
        rigidTransformationType = GraphOptimizer::SixDegreesOfFreedom;

        optimizer.setVerbose(false);

        // variable-size block solver
        //    linearSolver = new LinearSolverCholmod<BlockSolverX::PoseMatrixType>();
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
        solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        optimizer.setAlgorithm(solver);

        //Set the vertex index to 0
        vertexIdx=0;
    }

    /*! Adds a new vertex to the graph. The provided 4x4 matrix will be considered as the pose of the new added vertex. It returns the index of the added vertex.*/
    int addVertex(const Eigen::Matrix<double,4,4> vertexPose)
    {
        if(rigidTransformationType==GraphOptimizer::SixDegreesOfFreedom) //Rigid transformation 6DoF
        {
            //Transform pose from Eigen::Matrix<datatype,4,4> into Isometry3d to set up rotation and translation for this node
            Eigen::Isometry3d cam; // camera pose
            Eigen::Matrix3d m_rot = vertexPose.block(0,0,3,3);
            cam = Eigen::Quaterniond(m_rot);
            cam.translation() = vertexPose.block(0,3,3,1);
//            Transform<datatype,3,Isometry> cam; // camera pose
//            cam = Quaternion<datatype>(vertexPose.block(0,0,3,3));

            // set up node
            g2o::VertexSE3 *vc = new g2o::VertexSE3();
            vc->setEstimate(cam);

            vc->setId(vertex_id);      // vertex id

    //        cerr << vertexPose.block(0,3,3,1).transpose() << " | " << Quaterniond(vertexPose.block(0,0,3,3)).transpose() << endl;

            // set first cam pose fixed
            if (vertex_id==0)
              vc->setFixed(true);

            // add to optimizer
            optimizer.addVertex(vc);

            vertex_id++;
        }
    //    else //Rigid transformation 3DoF
    //    {
    //        //Transform Eigen::Matrix<datatype,4,4> into 2D translation and rotation for g2o
    //        SE2 pose(vertexPose(0,3),vertexPose(1,3),atan2(vertexPose(1,0),vertexPose(0,0)));

    //        // set up node
    //        VertexSE2 *vc = new VertexSE2();
    //        vc->estimate() = pose;
    //        vc->setId(vertexIdx);      // vertex id

    //        // set first pose fixed
    //        if (vertexIdx==0){
    //            vc->setFixed(true);
    //        }

    //        // add to optimizer
    //        optimizer.addVertex(vc);
    //    }

        //Update vertex index
        vertexIdx++;

        //Return the added vertex index
        return vertexIdx-1;
    }

    /*! Adds an edge that defines a spatial constraint between the vertices "fromIdx" and "toIdx" with information matrix that determines the weight of the added edge.*/
    void addEdge(const int fromIdx,
                 const int toIdx,
                 const Eigen::Matrix<double,4,4> relativePose,
                 const Eigen::Matrix<double,6,6> informationMatrix)
    {
        if(rigidTransformationType==GraphOptimizer::SixDegreesOfFreedom) //Rigid transformation 6DoF
        {
            //Transform Eigen::Matrix<datatype,4,4> into 3D traslation and rotation for g2o
            Eigen::Vector3d t(relativePose(0,3),relativePose(1,3),relativePose(2,3));
            Eigen::Matrix3d m_rot = relativePose.block(0,0,3,3);
            Eigen::Quaterniond q(m_rot);
            g2o::SE3Quat transf(q,t);	// relative transformation

            g2o::EdgeSE3* edge = new g2o::EdgeSE3;
            edge->vertices()[0] = optimizer.vertex(fromIdx);
            edge->vertices()[1] = optimizer.vertex(toIdx);
            edge->setMeasurement(transf);

            //Set the information matrix to identity
            edge->setInformation(informationMatrix);

            optimizer.addEdge(edge);
        }
    //    else //Rigid transformation 3DoF
    //    {
    //        //Transform Eigen::Matrix<datatype,4,4> into 2D translation and rotation for g2o
    //        SE2 transf(relativePose(0,3),relativePose(1,3),atan2(relativePose(1,0),relativePose(0,0))); // relative transformation

    //        EdgeSE2* edge = new EdgeSE2;
    //        edge->vertices()[0] = optimizer.vertex(fromIdx);
    //        edge->vertices()[1] = optimizer.vertex(toIdx);
    //        edge->setMeasurement(transf);

    //        //Set the information matrix to identity
    //        edge->setInformation(informationMatrix);

    //        optimizer.addEdge(edge);
    //    }
    }

    /*! Calls the graph optimization process to determine the pose configuration that best satisfies the constraints defined by the edges.*/
    void optimizeGraph()
    {
        //Prepare and run the optimization
        optimizer.initializeOptimization();

        //Set the initial Levenberg-Marquardt lambda
    //    optimizer.setUserLambdaInit(0.01);

        optimizer.computeActiveErrors();
        cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << endl;

        optimizer.setVerbose(true);

        //Run optimization
        optimizer.optimize(10);
    }

    /*! Returns a vector with all the optimized poses of the graph.*/
    void getPoses(std::vector<Eigen::Matrix<datatype,4,4>, Eigen::aligned_allocator<Eigen::Matrix<datatype,4,4> > > &poses)
    {
        poses.clear();
        poses.resize(vertexIdx);

        #pragma omp parallel for
        for(int poseIdx=0;poseIdx<vertexIdx;poseIdx++)
        {
            Eigen::Matrix<datatype,4,4> optimizedPose = Eigen::Matrix<datatype,4,4>::Zero();

            if(rigidTransformationType==GraphOptimizer::SixDegreesOfFreedom) //Rigid transformation 6DoF
            {
                //Transform the vertex pose from G2O quaternion and traslation vector to Eigen::Matrix<datatype,4,4>
                g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(poseIdx));
                double optimizedPoseQuaternion[7];
                vertex->getEstimateData(optimizedPoseQuaternion);

                double qx,qy,qz,qr,qx2,qy2,qz2,qr2;

                Eigen::Quaterniond quat;
                quat.x()=optimizedPoseQuaternion[3];
                quat.y()=optimizedPoseQuaternion[4];
                quat.z()=optimizedPoseQuaternion[5];
                quat.w()=optimizedPoseQuaternion[6];
                Eigen::Matrix3d rotationMatrix(quat);

                for(int i=0;i<3;i++)
                {
                    for(int j=0;j<3;j++)
                    {
                        optimizedPose(i,j)=rotationMatrix(i,j);
                    }
                }
                optimizedPose(0,3)=optimizedPoseQuaternion[0];
                optimizedPose(1,3)=optimizedPoseQuaternion[1];
                optimizedPose(2,3)=optimizedPoseQuaternion[2];
                optimizedPose(3,3)=1;
            }
            else //Rigid transformation 3DoF
            {
                //Transform the vertex pose from G2O SE2 pose to Eigen::Matrix<datatype,4,4>
                g2o::VertexSE2* vertex = dynamic_cast<g2o::VertexSE2*>(optimizer.vertex(poseIdx));
                double optimizedPoseSE2[3];
                vertex->getEstimateData(optimizedPoseSE2);

                optimizedPose(0,3)=optimizedPoseSE2[0];//x
                optimizedPose(1,3)=optimizedPoseSE2[1];//y
                optimizedPose(0,0)=cos(optimizedPoseSE2[2]);
                optimizedPose(0,1)=-sin(optimizedPoseSE2[2]);
                optimizedPose(1,0)=sin(optimizedPoseSE2[2]);
                optimizedPose(1,1)=cos(optimizedPoseSE2[2]);
                optimizedPose(2,2)=1;
                optimizedPose(3,3)=1;
            }

            //Set the optimized pose to the vector of poses
            poses[poseIdx] = optimizedPose;
        }
    }

    /*! Saves the graph to file.*/
    void saveGraph(std::string fileName)
    {
        //Save the graph to file
        optimizer.save(fileName.c_str(),0);
    }

    /*! Sets the rigid transformation type to the graph optimizer (6DoF or 3DoF).*/
    inline void setRigidTransformationType(RigidTransformationType rttype)
    {
        rigidTransformationType = rttype;
    }
};

#endif
