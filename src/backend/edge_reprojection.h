#ifndef SIMPLE_VIO_BACKEND_EDGE_REPROJECTION_H_
#define SIMPLE_VIO_BACKEND_EDGE_REPROJECTION_H_

#include "optimize/edge.h"

#include "utils/eigen_types.h"

namespace SimpleVIO {

// connected with 4 vertices: InvDepth(i), Pose(i), Pose(j), ExtrinsicPose
class EdgeReprojection : public Edge<2, 4> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeReprojection(const Vec3 &pts_i, const Vec3 &pts_j)
        : pts_i_(pts_i), pts_j_(pts_j) {
        SetVerticesTypes(std::vector<std::string>{"VertexInverseDepth",
                                                  "VertexPose",
                                                  "VertexPose",
                                                  "VertexPose"});
    }

    virtual std::string TypeInfo() const override {
        return "EdgeReprojection";
    }

    virtual void ComputeResidual() override;

    virtual void ComputeJacobians() override;

private:
    void GetParams(double &inv_depth_i,
                   Vec3 &Pi,
                   Qd &Qi,
                   Vec3 &Pj,
                   Qd &Qj,
                   Vec3 &tic,
                   Qd &qic);

private:
    // measurements
    Vec3 pts_i_, pts_j_;
};

// connected with 2 vertices: PointXYZ(i), Pose(i)
class EdgeReprojectionXYZ : public Edge<2, 2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeReprojectionXYZ(const Vec2 &pts_i,
                                 const Vec3 &tic,
                                 const Qd &qic)
        : pts_i_(pts_i), tic_(tic), qic_(qic) {
        SetVerticesTypes(std::vector<std::string>{"VertexXYZ", "VertexPose"});
    }

    virtual std::string TypeInfo() const override {
        return "EdgeReprojectionXYZ";
    }

    virtual void ComputeResidual() override;

    virtual void ComputeJacobians() override;

private:
    void GetParams(Vec3 &pts, Vec3 &Pi, Qd &Qi);

private:
    // transform from camera to imu
    Vec3 tic_;
    Qd qic_;

    // measurements
    Vec2 pts_i_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_BACKEND_EDGE_REPROJECTION_H_