#ifndef SIMPLE_VIO_BACKEND_EDGE_PRIOR_H_
#define SIMPLE_VIO_BACKEND_EDGE_PRIOR_H_

#include "optimize/edge.h"

#include "utils/eigen_types.h"

namespace SimpleVIO {

// connected with 1 vertices: Pose(i)
class EdgeSE3Prior : public Edge<6, 1> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeSE3Prior(const Vec3 &p, const Qd &q)
        : translation_prior_(p), rotation_prior_(q) {
        SetVerticesTypes(std::vector<std::string>{"VertexPose"});
    }

public:
    virtual std::string TypeInfo() const override {
        return "EdgeSE3Prior";
    }

    virtual void ComputeResidual() override;

    virtual void ComputeJacobians() override;

private:
    void GetParams(Vec3 &Pi, Qd &Qi);

private:
    Vec3 translation_prior_;
    Qd rotation_prior_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_BACKEND_EDGE_PRIOR_H_