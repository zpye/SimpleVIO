#ifndef SIMPLE_VIO_BACKEND_VERTEX_POSE_H_
#define SIMPLE_VIO_BACKEND_VERTEX_POSE_H_

#include "optimize/vertex.h"

namespace SimpleVIO {

/**
 * Pose vertex
 * parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
 * optimization is perform on manifold, so update is 6 DoF, left multiplication
 *
 * pose is represented as Twb in VIO case
 */
class VertexPose : public Vertex<7, 6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexPose() {}

    virtual ~VertexPose() {}

public:
    virtual std::string TypeInfo() const override {
        return "VertexPose";
    }

    virtual void Plus(double *delta) override;

    void SetFixRotation(bool fix_rotation = true);

    bool GetFixRotation();

    void SetFixTranslation(bool fix_translation = true);

    bool GetFixTranslation();

private:
    bool fix_rotation_    = false;
    bool fix_translation_ = false;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_BACKEND_VERTEX_POSE_H_