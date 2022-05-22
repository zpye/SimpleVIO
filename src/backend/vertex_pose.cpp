#include "vertex_pose.h"

#include "mini_sophus/se3.hpp"

#include "utils/eigen_types.h"

namespace SimpleVIO {

void VertexPose::Plus(double *delta) {
    Eigen::Map<LocalVec> d(delta);

    // translation
    if (!fix_translation_) {
        parameters_.head<3>() += d.head<3>();
    }

    // rotation
    if (!fix_rotation_) {
        Qd q(parameters_[6], parameters_[3], parameters_[4], parameters_[5]);

        // right multiplication with so3
        q = q * Sophus::SO3d::exp(Vec3(d[3], d[4], d[5])).unit_quaternion();
        q.normalize();

        parameters_[3] = q.x();
        parameters_[4] = q.y();
        parameters_[5] = q.z();
        parameters_[6] = q.w();
    }
}

void VertexPose::SetFixRotation(bool fix_rotation) {
    fix_rotation_ = fix_rotation;
}

bool VertexPose::GetFixRotation() {
    return fix_rotation_;
}

void VertexPose::SetFixTranslation(bool fix_translation) {
    fix_translation_ = fix_translation;
}

bool VertexPose::GetFixTranslation() {
    return fix_translation_;
}

}  // namespace SimpleVIO