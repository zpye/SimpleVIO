#include "edge_prior.h"

#include "mini_sophus/so3.hpp"

#include "utils/rotation_utils.h"

#define USE_SO3_JACOBIAN

namespace SimpleVIO {

void EdgeSE3Prior::ComputeResidual() {
    Vec3 Pi;
    Qd Qi;
    GetParams(Pi, Qi);

    // rotation error
#if defined(USE_SO3_JACOBIAN)
    Sophus::SO3d ri(Qi);
    Sophus::SO3d rp(rotation_prior_);
    Sophus::SO3d res_r = rp.inverse() * ri;

    residual_.block<3, 1>(0, 0) = res_r.log();
#else
    residual_.block<3, 1>(0, 0) = 2 * (rotation_prior_.inverse() * Qi).vec();
#endif

    // translation error
    residual_.block<3, 1>(3, 0) = Pi - translation_prior_;
}

void EdgeSE3Prior::ComputeJacobians() {
    Vec3 Pi;
    Qd Qi;
    GetParams(Pi, Qi);

    // w.r.t. pose i
    Eigen::Matrix<double, 6, 6> jacobian_pose_i =
        Eigen::Matrix<double, 6, 6>::Zero();

#if defined(USE_SO3_JACOBIAN)
    Sophus::SO3d ri(Qi);
    Sophus::SO3d rp(rotation_prior_);
    Sophus::SO3d res_r = rp.inverse() * ri;

    // http://rpg.ifi.uzh.ch/docs/RSS15_Forster.pdf  A.32
    jacobian_pose_i.block<3, 3>(0, 3) = Sophus::SO3d::JacobianRInv(res_r.log());
#else
    jacobian_pose_i.block<3, 3>(0, 3) =
        Qleft(rotation_prior_.inverse() * Qi).bottomRightCorner<3, 3>();
#endif

    jacobian_pose_i.block<3, 3>(3, 0) = Mat33::Identity();

    jacobians_[0] = jacobian_pose_i;
}

void EdgeSE3Prior::GetParams(Vec3 &Pi, Qd &Qi) {
    double pose[7] = {0};

    vertices_[0]->GetParameters(pose);
    Pi = Vec3(pose[0], pose[1], pose[2]);
    Qi = Qd(pose[6], pose[3], pose[4], pose[5]);
}

}  // namespace SimpleVIO