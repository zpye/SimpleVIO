#include "edge_reprojection.h"

#include "backend/vertex_pose.h"

#include "mini_sophus/se3.hpp"
#include "utils/rotation_utils.h"

namespace SimpleVIO {

void EdgeReprojection::ComputeResidual() {
    double inv_depth_i;
    Vec3 Pi;
    Qd Qi;
    Vec3 Pj;
    Qd Qj;
    Vec3 tic;
    Qd qic;
    GetParams(inv_depth_i, Pi, Qi, Pj, Qj, tic, qic);

    Vec3 pts_camera_i = pts_i_ / inv_depth_i;
    Vec3 pts_imu_i    = qic * pts_camera_i + tic;
    Vec3 pts_w        = Qi * pts_imu_i + Pi;
    Vec3 pts_imu_j    = Qj.inverse() * (pts_w - Pj);
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    residual_ = (pts_camera_j / pts_camera_j.z()).head<2>() - pts_j_.head<2>();
}

void EdgeReprojection::ComputeJacobians() {
    double inv_depth_i;
    Vec3 Pi;
    Qd Qi;
    Vec3 Pj;
    Qd Qj;
    Vec3 tic;
    Qd qic;
    GetParams(inv_depth_i, Pi, Qi, Pj, Qj, tic, qic);

    Vec3 pts_camera_i = pts_i_ / inv_depth_i;
    Vec3 pts_imu_i    = qic * pts_camera_i + tic;
    Vec3 pts_w        = Qi * pts_imu_i + Pi;
    Vec3 pts_imu_j    = Qj.inverse() * (pts_w - Pj);
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double depth_j = pts_camera_j.z();

    Mat33 Ri  = Qi.toRotationMatrix();
    Mat33 Rj  = Qj.toRotationMatrix();
    Mat33 ric = qic.toRotationMatrix();

    Eigen::Matrix<double, 2, 3> reduce;
    reduce << 1.0 / depth_j, 0.0, -pts_camera_j(0) / (depth_j * depth_j), 0.0,
        1.0 / depth_j, -pts_camera_j(1) / (depth_j * depth_j);

    // Jacobian 0
    {
        Vec2 jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri *
                                ric * pts_i_ * -1.0 /
                                (inv_depth_i * inv_depth_i);

        jacobians_[0] = jacobian_feature;
    }

    // Jacobian 1
    {
        Eigen::Matrix<double, 3, 6> jaco_i;
        jaco_i.leftCols<3>()  = ric.transpose() * Rj.transpose();
        jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri *
                                -Sophus::SO3d::hat(pts_imu_i);

        Eigen::Matrix<double, 2, 6> jacobian_pose_i = reduce * jaco_i;

        jacobians_[1] = jacobian_pose_i;
    }

    // Jacobian 2
    {
        Eigen::Matrix<double, 3, 6> jaco_j;
        jaco_j.leftCols<3>()  = ric.transpose() * -Rj.transpose();
        jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);

        Eigen::Matrix<double, 2, 6> jacobian_pose_j = reduce * jaco_j;

        jacobians_[2] = jacobian_pose_j;
    }

    // Jacobian 3
    {
        Eigen::Matrix<double, 3, 6> jaco_ex;
        jaco_ex.leftCols<3>() =
            ric.transpose() * (Rj.transpose() * Ri - Mat33::Identity());
        Mat33 tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
        jaco_ex.rightCols<3>() =
            -tmp_r * SkewSymmetric(pts_camera_i) +
            SkewSymmetric(tmp_r * pts_camera_i) +
            SkewSymmetric(ric.transpose() *
                          (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));

        Eigen::Matrix<double, 2, 6> jacobian_ex_pose = reduce * jaco_ex;

        jacobians_[3] = jacobian_ex_pose;
    }
}

void EdgeReprojection::GetParams(double &inv_depth_i,
                                 Vec3 &Pi,
                                 Qd &Qi,
                                 Vec3 &Pj,
                                 Qd &Qj,
                                 Vec3 &tic,
                                 Qd &qic) {
    double inv_depth[1];
    vertices_[0]->GetParameters(inv_depth);
    inv_depth_i = inv_depth[0];

    double pose[7] = {0};

    vertices_[1]->GetParameters(pose);
    Pi = Vec3(pose[0], pose[1], pose[2]);
    Qi = Qd(pose[6], pose[3], pose[4], pose[5]);

    vertices_[2]->GetParameters(pose);
    Pj = Vec3(pose[0], pose[1], pose[2]);
    Qj = Qd(pose[6], pose[3], pose[4], pose[5]);

    vertices_[3]->GetParameters(pose);
    tic = Vec3(pose[0], pose[1], pose[2]);
    qic = Qd(pose[6], pose[3], pose[4], pose[5]);
}

void EdgeReprojectionXYZ::ComputeResidual() {
    Vec3 pts_w;
    Vec3 Pi;
    Qd Qi;
    GetParams(pts_w, Pi, Qi);

    Vec3 pts_imu_i    = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = qic_.inverse() * (pts_imu_i - tic_);

    residual_ = (pts_camera_i / pts_camera_i.z()).head<2>() - pts_i_.head<2>();
}

void EdgeReprojectionXYZ::ComputeJacobians() {
    Vec3 pts_w;
    Vec3 Pi;
    Qd Qi;
    GetParams(pts_w, Pi, Qi);

    Vec3 pts_imu_i    = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = qic_.inverse() * (pts_imu_i - tic_);

    double depth_i = pts_camera_i.z();

    Mat33 Ri  = Qi.toRotationMatrix();
    Mat33 ric = qic_.toRotationMatrix();

    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    reduce << 1.0 / depth_i, 0.0, -pts_camera_i(0) / (depth_i * depth_i), 0.0,
        1.0 / depth_i, -pts_camera_i(1) / (depth_i * depth_i);

    std::shared_ptr<VertexPose> vertex_pose =
        std::dynamic_pointer_cast<VertexPose>(vertices_[1]);
    if (!vertex_pose) {
        LOGE("SimpleVIO", "Error vertex");
    }

    // Jacobian 0
    {
        // translation
        Mat23 jacobian_feature = Mat23::Zero();
        if (!vertex_pose->GetFixTranslation()) {
            jacobian_feature = -reduce * ric.transpose() * Ri.transpose();
        }

        jacobians_[0] = jacobian_feature;
    }

    // Jacobian 1
    {
        // rotation
        Mat26 jacobian_pose_i = Mat26::Zero();
        if (!vertex_pose->GetFixRotation()) {
            Mat36 jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Ri.transpose();
            jaco_i.rightCols<3>() =
                ric.transpose() * -Sophus::SO3d::hat(pts_imu_i);

            jacobian_pose_i = -reduce * jaco_i;
        }

        jacobians_[1] = jacobian_pose_i;
    }
}

void EdgeReprojectionXYZ::GetParams(Vec3 &pts, Vec3 &Pi, Qd &Qi) {
    double point[3] = {0};
    double pose[7]  = {0};

    vertices_[0]->GetParameters(point);
    pts = Vec3(point[0], point[1], point[2]);

    vertices_[1]->GetParameters(pose);
    Pi = Vec3(pose[0], pose[1], pose[2]);
    Qi = Qd(pose[6], pose[3], pose[4], pose[5]);
}

}  // namespace SimpleVIO