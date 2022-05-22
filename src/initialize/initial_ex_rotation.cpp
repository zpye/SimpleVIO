#include "initial_ex_rotation.h"

#include <cstddef>

#include <opencv2/opencv.hpp>

#include "solve_relative_pose.h"
#include "utils/rotation_utils.h"

namespace SimpleVIO {

InitialEXRotation::InitialEXRotation() {
    ric_ = Mat33::Identity();
}

bool InitialEXRotation::CalibrationExRotation(
    const std::vector<std::pair<Vec3, Vec3>> &corres,
    const Qd &delta_q_imu,
    Mat33 &calib_ric_result) {
    Mat33 R;
    SolveRelativeR(corres, R);

    Rc_.push_back(R);
    Rc_g_.push_back(ric_.inverse() * delta_q_imu * ric_);
    Rimu_.push_back(delta_q_imu.toRotationMatrix());

    const size_t frame_count = Rc_.size();

    MatXX A(frame_count * 4, 4);
    A.setZero();

    for (size_t i = 0; i < frame_count; ++i) {
        Qd r1(Rc_[i]);
        Qd r2(Rc_g_[i]);

        const double angular_distance = r1.angularDistance(r2) * 180.0 / M_PI;
        const double huber =
            (angular_distance > 5.0 ? 5.0 / angular_distance : 1.0);

        double w = Qd(Rc_[i]).w();
        Vec3 q   = Qd(Rc_[i]).vec();

        Mat44 L;
        L.block<3, 3>(0, 0) = w * Mat33::Identity() + SkewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3)             = w;

        Qd Rij(Rimu_[i]);
        w = Rij.w();
        q = Rij.vec();

        Mat44 R;
        R.block<3, 3>(0, 0) = w * Mat33::Identity() - SkewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3)             = w;

        A.block<4, 4>(i * 4, 0) = huber * (L - R);
    }

    Eigen::JacobiSVD<MatXX> svd(A, Eigen::ComputeFullV);

    Qd estimated_R(Vec4(svd.matrixV().col(3)));
    ric_ = estimated_R.toRotationMatrix().inverse();

    Vec3 ric_cov = svd.singularValues().tail<3>();
    if (ric_cov(1) > 0.25) {
        calib_ric_result = ric_;

        return true;
    } else {
        return false;
    }
}

}  // namespace SimpleVIO
