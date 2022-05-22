#include "edge_imu.h"

#include <iostream>

#include "imu/imu_utils.h"
#include "utils/logger.h"
#include "utils/rotation_utils.h"

namespace SimpleVIO {

void EdgeImu::ComputeResidual() {
    Vec3 Pi;
    Qd Qi;
    Vec3 Vi;
    Vec3 Bai;
    Vec3 Bgi;
    Vec3 Pj;
    Qd Qj;
    Vec3 Vj;
    Vec3 Baj;
    Vec3 Bgj;

    GetParams(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

    residual_ =
        imu_pre_integration_
            ->ComputeResidual(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

    IMUIntegration::CovarianceMatrix covariance_inv =
        imu_pre_integration_->GetCovariance().inverse();

    // std::cout << "cov:\n" << covariance_inv << "\n" << std::endl;

    SetInformation(covariance_inv.data());
}

void EdgeImu::ComputeJacobians() {
    Vec3 Pi;
    Qd Qi;
    Vec3 Vi;
    Vec3 Bai;
    Vec3 Bgi;
    Vec3 Pj;
    Qd Qj;
    Vec3 Vj;
    Vec3 Baj;
    Vec3 Bgj;

    GetParams(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

    const double sum_dt = imu_pre_integration_->GetSumDeltaTime();

    const Qd delta_q = imu_pre_integration_->GetDeltaQ();

    const Vec3 linearized_bg = imu_pre_integration_->GetLinearizedBg();

    const Vec3 &G = imu_pre_integration_->GetGravity();

    const IMUIntegration::JacobianMatrix &jacobian =
        imu_pre_integration_->GetJacobian();

    const Mat33 dp_dba =
        jacobian.block<3, 3>(StateOrder::O_P, StateOrder::O_BA);
    const Mat33 dp_dbg =
        jacobian.block<3, 3>(StateOrder::O_P, StateOrder::O_BG);

    const Mat33 dq_dbg =
        jacobian.block<3, 3>(StateOrder::O_R, StateOrder::O_BG);

    const Mat33 dv_dba =
        jacobian.block<3, 3>(StateOrder::O_V, StateOrder::O_BA);
    const Mat33 dv_dbg =
        jacobian.block<3, 3>(StateOrder::O_V, StateOrder::O_BG);

    if (jacobian.maxCoeff() > 1e8 || jacobian.minCoeff() < -1e8) {
        LOGW("SimpleVIO", "numerical unstable in preintegration");
    }

    const Qd corrected_delta_q =
        delta_q * DeltaQ(dq_dbg * (Bgi - linearized_bg));

    const Qd Qi_inv = Qi.inverse();
    const Qd Qj_inv = Qj.inverse();

    // jacobians 0
    {
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i;
        jacobian_pose_i.setZero();

        jacobian_pose_i.block<3, 3>(StateOrder::O_P, StateOrder::O_P) =
            -Qi_inv.toRotationMatrix();

        jacobian_pose_i.block<3, 3>(StateOrder::O_P, StateOrder::O_R) =
            SkewSymmetric(Qi_inv *
                          (Pj - Pi - Vi * sum_dt + 0.5 * G * sum_dt * sum_dt));

        jacobian_pose_i.block<3, 3>(StateOrder::O_R, StateOrder::O_R) =
            -(Qleft(Qj_inv * Qi) * Qright(corrected_delta_q))
                 .bottomRightCorner<3, 3>();

        jacobian_pose_i.block<3, 3>(StateOrder::O_V, StateOrder::O_R) =
            SkewSymmetric(Qi_inv * (Vj - Vi + G * sum_dt));

        // TODO: jacobian_pose_i = sqrt_info * jacobian_pose_i;

        if (jacobian_pose_i.maxCoeff() > 1e8 ||
            jacobian_pose_i.minCoeff() < -1e8) {
            LOGW("SimpleVIO", "numerical unstable in preintegration");
        }

        jacobians_[0] = jacobian_pose_i;
    }

    // jacobians 1
    {
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_i;
        jacobian_speedbias_i.setZero();
        jacobian_speedbias_i.block<3, 3>(StateOrder::O_P,
                                         StateOrder::O_V - StateOrder::O_V) =
            -Qi_inv.toRotationMatrix() * sum_dt;
        jacobian_speedbias_i.block<3, 3>(StateOrder::O_P,
                                         StateOrder::O_BA - StateOrder::O_V) =
            -dp_dba;
        jacobian_speedbias_i.block<3, 3>(StateOrder::O_P,
                                         StateOrder::O_BG - StateOrder::O_V) =
            -dp_dbg;

        jacobian_speedbias_i.block<3, 3>(StateOrder::O_R,
                                         StateOrder::O_BG - StateOrder::O_V) =
            -Qleft(Qj_inv * Qi * delta_q).bottomRightCorner<3, 3>() * dq_dbg;

        jacobian_speedbias_i.block<3, 3>(StateOrder::O_V,
                                         StateOrder::O_V - StateOrder::O_V) =
            -Qi_inv.toRotationMatrix();

        jacobian_speedbias_i.block<3, 3>(StateOrder::O_V,
                                         StateOrder::O_BA - StateOrder::O_V) =
            -dv_dba;

        jacobian_speedbias_i.block<3, 3>(StateOrder::O_V,
                                         StateOrder::O_BG - StateOrder::O_V) =
            -dv_dbg;

        jacobian_speedbias_i.block<3, 3>(StateOrder::O_BA,
                                         StateOrder::O_BA - StateOrder::O_V) =
            -Mat33::Identity();

        jacobian_speedbias_i.block<3, 3>(StateOrder::O_BG,
                                         StateOrder::O_BG - StateOrder::O_V) =
            -Mat33::Identity();

        // TODO: jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

        jacobians_[1] = jacobian_speedbias_i;
    }

    // jacobians 2
    {
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_j;
        jacobian_pose_j.setZero();

        jacobian_pose_j.block<3, 3>(StateOrder::O_P, StateOrder::O_P) =
            Qi_inv.toRotationMatrix();

        jacobian_pose_j.block<3, 3>(StateOrder::O_R, StateOrder::O_R) =
            Qleft(corrected_delta_q.inverse() * Qi_inv * Qj)
                .bottomRightCorner<3, 3>();

        // TODO: jacobian_pose_j = sqrt_info * jacobian_pose_j;

        jacobians_[2] = jacobian_pose_j;
    }

    // jacobians 3
    {
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_j;
        jacobian_speedbias_j.setZero();

        jacobian_speedbias_j.block<3, 3>(StateOrder::O_V,
                                         StateOrder::O_V - StateOrder::O_V) =
            Qi_inv.toRotationMatrix();

        jacobian_speedbias_j.block<3, 3>(StateOrder::O_BA,
                                         StateOrder::O_BA - StateOrder::O_V) =
            Mat33::Identity();

        jacobian_speedbias_j.block<3, 3>(StateOrder::O_BG,
                                         StateOrder::O_BG - StateOrder::O_V) =
            Mat33::Identity();

        // TODO: jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

        jacobians_[3] = jacobian_speedbias_j;
    }
}

void EdgeImu::GetParams(Vec3 &Pi,
                        Qd &Qi,
                        Vec3 &Vi,
                        Vec3 &Bai,
                        Vec3 &Bgi,
                        Vec3 &Pj,
                        Qd &Qj,
                        Vec3 &Vj,
                        Vec3 &Baj,
                        Vec3 &Bgj) {
    double pose[7]       = {0};
    double speed_bias[9] = {0};

    vertices_[0]->GetParameters(pose);
    Pi = Vec3(pose[0], pose[1], pose[2]);
    Qi = Qd(pose[6], pose[3], pose[4], pose[5]);

    vertices_[1]->GetParameters(speed_bias);
    Vi  = Vec3(speed_bias[0], speed_bias[1], speed_bias[2]);
    Bai = Vec3(speed_bias[3], speed_bias[4], speed_bias[5]);
    Bgi = Vec3(speed_bias[6], speed_bias[7], speed_bias[8]);

    vertices_[2]->GetParameters(pose);
    Pj = Vec3(pose[0], pose[1], pose[2]);
    Qj = Qd(pose[6], pose[3], pose[4], pose[5]);

    vertices_[3]->GetParameters(speed_bias);
    Vj  = Vec3(speed_bias[0], speed_bias[1], speed_bias[2]);
    Baj = Vec3(speed_bias[3], speed_bias[4], speed_bias[5]);
    Bgj = Vec3(speed_bias[6], speed_bias[7], speed_bias[8]);
}

}  // namespace SimpleVIO
