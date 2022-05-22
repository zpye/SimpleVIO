#include "imu_integration.h"

#include "imu_utils.h"
#include "utils/logger.h"
#include "utils/rotation_utils.h"

namespace SimpleVIO {

IMUIntegration::IMUIntegration(const Vec3 &acc_0,
                               const Vec3 &gyr_0,
                               const Vec3 &linearized_ba,
                               const Vec3 &linearized_bg,
                               const Vec3 &G,
                               const double acc_noise_sigma,
                               const double gyr_noise_sigma,
                               const double acc_random_walk_sigma,
                               const double gyr_random_walk_sigma)
    : linearized_acc_{acc_0},
      linearized_gyr_{gyr_0},
      acc_0_(acc_0),
      gyr_0_(gyr_0),
      linearized_ba_(linearized_ba),
      linearized_bg_(linearized_bg),
      G_(G) {
    const Mat33 mat_I = Mat33::Identity();

    noise_.block<3, 3>(0, 0) = (acc_noise_sigma * acc_noise_sigma) * mat_I;
    noise_.block<3, 3>(3, 3) = (gyr_noise_sigma * gyr_noise_sigma) * mat_I;
    noise_.block<3, 3>(6, 6) = (acc_noise_sigma * acc_noise_sigma) * mat_I;
    noise_.block<3, 3>(9, 9) = (gyr_noise_sigma * gyr_noise_sigma) * mat_I;
    noise_.block<3, 3>(12, 12) =
        (acc_random_walk_sigma * acc_random_walk_sigma) * mat_I;
    noise_.block<3, 3>(15, 15) =
        (gyr_random_walk_sigma * gyr_random_walk_sigma) * mat_I;
}

void IMUIntegration::AddMeasure(const double dt,
                                const Vec3 &acc,
                                const Vec3 &gyr) {
    dt_buf_.push_back(dt);
    acc_buf_.push_back(acc);
    gyr_buf_.push_back(gyr);

    Propagate(dt, acc, gyr);
}

void IMUIntegration::Propagate(const double dt,
                               const Vec3 &acc_1,
                               const Vec3 &gyr_1) {
    acc_1_ = acc_1;
    gyr_1_ = gyr_1;

    Vec3 out_delta_p;
    Qd out_delta_q;
    Vec3 out_delta_v;
    Vec3 out_linearized_ba;
    Vec3 out_linearized_bg;

    MidPointIntegration(dt,
                        acc_0_,
                        gyr_0_,
                        acc_1,
                        gyr_1,
                        delta_p_,
                        delta_q_,
                        delta_v_,
                        linearized_ba_,
                        linearized_bg_,
                        out_delta_p,
                        out_delta_q,
                        out_delta_v,
                        out_linearized_ba,
                        out_linearized_bg,
                        true);

    // update
    delta_p_       = out_delta_p;
    delta_q_       = out_delta_q.normalized();
    delta_v_       = out_delta_v;
    linearized_ba_ = out_linearized_ba;
    linearized_bg_ = out_linearized_bg;

    sum_dt_ += dt;
    acc_0_ = acc_1_;
    gyr_0_ = gyr_1_;
}

void IMUIntegration::Repropagate(const Vec3 &linearized_ba,
                                 const Vec3 &linearized_bg) {
    // reset
    sum_dt_ = 0.0;
    acc_0_  = linearized_acc_;
    gyr_0_  = linearized_gyr_;

    linearized_ba_ = linearized_ba;
    linearized_bg_ = linearized_bg;

    delta_p_.setZero();
    delta_q_.setIdentity();
    delta_v_.setZero();

    jacobian_.setIdentity();
    covariance_.setZero();

    // repropagate
    for (size_t i = 0; i < dt_buf_.size(); ++i) {
        Propagate(dt_buf_[i], acc_buf_[i], gyr_buf_[i]);
    }
}

void IMUIntegration::MidPointIntegration(const double dt,
                                         const Vec3 &acc_0,
                                         const Vec3 &gyr_0,
                                         const Vec3 &acc_1,
                                         const Vec3 &gyr_1,
                                         const Vec3 &delta_p,
                                         const Qd &delta_q,
                                         const Vec3 &delta_v,
                                         const Vec3 &linearized_ba,
                                         const Vec3 &linearized_bg,
                                         Vec3 &result_delta_p,
                                         Qd &result_delta_q,
                                         Vec3 &result_delta_v,
                                         Vec3 &result_linearized_ba,
                                         Vec3 &result_linearized_bg,
                                         bool update_jacobian) {
    Vec3 un_gyr   = 0.5 * (gyr_0 + gyr_1) - linearized_bg;
    Vec3 un_acc_0 = acc_0 - linearized_ba;
    Vec3 un_acc_1 = acc_1 - linearized_ba;

    result_delta_q = delta_q * Qd(1.0,
                                  un_gyr(0) * dt / 2.0,
                                  un_gyr(1) * dt / 2.0,
                                  un_gyr(2) * dt / 2.0);

    Vec3 un_acc_w  = 0.5 * (delta_q * un_acc_0 + result_delta_q * un_acc_1);
    result_delta_p = delta_p + delta_v * dt + 0.5 * un_acc_w * dt * dt;
    result_delta_v = delta_v + un_acc_w * dt;

    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    if (update_jacobian) {
        // cross product matrix
        const Mat33 R_w_x   = SkewSymmetric(un_gyr);
        const Mat33 R_a_0_x = SkewSymmetric(un_acc_0);
        const Mat33 R_a_1_x = SkewSymmetric(un_acc_1);

        const Mat33 mat_I = Mat33::Identity();
        const Mat33 dq_0  = delta_q.toRotationMatrix();
        const Mat33 dq_1  = result_delta_q.toRotationMatrix();

        JacobianTransMatrix F = JacobianTransMatrix::Zero();
        F.block<3, 3>(0, 0)   = mat_I;
        F.block<3, 3>(0, 3) =
            -0.25 * dq_0 * R_a_0_x * dt * dt +
            -0.25 * dq_1 * R_a_1_x * (mat_I - R_w_x * dt) * dt * dt;
        F.block<3, 3>(0, 6)  = mat_I * dt;
        F.block<3, 3>(0, 9)  = -0.25 * (dq_0 + dq_1) * dt * dt;
        F.block<3, 3>(0, 12) = -0.25 * dq_1 * R_a_1_x * dt * dt * -dt;
        F.block<3, 3>(3, 3)  = mat_I - R_w_x * dt;
        F.block<3, 3>(3, 12) = -1.0 * mat_I * dt;
        F.block<3, 3>(6, 3)  = -0.5 * dq_0 * R_a_0_x * dt +
                              -0.5 * dq_1 * R_a_1_x * (mat_I - R_w_x * dt) * dt;
        F.block<3, 3>(6, 6)   = mat_I;
        F.block<3, 3>(6, 9)   = -0.5 * (dq_0 + dq_1) * dt;
        F.block<3, 3>(6, 12)  = 0.5 * dq_1 * R_a_1_x * dt * dt;
        F.block<3, 3>(9, 9)   = mat_I;
        F.block<3, 3>(12, 12) = mat_I;

        NoiseTransMatrix V    = NoiseTransMatrix::Zero();
        V.block<3, 3>(0, 0)   = 0.25 * dq_0 * dt * dt;
        V.block<3, 3>(0, 3)   = 0.25 * -dq_1 * R_a_1_x * dt * dt * 0.5 * dt;
        V.block<3, 3>(0, 6)   = 0.25 * dq_1 * dt * dt;
        V.block<3, 3>(0, 9)   = V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3)   = 0.5 * mat_I * dt;
        V.block<3, 3>(3, 9)   = 0.5 * mat_I * dt;
        V.block<3, 3>(6, 0)   = 0.5 * dq_0 * dt;
        V.block<3, 3>(6, 3)   = 0.5 * -dq_1 * R_a_1_x * dt * 0.5 * dt;
        V.block<3, 3>(6, 6)   = 0.5 * dq_1 * dt;
        V.block<3, 3>(6, 9)   = V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12)  = mat_I * dt;
        V.block<3, 3>(12, 15) = mat_I * dt;

        // update Jacobian and Covariance
        jacobian_ = F * jacobian_;
        covariance_ =
            F * covariance_ * F.transpose() + V * noise_ * V.transpose();
    }
}

IMUIntegration::ResidualVector IMUIntegration::ComputeResidual(
    const Vec3 &Pi,
    const Qd &Qi,
    const Vec3 &Vi,
    const Vec3 &Bai,
    const Vec3 &Bgi,
    const Vec3 &Pj,
    const Qd &Qj,
    const Vec3 &Vj,
    const Vec3 &Baj,
    const Vec3 &Bgj) {
    Mat33 dp_dba = jacobian_.block<3, 3>(StateOrder::O_P, StateOrder::O_BA);
    Mat33 dp_dbg = jacobian_.block<3, 3>(StateOrder::O_P, StateOrder::O_BG);

    Mat33 dq_dbg = jacobian_.block<3, 3>(StateOrder::O_R, StateOrder::O_BG);

    Mat33 dv_dba = jacobian_.block<3, 3>(StateOrder::O_V, StateOrder::O_BA);
    Mat33 dv_dbg = jacobian_.block<3, 3>(StateOrder::O_V, StateOrder::O_BG);

    Vec3 dba = Bai - linearized_ba_;
    Vec3 dbg = Bgi - linearized_bg_;

    Qd corrected_delta_q   = delta_q_ * DeltaQ(dq_dbg * dbg);
    Vec3 corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
    Vec3 corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

    Qd Qi_inv = Qi.inverse();

    IMUIntegration::ResidualVector residual;
    residual.segment<3>(StateOrder::O_P) =
        Qi_inv * (Pj - Pi - Vi * sum_dt_ + 0.5 * G_ * sum_dt_ * sum_dt_) -
        corrected_delta_p;
    residual.segment<3>(StateOrder::O_R) =
        2.0 * (corrected_delta_q.inverse() * (Qi_inv * Qj)).vec();
    residual.segment<3>(StateOrder::O_V) =
        Qi_inv * (Vj - Vi + G_ * sum_dt_) - corrected_delta_v;
    residual.segment<3>(StateOrder::O_BA) = Baj - Bai;
    residual.segment<3>(StateOrder::O_BG) = Bgj - Bgi;

    /*LOGD("SimpleVIO",
         "imu residual (%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
         "%lf %lf)",
         residual(0),
         residual(1),
         residual(2),
         residual(3),
         residual(4),
         residual(5),
         residual(6),
         residual(7),
         residual(8),
         residual(9),
         residual(10),
         residual(11),
         residual(12),
         residual(13),
         residual(14));*/

    return residual;
}

}  // namespace SimpleVIO
