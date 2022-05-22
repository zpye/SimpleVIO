#ifndef SIMPLE_VIO_IMU_IMU_INTEGRATION_H_
#define SIMPLE_VIO_IMU_IMU_INTEGRATION_H_

#include <cstddef>

#include <Eigen/Eigen>

#include "parameters.h"
#include "utils/eigen_types.h"

namespace SimpleVIO {

class IMUIntegration {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using ResidualVector   = Eigen::Matrix<double, 15, 1>;
    using JacobianMatrix   = Eigen::Matrix<double, 15, 15>;
    using CovarianceMatrix = Eigen::Matrix<double, 15, 15>;
    using NoiseMatrix      = Eigen::Matrix<double, 18, 18>;

    using JacobianTransMatrix = Eigen::Matrix<double, 15, 15>;
    using NoiseTransMatrix    = Eigen::Matrix<double, 15, 18>;

public:
    IMUIntegration(const Vec3 &acc_0,
                   const Vec3 &gyr_0,
                   const Vec3 &linearized_ba,
                   const Vec3 &linearized_bg,
                   const Vec3 &G,
                   const double acc_noise_sigma,
                   const double gyr_noise_sigma,
                   const double acc_random_walk_sigma,
                   const double gyr_random_walk_sigma);

    void AddMeasure(const double dt, const Vec3 &acc, const Vec3 &gyr);

    void Propagate(const double dt, const Vec3 &acc_1, const Vec3 &gyr_1);

    void Repropagate(const Vec3 &linearized_ba, const Vec3 &linearized_bg);

    void MidPointIntegration(const double dt,
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
                             bool update_jacobian);

    ResidualVector ComputeResidual(const Vec3 &Pi,
                                   const Qd &Qi,
                                   const Vec3 &Vi,
                                   const Vec3 &Bai,
                                   const Vec3 &Bgi,
                                   const Vec3 &Pj,
                                   const Qd &Qj,
                                   const Vec3 &Vj,
                                   const Vec3 &Baj,
                                   const Vec3 &Bgj);

    JacobianMatrix &GetJacobian() {
        return jacobian_;
    }

    CovarianceMatrix &GetCovariance() {
        return covariance_;
    }

    Vec3 &GetLinearizedBa() {
        return linearized_ba_;
    }

    Vec3 &GetLinearizedBg() {
        return linearized_bg_;
    }

    Vec3 &GetDeltaP() {
        return delta_p_;
    }

    Qd &GetDeltaQ() {
        return delta_q_;
    }

    Vec3 &GetDeltaV() {
        return delta_v_;
    }

    double GetSumDeltaTime() {
        return sum_dt_;
    }

    const Vec3 &GetGravity() {
        return G_;
    }

private:
    // first measure
    const Vec3 linearized_acc_;
    const Vec3 linearized_gyr_;

    const Vec3 G_;

    Vec3 acc_0_;
    Vec3 gyr_0_;
    Vec3 acc_1_;
    Vec3 gyr_1_;

    Vec3 linearized_ba_;
    Vec3 linearized_bg_;

    JacobianMatrix jacobian_     = JacobianMatrix::Identity();
    CovarianceMatrix covariance_ = CovarianceMatrix::Zero();
    NoiseMatrix noise_           = NoiseMatrix::Zero();

    double sum_dt_ = 0.0;

    Vec3 delta_p_ = Vec3::Zero();
    Qd delta_q_   = Qd::Identity();
    Vec3 delta_v_ = Vec3::Zero();

    std::vector<double> dt_buf_;
    std::vector<Vec3> acc_buf_;
    std::vector<Vec3> gyr_buf_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_IMU_IMU_INTEGRATION_H_