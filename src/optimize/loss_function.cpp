#include "loss_function.h"

#include <cmath>

namespace SimpleVIO {

void HuberLoss::Compute(double err2, Eigen::Vector3d &rho) const {
    double delta2 = delta_ * delta_;
    if (err2 <= delta2) {
        // inlier
        rho[0] = err2;
        rho[1] = 1.0;
        rho[2] = 0.0;
    } else {
        // outlier
        double err = std::sqrt(err2);  // absolut value of the error

        // rho(e) = 2 * delta * e^(1/2) - delta^2
        rho[0] = 2.0 * err * delta_ - delta2;
        // rho'(e)  = delta / sqrt(e)
        rho[1] = delta_ / err;
        // rho''(e) = -1 / (2*e^(3/2)) = -1/2 * (delta/sqrt(e)) / e
        rho[2] = -0.5 * rho[1] / err2;
    }
}

void CauchyLoss::Compute(double err2, Eigen::Vector3d &rho) const {
    double delta2      = delta_ * delta_;           // c^2
    double delta2_reci = 1.0 / delta2;              // 1/c^2
    double aux         = delta2_reci * err2 + 1.0;  // 1 + e^2/c^2

    rho[0] = delta2 * std::log(aux);                // c^2 * log( 1 + e^2/c^2 )
    rho[1] = 1.0 / aux;                             // rho'
    rho[2] = -delta2_reci * std::pow(rho[1], 2.0);  // rho''
}

void TukeyLoss::Compute(double err2, Eigen::Vector3d &rho) const {
    const double delta2 = delta_ * delta_;

    const double err = std::sqrt(err2);
    if (err <= delta_) {
        const double aux = err2 / delta2;

        rho[0] = delta2 * (1.0 - std::pow((1.0 - aux), 3.0)) / 3.0;
        rho[1] = std::pow((1.0 - aux), 2.0);
        rho[2] = -2.0 * (1.0 - aux) / delta2;
    } else {
        rho[0] = delta2 / 3.0;
        rho[1] = 0.0;
        rho[2] = 0.0;
    }
}

}  // namespace SimpleVIO
