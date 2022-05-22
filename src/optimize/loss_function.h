#ifndef SIMPLE_VIO_OPTIMIZE_LOSS_FUNCTION_H_
#define SIMPLE_VIO_OPTIMIZE_LOSS_FUNCTION_H_

#include <Eigen/Eigen>

namespace SimpleVIO {

/**
 * compute the scaling factor for a error:
 * The error is e^T Omega e
 * The output rho is
 * rho[0]: The actual scaled error value
 * rho[1]: First derivative of the scaling function
 * rho[2]: Second derivative of the scaling function
 *
 */
class LossFunction {
public:
    virtual ~LossFunction() {}

    virtual void Compute(double err2, Eigen::Vector3d &rho) const = 0;
};

/**
 * same as setting nullptr
 * TrivalLoss(e) = e^2
 */
class TrivalLoss : public LossFunction {
public:
    virtual void Compute(double err2, Eigen::Vector3d &rho) const override {
        // TODO:: whether multiply 1/2
        rho[0] = err2;
        rho[1] = 1.0;
        rho[2] = 0.0;
    }
};

/**
 * Huber loss
 *
 * Huber(e) = e^2                      if e <= delta
 * huber(e) = delta*(2*e - delta)      if e > delta
 */
class HuberLoss : public LossFunction {
public:
    explicit HuberLoss(double delta) : delta_(delta) {}

    virtual void Compute(double err2, Eigen::Vector3d &rho) const override;

private:
    double delta_;
};

/*
 * Cauchy loss
 */
class CauchyLoss : public LossFunction {
public:
    explicit CauchyLoss(double delta) : delta_(delta) {}

    virtual void Compute(double err2, Eigen::Vector3d &rho) const override;

private:
    double delta_;
};

/*
 * Tukey loss
 */
class TukeyLoss : public LossFunction {
public:
    explicit TukeyLoss(double delta) : delta_(delta) {}

    virtual void Compute(double err2, Eigen::Vector3d &rho) const override;

private:
    double delta_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_LOSS_FUNCTION_H_
