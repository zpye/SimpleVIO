#ifndef SIMPLE_VIO_OPTIMIZE_EDGE_H_
#define SIMPLE_VIO_OPTIMIZE_EDGE_H_

#include "base_edge.h"

#include "utils//eigen_types.h"

namespace SimpleVIO {

template<int ResidualSize, int VertexNumber>
class Edge : public BaseEdge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using ResidualVec    = Eigen::Matrix<double, ResidualSize, 1>;
    using JacobianMat    = Eigen::Matrix<double, ResidualSize, Eigen::Dynamic>;
    using InformationMat = Eigen::Matrix<double, ResidualSize, ResidualSize>;

public:
    Edge() {
        vertices_.resize(VertexNumber);
        jacobians_.resize(VertexNumber);

        information_.setIdentity();
        sqrt_information_.setIdentity();
    }

    virtual ~Edge() {}

public:
    virtual std::string TypeInfo() const override = 0;

    virtual void ComputeResidual() override = 0;

    virtual int GetResidual(double *residual) const override {
        Eigen::Map<ResidualVec> r(residual);
        r = residual_;

        return 0;
    }

    virtual void ComputeJacobians() override = 0;

    virtual int GetJacobians(int i, double *jacobian) const override {
        Eigen::Map<JacobianMat> J(jacobian,
                                  ResidualSize,
                                  vertices_[i]->LocalDimension());
        J = jacobians_[i];

        return 0;
    }

    virtual void SetInformation(double *information) override {
        Eigen::Map<InformationMat> info(information);
        information_ = info;

        // sqrt
        sqrt_information_ =
            Eigen::LLT<InformationMat>(information_).matrixL().transpose();
    }

    virtual int GetInformation(double *information) const override {
        Eigen::Map<InformationMat> info(information);
        info = information_;

        return 0;
    }

    virtual int GetSqrtInformation(double *information) const override {
        Eigen::Map<InformationMat> info(information);
        info = sqrt_information_;

        return 0;
    }

public:
    virtual int GetResidualDimention() const override {
        return ResidualSize;
    }

    virtual size_t JacobiansSize() const override {
        return VertexNumber;
    }

    virtual double Chi2() const override {
        Scalar result = residual_.transpose() * information_ * residual_;
        return result(0);
    }

    virtual double RobustChi2() const override {
        double chi2 = this->Chi2();
        if (loss_function_) {
            Vec3 rho;
            loss_function_->Compute(chi2, rho);
            chi2 = rho[0];
        }

        return chi2;
    }

    virtual void ComputeRobustInformation(double &drho,
                                          double *information) const override {
        Eigen::Map<InformationMat> info(information);
        if (loss_function_) {
            double chi2 = this->Chi2();

            Vec3 rho;
            loss_function_->Compute(chi2, rho);
            ResidualVec weight_err = sqrt_information_ * residual_;

            InformationMat robust_info;
            robust_info.setIdentity();
            robust_info *= rho[1];

            if (rho[1] + 2.0 * rho[2] * chi2 > 0.0) {
                robust_info +=
                    2.0 * rho[2] * weight_err * weight_err.transpose();
            }

            drho = rho[1];
            info = robust_info * information_;
        } else {
            drho = 1.0;
            info = information_;
        }
    }

protected:
    ResidualVec residual_;

    std::vector<JacobianMat> jacobians_;

    InformationMat information_;

    InformationMat sqrt_information_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_EDGE_H_
