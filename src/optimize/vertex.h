#ifndef SIMPLE_VIO_OPTIMIZE_VERTEX_H_
#define SIMPLE_VIO_OPTIMIZE_VERTEX_H_

#include "base_vertex.h"

#include <Eigen/Eigen>

namespace SimpleVIO {

template<int GlobalSize, int LocalSize>
class Vertex : public BaseVertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using ParamVec = Eigen::Matrix<double, GlobalSize, 1>;
    using LocalVec = Eigen::Matrix<double, LocalSize, 1>;

public:
    Vertex() {}

    virtual ~Vertex() {}

public:
    virtual std::string TypeInfo() const override {
        return "Vertex";
    }

    virtual int Dimension() const override {
        return GlobalSize;
    }

    virtual int LocalDimension() const override {
        return LocalSize;
    }

    virtual void Plus(double *delta) override {
        // default LocalSize and GlobalSize are equal
        Eigen::Map<ParamVec> d(delta);
        parameters_ += d;
    }

    virtual void SetParameters(double *params) override {
        parameters_ = Eigen::Map<ParamVec>(params);
    }

    virtual int GetParameters(double *params) const override {
        Eigen::Map<ParamVec> p(params);
        p = parameters_;

        return 0;
    }

    virtual double ComputeParametersNorm() const override {
        return parameters_.norm();
    }

    virtual void BackUpParameters() override {
        parameters_backup_ = parameters_;
    }

    virtual void RollBackParameters() override {
        parameters_ = parameters_backup_;
    }

protected:
    ParamVec parameters_;
    ParamVec parameters_backup_;  // for rolling back
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_VERTEX_H_
