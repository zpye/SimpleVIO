#ifndef SIMPLE_VIO_OPTIMIZE_BASE_VERTEX_H_
#define SIMPLE_VIO_OPTIMIZE_BASE_VERTEX_H_

#include <cstddef>
#include <string>

namespace SimpleVIO {

extern size_t global_vertex_id;

class BaseVertex {
public:
    BaseVertex() {
        id_ = global_vertex_id++;
    }

    virtual ~BaseVertex() {}

public:
    virtual std::string TypeInfo() const = 0;

    virtual int Dimension() const = 0;

    virtual int LocalDimension() const = 0;

    virtual void Plus(double *delta) = 0;

    virtual void SetParameters(double *params) = 0;

    virtual int GetParameters(double *params) const = 0;

    virtual double ComputeParametersNorm() const = 0;

    virtual void BackUpParameters() = 0;

    virtual void RollBackParameters() = 0;

public:
    void SetOrderingId(size_t ordering_id) {
        ordering_id_ = ordering_id;
    }

    size_t OrderingId() const {
        return ordering_id_;
    }

    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }

    bool IsFixed() const {
        return fixed_;
    }

    size_t Id() const {
        return id_;
    }

protected:
    size_t ordering_id_ = 0;  // start position of jacobian block

    bool fixed_ = false;  // fixed during optimization

    size_t id_;  // global id
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_BASE_VERTEX_H_
