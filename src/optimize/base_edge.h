#ifndef SIMPLE_VIO_OPTIMIZE_BASE_EDGE_H_
#define SIMPLE_VIO_OPTIMIZE_BASE_EDGE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "base_vertex.h"
#include "loss_function.h"
#include "utils/logger.h"

namespace SimpleVIO {

extern size_t global_edge_id;

class BaseEdge {
public:
    BaseEdge() {
        id_ = global_edge_id++;
    }

    virtual ~BaseEdge() {}

    void SetVerticesTypes(const std::vector<std::string> &vertices_types) {
        vertices_types_ = vertices_types;
    }

    bool CheckValid() {
        if (!vertices_types_.empty()) {
            // check type info
            for (size_t i = 0; i < vertices_.size(); ++i) {
                if (vertices_types_[i] != vertices_[i]->TypeInfo()) {
                    LOGE("SimpleVIO",
                         "Vertex type does not match, should be [%s], but set "
                         "to [%s]",
                         vertices_types_[i].c_str(),
                         vertices_[i]->TypeInfo().c_str());

                    return false;
                }
            }
        }

        return true;
    }

public:
    virtual std::string TypeInfo() const = 0;

    virtual void ComputeResidual() = 0;

    virtual int GetResidual(double *residual) const = 0;

    virtual int GetResidualDimention() const = 0;

    virtual void ComputeJacobians() = 0;

    virtual int GetJacobians(int i, double *jacobian) const = 0;

    virtual size_t JacobiansSize() const = 0;

    virtual double Chi2() const = 0;

    virtual double RobustChi2() const = 0;

    virtual void SetInformation(double *information) = 0;

    virtual int GetInformation(double *information) const = 0;

    virtual int GetSqrtInformation(double *information) const = 0;

    virtual void ComputeRobustInformation(double &drho,
                                          double *information) const = 0;

public:
    void ClearVertices() {
        vertices_.clear();
    }

    void SetVertex(int i, const std::shared_ptr<BaseVertex> &vertex) {
        vertices_[i] = vertex;
    }

    std::shared_ptr<BaseVertex> &GetVertex(int i) {
        return vertices_[i];
    }

    void SetVertices(const std::vector<std::shared_ptr<BaseVertex>> &vertices) {
        vertices_ = vertices;
    }

    std::vector<std::shared_ptr<BaseVertex>> &GetVertices() {
        return vertices_;
    }

    size_t VerticesSize() const {
        return vertices_.size();
    }

    void SetLossFunction(const std::shared_ptr<LossFunction> &loss_function) {
        loss_function_ = loss_function;
    }

    std::shared_ptr<LossFunction> &GetLossFunction() {
        return loss_function_;
    }

    void SetOrderingId(size_t ordering_id) {
        ordering_id_ = ordering_id;
    };

    size_t OrderingId() const {
        return ordering_id_;
    }

    size_t Id() const {
        return id_;
    }

protected:
    std::vector<std::string> vertices_types_;

    std::vector<std::shared_ptr<BaseVertex>> vertices_;

    std::shared_ptr<LossFunction> loss_function_;

    size_t ordering_id_ = 0;  // start position of jacobian block

    size_t id_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_BASE_EDGE_H_
