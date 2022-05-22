#ifndef SIMPLE_VIO_BACKEND_VERTEX_INVERSE_DEPTH_H_
#define SIMPLE_VIO_BACKEND_VERTEX_INVERSE_DEPTH_H_

#include "optimize/vertex.h"

#include "utils/logger.h"

namespace SimpleVIO {

class VertexInverseDepth : public Vertex<1, 1> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexInverseDepth() {}

public:
    virtual std::string TypeInfo() const override {
        return "VertexInverseDepth";
    }
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_BACKEND_VERTEX_INVERSE_DEPTH_H_