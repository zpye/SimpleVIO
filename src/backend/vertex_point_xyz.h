#ifndef SIMPLE_VIO_BACKEND_VERTEX_POINT_XYZ_H_
#define SIMPLE_VIO_BACKEND_VERTEX_POINT_XYZ_H_

#include "optimize/vertex.h"

namespace SimpleVIO {

class VertexPointXYZ : public Vertex<3, 3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPointXYZ() {}

public:
    virtual std::string TypeInfo() const override {
        return "VertexPointXYZ";
    }
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_BACKEND_VERTEX_POINT_XYZ_H_
