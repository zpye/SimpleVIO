#ifndef SIMPLE_VIO_BACKEND_VERTEX_SPEED_BIAS_H_
#define SIMPLE_VIO_BACKEND_VERTEX_SPEED_BIAS_H_

#include "optimize/vertex.h"

namespace SimpleVIO {

/**
 * SpeedBias vertex
 * parameters: v, ba, bg 9 DoF
 */
class VertexSpeedBias : public Vertex<9, 9> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSpeedBias() {}

public:
    virtual std::string TypeInfo() const override {
        return "VertexSpeedBias";
    }
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_BACKEND_VERTEX_SPEED_BIAS_H_