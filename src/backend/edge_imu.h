#ifndef SIMPLE_VIO_BACKEND_EDGE_IMU_H_
#define SIMPLE_VIO_BACKEND_EDGE_IMU_H_

#include <memory>

#include "optimize/edge.h"

#include "imu/imu_integration.h"
#include "utils/eigen_types.h"

namespace SimpleVIO {

// connected with 4 vertices: Pose(i), SpeedBias(i), Pose(j), SpeedBias(j)
class EdgeImu : public Edge<15, 4> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    explicit EdgeImu(const std::shared_ptr<IMUIntegration> &imu_pre_integration)
        : imu_pre_integration_(imu_pre_integration) {
        SetVerticesTypes(std::vector<std::string>{"VertexPose",
                                                  "VertexSpeedBias",
                                                  "VertexPose",
                                                  "VertexSpeedBias"});
    }

public:
    virtual std::string TypeInfo() const override {
        return "EdgeImu";
    }

    virtual void ComputeResidual() override;

    virtual void ComputeJacobians() override;

private:
    void GetParams(Vec3 &Pi,
                   Qd &Qi,
                   Vec3 &Vi,
                   Vec3 &Bai,
                   Vec3 &Bgi,
                   Vec3 &Pj,
                   Qd &Qj,
                   Vec3 &Vj,
                   Vec3 &Baj,
                   Vec3 &Bgj);

private:
    std::shared_ptr<IMUIntegration> imu_pre_integration_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_BACKEND_EDGE_IMU_H_