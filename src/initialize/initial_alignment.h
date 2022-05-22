#ifndef SIMPLE_VIO_INITIALIZE_INITIAL_ALIGNMENT_H_
#define SIMPLE_VIO_INITIALIZE_INITIAL_ALIGNMENT_H_

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include "feature/feature_manager.h"
#include "imu/imu_integration.h"
#include "utils/eigen_types.h"

namespace SimpleVIO {

struct ImageIMUFrame {
    double t;

    std::map<int, std::vector<FeatureObservation>> feature_observations;

    std::shared_ptr<IMUIntegration> pre_integration;

    Mat33 R;
    Vec3 T;

    bool is_key_frame = false;
};

bool VisualIMUAlignment(const size_t window_size,
                        std::map<double, ImageIMUFrame> &all_image_frame,
                        std::vector<Vec3> &Bgs,
                        Vec3 &g,
                        VecX &x,
                        const Vec3 &G_const,
                        const Vec3 &tic0);

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_INITIALIZE_INITIAL_ALIGNMENT_H_
