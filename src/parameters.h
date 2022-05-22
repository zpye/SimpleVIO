#ifndef SIMPLE_VIO_PARAMETERS_H_
#define SIMPLE_VIO_PARAMETERS_H_

#include <string>
#include <vector>

#include "utils/eigen_types.h"

namespace SimpleVIO {

struct Parameters {
    std::string config_file;

    // system
    int frequency;

    std::string image_topic;
    std::string imu_topic;

    std::string vins_result_path;

    int show_track;

    // imu
    double acc_sigma_n;
    double acc_sigma_w;
    double gyr_sigma_n;
    double gyr_sigma_w;

    double acc_bias_threshold;
    double gyr_bias_threshold;

    Vec3 G{0.0, 0.0, 9.8};

    // camera
    int num_of_camera;
    std::vector<std::string> camera_names;
    std::vector<std::string> camera_types;

    // TODO: multi camera
    double fx;
    double fy;
    double cx;
    double cy;
    double focal_length;

    int rolling_shutter;
    double rolling_shutter_tr;

    int fisheye;
    std::string fisheye_mask;

    // extrinsic calibration
    int estimate_extrinsic;
    std::string extrinsic_calibration_result_path;

    std::vector<Mat33> initial_ric;
    std::vector<Vec3> initial_tic;

    // feature tracking
    int image_height;
    int image_width;
    int do_equalize;

    int max_feature_cnt;
    int min_feature_displacement;

    // estimation
    int window_size;

    double init_depth;
    double min_parallax;

    int estimate_td;
    double initial_td;

    double max_solve_time;
    int max_solve_iterations;

    double fundamental_ransac_threshold;

    // optimization
    int opt_method;

public:
    bool ReadParameters(const std::string& config_file);

    void ShowParameters();
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_PARAMETERS_H_
