#include "parameters.h"

#include <cmath>
#include <sstream>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "utils/logger.h"

namespace SimpleVIO {

bool Parameters::ReadParameters(const std::string& config_file) {
    cv::FileStorage fs_settings(config_file, cv::FileStorage::READ);
    if (!fs_settings.isOpened()) {
        LOGE("SimpleVIO", "ReadParameters Error, Wrong file path!!!");
        return false;
    }

    this->config_file = config_file;

    // system
    cv::read(fs_settings["freq"], frequency, 10);

    cv::read(fs_settings["image_topic"], image_topic, "");
    cv::read(fs_settings["imu_topic"], imu_topic, "");

    std::string output_path;
    cv::read(fs_settings["output_path"], output_path, ".");
    vins_result_path = output_path + "/vins_result_no_loop.txt";

    cv::read(fs_settings["show_track"], show_track, 0);

    // imu
    cv::read(fs_settings["acc_n"], acc_sigma_n, 1e-10);
    cv::read(fs_settings["acc_w"], acc_sigma_w, 1e-10);
    cv::read(fs_settings["gyr_n"], gyr_sigma_n, 1e-10);
    cv::read(fs_settings["gyr_w"], gyr_sigma_w, 1e-10);
    cv::read(fs_settings["acc_bias_threshold"], acc_bias_threshold, 0.1);
    cv::read(fs_settings["gyr_bias_threshold"], gyr_bias_threshold, 0.1);
    cv::read(fs_settings["g_norm"], G.z(), 9.81);

    // discrete noise
    // acc_sigma_n *= std::sqrt(200.0);
    // acc_sigma_w /= std::sqrt(200.0);
    // gyr_sigma_n *= std::sqrt(200.0);
    // gyr_sigma_w /= std::sqrt(200.0);

    // camera
    // TODO: support more than 1 camera
    cv::read(fs_settings["num_cameras"], num_of_camera, 1);

    camera_names.resize(num_of_camera);
    cv::read(fs_settings["camera_name"], camera_names[0], "unknown");

    camera_types.resize(num_of_camera);
    cv::read(fs_settings["model_type"], camera_types[0], "default");

    cv::read(fs_settings["rolling_shutter"], rolling_shutter, 0);
    if (0 != rolling_shutter) {
        cv::read(fs_settings["rolling_shutter_tr"], rolling_shutter_tr, 0.0);
    } else {
        rolling_shutter_tr = 0.0;
    }

    cv::read(fs_settings["fisheye"], fisheye, 0);

    // extrinsic calibration
    cv::read(fs_settings["estimate_extrinsic"], estimate_extrinsic, 0);
    if (2 == estimate_extrinsic) {
        LOGW("SimpleVIO",
             "Have no prior about extrinsic param, calibrate extrinsic param!");

        initial_ric.push_back(Mat33::Identity());
        initial_tic.push_back(Vec3::Zero());

        extrinsic_calibration_result_path =
            output_path + "/extrinsic_parameter.csv";
    } else {
        if (1 == estimate_extrinsic) {
            LOGW("SimpleVIO", "Optimize extrinsic param around initial guess!");

            extrinsic_calibration_result_path =
                output_path + "/extrinsic_parameter.csv";
        }

        if (0 == estimate_extrinsic) {
            LOGI("SimpleVIO", "Fix extrinsic param");
        }

        cv::Mat cv_R;
        cv::Mat cv_T;
        cv::read(fs_settings["extrinsicRotation"], cv_R);
        cv::read(fs_settings["extrinsicTranslation"], cv_T);

        Mat33 eigen_R;
        Vec3 eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);

        Qd Q(eigen_R);
        eigen_R = Q.normalized();

        initial_ric.push_back(eigen_R);
        initial_tic.push_back(eigen_T);
    }

    // feature tracking
    cv::read(fs_settings["image_height"], image_height, -1);
    cv::read(fs_settings["image_width"], image_width, -1);

    cv::read(fs_settings["equalize"], do_equalize, 0);

    cv::read(fs_settings["max_cnt"], max_feature_cnt, 150);
    cv::read(fs_settings["min_dist"], min_feature_displacement, 30);

    // estimation
    cv::read(fs_settings["window_size"], window_size, 10);

    cv::read(fs_settings["init_depth"], init_depth, 5.0);
    cv::read(fs_settings["keyframe_parallax"], min_parallax, 10.0);

    // optimization
    cv::read(fs_settings["optimization_method"], opt_method, 0);

    // adjust parallax
    auto proj_params = fs_settings["projection_parameters"];
    if (!proj_params.empty()) {
        cv::read(proj_params["fx"], fx, 460.0);
        cv::read(proj_params["fy"], fy, 460.0);
        cv::read(proj_params["cx"], cx, 376.0);
        cv::read(proj_params["cy"], cy, 240.0);

        focal_length = std::sqrt(fx * fy);
        min_parallax /= focal_length;
    }

    cv::read(fs_settings["estimate_td"], estimate_td, 0);
    cv::read(fs_settings["td"], initial_td, 0.0);

    cv::read(fs_settings["max_solver_time"], max_solve_time, 0.04);
    cv::read(fs_settings["max_num_iterations"], max_solve_iterations, 8);

    cv::read(fs_settings["F_threshold"], fundamental_ransac_threshold, 1.0);

    fs_settings.release();

    return true;
}

void Parameters::ShowParameters() {
    std::stringstream ss;

    LOGI("SimpleVIO", "config_file: %s", config_file.c_str());

    LOGI("SimpleVIO", "frequency: %d", frequency);
    LOGI("SimpleVIO", "show track: %d", show_track);

    LOGI("SimpleVIO", "acc_sigma_n: %lf", acc_sigma_n);
    LOGI("SimpleVIO", "acc_sigma_w: %lf", acc_sigma_w);
    LOGI("SimpleVIO", "gyr_sigma_n: %lf", gyr_sigma_n);
    LOGI("SimpleVIO", "gyr_sigma_w: %lf", gyr_sigma_w);

    LOGI("SimpleVIO", "acc_bias_threshold: %lf", acc_bias_threshold);
    LOGI("SimpleVIO", "gyr_bias_threshold: %lf", gyr_bias_threshold);

    ss.clear();
    ss.str("");
    ss << G.transpose();
    LOGI("SimpleVIO", "G: %s", ss.str().c_str());

    LOGI("SimpleVIO", "num_of_camera: %d", num_of_camera);
    for (int i = 0; i < num_of_camera; ++i) {
        LOGI("SimpleVIO", "camera name %d: %s", i, camera_names[i].c_str());
    }

    LOGI("SimpleVIO", "rolling_shutter: %d", rolling_shutter);
    LOGI("SimpleVIO", "rolling_shutter_tr: %lf", rolling_shutter_tr);
    LOGI("SimpleVIO", "fisheye: %d", fisheye);

    LOGI("SimpleVIO", "estimate_extrinsic: %d", estimate_extrinsic);

    for (int i = 0; i < num_of_camera; ++i) {
        ss.clear();
        ss.str("");
        ss << initial_ric[i];
        LOGI("SimpleVIO", "initial_ric %d: %s", i, ss.str().c_str());

        ss.clear();
        ss.str("");
        ss << initial_tic[i].transpose();
        LOGI("SimpleVIO", "initial_tic %d: %s", i, ss.str().c_str());
    }

    LOGI("SimpleVIO", "image_height: %d", image_height);
    LOGI("SimpleVIO", "image_width: %d", image_width);

    LOGI("SimpleVIO", "do_equalize: %d", do_equalize);

    LOGI("SimpleVIO", "max_feature_cnt: %d", max_feature_cnt);
    LOGI("SimpleVIO", "min_feature_displacement: %d", min_feature_displacement);

    LOGI("SimpleVIO", "window_size: %d", window_size);

    LOGI("SimpleVIO", "init_depth: %lf", init_depth);
    LOGI("SimpleVIO", "min_parallax: %lf", min_parallax);

    LOGI("SimpleVIO", "estimate_td: %d", estimate_td);
    LOGI("SimpleVIO", "initial_td: %lf", initial_td);

    LOGI("SimpleVIO", "max_solve_time: %lf", max_solve_time);
    LOGI("SimpleVIO", "max_solve_iterations: %d", max_solve_iterations);

    LOGI("SimpleVIO",
         "fundamental_ransac_threshold: %lf",
         fundamental_ransac_threshold);

    LOGI("SimpleVIO", "optimization method: %d", opt_method);
}

}  // namespace SimpleVIO