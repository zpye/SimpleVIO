#ifndef SIMPLE_VIO_SYSTEM_H_
#define SIMPLE_VIO_SYSTEM_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>

#include "utils/eigen_types.h"

#include "estimator/estimator.h"
#include "feature/feature_tracker.h"
#include "parameters.h"

#if defined(DRAW_RESULT)
#include <pangolin/pangolin.h>
#endif

namespace SimpleVIO {

// imu for vio
struct IMU_MSG {
    double header;
    Vec3 linear_acceleration;
    Vec3 angular_velocity;
};

using IMUPtr      = std::shared_ptr<IMU_MSG>;
using IMUConstPtr = std::shared_ptr<IMU_MSG const>;

// image for vio
struct IMG_MSG {
    double header;
    std::vector<Vec3> points;
    std::vector<int> id_of_point;
    std::vector<float> u_of_point;
    std::vector<float> v_of_point;
    std::vector<float> velocity_x_of_point;
    std::vector<float> velocity_y_of_point;
};

using ImagePtr      = std::shared_ptr<IMG_MSG>;
using ImageConstPtr = std::shared_ptr<IMG_MSG const>;

using MeasData =
    std::vector<std::pair<std::vector<IMUConstPtr>, ImageConstPtr>>;

class System {
public:
    System();

    ~System();

public:
    bool Initialize(const std::string &config_file);

    void Finish();

    void PubImageData(const double timestamp_sec, cv::Mat &img);
    void PubImageData(const double timestamp_sec,
                      const std::string &keyframes_file);

    void PubIMUData(const double timestamp_sec,
                    const Vec3 &acc,
                    const Vec3 &gyr);

private:
    size_t GetMeasurements(MeasData &meas_data);

    // thread: visual-inertial odometry
    void MainProcessLoop();

private:
    bool finished_ = false;

    Parameters params_;

    // publish imu
    double last_imu_t_ = -1.0;

    // publish image
    bool has_first_image_    = false;
    double last_image_time_  = -1.0;
    double first_image_time_ = -1.0;
    int pub_cnt_             = 1;

    // feature tracker
    std::vector<FeatureTracker> feature_trackers_;

    // estimator
    std::mutex estimator_mtx_;
    Estimator estimator_;

    // output
    std::ofstream ofs_pose_;
    std::vector<Vec3> path_to_draw_;

private:
    // measure
    std::shared_timed_mutex measure_mtx_;
    std::condition_variable_any measure_cond_;
    std::queue<IMUConstPtr> imu_buf_;
    std::queue<ImageConstPtr> image_buf_;

    // main process loop
    std::atomic_bool stop_main_process_loop_;
    std::thread main_process_loop_thread_;

    double current_time_ = -1.0;

private:
#if defined(DRAW_RESULT)
    void Draw();

    pangolin::OpenGlRenderState s_cam_;
    pangolin::View d_cam_;

    std::thread draw_thread_;

#endif
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_SYSTEM_H_
