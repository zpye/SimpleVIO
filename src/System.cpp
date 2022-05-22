#include "System.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <set>

#include "optimize/problem.h"

#include "utils/logger.h"
#include "utils/timer.h"

namespace SimpleVIO {

System::System() {}

System::~System() {
    if (!finished_) {
        Finish();
    }

    // debugging
    LOGD("SimpleVIO",
         "Average Hessian Cost %lf ms",
         Problem::t_total_hessian_cost_ / Problem::total_solve_count_);
    LOGD("SimpleVIO",
         "Average Solve Cost %lf ms",
         Problem::t_total_solve_cost_ / Problem::total_solve_count_);
}

bool System::Initialize(const std::string &config_file) {
    LOGI("SimpleVIO", "System Start Initialize");

    if (!params_.ReadParameters(config_file)) {
        LOGE("SimpleVIO", "System ReadParameters Error!!!");

        return false;
    }

    params_.ShowParameters();

    // feature tracker
    feature_trackers_.resize(params_.num_of_camera);
    for (auto &ft : feature_trackers_) {
        if (!ft.Initialize(&params_)) {
            LOGE("SimpleVIO", "Initialize Feature Tracker Error!!!");

            return false;
        }
    }

    // estimator
    if (!estimator_.SetParameters(&params_)) {
        LOGE("SimpleVIO", "Estimator SetParameters Error!!!");

        return false;
    }

    if (!estimator_.Initialize()) {
        LOGE("SimpleVIO", "Initialize Estimator Error!!!");

        return false;
    }

    // output file
    ofs_pose_.open("./pose_output.txt", std::fstream::out);
    if (!ofs_pose_.is_open()) {
        LOGE("SimpleVIO", "ofs_pose is not open!!!");

        return false;
    }

    // threads
    stop_main_process_loop_.store(false);
    main_process_loop_thread_ = std::thread(&System::MainProcessLoop, this);

#if defined(DRAW_RESULT)
    draw_thread_ = std::thread(&System::Draw, this);
#endif

    LOGI("SimpleVIO", "System Finish Initialize");

    return true;
}

void System::Finish() {
    LOGD("SimpleVIO", "Finish 0");
    stop_main_process_loop_.store(true);

    {
        std::unique_lock<std::shared_timed_mutex> lk(measure_mtx_);
        imu_buf_   = std::queue<IMUConstPtr>();
        image_buf_ = std::queue<ImageConstPtr>();
        measure_cond_.notify_all();
    }

    LOGD("SimpleVIO", "Finish 1");

    {
        std::unique_lock<std::mutex> lk(estimator_mtx_);
        estimator_.ClearState();
    }

    LOGD("SimpleVIO", "Finish 2");

    if (main_process_loop_thread_.joinable()) {
        main_process_loop_thread_.join();
    }

#if defined(DRAW_RESULT)
    pangolin::QuitAll();

    if (draw_thread_.joinable()) {
        draw_thread_.join();
    }
#endif

    ofs_pose_.close();

    finished_ = true;
}

void System::PubImageData(const double timestamp_sec, cv::Mat &img) {
    if (stop_main_process_loop_.load()) {
        return;
    }

    if (!has_first_image_) {
        LOGI("SimpleVIO",
             "PubImageData skip the first detected feature, which doesn't "
             "contain optical flow speed");

        last_image_time_  = timestamp_sec;
        first_image_time_ = timestamp_sec;

        // process first frame
        for (int i = 0; i < params_.num_of_camera; ++i) {
            feature_trackers_[i].SetInputImage(timestamp_sec, img, false);
        }

        has_first_image_ = true;

        return;
    }

    // detect unstable camera stream
    if (timestamp_sec - last_image_time_ > 1.0 ||
        timestamp_sec < last_image_time_) {
        LOGE("SimpleVIO",
             "PubImageData image discontinue! reset the feature tracker!");

        last_image_time_ = 0.0;

        pub_cnt_ = 1;

        has_first_image_ = false;

        return;
    }
    last_image_time_ = timestamp_sec;

    // frequency control
    bool pub_this_frame_ = false;
    double current_pub_frequencey =
        (double)pub_cnt_ / (timestamp_sec - first_image_time_);
    if (std::round(current_pub_frequencey) <= params_.frequency) {
        pub_this_frame_ = true;

        // reset the frequency control
        if (std::abs(current_pub_frequencey - params_.frequency) <
            0.01 * params_.frequency) {
            first_image_time_ = timestamp_sec;
            pub_cnt_          = 0;
        }
    } else {
        pub_this_frame_ = false;
    }

    // Timer t_r;
    for (int i = 0; i < params_.num_of_camera; ++i) {
        feature_trackers_[i].SetInputImage(timestamp_sec, img, pub_this_frame_);
    }

    if (pub_this_frame_) {
        pub_cnt_ += 1;

        ImagePtr feature_points = std::make_shared<IMG_MSG>();
        feature_points->header  = timestamp_sec;

        for (int i = 0; i < params_.num_of_camera; ++i) {
            auto &cur_pts    = feature_trackers_[i].GetCurrentPoints();
            auto &cur_un_pts = feature_trackers_[i].GetCurrentUndistortPoints();
            auto &pts_velocity = feature_trackers_[i].GetPointsVelocity();
            auto &ids          = feature_trackers_[i].GetIDs();
            auto &track_cnt    = feature_trackers_[i].GetTrackCount();

            for (size_t j = 0; j < ids.size(); ++j) {
                if (track_cnt[j] > 1) {
                    feature_points->points.push_back(
                        Vec3(cur_un_pts[j].x, cur_un_pts[j].y, 1.0));

                    feature_points->id_of_point.push_back(
                        ids[j] * params_.num_of_camera + i);

                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);

                    feature_points->velocity_x_of_point.push_back(
                        pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(
                        pts_velocity[j].y);
                }
            }

            // pubish
            {
                std::shared_lock<std::shared_timed_mutex> lk(measure_mtx_);
                image_buf_.push(feature_points);
                measure_cond_.notify_one();
            }
        }
    }

    // shwo image
    if (params_.show_track) {
        cv::Mat show_img;
        cv::cvtColor(img, show_img, cv::COLOR_GRAY2RGB);

        auto &cur_pts   = feature_trackers_[0].GetCurrentPoints();
        auto &track_cnt = feature_trackers_[0].GetTrackCount();
        for (size_t i = 0; i < cur_pts.size(); ++i) {
            double len =
                std::min(1.0, 1.0 * track_cnt[i] / params_.window_size);
            cv::circle(show_img,
                       cur_pts[i],
                       2,
                       cv::Scalar(255 * (1 - len), 0, 255 * len),
                       2);
        }

        cv::namedWindow("IMAGE", cv::WINDOW_AUTOSIZE);
        cv::imshow("IMAGE", show_img);
        cv::waitKey(1);
    }
}

void System::PubImageData(const double timestamp_sec,
                          const std::string &keyframes_file) {
    if (stop_main_process_loop_.load()) {
        return;
    }

    if (!has_first_image_) {
        LOGI("SimpleVIO",
             "PubImageData skip the first detected feature, which doesn't "
             "contain optical flow speed");

        last_image_time_  = timestamp_sec;
        first_image_time_ = timestamp_sec;

        // process first frame
        for (int i = 0; i < params_.num_of_camera; ++i) {
            feature_trackers_[i].SetInputImage(timestamp_sec, keyframes_file);
        }

        has_first_image_ = true;

        return;
    }

    // detect unstable camera stream
    if (timestamp_sec - last_image_time_ > 1.0 ||
        timestamp_sec < last_image_time_) {
        LOGE("SimpleVIO",
             "PubImageData image discontinue! reset the feature tracker!");

        last_image_time_ = 0.0;

        pub_cnt_ = 1;

        has_first_image_ = false;

        return;
    }
    last_image_time_ = timestamp_sec;

    // frequency control
    bool pub_this_frame_ = false;
    double current_pub_frequencey =
        (double)pub_cnt_ / (timestamp_sec - first_image_time_);
    if (std::round(current_pub_frequencey) <= params_.frequency) {
        pub_this_frame_ = true;

        // reset the frequency control
        if (std::abs(current_pub_frequencey - params_.frequency) <
            0.01 * params_.frequency) {
            first_image_time_ = timestamp_sec;
            pub_cnt_          = 0;
        }
    } else {
        pub_this_frame_ = false;
    }

    // Timer t_r;
    for (int i = 0; i < params_.num_of_camera; ++i) {
        feature_trackers_[i].SetInputImage(timestamp_sec, keyframes_file);
    }

    if (pub_this_frame_) {
        pub_cnt_ += 1;

        ImagePtr feature_points = std::make_shared<IMG_MSG>();
        feature_points->header  = timestamp_sec;

        for (int i = 0; i < params_.num_of_camera; ++i) {
            auto &cur_pts    = feature_trackers_[i].GetCurrentPoints();
            auto &cur_un_pts = feature_trackers_[i].GetCurrentUndistortPoints();
            auto &pts_velocity = feature_trackers_[i].GetPointsVelocity();
            auto &ids          = feature_trackers_[i].GetIDs();
            auto &track_cnt    = feature_trackers_[i].GetTrackCount();

            for (size_t j = 0; j < ids.size(); ++j) {
                if (track_cnt[j] > 1) {
                    feature_points->points.push_back(
                        Vec3(cur_un_pts[j].x, cur_un_pts[j].y, 1.0));

                    feature_points->id_of_point.push_back(
                        ids[j] * params_.num_of_camera + i);

                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);

                    feature_points->velocity_x_of_point.push_back(
                        pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(
                        pts_velocity[j].y);
                }
            }

            // pubish
            {
                std::shared_lock<std::shared_timed_mutex> lk(measure_mtx_);
                image_buf_.push(feature_points);
                measure_cond_.notify_one();
            }
        }
    }
}

void System::PubIMUData(const double timestamp_sec,
                        const Vec3 &acc,
                        const Vec3 &gyr) {
    if (stop_main_process_loop_.load()) {
        return;
    }

    IMUPtr imu_msg               = std::make_shared<IMU_MSG>();
    imu_msg->header              = timestamp_sec;
    imu_msg->linear_acceleration = acc;
    imu_msg->angular_velocity    = gyr;

    if (timestamp_sec <= last_imu_t_) {
        LOGE("SimpleVIO",
             "imu message in disorder! (%lf, %lf)",
             timestamp_sec,
             last_imu_t_);

        return;
    }
    last_imu_t_ = timestamp_sec;

    {
        std::shared_lock<std::shared_timed_mutex> lk(measure_mtx_);
        imu_buf_.push(imu_msg);
        measure_cond_.notify_one();
    }
}

size_t System::GetMeasurements(MeasData &meas_data) {
    meas_data.clear();

    while (true) {
        if (imu_buf_.empty() || image_buf_.empty()) {
            break;
        }

        if (imu_buf_.back()->header <=
            image_buf_.front()->header + estimator_.GetTd()) {
            LOGW("SimpleVIO",
                 "wait for imu, only should happen at the beginning (%lf, %lf)",
                 imu_buf_.back()->header,
                 image_buf_.front()->header);

            break;
        }

        if (imu_buf_.front()->header >=
            image_buf_.front()->header + estimator_.GetTd()) {
            LOGW("SimpleVIO",
                 "throw image, only should happen at the beginning (%lf, %lf)",
                 imu_buf_.front()->header,
                 image_buf_.front()->header);

            image_buf_.pop();

            continue;
        }

        // add measure
        ImageConstPtr image_msg = image_buf_.front();
        image_buf_.pop();

        std::vector<IMUConstPtr> imu_msgs;
        while (imu_buf_.front()->header <
               image_msg->header + estimator_.GetTd()) {
            imu_msgs.push_back(imu_buf_.front());
            imu_buf_.pop();
        }

        // one extra imu, no pop
        if (!imu_buf_.empty()) {
            imu_msgs.push_back(imu_buf_.front());
        }

        if (imu_msgs.empty()) {
            LOGW("SimpleVIO", "no imu between two image");
        }

        meas_data.emplace_back(imu_msgs, image_msg);
    }

    return meas_data.size();
}

// thread: visual-inertial odometry
void System::MainProcessLoop() {
    LOGI("SimpleVIO", "MainProcessLoop Start");

    while (!stop_main_process_loop_.load()) {
        // get measurements
        MeasData meas_data;
        {
            std::unique_lock<std::shared_timed_mutex> lk(measure_mtx_);
            measure_cond_.wait(lk, [&]() {
                return (GetMeasurements(meas_data) > 0 ||
                        stop_main_process_loop_.load());
            });
        }

        // estimation
        {
            std::unique_lock<std::mutex> lk(estimator_mtx_);
            for (auto &meas : meas_data) {
                auto &image_msg = meas.second;

                const double image_time =
                    image_msg->header + estimator_.GetTd();

                // previous imu
                double dx = 0.0;
                double dy = 0.0;
                double dz = 0.0;
                double rx = 0.0;
                double ry = 0.0;
                double rz = 0.0;
                for (auto &imu_msg : meas.first) {
                    double t = imu_msg->header;

                    if (t <= image_time) {
                        if (current_time_ < 0) {
                            current_time_ = t;
                        }

                        double dt     = t - current_time_;
                        current_time_ = t;

                        assert(dt >= 0.0);

                        dx = imu_msg->linear_acceleration.x();
                        dy = imu_msg->linear_acceleration.y();
                        dz = imu_msg->linear_acceleration.z();
                        rx = imu_msg->angular_velocity.x();
                        ry = imu_msg->angular_velocity.y();
                        rz = imu_msg->angular_velocity.z();

                        estimator_.ProcessIMU(dt,
                                              Vec3(dx, dy, dz),
                                              Vec3(rx, ry, rz));
                    } else {
                        // the last imu
                        double dt_1   = image_time - current_time_;
                        double dt_2   = t - image_time;
                        current_time_ = image_time;

                        assert(dt_1 >= 0);
                        assert(dt_2 >= 0);
                        assert(dt_1 + dt_2 > 0);

                        // interpolate
                        double w1 = dt_2 / (dt_1 + dt_2);
                        double w2 = dt_1 / (dt_1 + dt_2);

                        dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                        dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                        dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                        rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                        ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                        rz = w1 * rz + w2 * imu_msg->angular_velocity.z();

                        estimator_.ProcessIMU(dt_1,
                                              Vec3(dx, dy, dz),
                                              Vec3(rx, ry, rz));
                    }
                }

                LOGI("SimpleVIO",
                     "processing vision data with stamp: %lf, "
                     "img_msg->points.size: %zu",
                     image_msg->header,
                     image_msg->points.size());

                std::map<int, std::vector<FeatureObservation>> image_features;
                for (size_t i = 0; i < image_msg->points.size(); ++i) {
                    const int id         = image_msg->id_of_point[i];
                    const int feature_id = id / params_.num_of_camera;
                    const int camera_id  = id % params_.num_of_camera;

                    assert(1.0 == image_msg->points[i].z());

                    FeatureObservation feature_obs;
                    feature_obs.camera_idx  = camera_id;
                    feature_obs.point       = image_msg->points[i];
                    feature_obs.uv(0)       = image_msg->u_of_point[i];
                    feature_obs.uv(1)       = image_msg->v_of_point[i];
                    feature_obs.velocity(0) = image_msg->velocity_x_of_point[i];
                    feature_obs.velocity(1) = image_msg->velocity_y_of_point[i];

                    image_features[feature_id].push_back(feature_obs);
                }

                Timer t_processImage;
                estimator_.ProcessImage(image_msg->header, image_features);

                // output pose
                if (SolverFlag::NON_LINEAR == estimator_.GetSolverFlag()) {
                    std::vector<Mat33> &Rs = estimator_.GetRs();
                    std::vector<Vec3> &Ps  = estimator_.GetPs();

                    Vec3 p_wi = estimator_.GetPs()[params_.window_size];
                    Qd q_wi(estimator_.GetRs()[params_.window_size]);

                    path_to_draw_.push_back(p_wi);

                    double timestamp =
                        estimator_.GetHeaders()[params_.window_size];

                    LOGI("SimpleVIO",
                         "ProcessImage cost %lf ms, timestamp: %lf, p_wi: %lf "
                         "%lf %lf",
                         t_processImage.End(),
                         timestamp,
                         p_wi.transpose()(0),
                         p_wi.transpose()(1),
                         p_wi.transpose()(2));

                    LOGI("SimpleVIO", "==============================");

                    ofs_pose_ << std::fixed << timestamp << " " << p_wi[0]
                              << " " << p_wi[1] << " " << p_wi[2] << " "
                              << q_wi.x() << " " << q_wi.y() << " " << q_wi.z()
                              << " " << q_wi.w() << std::endl;
                }
            }
        }
    }

    LOGI("SimpleVIO", "MainProcessLoop Finish");
}

#if defined(DRAW_RESULT)
void System::Draw() {
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0));

    d_cam_ = pangolin::CreateDisplay()
                 .SetBounds(0.0,
                            1.0,
                            pangolin::Attach::Pix(175),
                            1.0,
                            -1024.0f / 768.0f)
                 .SetHandler(new pangolin::Handler3D(s_cam_));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam_.Activate(s_cam_);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);

        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = path_to_draw_.size();
        for (int i = 0; i < nPath_size - 1; ++i) {
            glVertex3f(path_to_draw_[i].x(),
                       path_to_draw_[i].y(),
                       path_to_draw_[i].z());
            glVertex3f(path_to_draw_[i + 1].x(),
                       path_to_draw_[i + 1].y(),
                       path_to_draw_[i + 1].z());
        }
        glEnd();

        // points
        if (estimator_.GetSolverFlag() == SolverFlag::NON_LINEAR) {
            glPointSize(5);
            glBegin(GL_POINTS);
            for (int i = 0; i < params_.window_size + 1; ++i) {
                Vec3 p_wi = estimator_.GetPs()[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0], p_wi[1], p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();

        // sleep 5ms
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    pangolin::DestroyWindow("Trajectory Viewer");
}
#endif

}  // namespace SimpleVIO
