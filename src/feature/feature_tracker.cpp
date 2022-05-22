#include "feature_tracker.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <utility>

#include "camodocal/camera_models/CameraFactory.h"
#include "utils/eigen_types.h"
#include "utils/logger.h"

namespace SimpleVIO {

int FeatureTracker::n_id = 0;

static const int kBorderSize = 1;
bool FeatureTracker::InBorder(const cv::Point2f &pt) {
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);

    return (kBorderSize <= img_x) &&
           (img_x < params_->image_width - kBorderSize) &&
           (kBorderSize <= img_y) &&
           (img_y < params_->image_height - kBorderSize);
}

template<typename T>
static void ReduceVector(std::vector<T> &vec,
                         const std::vector<unsigned char> &status) {
    size_t j = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (1 == status[i]) {
            vec[j++] = vec[i];
        }
    }

    vec.resize(j);
}

FeatureTracker::FeatureTracker() {}

bool FeatureTracker::Initialize(Parameters *params) {
    if (nullptr == params) {
        LOGE("SimpleVIO", "Empty params !!!");
        return false;
    }

    params_ = params;

    // create camera
    camera_ptr_ =
        camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
            params->config_file);
    if (nullptr == camera_ptr_) {
        LOGE("SimpleVIO", "Create camera failed !!!");
        return false;
    }

    return true;
}

void FeatureTracker::SetInputImage(const double timestamp,
                                   const cv::Mat &image,
                                   const bool pub_this_frame) {
    cur_time_ = timestamp;

    cv::Mat img;
    if (params_->do_equalize) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(image, img);
    } else {
        img = image;
    }

    if (cur_img_.empty()) {
        prev_img_ = cur_img_ = img;
    } else {
        cur_img_ = img;
    }

    cur_pts_.clear();

    // do tracking
    if (prev_pts_.size() > 0) {
        std::vector<unsigned char> status;

        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img_,
                                 cur_img_,
                                 prev_pts_,
                                 cur_pts_,
                                 status,
                                 err,
                                 cv::Size(21, 21),
                                 3);

        for (size_t i = 0; i < cur_pts_.size(); ++i) {
            if (1 == status[i] && !InBorder(cur_pts_[i])) {
                status[i] = 0;
            }
        }

        ReduceTrackedPointsByStatus(status);

        // increase tracked count
        for (auto &n : track_cnt_) {
            n += 1;
        }
    } else {
        SetImageMask();

        AddNewFeatures(params_->max_feature_cnt -
                       static_cast<int>(cur_pts_.size()));
    }

    // update tracked points
    if (pub_this_frame) {
        RejectWithF();

        ReorderTrackedPoints();

        SetImageMask();

        AddNewFeatures(params_->max_feature_cnt -
                       static_cast<int>(cur_pts_.size()));
    }

    UndistortPoints();

    CalculateVelocity();

    // previous <- current
    prev_img_        = cur_img_;
    prev_pts_        = cur_pts_;
    prev_un_pts_     = cur_un_pts_;
    prev_un_pts_map_ = cur_un_pts_map_;
    prev_time_       = cur_time_;

    UpdateAllIDs();
}

void FeatureTracker::SetInputImage(const double timestamp,
                                   const std::string &keyframes_file) {
    cur_time_ = timestamp;

    cur_pts_.clear();

    // do tracking
    {
        std::ifstream fs_keyframes;
        fs_keyframes.open(keyframes_file.c_str());
        if (!fs_keyframes.is_open()) {
            LOGE("SimpleVIO",
                 "Failed to open fsFeatures file [%s]! ",
                 keyframes_file.c_str());

            return;
        }

        ids_.clear();
        int id = 0;

        std::string keyframes_line;
        while (std::getline(fs_keyframes, keyframes_line) &&
               !keyframes_line.empty()) {
            std::istringstream keyframes_data(keyframes_line);

            float map_point[4];  // not used
            float keypoint[2];
            keyframes_data >> map_point[0] >> map_point[1] >> map_point[2] >>
                map_point[3] >> keypoint[0] >> keypoint[1];

            cv::Point2f image_point(params_->fx * keypoint[0] + params_->cx,
                                    params_->fy * keypoint[1] + params_->cy);

            cur_pts_.push_back(image_point);

            ids_.push_back(id);
            id += 1;
        }

        fs_keyframes.close();

        if (track_cnt_.size() != ids_.size()) {
            track_cnt_.clear();
            track_cnt_.resize(ids_.size(), 0);
        }
    }

    // increase tracked count
    for (auto &n : track_cnt_) {
        n += 1;
    }

    UndistortPoints();

    CalculateVelocity();

    // previous <- current
    prev_pts_        = cur_pts_;
    prev_un_pts_     = cur_un_pts_;
    prev_un_pts_map_ = cur_un_pts_map_;
    prev_time_       = cur_time_;
}

void FeatureTracker::ReduceTrackedPointsByStatus(
    const std::vector<unsigned char> &status) {
    ReduceVector(prev_pts_, status);
    ReduceVector(cur_pts_, status);
    ReduceVector(prev_un_pts_, status);
    ReduceVector(track_cnt_, status);
    ReduceVector(ids_, status);
}

void FeatureTracker::ReorderTrackedPoints() {
    int pts_cnt = (int)track_cnt_.size();

    std::vector<int> index(pts_cnt);
    std::iota(index.begin(), index.end(), 0);

    std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
        return (track_cnt_[a] > track_cnt_[b]);
    });

    std::vector<cv::Point2f> temp_cur_pts(pts_cnt);
    std::vector<int> temp_track_cnt(pts_cnt);
    std::vector<int> temp_ids(pts_cnt);

    for (int i = 0; i < pts_cnt; ++i) {
        temp_cur_pts[i]   = cur_pts_[index[i]];
        temp_track_cnt[i] = track_cnt_[index[i]];
        temp_ids[i]       = ids_[index[i]];
    }

    // update current
    cur_pts_   = temp_cur_pts;
    track_cnt_ = temp_track_cnt;
    ids_       = temp_ids;
}

void FeatureTracker::SetImageMask() {
    // TODO: fisheye mask
    img_mask_ = cv::Mat(params_->image_height,
                        params_->image_width,
                        CV_8UC1,
                        cv::Scalar(255));

    for (auto &p : cur_pts_) {
        if (255 == img_mask_.at<unsigned char>(p)) {
            cv::circle(img_mask_, p, params_->min_feature_displacement, 0, -1);
        }
    }
}

void FeatureTracker::AddNewFeatures(int needed_feature_cnt) {
    std::vector<cv::Point2f> added_pts;

    if (needed_feature_cnt > 0) {
        if (img_mask_.empty()) {
            LOGE("SimpleVIO", "mask is empty");
        }

        if (CV_8UC1 != img_mask_.type()) {
            LOGE("SimpleVIO", "mask type wrong");
        }

        if (img_mask_.size() != cur_img_.size()) {
            LOGE("SimpleVIO", "mask wrong size");
        }

        cv::goodFeaturesToTrack(cur_img_,
                                added_pts,
                                needed_feature_cnt,
                                0.01,
                                params_->min_feature_displacement,
                                img_mask_);
    }

    // add to current points
    for (auto &p : added_pts) {
        cur_pts_.push_back(p);
        track_cnt_.push_back(1);
        ids_.push_back(-1);
    }
}

void FeatureTracker::RejectWithF() {
    if (cur_pts_.size() >= 8) {
        assert(prev_pts_.size() == cur_pts_.size());
        std::vector<cv::Point2f> temp_prev_un_pts(prev_pts_.size());
        std::vector<cv::Point2f> temp_cur_un_pts(cur_pts_.size());

        for (size_t i = 0; i < cur_pts_.size(); ++i) {
            Vec3 temp_p;

            // previous
            camera_ptr_->liftProjective(Vec2(prev_pts_[i].x, prev_pts_[i].y),
                                        temp_p);

            temp_prev_un_pts[i] = cv::Point2f(
                params_->fx * temp_p.x() / temp_p.z() + params_->cx,
                params_->fy * temp_p.y() / temp_p.z() + params_->cy);

            // current
            camera_ptr_->liftProjective(Vec2(cur_pts_[i].x, cur_pts_[i].y),
                                        temp_p);

            temp_cur_un_pts[i] = cv::Point2f(
                params_->fx * temp_p.x() / temp_p.z() + params_->cx,
                params_->fy * temp_p.y() / temp_p.z() + params_->cy);
        }

        std::vector<unsigned char> status;
        cv::findFundamentalMat(temp_prev_un_pts,
                               temp_cur_un_pts,
                               cv::FM_RANSAC,
                               params_->fundamental_ransac_threshold,
                               0.99,
                               status);

        ReduceTrackedPointsByStatus(status);
    } else {
        LOGW("SimpleVIO", "not enough feature points (< 8)");
    }
}

void FeatureTracker::UpdateAllIDs() {
    for (auto &id : ids_) {
        if (-1 == id) {
            id = n_id++;
        }
    }
}

void FeatureTracker::UndistortPoints() {
    // undistort current points
    cur_un_pts_.clear();
    cur_un_pts_map_.clear();

    for (size_t i = 0; i < cur_pts_.size(); ++i) {
        Vec3 temp_p;

        camera_ptr_->liftProjective(Vec2(cur_pts_[i].x, cur_pts_[i].y), temp_p);

        cv::Point2f temp_un_pts(temp_p.x() / temp_p.z(),
                                temp_p.y() / temp_p.z());
        cur_un_pts_.push_back(temp_un_pts);
        cur_un_pts_map_.insert(std::make_pair(ids_[i], temp_un_pts));
    }
}

void FeatureTracker::CalculateVelocity() {
    pts_velocity_.clear();

    if (!prev_un_pts_map_.empty()) {
        double dt = cur_time_ - prev_time_;
        assert(dt > 0);

        for (size_t i = 0; i < cur_un_pts_.size(); ++i) {
            if (-1 != ids_[i]) {
                auto iter = prev_un_pts_map_.find(ids_[i]);
                if (prev_un_pts_map_.end() != iter) {
                    // both previous and current are tracked
                    pts_velocity_.push_back(
                        cv::Point2f((cur_un_pts_[i].x - iter->second.x) / dt,
                                    (cur_un_pts_[i].y - iter->second.y) / dt));
                } else {
                    // TODO: should not run into this line
                    pts_velocity_.push_back(cv::Point2f(0.0f, 0.0f));
                }
            } else {
                pts_velocity_.push_back(cv::Point2f(0.0f, 0.0f));
            }
        }
    } else {
        // first image
        pts_velocity_.resize(cur_un_pts_.size(), cv::Point2f(0.0f, 0.0f));
    }
}

}  // namespace SimpleVIO