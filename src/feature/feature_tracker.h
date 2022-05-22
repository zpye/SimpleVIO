#ifndef SIMPLE_VIO_FEATURE_FEATURE_TRACKER_H_
#define SIMPLE_VIO_FEATURE_FEATURE_TRACKER_H_

#include <cstddef>

#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/Camera.h"
#include "parameters.h"

namespace SimpleVIO {

class FeatureTracker {
public:
    FeatureTracker();

    bool Initialize(Parameters *params);

public:
    void SetInputImage(const double timestamp,
                       const cv::Mat &image,
                       const bool pub_this_frame);
    void SetInputImage(const double timestamp,
                       const std::string &keyframes_file);

    std::vector<cv::Point2f> &GetCurrentPoints() {
        return cur_pts_;
    }

    std::vector<cv::Point2f> &GetCurrentUndistortPoints() {
        return cur_un_pts_;
    }

    std::vector<cv::Point2f> &GetPointsVelocity() {
        return pts_velocity_;
    }

    std::vector<int> &GetTrackCount() {
        return track_cnt_;
    }

    std::vector<int> &GetIDs() {
        return ids_;
    }

private:
    void RejectWithF();

    void ReorderTrackedPoints();

    void SetImageMask();

    void AddNewFeatures(int needed_feature_cnt);

    void UndistortPoints();

    void CalculateVelocity();

    void UpdateAllIDs();

private:
    bool InBorder(const cv::Point2f &pt);

    void ReduceTrackedPointsByStatus(const std::vector<unsigned char> &status);

private:
    Parameters *params_ = nullptr;

private:
    camodocal::CameraPtr camera_ptr_;

private:
    cv::Mat img_mask_;

    cv::Mat prev_img_;
    cv::Mat cur_img_;

    std::vector<cv::Point2f> prev_pts_;
    std::vector<cv::Point2f> cur_pts_;

    std::vector<cv::Point2f> prev_un_pts_;
    std::vector<cv::Point2f> cur_un_pts_;

    std::vector<cv::Point2f> pts_velocity_;

    std::map<int, cv::Point2f> prev_un_pts_map_;
    std::map<int, cv::Point2f> cur_un_pts_map_;

    double prev_time_ = -1.0;
    double cur_time_  = -1.0;

    std::vector<int> track_cnt_;

    std::vector<int> ids_;

private:
    static int n_id;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_FEATURE_FEATURE_TRACKER_H_