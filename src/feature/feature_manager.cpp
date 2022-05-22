#include "feature_manager.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include "utils/logger.h"

namespace SimpleVIO {

bool FeatureManager::Initialize(Parameters *params) {
    if (nullptr == params) {
        LOGE("SimpleVIO", "Empty params !!!");
        return false;
    }

    params_ = params;

    ric_.resize(params->num_of_camera);

    return true;
}

void FeatureManager::SetRs(const std::vector<Mat33> &Rs) {
    Rs_ = Rs;
}

void FeatureManager::SetRic(const std::vector<Mat33> &ric) {
    for (int i = 0; i < params_->num_of_camera; ++i) {
        ric_[i] = ric[i];
    }
}

// TODO: multi camera
void FeatureManager::AddFeatures(
    double timestamp,
    int frame_count,
    const std::map<int, std::vector<FeatureObservation>> &features) {
    // recount track number
    last_track_num_ = 0;
    for (auto &id_obs : features) {
        const int feature_id = id_obs.first;

        auto iter = std::find_if(
            features_.begin(),
            features_.end(),
            [feature_id](const std::shared_ptr<TrackedFeaturePerId> &f) {
                return (f->feature_id == feature_id);
            });

        if (features_.end() == iter) {
            std::shared_ptr<TrackedFeaturePerId> f_per_id =
                std::make_shared<TrackedFeaturePerId>(feature_id, frame_count);
            f_per_id->feature_per_frame_vec.emplace_back(timestamp,
                                                         id_obs.second[0]);
            features_.push_back(f_per_id);
        } else if ((*iter)->feature_id == feature_id) {
            (*iter)->feature_per_frame_vec.emplace_back(timestamp,
                                                        id_obs.second[0]);
            last_track_num_ += 1;
        }
    }

    need_update_ = true;
}

void FeatureManager::UpdateFeatures() {
    available_feature_idx_.clear();

    int feature_idx = 0;
    for (auto &f_per_id : features_) {
        if (f_per_id->feature_per_frame_vec.size() >= 2 &&
            f_per_id->start_frame < params_->window_size - 2) {
            available_feature_idx_.push_back(feature_idx);
        }

        feature_idx += 1;
    }

    need_update_ = false;
}

bool FeatureManager::CheckParallax(int frame_count) {
    if (frame_count < 2 || last_track_num_ < 20) {
        return true;
    }

    double parallax_sum = 0.0;
    int parallax_num    = 0;

    for (auto &f_per_id : features_) {
        if (f_per_id->start_frame <= frame_count - 2 &&
            f_per_id->EndFrame() >= frame_count - 1) {
            parallax_sum += CompensatedParallax2(*f_per_id, frame_count);
            parallax_num += 1;
        }
    }

    if (0 == parallax_num) {
        return true;
    } else {
        // average parallax
        return (parallax_sum / parallax_num >= params_->min_parallax);
    }
}

void FeatureManager::ClearState() {
    last_track_num_ = 0;
    features_.clear();
}

int FeatureManager::GetFeatureCount() {
    if (need_update_) {
        UpdateFeatures();
    }

    return (int)available_feature_idx_.size();
}

std::vector<std::shared_ptr<TrackedFeaturePerId>>
    &FeatureManager::GetFeatures() {
    return features_;
}

VecX FeatureManager::GetDepthVector() {
    VecX ret(GetFeatureCount());

    int cnt = 0;
    for (auto &idx : available_feature_idx_) {
        ret(cnt) = 1.0 / features_[idx]->estimated_depth;
        cnt += 1;
    }

    return ret;
}

void FeatureManager::SetDepth(const VecX &x, bool update_flag) {
    if (need_update_) {
        UpdateFeatures();
    }

    int cnt = 0;
    for (auto &idx : available_feature_idx_) {
        features_[idx]->estimated_depth = 1.0 / x(cnt);
        cnt += 1;

        if (update_flag) {
            if (features_[idx]->estimated_depth >= 0.0) {
                features_[idx]->solve_flag = 1;
            } else {
                features_[idx]->solve_flag = 2;
            }
        }
    }
}

void FeatureManager::ScaleDepth(const double scale) {
    if (need_update_) {
        UpdateFeatures();
    }

    for (auto &idx : available_feature_idx_) {
        features_[idx]->estimated_depth *= scale;
    }
}

std::vector<std::pair<Vec3, Vec3>> FeatureManager::GetCorresponding(
    int frame_count_l,
    int frame_count_r) {
    assert(frame_count_l < frame_count_r);

    std::vector<std::pair<Vec3, Vec3>> ret;
    for (auto &iter : features_) {
        if (iter->start_frame <= frame_count_l &&
            iter->EndFrame() >= frame_count_r) {
            ret.push_back(std::make_pair(
                iter->feature_per_frame_vec[frame_count_l - iter->start_frame]
                    .feature_observation.point,
                iter->feature_per_frame_vec[frame_count_r - iter->start_frame]
                    .feature_observation.point));
        }
    }

    return ret;
}

void FeatureManager::Triangulate(const std::vector<Vec3> &Ps,
                                 const std::vector<Vec3> &tic,
                                 const std::vector<Mat33> &ric) {
    if (need_update_) {
        UpdateFeatures();
    }

    for (auto &idx : available_feature_idx_) {
        if (features_[idx]->estimated_depth > 0.0) {
            continue;
        }

        int start_idx = features_[idx]->start_frame;

        // TODO: multi camera
        assert(1 == params_->num_of_camera);
        MatXX svd_A(2 * features_[idx]->feature_per_frame_vec.size(), 4);

        int svd_idx    = 0;
        int offset_idx = 0;

        Mat33 R0 = Rs_[start_idx] * ric[0];
        Vec3 t0  = Rs_[start_idx] * tic[0] + Ps[start_idx];

        for (auto &f_per_frame : features_[idx]->feature_per_frame_vec) {
            const int cur_idx = start_idx + offset_idx;

            Mat33 R1 = Rs_[cur_idx] * ric[0];
            Vec3 t1  = Rs_[cur_idx] * tic[0] + Ps[cur_idx];

            Mat33 delta_R = R0.transpose() * R1;
            Vec3 delta_t  = R0.transpose() * (t1 - t0);

            Mat34 P;
            P.leftCols<3>()  = delta_R.transpose();
            P.rightCols<1>() = -delta_R.transpose() * delta_t;

            Vec3 unit_pts = f_per_frame.feature_observation.point.normalized();
            svd_A.row(svd_idx) =
                unit_pts[0] * P.row(2) - unit_pts[2] * P.row(0);
            svd_A.row(svd_idx + 1) =
                unit_pts[1] * P.row(2) - unit_pts[2] * P.row(1);

            svd_idx += 2;
            offset_idx += 1;
        }

        assert(svd_A.rows() == svd_idx);

        Vec4 svd_V = Eigen::JacobiSVD<MatXX>(svd_A, Eigen::ComputeThinV)
                         .matrixV()
                         .rightCols<1>();

        features_[idx]->estimated_depth = svd_V[2] / svd_V[3];
        if (features_[idx]->estimated_depth < 0.1) {
            features_[idx]->estimated_depth = params_->init_depth;
        }
    }
}

int FeatureManager::GetLastTrackNum() {
    return last_track_num_;
}

void FeatureManager::RemoveFailures() {
    for (auto iter = features_.begin(); iter != features_.end();) {
        if (2 == (*iter)->solve_flag) {
            iter = features_.erase(iter);
        } else {
            ++iter;
        }
    }

    need_update_ = true;
}

void FeatureManager::RemoveFront(int frame_count) {
    for (auto iter = features_.begin(); iter != features_.end();) {
        bool has_erased = false;
        if (frame_count == (*iter)->start_frame) {
            (*iter)->start_frame -= 1;
        } else if ((*iter)->EndFrame() >= frame_count - 1) {
            int offset = params_->window_size - 1 - (*iter)->start_frame;
            (*iter)->feature_per_frame_vec.erase(
                (*iter)->feature_per_frame_vec.begin() + offset);

            if ((*iter)->feature_per_frame_vec.empty()) {
                iter       = features_.erase(iter);
                has_erased = true;
            }
        }

        if (!has_erased) {
            ++iter;
        }
    }

    need_update_ = true;
}

void FeatureManager::RemoveBack() {
    for (auto iter = features_.begin(); iter != features_.end();) {
        bool has_erased = false;
        if (0 != (*iter)->start_frame) {
            (*iter)->start_frame -= 1;
        } else {
            (*iter)->feature_per_frame_vec.erase(
                (*iter)->feature_per_frame_vec.begin());

            if ((*iter)->feature_per_frame_vec.empty()) {
                iter       = features_.erase(iter);
                has_erased = true;
            }
        }

        if (!has_erased) {
            ++iter;
        }
    }

    need_update_ = true;
}

void FeatureManager::RemoveBackShiftDepth(const Mat33 &marg_R,
                                          const Vec3 &marg_P,
                                          const Mat33 &new_R,
                                          const Vec3 &new_P) {
    for (auto iter = features_.begin(); iter != features_.end();) {
        bool has_erased = false;
        if (0 != (*iter)->start_frame) {
            (*iter)->start_frame -= 1;
        } else {
            if ((*iter)->feature_per_frame_vec.empty()) {
                iter       = features_.erase(iter);
                has_erased = true;
            }

            Vec3 uv_i =
                (*iter)->feature_per_frame_vec[0].feature_observation.point;

            (*iter)->feature_per_frame_vec.erase(
                (*iter)->feature_per_frame_vec.begin());
            if ((*iter)->feature_per_frame_vec.size() < 2) {
                iter       = features_.erase(iter);
                has_erased = true;
            } else {
                Vec3 pts_i   = uv_i * (*iter)->estimated_depth;
                Vec3 w_pts_i = marg_R * pts_i + marg_P;
                Vec3 pts_j   = new_R.transpose() * (w_pts_i - new_P);

                if (pts_j(2) > 0.0f) {
                    (*iter)->estimated_depth = pts_j(2);
                } else {
                    (*iter)->estimated_depth = params_->init_depth;
                }
            }
        }

        if (!has_erased) {
            ++iter;
        }
    }

    need_update_ = true;
}

double FeatureManager::CompensatedParallax2(const TrackedFeaturePerId &f_per_id,
                                            int frame_count) {
    // check the second last frame is keyframe or not
    // parallax between second last frame and third last frame
    const Vec3 &frame_i =
        f_per_id.feature_per_frame_vec[frame_count - 2 - f_per_id.start_frame]
            .feature_observation.point;
    const Vec3 &frame_j =
        f_per_id.feature_per_frame_vec[frame_count - 1 - f_per_id.start_frame]
            .feature_observation.point;

    double u_j = frame_j(0) / frame_j(2);
    double v_j = frame_j(1) / frame_j(2);

    double u_i = frame_i(0) / frame_i(2);
    double v_i = frame_i(1) / frame_i(2);

    Vec3 p_i_comp = frame_i;
    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] *
    // ric[camera_id_i] * frame_i.point;
    double u_i_comp = p_i_comp(0) / p_i_comp(2);
    double v_i_comp = p_i_comp(1) / p_i_comp(2);

    double du      = u_i - u_j;
    double dv      = v_i - v_j;
    double du_comp = u_i_comp - u_j;
    double dv_comp = v_i_comp - v_j;

    double ans =
        std::max(0.0,
                 std::sqrt(std::min(du * du + dv * dv,
                                    du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}

}  // namespace SimpleVIO