#include "problem_DogLeg.h"

#include <cassert>
#include <cmath>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "utils/logger.h"
#include "utils/timer.h"

namespace SimpleVIO {

bool Problem_DogLeg::Solve(int iterations) {
    if (edges_.empty() || vertices_.empty()) {
        LOGE("SimpleVIO", "Cannot solve problem without edges or vertices !!!");

        return false;
    }

    Timer t_solve;

    SetOrdering();

    MakeHessianWithPrior();

    ComputeParamsInit();

    bool stop        = DogLegCondition1();
    int iter_cnt     = 0;
    double last_chi2 = 1e30;
    while (!stop && iter_cnt < iterations) {
        LOGD("SimpleVIO",
             "[Solve] iter: %d, chi2: %.9lf, trust region: %.9lf",
             iter_cnt,
             params_dogleg_.current_chi2,
             params_dogleg_.trust_region);

        // update lambda until success
        bool one_step_success = false;
        int false_cnt         = 0;

        // TODO: set iteration limit for lambda
        while (!stop && !one_step_success && false_cnt < 10) {
            SolveLinearSystem(ProblemType::SLAM_PROBLEM == problem_type_);

            if (IsGoodDelta()) {
                LOGD("SimpleVIO", "delta x is small enough");

                stop = true;
            } else {
                UpdateStates();

                // check if error decreased and update chi2
                one_step_success = IsGoodStep();
                if (one_step_success) {
                    MakeHessianWithPrior();

                    if (DogLegCondition1()) {
                        LOGD("SimpleVIO", "Satisfy DogLeg Condition 1");

                        stop = true;
                    }

                    if (last_chi2 - params_dogleg_.current_chi2 <
                        params_dogleg_.eps3) {
                        LOGI("SimpleVIO", "last_chi2 - current_chi2 < eps3");
                        stop = true;
                    }

                    last_chi2 = params_dogleg_.current_chi2;

                    false_cnt = 0;
                } else {
                    RollbackStates();

                    false_cnt += 1;
                }

                DogLegComputeXNorm();

                if (DogLegUpdateTrustRegion()) {
                    LOGD("SimpleVIO", "trust region is small enough");

                    stop = true;
                }
            }
        }

        iter_cnt += 1;
    }

    LOGD("SimpleVIO",
         "[Solve] iter: %d, chi2: %.9lf, trust region: %.9lf",
         iter_cnt,
         params_dogleg_.current_chi2,
         params_dogleg_.trust_region);

    // get cost
    double t_solve_cost = t_solve.End();
    LOGD("SimpleVIO", "problem solve cost %lf ms", t_solve_cost);
    LOGD("SimpleVIO", "make Hessian cost %lf ms", t_hessian_cost_);

    t_total_hessian_cost_ += t_hessian_cost_;
    t_total_solve_cost_ += t_solve_cost;
    total_solve_count_ += 1;

    t_hessian_cost_ = 0.0;

    return stop;
}

void Problem_DogLeg::ComputeParamsInit() {
    // initialize
    params_dogleg_.trust_region = 64.0;

    // current chi2
    params_dogleg_.current_chi2 = ComputeChi2(false);

    // eps
    params_dogleg_.eps1 = 1e-6;
    params_dogleg_.eps2 = 1e-6;
    params_dogleg_.eps3 = 1e-8 * params_dogleg_.current_chi2;

    DogLegComputeXNorm();
}

// Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
void Problem_DogLeg::SolveLinearSystem(bool use_shur_marg) {
    if (use_shur_marg) {
        // step1: schur marginalization --> Hpp, bp
        const size_t reserve_size = ordering_poses_;
        const size_t marg_size    = ordering_landmarks_;

        MatXX Hpp_schur;
        MatXX Hmp;
        MatXX Hmm_inv;
        VecX bp_schur;
        VecX bm;
        SchurDecomposition(Hessian_,
                           b_,
                           ordered_idx_landmark_vertices_,
                           reserve_size,
                           marg_size,
                           Hpp_schur,
                           Hmp,
                           Hmm_inv,
                           bp_schur,
                           bm);

        // step2: solve Hpp * delta_x_p = bp
        for (size_t i = 0; i < reserve_size; ++i) {
            Hpp_schur(i, i) += 1e-10;
        }

        VecX delta_x_p = Hpp_schur.ldlt().solve(bp_schur);

        // step3: solve Hmm * delta_x = bm - Hmp * delta_x_p;
        VecX delta_x_m = Hmm_inv * (bm - Hmp * delta_x_p);

        // result
        params_dogleg_.delta_x_gn = VecX::Zero(reserve_size + marg_size);
        params_dogleg_.delta_x_gn.head(reserve_size) = delta_x_p;
        params_dogleg_.delta_x_gn.tail(marg_size)    = delta_x_m;

        DogLegComputeDeltaX();
    } else {
        MatXX H = Hessian_;

        for (Eigen::Index i = 0; i < H.cols(); ++i) {
            H(i, i) += 1e-10;
        }

        // TODO: PCG solver
        // params_dogleg_.delta_x_gn = PCGSolver(H, b_, H.rows() * 2);
        params_dogleg_.delta_x_gn = H.ldlt().solve(b_);

        DogLegComputeDeltaX();
    }
}

bool Problem_DogLeg::IsGoodDelta() {
    return DogLegCondition2_h_dl();
}

bool Problem_DogLeg::IsGoodStep() {
    double scale = 1e-6;
    if (0 == params_dogleg_.state) {
        scale += params_dogleg_.current_chi2;
    } else if (1 == params_dogleg_.state) {
        scale += params_dogleg_.trust_region *
                 (2 * (params_dogleg_.alpha * b_).norm() -
                  params_dogleg_.trust_region) /
                 (2 * params_dogleg_.alpha);
    } else if (2 == params_dogleg_.state) {
        scale += 0.5 * params_dogleg_.alpha * (1.0 - params_dogleg_.beta) *
                     (1.0 - params_dogleg_.beta) * b_.squaredNorm() +
                 params_dogleg_.beta * (2.0 - params_dogleg_.beta) *
                     params_dogleg_.current_chi2;
    } else {
        LOGE("SimpleVIO", "Error State In DogLog [%d]", params_dogleg_.state);

        return false;
    }

    // recompute residuals after update state
    double temp_chi2 = ComputeChi2(true);

    params_dogleg_.rho = (params_dogleg_.current_chi2 - temp_chi2) / scale;

    if (params_dogleg_.rho > 0.0 && isfinite(temp_chi2)) {
        params_dogleg_.current_chi2 = temp_chi2;

        return true;
    } else {
        return false;
    }
}

void Problem_DogLeg::DogLegComputeXNorm() {
    params_dogleg_.norm_x = 0.0;
    for (auto &v : vertices_) {
        if (!v.second->IsFixed()) {
            params_dogleg_.norm_x += v.second->ComputeParametersNorm();
        }
    }
}

void Problem_DogLeg::DogLegComputeDeltaX() {
    // state 0 : || h_gn || <= trust region
    if (params_dogleg_.delta_x_gn.norm() <= params_dogleg_.trust_region) {
        delta_x_ = params_dogleg_.delta_x_gn;

        params_dogleg_.state = 0;

        return;
    }

    // compute alpha and h_sd
    params_dogleg_.alpha = b_.squaredNorm() / (b_.transpose() * Hessian_ * b_);
    params_dogleg_.delta_x_sd = params_dogleg_.alpha * b_;

    // state 1: ||alpha * h_sd|| >= trust region
    if (params_dogleg_.delta_x_sd.norm() >= params_dogleg_.trust_region) {
        delta_x_ = (params_dogleg_.trust_region / b_.norm()) * b_;

        params_dogleg_.state = 1;

        return;
    }

    // state 2: else
    VecX &a = params_dogleg_.delta_x_sd;
    VecX &b = params_dogleg_.delta_x_gn;

    double c = a.transpose() * (b - a);

    double tmp0 = (b - a).squaredNorm();
    double tmp1 = params_dogleg_.trust_region * params_dogleg_.trust_region -
                  a.squaredNorm();
    double tmp2 = c * c;

    if (c <= 0.0) {
        params_dogleg_.beta = (-c + std::sqrt(tmp2 + tmp0 * tmp1)) / tmp0;
    } else {
        params_dogleg_.beta = tmp1 / (c + std::sqrt(tmp2 + tmp0 * tmp1));
    }

    delta_x_ = a + params_dogleg_.beta * (b - a);

    params_dogleg_.state = 2;
}

bool Problem_DogLeg::DogLegCondition1() {
    return (b_.rowwise().lpNorm<1>().maxCoeff() <= params_dogleg_.eps1);
}

bool Problem_DogLeg::DogLegCondition2_h_dl() {
    return (delta_x_.norm() <= params_dogleg_.eps2 * (params_dogleg_.norm_x +
                                                      params_dogleg_.eps2));
}

bool Problem_DogLeg::DogLegCondition2_trust_region() {
    return (params_dogleg_.trust_region <=
            params_dogleg_.eps2 *
                (params_dogleg_.norm_x + params_dogleg_.eps2));
}

bool Problem_DogLeg::DogLegUpdateTrustRegion() {
    if (params_dogleg_.rho > 0.75) {
        params_dogleg_.trust_region =
            std::max(params_dogleg_.trust_region, 3.0 * delta_x_.norm());
    } else if (params_dogleg_.rho < 0.25) {
        params_dogleg_.trust_region *= 0.5;

        return DogLegCondition2_trust_region();
    }

    return false;
}

}  // namespace SimpleVIO