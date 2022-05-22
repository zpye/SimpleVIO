#include "problem_LM_Nielson.h"

#include <cassert>
#include <cmath>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "utils/logger.h"
#include "utils/timer.h"

namespace SimpleVIO {

bool Problem_LM_Nielson::Solve(int iterations) {
    if (edges_.empty() || vertices_.empty()) {
        LOGE("SimpleVIO", "Cannot solve problem without edges or vertices !!!");

        return false;
    }

    Timer t_solve;

    SetOrdering();

    MakeHessianWithPrior();

    ComputeParamsInit();

    bool stop        = false;
    int iter_cnt     = 0;
    double last_chi2 = 1e30;
    while (!stop && iter_cnt < iterations) {
        LOGD("SimpleVIO",
             "[Solve] iter: %2d, chi2: %.9lf, lambda: %.9lf",
             iter_cnt,
             params_LM_.current_chi2,
             params_LM_.current_lambda);

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

                    false_cnt = 0;
                } else {
                    RollbackStates();

                    false_cnt += 1;
                }
            }
        }

        if (last_chi2 - params_LM_.current_chi2 <
            params_LM_.stop_threshold_LM) {
            LOGI("SimpleVIO", "last_chi2 - current_chi2 < stop_threshold_LM");
            stop = true;
        }

        last_chi2 = params_LM_.current_chi2;

        iter_cnt += 1;
    }

    LOGD("SimpleVIO",
         "[Solve] iter: %d, chi2: %.9lf, lambda: %.9lf",
         iter_cnt,
         params_LM_.current_chi2,
         params_LM_.current_lambda);

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

void Problem_LM_Nielson::ComputeParamsInit() {
    // initialize
    params_LM_.ni = 2.0;

    // current chi2
    params_LM_.current_chi2 = ComputeChi2(false);

    // stop threshold
    params_LM_.stop_threshold_LM = 1e-10 * params_LM_.current_chi2;

    // current lambda
    if (Hessian_.rows() != Hessian_.cols()) {
        LOGE("SimpleVIO", "Hessian is not square");
        assert(Hessian_.rows() == Hessian_.cols());
    }

    double max_diagonal = 5e-10;
    for (Eigen::Index i = 0; i < Hessian_.cols(); ++i) {
        max_diagonal = std::max(std::fabs(Hessian_(i, i)), max_diagonal);
    }

    max_diagonal = std::min(5e10, max_diagonal);

    double tau                = 1e-5;
    params_LM_.current_lambda = tau * max_diagonal;
}

// Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
void Problem_LM_Nielson::SolveLinearSystem(bool use_shur_marg) {
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
            Hpp_schur(i, i) += params_LM_.current_lambda;
        }

        VecX delta_x_p = Hpp_schur.ldlt().solve(bp_schur);

        // step3: solve Hmm * delta_x = bm - Hmp * delta_x_p;
        VecX delta_x_m = Hmm_inv * (bm - Hmp * delta_x_p);

        // result
        delta_x_.head(reserve_size) = delta_x_p;
        delta_x_.tail(marg_size)    = delta_x_m;
    } else {
        MatXX H = Hessian_;

        for (Eigen::Index i = 0; i < H.cols(); ++i) {
            H(i, i) += params_LM_.current_lambda;
        }

        // TODO: PCG solver
        // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        delta_x_ = H.ldlt().solve(b_);
        std::cout << delta_x_ << "\n" << std::endl;
    }
}

bool Problem_LM_Nielson::IsGoodDelta() {
    return (delta_x_.norm() < 1e-30);
}

bool Problem_LM_Nielson::IsGoodStep() {
    double scale = 0.5 * delta_x_.transpose() *
                   (params_LM_.current_lambda * delta_x_ + b_);
    scale += 1e-6;  // make sure it's non-zero :)

    // recompute residuals after update state
    double temp_chi2 = ComputeChi2(true);

    double rho = (params_LM_.current_chi2 - temp_chi2) / scale;
    if (rho > 0.0 && std::isfinite(temp_chi2)) {
        // last step was good
        double alpha = 1.0 - std::pow((2.0 * rho - 1.0), 3.0);
        alpha        = std::min(alpha, 2.0 / 3.0);

        double scale_factor = (std::max)(1.0 / 3.0, alpha);  // range [1/3, 2/3]

        params_LM_.current_lambda *= scale_factor;
        params_LM_.current_chi2 = temp_chi2;
        params_LM_.ni           = 2.0;

        return true;
    } else {
        params_LM_.current_lambda =
            std::min(params_LM_.current_lambda * params_LM_.ni, 1e30);
        params_LM_.ni *= 2;

        return false;
    }

    return false;
}

}  // namespace SimpleVIO