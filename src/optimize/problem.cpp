#include "problem.h"

#include <cassert>
#include <cmath>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "utils/logger.h"
#include "utils/timer.h"

#if defined(USE_OPENMP)
#include <omp.h>
#endif

namespace SimpleVIO {

// debugging
double Problem::t_hessian_cost_       = 0.0;
double Problem::t_total_hessian_cost_ = 0.0;
double Problem::t_total_solve_cost_   = 0.0;
size_t Problem::total_solve_count_    = 0;

Problem::Problem(ProblemType problem_type) : problem_type_(problem_type) {
#if defined(USE_OPENMP)
    omp_set_num_threads(4);
#endif
}

Problem::~Problem() {
    // reset global id
    global_vertex_id = 0;
    global_edge_id   = 0;
}

bool Problem::AddVertex(const std::shared_ptr<BaseVertex> &vertex) {
    if (vertices_.end() != vertices_.find(vertex->Id())) {
        // already exists
        LOGW("SimpleVIO", "Vertex [%zu] already exists", vertex->Id());

        return false;
    } else {
        vertices_.insert(std::make_pair(vertex->Id(), vertex));
    }

    if (ProblemType::SLAM_PROBLEM == problem_type_) {
        if (IsPoseVertex(vertex)) {
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }

    return true;
}

bool Problem::RemoveVertex(const std::shared_ptr<BaseVertex> &vertex) {
    if (vertices_.end() == vertices_.find(vertex->Id())) {
        // not exists
        LOGW("SimpleVIO", "Vertex [%zu] not exists", vertex->Id());

        return false;
    }

    // remove connected edges
    std::vector<std::shared_ptr<BaseEdge>> remove_edges =
        GetConnectedEdges(vertex);
    for (auto &edge : remove_edges) {
        RemoveEdge(edge);
    }

    if (IsPoseVertex(vertex)) {
        ordered_idx_pose_vertices_.erase(vertex->Id());
    } else {
        ordered_idx_landmark_vertices_.erase(vertex->Id());
    }

    vertex_id_to_edge_.erase(vertex->Id());
    vertices_.erase(vertex->Id());

    return true;
}

bool Problem::AddEdge(const std::shared_ptr<BaseEdge> &edge) {
    if (edges_.end() != edges_.find(edge->Id())) {
        // already exists
        LOGW("SimpleVIO", "Edge [%zu] already exists", edge->Id());

        return false;
    } else {
        edges_.insert(std::make_pair(edge->Id(), edge));
    }

    // add related vertices
    for (auto vertex : edge->GetVertices()) {
        vertex_id_to_edge_.insert(std::make_pair(vertex->Id(), edge));
    }

    return true;
}

bool Problem::RemoveEdge(const std::shared_ptr<BaseEdge> &edge) {
    if (edges_.end() == edges_.find(edge->Id())) {
        // not exists
        LOGW("SimpleVIO", "Edge [%zu] not exists", edge->Id());

        return false;
    }

    edges_.erase(edge->Id());

    return true;
}

// marginalize frame, remove connected edges and marginalize vertices
bool Problem::Marginalize(
    std::vector<std::shared_ptr<BaseVertex>> &marg_vertices,
    const int frame_idx,
    const int pose_dim) {
    SetOrdering();

    // frame vertex contained pre-integration
    std::vector<std::shared_ptr<BaseEdge>> marg_edges =
        GetConnectedEdges(marg_vertices[frame_idx]);

    VertexMap marg_landmark_vertices;

    // reorder landmark id
    int marg_landmark_size = 0;
    for (size_t i = 0; i < marg_edges.size(); ++i) {
        auto &vertices = marg_edges[i]->GetVertices();
        for (auto &vertex : vertices) {
            if (IsLandmarkVertex(vertex) &&
                marg_landmark_vertices.end() ==
                    marg_landmark_vertices.find(vertex->Id())) {
                vertex->SetOrderingId(pose_dim + marg_landmark_size);
                marg_landmark_size += vertex->LocalDimension();

                marg_landmark_vertices.insert(
                    std::make_pair(vertex->Id(), vertex));
            }
        }
    }

    // make Hessian and b
    const int hessian_size = pose_dim + marg_landmark_size;

    MatXX H_marg(MatXX::Zero(hessian_size, hessian_size));
    VecX b_marg(VecX::Zero(hessian_size));

    MakeHessian(marg_edges, H_marg, b_marg, hessian_size);

    // marg landmark
    const int reserve_size = pose_dim;

    if (marg_landmark_size > 0) {
        const int marg_size = marg_landmark_size;

        MatXX Hpp_schur;
        MatXX Hmp;
        MatXX Hmm_inv;
        VecX bp_schur;
        VecX bm;
        SchurDecomposition(H_marg,
                           b_marg,
                           marg_landmark_vertices,
                           reserve_size,
                           marg_size,
                           Hpp_schur,
                           Hmp,
                           Hmm_inv,
                           bp_schur,
                           bm);

        // update Hessian and b
        H_marg = Hpp_schur;
        b_marg = bp_schur;
    }

    // add prior
    if (H_prior_.rows() > 0) {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }

    // marg frame and speedbias
    int marg_dim =
        UpdateMarginalizeMatrix(marg_vertices, H_marg, b_marg, reserve_size);

    // update prior
    UpdatePrior(H_marg, b_marg, reserve_size - marg_dim, marg_dim);

    // remove vertex and remove edge
    for (auto &vertex : marg_vertices) {
        RemoveVertex(vertex);
    }

    for (auto &id_vertex : marg_landmark_vertices) {
        RemoveVertex(id_vertex.second);
    }

    return true;
}

bool Problem::IsPoseVertex(const std::shared_ptr<BaseVertex> &vertex) {
    std::string type = vertex->TypeInfo();
    return (std::string("VertexPose") == type ||
            std::string("VertexSpeedBias") == type);
}

bool Problem::IsLandmarkVertex(const std::shared_ptr<BaseVertex> &vertex) {
    std::string type = vertex->TypeInfo();
    return (std::string("VertexPointXYZ") == type ||
            std::string("VertexInverseDepth") == type);
}

void Problem::SetOrdering() {
    // reset ordering counter
    ordering_poses_     = 0;
    ordering_landmarks_ = 0;
    ordering_generic_   = 0;

    // map is ordered by vertex id
    for (auto vertex : vertices_) {
        ordering_generic_ += vertex.second->LocalDimension();

        if (ProblemType::SLAM_PROBLEM == problem_type_) {
            auto &v = vertex.second;
            if (IsPoseVertex(v)) {
                v->SetOrderingId(ordering_poses_);
                ordering_poses_ += v->LocalDimension();

                ordered_idx_pose_vertices_.insert(std::make_pair(v->Id(), v));
            } else if (IsLandmarkVertex(v)) {
                v->SetOrderingId(ordering_landmarks_);
                ordering_landmarks_ += v->LocalDimension();

                ordered_idx_landmark_vertices_.insert(
                    std::make_pair(v->Id(), v));
            } else {
                LOGW("SimpleVIO",
                     "unsupport vertex type [%s]",
                     v->TypeInfo().c_str());
            }
        }
    }

    if (ProblemType::SLAM_PROBLEM == problem_type_) {
        // make landmark id after pose id
        for (auto &vertex : ordered_idx_landmark_vertices_) {
            vertex.second->SetOrderingId(vertex.second->OrderingId() +
                                         ordering_poses_);
        }
    }

    // doe debugging
    if (!CheckOrdering()) {
        LOGE("SimpleVIO", "Wrong ordering of vertices");
    }
}

bool Problem::CheckOrdering() {
    if (ProblemType::SLAM_PROBLEM == problem_type_) {
        int current_ordering = 0;
        for (auto &v : ordered_idx_pose_vertices_) {
            if (v.second->OrderingId() != current_ordering) {
                assert(v.second->OrderingId() == current_ordering);
                return false;
            }

            current_ordering += v.second->LocalDimension();
        }

        for (auto &v : ordered_idx_landmark_vertices_) {
            if (v.second->OrderingId() != current_ordering) {
                assert(v.second->OrderingId() == current_ordering);
                return false;
            }

            current_ordering += v.second->LocalDimension();
        }
    }

    return true;
}

std::vector<std::shared_ptr<BaseEdge>> Problem::GetConnectedEdges(
    const std::shared_ptr<BaseVertex> &vertex) {
    std::vector<std::shared_ptr<BaseEdge>> ret;

    auto range = vertex_id_to_edge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {
        if (edges_.end() != edges_.find(iter->second->Id())) {
            ret.push_back(iter->second);
        }
    }

    return ret;
}

void Problem::ResizePoseHessiansWhenAddingPose(
    const std::shared_ptr<BaseVertex> &v) {
    Eigen::Index size = H_prior_.rows() + v->LocalDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->LocalDimension()).setZero();
    H_prior_.rightCols(v->LocalDimension()).setZero();
    H_prior_.bottomRows(v->LocalDimension()).setZero();
}

void Problem::ExtendHessiansPriorSize(int dim) {
    Eigen::Index size = H_prior_.rows() + dim;
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(dim).setZero();
    H_prior_.rightCols(dim).setZero();
    H_prior_.bottomRows(dim).setZero();
}

void Problem::MakeHessianWithPrior() {
    // Hessian and b
    Timer t_H;
    MakeHessian(edges_, Hessian_, b_, ordering_generic_);
    t_hessian_cost_ += t_H.End();

    // add prior
    if (H_prior_.rows() > 0) {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp  = b_prior_;

        for (auto &id_vertex : vertices_) {
            auto &vertex = id_vertex.second;

            // if pose vertex fixed, set 0 to prior
            if (IsPoseVertex(vertex) && vertex->IsFixed()) {
                size_t idx = vertex->OrderingId();
                int dim    = vertex->LocalDimension();

                H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx, dim).setZero();
            }
        }

        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_).noalias() +=
            H_prior_tmp;
        b_.head(ordering_poses_).noalias() += b_prior_tmp;
    }

    // reset delta x to 0
    delta_x_ = VecX::Zero(ordering_generic_);
}

void Problem::MakeHessian(EdgeMap &edges,
                          MatXX &H,
                          VecX &b,
                          const size_t size) {
    H = MatXX::Zero(size, size);
    b = VecX::Zero(size);

    // TODO:: accelate, accelate, accelate
#if defined(USE_OPENMP)
#pragma omp parallel
    {
        int cnt            = 0;
        const int ithread  = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        for (auto iter = edges.begin(); iter != edges.end(); ++iter, ++cnt) {
            if (ithread != cnt % nthreads) {
                continue;
            }

            AddHessianBlock(iter->second, H, b);
        }
    }
#else
    for (auto &id_edge : edges) {
        AddHessianBlock(id_edge.second, H, b);
    }
#endif
}

void Problem::MakeHessian(std::vector<std::shared_ptr<BaseEdge>> &edges,
                          MatXX &H,
                          VecX &b,
                          const size_t size) {
    H = MatXX::Zero(size, size);
    b = VecX::Zero(size);

    // TODO:: accelate, accelate, accelate
#if defined(USE_OPENMP)
#pragma omp parallel
    {
        int cnt            = 0;
        const int ithread  = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        for (auto iter = edges.begin(); iter != edges.end(); ++iter, ++cnt) {
            if (ithread != cnt % nthreads) {
                continue;
            }

            AddHessianBlock(*iter, H, b);
        }
    }
#else
    for (auto &edge : edges) {
        AddHessianBlock(edge, H, b);
    }
#endif
}

void Problem::AddHessianBlock(std::shared_ptr<BaseEdge> &edge,
                              MatXX &H,
                              VecX &b) {
    // check
    if (edge->JacobiansSize() != edge->VerticesSize()) {
        LOGE("SimpleVIO",
             "Edge: JacobiansSize [%zu] != VerticesSize [%zu]",
             edge->JacobiansSize(),
             edge->VerticesSize());
        assert(edge->JacobiansSize() == edge->VerticesSize());
    }

    // TODO:: robust cost
    edge->ComputeResidual();
    edge->ComputeJacobians();

    const int residual_dim = edge->GetResidualDimention();

    VecX residual(residual_dim);
    edge->GetResidual(residual.data());

    // robust information
    double drho = 1.0;
    MatXX robust_info(residual_dim, residual_dim);
    edge->ComputeRobustInformation(drho, robust_info.data());

    auto vertices = edge->GetVertices();
    for (size_t i = 0; i < vertices.size(); ++i) {
        auto &v_i = vertices[i];
        if (v_i->IsFixed()) {
            // Jacobian = 0
            continue;
        }

        const size_t index_i = v_i->OrderingId();
        const int dim_i      = v_i->LocalDimension();

        MatXX jacobian_i(residual_dim, dim_i);
        edge->GetJacobians((int)i, jacobian_i.data());

        MatXX JtW = jacobian_i.transpose() * robust_info;

        for (size_t j = i; j < vertices.size(); ++j) {
            auto &v_j = vertices[j];
            if (v_j->IsFixed()) {
                // Jacobian = 0
                continue;
            }

            const size_t index_j = v_j->OrderingId();
            const int dim_j      = v_j->LocalDimension();

            MatXX jacobian_j(residual_dim, dim_j);
            edge->GetJacobians((int)j, jacobian_j.data());

            MatXX hessian_block = JtW * jacobian_j;

            // accumulate Hessian block
#if defined(USE_OPENMP)
#pragma omp critical
#endif
            {
                H.block(index_i, index_j, dim_i, dim_j).noalias() +=
                    hessian_block;
                if (j != i) {
                    // symmetry
                    H.block(index_j, index_i, dim_j, dim_i).noalias() +=
                        hessian_block.transpose();
                }
            }
        }

        // accumulate b block
#if defined(USE_OPENMP)
#pragma omp critical
#endif
        { b.segment(index_i, dim_i).noalias() -= JtW * residual; }
    }
}

void Problem::SchurDecomposition(const MatXX &H,
                                 const VecX &b,
                                 const VertexMap &marg_vertices,
                                 const size_t reserve_size,
                                 const size_t marg_size,
                                 MatXX &Hpp_schur,
                                 MatXX &Hmp,
                                 MatXX &Hmm_inv,
                                 VecX &bp_schur,
                                 VecX &bm) {
    // 4 sub-blocks H(rows)(cols):
    // Hpp Hpm
    // Hmp Hmm

    // 2 sub-blocks b(rows):
    // bp
    // bm

    Hmp = H.block(reserve_size, 0, marg_size, reserve_size);
    bm  = b.segment(reserve_size, marg_size);

    MatXX Hpm = H.block(0, reserve_size, reserve_size, marg_size);
    MatXX Hmm = H.block(reserve_size, reserve_size, marg_size, marg_size);
    VecX bp   = b.segment(0, reserve_size);

    // compute inverse of Hmm
    Hmm_inv = MatXX::Zero(marg_size, marg_size);

// TODO:: use openMP
#if defined(USE_OPENMP)
#pragma omp parallel
    {
        int cnt            = 0;
        const int ithread  = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        for (auto iter = marg_vertices.begin(); iter != marg_vertices.end();
             ++iter, ++cnt) {
            if (ithread != cnt % nthreads) {
                continue;
            }

            // idx should start from 0
            auto &vertex     = iter->second;
            const size_t idx = vertex->OrderingId() - reserve_size;
            const int size   = vertex->LocalDimension();

            Hmm_inv.block(idx, idx, size, size).noalias() =
                Hmm.block(idx, idx, size, size).inverse();
        }
    }
#else
    for (auto &vertex : marg_vertices) {
        // idx should start from 0
        const size_t idx = vertex.second->OrderingId() - reserve_size;
        const int size   = vertex.second->LocalDimension();

        Hmm_inv.block(idx, idx, size, size).noalias() =
            Hmm.block(idx, idx, size, size).inverse();
    }
#endif

    MatXX tempH = Hpm * Hmm_inv;

    Hpp_schur = H.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
    bp_schur  = bp - tempH * bm;
}

/**
 *  conjugate gradient with perconditioning
 *  the jacobi PCG method
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int max_iter) {
    if (A.rows() != A.cols()) {
        LOGE("SimpleVIO", "PCG solver ERROR: A is not a square matrix");
        assert(A.rows() == A.cols());
    }

    const int rows = (int)b.rows();
    const int n    = ((max_iter < 0) ? (rows) : (max_iter));

    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();

    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w       = A * p;
    double r0z0  = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1      = r0 - alpha * w;

    const double threshold = 1e-6 * r0.norm();

    int i = 0;
    while (r1.norm() > threshold && i < n) {
        VecX z1      = M_inv * r1;
        double r1z1  = r1.dot(z1);
        double belta = r1z1 / r0z0;

        z0   = z1;
        r0z0 = r1z1;
        r0   = r1;

        p     = belta * p + z1;
        w     = A * p;
        alpha = r1z1 / p.dot(w);

        x += alpha * p;
        r1 -= alpha * w;

        i += 1;
    }

    return x;
}

void Problem::UpdateStates() {
    // update vertex
    for (auto &id_vertex : vertices_) {
        auto &vertex = id_vertex.second;
        vertex->BackUpParameters();

        VecX delta =
            delta_x_.segment(vertex->OrderingId(), vertex->LocalDimension());
        vertex->Plus(delta.data());
    }

    // update prior
    if (err_prior_.rows() > 0) {
        b_prior_backup_   = b_prior_;
        err_prior_backup_ = err_prior_;

        // update with first order Taylor, b' = b + \frac{\delta b}{\delta x} *
        // \delta x \delta x = Computes the linearized deviation from the
        // references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(H_prior_.cols());
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(Jt_prior_inv_.cols());
    }
}

void Problem::RollbackStates() {
    // roll back vertex
    for (auto &vertex : vertices_) {
        vertex.second->RollBackParameters();
    }

    // roll back prior
    if (err_prior_backup_.rows() > 0) {
        b_prior_   = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}

int Problem::UpdateMarginalizeMatrix(
    const std::vector<std::shared_ptr<BaseVertex>> &marg_vertices,
    MatXX &H_marg,
    VecX &b_marg,
    const int size) {
    int marg_dim = 0;

    // move larger index first
    for (int i = (int)(marg_vertices.size() - 1); i >= 0; --i) {
        // move the marg pose to the Hmm bottown right
        const size_t idx = marg_vertices[i]->OrderingId();
        const int dim    = marg_vertices[i]->LocalDimension();

        marg_dim += dim;

        // move row(i) to bottom
        MatXX temp_rows = H_marg.block(idx, 0, dim, size);
        MatXX temp_bottom_rows =
            H_marg.block(idx + dim, 0, size - idx - dim, size);
        H_marg.block(idx, 0, size - idx - dim, size).noalias() =
            temp_bottom_rows;
        H_marg.block(size - dim, 0, dim, size).noalias() = temp_rows;

        // move col(i) to rightmost
        MatXX temp_cols = H_marg.block(0, idx, size, dim);
        MatXX temp_right_cols =
            H_marg.block(0, idx + dim, size, size - idx - dim);
        H_marg.block(0, idx, size, size - idx - dim).noalias() =
            temp_right_cols;
        H_marg.block(0, size - dim, size, dim).noalias() = temp_cols;

        VecX temp_b      = b_marg.segment(idx, dim);
        VecX temp_b_tail = b_marg.segment(idx + dim, size - idx - dim);
        b_marg.segment(idx, size - idx - dim).noalias() = temp_b_tail;
        b_marg.segment(size - dim, dim).noalias()       = temp_b;
    }

    return marg_dim;
}

void Problem::UpdatePrior(const MatXX &H_marg,
                          const VecX &b_marg,
                          const int reserve_size,
                          const int marg_size) {
    MatXX Arr = H_marg.block(0, 0, reserve_size, reserve_size);
    MatXX Arm = H_marg.block(0, reserve_size, reserve_size, marg_size);
    MatXX Amr = H_marg.block(reserve_size, 0, marg_size, reserve_size);

    VecX br = b_marg.segment(0, reserve_size);
    VecX bm = b_marg.segment(reserve_size, marg_size);

    const double eps = 1e-8;

    MatXX Amm =
        0.5 * (H_marg.block(reserve_size, reserve_size, marg_size, marg_size) +
               H_marg.block(reserve_size, reserve_size, marg_size, marg_size)
                   .transpose());
    Eigen::SelfAdjointEigenSolver<MatXX> eigen_solver(Amm);
    MatXX Amm_inv =
        eigen_solver.eigenvectors() *
        VecX((eigen_solver.eigenvalues().array() > eps)
                 .select(eigen_solver.eigenvalues().array().inverse(), 0.0))
            .asDiagonal() *
        eigen_solver.eigenvectors().transpose();

    MatXX tempB = Arm * Amm_inv;

    H_prior_ = Arr - tempB * Amr;
    b_prior_ = br - tempB * bm;

    Eigen::SelfAdjointEigenSolver<MatXX> eigen_solver2(H_prior_);
    VecX S((eigen_solver2.eigenvalues().array() > eps)
               .select(eigen_solver2.eigenvalues().array(), 0.0));
    VecX S_inv =
        VecX((eigen_solver2.eigenvalues().array() > eps)
                 .select(eigen_solver2.eigenvalues().array().inverse(), 0.0));

    VecX S_sqrt     = S.cwiseSqrt();
    VecX S_inv_sqrt = S_inv.cwiseSqrt();

    Jt_prior_inv_ =
        S_inv_sqrt.asDiagonal() * eigen_solver2.eigenvectors().transpose();
    err_prior_ = -Jt_prior_inv_ * b_prior_;

    MatXX J   = S_sqrt.asDiagonal() * eigen_solver2.eigenvectors().transpose();
    MatXX JTJ = J.transpose() * J;
    H_prior_  = MatXX((JTJ.array().abs() > 1e-9).select(JTJ.array(), 0.0));
}

double Problem::ComputeChi2(bool need_compute_residual) {
    double chi2 = 0.0;
    for (auto &edge : edges_) {
        if (need_compute_residual) {
            edge.second->ComputeResidual();
        }

        chi2 += edge.second->RobustChi2();
    }

    if (err_prior_.size() > 0) {
        chi2 += err_prior_.norm();
    }

    chi2 *= 0.5;  // 1/2 * err^2

    return chi2;
}

}  // namespace SimpleVIO
