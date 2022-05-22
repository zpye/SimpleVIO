#ifndef SIMPLE_VIO_OPTIMIZE_PROBLEM_H_
#define SIMPLE_VIO_OPTIMIZE_PROBLEM_H_

#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>

#include "base_edge.h"
#include "base_vertex.h"
#include "loss_function.h"
#include "utils/eigen_types.h"

namespace SimpleVIO {

class Problem {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using VertexMap = std::map<size_t, std::shared_ptr<BaseVertex>>;
    using EdgeMap   = std::unordered_map<size_t, std::shared_ptr<BaseEdge>>;
    using VertexIdEdgeMap =
        std::unordered_multimap<size_t, std::shared_ptr<BaseEdge>>;

public:
    enum class ProblemType { GENERIC_PROBLEM = 0, SLAM_PROBLEM = 1 };

    enum class OptimizationMethod {
        LM_NIELSON   = 0,
        LM_MARQUARDT = 1,
        DOGLEG       = 2
    };

    Problem(ProblemType problem_type);

    ~Problem();

public:
    virtual bool Solve(int iterations = 10) = 0;

public:
    bool AddVertex(const std::shared_ptr<BaseVertex> &vertex);

    bool RemoveVertex(const std::shared_ptr<BaseVertex> &vertex);

    bool AddEdge(const std::shared_ptr<BaseEdge> &edge);

    bool RemoveEdge(const std::shared_ptr<BaseEdge> &edge);

    bool Marginalize(std::vector<std::shared_ptr<BaseVertex>> &marg_vertices,
                     const int frame_idx,
                     const int pose_dim);

    void SetHessianPrior(const MatXX &H) {
        H_prior_ = H;
    }

    MatXX GetHessianPrior() {
        return H_prior_;
    }

    void SetbPrior(const VecX &b) {
        b_prior_ = b;
    }

    VecX GetbPrior() {
        return b_prior_;
    }

    void SetErrPrior(const VecX &b) {
        err_prior_ = b;
    }

    VecX GetErrPrior() {
        return err_prior_;
    }

    void SetJtPrior(const MatXX &J) {
        Jt_prior_inv_ = J;
    }

    MatXX GetJtPrior() {
        return Jt_prior_inv_;
    }

    void ExtendHessiansPriorSize(int dim);

protected:
    bool IsPoseVertex(const std::shared_ptr<BaseVertex> &vertex);

    bool IsLandmarkVertex(const std::shared_ptr<BaseVertex> &vertex);

    void SetOrdering();

    bool CheckOrdering();

    std::vector<std::shared_ptr<BaseEdge>> GetConnectedEdges(
        const std::shared_ptr<BaseVertex> &vertex);

    void ResizePoseHessiansWhenAddingPose(const std::shared_ptr<BaseVertex> &v);

protected:
    void MakeHessianWithPrior();

    void MakeHessian(EdgeMap &edges, MatXX &H, VecX &b, const size_t size);

    void MakeHessian(std::vector<std::shared_ptr<BaseEdge>> &edges,
                     MatXX &H,
                     VecX &b,
                     const size_t size);

    void AddHessianBlock(std::shared_ptr<BaseEdge> &edge, MatXX &H, VecX &b);

    void SchurDecomposition(const MatXX &H,
                            const VecX &b,
                            const VertexMap &marg_vertices,
                            const size_t reserve_size,
                            const size_t marg_size,
                            MatXX &Hpp_schur,
                            MatXX &Hmp,
                            MatXX &Hmm_inv,
                            VecX &bp_schur,
                            VecX &bm);

    VecX PCGSolver(const MatXX &A, const VecX &b, int max_iter = -1);

    void UpdateStates();

    void RollbackStates();

    int UpdateMarginalizeMatrix(
        const std::vector<std::shared_ptr<BaseVertex>> &marg_vertices,
        MatXX &H_marg,
        VecX &b_marg,
        const int size);

    void UpdatePrior(const MatXX &H_marg,
                     const VecX &b_marg,
                     const int reserve_size,
                     const int marg_size);

    double ComputeChi2(bool need_compute_residual);

protected:
    ProblemType problem_type_;

    // solve Hx=b, use the result x to update state vector
    MatXX Hessian_;
    VecX b_;
    VecX delta_x_;

    // prior
    MatXX H_prior_;
    VecX b_prior_;
    VecX b_prior_backup_;  // for rolling back
    VecX err_prior_;
    VecX err_prior_backup_;  // for rolling back
    MatXX Jt_prior_inv_;

protected:
    // all vertices
    VertexMap vertices_;

    // all edges
    EdgeMap edges_;

    // vertex id to edge
    VertexIdEdgeMap vertex_id_to_edge_;

    ///// for marginalization /////
    // ordering related
    size_t ordering_poses_     = 0;
    size_t ordering_landmarks_ = 0;
    size_t ordering_generic_   = 0;

    VertexMap ordered_idx_pose_vertices_;
    VertexMap ordered_idx_landmark_vertices_;

    // verticies need to marg. <Ordering_id_, Vertex>
    VertexMap vertices_marg_;

public:
    // for debugging
    static double t_hessian_cost_;
    static double t_total_hessian_cost_;
    static double t_total_solve_cost_;
    static size_t total_solve_count_;
};

using ProblemPtr = std::shared_ptr<Problem>;

ProblemPtr CreateProblemPtr(Problem::OptimizationMethod opt_method,
                            Problem::ProblemType problem_type);

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_PROBLEM_H_
