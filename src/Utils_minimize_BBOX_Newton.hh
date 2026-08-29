/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2026                                                      |
 |                                                                          |
 |      Enrico Bertolazzi                                                   |
 |      Dipartimento di Ingegneria Industriale                              |
 |      Universita degli Studi di Trento                                    |
 |                                                                          |
\*--------------------------------------------------------------------------*/

/**
 * @file Utils_minimize_BBOX_Newton_TRON_v3_doxygen.hh
 * @brief Dense box-constrained Newton solver with cubic regularization,
 *        semismooth projected Newton steps, TRON-inspired safeguards, and
 *        terminal high-accuracy polishing.
 *
 * @details
 * This header implements a dense second-order method for the bound-constrained
 * optimization problem
 *
 * @f[
 *   \min_{x\in\mathbb{R}^n} f(x)
 *   \qquad\text{subject to}\qquad
 *   \ell \le x \le u,
 * @f]
 *
 * where @f$f:\mathbb{R}^n\to\mathbb{R}@f$ is assumed sufficiently smooth and
 * the user provides the objective, gradient and dense Hessian.  The algorithm
 * is intentionally written so that the box enters only through the Euclidean
 * projection
 *
 * @f[
 *   P_{[\ell,u]}(z)_i = \min\{u_i,\max\{\ell_i,z_i\}\}.
 * @f]
 *
 * The central stationarity mapping is
 *
 * @f[
 *   p(x) = x - P_{[\ell,u]}\bigl(x-\nabla f(x)\bigr).
 * @f]
 *
 * A point is first-order stationary for the box problem if and only if
 * @f$p(x)=0@f$.  This formulation is the key design principle of the solver:
 * when the box becomes unbounded, @f$[\ell,u]=\mathbb{R}^n@f$, the projection
 * is the identity and therefore
 *
 * @f[
 *   p(x)=\nabla f(x).
 * @f]
 *
 * Consequently, every projected formula below reduces smoothly to its
 * unconstrained counterpart without a dedicated constrained/unconstrained
 * branch.
 *
 * @section bbox_newton_overview Algorithmic overview
 *
 * Each outer iteration uses the following hierarchy.
 *
 * 1. **Terminal semismooth Newton polishing.**  If the projected residual is
 *    sufficiently small, the algorithm tries a high-accuracy Newton correction
 *    on the generalized Jacobian of @f$p(x)@f$.
 *
 * 2. **Primary semismooth Newton candidate.**  When the free Hessian block is
 *    numerically positive definite, a projected Newton step is attempted first.
 *    The candidate is accepted by a TRON-like actual/predicted reduction test.
 *
 * 3. **Adaptive cubic-regularized Newton step.**  If the primary Newton step is
 *    unavailable or rejected, the solver computes
 *
 *    @f[
 *      \lambda_k = \sqrt{M_k\,\|p_k\|_2},
 *    @f]
 *
 *    and solves
 *
 *    @f[
 *      \bigl(H_k+\lambda_k I\bigr)d_k = p_k,
 *      \qquad H_k=\nabla^2 f(x_k).
 *    @f]
 *
 *    The actual trial point is always projected,
 *
 *    @f[
 *      x_k^+ = P_{[\ell,u]}(x_k-d_k),
 *      \qquad s_k=x_k-x_k^+.
 *    @f]
 *
 *    For an unbounded box, @f$p_k=g_k@f$, @f$x_k^+=x_k-d_k@f$ and
 *    @f$s_k=d_k@f$, hence the step is exactly the unconstrained
 *    cubic-regularized Newton step.
 *
 * 4. **Semismooth-Newton rescue.**  If a bounded number of cubic trials fails,
 *    a few semismooth Newton corrections are attempted using the same
 *    generalized Jacobian employed by the terminal polishing phase.
 *
 * 5. **Projected Cauchy rescue.**  As a final globalization safeguard, a
 *    projected gradient/Cauchy point is searched using only the frozen quadratic
 *    model before paying for an objective evaluation.  This mirrors one of the
 *    most effective safeguards of TRON.
 *
 * 6. **Safe fallback.**  A previously computed trial may be accepted only if it
 *    is non-increasing in the objective up to roundoff and strictly improves the
 *    projected residual.  A rejected/ascent trial is never promoted merely
 *    because the internal search budget was exhausted.
 *
 * @section bbox_newton_stationarity Projected first-order stationarity
 *
 * Define
 *
 * @f[
 *   y(x)=x-g(x),\qquad g(x)=\nabla f(x),
 * @f]
 *
 * and
 *
 * @f[
 *   p(x)=x-P_{[\ell,u]}(y(x)).
 * @f]
 *
 * Componentwise, @f$p_i(x)=0@f$ is equivalent to the usual KKT conditions:
 *
 * @f[
 * \begin{cases}
 *   g_i(x)=0, & \ell_i<x_i<u_i,\\
 *   g_i(x)\ge 0, & x_i=\ell_i,\\
 *   g_i(x)\le 0, & x_i=u_i.
 * \end{cases}
 * @f]
 *
 * The solver monitors both @f$\|p(x)\|_2@f$ and
 * @f$\|p(x)\|_\infty@f$.  The infinity norm drives termination, while the
 * Euclidean norm enters the cubic regularization parameter.
 *
 * @section bbox_newton_semismooth Semismooth Newton model
 *
 * Let @f$D@f$ be a diagonal generalized derivative of the projection evaluated
 * at @f$y=x-g(x)@f$.  Away from projection kinks,
 *
 * @f[
 *   D_{ii}=\begin{cases}
 *     1, & \ell_i<y_i<u_i,\\
 *     0, & y_i<\ell_i\ \text{or}\ y_i>u_i.
 *   \end{cases}
 * @f]
 *
 * The generalized Jacobian of @f$p@f$ is
 *
 * @f[
 *   J_p(x)
 *   = I-D(I-H)
 *   = (I-D)+DH,
 *   \qquad H=\nabla^2 f(x).
 * @f]
 *
 * After reordering variables into active components @f$A@f$ and free
 * components @f$F@f$, this matrix has the exact block structure
 *
 * @f[
 *   J_p =
 *   \begin{bmatrix}
 *     I      & 0\\
 *     H_{FA} & H_{FF}
 *   \end{bmatrix}.
 * @f]
 *
 * Therefore the semismooth Newton system @f$J_p d=p@f$ can be solved through
 *
 * @f[
 *   d_A=p_A,
 * @f]
 *
 * followed by
 *
 * @f[
 *   H_{FF}d_F = p_F-H_{FA}d_A.
 * @f]
 *
 * The primary Newton candidate is attempted only when @f$H_{FF}@f$ is
 * numerically positive definite.  This avoids relying on positivity of the
 * generally nonsymmetric full generalized Jacobian.  If the box is unbounded,
 * @f$A=\varnothing@f$, @f$F=\{1,\ldots,n\}@f$ and the same formulas reduce to
 *
 * @f[
 *   H d = g.
 * @f]
 *
 * @section bbox_newton_primary Primary Newton globalization
 *
 * The primary semismooth Newton direction is globalized on the frozen quadratic
 * model
 *
 * @f[
 *   q_k(s)=g_k^T s+\frac12 s^T H_k s.
 * @f]
 *
 * For a projected trial
 *
 * @f[
 *   x^+(\alpha)=P_{[\ell,u]}(x_k-\alpha d_k),
 *   \qquad s(\alpha)=x^+(\alpha)-x_k,
 * @f]
 *
 * model backtracking seeks a step satisfying a sufficient model decrease
 * condition of the form
 *
 * @f[
 *   q_k(s)\le \mu_0\,g_k^Ts,
 *   \qquad 0<\mu_0\ll1.
 * @f]
 *
 * Only after this inexpensive model-only search is successful is the objective
 * evaluated.  The candidate is then assessed through the trust-region-style
 * agreement ratio
 *
 * @f[
 *   \rho_k =
 *   \frac{f(x_k)-f(x_k^+)}{-q_k(s_k)}.
 * @f]
 *
 * A sufficiently positive ratio accepts the step.  Very good or poor model
 * agreement is also used to decrease or increase the cubic regularization
 * estimate @f$M_k@f$.
 *
 * @section bbox_newton_cubic Cubic-regularized Newton step
 *
 * The fallback second-order model follows the adaptive Newton/cubic
 * regularization structure
 *
 * @f[
 *   (H_k+\lambda_k I)d_k=p_k,
 *   \qquad
 *   \lambda_k=\sqrt{M_k\|p_k\|_2}.
 * @f]
 *
 * The projected trial is
 *
 * @f[
 *   x_k^+=P_{[\ell,u]}(x_k-d_k),
 *   \qquad
 *   s_k=x_k-x_k^+.
 * @f]
 *
 * A cheap quadratic-model safeguard is applied before any objective evaluation.
 * With @f$z_k=x_k^+-x_k=-s_k@f$,
 *
 * @f[
 *   q_k(z_k)=g_k^Tz_k+\frac12 z_k^T H_k z_k.
 * @f]
 *
 * Trials whose model reduction is not sufficiently descent-like are rejected
 * immediately and @f$M_k@f$ is increased.
 *
 * A finite trial that passes the model prescreen is accepted only if the two
 * adaptive cubic conditions hold:
 *
 * @f[
 *   \|p(x_k^+)\|_2
 *   \le c_g\,\lambda_k\|s_k\|_2,
 * @f]
 *
 * and
 *
 * @f[
 *   f(x_k^+)
 *   \le
 *   f(x_k)-c_f\,\lambda_k\|s_k\|_2^2 + \delta_{\rm fp},
 * @f]
 *
 * where the defaults are @f$c_g=2@f$ and @f$c_f=2/3@f$, and
 * @f$\delta_{\rm fp}@f$ is a scale-aware roundoff allowance.  For an unbounded
 * box these conditions become exactly the corresponding unconstrained formulas
 * with @f$p=g@f$ and @f$s=d@f$.
 *
 * @section bbox_newton_M Adaptation of the Hessian-Lipschitz estimate
 *
 * The scalar @f$M_k@f$ controls the cubic regularization through
 * @f$\lambda_k=\sqrt{M_k\|p_k\|}@f$.  A successful outer iteration starts the
 * next search from a smaller value,
 *
 * @f[
 *   M_{k+1}^{(0)} = \gamma_{\downarrow} M_k,
 *   \qquad 0<\gamma_{\downarrow}<1,
 * @f]
 *
 * while unsuccessful cubic trials increase it,
 *
 * @f[
 *   M \leftarrow \gamma_{\uparrow}M,
 *   \qquad \gamma_{\uparrow}>1.
 * @f]
 *
 * Factorization failures use a more aggressive multiplier.  After a successful
 * step the actual/predicted ratio may further decrease or increase @f$M@f$.
 * This mechanism plays a role analogous to trust-region expansion/contraction.
 *
 * @section bbox_newton_cauchy Projected Cauchy rescue
 *
 * If both semismooth and cubic Newton candidates fail, the solver searches
 * along the projected gradient path
 *
 * @f[
 *   x(\alpha)=P_{[\ell,u]}\bigl(x_k-\alpha g_k\bigr),
 *   \qquad s(\alpha)=x(\alpha)-x_k.
 * @f]
 *
 * Candidate values of @f$\alpha@f$ are tested first using only the frozen
 * quadratic model.  The search may shrink or geometrically enlarge
 * @f$\alpha@f$, and the accepted Cauchy scale is retained between iterations.
 * Only the selected candidate incurs an objective evaluation.  The final
 * acceptance test is an Armijo condition
 *
 * @f[
 *   f(x_k+s)\le f(x_k)+c_1 g_k^Ts+\delta_{\rm fp}.
 * @f]
 *
 * @section bbox_newton_polish Terminal high-accuracy polishing
 *
 * Once @f$\|p(x)\|_\infty@f$ is below the polishing trigger, the solver applies
 * semismooth Newton iterations directly to
 *
 * @f[
 *   J_p(x)d=p(x).
 * @f]
 *
 * The trial remains
 *
 * @f[
 *   x^+(\alpha)=P_{[\ell,u]}(x-\alpha d).
 * @f]
 *
 * Backtracking is driven primarily by contraction of the projected residual,
 * while allowing only roundoff-level objective increase.  This phase is what
 * allows the method to reach projected-gradient residuals near machine
 * precision.  On an unbounded box, @f$J_p=H@f$ and the polishing phase is the
 * ordinary damped Newton method.
 *
 * @section bbox_newton_fallback Safe fallback policy
 *
 * During cubic backtracking the best complete trial is retained according to a
 * merit measure combining residual and sufficient-decrease violations.  If all
 * formal acceptance tests fail, this trial can be promoted only when
 *
 * @f[
 *   f_{\rm trial}\le f_k + \delta_{\rm fp}
 * @f]
 *
 * and
 *
 * @f[
 *   \|p_{\rm trial}\|_2 < \|p_k\|_2.
 * @f]
 *
 * This rule prevents a finite-search budget from silently accepting an ascent
 * step.
 *
 * @section bbox_newton_reduction Smooth reduction to the unconstrained method
 *
 * The implementation deliberately contains no algorithmic branch that asks
 * whether the bounds are finite.  Setting
 *
 * @f[
 *   \ell_i=-\infty,\qquad u_i=+\infty
 * @f]
 *
 * for every component gives
 *
 * @f[
 *   P(z)=z,\qquad p(x)=g(x),\qquad D=I,
 * @f]
 *
 * and therefore
 *
 * @f[
 *   J_p=H,
 *   \qquad
 *   (H+\lambda I)d=g,
 *   \qquad
 *   x^+=x-d.
 * @f]
 *
 * Hence the box solver reduces algebraically to the corresponding unconstrained
 * Newton/cubic algorithm instead of switching to a separate implementation.
 *
 * @section bbox_newton_complexity Computational remarks
 *
 * The implementation targets small and medium dense problems.  The dominant
 * linear-algebra costs are dense factorizations of @f$H_{FF}@f$ or
 * @f$H+\lambda I@f$, nominally @f$O(n^3)@f$.  The Hessian is evaluated once per
 * outer iteration for the globalized step, while multiple cubic candidates can
 * reuse the same dense Hessian and differ only by the diagonal shift
 * @f$\lambda I@f$.  Model-only prescreens are used aggressively to avoid
 * unnecessary objective/gradient evaluations.
 *
 * @section bbox_newton_requirements Problem interface
 *
 * The problem object must provide
 *
 * @code{.cpp}
 * Real objective(ConstVectorRef<Real> x);
 * void gradient(ConstVectorRef<Real> x, VectorRef<Real> g);
 * void hessian(ConstVectorRef<Real> x, MatrixRef<Real> H);
 * @endcode
 *
 * The supplied Hessian is assumed to represent the dense Hessian of the
 * objective at the requested point.  The solver uses Eigen dense matrices and
 * requires Eigen 5 or newer.
 *
 * @note This documented file is algorithmically identical to the tested V3
 *       implementation; only Doxygen/documentation comments have been added.
 */

#pragma once

#ifndef UTILS_MINIMIZE_BBOX_NEWTON_DOT_HH
#define UTILS_MINIMIZE_BBOX_NEWTON_DOT_HH

#include "Utils_eigen.hh"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if EIGEN_MAJOR_VERSION < 5
#error "Utils::MinimizeNewton requires Eigen 5 or newer"
#endif

namespace Utils::MinimizeNewton
{

  template <typename Real> using Vector         = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
  template <typename Real> using Matrix         = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
  template <typename Real> using ConstVectorRef = Eigen::Ref<Vector<Real> const>;
  template <typename Real> using VectorRef      = Eigen::Ref<Vector<Real>>;
  template <typename Real> using MatrixRef      = Eigen::Ref<Matrix<Real>>;

  /**
   * @brief Compile-time interface required from an optimization problem.
   *
   * A compatible problem must provide objective, gradient and dense Hessian
   * evaluations at an Eigen vector.  The concept intentionally imposes no
   * ownership model and therefore works with lightweight wrappers as well as
   * full problem classes.
   */
  template <typename Problem, typename Real>
  concept ProblemFor = requires( Problem & p, ConstVectorRef<Real> x, VectorRef<Real> g, MatrixRef<Real> H ) {
    { p.objective( x ) } -> std::convertible_to<Real>;
    { p.gradient( x, g ) };
    { p.hessian( x, H ) };
  };

  /**
   * @brief Adapter turning three callables into a ProblemFor-compatible object.
   *
   * @tparam Real Scalar type.
   * @tparam Obj  Objective callable.
   * @tparam Grad Gradient callable.
   * @tparam Hess Hessian callable.
   */
  template <typename Real, typename Obj, typename Grad, typename Hess> class CallableProblem
  {
  public:
    CallableProblem( Obj obj, Grad grad, Hess hess )
      : m_obj( std::move( obj ) ), m_grad( std::move( grad ) ), m_hess( std::move( hess ) )
    {
    }

    Real objective( ConstVectorRef<Real> x ) { return static_cast<Real>( m_obj( x ) ); }
    void gradient( ConstVectorRef<Real> x, VectorRef<Real> g ) { m_grad( x, g ); }
    void hessian( ConstVectorRef<Real> x, MatrixRef<Real> H ) { m_hess( x, H ); }

  private:
    Obj  m_obj;
    Grad m_grad;
    Hess m_hess;
  };

  template <typename Real = double, typename Obj, typename Grad, typename Hess>
  auto make_problem( Obj obj, Grad grad, Hess hess )
  { return CallableProblem<Real, Obj, Grad, Hess>( std::move( obj ), std::move( grad ), std::move( hess ) ); }

  /** @brief Termination code returned by Solver::solve(). */
  enum class Status
  {
    unknown,
    converged,
    max_iterations,
    max_function_evaluations,
    no_progress,
    non_finite_objective,
    non_finite_gradient,
    non_finite_hessian,
    eigensolver_failure,
    user
  };

  [[nodiscard]] constexpr std::string_view to_string( Status status ) noexcept
  {
    switch ( status )
    {
      case Status::unknown: return "unknown";
      case Status::converged: return "converged";
      case Status::max_iterations: return "maximum iterations";
      case Status::max_function_evaluations: return "maximum function evaluations";
      case Status::no_progress: return "no acceptable projected Newton step";
      case Status::non_finite_objective: return "non-finite objective";
      case Status::non_finite_gradient: return "non-finite gradient";
      case Status::non_finite_hessian: return "non-finite Hessian";
      case Status::eigensolver_failure: return "linear solver failure";
      case Status::user: return "user request";
    }
    return "unknown";
  }

  /**
   * @brief Numerical and globalization parameters of the solver.
   *
   * The defaults are tuned for dense double-precision problems and mirror the
   * tested V3 configuration.  Several legacy fields are intentionally retained
   * for API compatibility even when the present algorithm does not use them.
   */
  template <typename Real = double> struct Options
  {
    static constexpr Real eps = std::numeric_limits<Real>::epsilon();

    // Keep these independent from the projection machinery.  If the same
    // tolerance is used by MinimizeNewtonCubic, the stopping test is identical
    // on an unbounded box because p(x) == g(x).
    Real absolute_tolerance = Real( 128 ) * eps;
    Real relative_tolerance = Real( 128 ) * eps;

    // Kept for API compatibility with the previous BBOX solver.
    Real step_tolerance      = Real( 0 );
    Real curvature_tolerance = Real( 0 );

    int max_iterations           = 400;
    int max_function_evaluations = -1;
    int max_sub_iterations       = 50;
    // Do not spend the whole backtracking budget before trying the Newton/Cauchy
    // rescues.  The absolute max_sub_iterations is still retained as an API cap.
    int cubic_trials_before_rescue = 6;

    // Same adaptive cubic parameters as MinimizeNewtonCubic.
    Real hessian_lipschitz_initial = Real( 1 );
    Real hessian_lipschitz_min     = std::sqrt( eps );  // compatibility
    Real hessian_lipschitz_max     = Real( 1 ) / eps;   // compatibility
    Real regularization_increase   = Real( 2 );
    Real regularization_decrease   = Real( 1 ) / Real( 2 );
    Real factorization_increase    = Real( 10 );
    Real roundoff_factor           = Real( 16 );
    Real linear_residual_factor    = Real( 128 );  // compatibility, deliberately unused
    Real maximum_step_norm         = Real( 100 );  // compatibility, deliberately unused

    Real gradient_acceptance_factor = Real( 2 );
    Real decrease_factor            = Real( 2 ) / Real( 3 );

    // TRON-inspired safeguards. They are written only through the projected
    // residual/step, so for an unbounded box they reduce to the corresponding
    // unconstrained formulas without a constrained/unconstrained branch.
    bool enable_model_safeguard = true;
    Real model_armijo           = Real( 1 ) / Real( 100 );

    // TRON-style projected Cauchy rescue.  The search is performed first on
    // the frozen quadratic model, therefore many candidate step lengths cost no
    // objective/gradient evaluations.  alpha is retained between outer iterations.
    bool enable_gradient_rescue = true;
    int  max_gradient_rescue    = 20;
    Real gradient_rescue_shrink = Real( 1 ) / Real( 10 );
    Real gradient_rescue_armijo = Real( 1e-4 );
    Real cauchy_expand          = Real( 10 );
    Real cauchy_alpha_min       = std::sqrt( std::numeric_limits<Real>::denorm_min() );
    Real cauchy_alpha_max       = Real( 1e8 );
    Real cauchy_radius_factor   = Real( 2 );

    // Primary semismooth-Newton candidate.  The generalized Jacobian is
    //
    //   J_p = (I-D) + D H,
    //
    // and, after ordering active/free components, has the block form
    //
    //   [ I      0   ]
    //   [ H_FA  H_FF ].
    //
    // Hence a safe Newton candidate exists when the free Hessian H_FF is
    // numerically positive definite.  For an unbounded box D=I, F={1,...,n},
    // and this test/solve becomes the ordinary pure Newton step H d = g.
    bool enable_primary_semismooth      = true;
    Real primary_pd_factor              = Real( 256 );
    Real primary_acceptance_threshold   = Real( 1e-4 );
    int  max_primary_model_backtracking = 12;
    Real primary_model_shrink           = Real( 1 ) / Real( 2 );

    // If the cubic backtracking still cannot produce a step, retain the more
    // permissive semismooth rescue used by the previous revision.
    bool enable_semismooth_rescue     = true;
    int  semismooth_rescue_iterations = 2;

    // Use TRON's actual/predicted reduction ratio only to adapt M after a cubic
    // step has already passed the AdaN acceptance tests.  Thus it changes speed,
    // not the fundamental acceptance criterion.
    Real ratio_decrease_threshold = Real( 1 ) / Real( 4 );
    Real ratio_increase_threshold = Real( 3 ) / Real( 4 );
    Real ratio_good_M_factor      = Real( 1 ) / Real( 2 );
    Real ratio_bad_M_factor       = Real( 2 );

    // Terminal high-accuracy Newton polishing.  The formulas are again written
    // only through the projected residual p.  For [-inf,+inf]^n they reduce to
    // the pure Newton polishing used by MinimizeNewtonCubic.
    bool enable_polishing          = true;
    int  max_polish_iterations     = 12;
    int  max_polish_backtracking   = 12;
    Real polish_trigger            = std::sqrt( std::sqrt( eps ) );
    Real polish_tolerance          = Real( 0 );  // 0 -> 128*eps
    Real polish_armijo             = Real( 1e-4 );
    Real projection_derivative_tol = Real( 64 ) * eps;
    int  linear_refinement_steps   = 0;           // compatibility; off to preserve exact reduction
    Real polish_bound_factor       = Real( 64 );  // compatibility

    int verbose = 0;

    void set_tolerances( Real tolerance )
    {
      absolute_tolerance = tolerance;
      relative_tolerance = tolerance;
    }
  };

  /**
   * @brief Complete solver result and diagnostic counters.
   *
   * Besides the final point and objective value, this structure exposes the
   * projected residual, regularization state, evaluation counters and the
   * number of rejected/rescue/polishing steps.  These counters are useful when
   * profiling globalization behaviour on difficult nonconvex problems.
   */
  template <typename Real = double> struct Result
  {
    Vector<Real> x;
    Real         objective                     = std::numeric_limits<Real>::quiet_NaN();
    Real         projected_gradient_norm       = std::numeric_limits<Real>::quiet_NaN();
    Real         projected_gradient_norm_inf   = std::numeric_limits<Real>::quiet_NaN();
    Real         primal_feasibility            = Real( 0 );
    Real         step_norm                     = std::numeric_limits<Real>::infinity();
    Real         lambda                        = Real( 0 );
    Real         hessian_lipschitz_estimate    = Real( 0 );
    Real         optimality_tolerance          = std::numeric_limits<Real>::quiet_NaN();
    Real         minimum_critical_eigenvalue   = std::numeric_limits<Real>::quiet_NaN();
    Real         effective_step_tolerance      = std::numeric_limits<Real>::quiet_NaN();
    Real         effective_curvature_tolerance = std::numeric_limits<Real>::quiet_NaN();
    int          iterations                    = 0;
    int          function_evaluations          = 0;
    int          gradient_evaluations          = 0;
    int          hessian_evaluations           = 0;
    int          rejected_steps                = 0;
    int          fallback_steps                = 0;
    int          model_rejections              = 0;
    int          gradient_rescue_steps         = 0;
    int          primary_semismooth_attempts   = 0;
    int          primary_semismooth_steps      = 0;
    int          primary_semismooth_rejected   = 0;
    int          semismooth_rescue_steps       = 0;
    int          polish_iterations             = 0;
    int          polish_backtracks             = 0;
    Status       status                        = Status::unknown;

    [[nodiscard]] bool solved() const noexcept { return status == Status::converged; }
  };

  namespace detail
  {
    // P_[l,u](x).  This is the ONLY place where the box enters the globalized
    // cubic iteration.  For l=-inf,u=+inf the expression is exactly x.
    template <typename Z, typename X, typename L, typename U> void project(
      Eigen::MatrixBase<Z> &       z,
      Eigen::MatrixBase<X> const & x,
      Eigen::MatrixBase<L> const & lower,
      Eigen::MatrixBase<U> const & upper )
    { z = x.cwiseMax( lower ).cwiseMin( upper ); }

  }  // namespace detail

  /**
   * @brief Dense box-constrained Newton/cubic solver.
   *
   * @tparam Real Floating-point scalar type.
   *
   * The class owns all dense work vectors/matrices so repeated calls can reuse
   * allocations after resize().  The algorithm itself is described in the
   * file-level documentation above.
   */
  template <typename Real = double> class Solver
  {
  public:
    using Vec = Vector<Real>;
    using Mat = Matrix<Real>;

    explicit Solver( Eigen::Index dimension = 0, Options<Real> options = {} ) : m_options( options )
    { resize( dimension ); }

    [[nodiscard]] Options<Real> &       options() noexcept { return m_options; }
    [[nodiscard]] Options<Real> const & options() const noexcept { return m_options; }

    /**
     * @brief Resize all internal dense workspaces.
     * @param dimension Number of optimization variables.
     */
    void resize( Eigen::Index dimension )
    {
      m_n = dimension;
      m_x.resize( m_n );
      m_trial.resize( m_n );
      m_gradient.resize( m_n );
      m_trial_gradient.resize( m_n );
      m_projected_gradient.resize( m_n );
      m_trial_projected_gradient.resize( m_n );
      m_direction.resize( m_n );
      m_step.resize( m_n );
      m_model_Hstep.resize( m_n );
      m_projection_slope.resize( m_n );
      m_reduced_rhs.resize( m_n );
      m_reduced_solution.resize( m_n );
      m_H.resize( m_n, m_n );
      m_shifted_H.resize( m_n, m_n );
      m_polish_J.resize( m_n, m_n );
      m_reduced_H.resize( m_n, m_n );
      m_free_indices.reserve( static_cast<std::size_t>( m_n ) );
    }

    template <typename Problem>
      requires ProblemFor<Problem, Real>
    /**
     * @brief Solve the box-constrained problem without a user callback.
     * @param problem Problem object providing objective/gradient/Hessian.
     * @param x0 Initial point; it is projected onto the box before evaluation.
     * @param lower Componentwise lower bounds.
     * @param upper Componentwise upper bounds.
     * @return Final Result structure.
     */
    Result<Real> solve(
      Problem &            problem,
      ConstVectorRef<Real> x0,
      ConstVectorRef<Real> lower,
      ConstVectorRef<Real> upper )
    {
      auto keep_going = []( Result<Real> const & ) { return true; };
      return solve( problem, x0, lower, upper, keep_going );
    }

    template <typename Problem, typename Callback>
      requires ProblemFor<Problem, Real>
    /**
     * @brief Solve the box-constrained problem with an iteration callback.
     *
     * @param problem Problem object providing objective/gradient/Hessian.
     * @param x0 Initial point; projected onto @f$[\ell,u]@f$ before use.
     * @param lower Lower bound vector.
     * @param upper Upper bound vector.
     * @param callback Invoked with a transient Result after initialization and
     *                 after each accepted outer step. Returning false stops the
     *                 solver with Status::user.
     *
     * @details
     * The method implements the complete globalization hierarchy documented at
     * file scope.  In particular, all first-order optimality tests are expressed
     * through @f$p(x)=x-P(x-g(x))@f$, ensuring algebraic reduction to the
     * unconstrained method when the bounds are infinite.
     */
    Result<Real> solve(
      Problem &            problem,
      ConstVectorRef<Real> x0,
      ConstVectorRef<Real> lower,
      ConstVectorRef<Real> upper,
      Callback &&          callback )
    {
      constexpr Real eps     = std::numeric_limits<Real>::epsilon();
      auto const &   options = m_options;

      resize( x0.size() );
      m_function_evaluations        = 0;
      m_gradient_evaluations        = 0;
      m_hessian_evaluations         = 0;
      m_rejected_steps              = 0;
      m_fallback_steps              = 0;
      m_model_rejections            = 0;
      m_gradient_rescue_steps       = 0;
      m_primary_semismooth_attempts = 0;
      m_primary_semismooth_steps    = 0;
      m_primary_semismooth_rejected = 0;
      m_semismooth_rescue_steps     = 0;
      m_polish_iterations           = 0;
      m_polish_backtracks           = 0;

      detail::project( m_x, x0, lower, upper );

      Real objective = evaluate_objective( problem, m_x );
      if ( !std::isfinite( objective ) ) return make_result( Status::non_finite_objective, objective );

      evaluate_gradient( problem, m_x, m_gradient );
      if ( !m_gradient.allFinite() ) return make_result( Status::non_finite_gradient, objective );

      // p(x) = x - P(x-g).
      // Unbounded box: P=I => p(x)=g(x), identically.
      projected_gradient( m_x, m_gradient, lower, upper, m_projected_gradient );

      Real projected_norm                = m_projected_gradient.norm();
      Real projected_norm_inf            = m_projected_gradient.template lpNorm<Eigen::Infinity>();
      Real H_estimate                    = std::max( options.hessian_lipschitz_initial, Real( 1 ) );
      Real H_previous                    = H_estimate;
      Real lambda                        = Real( 0 );
      Real last_step_norm                = Real( 0 );
      Real minimum_critical_eigenvalue   = std::numeric_limits<Real>::infinity();
      Real effective_curvature_tolerance = Real( 0 );
      Real cauchy_alpha                  = Real( 1 );
      int  iteration                     = 0;

      Real const initial_scale = std::max( Real( 1 ), projected_norm_inf );
      Real const tolerance     = std::max(
        Real( 64 ) * eps,
        std::min( options.absolute_tolerance, options.relative_tolerance * initial_scale ) );
      Real const automatic_polish_tolerance = Real( 128 ) * eps;
      Real const polish_tolerance           = options.polish_tolerance > Real( 0 )
                                                ? std::max( options.polish_tolerance, automatic_polish_tolerance )
                                                : automatic_polish_tolerance;
      Real const polish_entry               = std::max( options.polish_trigger, tolerance );

      Result<Real> result;
      auto         bound_violation = [&]() -> Real
      { return ( lower - m_x ).cwiseMax( m_x - upper ).cwiseMax( Real( 0 ) ).norm(); };

      auto fill = [&]( Status status ) -> Result<Real> &
      {
        result.x                           = m_x;
        result.objective                   = objective;
        result.projected_gradient_norm     = projected_norm;
        result.projected_gradient_norm_inf = projected_norm_inf;
        result.primal_feasibility          = bound_violation();
        result.step_norm                   = last_step_norm;
        result.lambda                      = lambda;
        result.hessian_lipschitz_estimate  = H_estimate;
        result.optimality_tolerance        = tolerance;
        result.minimum_critical_eigenvalue = minimum_critical_eigenvalue;
        result.effective_step_tolerance    = std::max(
          options.step_tolerance,
          Real( 64 ) * eps * std::max( Real( 1 ), m_x.norm() ) );
        result.effective_curvature_tolerance = effective_curvature_tolerance;
        result.iterations                    = iteration;
        result.function_evaluations          = m_function_evaluations;
        result.gradient_evaluations          = m_gradient_evaluations;
        result.hessian_evaluations           = m_hessian_evaluations;
        result.rejected_steps                = m_rejected_steps;
        result.fallback_steps                = m_fallback_steps;
        result.model_rejections              = m_model_rejections;
        result.gradient_rescue_steps         = m_gradient_rescue_steps;
        result.primary_semismooth_attempts   = m_primary_semismooth_attempts;
        result.primary_semismooth_steps      = m_primary_semismooth_steps;
        result.primary_semismooth_rejected   = m_primary_semismooth_rejected;
        result.semismooth_rescue_steps       = m_semismooth_rescue_steps;
        result.polish_iterations             = m_polish_iterations;
        result.polish_backtracks             = m_polish_backtracks;
        result.status                        = status;
        return result;
      };

      if ( !callback( fill( Status::unknown ) ) ) return fill( Status::user );

      enum class CertificateAction
      {
        certified,
        escaped,
        failed
      };

      // Certify second-order stationarity on the box critical cone, escape
      // along feasible negative curvature, or report that neither was safe.
      auto certify_or_escape = [&]() -> CertificateAction
      {
        evaluate_hessian( problem, m_x, m_H );
        if ( !m_H.allFinite() ) return CertificateAction::failed;

        std::vector<Eigen::Index> critical;
        std::vector<int>          cone_sign;  // +1: d>=0, -1: d<=0, 0: free
        critical.reserve( static_cast<std::size_t>( m_n ) );
        cone_sign.reserve( static_cast<std::size_t>( m_n ) );

        for ( Eigen::Index i = 0; i < m_n; ++i )
        {
          Real const scale = std::max(
            { Real( 1 ),
              std::abs( m_x[i] ),
              std::abs( m_gradient[i] ),
              std::isfinite( lower[i] ) ? std::abs( lower[i] ) : Real( 0 ),
              std::isfinite( upper[i] ) ? std::abs( upper[i] ) : Real( 0 ) } );
          Real const btol     = Real( 64 ) * eps * scale;
          bool const at_lower = std::isfinite( lower[i] ) && m_x[i] <= lower[i] + btol;
          bool const at_upper = std::isfinite( upper[i] ) && m_x[i] >= upper[i] - btol;

          // Strict complementarity removes the component from the critical
          // cone.  A weakly active component remains, with its feasible sign.
          bool const strongly_lower = at_lower && m_gradient[i] > tolerance;
          bool const strongly_upper = at_upper && m_gradient[i] < -tolerance;
          if ( strongly_lower || strongly_upper ) continue;

          critical.push_back( i );
          cone_sign.push_back( at_lower ? +1 : ( at_upper ? -1 : 0 ) );
        }

        if ( critical.empty() )
        {
          minimum_critical_eigenvalue   = std::numeric_limits<Real>::infinity();
          effective_curvature_tolerance = Real( 0 );
          last_step_norm                = Real( 0 );
          return CertificateAction::certified;
        }

        m_reduced_H = m_H( critical, critical );
        // Users occasionally return a Hessian with roundoff-level asymmetry.
        // The quadratic form depends only on its symmetric part.
        m_reduced_H                   = Real( 0.5 ) * ( m_reduced_H + m_reduced_H.transpose() ).eval();
        Real const hscale             = std::max( Real( 1 ), m_reduced_H.cwiseAbs().maxCoeff() );
        effective_curvature_tolerance = std::max(
          options.curvature_tolerance,
          Real( 256 ) * eps * hscale * Real( std::max<Eigen::Index>( 1, m_reduced_H.rows() ) ) );

        Eigen::SelfAdjointEigenSolver<Mat> eig( m_reduced_H );
        if ( eig.info() != Eigen::Success )
        {
          minimum_critical_eigenvalue = -std::numeric_limits<Real>::infinity();
          return CertificateAction::failed;
        }

        Eigen::Index const nc             = static_cast<Eigen::Index>( critical.size() );
        Vec                best_direction = Vec::Zero( nc );
        Real               best_curvature = std::numeric_limits<Real>::infinity();

        auto consider = [&]( Vec direction )
        {
          // Orthogonal projection onto the tangent/critical cone.
          for ( Eigen::Index k = 0; k < nc; ++k )
          {
            if ( cone_sign[static_cast<std::size_t>( k )] > 0 ) direction[k] = std::max( direction[k], Real( 0 ) );
            if ( cone_sign[static_cast<std::size_t>( k )] < 0 ) direction[k] = std::min( direction[k], Real( 0 ) );
          }
          Real const norm = direction.norm();
          if ( !( norm > Real( 0 ) ) ) return;
          direction /= norm;
          Real const curvature = direction.dot( m_reduced_H * direction );
          if ( curvature < best_curvature )
          {
            best_curvature = curvature;
            best_direction = direction;
          }
        };

        // Both orientations matter because one-sided weakly active variables
        // destroy the usual eigenvector sign symmetry.  Coordinate directions
        // additionally detect negative diagonal curvature on every cone face.
        for ( Eigen::Index j = 0; j < nc; ++j )
        {
          Vec v = eig.eigenvectors().col( j );
          consider( v );
          consider( -v );
        }
        for ( Eigen::Index j = 0; j < nc; ++j )
        {
          Vec e = Vec::Zero( nc );
          e[j]  = cone_sign[static_cast<std::size_t>( j )] < 0 ? Real( -1 ) : Real( 1 );
          consider( e );
        }

        minimum_critical_eigenvalue = best_curvature;
        if ( best_curvature >= -effective_curvature_tolerance )
        {
          // The final Newton correction required by the first-order mapping is
          // exactly zero.  Report that correction, not the previous outer step.
          last_step_norm = Real( 0 );
          return CertificateAction::certified;
        }

        // Lift the feasible negative-curvature direction to the full space and
        // try the farthest feasible point first.  This is important when a
        // stationary maximum lies on a weakly active bound: an infinitesimal
        // step is easily hidden by objective roundoff, whereas the box boundary
        // often gives the useful escape directly.
        m_direction.setZero();
        for ( Eigen::Index k = 0; k < nc; ++k )
          m_direction[critical[static_cast<std::size_t>( k )]] = best_direction[k];

        Real alpha_max = std::numeric_limits<Real>::infinity();
        for ( Eigen::Index i = 0; i < m_n; ++i )
        {
          if ( m_direction[i] > Real( 0 ) && std::isfinite( upper[i] ) )
            alpha_max = std::min( alpha_max, ( upper[i] - m_x[i] ) / m_direction[i] );
          else if ( m_direction[i] < Real( 0 ) && std::isfinite( lower[i] ) )
            alpha_max = std::min( alpha_max, ( lower[i] - m_x[i] ) / m_direction[i] );
        }
        Real alpha = std::isfinite( alpha_max ) ? alpha_max : Real( 1 );
        if ( !( alpha > Real( 0 ) ) ) return CertificateAction::failed;

        for ( int bt = 0; bt <= options.max_polish_backtracking; ++bt )
        {
          m_trial = m_x + alpha * m_direction;
          detail::project( m_trial, m_trial, lower, upper );
          m_step = m_trial - m_x;
          if ( m_step.norm() <= Real( 64 ) * eps * std::max( Real( 1 ), m_x.norm() ) ) break;

          Real const trial_objective = evaluate_objective( problem, m_trial );
          if ( std::isfinite( trial_objective ) && trial_objective < objective )
          {
            evaluate_gradient( problem, m_trial, m_trial_gradient );
            if ( !m_trial_gradient.allFinite() ) return CertificateAction::failed;
            projected_gradient( m_trial, m_trial_gradient, lower, upper, m_trial_projected_gradient );
            m_x                  = m_trial;
            m_gradient           = m_trial_gradient;
            m_projected_gradient = m_trial_projected_gradient;
            objective            = trial_objective;
            projected_norm       = m_projected_gradient.norm();
            projected_norm_inf   = m_projected_gradient.template lpNorm<Eigen::Infinity>();
            last_step_norm       = m_step.norm();
            lambda               = Real( 0 );
            return CertificateAction::escaped;
          }
          alpha *= Real( 0.5 );
        }
        return CertificateAction::failed;
      };

      for ( iteration = 1; iteration <= options.max_iterations; ++iteration )
      {
        // Same terminal structure as MinimizeNewtonCubic.  Only g is replaced by
        // p.  If P=I the call is pure Newton polishing on H d = g.
        if ( options.enable_polishing && projected_norm_inf <= polish_entry )
        {
          polish(
            problem,
            lower,
            upper,
            objective,
            projected_norm,
            projected_norm_inf,
            last_step_norm,
            polish_tolerance,
            -1 );
          if ( projected_norm_inf <= polish_tolerance )
          {
            CertificateAction const action = certify_or_escape();
            if ( action == CertificateAction::certified ) return fill( Status::converged );
            if ( action == CertificateAction::escaped ) continue;
            return fill( m_H.allFinite() ? Status::no_progress : Status::non_finite_hessian );
          }
        }

        if ( projected_norm_inf <= tolerance )
        {
          CertificateAction const action = certify_or_escape();
          if ( action == CertificateAction::certified ) return fill( Status::converged );
          if ( action == CertificateAction::escaped ) continue;
          return fill( m_H.allFinite() ? Status::no_progress : Status::non_finite_hessian );
        }

        if ( options.max_function_evaluations >= 0 && m_function_evaluations >= options.max_function_evaluations )
          return fill( Status::max_function_evaluations );

        Real const forcing_norm = projected_norm;
        if ( !std::isfinite( forcing_norm ) ) return fill( Status::non_finite_gradient );

        // TRON-like radius expansion translated to cubic regularization:
        // after a successful outer step, try a *smaller* M first.  The old
        // implementation multiplied by regularization_increase before its first
        // attempt, which effectively never reduced M and could make the method
        // unnecessarily conservative.
        Real const M_min = std::max( Real( 32 ) * eps, options.hessian_lipschitz_min );
        Real const M_max = std::max( M_min, options.hessian_lipschitz_max );
        H_estimate       = std::clamp( std::max( M_min, options.regularization_decrease * H_previous ), M_min, M_max );

        // x is fixed through the whole backtracking loop.
        evaluate_hessian( problem, m_x, m_H );
        if ( !m_H.allFinite() ) return fill( Status::non_finite_hessian );

        bool accepted   = false;
        bool have_trial = false;

        Vec  best_x( m_n ), best_gradient( m_n ), best_projected_gradient( m_n );
        Real best_objective = objective;
        Real best_step_norm = Real( 0 );
        Real best_lambda    = Real( 0 );
        Real best_H         = H_estimate;
        Real best_merit     = std::numeric_limits<Real>::infinity();

        // ------------------------------------------------------------------
        // Primary semismooth Newton candidate.
        //
        // p(x) = x-P(x-g),  J_p=(I-D)+DH.  We do not test positivity of J_p
        // directly because it is generally nonsymmetric on a mixed active/free
        // face.  Instead exploit its exact block-triangular structure: the
        // Newton system is safe when the symmetric free block H_FF is
        // numerically positive definite.  If all variables are free this is
        // exactly the ordinary Newton test/solve H d = g.
        // ------------------------------------------------------------------
        if ( options.enable_primary_semismooth )
        {
          ++m_primary_semismooth_attempts;

          m_free_indices.clear();
          m_direction.setZero();
          for ( Eigen::Index i = 0; i < m_n; ++i )
          {
            Real const y     = m_x[i] - m_gradient[i];
            Real const scale = std::max(
              { Real( 1 ),
                std::abs( m_x[i] ),
                std::abs( m_gradient[i] ),
                std::isfinite( lower[i] ) ? std::abs( lower[i] ) : Real( 0 ),
                std::isfinite( upper[i] ) ? std::abs( upper[i] ) : Real( 0 ) } );
            Real const dtol       = options.projection_derivative_tol * scale;
            bool const interior   = ( !std::isfinite( lower[i] ) || y > lower[i] + dtol ) &&
                                    ( !std::isfinite( upper[i] ) || y < upper[i] - dtol );
            m_projection_slope[i] = interior ? Real( 1 ) : Real( 0 );
            if ( interior )
              m_free_indices.push_back( i );
            else
              m_direction[i] = m_projected_gradient[i];
          }

          bool               primary_linear_ok = true;
          Eigen::Index const nfree             = static_cast<Eigen::Index>( m_free_indices.size() );
          if ( nfree > 0 )
          {
            m_reduced_H = m_H( m_free_indices, m_free_indices );
            Eigen::LDLT<Mat> ldlt( m_reduced_H );
            Real const       hscale = std::max( Real( 1 ), m_reduced_H.cwiseAbs().maxCoeff() );
            Real const pd_tol = options.primary_pd_factor * eps * hscale * Real( std::max<Eigen::Index>( 1, nfree ) );
            primary_linear_ok = ldlt.info() == Eigen::Success && ldlt.isPositive() && ldlt.vectorD().allFinite() &&
                                ldlt.vectorD().minCoeff() > pd_tol;

            if ( primary_linear_ok )
            {
              // Active rows of J_p are identities, hence d_A=p_A.  The free
              // equations are H_FF d_F = p_F - H_FA d_A.
              m_model_Hstep.noalias() = m_H * m_direction;
              m_reduced_rhs.resize( nfree );
              for ( Eigen::Index k = 0; k < nfree; ++k )
              {
                Eigen::Index const i = m_free_indices[static_cast<std::size_t>( k )];
                m_reduced_rhs[k]     = m_projected_gradient[i] - m_model_Hstep[i];
              }
              m_reduced_solution = ldlt.solve( m_reduced_rhs );
              primary_linear_ok  = ldlt.info() == Eigen::Success && m_reduced_solution.allFinite();
              if ( primary_linear_ok )
              {
                Real const rnorm  = ( m_reduced_H * m_reduced_solution - m_reduced_rhs ).norm();
                Real const rscale = Real( 1 ) + m_reduced_rhs.norm() + hscale * m_reduced_solution.norm();
                primary_linear_ok = rnorm <= Real( 512 ) * eps * rscale;
              }
              if ( primary_linear_ok )
                for ( Eigen::Index k = 0; k < nfree; ++k )
                  m_direction[m_free_indices[static_cast<std::size_t>( k )]] = m_reduced_solution[k];
            }
          }

          if ( primary_linear_ok && m_direction.allFinite() )
          {
            // SmallTRON-style projected model line-search: backtrack on the
            // frozen quadratic model only, then pay for one objective value.
            Real alpha           = Real( 1 );
            bool model_ok        = false;
            Real model_reduction = Real( 0 );
            Real model_slope     = Real( 0 );
            for ( int bt = 0; bt <= options.max_primary_model_backtracking; ++bt )
            {
              m_trial = m_x - alpha * m_direction;
              detail::project( m_trial, m_trial, lower, upper );
              m_step = m_trial - m_x;
              if ( m_step.squaredNorm() == Real( 0 ) ) break;

              m_model_Hstep.noalias() = m_H * m_step;
              model_slope             = m_gradient.dot( m_step );
              Real const qmodel       = model_slope + Real( 0.5 ) * m_step.dot( m_model_Hstep );
              model_reduction         = -qmodel;
              Real const mscale       = std::max( { Real( 1 ), std::abs( model_slope ), std::abs( qmodel ) } );
              Real const mround       = Real( 64 ) * eps * mscale;
              model_ok = model_slope < Real( 0 ) && qmodel <= options.model_armijo * model_slope + mround &&
                         model_reduction > Real( 0 );
              if ( model_ok ) break;
              alpha *= options.primary_model_shrink;
            }

            if ( model_ok )
            {
              if ( options.max_function_evaluations >= 0 && m_function_evaluations >= options.max_function_evaluations )
                return fill( Status::max_function_evaluations );

              Real const trial_objective = evaluate_objective( problem, m_trial );
              if ( std::isfinite( trial_objective ) )
              {
                Real const ro =
                  options.roundoff_factor * eps *
                  std::max(
                    { Real( 1 ), std::abs( objective ), std::abs( trial_objective ), std::abs( model_reduction ) } );
                Real const actual = objective - trial_objective;
                Real const ratio  = ( actual + ro ) / ( model_reduction + ro );

                if ( ratio >= options.primary_acceptance_threshold )
                {
                  evaluate_gradient( problem, m_trial, m_trial_gradient );
                  if ( !m_trial_gradient.allFinite() ) return fill( Status::non_finite_gradient );
                  projected_gradient( m_trial, m_trial_gradient, lower, upper, m_trial_projected_gradient );

                  m_x                  = m_trial;
                  m_gradient           = m_trial_gradient;
                  m_projected_gradient = m_trial_projected_gradient;
                  objective            = trial_objective;
                  projected_norm       = m_projected_gradient.norm();
                  projected_norm_inf   = m_projected_gradient.template lpNorm<Eigen::Infinity>();
                  last_step_norm       = m_step.norm();
                  lambda               = Real( 0 );
                  if ( ratio >= options.ratio_increase_threshold )
                    H_estimate = std::max( M_min, options.ratio_good_M_factor * H_estimate );
                  else if ( ratio < options.ratio_decrease_threshold )
                    H_estimate = std::min( M_max, options.ratio_bad_M_factor * H_estimate );
                  ++m_primary_semismooth_steps;
                  accepted = true;
                }
              }
            }
          }

          if ( !accepted ) ++m_primary_semismooth_rejected;
        }

        int const cubic_trial_limit = std::max(
          1,
          std::min( options.max_sub_iterations, options.cubic_trials_before_rescue ) );
        for ( int sub_iteration = 0; !accepted && sub_iteration < cubic_trial_limit; ++sub_iteration )
        {
          lambda = std::sqrt( H_estimate * forcing_norm );

          // (H + lambda I)d = p.
          // P=I => (H + lambda I)d = g, exactly MinimizeNewtonCubic.
          m_shifted_H = m_H;
          m_shifted_H.diagonal().array() += lambda;

          Eigen::FullPivLU<Mat> factorization( m_shifted_H );
          if ( !factorization.isInvertible() )
          {
            ++m_rejected_steps;
            H_estimate = std::min( M_max, options.factorization_increase * H_estimate );
            continue;
          }

          m_direction = factorization.solve( m_projected_gradient );
          if ( !m_direction.allFinite() )
          {
            ++m_rejected_steps;
            H_estimate = std::min( M_max, options.factorization_increase * H_estimate );
            continue;
          }

          // x1 = P(x-d).  Define s = x-x1.
          // P=I => x1=x-d and s=d, exactly.
          m_trial = m_x - m_direction;
          detail::project( m_trial, m_trial, lower, upper );
          m_step = m_x - m_trial;

          // Frozen quadratic model q(z), z=x_trial-x=-m_step.  Compute this
          // unconditionally: besides the cheap TRON prescreen it provides the
          // predicted reduction used to adapt M after a successful trial.
          m_model_Hstep.noalias()    = m_H * m_step;
          Real const model_slope     = -m_gradient.dot( m_step );
          Real const qmodel          = model_slope + Real( 0.5 ) * m_step.dot( m_model_Hstep );
          Real const model_reduction = -qmodel;
          if ( options.enable_model_safeguard )
          {
            Real const model_scale    = std::max( { Real( 1 ), std::abs( model_slope ), std::abs( qmodel ) } );
            Real const model_roundoff = Real( 64 ) * eps * model_scale;
            bool const model_ok       = model_slope < Real( 0 ) &&
                                        qmodel <= options.model_armijo * model_slope + model_roundoff;
            if ( !model_ok )
            {
              ++m_rejected_steps;
              ++m_model_rejections;
              H_estimate = std::min( M_max, options.regularization_increase * H_estimate );
              continue;
            }
          }

          if ( options.max_function_evaluations >= 0 && m_function_evaluations >= options.max_function_evaluations )
            return fill( Status::max_function_evaluations );

          Real const trial_objective = evaluate_objective( problem, m_trial );
          if ( !std::isfinite( trial_objective ) )
          {
            ++m_rejected_steps;
            H_estimate = std::min( M_max, options.factorization_increase * H_estimate );
            continue;
          }

          // An objective ascent cannot pass the sufficient-decrease test and is
          // not eligible for the safe fallback.  Reject it before evaluating g.
          Real const objective_roundoff_early = options.roundoff_factor * eps *
                                                std::max(
                                                  { Real( 1 ), std::abs( objective ), std::abs( trial_objective ) } );
          if ( trial_objective > objective + objective_roundoff_early )
          {
            ++m_rejected_steps;
            H_estimate = std::min( M_max, options.regularization_increase * H_estimate );
            continue;
          }

          evaluate_gradient( problem, m_trial, m_trial_gradient );
          if ( !m_trial_gradient.allFinite() ) return fill( Status::non_finite_gradient );

          projected_gradient( m_trial, m_trial_gradient, lower, upper, m_trial_projected_gradient );

          // Same AdaN tests, with g -> p and d -> s.
          // For P=I: p1=g1 and s=d, so the formulas are unchanged.
          Real const np1       = m_trial_projected_gradient.norm();
          Real const rp        = m_step.norm();
          Real const lr        = lambda * rp;
          Real const predicted = options.decrease_factor * lr * rp;
          Real const roundoff  = options.roundoff_factor * eps *
                                 std::max( { Real( 1 ), std::abs( objective ), std::abs( trial_objective ) } );

          bool const gradient_ok      = np1 <= options.gradient_acceptance_factor * lr;
          bool const decrease_ok      = trial_objective <= objective - predicted + roundoff;
          Real const actual_reduction = objective - trial_objective;
          Real const ratio_roundoff =
            options.roundoff_factor * eps *
            std::max( { Real( 1 ), std::abs( objective ), std::abs( trial_objective ), std::abs( model_reduction ) } );
          Real const agreement_ratio = model_reduction > Real( 0 )
                                         ? ( actual_reduction + ratio_roundoff ) / ( model_reduction + ratio_roundoff )
                                         : -std::numeric_limits<Real>::infinity();

          Real const tiny               = std::numeric_limits<Real>::min();
          Real const grad_scale         = std::max( options.gradient_acceptance_factor * lr, tiny );
          Real const grad_ratio         = np1 / grad_scale;
          Real const decrease_violation = std::max( Real( 0 ), trial_objective - ( objective - predicted + roundoff ) );
          Real const decrease_scale     = std::max( predicted + roundoff, tiny );
          Real const decrease_ratio     = Real( 1 ) + decrease_violation / decrease_scale;
          Real const merit              = std::max( grad_ratio, decrease_ratio );

          if ( !have_trial || merit < best_merit || ( merit == best_merit && trial_objective < best_objective ) )
          {
            best_x                  = m_trial;
            best_gradient           = m_trial_gradient;
            best_projected_gradient = m_trial_projected_gradient;
            best_objective          = trial_objective;
            best_step_norm          = rp;
            best_lambda             = lambda;
            best_H                  = H_estimate;
            best_merit              = merit;
            have_trial              = true;
          }

          if ( options.verbose > 1 )
            std::printf(
              "  sub=%2d f_trial=%14.7e p_trial=%10.3e step=%10.3e "
              "lambda=%10.3e H=%10.3e merit=%9.3e decrease=%d gradient=%d\n",
              sub_iteration,
              double( trial_objective ),
              double( np1 ),
              double( rp ),
              double( lambda ),
              double( H_estimate ),
              double( merit ),
              int( decrease_ok ),
              int( gradient_ok ) );

          if ( gradient_ok && decrease_ok )
          {
            m_x                  = m_trial;
            m_gradient           = m_trial_gradient;
            m_projected_gradient = m_trial_projected_gradient;
            objective            = trial_objective;
            projected_norm       = np1;
            projected_norm_inf   = m_projected_gradient.template lpNorm<Eigen::Infinity>();
            last_step_norm       = rp;
            if ( agreement_ratio >= options.ratio_increase_threshold )
              H_estimate = std::max( M_min, options.ratio_good_M_factor * H_estimate );
            else if ( agreement_ratio < options.ratio_decrease_threshold )
              H_estimate = std::min( M_max, options.ratio_bad_M_factor * H_estimate );
            accepted = true;
            break;
          }

          ++m_rejected_steps;
          H_estimate = std::min( M_max, options.regularization_increase * H_estimate );
        }

        // First rescue: semismooth Newton on the projected KKT mapping.  This
        // reuses exactly the same J_p=(I-D)+DH formula as terminal polishing,
        // but for only a few iterations.  On an unbounded box it is pure Newton.
        bool rescued = false;
        if ( !accepted && options.enable_semismooth_rescue )
        {
          Real const norm_before   = projected_norm_inf;
          int const  polish_before = m_polish_iterations;
          polish(
            problem,
            lower,
            upper,
            objective,
            projected_norm,
            projected_norm_inf,
            last_step_norm,
            polish_tolerance,
            options.semismooth_rescue_iterations );
          if ( projected_norm_inf < norm_before )
          {
            m_semismooth_rescue_steps += m_polish_iterations - polish_before;
            lambda  = Real( 0 );
            rescued = true;
          }
        }

        // Second rescue: SmallTRON-style projected Cauchy search.  Search for an
        // alpha satisfying the quadratic-model condition using H only, including
        // geometric extrapolation.  Only the selected alpha is tested on f.
        if ( !accepted && !rescued && options.enable_gradient_rescue )
        {
          Real const cauchy_radius   = options.cauchy_radius_factor *
                                       std::sqrt(
                                         std::max( forcing_norm, Real( 64 ) * eps ) / std::max( M_min, H_previous ) );
          auto       cauchy_model_ok = [&]( Real alpha ) -> bool
          {
            m_trial = m_x - alpha * m_gradient;
            detail::project( m_trial, m_trial, lower, upper );
            m_step = m_trial - m_x;
            if ( m_step.squaredNorm() == Real( 0 ) ) return false;
            if ( std::isfinite( cauchy_radius ) && m_step.norm() > cauchy_radius ) return false;
            m_model_Hstep.noalias() = m_H * m_step;
            Real const slope        = m_gradient.dot( m_step );
            Real const q            = slope + Real( 0.5 ) * m_step.dot( m_model_Hstep );
            Real const scale        = std::max( { Real( 1 ), std::abs( slope ), std::abs( q ) } );
            Real const ro           = Real( 64 ) * eps * scale;
            return slope < Real( 0 ) && q <= options.model_armijo * slope + ro;
          };

          Real alpha    = std::clamp( cauchy_alpha, options.cauchy_alpha_min, options.cauchy_alpha_max );
          bool model_ok = cauchy_model_ok( alpha );
          for ( int k = 0; !model_ok && k < options.max_gradient_rescue; ++k )
          {
            alpha *= options.gradient_rescue_shrink;
            if ( alpha < options.cauchy_alpha_min ) break;
            model_ok = cauchy_model_ok( alpha );
          }

          if ( model_ok )
          {
            Real alpha_ok = alpha;
            // TRON extrapolation: enlarge alpha while the frozen model continues
            // to satisfy sufficient decrease.  No f/g calls occur here.
            for ( int k = 0; k < options.max_gradient_rescue; ++k )
            {
              Real const next = alpha_ok * options.cauchy_expand;
              if ( !( next > alpha_ok ) || next > options.cauchy_alpha_max ) break;
              if ( !cauchy_model_ok( next ) ) break;
              alpha_ok = next;
            }
            alpha        = alpha_ok;
            cauchy_alpha = alpha_ok;
            cauchy_model_ok( alpha );  // restore selected trial/step

            Real const slope = m_gradient.dot( m_step );
            if ( options.max_function_evaluations >= 0 && m_function_evaluations >= options.max_function_evaluations )
              return fill( Status::max_function_evaluations );

            Real const trial_objective = evaluate_objective( problem, m_trial );
            if ( std::isfinite( trial_objective ) )
            {
              Real const roundoff = options.roundoff_factor * eps *
                                    std::max( { Real( 1 ), std::abs( objective ), std::abs( trial_objective ) } );
              if ( trial_objective <= objective + options.gradient_rescue_armijo * slope + roundoff )
              {
                evaluate_gradient( problem, m_trial, m_trial_gradient );
                if ( !m_trial_gradient.allFinite() ) return fill( Status::non_finite_gradient );
                projected_gradient( m_trial, m_trial_gradient, lower, upper, m_trial_projected_gradient );
                m_x                  = m_trial;
                m_gradient           = m_trial_gradient;
                m_projected_gradient = m_trial_projected_gradient;
                objective            = trial_objective;
                projected_norm       = m_projected_gradient.norm();
                projected_norm_inf   = m_projected_gradient.template lpNorm<Eigen::Infinity>();
                last_step_norm       = m_step.norm();
                lambda               = Real( 0 );
                ++m_gradient_rescue_steps;
                rescued = true;
              }
            }
          }
        }

        // Safe finite-search fallback.  It may only be used if it is not an
        // objective ascent (within roundoff); otherwise report no progress.
        if ( !accepted && !rescued )
        {
          if ( !have_trial ) return fill( Status::no_progress );
          Real const fallback_roundoff = options.roundoff_factor * eps *
                                         std::max( { Real( 1 ), std::abs( objective ), std::abs( best_objective ) } );
          bool const fallback_safe     = best_objective <= objective + fallback_roundoff &&
                                         best_projected_gradient.norm() < projected_norm;
          if ( !fallback_safe ) return fill( Status::no_progress );

          ++m_fallback_steps;
          m_x                  = best_x;
          m_gradient           = best_gradient;
          m_projected_gradient = best_projected_gradient;
          objective            = best_objective;
          projected_norm       = m_projected_gradient.norm();
          projected_norm_inf   = m_projected_gradient.template lpNorm<Eigen::Infinity>();
          last_step_norm       = best_step_norm;
          lambda               = best_lambda;
          H_estimate           = best_H;
        }

        H_previous = H_estimate;

        if ( options.verbose > 0 && iteration % options.verbose == 0 )
          std::printf(
            "%6d f=%14.7e p=%10.3e step=%10.3e lambda=%10.3e H=%10.3e\n",
            iteration,
            double( objective ),
            double( projected_norm_inf ),
            double( last_step_norm ),
            double( lambda ),
            double( H_estimate ) );

        if ( !callback( fill( Status::unknown ) ) ) return fill( Status::user );
      }

      if ( projected_norm_inf <= tolerance )
      {
        CertificateAction const action = certify_or_escape();
        if ( action == CertificateAction::certified ) return fill( Status::converged );
      }
      return fill( Status::max_iterations );
    }

  private:
    template <typename Problem>
    /**
     * @brief High-accuracy semismooth Newton polishing of the projected KKT map.
     *
     * @details
     * Builds one generalized Jacobian
     * @f$J_p=(I-D)+DH@f$, solves @f$J_p d=p@f$, and backtracks the projected
     * trial @f$P(x-\alpha d)@f$ until the projected residual contracts.  The
     * routine is also reused as a short rescue step when the cubic search stalls.
     * For an unbounded box, @f$D=I@f$ and the routine becomes ordinary damped
     * Newton polishing on @f$Hd=g@f$.
     */
    void polish(
      Problem &            problem,
      ConstVectorRef<Real> lower,
      ConstVectorRef<Real> upper,
      Real &               objective,
      Real &               projected_norm,
      Real &               projected_norm_inf,
      Real &               last_step_norm,
      Real                 tolerance,
      int                  max_iterations_override )
    {
      constexpr Real eps     = std::numeric_limits<Real>::epsilon();
      auto const &   options = m_options;

      int const max_pit = max_iterations_override > 0 ? max_iterations_override : options.max_polish_iterations;
      for ( int pit = 0; pit < max_pit && projected_norm_inf > tolerance; ++pit )
      {
        evaluate_hessian( problem, m_x, m_H );
        if ( !m_H.allFinite() ) return;

        // p(x)=x-P(y), y=x-g(x).
        // Let D be one generalized derivative of P at y. Then
        //
        //        J_p = I - D(I-H) = (I-D) + D H.
        //
        // This is the key smooth-transition formula.  For an unbounded box
        // D=I, hence J_p=H exactly, with NO algorithmic branch.
        for ( Eigen::Index i = 0; i < m_n; ++i )
        {
          Real const y     = m_x[i] - m_gradient[i];
          Real const scale = std::max(
            { Real( 1 ),
              std::abs( m_x[i] ),
              std::abs( m_gradient[i] ),
              std::isfinite( lower[i] ) ? std::abs( lower[i] ) : Real( 0 ),
              std::isfinite( upper[i] ) ? std::abs( upper[i] ) : Real( 0 ) } );
          Real const tol = options.projection_derivative_tol * scale;

          // D_ii=1 on the interior, D_ii=0 on a clamped branch.  At the kink
          // either value belongs to the Clarke generalized Jacobian; choosing
          // the clamped value is the stable semismooth-Newton convention.
          bool const interior   = ( !std::isfinite( lower[i] ) || y > lower[i] + tol ) &&
                                  ( !std::isfinite( upper[i] ) || y < upper[i] - tol );
          m_projection_slope[i] = interior ? Real( 1 ) : Real( 0 );
        }

        m_polish_J = m_projection_slope.asDiagonal() * m_H;
        m_polish_J.diagonal().array() += ( Real( 1 ) - m_projection_slope.array() );

        Eigen::FullPivLU<Mat> factorization( m_polish_J );
        if ( !factorization.isInvertible() ) return;

        m_direction = factorization.solve( m_projected_gradient );
        if ( !m_direction.allFinite() ) return;

        Real const step_norm = m_direction.norm();
        Real const x_scale   = std::max( Real( 1 ), m_x.norm() );
        if ( step_norm <= Real( 32 ) * eps * x_scale ) return;

        Real alpha          = Real( 1 );
        bool accepted       = false;
        Real best_norm_inf  = projected_norm_inf;
        Real best_f         = objective;
        Real best_step_norm = Real( 0 );
        Vec  best_x( m_n ), best_g( m_n ), best_p( m_n );
        bool have_best = false;

        for ( int bt = 0; bt <= options.max_polish_backtracking; ++bt )
        {
          // Same Newton trial formula, wrapped only by P.
          // P=I => x1=x-alpha*d exactly as in MinimizeNewtonCubic.
          m_trial = m_x - alpha * m_direction;
          detail::project( m_trial, m_trial, lower, upper );
          m_step = m_x - m_trial;

          if ( m_step.norm() <= Real( 32 ) * eps * x_scale ) break;

          if ( options.max_function_evaluations >= 0 && m_function_evaluations >= options.max_function_evaluations )
            return;

          Real const trial_objective = evaluate_objective( problem, m_trial );
          if ( !std::isfinite( trial_objective ) )
          {
            alpha *= Real( 0.5 );
            ++m_polish_backtracks;
            continue;
          }

          evaluate_gradient( problem, m_trial, m_trial_gradient );
          if ( !m_trial_gradient.allFinite() )
          {
            alpha *= Real( 0.5 );
            ++m_polish_backtracks;
            continue;
          }

          projected_gradient( m_trial, m_trial_gradient, lower, upper, m_trial_projected_gradient );
          Real const trial_norm_inf = m_trial_projected_gradient.template lpNorm<Eigen::Infinity>();
          Real const roundoff       = options.roundoff_factor * eps *
                                      std::max( { Real( 1 ), std::abs( objective ), std::abs( trial_objective ) } );

          if (
            !have_best || trial_norm_inf < best_norm_inf ||
            ( trial_norm_inf == best_norm_inf && trial_objective < best_f ) )
          {
            best_x         = m_trial;
            best_g         = m_trial_gradient;
            best_p         = m_trial_projected_gradient;
            best_f         = trial_objective;
            best_norm_inf  = trial_norm_inf;
            best_step_norm = m_step.norm();
            have_best      = true;
          }

          // Identical polishing test after the substitutions g -> p and
          // x-alpha*d -> P(x-alpha*d).  For P=I this is literally the same test.
          bool const residual_ok  = trial_norm_inf <=
                                      ( Real( 1 ) - options.polish_armijo * alpha ) * projected_norm_inf ||
                                    trial_norm_inf <= tolerance;
          bool const objective_ok = trial_objective <= objective + roundoff;

          if ( residual_ok && objective_ok )
          {
            accepted             = true;
            m_x                  = m_trial;
            m_gradient           = m_trial_gradient;
            m_projected_gradient = m_trial_projected_gradient;
            objective            = trial_objective;
            projected_norm       = m_projected_gradient.norm();
            projected_norm_inf   = trial_norm_inf;
            last_step_norm       = m_step.norm();
            ++m_polish_iterations;
            break;
          }

          alpha *= Real( 0.5 );
          ++m_polish_backtracks;
        }

        if ( !accepted )
        {
          if ( !have_best || !( best_norm_inf < projected_norm_inf ) ) return;
          m_x                  = best_x;
          m_gradient           = best_g;
          m_projected_gradient = best_p;
          objective            = best_f;
          projected_norm       = m_projected_gradient.norm();
          projected_norm_inf   = best_norm_inf;
          last_step_norm       = best_step_norm;
          ++m_polish_iterations;
        }
      }
    }

    /**
     * @brief Evaluate @f$f(x)@f$ and update the objective-evaluation counter.
     * @tparam Problem Problem type satisfying ProblemFor.
     * @param problem Optimization problem.
     * @param x Point at which the objective is evaluated.
     * @return Objective value.
     */
    template <typename Problem> Real evaluate_objective( Problem & problem, Vec const & x )
    {
      ++m_function_evaluations;
      return problem.objective( x );
    }

    /**
     * @brief Evaluate @f$g(x)=\nabla f(x)@f$ and update the gradient counter.
     * @param problem Optimization problem.
     * @param x Evaluation point.
     * @param gradient Output gradient vector.
     */
    template <typename Problem> void evaluate_gradient( Problem & problem, Vec const & x, Vec & gradient )
    {
      ++m_gradient_evaluations;
      problem.gradient( x, gradient );
    }

    /**
     * @brief Evaluate @f$H(x)=\nabla^2 f(x)@f$ and update the Hessian counter.
     * @param problem Optimization problem.
     * @param x Evaluation point.
     * @param H Output dense Hessian matrix.
     */
    template <typename Problem> void evaluate_hessian( Problem & problem, Vec const & x, Mat & H )
    {
      ++m_hessian_evaluations;
      problem.hessian( x, H );
    }

    /**
     * @brief Compute the projected first-order stationarity mapping.
     *
     * @param x Current point.
     * @param gradient Ordinary gradient @f$g(x)@f$.
     * @param lower Lower bounds.
     * @param upper Upper bounds.
     * @param projected Output vector @f$p(x)=x-P_{[\ell,u]}(x-g(x))@f$.
     *
     * @details
     * The sign convention is deliberate.  If all bounds are infinite then
     * @f$P(z)=z@f$ and this routine returns @f$p(x)=g(x)@f$ exactly.
     */
    //
    //   p(x) = x - P_[l,u](x-g(x)).
    //
    // No special treatment of the unconstrained case is required:
    // [-inf,+inf]^n => P=I => p(x)=g(x).
    static void projected_gradient(
      Vec const & x,
      Vec const & gradient,
      Vec const & lower,
      Vec const & upper,
      Vec &       projected )
    { projected = x - ( x - gradient ).cwiseMax( lower ).cwiseMin( upper ); }

    /**
     * @brief Construct a minimal Result object for early exits occurring before
     *        the main iteration state has been fully initialized.
     */
    Result<Real> make_result( Status status, Real objective ) const
    {
      Result<Real> result;
      result.x                    = m_x;
      result.objective            = objective;
      result.function_evaluations = m_function_evaluations;
      result.gradient_evaluations = m_gradient_evaluations;
      result.hessian_evaluations  = m_hessian_evaluations;
      result.status               = status;
      return result;
    }

    Options<Real> m_options{};
    Eigen::Index  m_n = 0;

    Vec                       m_x, m_trial;
    Vec                       m_gradient, m_trial_gradient;
    Vec                       m_projected_gradient, m_trial_projected_gradient;
    Vec                       m_direction, m_step, m_model_Hstep, m_projection_slope;
    Vec                       m_reduced_rhs, m_reduced_solution;
    Mat                       m_H, m_shifted_H, m_polish_J, m_reduced_H;
    std::vector<Eigen::Index> m_free_indices;

    int m_function_evaluations        = 0;
    int m_gradient_evaluations        = 0;
    int m_hessian_evaluations         = 0;
    int m_rejected_steps              = 0;
    int m_fallback_steps              = 0;
    int m_model_rejections            = 0;
    int m_gradient_rescue_steps       = 0;
    int m_primary_semismooth_attempts = 0;
    int m_primary_semismooth_steps    = 0;
    int m_primary_semismooth_rejected = 0;
    int m_semismooth_rescue_steps     = 0;
    int m_polish_iterations           = 0;
    int m_polish_backtracks           = 0;
  };

  template <typename Real = double, typename Problem>
    requires ProblemFor<Problem, Real>
  Result<Real> minimize(
    Problem &            problem,
    ConstVectorRef<Real> x0,
    ConstVectorRef<Real> lower,
    ConstVectorRef<Real> upper,
    Options<Real>        options = {} )
  {
    Solver<Real> solver( x0.size(), options );
    return solver.solve( problem, x0, lower, upper );
  }

}  // namespace Utils::MinimizeNewton

#endif
