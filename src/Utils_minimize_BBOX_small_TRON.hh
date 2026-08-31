/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2026                                                      |
 |                                                                          |
 |         , __                 , __                                        |
 |        /|/  \               /|/  \                                       |
 |         | __/ _   ,_         | __/ _   ,_                                |
 |         |   \|/  /  |  |   | |   \|/  /  |  |   |                        |
 |         |(__/|__/   |_/ \_/|/|(__/|__/   |_/ \_/|/                       |
 |                           /|                   /|                        |
 |                           \|                   \|                        |
 |                                                                          |
 |      Enrico Bertolazzi                                                   |
 |      Dipartimento di Ingegneria Industriale                              |
 |      Università degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

// Utils_minimize_BBOX_small_TRON.hh -- header-only C++20 port of the TRON trust-region solver
// for bound-constrained minimization:
//
//     min f(x)   s.t.   l <= x <= u
//
// Reference:
//   Chih-Jen Lin and Jorge J. More, "Newton's Method for Large
//   Bound-Constrained Optimization Problems", SIAM J. Optim., 9(4),
//   1100-1127, 1999. DOI: 10.1137/S1052623498345075
//
// Ported from the Julia implementation in JSOSolvers.jl (tron.jl), including
// the TRONTrustRegion update rules of SolverTools.jl. The free-variable
// trust-region subproblem is solved directly with Eigen 5.
//
// This revision targets small/medium dense problems: the problem supplies the
// full Hessian matrix H(x) rather than Hessian-vector products. Since the
// projected-Newton phase already assembles the reduced free-variable Hessian
// H(F,F) and solves it directly with a dense eigendecomposition, an
// explicit-Hessian interface removes the redundant matrix-free wrapper
// (reduced columns were previously reconstructed one canonical direction at a
// time) without changing the algorithm. For genuinely large-scale problems
// where forming H is impractical, a matrix-free/hprod variant remains the
// better choice.
//
// Dependencies: Eigen (Dense/Eigenvalues) and the C++20 standard library.

#pragma once

#ifndef UTILS_MINIMIZE_BBOX_SMALL_TRON_DOT_HH
#define UTILS_MINIMIZE_BBOX_SMALL_TRON_DOT_HH

#include "Utils_eigen.hh"
#include "Utils_minimize_BBOX_Common.hh"

#if EIGEN_MAJOR_VERSION < 5
#error "Utils::Minimize_BBOX_SmallTRON requires Eigen 5 or newer"
#endif

namespace Utils::SmallTRON_details
{

  /**
   * \brief Dynamic dense vector type used throughout the TRON implementation.
   *
   * Second-order information is supplied through Hessian-vector products. The
   * solver assembles only the reduced Hessian associated with the variables
   * currently free from their bounds.
   *
   * \tparam Real Scalar floating-point type.
   */
  // Reuse common dense types (Ref/Map zero-copy)
  template <typename Real> using Vector         = Utils::Vector<Real>;
  template <typename Real> using Matrix         = Utils::Matrix<Real>;
  template <typename Real> using ConstVectorRef = Utils::ConstVectorRef<Real>;
  template <typename Real> using VectorRef      = Utils::VectorRef<Real>;
  template <typename Real> using ConstMatrixRef = ::Utils::ConstMatrixRef<Real>;
  template <typename Real> using MatrixRef      = Utils::MatrixRef<Real>;

  // ---------------------------------------------------------------------------
  // Problem interface
  // ---------------------------------------------------------------------------

  /**
   * \brief Compile-time interface required by the TRON solver.
   *
   * A model satisfying this concept provides the objective value, the full
   * gradient, and the full (dense) Hessian matrix.  This is the natural
   * interface for small/medium problems with an explicitly available,
   * symmetric Hessian: the projected-Newton phase solves its free-variable
   * subproblem directly (dense eigendecomposition), so nothing is gained by
   * hiding the Hessian behind a matrix-free operator.
   *
   * The required operations are
   * \f[
   *   f(x), \qquad g(x)=\nabla f(x), \qquad H(x)=\nabla^2 f(x).
   * \f]
   *
   * \tparam Problem User problem type.
   * \tparam Real Scalar floating-point type.
   *
   * \note No dimension checks are imposed by the concept.  The problem methods
   *       must produce a vector/matrix compatible with the input dimension.
   * \note `H(x)` need not be supplied exactly symmetric: the solver
   *       symmetrizes it once, right after evaluation (see
   *       `Minimize_BBOX_SmallTRON::eval_hessian`), and relies on that single symmetrization
   *       point everywhere downstream.
   */
  template <typename Problem, typename Real>
  concept ProblemFor = requires( Problem & p, ConstVectorRef<Real> x, VectorRef<Real> g, MatrixRef<Real> H, Real & f ) {
    { p.objective( x, f ) } -> std::convertible_to<bool>;
    { p.gradient( x, g ) } -> std::convertible_to<bool>;
    { p.hessian( x, H ) } -> std::convertible_to<bool>;
  };

  /**
   * \brief Adapter that turns three independent callables into a TRON problem.
   *
   * This utility avoids the need to define a dedicated problem class.  The three
   * callables are stored by value and invoked by \ref objective, \ref gradient,
   * and \ref hessian.
   *
   * \tparam Real Scalar floating-point type.
   * \tparam Obj Callable implementing \f$f(x)\f$.
   * \tparam Grad Callable implementing \f$g=\nabla f(x)\f$.
   * \tparam Hess Callable implementing \f$H=\nabla^2 f(x)\f$.
   */
  template <typename Real, typename Obj, typename Grad, typename Hess> class CallableProblem
  {
  public:
    /**
     * \brief Construct the callable adapter.
     * \param obj Objective callable returning bool.
     * \param grad Gradient callable returning bool.
     * \param hess Hessian callable returning bool.
     */
    CallableProblem( Obj obj, Grad grad, Hess hess )
      : obj_( std::move( obj ) ), grad_( std::move( grad ) ), hess_( std::move( hess ) )
    {
    }

    /** \brief Evaluate the objective function at \p x. */
    bool objective( ConstVectorRef<Real> x, Real & f ) { return static_cast<bool>( obj_( x, f ) ); }

    /** \brief Evaluate the objective gradient at \p x. */
    bool gradient( ConstVectorRef<Real> x, VectorRef<Real> g ) { return static_cast<bool>( grad_( x, g ) ); }

    /** \brief Evaluate the full Hessian matrix \f$H(x)\f$. */
    bool hessian( ConstVectorRef<Real> x, MatrixRef<Real> H ) { return static_cast<bool>( hess_( x, H ) ); }

  private:
    Obj  obj_;
    Grad grad_;
    Hess hess_;
  };

  /**
   * \brief Deduce callable types and construct a \ref CallableProblem.
   * \tparam Real Scalar type used by the solver.
   * \param obj Objective callable.
   * \param grad Gradient callable.
   * \param hess Hessian callable.
   * \return A callable problem object satisfying \ref ProblemFor.
   */
  // make_problem moved to Utils::make_problem in Common.hh

  // ---------------------------------------------------------------------------
  // Status and options
  // ---------------------------------------------------------------------------

  /** \brief Termination status of the outer TRON iteration. */
  enum class Status
  {
    unknown,
    first_order,  // projected KKT residual, step, and critical curvature certified
    unbounded,    // objective seems unbounded below
    max_iter,
    max_eval,
    small_step,  // Cauchy point step underflowed
    neg_pred,    // non-negative predicted reduction
    direct_solver_failure,
    user_request
  };

  /**
   * \brief Convert a solver termination status to a human-readable string.
   * \param s Status code.
   * \return Static string describing \p s.
   */
  [[nodiscard]] constexpr std::string_view to_string( Status s ) noexcept
  {
    switch ( s )
    {
      case Status::unknown: return "unknown";
      case Status::first_order: return "certified constrained minimum";
      case Status::unbounded: return "unbounded";
      case Status::max_iter: return "maximum iterations";
      case Status::max_eval: return "maximum evaluations";
      case Status::small_step: return "small step";
      case Status::neg_pred: return "non-negative predicted reduction";
      case Status::direct_solver_failure: return "direct trust-region solver failure";
      case Status::user_request: return "user request";
    }
    return "unknown";
  }

  /**
   * \brief Termination status of the projected-Newton direct solve.
   */
  enum class NewtonStatus
  {
    unknown,
    stationary,
    boundary,
    max_iter,
    direct_solver_failure
  };

  /**
   * \brief Convert an inner projected-Newton status to a human-readable string.
   * \param s Inner iteration status.
   * \return Static string describing \p s.
   */
  [[nodiscard]] constexpr std::string_view to_string( NewtonStatus s ) noexcept
  {
    switch ( s )
    {
      case NewtonStatus::unknown: return "unknown";
      case NewtonStatus::stationary: return "stationary point found";
      case NewtonStatus::boundary: return "on trust-region boundary";
      case NewtonStatus::max_iter: return "maximum number of iterations";
      case NewtonStatus::direct_solver_failure: return "direct eigensolver failure";
    }
    return "unknown";
  }

  /**
   * \brief Numerical parameters, tolerances, and resource limits for TRON.
   *
   * The defaults follow the constants used by the referenced TRON trust-region
   * implementations.  The principal outer optimality test is based on the norm
   * of the projected-gradient mapping
   * \f[
   *   \pi(x)=\|P_{[l,u]}(x-g(x))-x\|_2.
   * \f]
   * Convergence is declared only when this quantity is no larger than the
   * configured general tolerance, the last trial step is small, the bound
   * violation is at machine-precision scale, and the Hessian has no significant
   * negative curvature on the critical free subspace.
   *
   * \tparam Real Scalar floating-point type.
   */
  template <typename Real = double> struct Options
  {
    static constexpr Real eps = std::numeric_limits<Real>::epsilon();

    // Algorithm parameters (see Lin & More).
    Real mu_0  = Real( 1 ) / Real( 100 );  // in (0, 1/2): sufficient decrease
    Real mu_1  = Real( 1 );                // in (0, inf): Cauchy step / radius ratio
    Real sigma = Real( 10 );               // in (1, inf): inter/extrapolation factor

    // Tolerances: stop when ||x - P(x - g)|| <= atol + rtol * ||x0 - P(x0 -
    // g0)||. Unified to 1e-9 in inf-norm for all BBOX solvers.
    Real atol = Real( 1e-9 );
    Real rtol = Real( 1e-9 );
    // A non-positive value selects the scale-aware machine-precision floor.
    // Positive values are absolute user overrides, never allowed below that
    // floor.  Keeping these tests independent of the first-order tolerance
    // forces one final, genuinely negligible Newton step and applies the
    // second-order test at the backward-error level of the eigensolver.
    Real step_tolerance             = Real( 0 );
    Real curvature_tolerance        = Real( 0 );
    Real projected_newton_tolerance = Real( 1 ) / Real( 10 );

    // Budgets.
    int max_iterations                  = 1000;
    int max_evaluations                 = -1;  // objective evaluations, -1 = unlimited
    int max_projected_newton_iterations = 50;

    // Trust region (TRONTrustRegion defaults).
    Real max_radius            = std::min( Real( 1 ) / std::sqrt( 2 * eps ), Real( 100 ) );
    Real acceptance_threshold  = Real( 1 ) / Real( 10000 );
    Real decrease_threshold    = Real( 1 ) / Real( 4 );
    Real increase_threshold    = Real( 3 ) / Real( 4 );
    Real large_decrease_factor = Real( 1 ) / Real( 4 );
    Real small_decrease_factor = Real( 1 ) / Real( 2 );
    Real increase_factor       = Real( 4 );

    // Optional cubic-regularization guidance.  The original TRON quadratic
    // model is unchanged: this only caps its trust-region radius by
    //
    //   Delta_cub = cubic_radius_factor * sqrt( ||pi(x)|| / M ),
    //
    // where M estimates the local Lipschitz constant of the Hessian from
    // consecutive explicit Hessians.  Thus no second diagonal shift is added
    // to the one already generated by the trust-region subproblem.
    bool use_cubic_radius       = true;
    Real cubic_initial_estimate = Real( 1 );
    Real cubic_min_estimate     = std::sqrt( eps );
    Real cubic_max_estimate     = Real( 1 ) / std::sqrt( eps );
    Real cubic_increase_factor  = Real( 2 );
    Real cubic_decrease_factor  = Real( 1 ) / Real( 2 );
    Real cubic_radius_factor    = Real( 1 );

    int verbose = 0;  // print every `verbose` iterations, 0 = silent

    /**
     * \brief Set all convergence tolerances consistently from a single value.
     * \param tol Base tolerance value.
     *
     * This method sets the first-order tolerances from \p tol:
     * - atol, rtol = tol
     * - projected_newton_tolerance = max(32 epsilon, 10 tol)
     *
     * The final-step and curvature tolerances remain at their default automatic
     * machine-precision levels.  Trust-region acceptance thresholds are
     * algorithmic constants and deliberately do not depend on the requested
     * stationarity tolerance.
     */
    void set_tolerances( Real tol )
    {
      atol                       = tol;
      rtol                       = tol;
      projected_newton_tolerance = std::max( Real( 32 ) * eps, tol * Real( 10 ) );
    }

    // Unified setters (same name across all BBOX solvers)
    void set_max_iterations( int n ) { max_iterations = n; }
    void set_absolute_tolerance( Real t ) { atol = t; }
    void set_relative_tolerance( Real t ) { rtol = t; }
  };

  /**
   * \brief Complete result and diagnostic counters returned by a solve.
   * \tparam Real Scalar floating-point type.
   */
  template <typename Real = double> struct Result
  {
    Vector<Real> x;
    Real         objective                     = std::numeric_limits<Real>::quiet_NaN();
    Real         dual_feas                     = std::numeric_limits<Real>::quiet_NaN();  // ||P(x - g) - x||
    Real         primal_feas                   = Real( 0 );                               // bound violation
    Real         step_norm                     = std::numeric_limits<Real>::infinity();   // norm of the last trial step
    Real         minimum_eigenvalue            = std::numeric_limits<Real>::quiet_NaN();
    Real         optimality_tolerance          = std::numeric_limits<Real>::quiet_NaN();
    Real         effective_step_tolerance      = std::numeric_limits<Real>::quiet_NaN();
    Real         effective_curvature_tolerance = std::numeric_limits<Real>::quiet_NaN();
    Real         radius                        = Real( 0 );
    Real         cubic_radius                  = std::numeric_limits<Real>::infinity();
    Real         cubic_lambda                  = Real( 0 );
    Real         hessian_lipschitz_estimate    = Real( 0 );
    int          iterations                    = 0;
    int          function_evaluations          = 0;
    int          gradient_evaluations          = 0;
    int          hessian_evaluations           = 0;
    Status       status                        = Status::unknown;

    /**
     * \brief Test whether first-order stationarity was reached.
     * \return `true` only for \ref Status::first_order.
     */
    [[nodiscard]] bool solved() const noexcept { return status == Status::first_order; }
  };

  // ---------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------

  namespace detail
  {

    /**
     * \brief Project a vector onto the closed bound box.
     *
     * Computes componentwise
     * \f[
     *   z_i=P_{[l_i,u_i]}(x_i)=\min\{u_i,\max\{l_i,x_i\}\}.
     * \f]
     * Infinite bounds are handled naturally by Eigen's componentwise operations.
     */
    template <typename Z, typename X, typename L, typename U> void project(
      Eigen::MatrixBase<Z> &       z,
      Eigen::MatrixBase<X> const & x,
      Eigen::MatrixBase<L> const & l,
      Eigen::MatrixBase<U> const & u )
    { z = x.cwiseMax( l ).cwiseMin( u ); }

    /**
     * \brief Form a feasible projected step along a search direction.
     *
     * Computes \f$s=P_{[l,u]}(x+\alpha d)-x\f$.  Consequently, \f$x+s\f$ is
     * feasible whenever the bounds are consistent.
     */
    template <typename S, typename X, typename D, typename L, typename U, typename Real> void project_step(
      Eigen::MatrixBase<S> &       s,
      Eigen::MatrixBase<X> const & x,
      Eigen::MatrixBase<D> const & d,
      Eigen::MatrixBase<L> const & l,
      Eigen::MatrixBase<U> const & u,
      Real                         alpha )
    { s = ( x + alpha * d ).cwiseMax( l ).cwiseMin( u ) - x; }

    /**
     * \brief Summary of nonzero-direction bound intersection parameters.
     *
     * A breakpoint is a scalar \f$t\f$ at which a component of \f$x+td\f$
     * reaches its lower or upper bound.
     */
    template <typename Real> struct Breakpoints
    {
      Eigen::Index count = 0;
      Real         min   = Real( 0 );
      Real         max   = Real( 0 );
    };

    /**
     * \brief Compute the range of bound-intersection step lengths along \p d.
     *
     * For each nonzero direction component, the routine evaluates the step at
     * which \f$x_i+t d_i\f$ reaches the bound in the direction of motion.  Only
     * the number of breakpoints and their minimum/maximum values are retained.
     * These values delimit piecewise-linear portions of the projected path.
     *
     * \return Breakpoint count together with the minimum and maximum step.
     */
    template <typename Real> [[nodiscard]] Breakpoints<Real> breakpoints(
      Vector<Real> const & x,
      Vector<Real> const & d,
      Vector<Real> const & l,
      Vector<Real> const & u )
    {
      constexpr Real    inf = std::numeric_limits<Real>::infinity();
      Breakpoints<Real> b{ 0, inf, Real( 0 ) };
      for ( Eigen::Index i = 0; i < x.size(); ++i )
      {
        Real t;
        if ( d[i] > Real( 0 ) ) { t = ( u[i] - x[i] ) / d[i]; }
        else if ( d[i] < Real( 0 ) ) { t = ( l[i] - x[i] ) / d[i]; }
        else
        {
          continue;
        }
        ++b.count;
        b.min = std::min( b.min, t );
        b.max = std::max( b.max, t );
      }
      if ( b.count == 0 ) return { 0, Real( 0 ), Real( 0 ) };
      return b;
    }

    /**
     * \brief Test whether a scalar variable is strongly active for minimization.
     *
     * Geometric activity alone is insufficient: at a lower bound a variable is
     * fixed only when the local-model gradient is positive (and conversely at
     * an upper bound).  A weakly active variable with zero model gradient must
     * remain free, otherwise negative curvature tangent to the feasible box
     * could be missed.  Bound proximity and the zero-gradient decision use
     * machine-scale tolerances rather than \f$\sqrt{\epsilon}\f$.
     */
    template <typename Real> [[nodiscard]] bool is_strongly_active( Real xi, Real model_gradient, Real li, Real ui )
    {
      const Real eps   = std::numeric_limits<Real>::epsilon();
      const Real scale = std::max(
        { Real( 1 ),
          std::isfinite( li ) ? std::abs( li ) : Real( 0 ),
          std::isfinite( ui ) ? std::abs( ui ) : Real( 0 ) } );
      const Real x_tol    = Real( 32 ) * eps * scale;
      const Real g_tol    = Real( 32 ) * eps;
      const bool fixed    = std::isfinite( li ) && std::isfinite( ui ) && ui - li <= x_tol;
      const bool at_lower = std::isfinite( li ) && xi <= li + x_tol;
      const bool at_upper = std::isfinite( ui ) && xi >= ui - x_tol;
      return fixed || ( at_lower && model_gradient > g_tol ) || ( at_upper && model_gradient < -g_tol );
    }

    enum class CurvatureStatus
    {
      certified,
      negative,
      failure
    };

    /**
     * \brief Apply the Hessian and evaluate the local quadratic model along a step.
     *
     * Given the (dense) Hessian \p H, computes \f$Hs\f$, the linear model term
     * \f$g^Ts\f$, and
     * \f[
     *   q(s)=g^Ts+\tfrac12 s^T Hs.
     * \f]
     * `H` may be the full-size Hessian or a reduced free-variable block: since
     * `s` is always zero on the variables not participating in the current
     * step, `s^T H s` only ever picks up the relevant block regardless.
     *
     * \return Pair `(slope,q)` with `slope = g.dot(s)`.
     */
    template <typename Real> std::pair<Real, Real> hs_slope_qs(
      Matrix<Real> const & H,
      Vector<Real> const & s,
      Vector<Real> const & g,
      Vector<Real> &       hs )
    {
      hs.noalias()     = H * s;
      const Real slope = g.dot( s );
      const Real qs    = Real( 0.5 ) * s.dot( hs ) + slope;
      return { slope, qs };
    }

    /**
     * \brief Solve a dense symmetric trust-region subproblem directly.
     *
     * Solves
     * \f[
     *   \min_z\; \tfrac12 z^T A z-b^Tz,
     *   \qquad \|z\|_2\le \Delta,
     * \f]
     * from the spectral decomposition \f$A=Q\operatorname{diag}(d)Q^T\f$
     * computed by Eigen 5's `SelfAdjointEigenSolver`.  If the unconstrained
     * Newton step is feasible it is returned directly.  Otherwise the routine
     * finds the nonnegative shift \f$\lambda\f$ satisfying
     * \f$\|(A+\lambda I)^{-1}b\|_2=\Delta\f$ by safeguarded bisection.  The
     * indefinite hard case is completed along an eigenvector associated with
     * the leftmost eigenvalue.
     *
     * \return `stationary` for an interior solution, `boundary` for a shifted
     *         solution, and `direct_solver_failure` if Eigen cannot compute a
     *         finite self-adjoint eigendecomposition.
     *
     * \pre `hessian` is (numerically) symmetric.  The solver enforces this
     *      contract at a single point, right after every Hessian evaluation
     *      (see `Minimize_BBOX_SmallTRON::eval_hessian`), so every reduced block extracted from
     *      it inherits the same symmetry and no further symmetrization is
     *      needed here.  In debug builds this precondition is checked with
     *      `assert`.
     */
    template <typename Real> NewtonStatus direct_trust_region(
      Matrix<Real> const & hessian,
      Vector<Real> const & rhs,
      Real                 radius,
      Vector<Real> &       solution )
    {
      using EigenSolver = Eigen::SelfAdjointEigenSolver<Matrix<Real>>;

      const Eigen::Index n = rhs.size();
      solution.setZero( n );
      if ( n == 0 ) return NewtonStatus::stationary;
      if ( !hessian.allFinite() || !rhs.allFinite() || !std::isfinite( radius ) )
        return NewtonStatus::direct_solver_failure;
      if ( radius <= Real( 0 ) ) return NewtonStatus::boundary;

#ifndef NDEBUG
      {
        const Real scale = std::max( Real( 1 ), hessian.cwiseAbs().maxCoeff() );
        const Real asym  = ( hessian - hessian.transpose() ).cwiseAbs().maxCoeff();
        assert(
          asym <= Real( 1024 ) * std::numeric_limits<Real>::epsilon() * scale &&
          "Utils::SmallTRON_details::detail::direct_trust_region: hessian is not "
          "symmetric; "
          "Problem::hessian(x,H) is expected to be symmetrized once by "
          "Minimize_BBOX_SmallTRON::eval_hessian." );
      }
#endif

      const Real algebra_tolerance = Real( 256 ) * std::numeric_limits<Real>::epsilon();
      const Real matrix_scale      = std::max( Real( 1 ), hessian.cwiseAbs().maxCoeff() );

      Eigen::LDLT<Matrix<Real>> ldlt( hessian );
      if ( ldlt.info() == Eigen::Success && ldlt.isPositive() )
      {
        Vector<Real> newton_step = ldlt.solve( rhs );
        if ( ldlt.info() == Eigen::Success && newton_step.allFinite() )
        {
          const Real residual       = ( hessian * newton_step - rhs ).norm();
          const Real residual_scale = Real( 1 ) + rhs.norm() + matrix_scale * newton_step.norm();
          if (
            residual <= algebra_tolerance * residual_scale &&
            newton_step.norm() <= radius * ( Real( 1 ) + algebra_tolerance ) )
          {
            solution = std::move( newton_step );
            return NewtonStatus::stationary;
          }
        }
      }

      EigenSolver eigensolver( hessian );
      if ( eigensolver.info() != Eigen::Success ) return NewtonStatus::direct_solver_failure;

      auto const & eigenvalues  = eigensolver.eigenvalues();
      auto const & eigenvectors = eigensolver.eigenvectors();
      if ( !eigenvalues.allFinite() || !eigenvectors.allFinite() ) return NewtonStatus::direct_solver_failure;

      Vector<Real> coefficients       = eigenvectors.transpose() * rhs;
      const Real   spectrum_scale     = std::max( Real( 1 ), eigenvalues.cwiseAbs().maxCoeff() );
      const Real   spectral_tolerance = Real( 64 ) * std::numeric_limits<Real>::epsilon() * spectrum_scale *
                                        Real( std::max<Eigen::Index>( 1, n ) );
      const Real   norm_tolerance     = algebra_tolerance;
      const Real   minimum_eigenvalue = eigenvalues[0];

      Vector<Real> spectral_step( n );
      if ( minimum_eigenvalue > spectral_tolerance )
      {
        spectral_step = coefficients.cwiseQuotient( eigenvalues );
        if ( spectral_step.allFinite() && spectral_step.norm() <= radius * ( Real( 1 ) + norm_tolerance ) )
        {
          solution.noalias() = eigenvectors * spectral_step;
          return NewtonStatus::stationary;
        }
      }

      const bool effectively_positive_semidefinite = minimum_eigenvalue >= -spectral_tolerance;
      const Real lower_shift       = effectively_positive_semidefinite ? Real( 0 ) : -minimum_eigenvalue;
      const Real forcing_tolerance = spectral_tolerance * ( Real( 1 ) + rhs.norm() );

      spectral_step.setZero();
      bool         singular_forcing     = false;
      Eigen::Index hard_direction       = 0;
      bool         hard_direction_found = false;
      for ( Eigen::Index i = 0; i < n; ++i )
      {
        const Real denominator = eigenvalues[i] + lower_shift;
        if ( denominator > spectral_tolerance ) { spectral_step[i] = coefficients[i] / denominator; }
        else
        {
          if ( !hard_direction_found )
          {
            hard_direction       = i;
            hard_direction_found = true;
          }
          singular_forcing = singular_forcing || std::abs( coefficients[i] ) > forcing_tolerance;
        }
      }

      const Real lower_step_norm = spectral_step.norm();
      if ( !singular_forcing && lower_step_norm <= radius * ( Real( 1 ) + norm_tolerance ) )
      {
        if ( effectively_positive_semidefinite )
        {
          solution.noalias() = eigenvectors * spectral_step;
          return NewtonStatus::stationary;
        }

        const Real remaining_squared  = std::max( Real( 0 ), radius * radius - lower_step_norm * lower_step_norm );
        spectral_step[hard_direction] = std::copysign(
          std::sqrt( remaining_squared ),
          coefficients[hard_direction] == Real( 0 ) ? Real( 1 ) : coefficients[hard_direction] );
        solution.noalias() = eigenvectors * spectral_step;
        return NewtonStatus::boundary;
      }

      auto shifted_norm = [&]( Real shift )
      {
        Real squared_norm = Real( 0 );
        for ( Eigen::Index i = 0; i < n; ++i )
        {
          const Real denominator = eigenvalues[i] + shift;
          if ( denominator <= Real( 0 ) ) return std::numeric_limits<Real>::infinity();
          const Real value = coefficients[i] / denominator;
          squared_norm += value * value;
          if ( !std::isfinite( squared_norm ) ) return std::numeric_limits<Real>::infinity();
        }
        return std::sqrt( squared_norm );
      };

      Real lower = lower_shift;
      Real upper = std::max( Real( 1 ), lower_shift + Real( 1 ) );
      for ( int k = 0; shifted_norm( upper ) > radius && k < 128; ++k )
      {
        const Real next = Real( 2 ) * upper + Real( 1 );
        if ( !std::isfinite( next ) ) return NewtonStatus::direct_solver_failure;
        upper = next;
      }
      if ( shifted_norm( upper ) > radius ) return NewtonStatus::direct_solver_failure;

      const int bisection_iterations = 2 * std::numeric_limits<Real>::digits;
      for ( int k = 0; k < bisection_iterations; ++k )
      {
        const Real middle = lower + Real( 0.5 ) * ( upper - lower );
        if ( shifted_norm( middle ) > radius ) { lower = middle; }
        else
        {
          upper = middle;
        }
        if ( upper - lower <= norm_tolerance * ( Real( 1 ) + upper ) ) break;
      }

      for ( Eigen::Index i = 0; i < n; ++i ) spectral_step[i] = coefficients[i] / ( eigenvalues[i] + upper );
      if ( !spectral_step.allFinite() ) return NewtonStatus::direct_solver_failure;

      const Real step_norm = spectral_step.norm();
      if ( step_norm > radius ) spectral_step *= radius / step_norm;
      solution.noalias() = eigenvectors * spectral_step;
      return solution.allFinite() ? NewtonStatus::boundary : NewtonStatus::direct_solver_failure;
    }

  }  // namespace detail

  // ---------------------------------------------------------------------------
  // Solver
  // ---------------------------------------------------------------------------

  /**
   * \brief Reusable TRON solver with an explicit dense Hessian interface.
   *
   * The algorithm minimizes \f$f(x)\f$ subject to \f$l\le x\le u\f$ using a
   * trust-region Newton framework specialized to bound constraints.  Each outer
   * iteration consists of:
   *
   * 1. freezing the Hessian \f$H=\nabla^2f(x_c)\f$ at the current accepted
   *    point (re-evaluated only when \f$x_c\f$ actually changed, i.e. skipped
   *    after a rejected trial, since the base point is unchanged);
   * 2. computing a feasible projected Cauchy step that guarantees model decrease;
   * 3. refining that step on the currently free variables by extracting the
   *    reduced Hessian block \f$H_{FF}\f$ and solving the corresponding
   *    trust-region subproblem directly via Eigen's dense eigendecomposition,
   *    followed by a projected line search;
   * 4. evaluating the trial objective and accepting/rejecting the step according
   *    to the ratio of actual to predicted reduction;
   * 5. updating the trust-region radius by TRON's interpolation rules.
   *
   * Because the free-variable subproblem is already solved densely/directly,
   * there is no benefit in hiding the Hessian behind a matrix-free
   * Hessian-vector-product interface: the reduced block would have to be
   * reconstructed one canonical direction at a time anyway.  This variant
   * therefore requires `Problem::hessian(x, H)` to supply the full Hessian
   * matrix, which is assembled once per outer iteration and sliced directly
   * (via Eigen's fancy indexing) to obtain \f$H_{FF}\f$ and to evaluate
   * \f$Hs\f$ everywhere it is needed. It targets small/medium dense problems;
   * a matrix-free variant remains preferable at large scale.
   *
   * All work vectors are owned by the solver and resized together, so repeated
   * calls with a fixed dimension reuse storage.
   *
   * \tparam Real Scalar floating-point type.
   */
  template <typename Real = double> class Minimize_BBOX_SmallTRON
  {
  public:
    using Vec = Vector<Real>;

    /**
     * \brief Construct a solver and allocate workspace for \p nvar variables.
     * \param nvar Problem dimension.
     * \param options Initial solver options.
     */
    explicit Minimize_BBOX_SmallTRON( Eigen::Index nvar = 0, Options<Real> options = {} ) : m_options( options )
    { resize( nvar ); }

    /** \brief Mutable access to the solver options. */
    [[nodiscard]] Options<Real> & options() noexcept { return m_options; }
    /** \brief Read-only access to the solver options. */
    [[nodiscard]] Options<Real> const & options() const noexcept { return m_options; }

    /**
     * \brief Resize all persistent work vectors to a new problem dimension.
     * \param nvar New number of decision variables.
     *
     * No algorithmic state is preserved beyond the option values.  This method
     * centralizes workspace sizing for the outer iteration, projected Newton
     * refinement, and dense direct subproblem.
     */
    void resize( Eigen::Index nvar )
    {
      m_n = nvar;
      m_x.resize( m_n );
      m_xc.resize( m_n );
      m_gx.resize( m_n );
      m_gpx.resize( m_n );
      m_s.resize( m_n );
      m_hs.resize( m_n );
      m_w.resize( m_n );
      m_rhs.resize( m_n );
      m_newton_direction.resize( m_n );
      m_tmp.resize( m_n );
      m_H.resize( m_n, m_n );
      m_H_previous.resize( m_n, m_n );
      m_free_indices.reserve( static_cast<std::size_t>( m_n ) );
    }

    /**
     * \brief Minimize an unconstrained problem.
     *
     * This convenience overload delegates to the bound-constrained method using
     * \f$(-\infty,+\infty)\f$ bounds for every variable.
     */
    template <typename Problem>
      requires ProblemFor<Problem, Real>
    Result<Real> solve( Problem & problem, ConstVectorRef<Real> x0 )
    {
      const Real inf   = std::numeric_limits<Real>::infinity();
      Vec        lower = Vec::Constant( x0.size(), -inf );
      Vec        upper = Vec::Constant( x0.size(), inf );
      return solve( problem, x0, lower, upper );
    }

    /**
     * \brief Minimize a problem over a box with per-iteration user control.
     *
     * The initial point is first projected onto the feasible box.  The method
     * then uses the projected-gradient mapping
     * \f$P(x-g)-x\f$ as its first-order stationarity measure.  The initial norm
     * defines the relative convergence scale, while the initial trust-region
     * radius is chosen from that norm and clipped by \ref Options::max_radius.
     *
     * At each outer iteration the Hessian evaluation point is frozen in `xc_`.
     * The Cauchy phase constructs a feasible descent step for the local quadratic
     * model.  The projected-Newton phase then works on variables not numerically
     * active at their bounds, using an explicitly assembled reduced Hessian and
     * Eigen's self-adjoint direct eigensolver.
     * The resulting trial point is accepted when the actual/predicted reduction
     * ratio exceeds \ref Options::acceptance_threshold.  The radius is updated
     * regardless of acceptance from this ratio and a one-dimensional quadratic
     * interpolation estimate.
     *
     * \param problem Objective/derivative provider.
     * \param x0 Initial point.
     * \param lower Componentwise lower bounds.
     * \param upper Componentwise upper bounds.
     * \return Final iterate, diagnostic counters, and termination status.
     *
     * \note The code assumes compatible vector dimensions and consistent bounds.
     * \note `max_eval` counts objective evaluations only.
     */
    template <typename Problem>
      requires ProblemFor<Problem, Real>
    Result<Real> solve(
      Problem &            problem,
      ConstVectorRef<Real> x0,
      ConstVectorRef<Real> lower,
      ConstVectorRef<Real> upper )
    {
      const Options<Real> & o   = m_options;
      constexpr Real        eps = std::numeric_limits<Real>::epsilon();

      resize( x0.size() );
      m_obj_evals = m_grad_evals = m_hess_evals = 0;
      m_hessian_valid                           = false;
      const Real cubic_min                      = std::max( Real( 0 ), o.cubic_min_estimate );
      const Real cubic_max                      = std::max( cubic_min, o.cubic_max_estimate );
      m_cubic_estimate                          = std::clamp( o.cubic_initial_estimate, cubic_min, cubic_max );
      m_cubic_radius                            = std::numeric_limits<Real>::infinity();
      m_cubic_lambda                            = Real( 0 );

      Result<Real> res;

      detail::project( m_x, x0, lower, upper );

      const Real machine_tolerance = Real( 32 ) * eps;
      m_radius                     = std::min( std::max( Real( 1 ), m_x.stableNorm() / Real( 10 ) ), o.max_radius );
      Real alpha_c                 = Real( 1 );
      m_ratio                      = Real( 0 );
      m_quad_min                   = Real( 0 );
      int  num_success             = 0;
      int  iter                    = 0;
      Real last_step_norm          = Real( 0 );
      Real minimum_eigenvalue      = std::numeric_limits<Real>::quiet_NaN();
      Real effective_curvature_tolerance = std::numeric_limits<Real>::quiet_NaN();
      NewtonStatus newton_status   = NewtonStatus::unknown;

      Real fx;
      if ( !eval_objective( problem, m_x, fx ) )
      {
        res.x                      = m_x;
        res.objective              = std::numeric_limits<Real>::quiet_NaN();
        res.dual_feas              = std::numeric_limits<Real>::quiet_NaN();
        res.primal_feas            = bound_violation( m_x, lower, upper );
        res.step_norm              = last_step_norm;
        res.minimum_eigenvalue     = minimum_eigenvalue;
        res.optimality_tolerance   = machine_tolerance;
        res.effective_step_tolerance = machine_tolerance;
        res.effective_curvature_tolerance = effective_curvature_tolerance;
        res.radius                 = m_radius;
        res.cubic_radius           = m_cubic_radius;
        res.cubic_lambda           = m_cubic_lambda;
        res.hessian_lipschitz_estimate = m_cubic_estimate;
        res.iterations             = iter;
        res.function_evaluations   = m_obj_evals;
        res.gradient_evaluations   = m_grad_evals;
        res.hessian_evaluations    = m_hess_evals;
        res.status                 = Status::user_request;
        return res;
      }
      if ( !eval_gradient( problem, m_x, m_gx ) )
      {
        res.x                      = m_x;
        res.objective              = fx;
        res.dual_feas              = std::numeric_limits<Real>::quiet_NaN();
        res.primal_feas            = bound_violation( m_x, lower, upper );
        res.step_norm              = last_step_norm;
        res.minimum_eigenvalue     = minimum_eigenvalue;
        res.optimality_tolerance   = machine_tolerance;
        res.effective_step_tolerance = machine_tolerance;
        res.effective_curvature_tolerance = effective_curvature_tolerance;
        res.radius                 = m_radius;
        res.cubic_radius           = m_cubic_radius;
        res.cubic_lambda           = m_cubic_lambda;
        res.hessian_lipschitz_estimate = m_cubic_estimate;
        res.iterations             = iter;
        res.function_evaluations   = m_obj_evals;
        res.gradient_evaluations   = m_grad_evals;
        res.hessian_evaluations    = m_hess_evals;
        res.status                 = Status::user_request;
        return res;
      }

      detail::project_step( m_gpx, m_x, m_gx, lower, upper, Real( -1 ) );
      Real pi_x   = m_gpx.stableNorm();
      Real primal = bound_violation( m_x, lower, upper );

      const Real relative_optimality_tolerance = o.rtol * std::max( Real( 1 ), pi_x );
      const Real optimality_tolerance          = std::max(
        machine_tolerance,
        std::min( o.atol, relative_optimality_tolerance ) );
      const Real fmin = std::min( Real( -1 ), fx ) / eps;

      auto effective_step_tolerance = [&]
      {
        const Real floor = machine_tolerance * std::max( Real( 1 ), m_x.stableNorm() );
        return o.step_tolerance > Real( 0 ) ? std::max( floor, o.step_tolerance ) : floor;
      };
      auto primal_tolerance = [&] { return machine_tolerance * std::max( Real( 1 ), m_x.stableNorm() ); };

      auto fill = [&]( Status st ) -> Result<Real> &
      {
        res.x                             = m_x;
        res.objective                     = fx;
        res.dual_feas                     = pi_x;
        res.primal_feas                   = primal;
        res.step_norm                     = last_step_norm;
        res.minimum_eigenvalue            = minimum_eigenvalue;
        res.optimality_tolerance          = optimality_tolerance;
        res.effective_step_tolerance      = effective_step_tolerance();
        res.effective_curvature_tolerance = effective_curvature_tolerance;
        res.radius                        = m_radius;
        res.cubic_radius                  = m_cubic_radius;
        res.cubic_lambda                  = m_cubic_lambda;
        res.hessian_lipschitz_estimate    = m_cubic_estimate;
        res.iterations                    = iter;
        res.function_evaluations          = m_obj_evals;
        res.gradient_evaluations          = m_grad_evals;
        res.hessian_evaluations           = m_hess_evals;
        res.status                        = st;
        return res;
      };

      auto stopping_metrics_are_small = [&]
      {
        return pi_x <= optimality_tolerance && last_step_norm <= effective_step_tolerance() &&
               primal <= primal_tolerance();
      };
      auto minimum_status = [&]() -> Status
      {
        if ( !stopping_metrics_are_small() ) return Status::unknown;
        if ( !m_hessian_valid )
        {
          m_xc = m_x;
          if ( !eval_hessian( problem, m_xc, m_H ) ) return Status::user_request;
          m_hessian_valid = true;
        }
        const auto curvature = critical_curvature(
          m_H,
          m_x,
          m_gx,
          lower,
          upper,
          optimality_tolerance,
          o.curvature_tolerance,
          minimum_eigenvalue,
          effective_curvature_tolerance );
        if ( curvature == detail::CurvatureStatus::failure ) return Status::direct_solver_failure;
        return curvature == detail::CurvatureStatus::certified ? Status::first_order : Status::unknown;
      };
      auto current_status = [&]
      {
        const Status candidate = minimum_status();
        if ( candidate != Status::unknown ) return candidate;
        if ( fx < fmin ) return Status::unbounded;
        if ( o.max_evaluations >= 0 && m_obj_evals >= o.max_evaluations ) return Status::max_eval;
        if ( iter >= o.max_iterations ) return Status::max_iter;
        return Status::unknown;
      };

      if ( o.verbose > 0 )
      {
        std::printf( "%6s  %14s  %10s  %10s  %10s  %s\n", "iter", "f(x)", "pi", "radius", "ratio", "Newton status" );
        std::printf(
          "%6d  %14.7e  %10.3e  %10.3e  %10s  %s\n",
          0,
          double( fx ),
          double( pi_x ),
          double( m_radius ),
          "-",
          "-" );
      }

      Status status = current_status();
      if ( status != Status::unknown ) return fill( status );

      while ( true )
      {
        const Real fc = fx;

        // The Hessian is frozen at xc_ for the whole outer iteration.  After an
        // accepted step, retain the old matrix long enough to estimate the local
        // Hessian Lipschitz constant from two consecutive accepted points.
        if ( !m_hessian_valid )
        {
          if ( m_hess_evals == 0 )
          {
            m_xc = m_x;
            if ( !eval_hessian( problem, m_xc, m_H ) ) return fill( Status::user_request );
          }
          else
          {
            const Real dx = ( m_x - m_xc ).stableNorm();
            m_H_previous  = m_H;
            m_xc          = m_x;
            if ( !eval_hessian( problem, m_xc, m_H ) ) return fill( Status::user_request );

            if ( o.use_cubic_radius && dx > effective_step_tolerance() )
            {
              const Real observed = ( m_H - m_H_previous ).stableNorm() / dx;
              if ( std::isfinite( observed ) )
                m_cubic_estimate = std::clamp( std::max( observed, m_cubic_estimate ), cubic_min, cubic_max );
            }
          }
          m_hessian_valid = true;
        }

        // Cubic regularization guides only the radius.  The direct solve below
        // still minimizes the unmodified TRON quadratic model, avoiding a
        // duplicate H + lambda I shift.
        if ( o.use_cubic_radius && pi_x > optimality_tolerance )
        {
          const Real pnorm = std::max( pi_x, machine_tolerance );
          const Real M     = std::clamp( m_cubic_estimate, cubic_min, cubic_max );
          if ( M > Real( 0 ) )
          {
            m_cubic_lambda = std::sqrt( M * pnorm );
            m_cubic_radius = o.cubic_radius_factor * std::sqrt( pnorm / M );
            if ( std::isfinite( m_cubic_radius ) && m_cubic_radius > Real( 0 ) )
              m_radius = std::min( m_radius, m_cubic_radius );
          }
        }
        else
        {
          m_cubic_lambda = Real( 0 );
          m_cubic_radius = std::numeric_limits<Real>::infinity();
        }

        const Real delta = m_radius;

        if ( !cauchy( m_H, m_x, m_gx, delta, alpha_c, lower, upper ) )
        {
          last_step_norm = m_s.stableNorm();
          status         = minimum_status();
          if ( status != Status::unknown ) return fill( status );
          if ( !stopping_metrics_are_small() ) return fill( Status::small_step );

          // A first-order stationary point with negative critical curvature is
          // not a minimum. Start the direct reduced solve from a zero Cauchy
          // step so that it can follow the negative-curvature direction.
          m_s.setZero();
          m_hs.setZero();
        }

        newton_status = projected_newton( m_H, m_x, m_gx, delta, lower, upper );
        if ( newton_status == NewtonStatus::direct_solver_failure )
        {
          m_x = m_xc;
          return fill( Status::direct_solver_failure );
        }

        const Real slope           = m_gx.dot( m_s );
        const Real qs              = Real( 0.5 ) * m_s.dot( m_hs ) + slope;
        const Real model_reduction = -qs;
        Real f_trial;
        if ( !eval_objective( problem, m_x, f_trial ) ) return fill( Status::user_request );

        // Require genuine descent in the quadratic model.  Objective changes
        // below its floating-point resolution are handled by adding the same
        // roundoff allowance to actual and predicted reductions; this lets the
        // derivative model refine a solution even when f(x+s)==f(x) because of
        // a large constant offset, without turning an ascent model into descent.
        if ( !( model_reduction > Real( 0 ) ) )
        {
          fx             = fc;
          m_x            = m_xc;
          last_step_norm = m_s.stableNorm();
          status         = minimum_status();
          if ( status != Status::unknown ) return fill( status );
          return fill( Status::neg_pred );
        }
        const Real objective_roundoff = Real( 16 ) * eps *
                                        std::max( { Real( 1 ), std::abs( fc ), std::abs( f_trial ) } );
        const Real actual_reduction   = fc - f_trial;
        m_ratio = ( actual_reduction + objective_roundoff ) / ( model_reduction + objective_roundoff );

        // Quadratic interpolation factor used by the TRON radius update.
        const Real gamma = f_trial - fc - slope;
        m_quad_min       = gamma <= Real( 0 ) ? o.increase_factor
                                              : std::max( o.large_decrease_factor, -slope / ( Real( 2 ) * gamma ) );

        if ( m_ratio >= o.acceptance_threshold )
        {
          ++num_success;
          fx = f_trial;
          if ( !eval_gradient( problem, m_x, m_gx ) ) return fill( Status::user_request );
          detail::project_step( m_gpx, m_x, m_gx, lower, upper, Real( -1 ) );
          pi_x            = m_gpx.stableNorm();
          m_hessian_valid = false;  // x_ moved: H_ must be re-evaluated at xc_ =
                                    // x_ next iteration

          if ( o.use_cubic_radius && m_ratio >= o.increase_threshold )
            m_cubic_estimate = std::clamp( o.cubic_decrease_factor * m_cubic_estimate, cubic_min, cubic_max );
        }
        else
        {
          fx  = fc;
          m_x = m_xc;  // x_ == xc_ again, so H_ (already evaluated there) stays
                       // valid

          if ( o.use_cubic_radius )
            m_cubic_estimate = std::clamp( o.cubic_increase_factor * m_cubic_estimate, cubic_min, cubic_max );
        }

        primal = bound_violation( m_x, lower, upper );
        ++iter;

        if ( o.verbose > 0 && iter % o.verbose == 0 )
        {
          std::printf(
            "%6d  %14.7e  %10.3e  %10.3e  %10.3e  %s\n",
            iter,
            double( fx ),
            double( pi_x ),
            double( delta ),
            double( m_ratio ),
            to_string( newton_status ).data() );
        }

        const Real s_norm = m_s.stableNorm();
        last_step_norm    = s_norm;
        if ( num_success == 0 ) m_radius = std::min( delta, s_norm );
        update_radius( s_norm );

        status = current_status();
        if ( status != Status::unknown ) break;
      }

      fill( status );
      if ( o.verbose > 0 )
      {
        std::printf(
          "%6d  %14.7e  %10.3e  %10.3e  -> %s\n",
          iter,
          double( fx ),
          double( pi_x ),
          double( m_radius ),
          to_string( status ).data() );
      }
      return res;
    }

    template <typename Obj, typename Grad, typename Hess> Result<Real> solve(
      Obj &&               obj,
      Grad &&              grad,
      Hess &&              hess,
      ConstVectorRef<Real> x0,
      ConstVectorRef<Real> lower,
      ConstVectorRef<Real> upper )
    {
      auto prob =
        ::Utils::make_problem<Real>( std::forward<Obj>( obj ), std::forward<Grad>( grad ), std::forward<Hess>( hess ) );
      return solve( prob, x0, lower, upper );
    }

    // Unified setters (same name across all BBOX solvers)
    void set_tolerances( Real tol ) { this->options().set_tolerances( tol ); }
    void set_max_iterations( int n ) { this->options().set_max_iterations( n ); }


  private:
    // --- objective / derivative wrappers -------------------------------------
    /** \brief Evaluate the objective and increment the objective counter. */
    template <typename Problem> bool eval_objective( Problem & p, Vec const & x, Real & f )
    {
      ++m_obj_evals;
      return p.objective( x, f );
    }

    /** \brief Evaluate the gradient and increment the gradient counter. */
    template <typename Problem> bool eval_gradient( Problem & p, Vec const & x, Vec & g )
    {
      ++m_grad_evals;
      return p.gradient( x, g );
    }

    /**
     * \brief Evaluate the Hessian, increment its counter, and symmetrize it.
     *
     * This is the single point in the solver where `H = (H + H^T)/2` is
     * enforced.  Every reduced block later extracted from `H_` (in \ref
     * projected_newton) therefore inherits exact symmetry, and no further
     * symmetrization is needed anywhere downstream (see \ref
     * detail::direct_trust_region).
     */
    template <typename Problem> bool eval_hessian( Problem & p, Vec const & x, Matrix<Real> & H )
    {
      ++m_hess_evals;
      bool ok = p.hessian( x, H );
      H = ( Real( 0.5 ) * ( H + H.transpose() ) ).eval();
      return ok;
    }

    /**
     * \brief Compute the Euclidean norm of componentwise box infeasibility.
     *
     * The violation vector is
     * \f$\max\{l-x,\;x-u,\;0\}\f$ componentwise.
     */
    [[nodiscard]] static Real bound_violation( Vec const & x, Vec const & l, Vec const & u )
    { return ( l - x ).cwiseMax( x - u ).cwiseMax( Real( 0 ) ).norm(); }

    /**
     * \brief Check second-order necessary curvature on the box critical cone.
     *
     * Variables whose active-bound gradient points strictly out of the feasible
     * box are strongly active and are removed. Interior, weakly active, and
     * degenerate active variables form a conservative linearized critical
     * subspace. Requiring its principal Hessian block to be positive
     * semidefinite is sufficient to reject interior saddles and degenerate
     * boundary maxima, while allowing ordinary minima with a nonzero gradient
     * normal to an active bound.
     */
    detail::CurvatureStatus critical_curvature(
      Matrix<Real> const & H,
      Vec const &          x,
      Vec const &          g,
      Vec const &          l,
      Vec const &          u,
      Real                 optimality_tolerance,
      Real                 requested_curvature_tolerance,
      Real &               minimum_eigenvalue,
      Real &               effective_curvature_tolerance )
    {
      const Real activity_tolerance = std::max(
        optimality_tolerance,
        Real( 32 ) * std::numeric_limits<Real>::epsilon() );

      m_free_indices.clear();
      for ( Eigen::Index i = 0; i < m_n; ++i )
      {
        const Real bound_scale = std::max(
          { Real( 1 ),
            std::isfinite( l[i] ) ? std::abs( l[i] ) : Real( 0 ),
            std::isfinite( u[i] ) ? std::abs( u[i] ) : Real( 0 ) } );
        const bool fixed           = std::isfinite( l[i] ) && std::isfinite( u[i] ) &&
                                     std::abs( u[i] - l[i] ) <= activity_tolerance * bound_scale;
        const bool at_lower        = std::isfinite( l[i] ) && x[i] <= l[i] + activity_tolerance * bound_scale;
        const bool at_upper        = std::isfinite( u[i] ) && x[i] >= u[i] - activity_tolerance * bound_scale;
        const bool strongly_active = fixed || ( at_lower && g[i] > optimality_tolerance ) ||
                                     ( at_upper && g[i] < -optimality_tolerance );
        if ( !strongly_active ) m_free_indices.push_back( i );
      }

      if ( m_free_indices.empty() )
      {
        minimum_eigenvalue            = std::numeric_limits<Real>::infinity();
        effective_curvature_tolerance = Real( 0 );
        return detail::CurvatureStatus::certified;
      }

      m_reduced_hessian = H( m_free_indices, m_free_indices );
      if ( !m_reduced_hessian.allFinite() ) return detail::CurvatureStatus::failure;

      Eigen::SelfAdjointEigenSolver<Matrix<Real>> eigensolver( m_reduced_hessian, Eigen::EigenvaluesOnly );
      if ( eigensolver.info() != Eigen::Success || !eigensolver.eigenvalues().allFinite() )
        return detail::CurvatureStatus::failure;

      minimum_eigenvalue            = eigensolver.eigenvalues()[0];
      const Real spectrum_scale     = std::max( Real( 1 ), eigensolver.eigenvalues().cwiseAbs().maxCoeff() );
      const Real backward_error     = Real( 64 ) * std::numeric_limits<Real>::epsilon() * spectrum_scale *
                                      Real( std::max<Eigen::Index>( 1, m_reduced_hessian.rows() ) );
      effective_curvature_tolerance = requested_curvature_tolerance > Real( 0 )
                                        ? std::max( requested_curvature_tolerance, backward_error )
                                        : backward_error;
      return minimum_eigenvalue >= -effective_curvature_tolerance ? detail::CurvatureStatus::certified
                                                                  : detail::CurvatureStatus::negative;
    }

    // --- trust-region radius update (TRONTrustRegion) ------------------------
    /**
     * \brief Update the trust-region radius using TRON interpolation rules.
     *
     * The update combines the agreement ratio `ratio_` between actual and
     * predicted reduction with `quad_min_`, the minimizer/interpolation factor of
     * a scalar quadratic model along the accepted trial direction.  Poor model
     * agreement contracts the radius, intermediate agreement limits its growth,
     * and strong agreement permits expansion.  The final radius is always capped
     * by \ref Options::max_radius.
     *
     * \param step_norm Norm of the most recently computed total step.
     */
    void update_radius( Real step_norm )
    {
      const Options<Real> & o = m_options;
      const Real            a = m_quad_min;
      if ( m_ratio <= o.acceptance_threshold )
      {
        m_radius = std::min( std::max( a, o.large_decrease_factor ) * step_norm, o.small_decrease_factor * m_radius );
      }
      else if ( m_ratio < o.decrease_threshold )
      {
        m_radius = std::max(
          o.large_decrease_factor * m_radius,
          std::min( a * step_norm, o.small_decrease_factor * m_radius ) );
      }
      else if ( m_ratio < o.increase_threshold )
      {
        m_radius = std::max(
          o.large_decrease_factor * m_radius,
          std::min( a * step_norm, o.increase_factor * m_radius ) );
      }
      else
      {
        m_radius = std::max( m_radius, std::min( a * step_norm, o.increase_factor * m_radius ) );
      }
      m_radius = std::min( m_radius, o.max_radius );
    }

    // --- Cauchy point --------------------------------------------------------
    /**
     * \brief Compute a projected Cauchy step for the local quadratic model.
     *
     * Along the projected negative-gradient path
     * \f[
     *   s(\alpha)=P_{[l,u]}(x-\alpha g)-x,
     * \f]
     * the routine searches geometrically in \f$\alpha\f$ for a step satisfying
     * both
     * \f[
     *   q(s) \le \mu_0 g^Ts,
     *   \qquad \|s\|_2 \le \mu_1\Delta.
     * \f]
     * If the incoming step is too long or fails sufficient model decrease,
     * \f$\alpha\f$ is reduced by `sigma`.  Otherwise the routine extrapolates by
     * `sigma` up to the last relevant breakpoint, retaining the largest tested
     * acceptable value.  This reproduces the TRON projected Cauchy-point search.
     *
     * On exit, `s_` contains the selected projected step and `hs_` contains its
     * Hessian product.  The scalar \p alpha is updated for reuse as the initial
     * Cauchy parameter in the following outer iteration.
     *
     * \return `false` only if repeated contraction drives \p alpha below the
     *         underflow safeguard based on `denorm_min()`.
     */
    bool cauchy(
      Matrix<Real> const & H,
      Vec const &          x,
      Vec const &          g,
      Real                 delta,
      Real &               alpha,
      Vec const &          l,
      Vec const &          u )
    {
      const Options<Real> & o           = m_options;
      const Real            alpha_start = alpha;

      m_tmp              = -g;
      const Real brk_max = detail::breakpoints( x, m_tmp, l, u ).max;

      m_s.setZero();
      m_hs.setZero();

      detail::project_step( m_s, x, g, l, u, -alpha );
      if ( m_s.squaredNorm() == Real( 0 ) ) return false;

      bool interpolate;
      if ( m_s.norm() > o.mu_1 * delta ) { interpolate = true; }
      else
      {
        const auto [slope, qs] = detail::hs_slope_qs( H, m_s, g, m_hs );
        interpolate            = qs >= o.mu_0 * slope;
      }

      if ( interpolate )
      {
        const Real alpha_min = std::sqrt( std::numeric_limits<Real>::denorm_min() );
        bool       search    = true;
        while ( search )
        {
          alpha /= o.sigma;
          detail::project_step( m_s, x, g, l, u, -alpha );
          if ( m_s.norm() <= o.mu_1 * delta )
          {
            const auto [slope, qs] = detail::hs_slope_qs( H, m_s, g, m_hs );
            search                 = qs >= o.mu_0 * slope;
          }
          if ( alpha < alpha_min )
          {
            alpha = alpha_start;
            return false;
          }
        }
      }
      else
      {
        Real alpha_ok = alpha;
        bool search   = true;
        while ( search && alpha <= brk_max )
        {
          alpha *= o.sigma;
          detail::project_step( m_s, x, g, l, u, -alpha );
          if ( m_s.norm() <= o.mu_1 * delta )
          {
            const auto [slope, qs] = detail::hs_slope_qs( H, m_s, g, m_hs );
            if ( qs <= o.mu_0 * slope ) alpha_ok = alpha;
          }
          else
          {
            search = false;
          }
        }
        alpha = alpha_ok;
        detail::project_step( m_s, x, g, l, u, -alpha );
        detail::hs_slope_qs( H, m_s, g, m_hs );
      }
      return true;
    }

    // --- projected line search ----------------------------------------------
    /**
     * \brief Perform a projected backtracking line search on the quadratic model.
     *
     * Starting from a unit step, forms
     * \f$w(\alpha)=P_{[l,u]}(x+\alpha d)-x\f$ and halves \f$\alpha\f$ until
     * the sufficient-decrease condition
     * \f[
     *   \tfrac12 w^THw+w^Tg \le \mu_0 w^Tg
     * \f]
     * holds, or until the first breakpoint of the projected path is crossed.
     * The accepted projected displacement is then applied directly to \p x.
     *
     * The method operates entirely on the local quadratic model; it performs no
     * objective-function evaluations.
     */
    void projected_line_search(
      Matrix<Real> const & H,
      Vec &                x,
      Vec const &          g,
      Vec const &          d,
      Vec const &          l,
      Vec const &          u,
      Vec &                hw,
      Vec &                w )
    {
      const Real brk_min = detail::breakpoints( x, d, l, u ).min;
      Real       alpha   = Real( 1 );

      w.setZero();
      hw.setZero();

      bool search = true;
      while ( search && alpha > brk_min )
      {
        detail::project_step( w, x, d, l, u, alpha );
        const auto [slope, qs] = detail::hs_slope_qs( H, w, g, hw );
        if ( qs <= m_options.mu_0 * slope ) { search = false; }
        else
        {
          alpha /= Real( 2 );
        }
      }
      if ( alpha < Real( 1 ) && alpha < brk_min )
      {
        alpha = brk_min;
        detail::project_step( w, x, d, l, u, alpha );
        detail::hs_slope_qs( H, w, g, hw );
      }

      detail::project_step( w, x, d, l, u, alpha );
      x += w;
    }

    // --- projected Newton step ----------------------------------------------
    /**
     * \brief Refine the Cauchy step by a projected Newton iteration on free
     * variables.
     *
     * The routine first advances \p x by the previously computed Cauchy step
     * `s_`.  At every refinement iteration it classifies variables that are
     * numerically on a bound and builds the diagonal mask
     * \f$Z=\operatorname{diag}(m_i)\f$, with \f$m_i=0\f$ for active variables
     * and \f$m_i=1\f$ otherwise.  Hessian products are then restricted to the
     * free subspace through
     * \f[
     *   v \mapsto Z H Z v.
     * \f]
     *
     * The Newton correction solves
     * \f[
     *   ZHZ\,d = -Z(g+Hs)
     * \f]
     * by assembling the symmetric reduced Hessian and applying
     * \ref detail::direct_trust_region.  A projected quadratic-model line search
     * globalizes that correction and the cumulative step `s_` is updated.
     * Refinement stops when the masked model gradient is sufficiently small, the
     * direct solution reaches the trust-region boundary, or the configured
     * projected-Newton iteration limit is met.
     *
     * \param H Hessian matrix, frozen at the outer iteration base point.
     * \param x Current trial point; modified in place.
     * \param g Gradient at the outer iteration base point.
     * \param delta Trust-region radius supplied to each direct reduced solve.
     * \param l Lower bounds.
     * \param u Upper bounds.
     * \return Inner iteration termination reason.
     *
     * \note With an explicit Hessian, the reduced free-variable block
     *       \f$H_{FF}\f$ is simply the submatrix of `H` indexed by the free
     *       variables (`H(free_indices_, free_indices_)`, using Eigen's
     *       generalized/"fancy" indexing) -- there is no need to reconstruct it
     *       one column at a time from Hessian-vector products.  Likewise, the
     *       projected line search can consume the full-size `H` directly: the
     *       Newton direction and every projected displacement `w` it produces
     *       are zero on the fixed variables by construction, so `w^T H w`
     *       already reduces to `w_F^T H_{FF} w_F` without needing a masked
     *       operator.
     */
    NewtonStatus projected_newton(
      Matrix<Real> const & H,
      Vec &                x,
      Vec const &          g,
      Real                 delta,
      Vec const &          l,
      Vec const &          u )
    {
      const Options<Real> & o = m_options;

      m_hs.noalias() = H * m_s;
      x += m_s;
      detail::project( x, x, l, u );

      NewtonStatus status = NewtonStatus::unknown;
      int          iters  = 0;
      while ( true )
      {
        m_free_indices.clear();
        for ( Eigen::Index i = 0; i < m_n; ++i )
        {
          const Real model_gradient = g[i] + m_hs[i];
          if ( !detail::is_strongly_active( x[i], model_gradient, l[i], u[i] ) ) m_free_indices.push_back( i );
        }
        if ( m_free_indices.empty() ) return NewtonStatus::stationary;

        const Eigen::Index nfree = static_cast<Eigen::Index>( m_free_indices.size() );

        // Reduced right-hand side -(g + H s) on the free variables, and the
        // norm of the free part of -g alone (used for the relative stopping
        // test below), built directly -- no full-size mask vector needed.
        m_reduced_rhs.resize( nfree );
        Real gfnorm2 = Real( 0 );
        for ( Eigen::Index k = 0; k < nfree; ++k )
        {
          const Eigen::Index i  = m_free_indices[k];
          const Real         gi = -g[i];
          gfnorm2 += gi * gi;
          m_reduced_rhs[k] = -( g[i] + m_hs[i] );
        }
        const Real gfnorm = std::sqrt( gfnorm2 );

        // H_{FF}: a direct submatrix of the explicit Hessian, symmetric
        // because H itself was symmetrized once in eval_hessian().
        m_reduced_hessian = H( m_free_indices, m_free_indices );
        if ( !m_reduced_hessian.allFinite() ) return NewtonStatus::direct_solver_failure;

        const NewtonStatus direct_status =
          detail::direct_trust_region( m_reduced_hessian, m_reduced_rhs, delta, m_reduced_solution );
        if ( direct_status == NewtonStatus::direct_solver_failure ) return direct_status;

        m_newton_direction.setZero();
        m_newton_direction( m_free_indices ) = m_reduced_solution;
        ++iters;

        // Gradient of the local quadratic model, g + H s, restricted to the
        // free variables (zero elsewhere): the sign of reduced_rhs_ above.
        m_rhs.setZero();
        for ( Eigen::Index k = 0; k < nfree; ++k )
        {
          const Eigen::Index i = m_free_indices[k];
          m_rhs[i]             = g[i] + m_hs[i];
        }
        projected_line_search( H, x, m_rhs, m_newton_direction, l, u, m_hs, m_w );
        m_s += m_w;

        m_hs.noalias() = H * m_s;
        Real newnorm2  = Real( 0 );
        for ( Eigen::Index k = 0; k < nfree; ++k )
        {
          const Eigen::Index i = m_free_indices[k];
          const Real         v = m_hs[i] + g[i];
          newnorm2 += v * v;
        }
        const Real newnorm = std::sqrt( newnorm2 );

        if ( newnorm <= o.projected_newton_tolerance * gfnorm ) { status = NewtonStatus::stationary; }
        else if ( direct_status == NewtonStatus::boundary ) { status = NewtonStatus::boundary; }
        else if ( iters >= o.max_projected_newton_iterations ) { status = NewtonStatus::max_iter; }
        if ( status != NewtonStatus::unknown ) return status;
      }
    }

    Options<Real> m_options{};
    Eigen::Index  m_n = 0;

    Vec m_x, m_xc, m_gx, m_gpx, m_s, m_hs, m_w, m_tmp;
    Vec m_rhs, m_newton_direction;
    Vec m_reduced_rhs, m_reduced_solution;

    Matrix<Real>              m_H;  // Hessian, frozen at xc for the outer iteration
    Matrix<Real>              m_H_previous;
    Matrix<Real>              m_reduced_hessian;
    std::vector<Eigen::Index> m_free_indices;

    Real m_radius         = Real( 1 );
    Real m_ratio          = Real( 0 );
    Real m_quad_min       = Real( 0 );
    Real m_cubic_estimate = Real( 1 );
    Real m_cubic_radius   = std::numeric_limits<Real>::infinity();
    Real m_cubic_lambda   = Real( 0 );

    bool m_hessian_valid = false;  // true when H is up to date at xc == x

    int m_obj_evals  = 0;
    int m_grad_evals = 0;
    int m_hess_evals = 0;
  };

  // ---------------------------------------------------------------------------
  // Free-function entry points
  // ---------------------------------------------------------------------------

  /**
   * \brief One-shot box-constrained minimization convenience function.
   *
   * Constructs a temporary \ref Solver, solves the problem, and returns its
   * result.  Use \ref Solver directly when repeated solves should reuse
   * workspace.
   */
  template <typename Real, typename Problem>
    requires ProblemFor<Problem, Real>
  Result<Real> minimize(
    Problem &             problem,
    ConstVectorRef<Real>  x0,
    ConstVectorRef<Real>  lower,
    ConstVectorRef<Real>  upper,
    Options<Real> const & options = {} )
  {
    Minimize_BBOX_SmallTRON<Real> solver( x0.size(), options );
    return solver.solve( problem, x0, lower, upper );
  }

  /**
   * \brief One-shot unconstrained minimization convenience function.
   *
   * Equivalent to constructing a temporary solver and calling the unconstrained
   * \ref Solver::solve overload.
   */
  template <typename Real, typename Problem>
    requires ProblemFor<Problem, Real>
  Result<Real> minimize( Problem & problem, ConstVectorRef<Real> x0, Options<Real> const & options = {} )
  {
    Minimize_BBOX_SmallTRON<Real> solver( x0.size(), options );
    return solver.solve( problem, x0 );
  }

}  // namespace Utils::SmallTRON_details

namespace Utils
{
  namespace SmallTRON = SmallTRON_details;  // source compatibility

  template <typename Real = double> using Minimize_BBOX_SmallTRON = SmallTRON_details::Minimize_BBOX_SmallTRON<Real>;

  using SmallTRON_details::to_string;

}  // namespace Utils

#endif  // UTILS_MINIMIZE_BBOX_SMALL_TRON_DOT_HH
