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

// Utils_TRON.hh -- header-only C++20 port of the TRON trust-region solver for
// bound-constrained minimization:
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
// Dependencies: Eigen (Dense/Eigenvalues) and the C++20 standard library.

#pragma once

#ifndef UTILS_TRON_DOT_HH
#define UTILS_TRON_DOT_HH

#include "Utils_eigen.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <functional>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if EIGEN_MAJOR_VERSION < 5
#error "Utils::TRON requires Eigen 5 or newer"
#endif

namespace Utils::TRON
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
  template <typename Real> using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
  template <typename Real> using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

  // ---------------------------------------------------------------------------
  // Problem interface
  // ---------------------------------------------------------------------------

  /**
   * \brief Compile-time interface required by the TRON solver.
   *
   * A model satisfying this concept provides the objective value, the full
   * gradient, and Hessian-vector products.  The solver never requires an
   * explicitly assembled Hessian matrix; this is important both for large-scale
   * problems and for applications in which a Hessian action is substantially
   * cheaper than Hessian formation.
   *
   * The required operations are
   * \f[
   *   f(x), \qquad g(x)=\nabla f(x), \qquad H(x)v=\nabla^2 f(x)v.
   * \f]
   * The Hessian is assumed to represent the symmetric Hessian of the objective.
   *
   * \tparam Problem User problem type.
   * \tparam Real Scalar floating-point type.
   *
   * \note No dimension checks are imposed by the concept.  The problem methods
   *       must produce vectors compatible with the input dimension.
   */
  template <typename Problem, typename Real>
  concept ProblemFor = requires( Problem & p, Vector<Real> const & x, Vector<Real> const & v, Vector<Real> & out ) {
    { p.objective( x ) } -> std::convertible_to<Real>;
    { p.gradient( x, out ) };
    { p.hprod( x, v, out ) };
  };

  /**
   * \brief Adapter that turns three independent callables into a TRON problem.
   *
   * This utility avoids the need to define a dedicated problem class.  The three
   * callables are stored by value and invoked by \ref objective, \ref gradient,
   * and \ref hprod.
   *
   * \tparam Real Scalar floating-point type.
   * \tparam Obj Callable implementing \f$f(x)\f$.
   * \tparam Grad Callable implementing \f$g=\nabla f(x)\f$.
   * \tparam Hprod Callable implementing \f$hv=H(x)v\f$.
   */
  template <typename Real, typename Obj, typename Grad, typename Hprod> class CallableProblem
  {
  public:
    /**
     * \brief Construct the callable adapter.
     * \param obj Objective callable.
     * \param grad Gradient callable.
     * \param hprod Hessian-vector-product callable.
     */
    CallableProblem( Obj obj, Grad grad, Hprod hprod )
      : obj_( std::move( obj ) ), grad_( std::move( grad ) ), hprod_( std::move( hprod ) )
    {
    }

    /** \brief Evaluate the objective function at \p x. */
    Real objective( Vector<Real> const & x ) { return static_cast<Real>( obj_( x ) ); }

    /** \brief Evaluate the objective gradient at \p x. */
    void gradient( Vector<Real> const & x, Vector<Real> & g ) { grad_( x, g ); }

    /** \brief Evaluate the Hessian-vector product \f$H(x)v\f$. */
    void hprod( Vector<Real> const & x, Vector<Real> const & v, Vector<Real> & hv ) { hprod_( x, v, hv ); }

  private:
    Obj   obj_;
    Grad  grad_;
    Hprod hprod_;
  };

  /**
   * \brief Deduce callable types and construct a \ref CallableProblem.
   * \tparam Real Scalar type used by the solver.
   * \param obj Objective callable.
   * \param grad Gradient callable.
   * \param hprod Hessian-vector-product callable.
   * \return A callable problem object satisfying \ref ProblemFor.
   */
  template <typename Real = double, typename Obj, typename Grad, typename Hprod>
  auto make_problem( Obj obj, Grad grad, Hprod hprod )
  { return CallableProblem<Real, Obj, Grad, Hprod>( std::move( obj ), std::move( grad ), std::move( hprod ) ); }

  // ---------------------------------------------------------------------------
  // Status and options
  // ---------------------------------------------------------------------------

  /** \brief Termination status of the outer TRON iteration. */
  enum class Status
  {
    unknown,
    first_order,  // projected gradient below tolerance
    unbounded,    // objective seems unbounded below
    max_iter,
    max_eval,
    max_time,
    small_step,  // Cauchy point step underflowed
    neg_pred,    // non-negative predicted reduction
    direct_solver_failure,
    user  // stopped by the user callback
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
      case Status::first_order: return "first-order stationary";
      case Status::unbounded: return "unbounded";
      case Status::max_iter: return "maximum iterations";
      case Status::max_eval: return "maximum evaluations";
      case Status::max_time: return "maximum time";
      case Status::small_step: return "small step";
      case Status::neg_pred: return "non-negative predicted reduction";
      case Status::direct_solver_failure: return "direct trust-region solver failure";
      case Status::user: return "user request";
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
    max_time,
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
      case NewtonStatus::max_time: return "time limit exceeded";
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
   * Convergence is declared when this quantity is below an absolute/relative
   * tolerance and the bound violation is at machine-precision scale.
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
    // g0)||.
    Real atol                       = std::sqrt( eps );
    Real rtol                       = std::sqrt( eps );
    Real projected_newton_tolerance = Real( 1 ) / Real( 10 );

    // Budgets.
    int    max_iter                        = 1000;
    int    max_eval                        = -1;  // objective evaluations, -1 = unlimited
    int    max_projected_newton_iterations = 50;
    double max_time                        = 30.0;  // seconds

    // Trust region (TRONTrustRegion defaults).
    Real max_radius            = std::min( Real( 1 ) / std::sqrt( 2 * eps ), Real( 100 ) );
    Real acceptance_threshold  = Real( 1 ) / Real( 10000 );
    Real decrease_threshold    = Real( 1 ) / Real( 4 );
    Real increase_threshold    = Real( 3 ) / Real( 4 );
    Real large_decrease_factor = Real( 1 ) / Real( 4 );
    Real small_decrease_factor = Real( 1 ) / Real( 2 );
    Real increase_factor       = Real( 4 );

    int verbose = 0;  // print every `verbose` iterations, 0 = silent
  };

  /**
   * \brief Complete result and diagnostic counters returned by a solve.
   * \tparam Real Scalar floating-point type.
   */
  template <typename Real = double> struct Result
  {
    Vector<Real> x;
    Real         objective   = std::numeric_limits<Real>::quiet_NaN();
    Real         dual_feas   = std::numeric_limits<Real>::quiet_NaN();  // ||P(x - g) - x||
    Real         primal_feas = Real( 0 );                               // bound violation
    Real         radius      = Real( 0 );
    int          iter        = 0;
    int          obj_evals   = 0;
    int          grad_evals  = 0;
    int          hprod_evals = 0;
    double       elapsed     = 0.0;
    Status       status      = Status::unknown;

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
    template <typename Real>
    void project( Vector<Real> & z, Vector<Real> const & x, Vector<Real> const & l, Vector<Real> const & u )
    { z = x.cwiseMax( l ).cwiseMin( u ); }

    /**
     * \brief Form a feasible projected step along a search direction.
     *
     * Computes \f$s=P_{[l,u]}(x+\alpha d)-x\f$.  Consequently, \f$x+s\f$ is
     * feasible whenever the bounds are consistent.
     */
    template <typename Real> void project_step(
      Vector<Real> &       s,
      Vector<Real> const & x,
      Vector<Real> const & d,
      Vector<Real> const & l,
      Vector<Real> const & u,
      Real                 alpha )
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
     * \brief Test whether a scalar variable is numerically active at either bound.
     *
     * A relative tolerance of \f$\sqrt{\epsilon}\f$ is used, scaled by the
     * magnitude of the finite bound.  Infinite bounds are explicitly excluded
     * from the activity test.
     */
    template <typename Real> [[nodiscard]] bool is_active( Real xi, Real li, Real ui )
    {
      const Real tol      = std::sqrt( std::numeric_limits<Real>::epsilon() );
      const bool at_lower = std::isfinite( li ) && xi <= li + tol * std::max( Real( 1 ), std::abs( li ) );
      const bool at_upper = std::isfinite( ui ) && xi >= ui - tol * std::max( Real( 1 ), std::abs( ui ) );
      return at_lower || at_upper;
    }

    /**
     * \brief Apply the Hessian and evaluate the local quadratic model along a step.
     *
     * Given a Hessian operator \p hop, computes \f$Hs\f$, the linear model term
     * \f$g^Ts\f$, and
     * \f[
     *   q(s)=g^Ts+\tfrac12 s^T Hs.
     * \f]
     *
     * \return Pair `(slope,q)` with `slope = g.dot(s)`.
     */
    template <typename Real, typename Operator> std::pair<Real, Real> hs_slope_qs(
      Operator &&          hop,
      Vector<Real> const & s,
      Vector<Real> const & g,
      Vector<Real> &       hs )
    {
      hop( s, hs );
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

      Matrix<Real> symmetric_hessian = Real( 0.5 ) * ( hessian + hessian.transpose() );
      const Real   algebra_tolerance = Real( 256 ) * std::numeric_limits<Real>::epsilon();
      const Real   matrix_scale      = std::max( Real( 1 ), symmetric_hessian.cwiseAbs().maxCoeff() );

      Eigen::LDLT<Matrix<Real>> ldlt( symmetric_hessian );
      if ( ldlt.info() == Eigen::Success && ldlt.isPositive() )
      {
        Vector<Real> newton_step = ldlt.solve( rhs );
        if ( ldlt.info() == Eigen::Success && newton_step.allFinite() )
        {
          const Real residual       = ( symmetric_hessian * newton_step - rhs ).norm();
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

      EigenSolver eigensolver( symmetric_hessian );
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
   * \brief Reusable TRON solver with a matrix-free Hessian interface.
   *
   * The algorithm minimizes \f$f(x)\f$ subject to \f$l\le x\le u\f$ using a
   * trust-region Newton framework specialized to bound constraints.  Each outer
   * iteration consists of:
   *
   * 1. freezing the Hessian at the current accepted point;
   * 2. computing a feasible projected Cauchy step that guarantees model decrease;
   * 3. refining that step on the currently free variables using a projected
   *    Newton process with a direct Eigen trust-region solve;
   * 4. evaluating the trial objective and accepting/rejecting the step according
   *    to the ratio of actual to predicted reduction;
   * 5. updating the trust-region radius by TRON's interpolation rules.
   *
   * All work vectors are owned by the solver and resized together, so repeated
   * calls with a fixed dimension reuse storage.
   *
   * \tparam Real Scalar floating-point type.
   */
  template <typename Real = double> class Solver
  {
  public:
    using Vec = Vector<Real>;

    /**
     * \brief Construct a solver and allocate workspace for \p nvar variables.
     * \param nvar Problem dimension.
     * \param options Initial solver options.
     */
    explicit Solver( Eigen::Index nvar, Options<Real> options = {} ) : options_( options ) { resize( nvar ); }

    /** \brief Mutable access to the solver options. */
    [[nodiscard]] Options<Real> & options() noexcept { return options_; }
    /** \brief Read-only access to the solver options. */
    [[nodiscard]] Options<Real> const & options() const noexcept { return options_; }

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
      n_ = nvar;
      x_.resize( n_ );
      xc_.resize( n_ );
      gx_.resize( n_ );
      gpx_.resize( n_ );
      s_.resize( n_ );
      hs_.resize( n_ );
      w_.resize( n_ );
      mask_.resize( n_ );
      rhs_.resize( n_ );
      newton_direction_.resize( n_ );
      basis_.resize( n_ );
      operator_column_.resize( n_ );
      tmp_.resize( n_ );
      free_indices_.reserve( static_cast<std::size_t>( n_ ) );
    }

    /**
     * \brief Minimize an unconstrained problem.
     *
     * This convenience overload delegates to the bound-constrained method using
     * \f$(-\infty,+\infty)\f$ bounds for every variable.
     */
    template <typename Problem>
      requires ProblemFor<Problem, Real>
    Result<Real> solve( Problem & problem, Vec const & x0 )
    {
      const Real inf = std::numeric_limits<Real>::infinity();
      return solve( problem, x0, Vec::Constant( x0.size(), -inf ), Vec::Constant( x0.size(), inf ) );
    }

    /**
     * \brief Minimize a problem over a box without an iteration callback.
     * \param problem Objective/derivative provider.
     * \param x0 Initial point; it is projected onto the box before evaluation.
     * \param lower Componentwise lower bounds.
     * \param upper Componentwise upper bounds.
     * \return Final iterate, termination status, and evaluation statistics.
     */
    template <typename Problem>
      requires ProblemFor<Problem, Real>
    Result<Real> solve( Problem & problem, Vec const & x0, Vec const & lower, Vec const & upper )
    {
      return solve( problem, x0, lower, upper, []( Result<Real> const & ) { return true; } );
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
     * The supplied callback is invoked with a snapshot after initialization and
     * after each completed outer iteration.  Returning `false` terminates with
     * \ref Status::user.
     *
     * \param problem Objective/derivative provider.
     * \param x0 Initial point.
     * \param lower Componentwise lower bounds.
     * \param upper Componentwise upper bounds.
     * \param callback Callable receiving `Result<Real> const&`; return `false`
     *        to request termination.
     * \return Final iterate, diagnostic counters, and termination status.
     *
     * \note The code assumes compatible vector dimensions and consistent bounds.
     * \note `max_eval` counts objective evaluations only.
     */
    template <typename Problem, typename Callback>
      requires ProblemFor<Problem, Real>
    Result<Real> solve( Problem & problem, Vec const & x0, Vec const & lower, Vec const & upper, Callback && callback )
    {
      using Clock                 = std::chrono::steady_clock;
      const auto            start = Clock::now();
      const Options<Real> & o     = options_;
      constexpr Real        eps   = std::numeric_limits<Real>::epsilon();

      resize( x0.size() );
      obj_evals_ = grad_evals_ = hprod_evals_ = 0;

      Result<Real> res;
      detail::project( x_, x0, lower, upper );

      Real fx = eval_objective( problem, x_ );
      eval_gradient( problem, x_, gx_ );

      detail::project_step( gpx_, x_, gx_, lower, upper, Real( -1 ) );
      Real pi_x   = gpx_.norm();
      Real primal = bound_violation( x_, lower, upper );

      const Real tol  = o.atol + o.rtol * pi_x;
      const Real fmin = std::min( Real( -1 ), fx ) / eps;

      radius_                    = std::min( std::max( Real( 1 ), pi_x / Real( 10 ) ), o.max_radius );
      Real alpha_c               = Real( 1 );
      ratio_                     = Real( 0 );
      quad_min_                  = Real( 0 );
      int          num_success   = 0;
      int          iter          = 0;
      NewtonStatus newton_status = NewtonStatus::unknown;

      auto elapsed = [&] { return std::chrono::duration<double>( Clock::now() - start ).count(); };
      auto fill    = [&]( Status st ) -> Result<Real> &
      {
        res.x           = x_;
        res.objective   = fx;
        res.dual_feas   = pi_x;
        res.primal_feas = primal;
        res.radius      = radius_;
        res.iter        = iter;
        res.obj_evals   = obj_evals_;
        res.grad_evals  = grad_evals_;
        res.hprod_evals = hprod_evals_;
        res.elapsed     = elapsed();
        res.status      = st;
        return res;
      };
      auto current_status = [&]
      {
        if ( pi_x <= tol && primal <= std::sqrt( eps ) ) return Status::first_order;
        if ( fx < fmin ) return Status::unbounded;
        if ( o.max_eval >= 0 && obj_evals_ >= o.max_eval ) return Status::max_eval;
        if ( iter >= o.max_iter ) return Status::max_iter;
        if ( elapsed() >= o.max_time ) return Status::max_time;
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
          double( radius_ ),
          "-",
          "-" );
      }

      Status status = current_status();
      if ( status != Status::unknown ) return fill( status );
      if ( !callback( fill( Status::unknown ) ) ) return fill( Status::user );

      while ( true )
      {
        xc_              = x_;  // Hessian is frozen at xc_ for the whole iteration
        const Real fc    = fx;
        const Real delta = radius_;

        auto hop = [&]( Vec const & v, Vec & hv ) { eval_hprod( problem, xc_, v, hv ); };

        if ( !cauchy( hop, x_, gx_, delta, alpha_c, lower, upper ) ) return fill( Status::small_step );

        newton_status = projected_newton( hop, x_, gx_, delta, lower, upper, o.max_time - elapsed() );
        if ( newton_status == NewtonStatus::direct_solver_failure )
        {
          x_ = xc_;
          return fill( Status::direct_solver_failure );
        }

        const Real slope   = gx_.dot( s_ );
        const Real qs      = Real( 0.5 ) * s_.dot( hs_ ) + slope;
        const Real f_trial = eval_objective( problem, x_ );

        // Predicted / actual reduction, guarded against cancellation.
        const Real guard = std::max( Real( 1 ), std::abs( fc ) ) * Real( 10 ) * eps;
        const Real pred  = qs - guard;
        const Real ared  = f_trial - fc - guard;
        if ( pred >= Real( 0 ) )
        {
          fx = fc;
          x_ = xc_;
          return fill( Status::neg_pred );
        }
        ratio_ = ared / pred;

        // Quadratic interpolation factor used by the TRON radius update.
        const Real gamma = f_trial - fc - slope;
        quad_min_        = gamma <= Real( 0 ) ? o.increase_factor
                                              : std::max( o.large_decrease_factor, -slope / ( Real( 2 ) * gamma ) );

        if ( ratio_ >= o.acceptance_threshold )
        {
          ++num_success;
          fx = f_trial;
          eval_gradient( problem, x_, gx_ );
          detail::project_step( gpx_, x_, gx_, lower, upper, Real( -1 ) );
          pi_x = gpx_.norm();
        }
        else
        {
          fx = fc;
          x_ = xc_;
        }

        primal = bound_violation( x_, lower, upper );
        ++iter;

        if ( o.verbose > 0 && iter % o.verbose == 0 )
        {
          std::printf(
            "%6d  %14.7e  %10.3e  %10.3e  %10.3e  %s\n",
            iter,
            double( fx ),
            double( pi_x ),
            double( delta ),
            double( ratio_ ),
            to_string( newton_status ).data() );
        }

        const Real s_norm = s_.norm();
        if ( num_success == 0 ) radius_ = std::min( delta, s_norm );
        update_radius( s_norm );

        status = current_status();
        if ( status != Status::unknown ) break;
        if ( !callback( fill( Status::unknown ) ) )
        {
          status = Status::user;
          break;
        }
      }

      fill( status );
      if ( o.verbose > 0 )
      {
        std::printf(
          "%6d  %14.7e  %10.3e  %10.3e  -> %s\n",
          iter,
          double( fx ),
          double( pi_x ),
          double( radius_ ),
          to_string( status ).data() );
      }
      return res;
    }

  private:
    // --- objective / derivative wrappers -------------------------------------
    /** \brief Evaluate the objective and increment the objective counter. */
    template <typename Problem> Real eval_objective( Problem & p, Vec const & x )
    {
      ++obj_evals_;
      return p.objective( x );
    }

    /** \brief Evaluate the gradient and increment the gradient counter. */
    template <typename Problem> void eval_gradient( Problem & p, Vec const & x, Vec & g )
    {
      ++grad_evals_;
      p.gradient( x, g );
    }

    /** \brief Evaluate a Hessian-vector product and increment its counter. */
    template <typename Problem> void eval_hprod( Problem & p, Vec const & x, Vec const & v, Vec & hv )
    {
      ++hprod_evals_;
      p.hprod( x, v, hv );
    }

    /**
     * \brief Compute the Euclidean norm of componentwise box infeasibility.
     *
     * The violation vector is
     * \f$\max\{l-x,\;x-u,\;0\}\f$ componentwise.
     */
    [[nodiscard]] static Real bound_violation( Vec const & x, Vec const & l, Vec const & u )
    { return ( l - x ).cwiseMax( x - u ).cwiseMax( Real( 0 ) ).norm(); }

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
      const Options<Real> & o = options_;
      const Real            a = quad_min_;
      if ( ratio_ <= o.acceptance_threshold )
      {
        radius_ = std::min( std::max( a, o.large_decrease_factor ) * step_norm, o.small_decrease_factor * radius_ );
      }
      else if ( ratio_ < o.decrease_threshold )
      {
        radius_ = std::max(
          o.large_decrease_factor * radius_,
          std::min( a * step_norm, o.small_decrease_factor * radius_ ) );
      }
      else if ( ratio_ < o.increase_threshold )
      {
        radius_ = std::max( o.large_decrease_factor * radius_, std::min( a * step_norm, o.increase_factor * radius_ ) );
      }
      else
      {
        radius_ = std::max( radius_, std::min( a * step_norm, o.increase_factor * radius_ ) );
      }
      radius_ = std::min( radius_, o.max_radius );
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
    template <typename Operator>
    bool cauchy( Operator && hop, Vec const & x, Vec const & g, Real delta, Real & alpha, Vec const & l, Vec const & u )
    {
      const Options<Real> & o = options_;

      tmp_               = -g;
      const Real brk_max = detail::breakpoints( x, tmp_, l, u ).max;

      s_.setZero();
      hs_.setZero();

      detail::project_step( s_, x, g, l, u, -alpha );

      bool interpolate;
      if ( s_.norm() > o.mu_1 * delta ) { interpolate = true; }
      else
      {
        const auto [slope, qs] = detail::hs_slope_qs( hop, s_, g, hs_ );
        interpolate            = qs >= o.mu_0 * slope;
      }

      if ( interpolate )
      {
        const Real alpha_min = std::sqrt( std::numeric_limits<Real>::denorm_min() );
        bool       search    = true;
        while ( search )
        {
          alpha /= o.sigma;
          detail::project_step( s_, x, g, l, u, -alpha );
          if ( s_.norm() <= o.mu_1 * delta )
          {
            const auto [slope, qs] = detail::hs_slope_qs( hop, s_, g, hs_ );
            search                 = qs >= o.mu_0 * slope;
          }
          if ( alpha < alpha_min ) return false;
        }
      }
      else
      {
        Real alpha_ok = alpha;
        bool search   = true;
        while ( search && alpha <= brk_max )
        {
          alpha *= o.sigma;
          detail::project_step( s_, x, g, l, u, -alpha );
          if ( s_.norm() <= o.mu_1 * delta )
          {
            const auto [slope, qs] = detail::hs_slope_qs( hop, s_, g, hs_ );
            if ( qs <= o.mu_0 * slope ) alpha_ok = alpha;
          }
          else
          {
            search = false;
          }
        }
        alpha = alpha_ok;
        detail::project_step( s_, x, g, l, u, -alpha );
        detail::hs_slope_qs( hop, s_, g, hs_ );
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
    template <typename Operator> void projected_line_search(
      Operator && hop,
      Vec &       x,
      Vec const & g,
      Vec const & d,
      Vec const & l,
      Vec const & u,
      Vec &       hw,
      Vec &       w )
    {
      const Real brk_min = detail::breakpoints( x, d, l, u ).min;
      Real       alpha   = Real( 1 );

      w.setZero();
      hw.setZero();

      bool search = true;
      while ( search && alpha > brk_min )
      {
        detail::project_step( w, x, d, l, u, alpha );
        const auto [slope, qs] = detail::hs_slope_qs( hop, w, g, hw );
        if ( qs <= options_.mu_0 * slope ) { search = false; }
        else
        {
          alpha /= Real( 2 );
        }
      }
      if ( alpha < Real( 1 ) && alpha < brk_min )
      {
        alpha = brk_min;
        detail::project_step( w, x, d, l, u, alpha );
        detail::hs_slope_qs( hop, w, g, hw );
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
     * direct solution reaches the trust-region boundary, the configured
     * projected-Newton iteration limit is met, or the remaining time budget is
     * exhausted.
     *
     * \param hop Hessian-vector-product operator with Hessian frozen at the outer
     *        iteration base point.
     * \param x Current trial point; modified in place.
     * \param g Gradient at the outer iteration base point.
     * \param delta Trust-region radius supplied to each direct reduced solve.
     * \param l Lower bounds.
     * \param u Upper bounds.
     * \param max_time Remaining wall-clock budget for this refinement.
     * \return Inner iteration termination reason.
     */
    template <typename Operator> NewtonStatus projected_newton(
      Operator && hop,
      Vec &       x,
      Vec const & g,
      Real        delta,
      Vec const & l,
      Vec const & u,
      double      max_time )
    {
      using Clock                 = std::chrono::steady_clock;
      const auto            start = Clock::now();
      const Options<Real> & o     = options_;

      hop( s_, hs_ );
      x += s_;
      detail::project( x, x, l, u );

      NewtonStatus status = NewtonStatus::unknown;
      int          iters  = 0;
      while ( true )
      {
        const double elapsed = std::chrono::duration<double>( Clock::now() - start ).count();
        if ( elapsed >= max_time ) return NewtonStatus::max_time;

        free_indices_.clear();
        Eigen::Index nfixed = 0;
        for ( Eigen::Index i = 0; i < n_; ++i )
        {
          const bool fixed = detail::is_active( x[i], l[i], u[i] );
          mask_[i]         = fixed ? Real( 0 ) : Real( 1 );
          nfixed += fixed ? 1 : 0;
          if ( !fixed ) free_indices_.push_back( i );
        }
        if ( nfixed == n_ ) return NewtonStatus::stationary;

        // Right-hand side: -(g + H s) on the free variables.
        rhs_              = mask_.cwiseProduct( -g );
        const Real gfnorm = rhs_.norm();
        rhs_ -= mask_.cwiseProduct( hs_ );

        const Eigen::Index nfree = static_cast<Eigen::Index>( free_indices_.size() );
        reduced_hessian_.resize( nfree, nfree );
        reduced_rhs_.resize( nfree );
        reduced_solution_.resize( nfree );
        reduced_input_.resize( nfree );
        reduced_output_.resize( nfree );

        for ( Eigen::Index i = 0; i < nfree; ++i ) reduced_rhs_[i] = rhs_[free_indices_[i]];

        basis_.setZero();
        for ( Eigen::Index j = 0; j < nfree; ++j )
        {
          const Eigen::Index variable = free_indices_[j];
          basis_[variable]            = Real( 1 );
          hop( basis_, operator_column_ );
          basis_[variable] = Real( 0 );
          if ( !operator_column_.allFinite() ) return NewtonStatus::direct_solver_failure;
          for ( Eigen::Index i = 0; i < nfree; ++i ) reduced_hessian_( i, j ) = operator_column_[free_indices_[i]];
        }
        reduced_hessian_ = ( Real( 0.5 ) * ( reduced_hessian_ + reduced_hessian_.transpose() ) ).eval();

        const NewtonStatus direct_status =
          detail::direct_trust_region( reduced_hessian_, reduced_rhs_, delta, reduced_solution_ );
        if ( direct_status == NewtonStatus::direct_solver_failure ) return direct_status;

        newton_direction_.setZero();
        for ( Eigen::Index i = 0; i < nfree; ++i ) newton_direction_[free_indices_[i]] = reduced_solution_[i];
        ++iters;

        auto reduced_hessian_action = [&]( Vec const & v, Vec & out )
        {
          for ( Eigen::Index i = 0; i < nfree; ++i ) reduced_input_[i] = v[free_indices_[i]];
          reduced_output_.noalias() = reduced_hessian_ * reduced_input_;
          out.setZero();
          for ( Eigen::Index i = 0; i < nfree; ++i ) out[free_indices_[i]] = reduced_output_[i];
        };

        // The sign flip makes rhs_ the gradient of the correction model.
        rhs_ = -rhs_;
        projected_line_search( reduced_hessian_action, x, rhs_, newton_direction_, l, u, hs_, w_ );
        s_ += w_;

        hop( s_, hs_ );
        const Real newnorm = mask_.cwiseProduct( hs_ + g ).norm();

        const double updated_elapsed = std::chrono::duration<double>( Clock::now() - start ).count();
        if ( newnorm <= o.projected_newton_tolerance * gfnorm ) { status = NewtonStatus::stationary; }
        else if ( direct_status == NewtonStatus::boundary ) { status = NewtonStatus::boundary; }
        else if ( iters >= o.max_projected_newton_iterations ) { status = NewtonStatus::max_iter; }
        else if ( updated_elapsed >= max_time ) { status = NewtonStatus::max_time; }
        if ( status != NewtonStatus::unknown ) return status;
      }
    }

    Options<Real> options_{};
    Eigen::Index  n_ = 0;

    Vec x_, xc_, gx_, gpx_, s_, hs_, w_, tmp_;
    Vec mask_, rhs_, newton_direction_, basis_, operator_column_;
    Vec reduced_rhs_, reduced_solution_, reduced_input_, reduced_output_;

    Matrix<Real>              reduced_hessian_;
    std::vector<Eigen::Index> free_indices_;

    Real radius_   = Real( 1 );
    Real ratio_    = Real( 0 );
    Real quad_min_ = Real( 0 );

    int obj_evals_   = 0;
    int grad_evals_  = 0;
    int hprod_evals_ = 0;
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
    Vector<Real> const &  x0,
    Vector<Real> const &  lower,
    Vector<Real> const &  upper,
    Options<Real> const & options = {} )
  {
    Solver<Real> solver( x0.size(), options );
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
  Result<Real> minimize( Problem & problem, Vector<Real> const & x0, Options<Real> const & options = {} )
  {
    Solver<Real> solver( x0.size(), options );
    return solver.solve( problem, x0 );
  }

}  // namespace Utils::TRON

#endif  // UTILS_TRON_DOT_HH
