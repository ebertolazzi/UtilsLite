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

/**
 * \file Utils_minimize_BBOX_TRON.hh
 * \brief Header-only C++20 implementation of the TRON algorithm for smooth
 *        bound-constrained nonlinear minimization.
 *
 * \details
 * This file implements a matrix-free variant of TRON (Trust Region Newton)
 * for problems of the form
 * \f[
 *   \min_{x\in\mathbb{R}^n} f(x)
 *   \qquad\text{subject to}\qquad
 *   \ell \le x \le u,
 * \f]
 * where the inequalities are interpreted componentwise, \f$f\f$ is assumed
 * twice continuously differentiable in a neighbourhood of the feasible box,
 * and the Hessian is accessed through Hessian-vector products.
 *
 * \section tron_algorithm_overview Algorithmic overview
 *
 * At an outer iterate \f$x_k\f$, TRON builds the quadratic model
 * \f[
 *   m_k(s)
 *   = f(x_k) + g_k^T s + \frac12 s^T H_k s,
 *   \qquad
 *   g_k = \nabla f(x_k),
 *   \qquad
 *   H_k = \nabla^2 f(x_k),
 * \f]
 * and approximately minimizes it subject to both the box constraints and a
 * trust-region constraint,
 * \f[
 *   \ell \le x_k+s \le u,
 *   \qquad
 *   \|s\|_2 \le \Delta_k.
 * \f]
 * The implementation follows the classical TRON decomposition into a
 * projected Cauchy phase and a projected Newton phase on an identified face.
 *
 * \subsection tron_projection Projection and stationarity measure
 *
 * The Euclidean projection onto the box is
 * \f[
 *   P_{[\ell,u]}(z)_i = \min\{u_i,\max\{\ell_i,z_i\}\}.
 * \f]
 * First-order stationarity is measured by the projected-gradient mapping
 * \f[
 *   G_P(x) = P_{[\ell,u]}(x-\nabla f(x)) - x,
 * \f]
 * and the stopping test uses \f$\|G_P(x_k)\|_2\f$.  The initial norm is also
 * used to scale the absolute/relative termination tolerance and to initialize
 * the trust-region radius.
 *
 * \subsection tron_cauchy Projected Cauchy step
 *
 * The first trial step follows the projected negative-gradient path
 * \f[
 *   s(\alpha)
 *   = P_{[\ell,u]}(x_k-\alpha g_k)-x_k.
 * \f]
 * The scalar \f$\alpha\f$ is adapted geometrically.  The step must satisfy
 * both a trust-region size condition and a sufficient decrease condition for
 * the quadratic model.  If the current \f$\alpha\f$ is too aggressive it is
 * reduced; otherwise it may be enlarged until a breakpoint of the projected
 * path or the trust-region/model conditions prevent further expansion.
 * Breakpoints are values of \f$\alpha\f$ at which one component reaches a
 * lower or upper bound.
 *
 * \subsection tron_face Projected Newton refinement on the active face
 *
 * After the Cauchy point, variables sufficiently close to a bound are marked
 * active.  The remaining free variables define a face of the feasible box.
 * On that face, the Newton correction approximately solves the reduced system
 * \f[
 *   H_{FF} p_F = -\bigl(g_k + H_k s\bigr)_F,
 * \f]
 * where \f$F\f$ denotes the free-variable index set and \f$s\f$ is the step
 * accumulated from \f$x_k\f$.  The reduced Hessian is never formed explicitly:
 * masking before and after a Hessian-vector product realizes the operator
 * \f$H_{FF}\f$ in matrix-free form.
 *
 * The reduced Newton system is solved by a Steihaug-Toint truncated conjugate
 * gradient iteration.  CG terminates when one of the following occurs:
 * - the reduced residual is sufficiently small;
 * - the trust-region boundary is reached;
 * - non-positive curvature is detected;
 * - the configured CG iteration limit is reached;
 * - a non-finite Hessian-vector product is encountered.
 *
 * When the unconstrained CG update would leave the trust region, the method
 * computes \f$\tau\ge0\f$ such that
 * \f[
 *   \|p+\tau d\|_2 = \Delta_k,
 * \f]
 * and returns the boundary point.  The same boundary construction is used
 * when negative curvature is detected.
 *
 * A projected line search then maps the reduced Newton direction back into
 * the feasible box and enforces sufficient decrease of the quadratic model.
 * The active set is recomputed after every face step, allowing the projected
 * Newton phase to move from one face of the box to another.
 *
 * \subsection tron_acceptance Trial acceptance and trust-region update
 *
 * Let \f$s_k\f$ be the final trial step.  The predicted model change is
 * \f[
 *   \operatorname{pred}_k
 *   = g_k^T s_k + \frac12 s_k^T H_k s_k,
 * \f]
 * while the actual objective change is
 * \f[
 *   \operatorname{ared}_k = f(x_k+s_k)-f(x_k).
 * \f]
 * The trust-region ratio is therefore
 * \f[
 *   \rho_k = \frac{\operatorname{ared}_k}{\operatorname{pred}_k}.
 * \f]
 * Both quantities are negative for a successful descent step.  The code adds
 * a small roundoff safeguard before forming the ratio.  If function-value
 * differences are dominated by floating-point cancellation, a symmetric
 * gradient-based estimate of the actual change is used instead.
 *
 * A trial point is accepted when \f$\rho_k\f$ exceeds the configured
 * acceptance threshold.  The trust-region radius is then reduced, retained,
 * or enlarged according to the usual TRON thresholds and safeguarded
 * interpolation factor.  Rejected steps leave \f$x_k\f$ unchanged but still
 * update the radius.
 *
 * \subsection tron_stopping Termination
 *
 * The solver terminates when one of the following conditions is detected:
 * - projected-gradient convergence;
 * - maximum outer iterations;
 * - maximum objective evaluations;
 * - an objective value interpreted as unbounded below;
 * - collapse of the projected Cauchy step;
 * - a non-descent quadratic model;
 * - non-finite objective, gradient, or Hessian-vector information.
 *
 * \section tron_interfaces Interfaces
 *
 * The low-level kernel accepts three callbacks:
 * \code{.cpp}
 * Scalar value(Vector const& x);
 * void gradient(Vector const& x, Vector& g);
 * void hessian_vector(Vector const& x, Vector const& v, Vector& Hv);
 * \endcode
 * The public \c Utils::Minimize_BBOX_TRON wrapper additionally provides dense
 * Hessian and problem-object interfaces.  Dense Hessians are converted to
 * Hessian-vector products internally.
 *
 * \section tron_reference Reference
 *
 * Chih-Jen Lin and Jorge J. More, "Newton's Method for Large Bound-Constrained
 * Optimization Problems", SIAM Journal on Optimization, 9(4), 1100-1127,
 * 1999. DOI: 10.1137/S1052623498345075.
 *
 * This implementation also follows the trust-region update conventions used
 * by SolverTools.jl and the Steihaug-Toint truncated-CG structure used by
 * Krylov.jl / JSOSolvers.jl.
 *
 * \note Requires Eigen 5 or newer and C++20.
 */

#pragma once

#ifndef UTILS_MINIMIZE_BBOX_TRON_DOT_HH
#define UTILS_MINIMIZE_BBOX_TRON_DOT_HH

#include "Utils_eigen.hh"
#include "Utils_minimize_BBOX_Common.hh"

#if EIGEN_MAJOR_VERSION < 5
#error "Utils::Minimize_BBOX_TRON requires Eigen 5 or newer"
#endif

namespace Utils::TRON_details
{

  template <std::floating_point Scalar> using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  /** \brief Final termination code returned by the TRON solver. */
  enum class Status
  {
    converged,
    max_iterations,
    max_function_evaluations,
    unbounded,
    small_step,
    non_descent_model,
    non_finite_objective,
    non_finite_gradient,
    non_finite_hessian
  };

  /**
   * \brief Convert a solver termination code to a human-readable message.
   * \param status Termination code to convert.
   * \return A static textual description of \p status.
   * \note The function is \c constexpr and performs no allocation.
   */
  [[nodiscard]] constexpr std::string_view to_string( Status status ) noexcept
  {
    switch ( status )
    {
      case Status::converged: return "converged";
      case Status::max_iterations: return "maximum number of iterations";
      case Status::max_function_evaluations: return "maximum number of function evaluations";
      case Status::unbounded: return "objective appears unbounded below";
      case Status::small_step: return "Cauchy step is too small";
      case Status::non_descent_model: return "quadratic model does not predict descent";
      case Status::non_finite_objective: return "objective returned a non-finite value";
      case Status::non_finite_gradient: return "gradient contains a non-finite value";
      case Status::non_finite_hessian: return "Hessian-vector product contains a non-finite value";
    }
    return "unknown";
  }

  /**
   * \brief Numerical parameters controlling TRON, projected Newton, and CG.
   *
   * The defaults reproduce the constants used by the reference Julia
   * implementations from which this port was derived.  Iteration and function
   * evaluation limits are deliberately independent; no timing criterion is
   * used by this implementation.
   *
   * \tparam Scalar Floating-point scalar type.
   */
  template <std::floating_point Scalar> struct Options
  {
    // TRON parameters from JSOSolvers.jl.
    Scalar sufficient_decrease{ Scalar( 0.01 ) };  // mu_0
    Scalar cauchy_radius_factor{ Scalar( 1 ) };    // mu_1
    Scalar cauchy_step_factor{ Scalar( 10 ) };     // sigma

    // TRONTrustRegion parameters from SolverTools.jl.
    Scalar max_radius{ Scalar( 100 ) };
    Scalar acceptance_threshold{ Scalar( 1.0e-4 ) };
    Scalar decrease_threshold{ Scalar( 0.25 ) };
    Scalar increase_threshold{ Scalar( 0.75 ) };
    Scalar large_decrease_factor{ Scalar( 0.25 ) };
    Scalar small_decrease_factor{ Scalar( 0.5 ) };
    Scalar increase_factor{ Scalar( 4 ) };

    Scalar absolute_tolerance{ Scalar( 1e-9 ) };
    Scalar relative_tolerance{ Scalar( 1e-9 ) };
    Scalar cg_tolerance{ Scalar( 0.1 ) };
    Scalar active_absolute_tolerance{ std::sqrt( std::numeric_limits<Scalar>::epsilon() ) };
    Scalar active_relative_tolerance{ std::sqrt( std::numeric_limits<Scalar>::epsilon() ) };

    std::size_t max_iterations{ std::numeric_limits<std::size_t>::max() };
    std::size_t max_function_evaluations{ std::numeric_limits<std::size_t>::max() };
    std::size_t max_projected_newton_iterations{ 50 };
    // Zero selects 2*n, matching Krylov.cg!'s default in the Julia version.
    std::size_t max_cg_iterations{ 0 };

    // Unified setters (same name across all BBOX solvers)
    /**
     * \brief Set both absolute and relative projected-gradient tolerances.
     * \param tol Non-negative tolerance assigned to both stopping parameters.
     */
    void set_tolerances( Scalar tol ) { absolute_tolerance = relative_tolerance = tol; }
    /**
     * \brief Set the maximum number of outer TRON iterations.
     * \param n Maximum number of attempted trust-region iterations.
     */
    void set_max_iterations( std::size_t n ) { max_iterations = n; }
  };

  /**
   * \brief Complete numerical result and accounting information.
   * \tparam Scalar Floating-point scalar type.
   *
   * Counters include accepted outer iterations, objective/gradient evaluations,
   * Hessian-vector products, and the cumulative number of inner CG iterations.
   */
  template <std::floating_point Scalar> struct Result
  {
    Vector<Scalar> x;
    Status         status{ Status::max_iterations };
    Scalar         objective{ std::numeric_limits<Scalar>::quiet_NaN() };
    Scalar         projected_gradient_norm{ std::numeric_limits<Scalar>::infinity() };
    Scalar         trust_region_radius{ Scalar( 0 ) };
    Scalar         last_ratio{ Scalar( 0 ) };
    std::size_t    iterations{ 0 };
    std::size_t    accepted_iterations{ 0 };
    std::size_t    function_evaluations{ 0 };
    std::size_t    gradient_evaluations{ 0 };
    std::size_t    hessian_vector_evaluations{ 0 };
    std::size_t    cg_iterations{ 0 };

    /**
     * \brief Test whether the solver satisfied its convergence criterion.
     * \return \c true only when \c status is \c Status::converged.
     */
    [[nodiscard]] bool success() const noexcept { return status == Status::converged; }
  };

  namespace detail
  {

    /**
     * \brief Check that every component of a vector is finite.
     * \param v Vector to inspect.
     * \return \c true when no component is NaN or infinite.
     */
    template <std::floating_point Scalar> [[nodiscard]] bool all_finite( Vector<Scalar> const & v )
    { return v.array().isFinite().all(); }

    /**
     * \brief Project a vector onto the closed box \f$[\ell,u]\f$ in place.
     *
     * Each component is replaced by
     * \f$\min(u_i,\max(\ell_i,x_i))\f$.
     *
     * \param[in,out] x Vector to project.
     * \param lower Componentwise lower bounds.
     * \param upper Componentwise upper bounds.
     */
    template <std::floating_point Scalar>
    void project_in_place( Vector<Scalar> & x, Vector<Scalar> const & lower, Vector<Scalar> const & upper )
    { x = x.cwiseMax( lower ).cwiseMin( upper ); }

    /**
     * \brief Build a feasible projected displacement along a direction.
     *
     * Computes
     * \f[
     *   s = P_{[\ell,u]}(x+\alpha d)-x.
     * \f]
     * The returned displacement is therefore feasible by construction even
     * when the unprojected point crosses one or more bounds.
     *
     * \param[out] step Projected displacement.
     * \param x Current feasible point.
     * \param direction Search direction.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \param alpha Step length applied before projection.
     */
    template <std::floating_point Scalar> void project_step(
      Vector<Scalar> &       step,
      Vector<Scalar> const & x,
      Vector<Scalar> const & direction,
      Vector<Scalar> const & lower,
      Vector<Scalar> const & upper,
      Scalar                 alpha )
    { step = ( x + alpha * direction ).cwiseMax( lower ).cwiseMin( upper ) - x; }

    /**
     * \brief Compute the norm of the projected-gradient stationarity mapping.
     *
     * The method evaluates
     * \f[
     *   \|P_{[\ell,u]}(x-g)-x\|_2,
     * \f]
     * where \f$g=\nabla f(x)\f$.  This quantity vanishes exactly at a
     * first-order stationary point for box constraints.
     *
     * \param x Current point.
     * \param gradient Gradient at \p x.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \param[out] work Workspace receiving the projected displacement.
     * \return Euclidean norm of the projected-gradient mapping.
     */
    template <std::floating_point Scalar> [[nodiscard]] Scalar projected_gradient_norm(
      Vector<Scalar> const & x,
      Vector<Scalar> const & gradient,
      Vector<Scalar> const & lower,
      Vector<Scalar> const & upper,
      Vector<Scalar> &       work )
    {
      project_step( work, x, gradient, lower, upper, Scalar( -1 ) );
      return work.stableNorm();
    }

    /**
     * \brief Summary of positive breakpoints along a projected search path.
     *
     * A breakpoint is a step length at which one variable first reaches a
     * finite lower or upper bound while moving along a direction.
     */
    template <std::floating_point Scalar> struct Breakpoints
    {
      std::size_t count{ 0 };
      Scalar      minimum{ std::numeric_limits<Scalar>::infinity() };
      Scalar      maximum{ Scalar( 0 ) };
    };

    /**
     * \brief Determine the positive bound-intersection step lengths.
     *
     * For each moving variable the routine solves either
     * \f$ x_i+\alpha d_i=u_i \f$ or
     * \f$ x_i+\alpha d_i=\ell_i \f$, retaining the smallest and largest
     * positive breakpoint encountered.
     *
     * \return Number, minimum, and maximum of the valid breakpoints.
     */
    template <std::floating_point Scalar> [[nodiscard]] Breakpoints<Scalar> breakpoints(
      Vector<Scalar> const & x,
      Vector<Scalar> const & direction,
      Vector<Scalar> const & lower,
      Vector<Scalar> const & upper )
    {
      Breakpoints<Scalar> result;
      for ( Eigen::Index i = 0; i < x.size(); ++i )
      {
        Scalar alpha{};
        bool   exists = false;
        if ( direction[i] > Scalar( 0 ) && x[i] < upper[i] )
        {
          alpha  = ( upper[i] - x[i] ) / direction[i];
          exists = true;
        }
        else if ( direction[i] < Scalar( 0 ) && x[i] > lower[i] )
        {
          alpha  = ( lower[i] - x[i] ) / direction[i];
          exists = true;
        }
        if ( exists )
        {
          ++result.count;
          result.minimum = std::min( result.minimum, alpha );
          result.maximum = std::max( result.maximum, alpha );
        }
      }
      return result;
    }

    /**
     * \brief Identify variables belonging to the current active face.
     *
     * A variable is marked fixed when it is exactly fixed by equal bounds or
     * lies within a scale-aware tolerance of either bound.  For a finite box
     * interval the tolerance is the minimum of an absolute threshold and a
     * relative fraction of the interval width.
     *
     * \param x Point whose active set is identified.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \param relative_tolerance Relative active-set tolerance.
     * \param absolute_tolerance Absolute active-set tolerance.
     * \param[out] fixed Boolean active/fixed mask.
     */
    template <std::floating_point Scalar> void active_mask(
      Vector<Scalar> const &                  x,
      Vector<Scalar> const &                  lower,
      Vector<Scalar> const &                  upper,
      Scalar                                  relative_tolerance,
      Scalar                                  absolute_tolerance,
      Eigen::Array<bool, Eigen::Dynamic, 1> & fixed )
    {
      for ( Eigen::Index i = 0; i < x.size(); ++i )
      {
        Scalar delta = absolute_tolerance;
        if ( std::isfinite( lower[i] ) && std::isfinite( upper[i] ) && lower[i] < upper[i] )
        {
          delta = std::min( relative_tolerance * ( upper[i] - lower[i] ), absolute_tolerance );
        }
        fixed[i] = ( lower[i] == x[i] && x[i] == upper[i] ) || x[i] <= lower[i] + delta || x[i] >= upper[i] - delta;
      }
    }

    /**
     * \brief Evaluate the quadratic model change for a displacement.
     *
     * Given an operator representing the Hessian, computes
     * \f[
     *   \text{slope}=g^Ts,
     *   \qquad
     *   q(s)=g^Ts+\frac12 s^THs.
     * \f]
     *
     * \return \c false if the operator result, slope, or model value is
     * non-finite; \c true otherwise.
     */
    template <std::floating_point Scalar, class ApplyOperator> [[nodiscard]] bool quadratic_model(
      ApplyOperator &&       apply,
      Vector<Scalar> const & step,
      Vector<Scalar> const & gradient,
      Vector<Scalar> &       operator_step,
      Scalar &               slope,
      Scalar &               value )
    {
      std::invoke( apply, step, operator_step );
      if ( !all_finite( operator_step ) ) return false;
      slope = gradient.dot( step );
      value = Scalar( 0.5 ) * step.dot( operator_step ) + slope;
      return std::isfinite( slope ) && std::isfinite( value );
    }

    /** \brief Internal termination code for projected Cauchy-step construction. */
    enum class CauchyStatus
    {
      success,
      small_step,
      non_finite_hessian
    };

    /**
     * \brief Compute a projected Cauchy step satisfying TRON safeguards.
     *
     * Starting from the projected negative-gradient path
     * \f$s(\alpha)=P(x-\alpha g)-x\f$, the method adapts \p alpha by the
     * multiplicative factor \c cauchy_step_factor.  It contracts the step when
     * the trust-region size or sufficient model-decrease test fails and expands
     * it when the current step remains safely acceptable.  Expansion is bounded
     * by the largest projection breakpoint.
     *
     * \param x Base point.
     * \param gradient Gradient at the base point.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \param radius Current trust-region radius.
     * \param[in,out] alpha Persistent Cauchy path parameter reused across outer iterations.
     * \param options Solver parameters.
     * \param apply_hessian Hessian-vector product operator at the base point.
     * \param[out] step Selected projected Cauchy displacement.
     * \param[out] hessian_step Hessian applied to \p step when evaluated.
     * \param[out] direction Workspace storing the negative gradient direction.
     */
    template <std::floating_point Scalar, class ApplyHessian> [[nodiscard]] CauchyStatus cauchy_step(
      Vector<Scalar> const &  x,
      Vector<Scalar> const &  gradient,
      Vector<Scalar> const &  lower,
      Vector<Scalar> const &  upper,
      Scalar                  radius,
      Scalar &                alpha,
      Options<Scalar> const & options,
      ApplyHessian &&         apply_hessian,
      Vector<Scalar> &        step,
      Vector<Scalar> &        hessian_step,
      Vector<Scalar> &        direction )
    {
      direction         = -gradient;
      auto const points = breakpoints( x, direction, lower, upper );

      project_step( step, x, gradient, lower, upper, -alpha );
      Scalar slope{};
      Scalar model{};
      Scalar step_norm   = step.stableNorm();
      bool   interpolate = step_norm > options.cauchy_radius_factor * radius;
      if ( !interpolate )
      {
        if ( !quadratic_model( apply_hessian, step, gradient, hessian_step, slope, model ) )
        {
          return CauchyStatus::non_finite_hessian;
        }
        interpolate = model >= options.sufficient_decrease * slope;
      }

      if ( interpolate )
      {
        auto const minimum_alpha = std::sqrt( std::nextafter( Scalar( 0 ), std::numeric_limits<Scalar>::infinity() ) );
        for ( ;; )
        {
          alpha /= options.cauchy_step_factor;
          project_step( step, x, gradient, lower, upper, -alpha );
          step_norm   = step.stableNorm();
          bool search = step_norm > options.cauchy_radius_factor * radius;
          if ( !search )
          {
            if ( !quadratic_model( apply_hessian, step, gradient, hessian_step, slope, model ) )
            {
              return CauchyStatus::non_finite_hessian;
            }
            search = model >= options.sufficient_decrease * slope;
          }
          if ( alpha < minimum_alpha ) return CauchyStatus::small_step;
          if ( !search ) break;
        }
      }
      else
      {
        Scalar successful_alpha = alpha;
        bool   search           = true;
        while ( search && alpha <= points.maximum )
        {
          alpha *= options.cauchy_step_factor;
          project_step( step, x, gradient, lower, upper, -alpha );
          step_norm = step.stableNorm();
          if ( step_norm <= options.cauchy_radius_factor * radius )
          {
            if ( !quadratic_model( apply_hessian, step, gradient, hessian_step, slope, model ) )
            {
              return CauchyStatus::non_finite_hessian;
            }
            if ( model <= options.sufficient_decrease * slope ) successful_alpha = alpha;
          }
          else
          {
            search = false;
          }
        }
        alpha = successful_alpha;
        project_step( step, x, gradient, lower, upper, -alpha );
      }
      return CauchyStatus::success;
    }

    /** \brief Internal termination code for Steihaug-Toint truncated CG. */
    enum class CGStatus
    {
      converged,
      boundary,
      negative_curvature,
      iteration_limit,
      non_finite
    };

    /** \brief Status and iteration count returned by the inner CG solver. */
    template <std::floating_point Scalar> struct CGResult
    {
      CGStatus    status{ CGStatus::converged };
      std::size_t iterations{ 0 };
    };

    /**
     * \brief Solve a trust-region linear system by truncated conjugate gradient.
     *
     * The method approximately solves \f$A p=b\f$ from the zero initial guess,
     * while enforcing \f$\|p\|_2\le\Delta\f$.  It is the Steihaug-Toint variant:
     * positive-curvature CG steps are taken while they remain inside the ball;
     * if a step would cross the boundary, the exact intersection with the sphere
     * is returned.  Non-positive curvature also causes termination at the
     * forward trust-region boundary.
     *
     * \param apply Linear operator \f$v\mapsto Av\f$.
     * \param rhs Right-hand side \f$b\f$.
     * \param radius Trust-region radius \f$\Delta\f$.
     * \param relative_tolerance Relative residual stopping tolerance.
     * \param iteration_limit Maximum number of CG iterations.
     * \param[out] solution Computed truncated-CG correction.
     * \param[out] residual Residual workspace/final residual.
     * \param[out] direction CG search-direction workspace.
     * \param[out] operator_direction Workspace for \f$Ad\f$.
     * \return Inner-solver status and number of iterations performed.
     */
    template <std::floating_point Scalar, class ApplyOperator> [[nodiscard]] CGResult<Scalar> truncated_cg(
      ApplyOperator &&       apply,
      Vector<Scalar> const & rhs,
      Scalar                 radius,
      Scalar                 relative_tolerance,
      std::size_t            iteration_limit,
      Vector<Scalar> &       solution,
      Vector<Scalar> &       residual,
      Vector<Scalar> &       direction,
      Vector<Scalar> &       operator_direction )
    {
      CGResult<Scalar> result;
      solution.setZero();
      residual  = rhs;
      direction = residual;
      Scalar rr = residual.squaredNorm();
      if ( !std::isfinite( rr ) )
      {
        result.status = CGStatus::non_finite;
        return result;
      }
      Scalar const target = relative_tolerance * std::sqrt( rr );
      if ( std::sqrt( rr ) <= target || rr == Scalar( 0 ) ) return result;
      Vector<Scalar> candidate( solution.size() );

      for ( std::size_t k = 0; k < iteration_limit; ++k )
      {
        std::invoke( apply, direction, operator_direction );
        if ( !all_finite( operator_direction ) )
        {
          result.status = CGStatus::non_finite;
          return result;
        }
        Scalar const curvature = direction.dot( operator_direction );
        if ( !std::isfinite( curvature ) )
        {
          result.status = CGStatus::non_finite;
          return result;
        }

        Scalar const a            = direction.squaredNorm();
        Scalar const b            = Scalar( 2 ) * solution.dot( direction );
        Scalar const c            = solution.squaredNorm() - radius * radius;
        Scalar const discriminant = std::max( Scalar( 0 ), b * b - Scalar( 4 ) * a * c );
        Scalar const tau          = ( -b + std::sqrt( discriminant ) ) / ( Scalar( 2 ) * a );
        Scalar       alpha        = curvature > Scalar( 0 ) ? rr / curvature : std::numeric_limits<Scalar>::infinity();
        if ( curvature <= Scalar( 0 ) || !std::isfinite( alpha ) || alpha > tau )
        {
          solution.noalias() += tau * direction;
          result.iterations = k + 1;
          result.status     = curvature <= Scalar( 0 ) ? CGStatus::negative_curvature : CGStatus::boundary;
          return result;
        }

        candidate = solution + alpha * direction;
        solution.swap( candidate );
        residual.noalias() -= alpha * operator_direction;
        Scalar const rr_new = residual.squaredNorm();
        result.iterations   = k + 1;
        if ( !std::isfinite( rr_new ) )
        {
          result.status = CGStatus::non_finite;
          return result;
        }
        if ( std::sqrt( rr_new ) <= target ) return result;
        direction = residual + ( rr_new / rr ) * direction;
        rr        = rr_new;
      }
      result.status = CGStatus::iteration_limit;
      return result;
    }

    /**
     * \brief Perform projected backtracking on the quadratic model.
     *
     * The candidate path is \f$P(x+\alpha d)\f$.  Starting from unit step,
     * \f$\alpha\f$ is halved until the quadratic model satisfies the configured
     * sufficient-decrease condition.  The first projection breakpoint is used
     * as a safeguard against needlessly shrinking below the first face change.
     *
     * \param[in,out] x Current face point; updated to the accepted projected point.
     * \param model_gradient Gradient of the local quadratic model at \p x.
     * \param direction Search direction.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \param sufficient_decrease Armijo-like model decrease coefficient.
     * \param apply_operator Reduced-Hessian operator.
     * \param[out] step Accepted projected displacement.
     * \param[out] operator_step Workspace for the operator applied to \p step.
     * \return \c false only when a non-finite quadratic-model evaluation occurs.
     */
    template <std::floating_point Scalar, class ApplyOperator> [[nodiscard]] bool projected_line_search(
      Vector<Scalar> &       x,
      Vector<Scalar> const & model_gradient,
      Vector<Scalar> const & direction,
      Vector<Scalar> const & lower,
      Vector<Scalar> const & upper,
      Scalar                 sufficient_decrease,
      ApplyOperator &&       apply_operator,
      Vector<Scalar> &       step,
      Vector<Scalar> &       operator_step )
    {
      Scalar     alpha  = Scalar( 1 );
      auto const points = breakpoints( x, direction, lower, upper );
      Scalar     slope{};
      Scalar     model{};

      while ( alpha > points.minimum )
      {
        project_step( step, x, direction, lower, upper, alpha );
        if ( !quadratic_model( apply_operator, step, model_gradient, operator_step, slope, model ) ) return false;
        if ( model <= sufficient_decrease * slope ) break;
        alpha /= Scalar( 2 );
      }
      if ( alpha < Scalar( 1 ) && alpha < points.minimum ) alpha = points.minimum;
      project_step( step, x, direction, lower, upper, alpha );
      x += step;
      return true;
    }

    /** \brief Internal termination code for projected Newton face refinement. */
    enum class ProjectedNewtonStatus
    {
      stationary,
      boundary,
      iteration_limit,
      non_finite
    };

    /**
     * \brief Refine the Cauchy point by projected Newton iterations on box faces.
     *
     * The method starts from the feasible Cauchy point \f$x_k+s\f$, identifies
     * variables near their bounds, and solves the Newton equations restricted to
     * the complementary free set.  The reduced Hessian is implemented by
     * masking vectors before and after the matrix-free Hessian action.  Each
     * reduced system is solved by truncated CG and globalized by a projected
     * model line search.  The active set is then recomputed, so the iteration
     * may continue on a different face.
     *
     * \param base_x Outer TRON base point \f$x_k\f$.
     * \param gradient Gradient at \p base_x.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \param radius Trust-region radius.
     * \param options Solver parameters.
     * \param apply_hessian Hessian-vector product at \p base_x.
     * \param[out] trial_x Final feasible trial point.
     * \param[in,out] step Accumulated displacement from \p base_x.
     * \param[out] hessian_step Hessian applied to the accumulated displacement.
     * \param[in,out] total_cg_iterations Global counter of inner CG iterations.
     * \return Reason for terminating the projected Newton phase.
     */
    template <std::floating_point Scalar, class ApplyHessian> [[nodiscard]] ProjectedNewtonStatus projected_newton(
      Vector<Scalar> const &  base_x,
      Vector<Scalar> const &  gradient,
      Vector<Scalar> const &  lower,
      Vector<Scalar> const &  upper,
      Scalar                  radius,
      Options<Scalar> const & options,
      ApplyHessian &&         apply_hessian,
      Vector<Scalar> &        trial_x,
      Vector<Scalar> &        step,
      Vector<Scalar> &        hessian_step,
      std::size_t &           total_cg_iterations )
    {
      auto const n = base_x.size();
      trial_x      = base_x + step;
      project_in_place( trial_x, lower, upper );
      std::invoke( apply_hessian, step, hessian_step );
      if ( !all_finite( hessian_step ) ) return ProjectedNewtonStatus::non_finite;

      Eigen::Array<bool, Eigen::Dynamic, 1> fixed( n );
      Vector<Scalar>                        mask( n ), rhs( n ), reduced_gradient( n ), correction( n ), line_step( n );
      Vector<Scalar> cg_residual( n ), cg_direction( n ), operator_direction( n ), operator_step( n ), masked( n ),
        temp( n );

      std::size_t face_iterations = 0;
      for ( ;; )
      {
        active_mask(
          trial_x,
          lower,
          upper,
          options.active_relative_tolerance,
          options.active_absolute_tolerance,
          fixed );
        if ( fixed.count() == n ) return ProjectedNewtonStatus::stationary;

        for ( Eigen::Index i = 0; i < n; ++i ) mask[i] = fixed[i] ? Scalar( 0 ) : Scalar( 1 );
        Scalar free_gradient_norm_squared = Scalar( 0 );
        for ( Eigen::Index i = 0; i < n; ++i )
        {
          Scalar const free_gradient = fixed[i] ? Scalar( 0 ) : -gradient[i];
          free_gradient_norm_squared += free_gradient * free_gradient;
          rhs[i]              = free_gradient - ( fixed[i] ? Scalar( 0 ) : hessian_step[i] );
          reduced_gradient[i] = -rhs[i];
        }

        auto apply_reduced_hessian = [&]( Vector<Scalar> const & v, Vector<Scalar> & out )
        {
          masked = mask.array() * v.array();
          std::invoke( apply_hessian, masked, temp );
          out = mask.array() * temp.array();
        };
        std::size_t const cg_limit = options.max_cg_iterations == 0
                                       ? static_cast<std::size_t>( std::max<Eigen::Index>( 1, 2 * n ) )
                                       : options.max_cg_iterations;
        auto const        cg       = truncated_cg(
          apply_reduced_hessian,
          rhs,
          radius,
          options.cg_tolerance,
          cg_limit,
          correction,
          cg_residual,
          cg_direction,
          operator_direction );
        total_cg_iterations += cg.iterations;
        if ( cg.status == CGStatus::non_finite ) return ProjectedNewtonStatus::non_finite;

        if ( !projected_line_search(
               trial_x,
               reduced_gradient,
               correction,
               lower,
               upper,
               options.sufficient_decrease,
               apply_reduced_hessian,
               line_step,
               operator_step ) )
        {
          return ProjectedNewtonStatus::non_finite;
        }
        step += line_step;
        std::invoke( apply_hessian, step, hessian_step );
        if ( !all_finite( hessian_step ) ) return ProjectedNewtonStatus::non_finite;

        Scalar new_norm_squared = Scalar( 0 );
        for ( Eigen::Index i = 0; i < n; ++i )
        {
          if ( !fixed[i] )
          {
            Scalar const component = hessian_step[i] + gradient[i];
            new_norm_squared += component * component;
          }
        }
        ++face_iterations;
        if ( std::sqrt( new_norm_squared ) <= options.cg_tolerance * std::sqrt( free_gradient_norm_squared ) )
        {
          return ProjectedNewtonStatus::stationary;
        }
        if ( cg.status == CGStatus::boundary ) return ProjectedNewtonStatus::boundary;
        if ( face_iterations >= options.max_projected_newton_iterations )
        {
          return ProjectedNewtonStatus::iteration_limit;
        }
      }
    }

    /**
     * \brief Validate dimensions, bounds, initial point, and numerical options.
     *
     * \throws std::invalid_argument if vector sizes are inconsistent, the
     * initial point/bounds are invalid, or any algorithmic parameter violates
     * the ordering/range assumptions required by TRON.
     */
    template <std::floating_point Scalar> void validate(
      Vector<Scalar> const &  x,
      Vector<Scalar> const &  lower,
      Vector<Scalar> const &  upper,
      Options<Scalar> const & o )
    {
      if ( x.size() == 0 || lower.size() != x.size() || upper.size() != x.size() )
      {
        throw std::invalid_argument( "TRON: x, lower and upper must have the same nonzero size" );
      }
      for ( Eigen::Index i = 0; i < x.size(); ++i )
      {
        if ( !std::isfinite( x[i] ) || std::isnan( lower[i] ) || std::isnan( upper[i] ) || lower[i] > upper[i] )
        {
          throw std::invalid_argument( "TRON: invalid initial point or bounds" );
        }
      }
      if (
        !( o.sufficient_decrease > Scalar( 0 ) && o.sufficient_decrease < Scalar( 0.5 ) ) ||
        !std::isfinite( o.cauchy_radius_factor ) || !( o.cauchy_radius_factor > Scalar( 0 ) ) ||
        !std::isfinite( o.cauchy_step_factor ) || !( o.cauchy_step_factor > Scalar( 1 ) ) ||
        !std::isfinite( o.max_radius ) || !( o.max_radius > Scalar( 0 ) ) ||
        !( Scalar( 0 ) < o.acceptance_threshold && o.acceptance_threshold < o.decrease_threshold &&
           o.decrease_threshold < o.increase_threshold && o.increase_threshold < Scalar( 1 ) ) ||
        !( Scalar( 0 ) < o.large_decrease_factor && o.large_decrease_factor < o.small_decrease_factor &&
           o.small_decrease_factor < Scalar( 1 ) && Scalar( 1 ) < o.increase_factor &&
           std::isfinite( o.increase_factor ) ) ||
        !std::isfinite( o.absolute_tolerance ) || o.absolute_tolerance < Scalar( 0 ) ||
        !std::isfinite( o.relative_tolerance ) || o.relative_tolerance < Scalar( 0 ) ||
        !std::isfinite( o.active_absolute_tolerance ) || o.active_absolute_tolerance < Scalar( 0 ) ||
        !std::isfinite( o.active_relative_tolerance ) || o.active_relative_tolerance < Scalar( 0 ) ||
        !std::isfinite( o.cg_tolerance ) || !( o.cg_tolerance > Scalar( 0 ) ) ||
        o.max_projected_newton_iterations == 0 )
      {
        throw std::invalid_argument( "TRON: invalid solver options" );
      }
    }

  }  // namespace detail

  // Minimize f(x) subject to lower <= x <= upper.
  //
  // Callbacks have the signatures
  //   Scalar value(Vector<Scalar> const& x)
  //   void   gradient(Vector<Scalar> const& x, Vector<Scalar>& g)
  //   void   hessian_vector(Vector<Scalar> const& x,
  //                         Vector<Scalar> const& v, Vector<Scalar>& Hv)
  // The Hessian-vector callback makes the solver matrix-free.  A dense Eigen
  // Hessian can simply implement the last callback as `Hv.noalias() = H * v`.
  /**
   * \brief Low-level matrix-free TRON minimization kernel.
   *
   * The routine projects the supplied initial point, evaluates objective and
   * gradient information, then repeatedly computes a projected Cauchy point,
   * refines it by projected Newton/CG iterations, evaluates the trial objective,
   * accepts or rejects the trial from the ratio of actual to predicted decrease,
   * and updates the trust-region radius.
   *
   * \tparam Scalar Floating-point scalar type.
   * \tparam Value Objective callback type.
   * \tparam Gradient Gradient callback type.
   * \tparam HessianVector Hessian-vector callback type.
   * \param x Initial point; it is projected onto the feasible box before use.
   * \param lower Lower bounds.
   * \param upper Upper bounds.
   * \param value Callable returning \f$f(x)\f$.
   * \param gradient Callable writing \f$\nabla f(x)\f$.
   * \param hessian_vector Callable writing \f$\nabla^2f(x)v\f$.
   * \param options Numerical options.
   * \return Final point, termination status, norms, radius, and evaluation counters.
   */
  template <std::floating_point Scalar, class Value, class Gradient, class HessianVector>
  [[nodiscard]] Result<Scalar> minimize(
    Vector<Scalar>          x,
    Vector<Scalar> const &  lower,
    Vector<Scalar> const &  upper,
    Value &&                value,
    Gradient &&             gradient,
    HessianVector &&        hessian_vector,
    Options<Scalar> const & options = {} )
  {
    detail::validate( x, lower, upper, options );
    Result<Scalar> result;
    detail::project_in_place( x, lower, upper );
    auto evaluate_value = [&]( Vector<Scalar> const & point )
    {
      ++result.function_evaluations;
      return static_cast<Scalar>( std::invoke( value, point ) );
    };
    auto evaluate_gradient = [&]( Vector<Scalar> const & point, Vector<Scalar> & g )
    {
      ++result.gradient_evaluations;
      std::invoke( gradient, point, g );
      if ( g.size() != point.size() )
        throw std::invalid_argument( "TRON: gradient callback returned the wrong vector size" );
    };

    Vector<Scalar> g( x.size() ), trial_g( x.size() ), projected( x.size() );
    Scalar         f = evaluate_value( x );
    if ( !std::isfinite( f ) )
    {
      result.x         = std::move( x );
      result.objective = f;
      result.status = f == -std::numeric_limits<Scalar>::infinity() ? Status::unbounded : Status::non_finite_objective;
      return result;
    }
    evaluate_gradient( x, g );
    if ( !detail::all_finite( g ) )
    {
      result.x         = std::move( x );
      result.objective = f;
      result.status    = Status::non_finite_gradient;
      return result;
    }

    Scalar       projected_norm        = detail::projected_gradient_norm( x, g, lower, upper, projected );
    Scalar const stopping_tolerance    = options.absolute_tolerance + options.relative_tolerance * projected_norm;
    Scalar const unbounded_threshold   = std::min( Scalar( -1 ), f ) / std::numeric_limits<Scalar>::epsilon();
    Scalar       radius                = std::clamp( projected_norm / Scalar( 10 ), Scalar( 1 ), options.max_radius );
    Scalar       cauchy_alpha          = Scalar( 1 );
    Scalar       ratio                 = Scalar( 0 );
    std::size_t  successful_iterations = 0;

    Vector<Scalar> base_x( x.size() ), trial_x( x.size() ), step( x.size() ), hessian_step( x.size() ),
      direction( x.size() );

    auto finish = [&]( Status status )
    {
      result.x                       = x;
      result.status                  = status;
      result.objective               = f;
      result.projected_gradient_norm = projected_norm;
      result.trust_region_radius     = radius;
      result.last_ratio              = ratio;
      result.accepted_iterations     = successful_iterations;
      return result;
    };

    if ( projected_norm <= stopping_tolerance ) return finish( Status::converged );
    if ( f < unbounded_threshold || f == -std::numeric_limits<Scalar>::infinity() ) return finish( Status::unbounded );

    while ( result.iterations < options.max_iterations )
    {
      if ( result.function_evaluations >= options.max_function_evaluations )
      {
        return finish( Status::max_function_evaluations );
      }

      base_x                     = x;
      Scalar const f0            = f;
      Scalar const old_radius    = radius;
      auto         apply_hessian = [&]( Vector<Scalar> const & v, Vector<Scalar> & Hv )
      {
        ++result.hessian_vector_evaluations;
        std::invoke( hessian_vector, base_x, v, Hv );
        if ( Hv.size() != base_x.size() )
          throw std::invalid_argument( "TRON: Hessian-vector callback returned the wrong vector size" );
      };

      auto const cauchy_status = detail::cauchy_step(
        base_x,
        g,
        lower,
        upper,
        old_radius,
        cauchy_alpha,
        options,
        apply_hessian,
        step,
        hessian_step,
        direction );
      if ( cauchy_status == detail::CauchyStatus::small_step ) return finish( Status::small_step );
      if ( cauchy_status == detail::CauchyStatus::non_finite_hessian ) { return finish( Status::non_finite_hessian ); }

      auto const newton_status = detail::projected_newton(
        base_x,
        g,
        lower,
        upper,
        old_radius,
        options,
        apply_hessian,
        trial_x,
        step,
        hessian_step,
        result.cg_iterations );
      if ( newton_status == detail::ProjectedNewtonStatus::non_finite ) { return finish( Status::non_finite_hessian ); }

      Scalar const slope           = g.dot( step );
      Scalar const model_reduction = Scalar( 0.5 ) * step.dot( hessian_step ) + slope;
      Scalar const f_trial         = evaluate_value( trial_x );
      if ( std::isnan( f_trial ) ) return finish( Status::non_finite_objective );
      if ( f_trial == -std::numeric_limits<Scalar>::infinity() )
      {
        x = trial_x;
        f = f_trial;
        return finish( Status::unbounded );
      }

      Scalar const roundoff                 = std::max( Scalar( 1 ), std::abs( f0 ) ) * Scalar( 10 ) *
                                              std::numeric_limits<Scalar>::epsilon();
      Scalar       predicted                = model_reduction - roundoff;
      Scalar       actual                   = f_trial - f0 + roundoff;
      bool         trial_gradient_available = false;
      if (
        std::abs( model_reduction ) < Scalar( 10000 ) * std::numeric_limits<Scalar>::epsilon() ||
        std::abs( actual ) < Scalar( 10000 ) * std::numeric_limits<Scalar>::epsilon() * std::abs( f0 ) )
      {
        evaluate_gradient( trial_x, trial_g );
        if ( !detail::all_finite( trial_g ) ) return finish( Status::non_finite_gradient );
        actual                   = Scalar( 0.5 ) * ( trial_g.dot( step ) + slope );
        trial_gradient_available = true;
      }
      if ( !( predicted < Scalar( 0 ) ) ) return finish( Status::non_descent_model );

      ratio                          = actual / predicted;
      Scalar const gamma             = f_trial - f0 - slope;
      Scalar const quadratic_minimum = gamma <= Scalar( 0 )
                                         ? options.increase_factor
                                         : std::max( options.large_decrease_factor, -slope / ( Scalar( 2 ) * gamma ) );

      if ( ratio >= options.acceptance_threshold )
      {
        ++successful_iterations;
        x = trial_x;
        f = f_trial;
        if ( trial_gradient_available ) { g = trial_g; }
        else
        {
          evaluate_gradient( x, g );
          if ( !detail::all_finite( g ) ) return finish( Status::non_finite_gradient );
        }
        projected_norm = detail::projected_gradient_norm( x, g, lower, upper, projected );
      }

      ++result.iterations;
      Scalar const step_norm = step.stableNorm();
      if ( successful_iterations == 0 ) radius = std::min( old_radius, step_norm );

      Scalar const sigma1 = options.large_decrease_factor;
      Scalar const sigma2 = options.small_decrease_factor;
      Scalar const sigma3 = options.increase_factor;
      if ( ratio < options.acceptance_threshold )
      {
        radius = std::min( std::max( quadratic_minimum, sigma1 ) * step_norm, sigma2 * radius );
      }
      else if ( ratio < options.decrease_threshold )
      {
        radius = std::max( sigma1 * radius, std::min( quadratic_minimum * step_norm, sigma2 * radius ) );
      }
      else if ( ratio < options.increase_threshold )
      {
        radius = std::min(
          options.max_radius,
          std::max( sigma1 * radius, std::min( quadratic_minimum * step_norm, sigma3 * radius ) ) );
      }
      else
      {
        radius = std::min(
          options.max_radius,
          std::max( radius, std::min( quadratic_minimum * step_norm, sigma3 * radius ) ) );
      }

      if ( projected_norm <= stopping_tolerance ) return finish( Status::converged );
      if ( f < unbounded_threshold ) return finish( Status::unbounded );
    }
    return finish( Status::max_iterations );
  }

}  // namespace Utils::TRON_details

namespace Utils
{
  /**
   * \brief Configurable public wrapper for matrix-free and dense TRON solves.
   *
   * The class owns configuration and the expected problem dimension; the
   * numerical kernel and result types remain in Utils::TRON_details.
   */
  template <std::floating_point Scalar = double> class Minimize_BBOX_TRON
  {
  public:
    using Vector         = TRON_details::Vector<Scalar>;
    using Matrix         = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ConstVectorRef = Eigen::Ref<Vector const>;
    using VectorRef      = Eigen::Ref<Vector>;
    using ConstMatrixRef = Eigen::Ref<Matrix const>;
    using MatrixRef      = Eigen::Ref<Matrix>;
    using Options        = TRON_details::Options<Scalar>;
    using Result         = TRON_details::Result<Scalar>;
    using Status         = TRON_details::Status;

    /**
     * \brief Construct a solver with optional fixed problem dimension.
     * \param dimension Expected dimension; zero disables dimension locking.
     * \param options Initial numerical options.
     * \throws std::invalid_argument if \p dimension is negative.
     */
    explicit Minimize_BBOX_TRON( Eigen::Index dimension = 0, Options options = {} )
      : m_options( options ), m_dimension( dimension )
    {
      if ( dimension < 0 ) throw std::invalid_argument( "Minimize_BBOX_TRON: negative dimension" );
    }

    /** \brief Obtain mutable access to the solver options. */
    [[nodiscard]] Options & options() noexcept { return m_options; }
    /** \brief Obtain read-only access to the solver options. */
    [[nodiscard]] Options const & options() const noexcept { return m_options; }

    /**
     * \brief Set both absolute and relative projected-gradient tolerances.
     * \param tol Tolerance forwarded to \c Options::set_tolerances().
     */
    void set_tolerances( Scalar tol ) { m_options.set_tolerances( tol ); }
    /**
     * \brief Set the outer TRON iteration limit.
     * \param n Maximum number of trust-region iterations.
     */
    void set_max_iterations( std::size_t n ) { m_options.set_max_iterations( n ); }

    /**
     * \brief Change the expected problem dimension.
     * \param dimension New expected dimension; zero permits any dimension.
     * \throws std::invalid_argument if \p dimension is negative.
     */
    void resize( Eigen::Index dimension )
    {
      if ( dimension < 0 ) throw std::invalid_argument( "Minimize_BBOX_TRON: negative dimension" );
      m_dimension = dimension;
    }

    /** \brief Return the configured expected problem dimension. */
    [[nodiscard]] Eigen::Index dimension() const noexcept { return m_dimension; }

    /**
     * \brief Solve a box-constrained problem through matrix-free callbacks.
     *
     * \param x0 Initial point.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \param value Objective callback.
     * \param gradient Gradient callback.
     * \param hessian_vector Hessian-vector product callback.
     * \return Detailed TRON result.
     */
    template <class Value, class Gradient, class HessianVector>
      requires std::is_invocable_r_v<Scalar, Value, Vector const &> &&
               std::is_invocable_v<Gradient, Vector const &, Vector &> &&
               std::is_invocable_v<HessianVector, Vector const &, Vector const &, Vector &>
    [[nodiscard]] Result solve(
      Vector const &   x0,
      Vector const &   lower,
      Vector const &   upper,
      Value &&         value,
      Gradient &&      gradient,
      HessianVector && hessian_vector ) const
    {
      check_dimension( x0.size() );
      return TRON_details::minimize(
        x0,
        lower,
        upper,
        std::forward<Value>( value ),
        std::forward<Gradient>( gradient ),
        std::forward<HessianVector>( hessian_vector ),
        m_options );
    }

    /**
     * \brief Alias of the matrix-free \c solve() interface.
     *
     * This method exists to support minimizer-oriented naming while preserving
     * exactly the same callback contract and numerical path as \c solve().
     */
    template <class Value, class Gradient, class HessianVector>
      requires std::is_invocable_r_v<Scalar, Value, Vector const &> &&
               std::is_invocable_v<Gradient, Vector const &, Vector &> &&
               std::is_invocable_v<HessianVector, Vector const &, Vector const &, Vector &>
    [[nodiscard]] Result minimize(
      Vector const &   x0,
      Vector const &   lower,
      Vector const &   upper,
      Value &&         value,
      Gradient &&      gradient,
      HessianVector && hessian_vector ) const
    {
      return solve(
        x0,
        lower,
        upper,
        std::forward<Value>( value ),
        std::forward<Gradient>( gradient ),
        std::forward<HessianVector>( hessian_vector ) );
    }

    /**
     * \brief Solve using objective, gradient, and dense-Hessian callbacks.
     *
     * A dense Hessian matrix is assembled whenever the matrix-free kernel asks
     * for a Hessian-vector product, after which \f$Hv\f$ is formed by Eigen.
     * This interface is convenient for small and medium dense problems but may
     * recompute the Hessian multiple times within one outer iteration.
     *
     * \param obj Objective callback.
     * \param grad Gradient callback.
     * \param hess_dense Dense Hessian callback.
     * \param x0 Initial point.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \return Detailed TRON result.
     */
    template <typename Obj, typename Grad, typename Hess>
      requires std::is_invocable_r_v<Scalar, Obj, ConstVectorRef> &&
               std::is_invocable_v<Grad, ConstVectorRef, VectorRef> &&
               std::is_invocable_v<Hess, ConstVectorRef, MatrixRef>
    [[nodiscard]] Result solve(
      Obj &&         obj,
      Grad &&        grad,
      Hess &&        hess_dense,
      ConstVectorRef x0,
      ConstVectorRef lower,
      ConstVectorRef upper ) const
    {
      auto value    = [&]( ConstVectorRef x ) -> Scalar { return static_cast<Scalar>( obj( x ) ); };
      auto gradient = [&]( ConstVectorRef x, VectorRef g ) -> void { grad( x, g ); };
      auto hess_vec = [&]( ConstVectorRef x, ConstVectorRef v, VectorRef Hv ) -> void
      {
        Matrix H( x.size(), x.size() );
        hess_dense( x, H );
        Hv.noalias() = H * v;
      };
      return solve( x0, lower, upper, value, gradient, hess_vec );
    }

    /**
     * \brief Solve a problem object exposing objective, gradient, and Hessian.
     *
     * The problem must provide \c objective(x), \c gradient(x,g), and
     * \c hessian(x,H).  As in the dense-callback overload, the Hessian is
     * assembled on demand and multiplied by the requested vector.
     *
     * \param problem User problem object satisfying the required interface.
     * \param x0 Initial point.
     * \param lower Lower bounds.
     * \param upper Upper bounds.
     * \return Detailed TRON result.
     */
    template <typename Problem>
      requires requires( Problem & p, ConstVectorRef x, VectorRef g, MatrixRef H ) {
        { p.objective( x ) } -> std::convertible_to<Scalar>;
        p.gradient( x, g );
        p.hessian( x, H );
      }
    [[nodiscard]] Result solve( Problem & problem, ConstVectorRef x0, ConstVectorRef lower, ConstVectorRef upper ) const
    {
      auto value    = [&]( ConstVectorRef x ) -> Scalar { return static_cast<Scalar>( problem.objective( x ) ); };
      auto gradient = [&]( ConstVectorRef x, VectorRef g ) -> void { problem.gradient( x, g ); };
      auto hess_vec = [&]( ConstVectorRef x, ConstVectorRef v, VectorRef Hv ) -> void
      {
        Matrix H( x.size(), x.size() );
        problem.hessian( x, H );
        Hv.noalias() = H * v;
      };
      return solve( x0, lower, upper, value, gradient, hess_vec );
    }

  private:
    /**
     * \brief Enforce the optional dimension lock configured in the solver.
     * \param dimension Dimension of the current initial point.
     * \throws std::invalid_argument if a nonzero configured dimension differs
     * from \p dimension.
     */
    void check_dimension( Eigen::Index dimension ) const
    {
      if ( m_dimension != 0 && dimension != m_dimension )
        throw std::invalid_argument( "Minimize_BBOX_TRON: initial point has the wrong dimension" );
    }

    Options      m_options{};
    Eigen::Index m_dimension{ 0 };
  };

  using TRON_details::to_string;
}  // namespace Utils

#endif
