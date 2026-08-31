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

// Utils_minimize_BBOX_TRON.hh -- header-only C++20 port of the TRON trust-region solver for
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
// the TRONTrustRegion update rules of SolverTools.jl and the Steihaug-Toint
// truncated conjugate gradient of Krylov.jl.
//
// Dependencies: Eigen (Core only) and the C++20 standard library.

#pragma once

#ifndef UTILS_MINIMIZE_BBOX_TRON_DOT_HH
#define UTILS_MINIMIZE_BBOX_TRON_DOT_HH

#include "Utils_eigen.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#if EIGEN_MAJOR_VERSION < 5
#error "Utils::Minimize_BBOX_TRON requires Eigen 5 or newer"
#endif

namespace Utils::TRON2_details
{

  template <std::floating_point Scalar> using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  enum class Status
  {
    converged,
    max_iterations,
    max_function_evaluations,
    max_time,
    unbounded,
    small_step,
    non_descent_model,
    non_finite_objective,
    non_finite_gradient,
    non_finite_hessian
  };

  [[nodiscard]] constexpr std::string_view to_string( Status status ) noexcept
  {
    switch ( status )
    {
      case Status::converged: return "converged";
      case Status::max_iterations: return "maximum number of iterations";
      case Status::max_function_evaluations: return "maximum number of function evaluations";
      case Status::max_time: return "time limit exceeded";
      case Status::unbounded: return "objective appears unbounded below";
      case Status::small_step: return "Cauchy step is too small";
      case Status::non_descent_model: return "quadratic model does not predict descent";
      case Status::non_finite_objective: return "objective returned a non-finite value";
      case Status::non_finite_gradient: return "gradient contains a non-finite value";
      case Status::non_finite_hessian: return "Hessian-vector product contains a non-finite value";
    }
    return "unknown";
  }

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

    Scalar absolute_tolerance{ std::sqrt( std::numeric_limits<Scalar>::epsilon() ) };
    Scalar relative_tolerance{ std::sqrt( std::numeric_limits<Scalar>::epsilon() ) };
    Scalar cg_tolerance{ Scalar( 0.1 ) };
    Scalar active_absolute_tolerance{ std::sqrt( std::numeric_limits<Scalar>::epsilon() ) };
    Scalar active_relative_tolerance{ std::sqrt( std::numeric_limits<Scalar>::epsilon() ) };

    std::size_t max_iterations{ std::numeric_limits<std::size_t>::max() };
    std::size_t max_function_evaluations{ std::numeric_limits<std::size_t>::max() };
    std::size_t max_projected_newton_iterations{ 50 };
    // Zero selects 2*n, matching Krylov.cg!'s default in the Julia version.
    std::size_t max_cg_iterations{ 0 };
    double      max_time_seconds{ 30.0 };

    // Unified setters (same name across all BBOX solvers)
    void set_tolerances( Scalar tol )
    {
      absolute_tolerance = relative_tolerance = tol;
    }
    void set_max_iterations( std::size_t n ) { max_iterations = n; }
  };

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
    double         elapsed_seconds{ 0 };

    [[nodiscard]] bool success() const noexcept { return status == Status::converged; }
  };

  namespace detail
  {

    template <std::floating_point Scalar> [[nodiscard]] bool all_finite( Vector<Scalar> const & v )
    { return v.array().isFinite().all(); }

    template <std::floating_point Scalar>
    void project_in_place( Vector<Scalar> & x, Vector<Scalar> const & lower, Vector<Scalar> const & upper )
    { x = x.cwiseMax( lower ).cwiseMin( upper ); }

    template <std::floating_point Scalar> void project_step(
      Vector<Scalar> &       step,
      Vector<Scalar> const & x,
      Vector<Scalar> const & direction,
      Vector<Scalar> const & lower,
      Vector<Scalar> const & upper,
      Scalar                 alpha )
    { step = ( x + alpha * direction ).cwiseMax( lower ).cwiseMin( upper ) - x; }

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

    template <std::floating_point Scalar> struct Breakpoints
    {
      std::size_t count{ 0 };
      Scalar      minimum{ std::numeric_limits<Scalar>::infinity() };
      Scalar      maximum{ Scalar( 0 ) };
    };

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

    enum class CauchyStatus
    {
      success,
      small_step,
      non_finite_hessian
    };

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

    enum class CGStatus
    {
      converged,
      boundary,
      negative_curvature,
      iteration_limit,
      non_finite
    };

    template <std::floating_point Scalar> struct CGResult
    {
      CGStatus    status{ CGStatus::converged };
      std::size_t iterations{ 0 };
    };

    template <std::floating_point Scalar, class ApplyOperator, class TimeExpired>
    [[nodiscard]] CGResult<Scalar> truncated_cg(
      ApplyOperator &&       apply,
      Vector<Scalar> const & rhs,
      Scalar                 radius,
      Scalar                 relative_tolerance,
      std::size_t            iteration_limit,
      TimeExpired &&         time_expired,
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
        if ( std::invoke( time_expired ) )
        {
          result.status = CGStatus::iteration_limit;
          return result;
        }
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

    enum class ProjectedNewtonStatus
    {
      stationary,
      boundary,
      iteration_limit,
      time_limit,
      non_finite
    };

    template <std::floating_point Scalar, class ApplyHessian, class TimeExpired>
    [[nodiscard]] ProjectedNewtonStatus projected_newton(
      Vector<Scalar> const &  base_x,
      Vector<Scalar> const &  gradient,
      Vector<Scalar> const &  lower,
      Vector<Scalar> const &  upper,
      Scalar                  radius,
      Options<Scalar> const & options,
      ApplyHessian &&         apply_hessian,
      TimeExpired &&          time_expired,
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
        if ( std::invoke( time_expired ) ) return ProjectedNewtonStatus::time_limit;
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
          time_expired,
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

    template <std::floating_point Scalar> void validate(
      Vector<Scalar> const &  x,
      Vector<Scalar> const &  lower,
      Vector<Scalar> const &  upper,
      Options<Scalar> const & o )
    {
      if ( x.size() == 0 || lower.size() != x.size() || upper.size() != x.size() )
      {
        throw std::invalid_argument( "TRON2: x, lower and upper must have the same nonzero size" );
      }
      for ( Eigen::Index i = 0; i < x.size(); ++i )
      {
        if ( !std::isfinite( x[i] ) || std::isnan( lower[i] ) || std::isnan( upper[i] ) || lower[i] > upper[i] )
        {
          throw std::invalid_argument( "TRON2: invalid initial point or bounds" );
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
        o.max_projected_newton_iterations == 0 || !( o.max_time_seconds >= 0 ) )
      {
        throw std::invalid_argument( "TRON2: invalid solver options" );
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
    using Clock             = std::chrono::steady_clock;
    auto const start        = Clock::now();
    auto       elapsed      = [&]() { return std::chrono::duration<double>( Clock::now() - start ).count(); };
    auto       time_expired = [&]() { return elapsed() >= options.max_time_seconds; };

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
        throw std::invalid_argument( "TRON2: gradient callback returned the wrong vector size" );
    };

    Vector<Scalar> g( x.size() ), trial_g( x.size() ), projected( x.size() );
    Scalar         f = evaluate_value( x );
    if ( !std::isfinite( f ) )
    {
      result.x         = std::move( x );
      result.objective = f;
      result.status = f == -std::numeric_limits<Scalar>::infinity() ? Status::unbounded : Status::non_finite_objective;
      result.elapsed_seconds = elapsed();
      return result;
    }
    evaluate_gradient( x, g );
    if ( !detail::all_finite( g ) )
    {
      result.x               = std::move( x );
      result.objective       = f;
      result.status          = Status::non_finite_gradient;
      result.elapsed_seconds = elapsed();
      return result;
    }

    Scalar       projected_norm      = detail::projected_gradient_norm( x, g, lower, upper, projected );
    Scalar const stopping_tolerance  = options.absolute_tolerance + options.relative_tolerance * projected_norm;
    Scalar const unbounded_threshold = std::min( Scalar( -1 ), f ) / std::numeric_limits<Scalar>::epsilon();
    Scalar       radius       = std::min( std::max( Scalar( 1 ), projected_norm / Scalar( 10 ) ), options.max_radius );
    Scalar       cauchy_alpha = Scalar( 1 );
    Scalar       ratio        = Scalar( 0 );
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
      result.elapsed_seconds         = elapsed();
      return result;
    };

    if ( projected_norm <= stopping_tolerance ) return finish( Status::converged );
    if ( f < unbounded_threshold || f == -std::numeric_limits<Scalar>::infinity() ) return finish( Status::unbounded );
    if ( time_expired() ) return finish( Status::max_time );

    while ( result.iterations < options.max_iterations )
    {
      if ( result.function_evaluations >= options.max_function_evaluations )
      {
        return finish( Status::max_function_evaluations );
      }
      if ( time_expired() ) return finish( Status::max_time );

      base_x                     = x;
      Scalar const f0            = f;
      Scalar const old_radius    = radius;
      auto         apply_hessian = [&]( Vector<Scalar> const & v, Vector<Scalar> & Hv )
      {
        ++result.hessian_vector_evaluations;
        std::invoke( hessian_vector, base_x, v, Hv );
        if ( Hv.size() != base_x.size() )
          throw std::invalid_argument( "TRON2: Hessian-vector callback returned the wrong vector size" );
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
        time_expired,
        trial_x,
        step,
        hessian_step,
        result.cg_iterations );
      if ( newton_status == detail::ProjectedNewtonStatus::time_limit ) return finish( Status::max_time );
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

}  // namespace Utils::TRON2_details

namespace Utils
{
  namespace TRON2 = TRON2_details; // source compatibility

  /** Matrix-free bound-constrained TRON solver.
   *
   * The class owns configuration and the expected problem dimension; the
   * numerical kernel and result types remain in Utils::TRON2_details.
   */
  template <std::floating_point Scalar = double> class Minimize_BBOX_TRON
  {
  public:
    using Vector  = TRON2_details::Vector<Scalar>;
    using Matrix  = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Options = TRON2_details::Options<Scalar>;
    using Result  = TRON2_details::Result<Scalar>;
    using Status  = TRON2_details::Status;

    explicit Minimize_BBOX_TRON( Eigen::Index dimension = 0, Options options = {} )
      : m_options( options ), m_dimension( dimension )
    {
      if ( dimension < 0 ) throw std::invalid_argument( "Minimize_BBOX_TRON: negative dimension" );
    }

    [[nodiscard]] Options &       options() noexcept { return m_options; }
    [[nodiscard]] Options const & options() const noexcept { return m_options; }

    void set_tolerances( Scalar tol ) { m_options.set_tolerances( tol ); }
    void set_max_iterations( std::size_t n ) { m_options.set_max_iterations( n ); }

    void resize( Eigen::Index dimension )
    {
      if ( dimension < 0 ) throw std::invalid_argument( "Minimize_BBOX_TRON: negative dimension" );
      m_dimension = dimension;
    }

    [[nodiscard]] Eigen::Index dimension() const noexcept { return m_dimension; }

    template <class Value, class Gradient, class HessianVector>
    [[nodiscard]] Result solve(
      Vector const &  x0,
      Vector const &  lower,
      Vector const &  upper,
      Value &&        value,
      Gradient &&     gradient,
      HessianVector && hessian_vector ) const
    {
      check_dimension( x0.size() );
      return TRON2_details::minimize(
        x0,
        lower,
        upper,
        std::forward<Value>( value ),
        std::forward<Gradient>( gradient ),
        std::forward<HessianVector>( hessian_vector ),
        m_options );
    }

    template <class Value, class Gradient, class HessianVector>
    [[nodiscard]] Result minimize(
      Vector const &  x0,
      Vector const &  lower,
      Vector const &  upper,
      Value &&        value,
      Gradient &&     gradient,
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

    // --- unified dense interface: f(x), grad(x,g), hess(x,H) with dense Matrix ---
    template <typename Obj, typename Grad, typename Hess>
    [[nodiscard]] Result solve(
      Obj &&         obj,
      Grad &&        grad,
      Hess &&        hess_dense,
      Vector const & x0,
      Vector const & lower,
      Vector const & upper ) const
    {
      auto value = [&]( Vector const & x ) -> Scalar { return static_cast<Scalar>( obj( x ) ); };
      auto gradient = [&]( Vector const & x, Vector & g ) -> void { grad( x, g ); };
      auto hess_vec = [&]( Vector const & x, Vector const & v, Vector & Hv ) -> void {
        Matrix H( x.size(), x.size() );
        hess_dense( x, H );
        Hv.noalias() = H * v;
      };
      return solve( x0, lower, upper, value, gradient, hess_vec );
    }

    // Problem-based dense interface (expects problem.objective/gradient/hessian with dense Matrix)
    template <typename Problem>
      requires requires( Problem & p, Vector const & x, Vector & g, Matrix & H ) {
        { p.objective( x ) } -> std::convertible_to<Scalar>;
        p.gradient( x, g );
        p.hessian( x, H );
      }
    [[nodiscard]] Result solve(
      Problem &      problem,
      Vector const & x0,
      Vector const & lower,
      Vector const & upper ) const
    {
      auto value = [&]( Vector const & x ) -> Scalar { return static_cast<Scalar>( problem.objective( x ) ); };
      auto gradient = [&]( Vector const & x, Vector & g ) -> void { problem.gradient( x, g ); };
      auto hess_vec = [&]( Vector const & x, Vector const & v, Vector & Hv ) -> void {
        Matrix H( x.size(), x.size() );
        problem.hessian( x, H );
        Hv.noalias() = H * v;
      };
      return solve( x0, lower, upper, value, gradient, hess_vec );
    }

  private:
    void check_dimension( Eigen::Index dimension ) const
    {
      if ( m_dimension != 0 && dimension != m_dimension )
        throw std::invalid_argument( "Minimize_BBOX_TRON: initial point has the wrong dimension" );
    }

    Options      m_options{};
    Eigen::Index m_dimension{ 0 };
  };
}

#endif
