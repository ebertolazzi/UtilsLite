/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2018                                                      |
 |                                                                          |
 |      Enrico Bertolazzi                                                   |
 |      Dipartimento di Ingegneria Industriale                              |
 |      Università degli Studi di Trento                                    |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#pragma once

#ifndef UTILS_MINIMIZE_BBOX_NEWTON_CUBIC_DOT_HH
#define UTILS_MINIMIZE_BBOX_NEWTON_CUBIC_DOT_HH

#include "Utils_minimize_BBOX_Newton.hh"

#include <functional>

namespace Utils
{

  /**
   * Newton minimizer with adaptive cubic regularization.
   *
   * The algorithm is the original MinimizeNewtonCubic method; its public API
   * matches Utils::Minimize_BBOX_Newton<Real>, so the two solver types are
   * interchangeable at a call site.
   *
   * Header-only: all member functions are defined inline in this header, so
   * no separate .cc file / translation unit is required.
   *
   * The initial point and every trial point are projected onto [lower, upper].
   * Convergence is measured with the projected KKT residual
   *
   *   p(x) = x - P_[lower,upper](x - gradient(x)).
   *
   * Infinite one-sided or two-sided bounds are supported.
   */
  template <typename Real = double> class Minimize_BBOX_NewtonCubic
  {
  public:
    using Vec     = Utils::Vector<Real>;
    using Mat     = Utils::Matrix<Real>;
    using Options = Utils::Options<Real>;
    using Result  = Utils::Result<Real>;

    explicit Minimize_BBOX_NewtonCubic( Eigen::Index dimension = 0, Options options = {} )
      : m_options( std::move( options ) )
    { resize( dimension ); }

    [[nodiscard]] Options &       options() noexcept { return m_options; }
    [[nodiscard]] Options const & options() const noexcept { return m_options; }

    void set_tolerances( Real tol ) { m_options.set_tolerances( tol ); }
    void set_max_iterations( int n ) { m_options.max_iterations = n; }

    void resize( Eigen::Index dimension )
    {
      m_x.resize( dimension );
      m_trial.resize( dimension );
      m_gradient.resize( dimension );
      m_trial_gradient.resize( dimension );
      m_projected_gradient.resize( dimension );
      m_trial_projected_gradient.resize( dimension );
      m_direction.resize( dimension );
      m_hessian.resize( dimension, dimension );
      m_shifted_hessian.resize( dimension, dimension );
    }

    template <typename Problem>
      requires Utils::ProblemFor<Problem, Real>
    Result solve(
      Problem &                   problem,
      Utils::ConstVectorRef<Real> x0,
      Utils::ConstVectorRef<Real> lower,
      Utils::ConstVectorRef<Real> upper )
    {
      auto keep_going = []( Result const & ) { return true; };
      return solve( problem, x0, lower, upper, keep_going );
    }

    template <typename Problem, typename Callback>
      requires Utils::ProblemFor<Problem, Real>
    Result solve(
      Problem &                   problem,
      Utils::ConstVectorRef<Real> x0,
      Utils::ConstVectorRef<Real> lower,
      Utils::ConstVectorRef<Real> upper,
      Callback &&                 callback )
    {
      Objective objective = [&problem]( Utils::ConstVectorRef<Real> x ) { return problem.objective( x ); };
      Gradient  gradient  = [&problem]( Utils::ConstVectorRef<Real> x, Utils::VectorRef<Real> g )
      { problem.gradient( x, g ); };
      Hessian hessian = [&problem]( Utils::ConstVectorRef<Real> x, Utils::MatrixRef<Real> h )
      { problem.hessian( x, h ); };
      IterationCallback iteration_callback = [&callback]( Result const & result )
      { return static_cast<bool>( callback( result ) ); };
      return solve_impl( objective, gradient, hessian, x0, lower, upper, iteration_callback );
    }

    // --- unified lambda interface (dense Vector / Matrix) ---
    template <typename Obj, typename Grad, typename Hess>
    Result solve(
      Obj &&                      obj,
      Grad &&                     grad,
      Hess &&                     hess,
      Utils::ConstVectorRef<Real> x0,
      Utils::ConstVectorRef<Real> lower,
      Utils::ConstVectorRef<Real> upper )
    {
      auto prob = Utils::make_problem<Real>(
        std::forward<Obj>( obj ), std::forward<Grad>( grad ), std::forward<Hess>( hess ) );
      return solve( prob, x0, lower, upper );
    }

    template <typename Obj, typename Grad, typename Hess, typename Callback>
    Result solve(
      Obj &&                      obj,
      Grad &&                     grad,
      Hess &&                     hess,
      Utils::ConstVectorRef<Real> x0,
      Utils::ConstVectorRef<Real> lower,
      Utils::ConstVectorRef<Real> upper,
      Callback &&                 callback )
    {
      auto prob = Utils::make_problem<Real>(
        std::forward<Obj>( obj ), std::forward<Grad>( grad ), std::forward<Hess>( hess ) );
      return solve( prob, x0, lower, upper, std::forward<Callback>( callback ) );
    }

  private:
    using Objective         = std::function<Real( Utils::ConstVectorRef<Real> )>;
    using Gradient          = std::function<void( Utils::ConstVectorRef<Real>, Utils::VectorRef<Real> )>;
    using Hessian           = std::function<void( Utils::ConstVectorRef<Real>, Utils::MatrixRef<Real> )>;
    using IterationCallback = std::function<bool( Result const & )>;

    inline Result solve_impl(
      Objective const &           objective,
      Gradient const &            gradient,
      Hessian const &             hessian,
      Utils::ConstVectorRef<Real> x0,
      Utils::ConstVectorRef<Real> lower,
      Utils::ConstVectorRef<Real> upper,
      IterationCallback const &   callback )
    {
      using std::max;
      using std::min;
      using std::sqrt;

      Utils::Check(
        x0.size() == lower.size() && x0.size() == upper.size(),
        "Minimize_BBOX_NewtonCubic::solve: incompatible vector dimensions" );
      Utils::Check(
        !lower.hasNaN() && !upper.hasNaN() && ( lower.array() <= upper.array() ).all(),
        "Minimize_BBOX_NewtonCubic::solve: invalid lower/upper bounds" );

      resize( x0.size() );
      m_x = x0.cwiseMax( lower ).cwiseMin( upper );

      Result result;
      int    function_evaluations        = 0;
      int    gradient_evaluations        = 0;
      int    hessian_evaluations         = 0;
      int    rejected_steps              = 0;
      int    fallback_steps              = 0;
      int    model_rejections            = 0;
      int    gradient_rescue_steps       = 0;
      int    primary_semismooth_attempts = 0;
      int    primary_semismooth_steps    = 0;
      int    primary_semismooth_rejected = 0;

      Real f = objective( m_x );
      ++function_evaluations;
      if ( !Utils::is_finite( f ) )
      {
        result.x                    = m_x;
        result.objective            = f;
        result.function_evaluations = function_evaluations;
        result.status               = Utils::Status::non_finite_objective;
        return result;
      }

      gradient( m_x, m_gradient );
      ++gradient_evaluations;
      if ( !m_gradient.allFinite() )
      {
        result.x                    = m_x;
        result.objective            = f;
        result.function_evaluations = function_evaluations;
        result.gradient_evaluations = gradient_evaluations;
        result.status               = Utils::Status::non_finite_gradient;
        return result;
      }

      m_projected_gradient         = m_x - ( m_x - m_gradient ).cwiseMax( lower ).cwiseMin( upper );
      Real       gradient_norm     = m_projected_gradient.norm();
      Real       gradient_norm_inf = m_projected_gradient.template lpNorm<Eigen::Infinity>();
      Real const initial_scale     = max( Real( 1 ), gradient_norm_inf );
      Real const tolerance         = max(
        Real( 64 ) * std::numeric_limits<Real>::epsilon(),
        min( m_options.absolute_tolerance, m_options.relative_tolerance * initial_scale ) );
      Real hessian_estimate              = max( m_options.hessian_lipschitz_initial, Real( 1 ) );
      Real lambda                        = Real( 0 );
      Real step_norm                     = std::numeric_limits<Real>::infinity();
      Real minimum_critical_eigenvalue   = std::numeric_limits<Real>::quiet_NaN();
      Real effective_curvature_tolerance = Real( 0 );

      auto fill_result = [&]( Utils::Status status, int iterations ) -> Result &
      {
        result.x                             = m_x;
        result.objective                     = f;
        result.projected_gradient_norm       = gradient_norm;
        result.projected_gradient_norm_inf   = gradient_norm_inf;
        result.primal_feasibility            = ( lower - m_x ).cwiseMax( m_x - upper ).cwiseMax( Real( 0 ) ).norm();
        result.step_norm                     = step_norm;
        result.lambda                        = lambda;
        result.hessian_lipschitz_estimate    = hessian_estimate;
        result.optimality_tolerance          = tolerance;
        result.minimum_critical_eigenvalue   = minimum_critical_eigenvalue;
        result.effective_curvature_tolerance = effective_curvature_tolerance;
        result.iterations                    = iterations;
        result.function_evaluations          = function_evaluations;
        result.gradient_evaluations          = gradient_evaluations;
        result.hessian_evaluations           = hessian_evaluations;
        result.rejected_steps                = rejected_steps;
        result.fallback_steps                = fallback_steps;
        result.model_rejections              = model_rejections;
        result.gradient_rescue_steps         = gradient_rescue_steps;
        result.primary_semismooth_attempts   = primary_semismooth_attempts;
        result.primary_semismooth_steps      = primary_semismooth_steps;
        result.primary_semismooth_rejected   = primary_semismooth_rejected;
        result.status                        = status;
        return result;
      };

      enum class CertificateAction
      {
        certified,
        escaped,
        failed
      };
      auto certify_or_escape = [&]() -> CertificateAction
      {
        hessian( m_x, m_hessian );
        ++hessian_evaluations;
        if ( !m_hessian.allFinite() ) return CertificateAction::failed;

        std::vector<Eigen::Index> critical;
        std::vector<int>          cone_sign;
        critical.reserve( static_cast<std::size_t>( m_x.size() ) );
        cone_sign.reserve( static_cast<std::size_t>( m_x.size() ) );
        Real const eps = std::numeric_limits<Real>::epsilon();
        for ( Eigen::Index i = 0; i < m_x.size(); ++i )
        {
          if ( lower[i] == upper[i] ) continue;
          Real const scale = max(
            { Real( 1 ),
              std::abs( m_x[i] ),
              std::abs( m_gradient[i] ),
              std::isfinite( lower[i] ) ? std::abs( lower[i] ) : Real( 0 ),
              std::isfinite( upper[i] ) ? std::abs( upper[i] ) : Real( 0 ) } );
          Real const bound_tolerance = Real( 64 ) * eps * scale;
          bool const at_lower        = std::isfinite( lower[i] ) && m_x[i] <= lower[i] + bound_tolerance;
          bool const at_upper        = std::isfinite( upper[i] ) && m_x[i] >= upper[i] - bound_tolerance;
          bool const strongly_lower  = at_lower && m_gradient[i] > tolerance;
          bool const strongly_upper  = at_upper && m_gradient[i] < -tolerance;
          if ( !strongly_lower && !strongly_upper )
          {
            critical.push_back( i );
            cone_sign.push_back( at_lower ? +1 : ( at_upper ? -1 : 0 ) );
          }
        }

        if ( critical.empty() )
        {
          minimum_critical_eigenvalue   = std::numeric_limits<Real>::infinity();
          effective_curvature_tolerance = Real( 0 );
          step_norm                     = Real( 0 );
          return CertificateAction::certified;
        }

        Mat reduced                   = m_hessian( critical, critical );
        reduced                       = Real( 0.5 ) * ( reduced + reduced.transpose() ).eval();
        Real const hessian_scale      = max( Real( 1 ), reduced.cwiseAbs().maxCoeff() );
        effective_curvature_tolerance = max(
          m_options.curvature_tolerance,
          Real( 256 ) * eps * hessian_scale * Real( std::max<Eigen::Index>( 1, reduced.rows() ) ) );
        Eigen::SelfAdjointEigenSolver<Mat> eig( reduced );
        if ( eig.info() != Eigen::Success )
        {
          minimum_critical_eigenvalue = -std::numeric_limits<Real>::infinity();
          return CertificateAction::failed;
        }

        Eigen::Index const nc             = static_cast<Eigen::Index>( critical.size() );
        Vec                best_direction = Vec::Zero( nc );
        Real               best_curvature = std::numeric_limits<Real>::infinity();
        Real const         rayleigh_step  = Real( 0.25 ) / hessian_scale;

        auto consider = [&]( Vec direction )
        {
          for ( Eigen::Index k = 0; k < nc; ++k )
          {
            if ( cone_sign[static_cast<std::size_t>( k )] > 0 ) direction[k] = max( direction[k], Real( 0 ) );
            if ( cone_sign[static_cast<std::size_t>( k )] < 0 ) direction[k] = min( direction[k], Real( 0 ) );
          }
          Real direction_norm = direction.norm();
          if ( !( direction_norm > Real( 0 ) ) ) return;
          direction /= direction_norm;

          for ( int refinement = 0; refinement < 16; ++refinement )
          {
            Vec const  Hd        = reduced * direction;
            Real const curvature = direction.dot( Hd );
            if ( curvature < best_curvature )
            {
              best_curvature = curvature;
              best_direction = direction;
            }
            Vec candidate = direction - rayleigh_step * ( Hd - curvature * direction );
            for ( Eigen::Index k = 0; k < nc; ++k )
            {
              if ( cone_sign[static_cast<std::size_t>( k )] > 0 ) candidate[k] = max( candidate[k], Real( 0 ) );
              if ( cone_sign[static_cast<std::size_t>( k )] < 0 ) candidate[k] = min( candidate[k], Real( 0 ) );
            }
            Real const candidate_norm = candidate.norm();
            if ( !( candidate_norm > Real( 0 ) ) ) break;
            candidate /= candidate_norm;
            if ( ( candidate - direction ).norm() <= Real( 32 ) * eps ) break;
            direction.swap( candidate );
          }
        };

        for ( Eigen::Index j = 0; j < nc; ++j )
        {
          Vec const eigenvector = eig.eigenvectors().col( j );
          consider( eigenvector );
          consider( -eigenvector );
        }
        for ( Eigen::Index j = 0; j < nc; ++j )
        {
          Vec coordinate = Vec::Zero( nc );
          coordinate[j]  = cone_sign[static_cast<std::size_t>( j )] < 0 ? Real( -1 ) : Real( 1 );
          consider( coordinate );
        }

        minimum_critical_eigenvalue = best_curvature;
        if ( minimum_critical_eigenvalue >= -effective_curvature_tolerance )
        {
          step_norm = Real( 0 );
          return CertificateAction::certified;
        }

        for ( int orientation : { -1, 1 } )
        {
          m_direction.setZero();
          for ( Eigen::Index k = 0; k < static_cast<Eigen::Index>( critical.size() ); ++k )
            m_direction[critical[static_cast<std::size_t>( k )]] = Real( orientation ) * best_direction[k];

          Real alpha_max = std::numeric_limits<Real>::infinity();
          for ( Eigen::Index i = 0; i < m_x.size(); ++i )
          {
            if ( m_direction[i] > Real( 0 ) && std::isfinite( upper[i] ) )
              alpha_max = min( alpha_max, ( upper[i] - m_x[i] ) / m_direction[i] );
            else if ( m_direction[i] < Real( 0 ) && std::isfinite( lower[i] ) )
              alpha_max = min( alpha_max, ( lower[i] - m_x[i] ) / m_direction[i] );
          }
          Real alpha = std::isfinite( alpha_max ) ? alpha_max : Real( 1 );
          if ( !( alpha > Real( 0 ) ) ) continue;

          for ( int backtrack = 0; backtrack <= m_options.max_polish_backtracking; ++backtrack )
          {
            m_trial                   = ( m_x + alpha * m_direction ).cwiseMax( lower ).cwiseMin( upper );
            Real const candidate_step = ( m_trial - m_x ).norm();
            if ( !( candidate_step > Real( 64 ) * eps * max( Real( 1 ), m_x.norm() ) ) ) break;
            if ( m_options.max_function_evaluations >= 0 && function_evaluations >= m_options.max_function_evaluations )
              return CertificateAction::failed;

            Real const trial_f = objective( m_trial );
            ++function_evaluations;
            Real const roundoff = m_options.roundoff_factor * eps *
                                  max( { Real( 1 ), std::abs( f ), std::abs( trial_f ) } );
            if ( Utils::is_finite( trial_f ) && trial_f < f - roundoff )
            {
              gradient( m_trial, m_trial_gradient );
              ++gradient_evaluations;
              if ( !m_trial_gradient.allFinite() ) return CertificateAction::failed;
              m_trial_projected_gradient = m_trial - ( m_trial - m_trial_gradient ).cwiseMax( lower ).cwiseMin( upper );
              m_x.swap( m_trial );
              m_gradient.swap( m_trial_gradient );
              m_projected_gradient.swap( m_trial_projected_gradient );
              f                 = trial_f;
              gradient_norm     = m_projected_gradient.norm();
              gradient_norm_inf = m_projected_gradient.template lpNorm<Eigen::Infinity>();
              step_norm         = candidate_step;
              lambda            = Real( 0 );
              return CertificateAction::escaped;
            }
            alpha *= Real( 0.5 );
          }
        }
        return CertificateAction::failed;
      };

      if ( !callback( fill_result( Utils::Status::unknown, 0 ) ) ) { return fill_result( Utils::Status::user, 0 ); }

      for ( int iteration = 1; iteration <= m_options.max_iterations; ++iteration )
      {
        if ( gradient_norm_inf < tolerance )
        {
          CertificateAction const action = certify_or_escape();
          if ( action == CertificateAction::certified ) return fill_result( Utils::Status::converged, iteration - 1 );
          if ( action == CertificateAction::failed )
            return fill_result(
              m_hessian.allFinite() ? Utils::Status::no_progress : Utils::Status::non_finite_hessian,
              iteration - 1 );
          if ( !callback( fill_result( Utils::Status::unknown, iteration - 1 ) ) )
            return fill_result( Utils::Status::user, iteration - 1 );
          continue;
        }

        hessian_estimate = max( m_options.hessian_lipschitz_min, m_options.regularization_decrease * hessian_estimate );
        bool accepted    = false;

        hessian( m_x, m_hessian );
        ++hessian_evaluations;
        if ( !m_hessian.allFinite() ) return fill_result( Utils::Status::non_finite_hessian, iteration - 1 );

        // Try the inexpensive pure Newton step first when the free Hessian is
        // positive definite.  Model-only backtracking avoids repeated f/g calls.
        if ( m_options.enable_primary_semismooth )
        {
          ++primary_semismooth_attempts;
          m_shifted_hessian = Real( 0.5 ) * ( m_hessian + m_hessian.transpose() ).eval();
          m_direction       = m_projected_gradient;
          for ( Eigen::Index i = 0; i < m_x.size(); ++i )
          {
            bool const fixed        = lower[i] == upper[i];
            bool const active_lower = m_x[i] <= lower[i] && m_gradient[i] >= Real( 0 );
            bool const active_upper = m_x[i] >= upper[i] && m_gradient[i] <= Real( 0 );
            if ( fixed || active_lower || active_upper )
            {
              m_shifted_hessian.row( i ).setZero();
              m_shifted_hessian.col( i ).setZero();
              m_shifted_hessian( i, i ) = Real( 1 );
              m_direction[i]            = Real( 0 );
            }
          }

          Eigen::LDLT<Mat> ldlt( m_shifted_hessian );
          bool             primary_ok = ldlt.info() == Eigen::Success && ldlt.isPositive();
          if ( primary_ok )
          {
            m_direction = ldlt.solve( m_direction );
            primary_ok  = ldlt.info() == Eigen::Success && m_direction.allFinite();
          }

          Real model_reduction = Real( 0 );
          if ( primary_ok )
          {
            Real alpha = Real( 1 );
            primary_ok = false;
            for ( int backtrack = 0; backtrack <= m_options.max_primary_model_backtracking; ++backtrack )
            {
              m_trial        = ( m_x - alpha * m_direction ).cwiseMax( lower ).cwiseMin( upper );
              Vec const step = m_trial - m_x;
              if ( !( step.squaredNorm() > Real( 0 ) ) ) break;
              Real const slope           = m_gradient.dot( step );
              Real const quadratic_model = slope + Real( 0.5 ) * step.dot( m_hessian * step );
              model_reduction            = -quadratic_model;
              if (
                slope < Real( 0 ) && model_reduction > Real( 0 ) && quadratic_model <= m_options.model_armijo * slope )
              {
                primary_ok = true;
                break;
              }
              alpha *= m_options.primary_model_shrink;
            }
          }

          if (
            primary_ok &&
            ( m_options.max_function_evaluations < 0 || function_evaluations < m_options.max_function_evaluations ) )
          {
            Real const trial_f = objective( m_trial );
            ++function_evaluations;
            Real const roundoff = m_options.roundoff_factor * std::numeric_limits<Real>::epsilon() *
                                  max( { Real( 1 ), std::abs( f ), std::abs( trial_f ), std::abs( model_reduction ) } );
            Real const ratio    = ( f - trial_f + roundoff ) / ( model_reduction + roundoff );
            if ( Utils::is_finite( trial_f ) && ratio >= m_options.primary_acceptance_threshold )
            {
              gradient( m_trial, m_trial_gradient );
              ++gradient_evaluations;
              if ( !m_trial_gradient.allFinite() )
                return fill_result( Utils::Status::non_finite_gradient, iteration - 1 );
              m_trial_projected_gradient = m_trial - ( m_trial - m_trial_gradient ).cwiseMax( lower ).cwiseMin( upper );
              step_norm                  = ( m_trial - m_x ).norm();
              m_x.swap( m_trial );
              m_gradient.swap( m_trial_gradient );
              m_projected_gradient.swap( m_trial_projected_gradient );
              f                 = trial_f;
              gradient_norm     = m_projected_gradient.norm();
              gradient_norm_inf = m_projected_gradient.template lpNorm<Eigen::Infinity>();
              lambda            = Real( 0 );
              ++primary_semismooth_steps;
              accepted = true;
            }
          }
          if ( !accepted ) ++primary_semismooth_rejected;
        }

        for ( int sub_iteration = 0; !accepted && sub_iteration < m_options.max_sub_iterations; ++sub_iteration )
        {
          lambda = sqrt( hessian_estimate * gradient_norm );

          m_shifted_hessian = m_hessian;
          m_shifted_hessian.diagonal().array() += lambda;
          m_direction = m_projected_gradient;

          // Variables satisfying the outward-pointing KKT condition are fixed
          // for this Newton system.  Zeroing both the row and column produces
          // the reduced free-variable Newton equation without reallocations.
          for ( Eigen::Index i = 0; i < m_x.size(); ++i )
          {
            bool const fixed        = lower[i] == upper[i];
            bool const active_lower = m_x[i] <= lower[i] && m_gradient[i] >= Real( 0 );
            bool const active_upper = m_x[i] >= upper[i] && m_gradient[i] <= Real( 0 );
            if ( fixed || active_lower || active_upper )
            {
              m_shifted_hessian.row( i ).setZero();
              m_shifted_hessian.col( i ).setZero();
              m_shifted_hessian( i, i ) = Real( 1 );
              m_direction[i]            = Real( 0 );
            }
          }

          Eigen::FullPivLU<Mat> factorization( m_shifted_hessian );
          if ( !factorization.isInvertible() )
          {
            hessian_estimate *= m_options.factorization_increase;
            ++rejected_steps;
            ++fallback_steps;
            continue;
          }

          m_direction = factorization.solve( m_direction );
          if ( !m_direction.allFinite() )
          {
            hessian_estimate *= m_options.factorization_increase;
            ++rejected_steps;
            ++fallback_steps;
            continue;
          }

          m_trial     = ( m_x - m_direction ).cwiseMax( lower ).cwiseMin( upper );
          m_direction = m_x - m_trial;

          Real const model_slope     = -m_gradient.dot( m_direction );
          Real const quadratic_model = model_slope + Real( 0.5 ) * m_direction.dot( m_hessian * m_direction );
          if (
            m_options.enable_model_safeguard &&
            ( !( model_slope < Real( 0 ) ) || quadratic_model > m_options.model_armijo * model_slope ) )
          {
            hessian_estimate *= m_options.regularization_increase;
            ++rejected_steps;
            ++model_rejections;
            continue;
          }

          if ( m_options.max_function_evaluations >= 0 && function_evaluations >= m_options.max_function_evaluations )
          {
            return fill_result( Utils::Status::max_function_evaluations, iteration - 1 );
          }

          Real const trial_f = objective( m_trial );
          ++function_evaluations;
          if ( !Utils::is_finite( trial_f ) )
          {
            hessian_estimate *= m_options.factorization_increase;
            ++rejected_steps;
            continue;
          }

          gradient( m_trial, m_trial_gradient );
          ++gradient_evaluations;
          if ( !m_trial_gradient.allFinite() )
          {
            hessian_estimate *= m_options.factorization_increase;
            ++rejected_steps;
            continue;
          }

          m_trial_projected_gradient     = m_trial - ( m_trial - m_trial_gradient ).cwiseMax( lower ).cwiseMin( upper );
          Real const trial_gradient_norm = m_trial_projected_gradient.norm();
          step_norm                      = m_direction.norm();
          Real const lambda_step         = lambda * step_norm;
          if (
            trial_gradient_norm <= m_options.gradient_acceptance_factor * lambda_step &&
            trial_f <= f - m_options.decrease_factor * lambda_step * step_norm )
          {
            m_x.swap( m_trial );
            m_gradient.swap( m_trial_gradient );
            m_projected_gradient.swap( m_trial_projected_gradient );
            f                 = trial_f;
            gradient_norm     = trial_gradient_norm;
            gradient_norm_inf = m_projected_gradient.template lpNorm<Eigen::Infinity>();
            hessian_estimate  = max(
              m_options.hessian_lipschitz_min,
              m_options.regularization_decrease * hessian_estimate );
            accepted = true;
            break;
          }
          hessian_estimate *= m_options.regularization_increase;
          ++rejected_steps;
        }

        // A projected-gradient Armijo step is a safe BBOX fallback when the
        // regularized Newton model cannot produce an acceptable feasible step.
        if ( !accepted && m_options.enable_gradient_rescue )
        {
          Real alpha = Real( 1 );
          for ( int rescue = 0; rescue < m_options.max_gradient_rescue; ++rescue )
          {
            m_trial     = ( m_x - alpha * m_gradient ).cwiseMax( lower ).cwiseMin( upper );
            m_direction = m_x - m_trial;
            step_norm   = m_direction.norm();
            if ( !( step_norm > Real( 0 ) ) ) break;

            Real const slope           = m_gradient.dot( m_trial - m_x );
            Real const quadratic_model = slope + Real( 0.5 ) * ( m_trial - m_x ).dot( m_hessian * ( m_trial - m_x ) );
            if ( !( slope < Real( 0 ) ) || quadratic_model > m_options.model_armijo * slope )
            {
              alpha *= m_options.gradient_rescue_shrink;
              ++model_rejections;
              continue;
            }

            if ( m_options.max_function_evaluations >= 0 && function_evaluations >= m_options.max_function_evaluations )
            {
              return fill_result( Utils::Status::max_function_evaluations, iteration - 1 );
            }

            Real const trial_f = objective( m_trial );
            ++function_evaluations;
            if (
              Utils::is_finite( trial_f ) && slope < Real( 0 ) &&
              trial_f <= f + m_options.gradient_rescue_armijo * slope )
            {
              gradient( m_trial, m_trial_gradient );
              ++gradient_evaluations;
              if ( !m_trial_gradient.allFinite() )
                return fill_result( Utils::Status::non_finite_gradient, iteration - 1 );

              m_trial_projected_gradient = m_trial - ( m_trial - m_trial_gradient ).cwiseMax( lower ).cwiseMin( upper );
              m_x.swap( m_trial );
              m_gradient.swap( m_trial_gradient );
              m_projected_gradient.swap( m_trial_projected_gradient );
              f                 = trial_f;
              gradient_norm     = m_projected_gradient.norm();
              gradient_norm_inf = m_projected_gradient.template lpNorm<Eigen::Infinity>();
              ++fallback_steps;
              ++gradient_rescue_steps;
              accepted = true;
              break;
            }

            alpha *= m_options.gradient_rescue_shrink;
            ++rejected_steps;
          }
        }

        if ( !accepted ) return fill_result( Utils::Status::no_progress, iteration - 1 );
        if ( !callback( fill_result( Utils::Status::unknown, iteration ) ) )
        {
          return fill_result( Utils::Status::user, iteration );
        }
      }

      if ( gradient_norm_inf <= tolerance )
      {
        CertificateAction const action = certify_or_escape();
        if ( action == CertificateAction::certified )
          return fill_result( Utils::Status::converged, m_options.max_iterations );
      }
      return fill_result( Utils::Status::max_iterations, m_options.max_iterations );
    }

    Options m_options{};
    Vec     m_x{};
    Vec     m_trial{};
    Vec     m_gradient{};
    Vec     m_trial_gradient{};
    Vec     m_projected_gradient{};
    Vec     m_trial_projected_gradient{};
    Vec     m_direction{};
    Mat     m_hessian{};
    Mat     m_shifted_hessian{};
  };

}  // namespace Utils

#endif
