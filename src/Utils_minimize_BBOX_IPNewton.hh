/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2025                                                      |
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

//
// file: Utils_minimize_BBOX_IPNewton.hh
//

#pragma once

#ifndef UTILS_MINIMIZE_BBOX_IPNEWTON_HH
#define UTILS_MINIMIZE_BBOX_IPNEWTON_HH

#include "Utils_minimize_BBOX.hh"
#include "Utils_LBFGS.hh"
#include "Utils_minimize_Newton.hh"
#include <random>
#include <algorithm>
#include <deque>
#include <optional>

namespace Utils
{

  // ===========================================================================
  // Interior Point Newton Method for Box Constraints - ROBUST VERSION
  // ===========================================================================
  template <typename Scalar = double> class minimize_BBOX_IPNewton
  {
  public:
    using integer        = Eigen::Index;
    using Vector         = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix         = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SparseMatrix   = Eigen::SparseMatrix<Scalar>; // kept for compat, not used in new interface
    using ConstVectorRef = Eigen::Ref<Vector const>;
    using VectorRef      = Eigen::Ref<Vector>;
    using ConstMatrixRef = Eigen::Ref<Matrix const>;
    using MatrixRef      = Eigen::Ref<Matrix>;
    using Callback       = std::function<Scalar( Vector const &, Vector *, Matrix * )>;

    enum class Status
    {
      CONVERGED         = 0,
      MAX_ITERATIONS    = 1,
      BARRIER_FAILED    = 2,
      DUAL_INFEASIBLE   = 3,
      PRIMAL_INFEASIBLE = 4,
      FAILED            = 5,
      NOT_STARTED       = 6
    };

    static std::string to_string( Status s )
    {
      switch ( s )
      {
        case Status::CONVERGED: return "CONVERGED";
        case Status::MAX_ITERATIONS: return "MAX_ITERATIONS";
        case Status::BARRIER_FAILED: return "BARRIER_FAILED";
        case Status::DUAL_INFEASIBLE: return "DUAL_INFEASIBLE";
        case Status::PRIMAL_INFEASIBLE: return "PRIMAL_INFEASIBLE";
        case Status::FAILED: return "FAILED";
        case Status::NOT_STARTED: return "NOT_STARTED";
        default: return "UNKNOWN";
      }
    }

    struct Options
    {
      // General parameters
      integer max_outer_iterations = 100;   // Increased for difficult problems
      integer max_inner_iterations = 500;   // Increased for difficult subproblems
      Scalar  tol                  = 1e-6;  // Relaxed tolerance
      Scalar  f_tol                = 1e-10;
      Scalar  x_tol                = 1e-8;
      Scalar  g_max                = 1e-4;  // Relaxed gradient tolerance

      // Barrier parameter
      Scalar mu_init            = 1.0;    // Start with larger mu for stability
      Scalar mu_min             = 1e-10;  // Less aggressive minimum
      Scalar mu_decrease_factor = 0.5;    // Slower reduction
      Scalar sigma              = 0.5;

      // Feasibility
      Scalar epsilon_feas   = 1e-6;  // Relaxed feasibility margin
      Scalar epsilon_infeas = 1e-4;  // Relaxed infeasibility tolerance

      // Newton step parameters
      Scalar  alpha_min                  = 1e-8;
      Scalar  alpha_reduction            = 0.5;
      integer max_line_search_iterations = 20;

      // Predictor-corrector
      bool   use_predictor_corrector = true;
      Scalar corrector_max_ratio     = 10.0;

      // Verbosity
      integer verbosity = 1;

      // Adaptive parameters
      bool    adaptive_mu_update   = true;
      integer max_barrier_failures = 5;

      // Line search enhancement
      Scalar armijo_c1 = 1e-4;

      // Progress monitoring
      integer max_no_progress_iterations = 15;  // Increased
      Scalar  progress_tol               = 1e-8;

      // Recovery strategies
      bool    enable_recovery       = true;  // Enable recovery from failed subproblems
      Scalar  recovery_factor       = 10.0;  // Factor to increase mu when recovering
      integer max_recovery_attempts = 3;     // Max recovery attempts per barrier failure

      // --- Unified setters (same name across all BBOX solvers) ---
      void set_tolerances( Scalar t )
      {
        tol                  = t;
        f_tol                = t * Scalar( 1e-4 );
        x_tol                = t * Scalar( 1e-2 );
        g_max                = t;
        epsilon_feas         = t;
        epsilon_infeas       = t * Scalar( 1e2 );
      }
      void set_max_iterations( integer n ) { max_outer_iterations = n; }
    };

    struct Result
    {
      Vector  x;
      Status  status{ Status::NOT_STARTED };
      Scalar  objective{ std::numeric_limits<Scalar>::quiet_NaN() };
      Scalar  projected_gradient_norm{ std::numeric_limits<Scalar>::infinity() };
      Scalar  primal_feasibility{ std::numeric_limits<Scalar>::infinity() };
      Scalar  dual_infeasibility{ std::numeric_limits<Scalar>::infinity() };
      Scalar  duality_gap{ std::numeric_limits<Scalar>::infinity() };
      Scalar  barrier_parameter{ Scalar( 0 ) };
      integer iterations{ 0 };
      integer inner_iterations{ 0 };
      integer function_evaluations{ 0 };
      integer hessian_evaluations{ 0 };

      [[nodiscard]] bool solved() const noexcept { return status == Status::CONVERGED; }
    };

  private:
    Options m_opts;
    integer m_dimension{ 0 };
    Vector  m_lower;
    Vector  m_upper;
    bool    m_bounds_initialized{ false };
    Scalar  m_epsi{ std::numeric_limits<Scalar>::epsilon() };

    // Results
    Status  m_status{ Status::NOT_STARTED };
    integer m_outer_iterations       = 0;
    integer m_total_inner_iterations = 0;
    integer m_function_evals         = 0;
    integer m_hessian_evals          = 0;
    Scalar  m_final_f                = 0;
    Scalar  m_initial_f              = 0;
    Scalar  m_final_mu               = 0;
    Scalar  m_duality_gap            = 0;
    Scalar  m_primal_infeasibility   = 0;
    Scalar  m_dual_infeasibility     = 0;
    Scalar  m_gradient_norm          = 0;
    Vector  m_x;
    Vector  m_lambda_lower;
    Vector  m_lambda_upper;
    integer m_newton_steps    = 0;
    integer m_gradient_steps  = 0;
    integer m_centering_steps = 0;

    // Best point tracking
    Vector m_best_x;
    Scalar m_best_f  = std::numeric_limits<Scalar>::max();
    Scalar m_last_df = 0;

    void reset_results()
    {
      m_status                 = Status::NOT_STARTED;
      m_outer_iterations       = 0;
      m_total_inner_iterations = 0;
      m_function_evals         = 0;
      m_hessian_evals          = 0;
      m_final_f                = 0;
      m_initial_f              = 0;
      m_final_mu               = 0;
      m_duality_gap            = 0;
      m_primal_infeasibility   = 0;
      m_dual_infeasibility     = 0;
      m_gradient_norm          = 0;
      m_x.resize( 0 );
      m_lambda_lower.resize( 0 );
      m_lambda_upper.resize( 0 );
      m_newton_steps    = 0;
      m_gradient_steps  = 0;
      m_centering_steps = 0;
      m_best_f          = std::numeric_limits<Scalar>::max();
      m_best_x.resize( 0 );
      m_last_df = 0;
    }

    // Logarithmic barrier function evaluator
    class BarrierEvaluator
    {
    private:
      Callback       m_callback;
      Vector const & m_lower;
      Vector const & m_upper;
      Scalar         m_mu;
      Scalar         m_epsilon_feas;

    public:
      BarrierEvaluator(
        Callback const & callback,
        Vector const &   lower,
        Vector const &   upper,
        Scalar           mu,
        Scalar           epsilon_feas )
        : m_callback( callback ), m_lower( lower ), m_upper( upper ), m_mu( mu ), m_epsilon_feas( epsilon_feas )
      {
      }

      Scalar operator()( Vector const & x, Vector * grad, Matrix * hess ) const
      {
        integer n = x.size();
        Scalar  f = m_callback( x, grad, hess );

        Scalar barrier = 0;
        for ( integer i = 0; i < n; ++i )
        {
          if ( std::isfinite( m_lower[i] ) )
          {
            Scalar const s_lower = std::max( m_epsilon_feas, x[i] - m_lower[i] );
            barrier -= m_mu * std::log( s_lower );
          }
          if ( std::isfinite( m_upper[i] ) )
          {
            Scalar const s_upper = std::max( m_epsilon_feas, m_upper[i] - x[i] );
            barrier -= m_mu * std::log( s_upper );
          }
        }

        f += barrier;

        if ( grad )
        {
          for ( integer i = 0; i < n; ++i )
          {
            if ( std::isfinite( m_lower[i] ) )
            {
              Scalar const s_lower = std::max( m_epsilon_feas, x[i] - m_lower[i] );
              ( *grad )[i] -= m_mu / s_lower;
            }
            if ( std::isfinite( m_upper[i] ) )
            {
              Scalar const s_upper = std::max( m_epsilon_feas, m_upper[i] - x[i] );
              ( *grad )[i] += m_mu / s_upper;
            }
          }
        }

        if ( hess )
        {
          // hess already contains dense Hessian of original objective (if any)
          // add barrier diagonal contribution in place, no allocation via triplets
          for ( integer i = 0; i < n; ++i )
          {
            Scalar diag_term = 0;
            if ( std::isfinite( m_lower[i] ) )
            {
              Scalar const s_lower = std::max( m_epsilon_feas, x[i] - m_lower[i] );
              diag_term += m_mu / ( s_lower * s_lower );
            }
            if ( std::isfinite( m_upper[i] ) )
            {
              Scalar const s_upper = std::max( m_epsilon_feas, m_upper[i] - x[i] );
              diag_term += m_mu / ( s_upper * s_upper );
            }
            if ( diag_term != Scalar( 0 ) ) ( *hess )( i, i ) += diag_term;
          }
        }

        return f;
      }
    };

    // KKT conditions evaluator
    struct KKTResiduals
    {
      Scalar primal_feasibility;
      Scalar dual_feasibility;
      Scalar complementarity;
      Scalar duality_gap;
      Scalar gradient_norm;

      KKTResiduals()
        : primal_feasibility( 0 ), dual_feasibility( 0 ), complementarity( 0 ), duality_gap( 0 ), gradient_norm( 0 )
      {
      }
    };

    KKTResiduals compute_KKT_residuals(
      Vector const & x,
      Vector const & grad,
      Vector const & lam_lower,
      Vector const & lam_upper ) const
    {
      KKTResiduals res;
      integer      n = x.size();

      // Stationarity of the Lagrangian: g-lambda_lower+lambda_upper.
      res.gradient_norm = ( grad - lam_lower + lam_upper ).norm();

      // Primal feasibility
      Scalar max_primal_viol = 0;
      for ( integer i = 0; i < n; ++i )
      {
        if ( std::isfinite( m_lower[i] ) ) max_primal_viol = std::max( max_primal_viol, m_lower[i] - x[i] );
        if ( std::isfinite( m_upper[i] ) ) max_primal_viol = std::max( max_primal_viol, x[i] - m_upper[i] );
      }
      res.primal_feasibility = std::max( Scalar( 0 ), max_primal_viol );

      // Dual feasibility
      Scalar max_dual_viol = 0;
      for ( integer i = 0; i < n; ++i )
      {
        max_dual_viol = std::max( max_dual_viol, -lam_lower[i] );
        max_dual_viol = std::max( max_dual_viol, -lam_upper[i] );
      }
      res.dual_feasibility = std::max( Scalar( 0 ), max_dual_viol );

      // Complementarity
      Scalar max_comp  = 0;
      Scalar total_gap = 0;
      for ( integer i = 0; i < n; ++i )
      {
        Scalar const comp_lower = std::isfinite( m_lower[i] ) ? std::abs( lam_lower[i] * ( x[i] - m_lower[i] ) ) : 0;
        Scalar const comp_upper = std::isfinite( m_upper[i] ) ? std::abs( lam_upper[i] * ( m_upper[i] - x[i] ) ) : 0;
        max_comp          = std::max( max_comp, std::max( comp_lower, comp_upper ) );
        total_gap += comp_lower + comp_upper;
      }
      res.complementarity = max_comp;
      res.duality_gap     = total_gap;

      return res;
    }

    void compute_dual_variables_corrected(
      Vector const & x,
      Vector const & grad,
      Vector &       lam_lower,
      Vector &       lam_upper ) const
    {
      integer n = x.size();
      lam_lower.resize( n );
      lam_upper.resize( n );

      for ( integer i = 0; i < n; ++i )
      {
        // Per lower bound: λ_lower > 0 solo se x_i ≈ lower_i E grad_i < 0
        if ( std::isfinite( m_lower[i] ) && x[i] <= m_lower[i] + m_opts.epsilon_feas )
        {
          lam_lower[i] = std::max( Scalar( 0 ), grad[i] );
          lam_upper[i] = 0;
        }
        // Per upper bound: λ_upper > 0 solo se x_i ≈ upper_i E grad_i > 0
        else if ( std::isfinite( m_upper[i] ) && x[i] >= m_upper[i] - m_opts.epsilon_feas )
        {
          lam_lower[i] = 0;
          lam_upper[i] = std::max( Scalar( 0 ), -grad[i] );
        }
        // Variabile libera: moltiplicatori = 0
        else
        {
          lam_lower[i] = 0;
          lam_upper[i] = 0;
        }

        // Stabilità numerica: assicurarsi che non siano troppo piccoli
        if ( lam_lower[i] < m_opts.epsilon_feas ) lam_lower[i] = 0;
        if ( lam_upper[i] < m_opts.epsilon_feas ) lam_upper[i] = 0;
      }
    }

    bool is_strictly_feasible( Vector const & x ) const
    {
      for ( integer i = 0; i < x.size(); ++i )
      {
        if ( std::isfinite( m_lower[i] ) && x[i] <= m_lower[i] + m_epsi ) return false;
        if ( std::isfinite( m_upper[i] ) && x[i] >= m_upper[i] - m_epsi ) return false;
      }
      return true;
    }

    bool make_strictly_feasible( Vector & x ) const
    {
      bool   modified      = false;
      Scalar margin_factor = 0.1;

      for ( integer i = 0; i < x.size(); ++i )
      {
        bool const finite_lower = std::isfinite( m_lower[i] );
        bool const finite_upper = std::isfinite( m_upper[i] );
        Scalar const scale = std::max( Scalar( 1 ), std::abs( x[i] ) );
        Scalar const margin = finite_lower && finite_upper
                                ? std::max( Scalar( 1e-3 ), margin_factor * ( m_upper[i] - m_lower[i] ) )
                                : std::max( Scalar( 1e-3 ), margin_factor * scale );

        if ( finite_lower && x[i] <= m_lower[i] + m_opts.epsilon_feas )
        {
          x[i]     = m_lower[i] + margin;
          modified = true;
        }
        else if ( finite_upper && x[i] >= m_upper[i] - m_opts.epsilon_feas )
        {
          x[i]     = m_upper[i] - margin;
          modified = true;
        }
      }
      return modified;
    }

    [[nodiscard]] bool has_finite_bound() const
    {
      for ( integer i = 0; i < m_lower.size(); ++i )
        if ( std::isfinite( m_lower[i] ) || std::isfinite( m_upper[i] ) ) return true;
      return false;
    }

    void print_header( integer n, Scalar f0 ) const
    {
      if ( m_opts.verbosity < 1 ) return;
      fmt::print( "╔══════════════════════════════════════════════════════════════════════╗\n" );
      fmt::print( "║                 IPNewton (Logarithmic Barrier) - ROBUST             ║\n" );
      fmt::print( "╠══════════════════════════════════════════════════════════════════════╣\n" );
      fmt::print(
        "║ Dimension: {:4d}     Outer Max: {:3d}     Tol: {:<12.4g}     ║\n",
        n,
        m_opts.max_outer_iterations,
        m_opts.tol );
      fmt::print(
        "║ Barrier:  μ₀={:<12.4g}  μ_min={:<12.4g}  Adaptive: {:<3} ║\n",
        m_opts.mu_init,
        m_opts.mu_min,
        m_opts.adaptive_mu_update ? "ON" : "OFF" );
      fmt::print( "║ Initial F = {:<15.6g}                                  ║\n", f0 );
      fmt::print( "╚══════════════════════════════════════════════════════════════════════╝\n" );
    }

    void print_outer_iteration(
      integer iter,
      Scalar  mu,
      Scalar  f,
      Scalar  gap,
      Scalar  grad_norm,
      Scalar  primal_inf,
      Scalar  dual_inf ) const
    {
      if ( m_opts.verbosity < 1 ) return;

      fmt::print(
        "[Outer {:3d}] μ={:.2e} F={:.6e} ‖g‖={:.2e} Gap={:.2e} Primal={:.2e} Dual={:.2e}\n",
        iter,
        mu,
        f,
        grad_norm,
        gap,
        primal_inf,
        dual_inf );
    }

    void print_convergence_info( Status status, integer outer_iter, Scalar f, Scalar gap ) const
    {
      if ( m_opts.verbosity < 1 ) return;

      auto color = ( status == Status::CONVERGED )        ? PrintColors::SUCCESS
                   : ( status == Status::MAX_ITERATIONS ) ? PrintColors::WARNING
                                                          : PrintColors::ERROR;

      fmt::print( color, "[CONV] Outer={} F={:.6e} Gap={:.2e} Status={}\n", outer_iter, f, gap, to_string( status ) );
    }

    // Solve barrier subproblem for fixed mu
    bool solve_barrier_subproblem(
      Vector &                 x,
      Scalar                   mu,
      BarrierEvaluator const & evaluator,
      integer &                inner_iterations ) const
    {
      typename Newton_minimizer<Scalar>::Options newton_opts;

      newton_opts.max_iter  = m_opts.max_inner_iterations;
      newton_opts.g_tol     = has_finite_bound() ? std::max( m_opts.tol * 0.1, mu * 0.1 ) : m_opts.tol;
      newton_opts.f_tol     = m_opts.f_tol * 100.0;
      newton_opts.verbosity = ( m_opts.verbosity > 2 ) ? m_opts.verbosity - 2 : 0;
      Newton_minimizer<Scalar> newton_solver( newton_opts );

      // Set bounds with safe margin
      Vector lower_with_margin = m_lower.array() + m_opts.epsilon_feas;
      Vector upper_with_margin = m_upper.array() - m_opts.epsilon_feas;

      newton_solver.set_bounds( lower_with_margin, upper_with_margin );

      // Create barrier callback (dense, no SparseMatrix)
      auto barrier_callback = [&]( Vector const & x_eval, Vector * grad, Matrix * hess ) -> Scalar
      { return evaluator( x_eval, grad, hess ); };

      newton_solver.minimize( x, barrier_callback );

      inner_iterations = newton_solver.iterations();

      bool converged = newton_solver.status() == Newton_minimizer<Scalar>::Status::CONVERGED;

      if ( converged )
      {
        x = newton_solver.solution();
        return true;
      }

      return false;
    }

    // Recovery strategy for failed barrier subproblems
    bool recover_from_barrier_failure( Vector & x, Scalar & mu, Callback const & callback )
    {
      if ( !m_opts.enable_recovery ) return false;

      if ( m_opts.verbosity >= 2 ) fmt::print( "  Attempting recovery from barrier failure...\n" );

      // 1. Ripristina il miglior punto trovato
      if ( m_best_f < std::numeric_limits<Scalar>::max() )
      {
        x = m_best_x;
        if ( m_opts.verbosity >= 2 ) fmt::print( "    Restored best point f={:.6e}\n", m_best_f );
      }

      // 2. Aumenta μ per stabilizzare
      Scalar old_mu = mu;
      mu            = std::min( mu * m_opts.recovery_factor, m_opts.mu_init );
      if ( m_opts.verbosity >= 2 ) fmt::print( "    Increased μ from {:.2e} to {:.2e}\n", old_mu, mu );

      // 3. Valuta gradiente al punto corrente
      Vector grad( x.size() );
      Scalar f = callback( x, &grad, nullptr );

      // 4. Calcola direzione di gradiente proiettata
      Vector dir = -grad;
      for ( integer i = 0; i < x.size(); ++i )
      {
        if ( std::isfinite( m_lower[i] ) && x[i] <= m_lower[i] + m_opts.epsilon_feas && dir[i] < 0 )
          dir[i] = 0;
        else if ( std::isfinite( m_upper[i] ) && x[i] >= m_upper[i] - m_opts.epsilon_feas && dir[i] > 0 )
          dir[i] = 0;
      }

      // Normalizza la direzione
      Scalar dir_norm = dir.norm();
      if ( dir_norm < m_epsi ) return false;  // Nessuna direzione valida
      dir /= dir_norm;

      // 5. Line search semplice
      Scalar alpha   = 1.0;
      Scalar f0      = f;
      bool   success = false;

      for ( integer ls = 0; ls < 10; ++ls )
      {
        Vector x_trial = x + alpha * dir;

        // Assicura fattibilità
        for ( integer i = 0; i < x_trial.size(); ++i )
        {
          if ( std::isfinite( m_lower[i] ) && x_trial[i] < m_lower[i] )
            x_trial[i] = m_lower[i] + m_opts.epsilon_feas;
          else if ( std::isfinite( m_upper[i] ) && x_trial[i] > m_upper[i] )
            x_trial[i] = m_upper[i] - m_opts.epsilon_feas;
        }

        Scalar f_trial = callback( x_trial, nullptr, nullptr );

        if ( f_trial < f0 - m_opts.armijo_c1 * alpha * grad.dot( dir ) )
        {
          x       = x_trial;
          success = true;
          if ( m_opts.verbosity >= 2 )
            fmt::print( "    Recovery successful with α={:.2e}, f={:.6e}→{:.6e}\n", alpha, f0, f_trial );
          break;
        }
        alpha *= 0.5;
      }

      return success;
    }

    Scalar compute_robust_mu_update(
      Scalar current_mu,
      Scalar complementarity,
      Scalar primal_inf,
      Scalar dual_inf,
      Scalar gradient_norm ) const
    {
      if ( !m_opts.adaptive_mu_update ) return std::max( m_opts.mu_min, m_opts.mu_decrease_factor * current_mu );

      // Se la complementarità è grande rispetto a μ, riduci μ lentamente
      Scalar comp_ratio = complementarity / current_mu;

      Scalar reduction = m_opts.mu_decrease_factor;

      if ( comp_ratio > 10.0 )
      {
        // Complementarità grande → riduci μ moderatamente
        reduction = 0.7;
      }
      else if ( comp_ratio < 0.1 && gradient_norm < m_opts.g_max )
      {
        // Complementarità piccola E gradiente buono → riduci μ rapidamente
        reduction = 0.2;
      }
      else if ( primal_inf > 1e-4 || dual_inf > 1e-4 )
      {
        // Infattibilità → riduci μ lentamente
        reduction = 0.9;
      }

      reduction = std::clamp( reduction, 0.1, 0.9 );
      return std::max( m_opts.mu_min, reduction * current_mu );
    }

    bool check_convergence( KKTResiduals const & kkt, Scalar mu ) const
    {
      // Criterio 1: KKT soddisfatte con tolleranza
      Scalar kkt_tol = std::max( m_opts.tol * 10, mu * 100 );

      bool kkt_satisfied = kkt.primal_feasibility <= m_opts.epsilon_infeas &&
                           kkt.dual_feasibility <= m_opts.epsilon_infeas && kkt.complementarity <= kkt_tol &&
                           kkt.gradient_norm <= m_opts.g_max;

      // Criterio 2: μ molto piccolo E progresso minimo
      bool mu_small_and_stable = mu <= m_opts.mu_min * 10 &&
                                 std::abs( m_last_df ) < m_opts.progress_tol * ( 1 + std::abs( m_final_f ) );

      return kkt_satisfied || mu_small_and_stable;
    }

  public:
    explicit minimize_BBOX_IPNewton( integer dimension = 0, Options opts = {} )
      : m_opts( std::move( opts ) )
    { resize( dimension ); }

    explicit minimize_BBOX_IPNewton( Options opts ) : minimize_BBOX_IPNewton( 0, std::move( opts ) ) {}

    [[nodiscard]] Options &       options() noexcept { return m_opts; }
    [[nodiscard]] Options const & options() const noexcept { return m_opts; }

    // Unified setters (same name across all BBOX solvers)
    void set_tolerances( Scalar tol ) { m_opts.set_tolerances( tol ); }
    void set_max_iterations( integer n ) { m_opts.set_max_iterations( n ); }

    void resize( integer dimension )
    {
      Utils::Check( dimension >= 0, "minimize_BBOX_IPNewton::resize: negative dimension" );
      if ( dimension != m_dimension )
      {
        m_lower.resize( 0 );
        m_upper.resize( 0 );
        m_bounds_initialized = false;
      }
      m_dimension = dimension;
    }

    void set_bounds( ConstVectorRef lower, ConstVectorRef upper )
    {
      Utils::Check(
        lower.size() == upper.size() && ( m_dimension == 0 || lower.size() == m_dimension ),
        "minimize_BBOX_IPNewton::set_bounds: incompatible vector dimensions" );
      Utils::Check(
        !lower.hasNaN() && !upper.hasNaN() && ( lower.array() < upper.array() ).all(),
        "minimize_BBOX_IPNewton::set_bounds: bounds must not be NaN and lower must be strictly smaller than upper" );
      m_dimension = lower.size();
      m_lower = lower;
      m_upper = upper;
      m_bounds_initialized = true;
    }

    // Getters
    Status         status() const { return m_status; }
    integer        outer_iterations() const { return m_outer_iterations; }
    integer        total_inner_iterations() const { return m_total_inner_iterations; }
    integer        function_evals() const { return m_function_evals; }
    integer        hessian_evals() const { return m_hessian_evals; }
    Scalar         final_f() const { return m_final_f; }
    Scalar         initial_f() const { return m_initial_f; }
    Scalar         final_mu() const { return m_final_mu; }
    Scalar         duality_gap() const { return m_duality_gap; }
    Scalar         primal_infeasibility() const { return m_primal_infeasibility; }
    Scalar         dual_infeasibility() const { return m_dual_infeasibility; }
    Scalar         gradient_norm() const { return m_gradient_norm; }
    Vector const & solution() const { return m_x; }
    Vector const & lambda_lower() const { return m_lambda_lower; }
    Vector const & lambda_upper() const { return m_lambda_upper; }
    integer        newton_steps() const { return m_newton_steps; }
    integer        gradient_steps() const { return m_gradient_steps; }
    integer        centering_steps() const { return m_centering_steps; }
    Vector const & best_solution() const { return m_best_x; }
    Scalar         best_f() const { return m_best_f; }

    [[nodiscard]] Result result() const
    {
      Result out;
      out.x                       = m_x;
      out.status                  = m_status;
      out.objective               = m_final_f;
      out.projected_gradient_norm = m_gradient_norm;
      out.primal_feasibility      = m_primal_infeasibility;
      out.dual_infeasibility      = m_dual_infeasibility;
      out.duality_gap             = m_duality_gap;
      out.barrier_parameter       = m_final_mu;
      out.iterations              = m_outer_iterations;
      out.inner_iterations        = m_total_inner_iterations;
      out.function_evaluations    = m_function_evals;
      out.hessian_evaluations     = m_hessian_evals;
      return out;
    }

    template <typename Problem>
      requires requires( Problem & problem, Vector const & x, Vector & gradient, Matrix & hessian ) {
        { problem.objective( x ) } -> std::convertible_to<Scalar>;
        problem.gradient( x, gradient );
        problem.hessian( x, hessian );
      }
    Result solve(
      Problem &      problem,
      Vector const & x0,
      Vector const & lower,
      Vector const & upper )
    {
      resize( x0.size() );
      set_bounds( lower, upper );
      Callback callback = [&problem]( Vector const & x, Vector * gradient, Matrix * hessian ) -> Scalar
      {
        if ( gradient != nullptr ) problem.gradient( x, *gradient );
        if ( hessian != nullptr ) problem.hessian( x, *hessian );
        return static_cast<Scalar>( problem.objective( x ) );
      };
      minimize( x0, callback );
      return result();
    }

    // --- unified dense lambda interface: f(x), grad(x,g), hess(x,H) ---
    template <typename Obj, typename Grad, typename Hess>
    Result solve(
      Obj &&         obj,
      Grad &&        grad,
      Hess &&        hess,
      ConstVectorRef x0,
      ConstVectorRef lower,
      ConstVectorRef upper )
    {
      struct Adapter
      {
        std::decay_t<Obj>  o;
        std::decay_t<Grad> g;
        std::decay_t<Hess> h;
        Scalar objective( Vector const & x ) { return static_cast<Scalar>( o( x ) ); }
        void gradient( Vector const & x, Vector & gg ) { g( x, gg ); }
        void hessian( Vector const & x, Matrix & HH ) { h( x, HH ); }
      } adapt{ std::forward<Obj>( obj ), std::forward<Grad>( grad ), std::forward<Hess>( hess ) };
      return solve( adapt, x0, lower, upper );
    }

    std::string summary() const
    {
      return fmt::format(
        "Status: {} | Outer iterations: {} | Inner iterations: {} | Function evaluations: {} | "
        "Hessian evaluations: {} | Final f: {:.6e} | Initial f: {:.6e} | Final μ: {:.1e} | "
        "Duality gap: {:.2e} | Primal infeasibility: {:.2e} | Dual infeasibility: {:.2e} | "
        "Gradient norm: {:.2e} | Newton steps: {} | Gradient steps: {} | Centering steps: {}",
        to_string( m_status ),
        m_outer_iterations,
        m_total_inner_iterations,
        m_function_evals,
        m_hessian_evals,
        m_final_f,
        m_initial_f,
        m_final_mu,
        m_duality_gap,
        m_primal_infeasibility,
        m_dual_infeasibility,
        m_gradient_norm,
        m_newton_steps,
        m_gradient_steps,
        m_centering_steps );
    }

    void minimize( Vector const & x0, Callback const & callback )
    {
      reset_results();

      Utils::Check( static_cast<bool>( callback ), "minimize_BBOX_IPNewton::minimize: callback is not initialized" );
      Utils::Check( x0.size() > 0 && x0.allFinite(), "minimize_BBOX_IPNewton::minimize: invalid initial point" );
      Utils::Check(
        m_bounds_initialized && m_dimension == x0.size() && m_lower.size() == x0.size() && m_upper.size() == x0.size(),
        "minimize_BBOX_IPNewton::minimize: bounds are not initialized or have the wrong dimension" );
      Utils::Check(
        !m_lower.hasNaN() && !m_upper.hasNaN() && ( m_lower.array() < m_upper.array() ).all(),
        "minimize_BBOX_IPNewton::minimize: invalid bounds" );
      Utils::Check(
        m_opts.max_outer_iterations > 0 && m_opts.max_inner_iterations > 0 && m_opts.tol >= Scalar( 0 ) &&
          m_opts.mu_init > Scalar( 0 ) && m_opts.mu_min > Scalar( 0 ) && m_opts.epsilon_feas > Scalar( 0 ),
        "minimize_BBOX_IPNewton::minimize: invalid solver options" );

      integer n = x0.size();
      Vector  x = x0;
      Vector  grad( n );

      if ( !is_strictly_feasible( x ) )
      {
        if ( m_opts.verbosity > 0 ) { fmt::print( "Adjusting initial point for feasibility...\n" ); }
        make_strictly_feasible( x );
      }

      Scalar f        = callback( x, &grad, nullptr );
      Utils::Check(
        std::isfinite( f ) && grad.size() == n && grad.allFinite(),
        "minimize_BBOX_IPNewton::minimize: callback returned an invalid objective or gradient" );
      m_initial_f     = f;
      integer f_evals = 1;
      integer h_evals = 0;

      m_best_f = f;
      m_best_x = x;

      print_header( n, m_initial_f );

      Vector lam_lower( n ), lam_upper( n );
      compute_dual_variables_corrected( x, grad, lam_lower, lam_upper );

      bool const barrier_active      = has_finite_bound();
      Scalar  mu                     = barrier_active ? m_opts.mu_init : Scalar( 0 );
      Status  status                 = Status::MAX_ITERATIONS;
      integer outer_iter             = 0;
      integer total_inner_iterations = 0;
      integer newton_steps           = 0;
      integer gradient_steps         = 0;
      integer centering_steps        = 0;

      integer no_progress_count = 0;
      integer barrier_failures  = 0;
      integer recovery_attempts = 0;

      for ( outer_iter = 0; outer_iter < m_opts.max_outer_iterations; ++outer_iter )
      {
        BarrierEvaluator evaluator( callback, m_lower, m_upper, mu, m_opts.epsilon_feas );

        integer inner_iterations = 0;

        bool success = solve_barrier_subproblem( x, mu, evaluator, inner_iterations );

        total_inner_iterations += inner_iterations;

        if ( !success )
        {
          barrier_failures++;
          if ( m_opts.verbosity >= 1 ) fmt::print( "  Barrier subproblem failed (failure #{})\n", barrier_failures );

          if ( barrier_failures >= m_opts.max_barrier_failures )
          {
            if ( m_opts.enable_recovery && recovery_attempts < m_opts.max_recovery_attempts )
            {
              bool recovery_success = recover_from_barrier_failure( x, mu, callback );
              if ( recovery_success )
              {
                recovery_attempts++;
                barrier_failures = 0;
                continue;
              }
            }

            status = Status::BARRIER_FAILED;
            break;
          }

          mu = std::max( m_opts.mu_min, mu * 0.5 );
          if ( m_opts.verbosity > 0 ) { fmt::print( "  Reducing μ to {:.1e}\n", mu ); }
          continue;
        }

        // Reset failures on success
        barrier_failures  = 0;
        recovery_attempts = 0;

        f = callback( x, &grad, nullptr );
        f_evals++;

        // Update best point
        if ( f < m_best_f - m_opts.progress_tol )
        {
          m_last_df         = m_best_f - f;
          m_best_f          = f;
          m_best_x          = x;
          no_progress_count = 0;
        }
        else
        {
          no_progress_count++;
        }

        compute_dual_variables_corrected( x, grad, lam_lower, lam_upper );
        auto kkt_res = compute_KKT_residuals( x, grad, lam_lower, lam_upper );

        print_outer_iteration(
          outer_iter,
          mu,
          f,
          kkt_res.duality_gap,
          kkt_res.gradient_norm,
          kkt_res.primal_feasibility,
          kkt_res.dual_feasibility );

        // Check convergence
        if ( check_convergence( kkt_res, mu ) )
        {
          status = Status::CONVERGED;
          break;
        }

        // With no finite side the logarithmic barrier is identically zero:
        // one unconstrained Newton solve is the complete subproblem.
        if ( !barrier_active )
        {
          status = kkt_res.gradient_norm <= m_opts.g_max ? Status::CONVERGED : Status::FAILED;
          break;
        }

        // Check for infeasibility
        if ( kkt_res.primal_feasibility > 0.1 && outer_iter > m_opts.max_outer_iterations / 2 )
        {
          status = Status::PRIMAL_INFEASIBLE;
          break;
        }

        if ( kkt_res.dual_feasibility > 0.1 && outer_iter > m_opts.max_outer_iterations / 2 )
        {
          status = Status::DUAL_INFEASIBLE;
          break;
        }

        // Check for no progress
        if ( no_progress_count >= m_opts.max_no_progress_iterations )
        {
          if ( m_opts.verbosity >= 1 )
            fmt::print( "  No progress for {} iterations, checking convergence...\n", no_progress_count );

          if ( kkt_res.gradient_norm <= m_opts.g_max * 10 )
          {
            status = Status::CONVERGED;
            break;
          }
          else
          {
            // Try to recover by increasing mu
            mu                = std::max( m_opts.mu_min, mu * 2.0 );
            no_progress_count = 0;
            if ( m_opts.verbosity >= 1 ) fmt::print( "  Increasing μ to {:.1e} to escape stagnation\n", mu );
          }
        }

        // Update mu
        Scalar prev_mu = mu;
        mu             = compute_robust_mu_update(
          mu,
          kkt_res.complementarity,
          kkt_res.primal_feasibility,
          kkt_res.dual_feasibility,
          kkt_res.gradient_norm );

        if ( mu <= m_opts.mu_min )
        {
          // Final convergence check with relaxed criteria
          if ( kkt_res.gradient_norm <= m_opts.g_max * 5 ) { status = Status::CONVERGED; }
          else
          {
            status = Status::BARRIER_FAILED;
          }
          break;
        }

        // Check for numerical stagnation
        if ( std::abs( mu - prev_mu ) < 1e-12 * prev_mu && no_progress_count > 5 )
        {
          mu = prev_mu * 0.1;
          if ( m_opts.verbosity >= 1 )
          {
            fmt::print( "  Numerical stagnation, aggressive μ reduction to {:.1e}\n", mu );
          }
        }
      }

      // Final evaluation
      f = callback( x, &grad, nullptr );
      f_evals++;
      compute_dual_variables_corrected( x, grad, lam_lower, lam_upper );

      auto final_kkt_res = compute_KKT_residuals( x, grad, lam_lower, lam_upper );

      // Final status determination
      if ( status == Status::CONVERGED )
      {
        if ( final_kkt_res.gradient_norm > m_opts.g_max * 5 )
        {
          if ( m_opts.verbosity >= 1 )
            fmt::print( "  Final gradient too large (‖g‖={:.2e}), marking as FAILED\n", final_kkt_res.gradient_norm );
          status = Status::FAILED;
        }
      }
      else if ( status == Status::MAX_ITERATIONS )
      {
        // Check if we have acceptable solution
        if ( final_kkt_res.gradient_norm <= m_opts.g_max && final_kkt_res.primal_feasibility <= m_opts.epsilon_infeas )
        {
          status = Status::CONVERGED;
        }
      }

      print_convergence_info( status, outer_iter, f, final_kkt_res.duality_gap );

      // Save results
      m_status                 = status;
      m_outer_iterations       = outer_iter;
      m_total_inner_iterations = total_inner_iterations;
      m_function_evals         = f_evals;
      m_hessian_evals          = h_evals;
      m_final_f                = f;
      m_final_mu               = mu;
      m_duality_gap            = final_kkt_res.duality_gap;
      m_primal_infeasibility   = final_kkt_res.primal_feasibility;
      m_dual_infeasibility     = final_kkt_res.dual_feasibility;
      m_gradient_norm          = final_kkt_res.gradient_norm;
      m_x                      = x;
      m_lambda_lower           = lam_lower;
      m_lambda_upper           = lam_upper;
      m_newton_steps           = newton_steps;
      m_gradient_steps         = gradient_steps;
      m_centering_steps        = centering_steps;
    }
  };

  // Consistency alias (capital M) for uniform naming
  template <typename Scalar = double>
  using Minimize_BBOX_IPNewton = minimize_BBOX_IPNewton<Scalar>;

}  // namespace Utils

#endif
