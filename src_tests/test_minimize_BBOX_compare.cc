// SPDX-License-Identifier: MPL-2.0

#include "Utils_fmt.hh"
#include "Utils_minimize_BBOX_Newton.hh"
#include "Utils_minimize_BBOX_TRON.hh"
#include "Utils_minimize_BBOX_small_TRON.hh"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ND_func.cxx"

namespace
{

  using Scalar         = double;
  using Vector         = Eigen::VectorXd;
  using SparseMatrix   = Eigen::SparseMatrix<Scalar>;
  using ConstVectorRef = Utils::SmallTRON::ConstVectorRef<Scalar>;
  using VectorRef      = Utils::SmallTRON::VectorRef<Scalar>;
  using MatrixRef      = Utils::SmallTRON::MatrixRef<Scalar>;

  constexpr std::size_t MAX_ITERATIONS = 400;
  constexpr int         NAME_WIDTH     = 32;
  constexpr int         STATUS_WIDTH   = 11;

  enum class Outcome
  {
    converged,
    stopped,
    failed
  };

  struct Metrics
  {
    std::string status;
    Outcome     outcome{ Outcome::failed };
    std::size_t iterations{ 0 };
    std::size_t function_evaluations{ 0 };
    std::size_t gradient_evaluations{ 0 };
    std::size_t second_order_evaluations{ 0 };
    Scalar      objective{ std::numeric_limits<Scalar>::quiet_NaN() };
    Scalar      projected_gradient_norm{ std::numeric_limits<Scalar>::quiet_NaN() };
    Vector      x;
  };

  struct Comparison
  {
    std::string name;
    int         dimension{ 0 };
    Metrics     tron;
    Metrics     tron2;
    Metrics     newton;
  };

  std::vector<Comparison> comparisons;

  class NDProblemAdapter
  {
  public:
    explicit NDProblemAdapter( std::shared_ptr<NDbase<Scalar>> problem ) : m_problem( std::move( problem ) ) {}

    Scalar objective( ConstVectorRef x ) { return ( *m_problem )( x ); }

    void gradient( ConstVectorRef x, VectorRef g ) { g = m_problem->gradient( x ); }

    void hessian( ConstVectorRef x, MatrixRef H )
    {
      if ( !m_cache_valid || m_cached_x.size() != x.size() || ( m_cached_x.array() != x.array() ).any() )
      {
        m_cached_x       = x;
        m_cached_hessian = m_problem->hessian( x );
        m_cache_valid    = true;
      }
      H = m_cached_hessian;
    }

    void hprod( Vector const & x, Vector const & v, Vector & Hv )
    {
      if ( !m_cache_valid || m_cached_x.size() != x.size() || ( m_cached_x.array() != x.array() ).any() )
      {
        m_cached_x       = x;
        m_cached_hessian = m_problem->hessian( x );
        m_cache_valid    = true;
      }
      Hv.noalias() = m_cached_hessian * v;
    }

  private:
    std::shared_ptr<NDbase<Scalar>> m_problem;
    Vector                          m_cached_x;
    SparseMatrix                    m_cached_hessian;
    bool                            m_cache_valid{ false };
  };

  [[nodiscard]] fmt::text_style outcome_style( Outcome outcome )
  {
    switch ( outcome )
    {
      case Outcome::converged: return fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold;
      case Outcome::stopped: return fmt::fg( fmt::color::gold );
      case Outcome::failed: return fmt::fg( fmt::color::red ) | fmt::emphasis::bold;
    }
    return fmt::fg( fmt::color::white );
  }

  [[nodiscard]] Metrics run_tron2(
    std::shared_ptr<NDbase<Scalar>> const & problem,
    Vector const &                          x0,
    Vector const &                          lower,
    Vector const &                          upper )
  {
    NDProblemAdapter adapter( problem );

    Utils::TRON2::Options<Scalar> options;
    options.max_iterations                  = MAX_ITERATIONS;
    options.max_function_evaluations        = MAX_ITERATIONS + 1;
    options.absolute_tolerance              = 1e-12;
    options.relative_tolerance              = 1e-12;
    options.cg_tolerance                    = 1e-6;
    options.max_projected_newton_iterations = 50;
    options.max_time_seconds                = 30.0;

    auto const result = Utils::TRON2::minimize(
      x0,
      lower,
      upper,
      [&]( Vector const & x ) { return adapter.objective( x ); },
      [&]( Vector const & x, Vector & g ) { adapter.gradient( x, g ); },
      [&]( Vector const & x, Vector const & v, Vector & Hv ) { adapter.hprod( x, v, Hv ); },
      options );

    Metrics metrics;
    switch ( result.status )
    {
      case Utils::TRON2::Status::converged: metrics.status = "CONVERGED"; break;
      case Utils::TRON2::Status::max_iterations: metrics.status = "ITER LIMIT"; break;
      case Utils::TRON2::Status::max_function_evaluations: metrics.status = "EVAL LIMIT"; break;
      case Utils::TRON2::Status::max_time: metrics.status = "TIME LIMIT"; break;
      case Utils::TRON2::Status::unbounded: metrics.status = "UNBOUNDED"; break;
      case Utils::TRON2::Status::small_step: metrics.status = "SMALL STEP"; break;
      case Utils::TRON2::Status::non_descent_model: metrics.status = "BAD MODEL"; break;
      case Utils::TRON2::Status::non_finite_objective: metrics.status = "NONFINITE F"; break;
      case Utils::TRON2::Status::non_finite_gradient: metrics.status = "NONFINITE G"; break;
      case Utils::TRON2::Status::non_finite_hessian: metrics.status = "NONFINITE H"; break;
    }
    metrics.iterations               = result.iterations;
    metrics.function_evaluations     = result.function_evaluations;
    metrics.gradient_evaluations     = result.gradient_evaluations;
    metrics.second_order_evaluations = result.hessian_vector_evaluations;
    metrics.objective                = result.objective;
    metrics.projected_gradient_norm  = result.projected_gradient_norm;
    metrics.x                        = result.x;

    using Status = Utils::TRON2::Status;
    if ( result.status == Status::converged )
      metrics.outcome = Outcome::converged;
    else if (
      result.status == Status::max_iterations || result.status == Status::max_function_evaluations ||
      result.status == Status::small_step )
      metrics.outcome = Outcome::stopped;

    return metrics;
  }

  [[nodiscard]] Metrics run_tron(
    std::shared_ptr<NDbase<Scalar>> const & problem,
    Vector const &                          x0,
    Vector const &                          lower,
    Vector const &                          upper )
  {
    NDProblemAdapter adapter( problem );

    Utils::SmallTRON::Options<Scalar> options;
    options.max_iter                        = static_cast<int>( MAX_ITERATIONS );
    options.max_eval                        = static_cast<int>( MAX_ITERATIONS + 1 );
    options.max_projected_newton_iterations = 50;
    options.set_tolerances( 1e-12 );

    Utils::SmallTRON::Solver<Scalar> solver( x0.size(), options );
    auto const                       result = solver.solve( adapter, x0, lower, upper );

    Metrics metrics;
    switch ( result.status )
    {
      case Utils::SmallTRON::Status::unknown: metrics.status = "UNKNOWN"; break;
      case Utils::SmallTRON::Status::first_order: metrics.status = "CONVERGED"; break;
      case Utils::SmallTRON::Status::unbounded: metrics.status = "UNBOUNDED"; break;
      case Utils::SmallTRON::Status::max_iter: metrics.status = "ITER LIMIT"; break;
      case Utils::SmallTRON::Status::max_eval: metrics.status = "EVAL LIMIT"; break;
      case Utils::SmallTRON::Status::small_step: metrics.status = "SMALL STEP"; break;
      case Utils::SmallTRON::Status::neg_pred: metrics.status = "BAD MODEL"; break;
      case Utils::SmallTRON::Status::direct_solver_failure: metrics.status = "DIRECT FAIL"; break;
      case Utils::SmallTRON::Status::user: metrics.status = "USER STOP"; break;
    }
    metrics.iterations               = static_cast<std::size_t>( result.iter );
    metrics.function_evaluations     = static_cast<std::size_t>( result.obj_evals );
    metrics.gradient_evaluations     = static_cast<std::size_t>( result.grad_evals );
    metrics.second_order_evaluations = static_cast<std::size_t>( result.hess_evals );
    metrics.objective                = result.objective;
    metrics.projected_gradient_norm  = result.dual_feas;
    metrics.x                        = result.x;

    using Status = Utils::SmallTRON::Status;
    if ( result.status == Status::first_order )
      metrics.outcome = Outcome::converged;
    else if (
      result.status == Status::max_iter || result.status == Status::max_eval || result.status == Status::small_step )
      metrics.outcome = Outcome::stopped;

    return metrics;
  }

  [[nodiscard]] Metrics run_newton(
    std::shared_ptr<NDbase<Scalar>> const & problem,
    Vector const &                          x0,
    Vector const &                          lower,
    Vector const &                          upper )
  {
    NDProblemAdapter adapter( problem );

    Utils::Options options;
    options.max_iterations           = static_cast<int>( MAX_ITERATIONS );
    options.max_function_evaluations = static_cast<int>( MAX_ITERATIONS + 1 );
    options.set_tolerances( 1e-12 );

    Utils::Minimize_BBOX_Newton solver( x0.size(), options );
    auto const                  result = solver.solve( adapter, x0, lower, upper );

    Metrics metrics;
    using Status = Utils::Status;
    switch ( result.status )
    {
      case Status::unknown: metrics.status = "UNKNOWN"; break;
      case Status::converged: metrics.status = "CONVERGED"; break;
      case Status::max_iterations: metrics.status = "ITER LIMIT"; break;
      case Status::max_function_evaluations: metrics.status = "EVAL LIMIT"; break;
      case Status::no_progress: metrics.status = "NO PROGRESS"; break;
      case Status::non_finite_objective: metrics.status = "NONFINITE F"; break;
      case Status::non_finite_gradient: metrics.status = "NONFINITE G"; break;
      case Status::non_finite_hessian: metrics.status = "NONFINITE H"; break;
      case Status::eigensolver_failure: metrics.status = "EIGEN FAIL"; break;
      case Status::user: metrics.status = "USER STOP"; break;
    }
    metrics.iterations               = static_cast<std::size_t>( result.iterations );
    metrics.function_evaluations     = static_cast<std::size_t>( result.function_evaluations );
    metrics.gradient_evaluations     = static_cast<std::size_t>( result.gradient_evaluations );
    metrics.second_order_evaluations = static_cast<std::size_t>( result.hessian_evaluations );
    metrics.objective                = result.objective;
    metrics.projected_gradient_norm  = result.projected_gradient_norm;
    metrics.x                        = result.x;

    using Status = Utils::Status;
    if ( result.status == Status::converged )
      metrics.outcome = Outcome::converged;
    else if (
      result.status == Status::max_iterations || result.status == Status::max_function_evaluations ||
      result.status == Status::no_progress )
      metrics.outcome = Outcome::stopped;
    return metrics;
  }

  void print_rule( std::string_view fill = "─" )
  {
    for ( int i = 0; i < 202; ++i ) fmt::print( "{}", fill );
    fmt::print( "\n" );
  }

  void print_header()
  {
    print_rule();
    fmt::print(
      fmt::emphasis::bold,
      "{:<{}} {:>4} │ {:^{}} {:>5} {:>5} {:>7} {:>10} {:>9} │ {:^{}} "
      "{:>5} {:>5} {:>7} {:>10} {:>9} │ {:^{}} {:>5} {:>5} {:>7} "
      "{:>10} {:>9}\n",
      "Problem",
      NAME_WIDTH,
      "Dim",
      "SmallTRON",
      STATUS_WIDTH,
      "Iter",
      "F",
      "H",
      "f(x)",
      "ǁPgradǁ",
      "TRON",
      STATUS_WIDTH,
      "Iter",
      "F",
      "Hv",
      "f(x)",
      "ǁPgradǁ",
      "BBOX Newton",
      STATUS_WIDTH,
      "Iter",
      "F",
      "H",
      "f(x)",
      "ǁPgradǁ" );
    print_rule();
  }

  void print_status( Metrics const & metrics )
  {
    std::string_view label = metrics.status;
    if ( label.size() > static_cast<std::size_t>( STATUS_WIDTH ) ) label = label.substr( 0, STATUS_WIDTH );
    fmt::print( outcome_style( metrics.outcome ), "{:<{}}", label, STATUS_WIDTH );
  }

  void print_comparison( Comparison const & comparison )
  {
    fmt::print( "{:<{}} {:>4} │ ", comparison.name, NAME_WIDTH, comparison.dimension );
    print_status( comparison.tron );
    fmt::print(
      " {:>5} {:>5} {:>7} {:>10.2e} {:>9.2e} │ ",
      comparison.tron.iterations,
      comparison.tron.function_evaluations,
      comparison.tron.second_order_evaluations,
      comparison.tron.objective,
      comparison.tron.projected_gradient_norm );
    print_status( comparison.tron2 );
    fmt::print(
      " {:>5} {:>5} {:>7} {:>10.2e} {:>9.2e} │ ",
      comparison.tron2.iterations,
      comparison.tron2.function_evaluations,
      comparison.tron2.second_order_evaluations,
      comparison.tron2.objective,
      comparison.tron2.projected_gradient_norm );
    print_status( comparison.newton );
    fmt::print(
      " {:>5} {:>5} {:>7} {:>10.2e} {:>9.2e}\n",
      comparison.newton.iterations,
      comparison.newton.function_evaluations,
      comparison.newton.second_order_evaluations,
      comparison.newton.objective,
      comparison.newton.projected_gradient_norm );
  }

  struct Totals
  {
    std::size_t converged{ 0 };
    std::size_t stopped{ 0 };
    std::size_t failed{ 0 };
    std::size_t iterations{ 0 };
    std::size_t function_evaluations{ 0 };
    std::size_t gradient_evaluations{ 0 };
    std::size_t second_order_evaluations{ 0 };
  };

  void accumulate( Totals & totals, Metrics const & metrics )
  {
    switch ( metrics.outcome )
    {
      case Outcome::converged: ++totals.converged; break;
      case Outcome::stopped: ++totals.stopped; break;
      case Outcome::failed: ++totals.failed; break;
    }
    totals.iterations += metrics.iterations;
    totals.function_evaluations += metrics.function_evaluations;
    totals.gradient_evaluations += metrics.gradient_evaluations;
    totals.second_order_evaluations += metrics.second_order_evaluations;
  }

  void print_totals( std::string_view name, std::string_view second_order_label, Totals const & totals )
  {
    fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "  {:<10}", name );
    fmt::print( outcome_style( Outcome::converged ), "{:>2} converged", totals.converged );
    fmt::print( ", " );
    fmt::print( outcome_style( Outcome::stopped ), "{:>2} stopped", totals.stopped );
    fmt::print( ", " );
    fmt::print( outcome_style( Outcome::failed ), "{:>2} failed", totals.failed );
    fmt::print(
      "; {:>5} iter, {:>5} f, {:>5} g, {:>7} {}\n",
      totals.iterations,
      totals.function_evaluations,
      totals.gradient_evaluations,
      totals.second_order_evaluations,
      second_order_label );
  }

  void print_summary( std::size_t skipped )
  {
    Totals      tron_totals;
    Totals      tron2_totals;
    Totals      newton_totals;
    std::size_t same_outcome{ 0 };
    std::size_t tron_best{ 0 };
    std::size_t tron2_best{ 0 };
    std::size_t newton_best{ 0 };

    for ( auto const & comparison : comparisons )
    {
      accumulate( tron_totals, comparison.tron );
      accumulate( tron2_totals, comparison.tron2 );
      accumulate( newton_totals, comparison.newton );
      if ( comparison.tron.outcome == comparison.tron2.outcome && comparison.tron.outcome == comparison.newton.outcome )
        ++same_outcome;

      Scalar const scale = std::max(
        { Scalar( 1 ),
          std::abs( comparison.tron.objective ),
          std::abs( comparison.tron2.objective ),
          std::abs( comparison.newton.objective ) } );
      Scalar const tol  = 1e-10 * scale;
      Scalar const best = std::min(
        { comparison.tron.objective, comparison.tron2.objective, comparison.newton.objective } );
      if ( comparison.tron.objective <= best + tol ) ++tron_best;
      if ( comparison.tron2.objective <= best + tol ) ++tron2_best;
      if ( comparison.newton.objective <= best + tol ) ++newton_best;
    }

    print_rule( "═" );
    fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "SmallTRON vs TRON vs BBOX Newton SUMMARY\n" );
    fmt::print(
      "  Problems: {} compared, {} skipped; same outcome class for all "
      "three: {}/{}\n",
      comparisons.size(),
      skipped,
      same_outcome,
      comparisons.size() );
    print_totals( "SmallTRON", "H", tron_totals );
    print_totals( "TRON", "Hv", tron2_totals );
    print_totals( "Newton", "H", newton_totals );
    fmt::print(
      "  Best objective (ties included): SmallTRON {}, TRON {}, BBOX "
      "Newton {}\n",
      tron_best,
      tron2_best,
      newton_best );
    fmt::print( "  Limit:     {} outer iterations per problem\n", MAX_ITERATIONS );
    print_rule( "═" );
  }

}  // namespace

int main()
{
  fmt::print( "\n" );
  print_rule( "═" );
  fmt::print(
    fmt::emphasis::bold | fmt::fg( fmt::color::cyan ),
    "SmallTRON vs TRON vs BBOX Newton — SAME ND_func PROBLEMS AND "
    "SOLVER BUDGETS ({})\n",
    NL_list.size() );
  fmt::print( "Maximum iterations per problem: {}\n", MAX_ITERATIONS );
  print_header();

  std::size_t skipped{ 0 };
  for ( auto const & [problem, name] : NL_list )
  {
    if ( name == "Katsuura10D" || name == "MichalewiczN10D" )
    {
      ++skipped;
      continue;
    }

    Vector const lower = problem->lower();
    Vector const upper = problem->upper();
    Vector const x0    = problem->init();

    Comparison comparison;
    comparison.name      = name;
    comparison.dimension = static_cast<int>( x0.size() );
    comparison.tron      = run_tron( problem, x0, lower, upper );
    comparison.tron2     = run_tron2( problem, x0, lower, upper );
    comparison.newton    = run_newton( problem, x0, lower, upper );
    comparisons.emplace_back( std::move( comparison ) );
    print_comparison( comparisons.back() );
  }

  print_summary( skipped );

  bool const fatal_failure = std::any_of(
    comparisons.begin(),
    comparisons.end(),
    []( Comparison const & comparison )
    {
      return comparison.tron.outcome == Outcome::failed || comparison.tron2.outcome == Outcome::failed ||
             comparison.newton.outcome == Outcome::failed || !std::isfinite( comparison.tron.objective ) ||
             !std::isfinite( comparison.tron2.objective ) || !std::isfinite( comparison.newton.objective );
    } );
  return fatal_failure ? 1 : 0;
}
