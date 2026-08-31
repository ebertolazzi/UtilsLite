// SPDX-License-Identifier: MPL-2.0

#include "Utils_fmt.hh"
#include "Utils_minimize_BBOX_Newton.hh"
#include "Utils_minimize_BBOX_NewtonCubic.hh"

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
  using Vector         = Utils::Vector<Scalar>;
  using SparseMatrix   = Eigen::SparseMatrix<Scalar>;
  using ConstVectorRef = Utils::ConstVectorRef<Scalar>;
  using VectorRef      = Utils::VectorRef<Scalar>;
  using MatrixRef      = Utils::MatrixRef<Scalar>;

  constexpr std::size_t MAX_ITERATIONS = 400;
  constexpr int         NAME_WIDTH     = 38;
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
    std::size_t hessian_evaluations{ 0 };
    Scalar      objective{ std::numeric_limits<Scalar>::quiet_NaN() };
    Scalar      projected_gradient_norm{ std::numeric_limits<Scalar>::quiet_NaN() };
    Scalar      minimum_critical_eigenvalue{ std::numeric_limits<Scalar>::quiet_NaN() };
    bool        second_order_minimum{ false };
    Vector      x;
  };

  struct Comparison
  {
    std::string name;
    int         dimension{ 0 };
    Metrics     newton;
    Metrics     cubic;
  };

  std::vector<Comparison> comparisons;

  class NDProblemAdapter
  {
  public:
    explicit NDProblemAdapter( std::shared_ptr<NDbase<Scalar>> problem ) : m_problem( std::move( problem ) ) {}

    Scalar objective( ConstVectorRef x ) { return ( *m_problem )( x ); }

    void gradient( ConstVectorRef x, VectorRef gradient ) { gradient = m_problem->gradient( x ); }

    void hessian( ConstVectorRef x, MatrixRef hessian )
    {
      if ( !m_cache_valid || m_cached_x.size() != x.size() || ( m_cached_x.array() != x.array() ).any() )
      {
        m_cached_x       = x;
        m_cached_hessian = m_problem->hessian( x );
        m_cache_valid    = true;
      }
      hessian = m_cached_hessian;
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

  [[nodiscard]] Metrics make_metrics( Utils::Result<Scalar> const & result )
  {
    Metrics metrics;
    switch ( result.status )
    {
      case Utils::Status::unknown: metrics.status = "UNKNOWN"; break;
      case Utils::Status::converged: metrics.status = "CONVERGED"; break;
      case Utils::Status::max_iterations: metrics.status = "ITER LIMIT"; break;
      case Utils::Status::max_function_evaluations: metrics.status = "EVAL LIMIT"; break;
      case Utils::Status::no_progress: metrics.status = "NO PROGRESS"; break;
      case Utils::Status::non_finite_objective: metrics.status = "NONFINITE F"; break;
      case Utils::Status::non_finite_gradient: metrics.status = "NONFINITE G"; break;
      case Utils::Status::non_finite_hessian: metrics.status = "NONFINITE H"; break;
      case Utils::Status::eigensolver_failure: metrics.status = "EIGEN FAIL"; break;
      case Utils::Status::user: metrics.status = "USER STOP"; break;
    }

    metrics.iterations              = static_cast<std::size_t>( result.iterations );
    metrics.function_evaluations    = static_cast<std::size_t>( result.function_evaluations );
    metrics.gradient_evaluations    = static_cast<std::size_t>( result.gradient_evaluations );
    metrics.hessian_evaluations     = static_cast<std::size_t>( result.hessian_evaluations );
    metrics.objective               = result.objective;
    metrics.projected_gradient_norm = result.projected_gradient_norm;
    metrics.x                       = result.x;

    if ( result.status == Utils::Status::converged )
      metrics.outcome = Outcome::converged;
    else if (
      result.status == Utils::Status::max_iterations || result.status == Utils::Status::max_function_evaluations ||
      result.status == Utils::Status::no_progress )
      metrics.outcome = Outcome::stopped;

    return metrics;
  }

  void check_second_order_minimum(
    NDProblemAdapter & adapter,
    Vector const &     x,
    Vector const &     lower,
    Vector const &     upper,
    Scalar             first_order_tolerance,
    Metrics &          metrics )
  {
    Vector gradient( x.size() );
    adapter.gradient( x, gradient );

    std::vector<Eigen::Index> critical;
    critical.reserve( static_cast<std::size_t>( x.size() ) );
    Scalar const eps = std::numeric_limits<Scalar>::epsilon();
    for ( Eigen::Index i = 0; i < x.size(); ++i )
    {
      if ( lower[i] == upper[i] ) continue;
      Scalar const scale = std::max(
        { Scalar( 1 ),
          std::abs( x[i] ),
          std::abs( gradient[i] ),
          std::isfinite( lower[i] ) ? std::abs( lower[i] ) : Scalar( 0 ),
          std::isfinite( upper[i] ) ? std::abs( upper[i] ) : Scalar( 0 ) } );
      Scalar const bound_tolerance = Scalar( 64 ) * eps * scale;
      bool const   at_lower        = std::isfinite( lower[i] ) && x[i] <= lower[i] + bound_tolerance;
      bool const   at_upper        = std::isfinite( upper[i] ) && x[i] >= upper[i] - bound_tolerance;
      bool const   strongly_lower  = at_lower && gradient[i] > first_order_tolerance;
      bool const   strongly_upper  = at_upper && gradient[i] < -first_order_tolerance;
      if ( !strongly_lower && !strongly_upper ) critical.push_back( i );
    }

    if ( critical.empty() )
    {
      metrics.minimum_critical_eigenvalue = std::numeric_limits<Scalar>::infinity();
      metrics.second_order_minimum        = true;
      return;
    }

    Utils::Matrix<Scalar> hessian( x.size(), x.size() );
    adapter.hessian( x, hessian );
    Utils::Matrix<Scalar> reduced = hessian( critical, critical );
    reduced                       = Scalar( 0.5 ) * ( reduced + reduced.transpose() ).eval();
    Eigen::SelfAdjointEigenSolver<Utils::Matrix<Scalar>> eig( reduced );
    if ( eig.info() != Eigen::Success ) return;

    metrics.minimum_critical_eigenvalue = eig.eigenvalues().minCoeff();
    Scalar const hessian_scale          = std::max( Scalar( 1 ), reduced.cwiseAbs().maxCoeff() );
    Scalar const curvature_tolerance    = Scalar( 256 ) * eps * hessian_scale *
                                          Scalar( std::max<Eigen::Index>( 1, reduced.rows() ) );
    metrics.second_order_minimum        = metrics.minimum_critical_eigenvalue >= -curvature_tolerance;
  }

  template <typename Solver> [[nodiscard]] Metrics run_solver(
    std::shared_ptr<NDbase<Scalar>> const & problem,
    Vector const &                          x0,
    Vector const &                          lower,
    Vector const &                          upper )
  {
    NDProblemAdapter adapter( problem );

    Utils::Options<Scalar> options;
    options.max_iterations           = static_cast<int>( MAX_ITERATIONS );
    options.max_function_evaluations = static_cast<int>( MAX_ITERATIONS + 1 );
    options.set_tolerances( 1e-12 );

    Solver     solver( x0.size(), options );
    auto const result  = solver.solve( adapter, x0, lower, upper );
    Metrics    metrics = make_metrics( result );
    if ( result.status == Utils::Status::converged )
      check_second_order_minimum( adapter, result.x, lower, upper, result.optimality_tolerance, metrics );
    return metrics;
  }

  void print_rule( std::string_view fill = "─" )
  {
    for ( int i = 0; i < 168; ++i ) fmt::print( "{}", fill );
    fmt::print( "\n" );
  }

  void print_header()
  {
    print_rule();
    fmt::print(
      fmt::emphasis::bold,
      "{:<{}} {:>4} │ {:^{}} {:>5} {:>5} {:>5} {:>10} {:>9} {:>10} │ "
      "{:^{}} {:>5} {:>5} {:>5} {:>10} {:>9} {:>10} │ {:>10}\n",
      "Problem",
      NAME_WIDTH,
      "Dim",
      "Newton",
      STATUS_WIDTH,
      "Iter",
      "F",
      "H",
      "f(x)",
      "ǁPgradǁ",
      "λcrit",
      "NewtonCubic",
      STATUS_WIDTH,
      "Iter",
      "F",
      "H",
      "f(x)",
      "ǁPgradǁ",
      "λcrit",
      "ǁΔxǁ∞" );
    print_rule();
  }

  void print_status( Metrics const & metrics )
  {
    std::string_view label = metrics.status;
    if ( label.size() > static_cast<std::size_t>( STATUS_WIDTH ) ) label = label.substr( 0, STATUS_WIDTH );
    fmt::print( outcome_style( metrics.outcome ), "{:<{}}", label, STATUS_WIDTH );
  }

  void print_metrics( Metrics const & metrics )
  {
    print_status( metrics );
    fmt::print(
      " {:>5} {:>5} {:>5} {:>10.2e} {:>9.2e} {:>10.2e}",
      metrics.iterations,
      metrics.function_evaluations,
      metrics.hessian_evaluations,
      metrics.objective,
      metrics.projected_gradient_norm,
      metrics.minimum_critical_eigenvalue );
  }

  void print_comparison( Comparison const & comparison )
  {
    fmt::print( "{:<{}} {:>4} │ ", comparison.name, NAME_WIDTH, comparison.dimension );
    print_metrics( comparison.newton );
    fmt::print( " │ " );
    print_metrics( comparison.cubic );
    if ( comparison.newton.outcome == Outcome::converged && comparison.cubic.outcome == Outcome::converged )
    {
      Scalar const solution_difference =
        ( comparison.newton.x - comparison.cubic.x ).template lpNorm<Eigen::Infinity>();
      fmt::print( " │ {:>10.2e}", solution_difference );
    }
    else
      fmt::print( " │ {:>10}", "-" );
    fmt::print( "\n" );
  }

  struct Totals
  {
    std::size_t converged{ 0 };
    std::size_t stopped{ 0 };
    std::size_t failed{ 0 };
    std::size_t iterations{ 0 };
    std::size_t function_evaluations{ 0 };
    std::size_t gradient_evaluations{ 0 };
    std::size_t hessian_evaluations{ 0 };
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
    totals.hessian_evaluations += metrics.hessian_evaluations;
  }

  void print_totals( std::string_view name, Totals const & totals )
  {
    fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "  {:<18}", name );
    fmt::print( outcome_style( Outcome::converged ), "{:>2} converged", totals.converged );
    fmt::print( ", " );
    fmt::print( outcome_style( Outcome::stopped ), "{:>2} stopped", totals.stopped );
    fmt::print( ", " );
    fmt::print( outcome_style( Outcome::failed ), "{:>2} failed", totals.failed );
    fmt::print(
      "; {:>5} iter, {:>5} f, {:>5} g, {:>5} H\n",
      totals.iterations,
      totals.function_evaluations,
      totals.gradient_evaluations,
      totals.hessian_evaluations );
  }

  void print_summary( std::size_t skipped )
  {
    Totals      newton_totals;
    Totals      cubic_totals;
    std::size_t same_outcome{ 0 };
    std::size_t newton_best{ 0 };
    std::size_t cubic_best{ 0 };

    for ( auto const & comparison : comparisons )
    {
      accumulate( newton_totals, comparison.newton );
      accumulate( cubic_totals, comparison.cubic );
      if ( comparison.newton.outcome == comparison.cubic.outcome ) ++same_outcome;

      Scalar const scale = std::max(
        { Scalar( 1 ), std::abs( comparison.newton.objective ), std::abs( comparison.cubic.objective ) } );
      Scalar const tolerance = 1e-10 * scale;
      Scalar const best      = std::min( comparison.newton.objective, comparison.cubic.objective );
      if ( comparison.newton.objective <= best + tolerance ) ++newton_best;
      if ( comparison.cubic.objective <= best + tolerance ) ++cubic_best;
    }

    print_rule( "═" );
    fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "BBOX Newton vs BBOX NewtonCubic SUMMARY\n" );
    fmt::print(
      "  Problems: {} compared, {} skipped; same outcome class: {}/{}\n",
      comparisons.size(),
      skipped,
      same_outcome,
      comparisons.size() );
    print_totals( "BBOX Newton", newton_totals );
    print_totals( "BBOX NewtonCubic", cubic_totals );
    fmt::print( "  Best objective (ties included): BBOX Newton {}, BBOX NewtonCubic {}\n", newton_best, cubic_best );
    fmt::print( "  Limit: {} outer iterations per problem\n", MAX_ITERATIONS );
    print_rule( "═" );
  }

}  // namespace

int main()
{
  fmt::print( "\n" );
  print_rule( "═" );
  fmt::print(
    fmt::emphasis::bold | fmt::fg( fmt::color::cyan ),
    "BBOX Newton vs BBOX NewtonCubic — SAME ND_func PROBLEMS AND SOLVER BUDGETS ({})\n",
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
    comparison.newton    = run_solver<Utils::Minimize_BBOX_Newton<Scalar>>( problem, x0, lower, upper );
    comparison.cubic     = run_solver<Utils::Minimize_BBOX_NewtonCubic<Scalar>>( problem, x0, lower, upper );
    comparisons.emplace_back( std::move( comparison ) );
    print_comparison( comparisons.back() );
  }

  print_summary( skipped );

  bool const fatal_failure = std::any_of(
    comparisons.begin(),
    comparisons.end(),
    []( Comparison const & comparison )
    {
      return comparison.newton.outcome == Outcome::failed || comparison.cubic.outcome == Outcome::failed ||
             !std::isfinite( comparison.newton.objective ) || !std::isfinite( comparison.cubic.objective ) ||
             ( comparison.newton.outcome == Outcome::converged && !comparison.newton.second_order_minimum ) ||
             ( comparison.cubic.outcome == Outcome::converged && !comparison.cubic.second_order_minimum );
    } );
  return fatal_failure ? 1 : 0;
}
