// SPDX-License-Identifier: MPL-2.0

#include "Utils_minimize_BBOX_TRON.hh"
#include "Utils_fmt.hh"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
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

  using Scalar       = double;
  using Vector       = Utils::TRON_details::Vector<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using Status       = Utils::TRON_details::Status;

  constexpr std::size_t MAX_ITERATIONS = 1000;
  constexpr int         NAME_WIDTH     = 39;
  constexpr int         STATUS_WIDTH   = 14;

  struct TestResult
  {
    std::string problem_name;
    Status      status{ Status::non_descent_model };
    int         dimension{ 0 };
    std::size_t iterations{ 0 };
    std::size_t function_evaluations{ 0 };
    std::size_t gradient_evaluations{ 0 };
    std::size_t hessian_vector_evaluations{ 0 };
    Scalar      final_function_value{ 0 };
    Scalar      projected_gradient_norm{ 0 };
    Scalar      solution_error{ 0 };
    double      elapsed_seconds{ 0 };
  };

  std::vector<TestResult> global_test_results;

  [[nodiscard]] fmt::text_style status_style( Status status )
  {
    switch ( status )
    {
      case Status::converged: return fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold;
      case Status::max_iterations:
      case Status::max_function_evaluations:
      case Status::small_step: return fmt::fg( fmt::color::gold );
      case Status::unbounded: return fmt::fg( fmt::color::magenta );
      case Status::non_descent_model:
      case Status::non_finite_objective:
      case Status::non_finite_gradient:
      case Status::non_finite_hessian: return fmt::fg( fmt::color::red ) | fmt::emphasis::bold;
    }
    return fmt::fg( fmt::color::white );
  }

  void print_rule( std::string_view fill = "─" )
  {
    for ( int i = 0; i < 128; ++i ) fmt::print( "{}", fill );
    fmt::print( "\n" );
  }

  void print_table_header()
  {
    print_rule();
    fmt::print(
      fmt::emphasis::bold,
      "{:<{}} {:<{}} {:>4} {:>6} {:>7} {:>7} {:>13} {:>11} {:>11} {:>8}\n",
      "Problem",
      NAME_WIDTH,
      "Status",
      STATUS_WIDTH,
      "Dim",
      "Iter",
      "F eval",
      "Hv eval",
      "f(x)",
      "||Pgrad||",
      "||x-x*||",
      "Time" );
    print_rule();
  }

  void print_result( TestResult const & result )
  {
    fmt::print( "{:<{}} ", result.problem_name, NAME_WIDTH );
    fmt::print( status_style( result.status ), "{:<{}}", to_string( result.status ), STATUS_WIDTH );
    fmt::print(
      " {:>4} {:>6} {:>7} {:>7} {:>13.5e} {:>11.3e} {:>11.3e} {:>7.3f}s\n",
      result.dimension,
      result.iterations,
      result.function_evaluations,
      result.hessian_vector_evaluations,
      result.final_function_value,
      result.projected_gradient_norm,
      result.solution_error,
      result.elapsed_seconds );
  }

  void print_skipped( std::string_view name, std::string_view reason )
  {
    fmt::print( "{:<{}} ", name, NAME_WIDTH );
    fmt::print( fmt::fg( fmt::color::gray ), "{:<{}}", "SKIPPED", STATUS_WIDTH );
    fmt::print( fmt::fg( fmt::color::gray ), " {}\n", reason );
  }

  void run_test( std::string const & name, std::shared_ptr<NDbase<Scalar>> const & problem )
  {
    Vector const lower = problem->lower();
    Vector const upper = problem->upper();
    Vector const x0    = problem->init();
    Vector const exact = problem->exact();

    // TRON applies the Hessian repeatedly at the same accepted iterate. Cache
    // the sparse matrix so ND_func.cxx constructs it only once per outer step.
    Vector       cached_x;
    SparseMatrix cached_hessian;
    bool         cache_valid    = false;
    auto         hessian_vector = [&]( Vector const & x, Vector const & v, Vector & Hv )
    {
      if ( !cache_valid || cached_x.size() != x.size() || ( cached_x.array() != x.array() ).any() )
      {
        cached_x       = x;
        cached_hessian = problem->hessian( x );
        cache_valid    = true;
      }
      Hv.noalias() = cached_hessian * v;
    };

    Utils::TRON_details::Options<Scalar> options;
    options.max_iterations           = MAX_ITERATIONS;
    options.max_function_evaluations = MAX_ITERATIONS + 1;
    options.absolute_tolerance       = 1e-12;
    options.relative_tolerance       = 1e-12;
    options.cg_tolerance             = 1e-6;

    Utils::Minimize_BBOX_TRON<Scalar> solver( x0.size(), options );
    auto const                        start       = std::chrono::steady_clock::now();
    auto const                        tron_result = solver.solve(
      x0,
      lower,
      upper,
      [&]( Vector const & x ) { return ( *problem )( x ); },
      [&]( Vector const & x, Vector & g ) { g = problem->gradient( x ); },
      hessian_vector );
    double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();

    TestResult result;
    result.problem_name               = name;
    result.status                     = tron_result.status;
    result.dimension                  = static_cast<int>( lower.size() );
    result.iterations                 = tron_result.iterations;
    result.function_evaluations       = tron_result.function_evaluations;
    result.gradient_evaluations       = tron_result.gradient_evaluations;
    result.hessian_vector_evaluations = tron_result.hessian_vector_evaluations;
    result.final_function_value       = tron_result.objective;
    result.projected_gradient_norm    = tron_result.projected_gradient_norm;
    result.solution_error             = exact.size() == tron_result.x.size() ? ( tron_result.x - exact ).stableNorm()
                                                                             : std::numeric_limits<Scalar>::quiet_NaN();
    result.elapsed_seconds            = elapsed;

    global_test_results.emplace_back( result );
    print_result( result );
  }

  void print_summary( std::size_t skipped )
  {
    std::size_t converged{ 0 };
    std::size_t limited{ 0 };
    std::size_t failed{ 0 };
    std::size_t total_iterations{ 0 };
    std::size_t total_function_evaluations{ 0 };
    std::size_t total_gradient_evaluations{ 0 };
    std::size_t total_hessian_products{ 0 };
    double      total_time{ 0 };

    for ( auto const & result : global_test_results )
    {
      if ( result.status == Status::converged )
        ++converged;
      else if (
        result.status == Status::max_iterations || result.status == Status::max_function_evaluations ||
        result.status == Status::small_step )
        ++limited;
      else
        ++failed;
      total_iterations += result.iterations;
      total_function_evaluations += result.function_evaluations;
      total_gradient_evaluations += result.gradient_evaluations;
      total_hessian_products += result.hessian_vector_evaluations;
      total_time += result.elapsed_seconds;
    }

    auto const   executed           = global_test_results.size();
    double const success_rate       = executed == 0
                                        ? 0.0
                                        : 100.0 * static_cast<double>( converged ) / static_cast<double>( executed );
    double const average_iterations = executed == 0
                                        ? 0.0
                                        : static_cast<double>( total_iterations ) / static_cast<double>( executed );

    print_rule( "═" );
    fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "TRON SUMMARY\n" );
    fmt::print( "  Problems: {} executed, {} skipped\n", executed, skipped );
    fmt::print( "  Status:   " );
    fmt::print( fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold, "{} converged", converged );
    fmt::print( ", " );
    fmt::print( fmt::fg( fmt::color::gold ), "{} stopped", limited );
    fmt::print( ", " );
    fmt::print( fmt::fg( fmt::color::red ) | fmt::emphasis::bold, "{} failed", failed );
    fmt::print( "  ({:.1f}% converged)\n", success_rate );
    fmt::print(
      "  Work:     {} iterations (avg {:.1f}), {} f, {} g, {} Hv evaluations\n",
      total_iterations,
      average_iterations,
      total_function_evaluations,
      total_gradient_evaluations,
      total_hessian_products );
    fmt::print( "  Time:     {:.3f} s total\n", total_time );
    fmt::print( "  Limit:    {} outer iterations per problem\n", MAX_ITERATIONS );
    print_rule( "═" );
  }

}  // namespace

int main()
{
  fmt::print( "\n" );
  print_rule( "═" );
  fmt::print(
    fmt::emphasis::bold | fmt::fg( fmt::color::cyan ),
    "TRON — BOUND-CONSTRAINED OPTIMIZATION ON ND_func ({})\n",
    NL_list.size() );
  fmt::print( "Newton model, projected Cauchy step and active-face truncated CG\n" );
  fmt::print( "Maximum iterations per problem: {}\n", MAX_ITERATIONS );
  print_table_header();

  std::size_t skipped{ 0 };
  for ( auto const & [problem, name] : NL_list )
  {
    // Katsuura is not differentiable; Michalewicz::exact() is unavailable in
    // the shared ND_func catalog.
    if ( name == "Katsuura10D" )
    {
      ++skipped;
      print_skipped( name, "non-differentiable" );
      continue;
    }
    if ( name == "MichalewiczN10D" )
    {
      ++skipped;
      print_skipped( name, "exact solution unavailable" );
      continue;
    }
    run_test( name, problem );
  }

  print_summary( skipped );

  // ND_func contains multimodal and nonsmooth benchmarks for which convergence
  // to the catalogued global minimizer is not expected from a local Newton
  // method. The detailed status table is the regression artifact; fatal
  // numerical states still make CTest fail.
  bool const fatal_failure = std::any_of(
    global_test_results.begin(),
    global_test_results.end(),
    []( TestResult const & result )
    {
      return result.status == Status::unbounded || result.status == Status::non_descent_model ||
             result.status == Status::non_finite_objective || result.status == Status::non_finite_gradient ||
             result.status == Status::non_finite_hessian;
    } );
  return fatal_failure ? 1 : 0;
}
