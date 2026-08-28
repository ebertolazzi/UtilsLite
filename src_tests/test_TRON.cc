// SPDX-License-Identifier: MPL-2.0

#include "Utils_minimize_BBOX_small_TRON.hh"
#include "Utils_fmt.hh"

#include <Eigen/Core>
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
  using Vector         = Utils::SmallTRON::Vector<Scalar>;
  using ConstVectorRef = Utils::SmallTRON::ConstVectorRef<Scalar>;
  using VectorRef      = Utils::SmallTRON::VectorRef<Scalar>;
  using MatrixRef      = Utils::SmallTRON::MatrixRef<Scalar>;
  using Status         = Utils::SmallTRON::Status;

  constexpr std::size_t MAX_ITERATIONS = 400;
  constexpr int         NAME_WIDTH     = 39;
  constexpr int         STATUS_WIDTH   = 14;

  struct TestResult
  {
    std::string problem_name;
    Status      status{ Status::unknown };
    int         dimension{ 0 };
    std::size_t iterations{ 0 };
    std::size_t function_evaluations{ 0 };
    std::size_t gradient_evaluations{ 0 };
    std::size_t hessian_evaluations{ 0 };
    Scalar      final_function_value{ 0 };
    Scalar      projected_gradient_norm{ 0 };
    Scalar      solution_error{ 0 };
  };

  std::vector<TestResult> global_test_results;

  [[nodiscard]] std::string_view status_label( Status status )
  {
    switch ( status )
    {
      case Status::unknown: return "UNKNOWN";
      case Status::first_order: return "CONVERGED";
      case Status::max_iter: return "ITER LIMIT";
      case Status::max_eval: return "EVAL LIMIT";
      case Status::unbounded: return "UNBOUNDED";
      case Status::small_step: return "SMALL STEP";
      case Status::neg_pred: return "BAD MODEL";
      case Status::direct_solver_failure: return "DIRECT FAIL";
      case Status::user: return "USER STOP";
    }
    return "UNKNOWN";
  }

  [[nodiscard]] fmt::text_style status_style( Status status )
  {
    switch ( status )
    {
      case Status::first_order: return fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold;
      case Status::max_iter:
      case Status::max_eval:
      case Status::small_step: return fmt::fg( fmt::color::gold );
      case Status::unbounded: return fmt::fg( fmt::color::magenta );
      case Status::unknown:
      case Status::neg_pred:
      case Status::direct_solver_failure:
      case Status::user: return fmt::fg( fmt::color::red ) | fmt::emphasis::bold;
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
      "{:<{}} {:<{}} {:>4} {:>6} {:>7} {:>7} {:>13} {:>11} {:>11}\n",
      "Problem",
      NAME_WIDTH,
      "Status",
      STATUS_WIDTH,
      "Dim",
      "Iter",
      "F eval",
      "H eval",
      "f(x)",
      "||Pgrad||",
      "||x-x*||" );
    print_rule();
  }

  void print_result( TestResult const & result )
  {
    fmt::print( "{:<{}} ", result.problem_name, NAME_WIDTH );
    fmt::print( status_style( result.status ), "{:<{}}", status_label( result.status ), STATUS_WIDTH );
    fmt::print(
      " {:>4} {:>6} {:>7} {:>7} {:>13.5e} {:>11.3e} {:>11.3e}\n",
      result.dimension,
      result.iterations,
      result.function_evaluations,
      result.hessian_evaluations,
      result.final_function_value,
      result.projected_gradient_norm,
      result.solution_error );
  }

  void print_skipped( std::string_view name, std::string_view reason )
  {
    fmt::print( "{:<{}} ", name, NAME_WIDTH );
    fmt::print( fmt::fg( fmt::color::gray ), "{:<{}}", "SKIPPED", STATUS_WIDTH );
    fmt::print( fmt::fg( fmt::color::gray ), " {}\n", reason );
  }

  [[nodiscard]] bool test_direct_subproblems()
  {
    using DenseMatrix  = Utils::SmallTRON::Matrix<Scalar>;
    using NewtonStatus = Utils::SmallTRON::NewtonStatus;

    bool passed = true;
    auto check  = [&]( bool condition ) { passed = passed && condition; };

    DenseMatrix positive( 2, 2 );
    positive << 2.0, 0.0, 0.0, 4.0;
    Vector rhs( 2 ), step;
    rhs << 2.0, 8.0;

    auto status = Utils::SmallTRON::detail::direct_trust_region( positive, rhs, 10.0, step );
    check( status == NewtonStatus::stationary );
    check( ( step - ( Vector( 2 ) << 1.0, 2.0 ).finished() ).norm() <= 1e-12 );

    status = Utils::SmallTRON::detail::direct_trust_region( positive, rhs, 1.0, step );
    check( status == NewtonStatus::boundary );
    check( std::abs( step.norm() - 1.0 ) <= 1e-6 );
    check( Scalar( 0.5 ) * step.dot( positive * step ) - rhs.dot( step ) < 0.0 );

    DenseMatrix indefinite( 2, 2 );
    indefinite << -2.0, 0.0, 0.0, 1.0;
    rhs.setZero();
    status = Utils::SmallTRON::detail::direct_trust_region( indefinite, rhs, 3.0, step );
    check( status == NewtonStatus::boundary );
    check( std::abs( step.norm() - 3.0 ) <= 1e-12 );
    check( Scalar( 0.5 ) * step.dot( indefinite * step ) < 0.0 );

    // Verify the Ref-based API directly on caller-owned raw storage.
    Scalar                   x0_data[]{ 2.0, -3.0 };
    Scalar                   lower_data[]{ -10.0, -10.0 };
    Scalar                   upper_data[]{ 10.0, 10.0 };
    Eigen::Map<Vector const> x0_map( x0_data, 2 );
    Eigen::Map<Vector const> lower_map( lower_data, 2 );
    Eigen::Map<Vector const> upper_map( upper_data, 2 );

    auto mapped_problem = Utils::SmallTRON::make_problem<Scalar>(
      []( ConstVectorRef x ) { return Scalar( 0.5 ) * ( x[0] * x[0] + Scalar( 2 ) * x[1] * x[1] ); },
      []( ConstVectorRef x, VectorRef g )
      {
        g[0] = x[0];
        g[1] = Scalar( 2 ) * x[1];
      },
      []( ConstVectorRef, MatrixRef H )
      {
        H.setZero();
        H( 0, 0 ) = Scalar( 1 );
        H( 1, 1 ) = Scalar( 2 );
      } );
    Utils::SmallTRON::Options<Scalar> strict_options;
    strict_options.set_tolerances( 1e-12 );
    Utils::SmallTRON::Solver<Scalar> mapped_solver( 2, strict_options );
    auto const mapped_result = mapped_solver.solve( mapped_problem, x0_map, lower_map, upper_map );
    check( mapped_result.status == Status::first_order );
    check( mapped_result.x.norm() <= 1e-12 );
    check( mapped_result.dual_feas <= mapped_result.optimality_tolerance );
    check( mapped_result.step_norm <= mapped_result.effective_step_tolerance );
    check( mapped_result.minimum_eigenvalue >= -strict_options.curvature_tolerance );

    // The objective cannot resolve changes of order one next to a 1e20 offset.
    // Acceptance must still permit derivative-driven Newton refinement instead
    // of stalling because f(x+s) and f(x) round to the same double.
    auto offset_problem = Utils::SmallTRON::make_problem<Scalar>(
      []( ConstVectorRef x )
      {
        return Scalar( 1e20 ) + Scalar( 0.5 ) * ( std::pow( x[0] - Scalar( 0.25 ), 2 ) +
                                                  Scalar( 3 ) * std::pow( x[1] + Scalar( 0.75 ), 2 ) );
      },
      []( ConstVectorRef x, VectorRef g )
      {
        g[0] = x[0] - Scalar( 0.25 );
        g[1] = Scalar( 3 ) * ( x[1] + Scalar( 0.75 ) );
      },
      []( ConstVectorRef, MatrixRef H )
      {
        H.setZero();
        H( 0, 0 ) = Scalar( 1 );
        H( 1, 1 ) = Scalar( 3 );
      } );
    Utils::SmallTRON::Solver<Scalar> offset_solver( 2, strict_options );
    auto const offset_result = offset_solver.solve( offset_problem, x0_map, lower_map, upper_map );
    Vector     offset_exact( 2 );
    offset_exact << Scalar( 0.25 ), Scalar( -0.75 );
    check( offset_result.status == Status::first_order );
    check( ( offset_result.x - offset_exact ).stableNorm() <= Scalar( 64 ) * std::numeric_limits<Scalar>::epsilon() );
    check( offset_result.dual_feas <= offset_result.optimality_tolerance );
    check( offset_result.step_norm <= offset_result.effective_step_tolerance );

    // At a constrained minimum the ordinary gradient need not vanish. The
    // projected KKT residual must vanish and the outward normal gradient makes
    // the active variable strongly active in the curvature test.
    Vector one_x0( 1 ), one_lower( 1 ), one_upper( 1 );
    one_x0[0]             = 0;
    one_lower[0]          = 0;
    one_upper[0]          = 1;
    auto boundary_problem = Utils::SmallTRON::make_problem<Scalar>(
      []( ConstVectorRef x ) { return Scalar( 0.5 ) * ( x[0] - Scalar( 2 ) ) * ( x[0] - Scalar( 2 ) ); },
      []( ConstVectorRef x, VectorRef g ) { g[0] = x[0] - Scalar( 2 ); },
      []( ConstVectorRef, MatrixRef H ) { H.setConstant( Scalar( 1 ) ); } );
    Utils::SmallTRON::Solver<Scalar> boundary_solver( 1, strict_options );
    auto const boundary_result = boundary_solver.solve( boundary_problem, one_x0, one_lower, one_upper );
    check( boundary_result.status == Status::first_order );
    check( std::abs( boundary_result.x[0] - Scalar( 1 ) ) <= 1e-12 );
    check( boundary_result.dual_feas <= boundary_result.optimality_tolerance );
    check( boundary_result.step_norm <= boundary_result.effective_step_tolerance );

    // A stationary point with negative curvature is not accepted. Starting at
    // the interior maximum of -x^2/2 must escape to a constrained minimum.
    one_x0[0]           = 0;
    one_lower[0]        = -1;
    one_upper[0]        = 1;
    auto saddle_problem = Utils::SmallTRON::make_problem<Scalar>(
      []( ConstVectorRef x ) { return -Scalar( 0.5 ) * x[0] * x[0]; },
      []( ConstVectorRef x, VectorRef g ) { g[0] = -x[0]; },
      []( ConstVectorRef, MatrixRef H ) { H.setConstant( Scalar( -1 ) ); } );
    Utils::SmallTRON::Solver<Scalar> saddle_solver( 1, strict_options );
    auto const saddle_result = saddle_solver.solve( saddle_problem, one_x0, one_lower, one_upper );
    check( saddle_result.status == Status::first_order );
    check( std::abs( std::abs( saddle_result.x[0] ) - Scalar( 1 ) ) <= 1e-12 );
    check( saddle_result.objective < Scalar( -0.49 ) );

    // The same negative-curvature escape must work at a weakly active bound.
    // Treating every geometrically active variable as fixed would incorrectly
    // accept x=0 here, even though the feasible direction +1 decreases f.
    one_x0[0]    = 0;
    one_lower[0] = 0;
    one_upper[0] = 1;
    Utils::SmallTRON::Solver<Scalar> weak_boundary_solver( 1, strict_options );
    auto const weak_boundary_result = weak_boundary_solver.solve( saddle_problem, one_x0, one_lower, one_upper );
    check( weak_boundary_result.status == Status::first_order );
    check(
      std::abs( weak_boundary_result.x[0] - Scalar( 1 ) ) <= Scalar( 64 ) * std::numeric_limits<Scalar>::epsilon() );
    check( weak_boundary_result.objective < Scalar( -0.49 ) );

    fmt::print(
      passed ? fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold
             : fmt::fg( fmt::color::red ) | fmt::emphasis::bold,
      "Direct Eigen, Ref/Map, cubic/roundoff, KKT/step and curvature checks: {}\n",
      passed ? "PASSED" : "FAILED" );
    return passed;
  }

  void run_test( std::string const & name, std::shared_ptr<NDbase<Scalar>> const & problem )
  {
    Vector const lower = problem->lower();
    Vector const upper = problem->upper();
    Vector const x0    = problem->init();
    Vector const exact = problem->exact();

    auto hessian = [&]( ConstVectorRef x, MatrixRef H ) { H = problem->hessian( x ); };

    Utils::SmallTRON::Options<Scalar> options;
    options.max_iter                        = static_cast<int>( MAX_ITERATIONS );
    options.max_eval                        = static_cast<int>( MAX_ITERATIONS + 1 );
    options.max_projected_newton_iterations = 50;
    options.set_tolerances( 1e-12 );
    options.use_cubic_radius = false;

    auto tron_problem = Utils::SmallTRON::make_problem<Scalar>(
      [&]( ConstVectorRef x ) { return ( *problem )( x ); },
      [&]( ConstVectorRef x, VectorRef g ) { g = problem->gradient( x ); },
      hessian );
    Utils::SmallTRON::Solver<Scalar> solver( x0.size(), options );

    auto const tron_result = solver.solve( tron_problem, x0, lower, upper );

    TestResult result;
    result.problem_name            = name;
    result.status                  = tron_result.status;
    result.dimension               = static_cast<int>( lower.size() );
    result.iterations              = static_cast<std::size_t>( tron_result.iter );
    result.function_evaluations    = static_cast<std::size_t>( tron_result.obj_evals );
    result.gradient_evaluations    = static_cast<std::size_t>( tron_result.grad_evals );
    result.hessian_evaluations     = static_cast<std::size_t>( tron_result.hess_evals );
    result.final_function_value    = tron_result.objective;
    result.projected_gradient_norm = tron_result.dual_feas;
    result.solution_error          = exact.size() == tron_result.x.size() ? ( tron_result.x - exact ).stableNorm()
                                                                          : std::numeric_limits<Scalar>::quiet_NaN();

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
    std::size_t total_hessian_evaluations{ 0 };

    for ( auto const & result : global_test_results )
    {
      if ( result.status == Status::first_order )
        ++converged;
      else if (
        result.status == Status::max_iter || result.status == Status::max_eval || result.status == Status::small_step )
        ++limited;
      else
        ++failed;
      total_iterations += result.iterations;
      total_function_evaluations += result.function_evaluations;
      total_gradient_evaluations += result.gradient_evaluations;
      total_hessian_evaluations += result.hessian_evaluations;
    }

    auto const   executed           = global_test_results.size();
    double const success_rate       = executed == 0
                                        ? 0.0
                                        : 100.0 * static_cast<double>( converged ) / static_cast<double>( executed );
    double const average_iterations = executed == 0
                                        ? 0.0
                                        : static_cast<double>( total_iterations ) / static_cast<double>( executed );

    print_rule( "═" );
    fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "SmallTRON SUMMARY\n" );
    fmt::print( "  Problems: {} executed, {} skipped\n", executed, skipped );
    fmt::print( "  Status:   " );
    fmt::print( fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold, "{} converged", converged );
    fmt::print( ", " );
    fmt::print( fmt::fg( fmt::color::gold ), "{} stopped", limited );
    fmt::print( ", " );
    fmt::print( fmt::fg( fmt::color::red ) | fmt::emphasis::bold, "{} failed", failed );
    fmt::print( "  ({:.1f}% converged)\n", success_rate );
    fmt::print(
      "  Work:     {} iterations (avg {:.1f}), {} f, {} g, {} H evaluations\n",
      total_iterations,
      average_iterations,
      total_function_evaluations,
      total_gradient_evaluations,
      total_hessian_evaluations );
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
    "SmallTRON — BOUND-CONSTRAINED OPTIMIZATION ON ND_func ({})\n",
    NL_list.size() );
  fmt::print(
    "Newton model, projected Cauchy step and direct Eigen solve on "
    "the active face\n" );
  fmt::print( "Maximum iterations per problem: {}\n", MAX_ITERATIONS );
  bool const direct_subproblems_passed = test_direct_subproblems();
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
      return result.status == Status::unknown || result.status == Status::unbounded ||
             result.status == Status::neg_pred || result.status == Status::direct_solver_failure ||
             result.status == Status::user || !std::isfinite( result.final_function_value ) ||
             !std::isfinite( result.projected_gradient_norm );
    } );
  return fatal_failure || !direct_subproblems_passed ? 1 : 0;
}
