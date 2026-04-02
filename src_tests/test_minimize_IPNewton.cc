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

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>
#include <optional>
#include <functional>
#include <limits>
#include <algorithm>
#include <numeric>

#include "Utils_minimize_IPNewton.hh"

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif

using std::map;
using std::pair;
using std::string;
using std::vector;
using Scalar       = double;
using MINIMIZER    = Utils::IPNewton_minimizer<Scalar>;
using integer      = MINIMIZER::integer;
using Vector       = typename MINIMIZER::Vector;
using SparseMatrix = typename MINIMIZER::SparseMatrix;

using Status = MINIMIZER::Status;

string status_to_string( Status s )
{
  using Status = MINIMIZER::Status;
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

struct TestResult
{
  string  problem_name;
  Vector  final_solution;
  integer dimension;
  Status  status;
  Scalar  final_gradient_norm{ 0 };

  // Dati ora ottenuti dal minimizer
  integer outer_iterations{ 0 };
  integer total_inner_iterations{ 0 };
  integer function_evals{ 0 };
  integer hessian_evals{ 0 };
  Scalar  final_f{ 0 };
  Scalar  initial_f{ 0 };
  Scalar  final_mu{ 0 };
  Scalar  duality_gap{ 0 };
  Scalar  primal_infeasibility{ 0 };
  Scalar  dual_infeasibility{ 0 };
  integer newton_steps{ 0 };
  integer gradient_steps{ 0 };
  integer centering_steps{ 0 };
};

vector<TestResult> global_test_results;

#include "ND_func.cxx"

inline string format_reduced_vector( Vector const & v, integer max_size = 10 )
{
  string  tmp{ "[" };
  integer v_size = v.size();

  if ( v_size == 0 ) { return "[]"; }

  if ( v_size <= max_size )
  {
    for ( integer i = 0; i < v_size; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
  }
  else
  {
    for ( integer i = 0; i < max_size - 3; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
    tmp += "..., ";
    for ( integer i = v_size - 3; i < v_size; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
  }

  if ( v_size > 0 )
  {
    tmp.pop_back();
    tmp.pop_back();
  }
  tmp += "]";
  return tmp;
}

template <typename T, typename = void> struct has_hessian : std::false_type
{
};

template <typename T>
struct has_hessian<T, std::void_t<decltype( std::declval<T>().hessian( std::declval<Vector>() ) )>> : std::true_type
{
};

template <typename Problem> static void test( Problem & tp, string const & name )
{
  MINIMIZER::Options opts;

  // Simplified options
  opts.max_outer_iterations = 50;
  opts.max_inner_iterations = 200;
  opts.tol                  = 1e-8;
  opts.f_tol                = 1e-10;
  opts.x_tol                = 1e-8;

  opts.mu_init            = 0.1;
  opts.mu_min             = 1e-12;
  opts.mu_decrease_factor = 0.2;
  opts.sigma              = 0.5;

  opts.epsilon_feas   = 1e-8;
  opts.epsilon_infeas = 1e-6;

  opts.alpha_min                  = 1e-8;
  opts.alpha_reduction            = 0.5;
  opts.max_line_search_iterations = 20;
  opts.armijo_c1                  = 1e-4;

  opts.adaptive_mu_update   = true;
  opts.max_barrier_failures = 5;

  opts.max_no_progress_iterations = 10;
  opts.progress_tol               = 1e-8;

  opts.verbosity = 3;

  Vector x0 = tp.init();

  Vector final_solution = x0;

  auto cb = [&tp, &final_solution]( Vector const & x, Vector * g, SparseMatrix * H ) -> Scalar
  {
    final_solution = x;

    if ( g ) *g = tp.gradient( x );

    if ( H )
    {
      if constexpr ( has_hessian<Problem>::value ) { *H = tp.hessian( x ); }
      else
      {
        H->resize( x.size(), x.size() );
        H->setZero();

        Scalar eps = std::sqrt( std::numeric_limits<Scalar>::epsilon() );
        Vector xp = x, xm = x;

        for ( integer i = 0; i < x.size(); ++i )
        {
          xp( i ) += eps;
          xm( i ) -= eps;

          Vector gp = tp.gradient( xp );
          Vector gm = tp.gradient( xm );

          for ( integer j = 0; j < x.size(); ++j ) H->coeffRef( j, i ) = ( gp( j ) - gm( j ) ) / ( 2 * eps );

          xp( i ) = xm( i ) = x( i );
        }

        SparseMatrix Ht = H->transpose();
        *H              = 0.5 * ( ( *H ) + Ht );
      }
    }
    return tp( x );
  };

  MINIMIZER m( opts );
  try
  {
    m.set_bounds( tp.lower(), tp.upper() );
  }
  catch ( ... )
  {
    // If no bounds, do nothing
  }

  m.minimize( x0, cb );

  // Estrai i risultati dal minimizer
  TestResult tr;
  tr.problem_name   = name;
  tr.final_solution = final_solution;
  tr.dimension      = tr.final_solution.size();
  tr.status         = m.status();

  // Copia i dati dal minimizer
  tr.outer_iterations       = m.outer_iterations();
  tr.total_inner_iterations = m.total_inner_iterations();
  tr.function_evals         = m.function_evals();
  tr.hessian_evals          = m.hessian_evals();
  tr.final_f                = m.final_f();
  tr.initial_f              = m.initial_f();
  tr.final_mu               = m.final_mu();
  tr.duality_gap            = m.duality_gap();
  tr.primal_infeasibility   = m.primal_infeasibility();
  tr.dual_infeasibility     = m.dual_infeasibility();
  tr.newton_steps           = m.newton_steps();
  tr.gradient_steps         = m.gradient_steps();
  tr.centering_steps        = m.centering_steps();

  // Calculate final gradient norm
  Vector grad_final      = tp.gradient( tr.final_solution );
  tr.final_gradient_norm = grad_final.norm();

  global_test_results.push_back( tr );

  fmt::print(
    "{}: {} | outer = {} | inner = {} | f = {:.6e} | ‖g‖ = {:.3e} | μ = {:.1e}\n",
    name,
    status_to_string( tr.status ),
    tr.outer_iterations,
    tr.total_inner_iterations,
    tr.final_f,
    tr.final_gradient_norm,
    tr.final_mu );

  if ( opts.verbosity >= 2 ) { fmt::print( "Solution: {}\n\n", format_reduced_vector( tr.final_solution ) ); }
}

void print_summary_table()
{
  fmt::print(
    "\n\n"
    "╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                         IP NEWTON SUMMARY RESULTS                                              ║\n"
    "╠════════════════════════╤════════╤══════════╤══════════╤════════════════╤═══════════════╤═══════════════════════╣\n"
    "║ Function               │ Dim    │ Outer    │ Inner    │ Final Value    │ ‖g_final‖     │ Status                ║\n"
    "╠════════════════════════╪════════╪══════════╪══════════╪════════════════╪═══════════════╪═══════════════════════╣"
    "\n" );

  for ( auto const & result : global_test_results )
  {
    string status_str = status_to_string( result.status );
    bool   converged  = result.status == Status::CONVERGED;

    auto status_color = converged                                     ? fmt::fg( fmt::color::green )
                        : ( result.status == Status::MAX_ITERATIONS ) ? fmt::fg( fmt::color::yellow )
                                                                      : fmt::fg( fmt::color::red );

    auto grad_color = ( result.final_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                      : ( result.final_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                      : ( result.final_gradient_norm < 1e-4 ) ? fmt::fg( fmt::color::orange )
                                                              : fmt::fg( fmt::color::red );

    string problem_name = result.problem_name;
    if ( problem_name.length() > 22 ) { problem_name = problem_name.substr( 0, 19 ) + "..."; }

    fmt::print(
      "║ {:<22} │ {:>6} │ {:>8} │ {:>8} │ {:<14.4e} │ ",
      problem_name,
      result.dimension,
      result.outer_iterations,
      result.total_inner_iterations,
      result.final_f );

    fmt::print( grad_color, "{:<13.2e}", result.final_gradient_norm );
    fmt::print( " │ " );

    fmt::print( status_color, "{:<21}", status_str );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    "╚════════════════════════╧════════╧══════════╧══════════╧════════════════╧═══════════════╧═══════════════════════╝"
    "\n" );

  integer total_tests     = global_test_results.size();
  integer converged_tests = std::count_if(
    global_test_results.begin(),
    global_test_results.end(),
    []( const TestResult & r ) { return r.status == Status::CONVERGED; } );

  integer accumulated_outer_iter{ 0 };
  integer accumulated_inner_iter{ 0 };
  integer accumulated_evals{ 0 };
  integer accumulated_hess_evals{ 0 };
  integer accumulated_newton_steps{ 0 };
  integer accumulated_gradient_steps{ 0 };

  for ( auto const & r : global_test_results )
  {
    if ( r.status == Status::CONVERGED )
    {
      accumulated_outer_iter += r.outer_iterations;
      accumulated_inner_iter += r.total_inner_iterations;
      accumulated_evals += r.function_evals;
    }
    accumulated_hess_evals += r.hessian_evals;
    accumulated_newton_steps += r.newton_steps;
    accumulated_gradient_steps += r.gradient_steps;
  }

  auto perc =
    100.0 * static_cast<double>( converged_tests ) / static_cast<double>( std::max<size_t>( total_tests, 1 ) );

  fmt::print( fmt::fg( fmt::color::light_blue ), "\n📊 Statistics:\n" );
  fmt::print( "   • Total problems: {}\n", total_tests );
  fmt::print( "   • Converged: {} ({:.1f}%)\n", converged_tests, perc );
  fmt::print( "   • Total outer iterations: {}\n", accumulated_outer_iter );
  fmt::print( "   • Total inner iterations: {}\n", accumulated_inner_iter );
  fmt::print( "   • Total function evaluations: {}\n", accumulated_evals );
  fmt::print( "   • Total hessian evaluations: {}\n", accumulated_hess_evals );
  fmt::print( "   • Total Newton steps: {}\n", accumulated_newton_steps );
  fmt::print( "   • Total gradient steps: {}\n", accumulated_gradient_steps );
}

int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║              IP NEWTON Optimization Test Suite                 ║\n"
    "║               (Logarithmic Barrier)                            ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  for ( auto [ptr, name] : NL_list ) test( *ptr, name );

  print_summary_table();

  return 0;
}
