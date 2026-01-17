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
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "Utils_fmt.hh"
#include "Utils_eigen.hh"
#include "Utils_minimize_NelderMead_BlockCoordinate.hh"
#include "ND_func.cxx"

#if defined( __clang__ )
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wunused-macros"
#elif defined( __llvm__ ) || defined( __GNUC__ )
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-macros"
#elif defined( _MSC_VER )
#pragma warning( disable : 4100 )
#pragma warning( disable : 4101 )
#pragma warning( disable : 4189 )
#endif


using Scalar   = double;
using NM_Block = Utils::NelderMead_BlockCoordinate<Scalar>;
using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

struct TestResult
{
  std::string problem_name;
  Scalar      final_value{ 0 };
  size_t      dimension{ 0 };
  size_t      outer_iters{ 0 };
  size_t      inner_iters{ 0 };
  size_t      total_evaluations{ 0 };
  std::string status_str;
};

static std::vector<TestResult> global_results;

// Safe initialization
template <typename ProblemFunc> Vector get_safe_initial_point( ProblemFunc & problem )
{
  try
  {
    return problem.init();
  }
  catch ( ... )
  {
  }
  Vector L   = problem.lower();
  Vector U   = problem.upper();
  size_t dim = L.size();
  Vector x0( dim );
  for ( size_t i = 0; i < dim; ++i )
  {
    if ( std::isfinite( L( i ) ) && std::isfinite( U( i ) ) )
      x0( i ) = ( L( i ) + U( i ) ) / 2.0;
    else if ( std::isfinite( L( i ) ) )
      x0( i ) = L( i ) + 1.0;
    else if ( std::isfinite( U( i ) ) )
      x0( i ) = U( i ) - 1.0;
    else
      x0( i ) = 0.5;
  }
  return x0;
}

template <typename ProblemFunc> void test( ProblemFunc & problem, std::string const & name, int verbosity_level = 1 )
{
  Vector L        = problem.lower();
  Vector U        = problem.upper();
  size_t dim      = L.size();
  Vector x0       = get_safe_initial_point( problem );
  Scalar init_val = problem( x0 );

  // 1. Print Header
  fmt::print( "\n" );
  fmt::print( "╔═════════════════════════════════════════════════════════════╗\n" );
  fmt::print( "║ TEST FUNCTION: {:<44} ║\n", name );
  fmt::print( "║ Dimension:     {:<44} ║\n", dim );
  fmt::print( "╚═════════════════════════════════════════════════════════════╝\n" );

  // Solver Configuration
  NM_Block::Options opts;

  opts.verbosity_level          = verbosity_level;
  opts.inner_progress_frequency = verbosity_level >= 2 ? 10 : 100;

  opts.sub_options.verbosity_level          = verbosity_level;
  opts.sub_options.inner_progress_frequency = verbosity_level >= 2 ? 10 : 100;

  opts.block_size               = 10;
  opts.max_outer_iterations     = 200;
  opts.max_function_evaluations = 500000;
  opts.tolerance                = 1e-7;
  opts.verbose                  = true;

  opts.sub_options.tolerance    = 1e-7;
  opts.sub_options.initial_step = 0.1;

  NM_Block solver( opts );
  solver.set_bounds( L, U );

  // 2. Run Optimization
  bool success = solver.minimize( x0, [&]( Vector const & x ) { return problem( x ); } );

  fmt::print( "\n" );
  fmt::print( "-> Initial Point:  {}\n", Utils::NelderMead::format_vector( x0 ) );
  fmt::print( "-> Final Point:    {}\n", Utils::NelderMead::format_vector( solver.get_solution() ) );
  fmt::print( "-> Initial Value:  {:.6e}\n", init_val );
  fmt::print( "-> Final Value:    {:.8e}\n", solver.get_final_function_value() );

  // 4. Print Final Point & Stats
  fmt::print( "-> Final Status:   {}\n", Utils::NelderMead::status_to_string( solver.get_status() ) );
  fmt::print( "-> Total Outer It: {}\n", solver.get_outer_iterations() );
  fmt::print( "-> Total Inner It: {}\n", solver.get_inner_iterations() );
  fmt::print( "-> Total Evals:    {}\n", solver.get_total_evaluations() );

  TestResult tr;
  tr.problem_name      = name;
  tr.dimension         = dim;
  tr.final_value       = solver.get_final_function_value();
  tr.status_str        = Utils::NelderMead::status_to_string( solver.get_status() );
  tr.outer_iters       = solver.get_outer_iterations();
  tr.inner_iters       = solver.get_inner_iterations();
  tr.total_evaluations = solver.get_total_evaluations();
  global_results.push_back( tr );
}

void print_summary_table()
{
  if ( global_results.empty() ) return;

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔═══════════════════════════════════════════════════════════════"
    "═══════════════════════╗\n"
    "║                                OPTIMIZATION SUMMARY           "
    "                       ║\n"
    "╠════════════════════════╤════════╤══════════╤══════════╤═══════"
    "═════════╤═════════════╣\n"
    "║ Function               │ Dim    │ Outer It │ Inner It │ Final "
    "Value    │ Status      ║\n"
    "╠════════════════════════╪════════╪══════════╪══════════╪═══════"
    "═════════╪═════════════╣\n" );

  for ( const auto & r : global_results )
  {
    auto status_color = r.status_str == "CONVERGED" ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::yellow );

    fmt::print(
      "║ {:<22} │ {:>6} │ {:>8} │ {:>8} │ {:<14.4e} │ ",
      r.problem_name.substr( 0, 22 ),
      r.dimension,
      r.outer_iters,
      r.inner_iters,
      r.final_value );
    fmt::print( status_color, "{:<11}", r.status_str );
    fmt::print( " ║\n" );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚════════════════════════╧════════╧══════════╧══════════╧═══════"
    "═════════╧═════════════╝\n" );

  // Aggiungere statistiche globali
  size_t total_evals     = 0;
  size_t converged_count = 0;
  for ( const auto & r : global_results )
  {
    total_evals += r.total_evaluations;
    if ( r.status_str == "CONVERGED" ) converged_count++;
  }

  fmt::print( "\n📊 Global Statistics:\n" );
  fmt::print( "   • Total problems: {}\n", global_results.size() );
  fmt::print( "   • Converged: {} ({:.1f}%)\n", converged_count, ( 100.0 * converged_count / global_results.size() ) );
  fmt::print( "   • Total evaluations: {}\n", total_evals );
}

int main()
{
  try
  {
    for ( auto [ptr, name] : NL_list ) test( *ptr, name );

    print_summary_table();
  }
  catch ( std::exception const & e )
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}
