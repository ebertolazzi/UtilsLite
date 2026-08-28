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

// -----------------------------------------------------------------------------
// File: test_SubspaceNewton.cc
// -----------------------------------------------------------------------------

#include "Utils_fmt.hh"
#include "Utils_nonlinear_system_NewtonSubspace.hh"
#include "Utils_nonlinear_system.hh"
#include "Utils_TicToc.hh"
#include "Utils_progress_bar.hh"

using namespace Utils;
using namespace std;

struct TestResult
{
  string test_name;
  int    num_equations;
  bool   converged;
  int    outer_iterations;
  int    total_iterations;
  int    function_evals;
  int    jacobian_evals;
  double final_residual;
  double elapsed_time_ms;
  int    initial_point_index;

  TestResult()
    : num_equations( 0 )
    , converged( false )
    , outer_iterations( 0 )
    , total_iterations( 0 )
    , function_evals( 0 )
    , jacobian_evals( 0 )
    , final_residual( 0.0 )
    , elapsed_time_ms( 0.0 )
    , initial_point_index( -1 )
  {
  }
};

struct Statistics
{
  int    total_tests;
  int    converged_tests;
  int    failed_tests;
  double success_rate;
  double avg_outer_iterations;
  double avg_total_iterations;
  double avg_function_evals;
  double avg_jacobian_evals;
  double avg_time_ms;
  double max_time_ms;
  double min_time_ms;

  Statistics()
    : total_tests( 0 )
    , converged_tests( 0 )
    , failed_tests( 0 )
    , success_rate( 0.0 )
    , avg_outer_iterations( 0.0 )
    , avg_total_iterations( 0.0 )
    , avg_function_evals( 0.0 )
    , avg_jacobian_evals( 0.0 )
    , avg_time_ms( 0.0 )
    , max_time_ms( 0.0 )
    , min_time_ms( numeric_limits<double>::max() )
  {
  }
};

string truncate_string( string const & str, size_t max_length )
{
  if ( str.length() <= max_length ) return str;
  return str.substr( 0, max_length - 3 ) + "...";
}

void print_progress( int current, int total )
{
  double progress = static_cast<double>( current ) / static_cast<double>( total );
  Utils::progress_bar( std::cout, progress, 50, "Progress:" );
}

void print_summary_table( const vector<TestResult> & results )
{
  constexpr int col_idx      = 5;
  constexpr int col_status   = 6;
  constexpr int col_neq      = 5;
  constexpr int col_oiter    = 5;
  constexpr int col_titer    = 5;
  constexpr int col_feval    = 6;
  constexpr int col_jeval    = 6;
  constexpr int col_residual = 10;
  constexpr int col_time     = 10;
  constexpr int col_name     = 40;

  constexpr int total_width = 2 + col_idx + 3 + col_status + 3 + col_neq + 3 + col_oiter + 3 + col_titer + 3 +
                              col_feval + 3 + col_jeval + 3 + col_residual + 3 + col_time + 3 + col_name + 2;

  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n\n"
    "┏{:━^{}}┓\n"
    "┠{:─^{}}┨\n",
    " TWO-LEVEL SUBSPACE NEWTON TEST RESULTS ",
    total_width - 2,
    "",
    total_width - 2 );

  fmt::print( fg( fmt::color::cyan ), "┃ " );
  fmt::print( "{:>{}} │ ", "#", col_idx );
  fmt::print( "{:^{}} │ ", "Status", col_status );
  fmt::print( "{:>{}} │ ", "NEQ", col_neq );
  fmt::print( "{:>{}} │ ", "Outer", col_oiter );
  fmt::print( "{:>{}} │ ", "Total", col_titer );
  fmt::print( "{:>{}} │ ", "F-Eval", col_feval );
  fmt::print( "{:>{}} │ ", "J-Eval", col_jeval );
  fmt::print( "{:>{}} │ ", "Residual", col_residual );
  fmt::print( "{:>{}} │ ", "Time(ms)", col_time );
  fmt::print( "{:<{}} ", "Test Name", col_name );
  fmt::print( fg( fmt::color::cyan ), "┃\n" );

  fmt::print( fg( fmt::color::cyan ), "┠{:─^{}}┨\n", "", total_width - 2 );

  for ( size_t i = 0; i < results.size(); ++i )
  {
    auto const & r = results[i];

    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:>{}} │ ", i + 1, col_idx );

    if ( r.converged ) { fmt::print( fg( fmt::color::green ), "{:^{}}", "✓ OK", col_status ); }
    else
    {
      fmt::print( fg( fmt::color::red ), "{:^{}}", "✗ FAIL", col_status );
    }
    fmt::print( fg( fmt::color::cyan ), " │ " );

    fmt::print( "{:>{}} │ ", r.num_equations, col_neq );
    fmt::print( "{:>{}} │ ", r.outer_iterations, col_oiter );
    fmt::print( "{:>{}} │ ", r.total_iterations, col_titer );
    fmt::print( "{:>{}} │ ", r.function_evals, col_feval );
    fmt::print( "{:>{}} │ ", r.jacobian_evals, col_jeval );
    fmt::print( "{:>10} │ ", fmt::format( "{:.2e}", r.final_residual ) );
    fmt::print( "{:>10} │ ", fmt::format( "{:.3f}", r.elapsed_time_ms ) );
    fmt::print( "{:<{}} ", truncate_string( r.test_name, col_name ), col_name );

    fmt::print( fg( fmt::color::cyan ), "┃\n" );
  }

  fmt::print( fg( fmt::color::cyan ), "┗{}┛\n", fmt::format( "{:━^{}}", "", total_width - 2 ) );
}

void print_statistics( const vector<TestResult> & results )
{
  Statistics stats;
  stats.total_tests = results.size();

  double total_outer_iter     = 0.0;
  double total_iter           = 0.0;
  double total_function_evals = 0.0;
  double total_jacobian_evals = 0.0;
  double total_time           = 0.0;

  for ( auto const & r : results )
  {
    if ( r.converged )
    {
      stats.converged_tests++;
      total_outer_iter += r.outer_iterations;
      total_iter += r.total_iterations;
      total_function_evals += r.function_evals;
      total_jacobian_evals += r.jacobian_evals;
      total_time += r.elapsed_time_ms;

      if ( r.elapsed_time_ms > stats.max_time_ms ) stats.max_time_ms = r.elapsed_time_ms;
      if ( r.elapsed_time_ms < stats.min_time_ms ) stats.min_time_ms = r.elapsed_time_ms;
    }
    else
    {
      stats.failed_tests++;
    }
  }

  stats.success_rate = 100.0 * stats.converged_tests / stats.total_tests;

  if ( stats.converged_tests > 0 )
  {
    stats.avg_outer_iterations = total_outer_iter / stats.converged_tests;
    stats.avg_total_iterations = total_iter / stats.converged_tests;
    stats.avg_function_evals   = total_function_evals / stats.converged_tests;
    stats.avg_jacobian_evals   = total_jacobian_evals / stats.converged_tests;
    stats.avg_time_ms          = total_time / stats.converged_tests;
  }

  constexpr int stat_col_label   = 25;
  constexpr int stat_col_value   = 12;
  constexpr int stat_total_width = stat_col_label + stat_col_value + 4;

  fmt::print( "\n" );
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "╭{:─^{}}╮\n",
    " STATISTICAL SUMMARY ",
    stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "├{}┤\n", fmt::format( "{:─^{}}", "", stat_total_width - 2 ) );
  fmt::print( fg( fmt::color::cyan ), "│" );
  fmt::print( "{:^{}}", "", stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "│\n" );

  fmt::print( fg( fmt::color::cyan ), "│ " );
  fmt::print( "{:<{}}", "Total Tests:", stat_col_label );
  fmt::print( fg( fmt::color::white ), "{:>{}}", stats.total_tests, stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " │\n" );

  fmt::print( fg( fmt::color::cyan ), "│ " );
  fmt::print( "{:<{}}", "Converged Tests:", stat_col_label );
  fmt::print(
    fg( fmt::color::green ),
    "{:>{}}",
    fmt::format( "{} ({:.1f}%)", stats.converged_tests, stats.success_rate ),
    stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " │\n" );

  fmt::print( fg( fmt::color::cyan ), "│ " );
  fmt::print( "{:<{}}", "Failed Tests:", stat_col_label );
  fmt::print(
    fg( fmt::color::red ),
    "{:>{}}",
    fmt::format( "{} ({:.1f}%)", stats.failed_tests, 100.0 - stats.success_rate ),
    stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " │\n" );

  fmt::print( fg( fmt::color::cyan ), "├{}┤\n", fmt::format( "{:─^{}}", "", stat_total_width - 2 ) );

  if ( stats.converged_tests > 0 )
  {
    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Avg Outer Iterations:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_outer_iterations, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Avg Total Iterations:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_total_iterations, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Avg Function Evals:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_function_evals, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Avg Jacobian Evals:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_jacobian_evals, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Avg Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Min Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.min_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Max Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.max_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );
  }

  fmt::print( fg( fmt::color::cyan ), "│" );
  fmt::print( "{:^{}}", "", stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "│\n" );
  fmt::print( fg( fmt::color::cyan ), "╰{}╯\n", fmt::format( "{:─^{}}", "", stat_total_width - 2 ) );
}

int main( int argc, char * argv[] )
{
  Utils::TicToc tm;

  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n\n"
    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "━━━━━━━━━━━┓\n"
    "┃                  TWO-LEVEL SUBSPACE NEWTON TEST SUITE         "
    "           ┃\n"
    "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "━━━━━━━━━━━┛\n"
    "\n" );

  init_nonlinear_system_tests();

  fmt::print( fg( fmt::color::yellow ), "Total number of test problems: " );
  fmt::print( fg( fmt::color::white ), "{}\n\n", nonlinear_system_tests.size() );

  // Default parameters
  double tolerance      = 1e-8;
  int    max_iterations = 100;
  bool   verbose_mode   = false;
  int    block_size     = 1;
  string strategy_name  = "random";

  // Inner solver parameters
  int    inner_max_iter   = 5;
  double inner_tol        = 0.1;
  double inner_damping    = 1e-6;
  bool   inner_linesearch = true;

  auto print_usage = [&]()
  {
    fmt::print( "Usage: {} [options]\n", argv[0] );
    fmt::print( "Options:\n" );
    fmt::print( "  --help                    Show help\n" );
    fmt::print( "  --verbose                 Enable verbose output\n" );
    fmt::print( "  --max-iter=N              Set max outer iterations\n" );
    fmt::print( "  --tolerance=VAL           Set residual tolerance\n" );
    fmt::print( "  --block-size=N            Set variable block size\n" );
    fmt::print( "  --strategy=cyclic|random|greedy\n" );
    fmt::print( "  --inner-max-iter=N        Set max inner iterations (default: 5)\n" );
    fmt::print( "  --inner-tolerance=VAL     Set inner tolerance (default: 0.1)\n" );
    fmt::print( "  --inner-damping=VAL       Set inner damping (default: 1e-6)\n" );
    fmt::print( "  --no-inner-linesearch     Disable inner line search\n" );
  };

  // Parse arguments
  for ( int i = 1; i < argc; ++i )
  {
    string arg = argv[i];
    if ( arg == "--help" || arg == "-h" )
    {
      print_usage();
      return 0;
    }
    else if ( arg == "--verbose" ) { verbose_mode = true; }
    else if ( arg.rfind( "--max-iter=", 0 ) == 0 ) { max_iterations = stoi( arg.substr( 11 ) ); }
    else if ( arg.rfind( "--tolerance=", 0 ) == 0 ) { tolerance = stod( arg.substr( 12 ) ); }
    else if ( arg.rfind( "--block-size=", 0 ) == 0 ) { block_size = stoi( arg.substr( 13 ) ); }
    else if ( arg.rfind( "--strategy=", 0 ) == 0 ) { strategy_name = arg.substr( 11 ); }
    else if ( arg.rfind( "--inner-max-iter=", 0 ) == 0 ) { inner_max_iter = stoi( arg.substr( 17 ) ); }
    else if ( arg.rfind( "--inner-tolerance=", 0 ) == 0 ) { inner_tol = stod( arg.substr( 18 ) ); }
    else if ( arg.rfind( "--inner-damping=", 0 ) == 0 ) { inner_damping = stod( arg.substr( 16 ) ); }
    else if ( arg == "--no-inner-linesearch" ) { inner_linesearch = false; }
    else
    {
      fmt::print( fg( fmt::color::yellow ), "Unknown option: {}\n", arg );
      print_usage();
      return 1;
    }
  }

  fmt::print( fg( fmt::color::yellow ), "Outer tolerance: " );
  fmt::print( "{}\n", tolerance );
  fmt::print( fg( fmt::color::yellow ), "Max outer iterations: " );
  fmt::print( "{}\n", max_iterations );
  fmt::print( fg( fmt::color::yellow ), "Block size: " );
  fmt::print( "{}\n", block_size );
  fmt::print( fg( fmt::color::yellow ), "Strategy: " );
  fmt::print( "{}\n", strategy_name );
  fmt::print( fg( fmt::color::yellow ), "Inner max iterations: " );
  fmt::print( "{}\n", inner_max_iter );
  fmt::print( fg( fmt::color::yellow ), "Inner tolerance: " );
  fmt::print( "{}\n", inner_tol );
  fmt::print( fg( fmt::color::yellow ), "Inner damping: " );
  fmt::print( "{}\n", inner_damping );
  fmt::print( fg( fmt::color::yellow ), "Inner line search: " );
  fmt::print( "{}\n\n", inner_linesearch ? "ON" : "OFF" );

  // Convert strategy
  TwoLevelSubspaceNewton::SelectionStrategy strategy = TwoLevelSubspaceNewton::RANDOM_UNIFORM;
  if ( strategy_name == "cyclic" )
    strategy = TwoLevelSubspaceNewton::CYCLIC;
  else if ( strategy_name == "greedy" )
    strategy = TwoLevelSubspaceNewton::GREEDY;

  // Results
  vector<TestResult> all_results;

  // Instantiate solver
  TwoLevelSubspaceNewton solver;
  solver.set_max_iterations( max_iterations );
  solver.set_tolerance( tolerance );
  solver.set_block_size( block_size );
  solver.set_verbose_level( verbose_mode ? 2 : 0 );
  solver.set_strategy( strategy );

  // Configure inner solver
  solver.set_inner_max_iterations( inner_max_iter );
  solver.set_inner_tolerance( inner_tol );
  solver.set_inner_damping( inner_damping );
  solver.enable_inner_line_search( inner_linesearch );

  // Loop over test suite
  for ( size_t test_idx = 0; test_idx < nonlinear_system_tests.size(); ++test_idx )
  {
    if ( !verbose_mode ) print_progress( test_idx, nonlinear_system_tests.size() );

    NonlinearSystem * system = nonlinear_system_tests[test_idx];

    vector<TwoLevelSubspaceNewton::Vector> initial_points;
    system->initial_points( initial_points );

    if ( initial_points.empty() )
    {
      TestResult r;
      r.test_name     = system->title();
      r.num_equations = system->num_equations();
      all_results.push_back( r );
      continue;
    }

    for ( size_t ip = 0; ip < initial_points.size(); ++ip )
    {
      TwoLevelSubspaceNewton::Vector x = initial_points[ip];

      tm.tic();
      bool converged = solver.solve( *system, x );
      tm.toc();

      TestResult r;
      r.test_name           = system->title();
      r.num_equations       = system->num_equations();
      r.converged           = converged;
      r.outer_iterations    = solver.get_outer_iterations();
      r.total_iterations    = solver.get_total_iterations();
      r.function_evals      = solver.get_function_evals();
      r.jacobian_evals      = solver.get_jacobian_evals();
      r.final_residual      = solver.final_residual();
      r.elapsed_time_ms     = tm.elapsed_ms();
      r.initial_point_index = ip;

      all_results.push_back( r );
    }
  }

  if ( !verbose_mode ) print_progress( nonlinear_system_tests.size(), nonlinear_system_tests.size() );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n\nAll tests completed!\n" );

  print_summary_table( all_results );
  print_statistics( all_results );

  for ( auto * sys : nonlinear_system_tests ) delete sys;
  fmt::print( fg( fmt::color::cyan ), "\nTest suite completed successfully.\n\n" );
  return 0;
}
