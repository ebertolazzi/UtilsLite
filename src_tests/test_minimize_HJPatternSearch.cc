/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022                                                      |
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
#include <chrono>
#include <random>
#include <functional>
#include <memory>
#include <vector>
#include <map>
#include <algorithm>

#include "Utils_fmt.hh"
#include "Utils_minimize_HJPatternSearch.hh"

using namespace std;
using namespace Utils;

using real_type = double;

// ============================================================================
// Strutture dati per raccolta risultati
// ============================================================================

struct TestResultHJ
{
  std::string                   test_name;
  std::string                   function_name;
  int                           dimension;
  int                           iterations;
  int                           function_evaluations;
  real_type                     final_value;
  real_type                     final_step;
  real_type                     tolerance;
  real_type                     rho;
  bool                          success;
  std::string                   status;
  std::chrono::duration<double> time_taken;
  std::vector<real_type>        solution;
};

struct FunctionStats
{
  std::string name;
  int         dimension;
  size_t      total_tests{ 0 };
  size_t      successful_tests{ 0 };
  size_t      total_iterations{ 0 };
  size_t      total_evaluations{ 0 };
  real_type   average_iterations{ 0.0 };
  real_type   success_rate{ 0.0 };
  real_type   min_final_value{ std::numeric_limits<real_type>::max() };
  real_type   max_final_value{ std::numeric_limits<real_type>::lowest() };
};

// Collettore globale dei risultati
std::vector<TestResultHJ>            global_test_results;
std::map<std::string, FunctionStats> function_statistics;
std::map<std::string, size_t>        configuration_stats;

// ============================================================================
// Test Functions
// ============================================================================

static real_type rosenbrock( real_type const X[] )
{
  real_type const x = X[0];
  real_type const y = X[1];
  return 100.0 * pow( y - x * x, 2 ) + pow( 1.0 - x, 2 );
}

static real_type sphere( real_type const X[], int n )
{
  real_type sum = 0.0;
  for ( int i = 0; i < n; ++i ) { sum += X[i] * X[i]; }
  return sum;
}

static real_type sphere2d( real_type const X[] )
{
  return sphere( X, 2 );
}

static real_type rastrigin( real_type const X[], int n )
{
  real_type const pi  = 3.14159265358979323846;
  real_type       sum = 10.0 * n;
  for ( int i = 0; i < n; ++i ) { sum += X[i] * X[i] - 10.0 * cos( 2.0 * pi * X[i] ); }
  return sum;
}

static real_type rastrigin2d( real_type const X[] )
{
  return rastrigin( X, 2 );
}

static real_type himmelblau( real_type const X[] )
{
  real_type const x = X[0];
  real_type const y = X[1];
  return pow( x * x + y - 11.0, 2 ) + pow( x + y * y - 7.0, 2 );
}

// static real_type beale( real_type const X[] )
//{
//   real_type const x = X[0];
//   real_type const y = X[1];
//   return pow( 1.5 - x + x * y, 2 ) + pow( 2.25 - x + x * y * y, 2 ) + pow( 2.625 - x + x * y * y * y, 2 );
// }

// ============================================================================
// Helper Functions
// ============================================================================

std::string get_status_symbol( bool success, real_type final_value, real_type tolerance = 1e-6 )
{
  if ( success && final_value < tolerance ) { return fmt::format( fmt::fg( fmt::color::green ), "✓" ); }
  else if ( success ) { return fmt::format( fmt::fg( fmt::color::yellow ), "⚠" ); }
  else
  {
    return fmt::format( fmt::fg( fmt::color::red ), "✗" );
  }
}

std::string format_vector( const std::vector<real_type> & vec, size_t max_show = 3 )
{
  if ( vec.empty() ) return "[]";

  std::string result = "[";
  size_t      show   = std::min( vec.size(), max_show );

  for ( size_t i = 0; i < show; ++i )
  {
    result += fmt::format( "{:.4f}", vec[i] );
    if ( i < show - 1 ) result += ", ";
  }

  if ( vec.size() > max_show ) { result += fmt::format( ", ... (+{})", vec.size() - show ); }

  result += "]";
  return result;
}

void print_section_header( std::string const & title, int width = 80 )
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║ {:^{}} ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n",
    title,
    width - 4 );
}

void update_statistics( const TestResultHJ & result )
{
  // Update function statistics
  auto & func_stats    = function_statistics[result.function_name];
  func_stats.name      = result.function_name;
  func_stats.dimension = result.dimension;
  func_stats.total_tests++;

  if ( result.success )
  {
    func_stats.successful_tests++;
    func_stats.total_iterations += result.iterations;
    func_stats.total_evaluations += result.function_evaluations;
    func_stats.min_final_value = std::min( func_stats.min_final_value, result.final_value );
    func_stats.max_final_value = std::max( func_stats.max_final_value, result.final_value );
  }

  // Update configuration statistics
  std::string config_key = fmt::format( "tol={:.0e},ρ={:.2f}", result.tolerance, result.rho );
  configuration_stats[config_key]++;
}

// ============================================================================
// Test Runner
// ============================================================================

TestResultHJ run_hj_test(
  const std::string &                           test_name,
  const std::string &                           func_name,
  std::function<real_type( real_type const[] )> func,
  int                                           dimension,
  const real_type *                             x0,
  real_type                                     h0,
  real_type                                     tolerance       = 1e-6,
  real_type                                     rho             = 0.9,
  int                                           max_iterations  = 1000,
  int                                           max_fevals      = 5000,
  int                                           max_stagnations = 20,
  int                                           verbose         = 0 )
{
  TestResultHJ result;
  result.test_name     = test_name;
  result.function_name = func_name;
  result.dimension     = dimension;
  result.tolerance     = tolerance;
  result.rho           = rho;

  auto start_time = std::chrono::high_resolution_clock::now();

  try
  {
    Utils::Console             console( &cout, verbose );
    HJPatternSearch<real_type> solver( test_name );

    solver.setup( dimension, func, &console );
    solver.set_tolerance( tolerance );
    solver.set_max_iterations( max_iterations );
    solver.set_max_fun_evaluations( max_fevals );
    solver.set_max_stagnations( max_stagnations );
    solver.set_rho( rho );

    std::vector<real_type> x0_vec( x0, x0 + dimension );
    solver.run( x0_vec.data(), h0 );

    result.solution.resize( dimension );
    result.final_value          = solver.get_last_solution( result.solution.data() );
    result.iterations           = solver.get_iteration_count();
    result.function_evaluations = solver.get_fun_evaluation_count();
    result.final_step           = solver.get_current_step();

    // Determine success based on final step and value
    result.success = ( result.final_step <= tolerance || result.final_value < 1e-3 );

    if ( result.final_step <= tolerance ) { result.status = "Converged (step)"; }
    else if ( result.iterations >= max_iterations ) { result.status = "Max iterations"; }
    else if ( result.function_evaluations >= max_fevals ) { result.status = "Max evaluations"; }
    else if ( result.final_value < 1e-3 ) { result.status = "Converged (value)"; }
    else
    {
      result.status = "Other";
    }
  }
  catch ( const std::exception & e )
  {
    result.success     = false;
    result.status      = fmt::format( "Error: {}", e.what() );
    result.final_value = std::numeric_limits<real_type>::max();
  }

  auto end_time     = std::chrono::high_resolution_clock::now();
  result.time_taken = end_time - start_time;

  global_test_results.push_back( result );
  update_statistics( result );

  return result;
}

// ============================================================================
// Test Suites
// ============================================================================

void test_basic_functions()
{
  print_section_header( "Basic Function Tests" );

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔═══════════════════════╤════════╤══════════╤════════════════╤══════════════╤══════════╤════════════════════╗\n"
    "║ Function              │ Dim    │ Iter     │ Final Value    │ ‖Step‖       │ F Evals  │ Time (ms)          ║\n"
    "╠═══════════════════════╪════════╪══════════╪════════════════╪══════════════╪══════════╪════════════════════╣\n" );

  // Rosenbrock
  {
    real_type x0[]   = { -1.2, 1.0 };
    auto      result = run_hj_test( "Rosenbrock", "Rosenbrock", rosenbrock, 2, x0, 0.5 );

    auto symbol = get_status_symbol( result.success, result.final_value );
    fmt::print(
      "║ {}{:<20} │ {:>6} │ {:>8} │ {:>14.2e} │ {:>12.2e} │ {:>8} │ {:>18.2f} ║\n",
      symbol,
      result.function_name,
      result.dimension,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations,
      result.time_taken.count() * 1000.0 );
  }

  // Himmelblau
  {
    real_type x0[]   = { 3.5, 2.5 };
    auto      result = run_hj_test( "Himmelblau", "Himmelblau", himmelblau, 2, x0, 0.3 );

    auto symbol = get_status_symbol( result.success, result.final_value );
    fmt::print(
      "║ {}{:<20} │ {:>6} │ {:>8} │ {:>14.2e} │ {:>12.2e} │ {:>8} │ {:>18.2f} ║\n",
      symbol,
      result.function_name,
      result.dimension,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations,
      result.time_taken.count() * 1000.0 );
  }

  // Sphere 2D
  {
    real_type x0[]   = { 2.0, 2.0 };
    auto      result = run_hj_test( "Sphere2D", "Sphere2D", sphere2d, 2, x0, 0.5 );

    auto symbol = get_status_symbol( result.success, result.final_value );
    fmt::print(
      "║ {}{:<20} │ {:>6} │ {:>8} │ {:>14.2e} │ {:>12.2e} │ {:>8} │ {:>18.2f} ║\n",
      symbol,
      result.function_name,
      result.dimension,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations,
      result.time_taken.count() * 1000.0 );
  }

  // Sphere 5D
  {
    real_type x0[]     = { 2.0, 2.0, 2.0, 2.0, 2.0 };
    auto      sphere5d = []( real_type const X[] ) { return sphere( X, 5 ); };
    auto      result   = run_hj_test( "Sphere5D", "Sphere5D", sphere5d, 5, x0, 1.0 );

    auto symbol = get_status_symbol( result.success, result.final_value );
    fmt::print(
      "║ {}{:<20} │ {:>6} │ {:>8} │ {:>14.2e} │ {:>12.2e} │ {:>8} │ {:>18.2f} ║\n",
      symbol,
      result.function_name,
      result.dimension,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations,
      result.time_taken.count() * 1000.0 );
  }

  // Rastrigin 2D
  {
    real_type x0[]   = { 2.0, 2.0 };
    auto      result = run_hj_test( "Rastrigin2D", "Rastrigin2D", rastrigin2d, 2, x0, 0.5 );

    auto symbol = get_status_symbol( result.success, result.final_value );
    fmt::print(
      "║ {}{:<20} │ {:>6} │ {:>8} │ {:>14.2e} │ {:>12.2e} │ {:>8} │ {:>18.2f} ║\n",
      symbol,
      result.function_name,
      result.dimension,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations,
      result.time_taken.count() * 1000.0 );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚═══════════════════════╧════════╧══════════╧════════════════╧══════════════╧══════════╧════════════════════╝\n" );
}

void test_different_starting_points()
{
  print_section_header( "Different Starting Points - Rosenbrock" );

  std::vector<std::pair<std::string, std::array<real_type, 2>>> test_cases = { { "Easy start", { -1.0, 1.0 } },
                                                                               { "Medium start", { -1.5, 2.5 } },
                                                                               { "Hard start", { 0.0, 0.0 } },
                                                                               { "Far start", { 5.0, 5.0 } } };

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔═══════════════════════╤════════════════════════╤══════════╤════════════════╤══════════════╤══════════╗\n"
    "║ Starting Point        │ Solution               │ Iter     │ Final Value    │ ‖Step‖       │ F Evals  ║\n"
    "╠═══════════════════════╪════════════════════════╪══════════╪════════════════╪══════════════╪══════════╣\n" );

  for ( const auto & [name, start] : test_cases )
  {
    auto result = run_hj_test( fmt::format( "Rosenbrock_{}", name ), "Rosenbrock", rosenbrock, 2, start.data(), 0.5 );

    auto        symbol  = get_status_symbol( result.success, result.final_value );
    std::string sol_str = fmt::format( "({:.3f},{:.3f})", result.solution[0], result.solution[1] );

    fmt::print(
      "║ {}{:<20} │ {:<22} │ {:>8} │ {:>14.2e} │ {:>12.2e} │ {:>8} ║\n",
      symbol,
      name,
      sol_str,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚═══════════════════════╧════════════════════════╧══════════╧════════════════╧══════════════╧══════════╝\n" );
}

void test_configurations()
{
  print_section_header( "Configuration Tests - Rosenbrock" );

  real_type x0[2] = { -1.5, 2.5 };

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔═══════════════════════╤══════════════╤══════════╤══════════════╤══════════════╤═══════════╤══════════╗\n"
    "║ Configuration         │ Tolerance    │ ρ        │ Iter         │ Final Value  │  ‖Step‖   │ F Evals  ║\n"
    "╠═══════════════════════╪══════════════╪══════════╪══════════════╪══════════════╪═══════════╪══════════╣\n" );

  // Test different tolerances
  for ( real_type tol : { 1e-4, 1e-6, 1e-8 } )
  {
    auto result = run_hj_test( fmt::format( "tol_{:.0e}", tol ), "Rosenbrock", rosenbrock, 2, x0, 0.5, tol );

    auto symbol = get_status_symbol( result.success, result.final_value, tol );
    fmt::print(
      "║ {}{:<20} │ {:>12.0e} │ {:>8.2f} │ {:>12} │ {:>12.2e} │ {:>9.2e} │ {:>8} ║\n",
      symbol,
      fmt::format( "Tolerance {:.0e}", tol ),
      tol,
      result.rho,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations );
  }

  // Test different rho values
  for ( real_type rho : { 0.5, 0.7, 0.9, 0.95 } )
  {
    auto result = run_hj_test( fmt::format( "rho_{:.2f}", rho ), "Rosenbrock", rosenbrock, 2, x0, 0.5, 1e-6, rho );

    auto symbol = get_status_symbol( result.success, result.final_value );
    fmt::print(
      "║ {}{:<20} │ {:>12.0e} │ {:>8.2f} │ {:>12} │ {:>12.2e} │ {:>9.2e} │ {:>8} ║\n",
      symbol,
      fmt::format( "ρ = {:.2f}", rho ),
      1e-6,
      rho,
      result.iterations,
      result.final_value,
      result.final_step,
      result.function_evaluations );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚═══════════════════════╧══════════════╧══════════╧══════════════╧══════════════╧═══════════╧══════════╝\n" );
}

// ============================================================================
// Summary Tables
// ============================================================================

void print_function_statistics()
{
  // Calculate averages
  for ( auto & [name, stats] : function_statistics )
  {
    if ( stats.successful_tests > 0 )
    {
      stats.average_iterations =
        static_cast<real_type>( stats.total_iterations ) / static_cast<real_type>( stats.successful_tests );
      stats.success_rate =
        ( 100.0 * static_cast<double>( stats.successful_tests ) ) / static_cast<double>( stats.total_tests );
    }
  }

  print_section_header( "Function Statistics" );

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔══════════════════╤════════╤══════════╤══════════════╤════════════════╤══════════════════════════════╗\n"
    "║ Function         │ Dim    │ Tests    │ Success %    │ Avg Iter       │ Final Value Range            ║\n"
    "╠══════════════════╪════════╪══════════╪══════════════╪════════════════╪══════════════════════════════╣\n" );

  for ( const auto & [name, stats] : function_statistics )
  {
    auto success_color = stats.success_rate >= 80.0   ? fmt::fg( fmt::color::green )
                         : stats.success_rate >= 60.0 ? fmt::fg( fmt::color::yellow )
                                                      : fmt::fg( fmt::color::red );

    std::string value_range = fmt::format( "[{:.2e}, {:.2e}]", stats.min_final_value, stats.max_final_value );

    fmt::print( "║ {:<16} │ {:>6} │ {:>8} │ ", name, stats.dimension, stats.total_tests );

    fmt::print( success_color, "{:>11.1f}%", stats.success_rate );

    fmt::print( " │ {:>14.1f} │ {:<28} ║\n", stats.average_iterations, value_range );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚══════════════════╧════════╧══════════╧══════════════╧════════════════╧══════════════════════════════╝\n" );
}

void print_configuration_statistics()
{
  print_section_header( "Configuration Statistics" );

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔══════════════════════════════════╤══════════╗\n"
    "║ Configuration                    │ Count    ║\n"
    "╠══════════════════════════════════╪══════════╣\n" );

  for ( const auto & [config, count] : configuration_stats ) { fmt::print( "║ {:<32} │ {:>8} ║\n", config, count ); }

  fmt::print( fmt::fg( fmt::color::light_blue ), "╚══════════════════════════════════╧══════════╝\n" );
}

void print_performance_comparison()
{
  print_section_header( "Performance Comparison" );

  // Group by function
  std::map<std::string, std::vector<const TestResultHJ *>> grouped_by_func;
  for ( const auto & result : global_test_results ) { grouped_by_func[result.function_name].push_back( &result ); }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔═══════════════════╤════════╤═══════════════╤═════════════╤════════════════════════════════════╗\n"
    "║ Function          │ Dim    │ Avg Time (ms) │ Avg F Evals │ Best Result                        ║\n"
    "╠═══════════════════╪════════╪═══════════════╪═════════════╪════════════════════════════════════╣\n" );

  for ( const auto & [func_name, results] : grouped_by_func )
  {
    real_type            avg_time    = 0.0;
    real_type            avg_fevals  = 0.0;
    real_type            best_value  = std::numeric_limits<real_type>::max();
    const TestResultHJ * best_result = nullptr;

    int successful = 0;
    for ( const auto & result : results )
    {
      if ( result->success )
      {
        avg_time += result->time_taken.count() * 1000.0;
        avg_fevals += result->function_evaluations;
        successful++;

        if ( result->final_value < best_value )
        {
          best_value  = result->final_value;
          best_result = result;
        }
      }
    }

    if ( successful > 0 )
    {
      avg_time /= successful;
      avg_fevals /= successful;
    }

    std::string best_str = "N/A";
    if ( best_result )
    {
      best_str = fmt::format( "f={:.2e} ({} iter)", best_result->final_value, best_result->iterations );
    }

    auto time_color = avg_time < 10.0   ? fmt::fg( fmt::color::green )
                      : avg_time < 50.0 ? fmt::fg( fmt::color::yellow )
                                        : fmt::fg( fmt::color::red );

    fmt::print( "║ {:<17} │ {:>6} │ ", func_name, results[0]->dimension );

    fmt::print( time_color, "{:>13.4g}", avg_time );

    fmt::print( " │ {:>11.3g} │ {:<34} ║\n", avg_fevals, best_str );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚═══════════════════╧════════╧═══════════════╧═════════════╧════════════════════════════════════╝\n" );
}

// ============================================================================
// Main Function
// ============================================================================

int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                  Hooke-Jeeves Pattern Search - Test Suite                    ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n\n" );

  try
  {
    // Run test suites
    test_basic_functions();
    test_different_starting_points();
    test_configurations();

    // Print summary tables
    print_function_statistics();
    print_configuration_statistics();
    print_performance_comparison();

    // Final summary
    size_t total_tests      = global_test_results.size();
    size_t successful_tests = std::count_if(
      global_test_results.begin(),
      global_test_results.end(),
      []( const TestResultHJ & r ) { return r.success; } );

    fmt::print(
      fmt::fg( fmt::color::light_blue ),
      "\n"
      "╔══════════════════════════════════════════════════════════════════════════════╗\n"
      "║                             Test Suite Summary                               ║\n"
      "╠══════════════════════════════════════════════════════════════════════════════╣\n" );

    fmt::print( "║ Total tests run:            {:>48} ║\n", total_tests );
    fmt::print( "║ Successful tests:           {:>48} ║\n", successful_tests );
    fmt::print(
      "║ Success rate:               {:>47.1f}% ║\n",
      ( 100.0 * static_cast<double>( successful_tests ) ) / static_cast<double>( total_tests ) );

    size_t total_iterations  = 0;
    size_t total_evaluations = 0;
    double total_time        = 0.0;

    for ( const auto & result : global_test_results )
    {
      total_iterations += result.iterations;
      total_evaluations += result.function_evaluations;
      total_time += result.time_taken.count();
    }

    fmt::print( "║ Total iterations:           {:>48} ║\n", total_iterations );
    fmt::print( "║ Total function evaluations: {:>48} ║\n", total_evaluations );
    fmt::print( "║ Total execution time:       {:>47.3f}s ║\n", total_time );

    fmt::print(
      fmt::fg( fmt::color::light_blue ),
      "╚══════════════════════════════════════════════════════════════════════════════╝\n" );
  }
  catch ( std::exception const & e )
  {
    fmt::print( fmt::fg( fmt::color::red ), "ERROR: Test suite failed with exception: {}\n", e.what() );
    return 1;
  }

  return 0;
}
