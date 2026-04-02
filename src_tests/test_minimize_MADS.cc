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

/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  MADS test suite per problemi di ottimizzazione senza gradienti          |
 |                                                                          |
 |  Adattamento migliorato con colori, Unicode e formattazione avanzata     |
\*--------------------------------------------------------------------------*/

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Utils_minimize_MADS.hh"

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif

using Utils::MADS_minimizer;
using integer = Eigen::Index;
using Scalar  = double;
using Vector  = MADS_minimizer<Scalar>::Vector;
using std::string;

// Struttura per raccogliere i risultati dei test
struct TestResult
{
  string  test_name;
  Scalar  initial_f;
  Scalar  final_f;
  Scalar  error;
  integer iterations;
  integer f_eval_count;
  integer dimension;
  bool    converged;
  string  message;
  Vector  final_solution;
  Scalar  mesh_size;
};

// Statistiche line search
struct LineSearchStats
{
  std::string name;
  integer     total_tests{ 0 };
  integer     successful_tests{ 0 };
  integer     total_iterations{ 0 };
  integer     total_function_evals{ 0 };
  Scalar      avg_iterations{ 0.0 };
  Scalar      success_rate{ 0.0 };
};

// Collettore globale
std::vector<TestResult>                global_test_results;
std::map<std::string, LineSearchStats> line_search_statistics;

// -------------------------------------------------------------------
// Aggiorna statistiche
// -------------------------------------------------------------------
void update_line_search_statistics( const TestResult & result )
{
  auto & stats = line_search_statistics["MADS"];
  stats.name   = "MADS";
  stats.total_tests++;
  if ( result.converged )
  {
    stats.successful_tests++;
    stats.total_iterations += result.iterations;
    stats.total_function_evals += result.f_eval_count;
  }

  // Calcola statistiche aggregate
  if ( stats.successful_tests > 0 )
  {
    stats.avg_iterations =
      static_cast<Scalar>( stats.total_iterations ) / static_cast<Scalar>( stats.successful_tests );
    stats.success_rate =
      100.0 * static_cast<double>( stats.successful_tests ) / static_cast<double>( stats.total_tests );
  }
}

// -------------------------------------------------------------------
// Stampa tabella riassuntiva con colori e Unicode
// -------------------------------------------------------------------
void print_summary_table()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                         MADS PATTERN SEARCH SUMMARY RESULTS                                    ║\n"
    "╠════════════════════════╤════════╤══════════╤══════════╤════════════════╤═══════════════╤═══════════════════════╣\n"
    "║ Function               │ Dim    │ Iter     │ Eval     │ f(x) Initial   │ f(x) Final    │ Status                ║\n"
    "╠════════════════════════╪════════╪══════════╪══════════╪════════════════╪═══════════════╪═══════════════════════╣"
    "\n" );

  for ( auto const & result : global_test_results )
  {
    string status_str = result.converged ? "✓ CONVERGED" : "✗ MAX_ITER";
    bool   converged  = result.converged;

    auto status_color = converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::yellow );

    auto final_f_color = ( std::abs( result.final_f - result.initial_f ) / ( std::abs( result.initial_f ) + 1e-10 ) >
                           0.1 )
                           ? fmt::fg( fmt::color::green )
                           : fmt::fg( fmt::color::yellow );

    string problem_name = result.test_name;
    if ( problem_name.length() > 22 ) { problem_name = problem_name.substr( 0, 19 ) + "..."; }

    fmt::print(
      "║ {:<22} │ {:>6} │ {:>8} │ {:>8} │ ",
      problem_name,
      result.dimension,
      result.iterations,
      result.f_eval_count );

    fmt::print( fmt::fg( fmt::color::light_gray ), "{:<14.4e} │ ", result.initial_f );

    fmt::print( final_f_color, "{:<13.4e}", result.final_f );
    fmt::print( " │ " );

    fmt::print( status_color, "{:<21}", status_str );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚════════════════════════╧════════╧══════════╧══════════╧════════════════╧═══════════════╧═══════════════════════╝"
    "\n" );

  // Statistiche globali
  integer total_tests     = global_test_results.size();
  integer converged_tests = std::count_if(
    global_test_results.begin(),
    global_test_results.end(),
    []( const TestResult & r ) { return r.converged; } );

  Scalar success_rate =
    100.0 * static_cast<Scalar>( converged_tests ) / static_cast<Scalar>( std::max<integer>( total_tests, 1 ) );

  integer total_iterations = 0;
  integer total_evals      = 0;
  Scalar  avg_mesh_size    = 0.0;

  for ( auto const & r : global_test_results )
  {
    if ( r.converged )
    {
      total_iterations += r.iterations;
      total_evals += r.f_eval_count;
    }
    avg_mesh_size += r.mesh_size;
  }

  if ( converged_tests > 0 ) { avg_mesh_size /= static_cast<Scalar>( converged_tests ); }

  fmt::print( fmt::fg( fmt::color::light_blue ), "\n📊 Global Statistics:\n" );
  fmt::print( fmt::fg( fmt::color::light_gray ), "   • Total problems: {}\n", total_tests );
  fmt::print( fmt::fg( fmt::color::light_gray ), "   • Converged: {} ({:.1f}%)\n", converged_tests, success_rate );
  fmt::print( fmt::fg( fmt::color::light_gray ), "   • Total iterations: {}\n", total_iterations );
  fmt::print( fmt::fg( fmt::color::light_gray ), "   • Total function evaluations: {}\n", total_evals );
  fmt::print( fmt::fg( fmt::color::light_gray ), "   • Average final mesh size: {:.2e}\n", avg_mesh_size );

  // Calcola riduzione media della funzione obiettivo
  Scalar  avg_reduction = 0.0;
  integer count         = 0;
  for ( auto const & r : global_test_results )
  {
    if ( r.converged && std::abs( r.initial_f ) > 1e-10 )
    {
      avg_reduction += std::abs( ( r.final_f - r.initial_f ) / r.initial_f );
      count++;
    }
  }
  if ( count > 0 )
  {
    avg_reduction = 100.0 * avg_reduction / static_cast<Scalar>( count );
    fmt::print( fmt::fg( fmt::color::light_gray ), "   • Average relative reduction: {:.1f}%\n", avg_reduction );
  }
}

// -------------------------------------------------------------------
// Stampa statistiche per algoritmo
// -------------------------------------------------------------------
void print_line_search_statistics()
{
  fmt::print( fmt::fg( fmt::color::light_blue ), "\n\n{:=^80}\n", " MADS STATISTICS BY ALGORITHM " );

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "{:<15} {:<8} {:<8} {:<12} {:<10} {:<12}\n",
    "Optimizer",
    "Tests",
    "Success",
    "Success%",
    "AvgIter",
    "AvgEvals" );

  fmt::print( fmt::fg( fmt::color::light_blue ), "{:-<80}\n", "" );

  for ( const auto & [name, stats] : line_search_statistics )
  {
    Scalar avg_evals = ( stats.successful_tests > 0 )
                         ? static_cast<Scalar>( stats.total_function_evals ) / static_cast<Scalar>( stats.successful_tests )
                         : 0.0;

    auto success_color = stats.success_rate > 80.0   ? fmt::fg( fmt::color::green )
                         : stats.success_rate > 50.0 ? fmt::fg( fmt::color::yellow )
                                                     : fmt::fg( fmt::color::red );

    fmt::print(
      fmt::fg( fmt::color::light_gray ),
      "{:<15} {:<8} {:<8} ",
      stats.name,
      stats.total_tests,
      stats.successful_tests );

    fmt::print( success_color, "{:<12.1f}", stats.success_rate );

    fmt::print( fmt::fg( fmt::color::light_gray ), " {:<10.1f} {:<12.1f}\n", stats.avg_iterations, avg_evals );
  }

  fmt::print( fmt::fg( fmt::color::light_blue ), "{:=^80}\n", "" );
}

#include "ND_func.cxx"

// -------------------------------------------------------------------
// Funzione di test MADS migliorata
// -------------------------------------------------------------------
void test( NDbase<Scalar> * prob, std::string const & problem_name )
{
  fmt::print( fmt::fg( fmt::color::light_blue ), "\n{:━^{}}\n", "", 60 );

  fmt::print( fmt::fg( fmt::color::cyan ), "🧪 Testing: {}\n", problem_name );

  // Parametri MADS
  Utils::MADS_minimizer<Scalar>::Options opts;
  opts.max_iter          = 500;
  opts.verbose           = false;  // Disabilitato per output più pulito
  opts.print_every       = 100;
  opts.max_evaluations   = 10000;
  opts.initial_mesh_size = 1.0;
  opts.min_mesh_size     = 1e-8;
  opts.tol_mesh          = 1e-8;
  opts.success_threshold = 0.01;
  opts.patience          = 20;

  Utils::MADS_minimizer<Scalar> optimizer( opts );
  optimizer.set_bounds( prob->lower(), prob->upper() );

  Vector x0        = prob->init();
  Scalar initial_f = prob->operator()( x0 );

  fmt::print(
    fmt::fg( fmt::color::light_gray ),
    "   • Dimension: {}\n"
    "   • Initial point: {}\n"
    "   • Initial f(x): {:.6e}\n",
    x0.size(),
    Utils::format_reduced_vector( x0 ),
    initial_f );

  // Esegui l'ottimizzazione
  optimizer.minimize( x0, [&prob]( Vector const & x ) { return prob->operator()( x ); } );

  // Ottieni i risultati tramite i metodi della classe
  TestResult result;
  result.test_name      = problem_name;
  result.initial_f      = initial_f;
  result.final_f        = optimizer.get_final_f();
  result.error          = 0.0;
  result.iterations     = optimizer.get_iterations();
  result.f_eval_count   = optimizer.get_f_eval_count();
  result.dimension      = static_cast<integer>( x0.size() );
  result.converged      = optimizer.get_converged();
  result.message        = optimizer.get_message();
  result.final_solution = optimizer.get_final_x();
  result.mesh_size      = optimizer.get_mesh_size();

  global_test_results.push_back( result );
  update_line_search_statistics( result );

  // Stampa risultati colorati
  auto improvement_color = ( result.final_f < result.initial_f - 1e-6 ) ? fmt::fg( fmt::color::green )
                                                                        : fmt::fg( fmt::color::yellow );

  auto status_color = result.converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::yellow );

  fmt::print(
    fmt::fg( fmt::color::light_gray ),
    "   • Iterations: {}\n"
    "   • Function evaluations: {}\n"
    "   • Final mesh size: {:.2e}\n",
    result.iterations,
    result.f_eval_count,
    result.mesh_size );

  fmt::print( improvement_color, "   • Final f(x): {:.6e} ", result.final_f );

  if ( result.final_f < result.initial_f )
  {
    Scalar improvement = 100.0 * ( result.initial_f - result.final_f ) / std::abs( result.initial_f );
    fmt::print( improvement_color, "(improvement: {:.1f}%)\n", improvement );
  }
  else
  {
    fmt::print( "\n" );
  }

  if ( x0.size() <= 10 )
  {
    fmt::print(
      fmt::fg( fmt::color::light_gray ),
      "   • Final solution: {}\n",
      Utils::format_reduced_vector( result.final_solution ) );
  }

  fmt::print( status_color, "   • Status: {}", result.message );

  if ( result.converged ) { fmt::print( fmt::fg( fmt::color::green ), " ✓\n" ); }
  else
  {
    fmt::print( fmt::fg( fmt::color::yellow ), " ⚠\n" );
  }
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║              MADS PATTERN SEARCH Test Suite                    ║\n"
    "║           (Mesh Adaptive Direct Search)                        ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  fmt::print(
    fmt::fg( fmt::color::light_gray ),
    "🔍 Testing derivative-free optimization on {} problems...\n\n",
    NL_list.size() );

  for ( auto [ptr, name] : NL_list ) test( ptr.get(), name );

  print_summary_table();
  print_line_search_statistics();

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n"
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║                   Test Suite Completed                         ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n" );

  return 0;
}
