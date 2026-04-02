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
 |  Test LBFGS - Versione Migliorata con Norma Gradiente                   |
 |  Enrico Bertolazzi - Università degli Studi di Trento                   |
\*--------------------------------------------------------------------------*/

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "Utils_minimize_LBFGS.hh"

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

using Scalar    = double;
using MINIMIZER = Utils::LBFGS_minimizer<Scalar>;
using integer   = MINIMIZER::integer;
using Vector    = typename MINIMIZER::Vector;
using Status    = MINIMIZER::Status;

// ===========================================================================
// MIGLIORAMENTO: Struttura TestResult estesa con norma gradiente
// ===========================================================================
struct TestResult
{
  string  problem_name;
  string  linesearch_name;
  Vector  final_solution;
  integer dimension;
  Status  status;
  integer total_iterations{ 0 };
  integer total_evaluations{ 0 };
  Scalar  final_function_value{ 0.0 };
  Scalar  initial_function_value{ 0.0 };
  Scalar  final_gradient_norm{ 0.0 };  // NUOVO CAMPO
};

// Struttura per statistiche delle line search
struct LineSearchStats
{
  string  name;
  integer total_tests{ 0 };
  integer successful_tests{ 0 };
  integer total_iterations{ 0 };
  integer total_evaluations{ 0 };
  Scalar  average_iterations{ 0 };
  Scalar  success_rate{ 0 };
  Scalar  avg_gradient_norm{ 0.0 };    // NUOVO: media norma gradiente
  Scalar  total_gradient_norm{ 0.0 };  // Per calcolare la media
};

// Collettore globale dei risultati
vector<TestResult>           global_test_results;
map<string, LineSearchStats> line_search_statistics;

#include "ND_func.cxx"

// ===========================================================================
// MIGLIORAMENTO: Aggiornamento statistiche con norma gradiente
// ===========================================================================
void update_line_search_statistics( const TestResult & result )
{
  auto & stats = line_search_statistics[result.linesearch_name];
  stats.name   = result.linesearch_name;
  stats.total_tests++;

  bool success = result.status == Status::CONVERGED;

  if ( success )
  {
    stats.successful_tests++;
    stats.total_iterations += result.total_iterations;
    stats.total_evaluations += result.total_evaluations;
    stats.total_gradient_norm += result.final_gradient_norm;
  }
}

// ===========================================================================
// MIGLIORAMENTO: Test runner che calcola la norma del gradiente finale
// ===========================================================================
template <typename Problem> static void test( Problem & tp, string const & problem_name )
{
  fmt::print(
    fmt::fg( fmt::color::cyan ),
    "\n"
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║ TEST FUNCTION: {:<47} ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n",
    problem_name );

  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  MINIMIZER::Options opts;
  opts.verbosity_level = 2;

  Vector x0 = tp.init();

  auto cb = [&tp]( Vector const & x, Vector * g ) -> Scalar
  {
    if ( g != nullptr ) *g = tp.gradient( x );
    return (tp) ( x );
  };

  // Lista di line search da testare
  vector<pair<
    string,
    std::function<std::optional<std::tuple<Scalar, integer>>(
      Scalar,
      Scalar,
      Vector const &,
      Vector const &,
      std::function<Scalar( Vector const &, Vector * )>,
      Scalar )>>>
    line_searches;

  // Inizializza le line search
  Utils::WeakWolfeLineSearch<Scalar>   wolfe_weak;
  Utils::StrongWolfeLineSearch<Scalar> wolfe_strong;
  Utils::ArmijoLineSearch<Scalar>      armijo;
  Utils::GoldsteinLineSearch<Scalar>   gold;
  Utils::HagerZhangLineSearch<Scalar>  HZ;
  Utils::MoreThuenteLineSearch<Scalar> More;

  line_searches.emplace_back( "Armijo", [&]( auto... a ) { return armijo( a... ); } );
  line_searches.emplace_back( "WeakWolfe", [&]( auto... a ) { return wolfe_weak( a... ); } );
  line_searches.emplace_back( "StrongWolfe", [&]( auto... a ) { return wolfe_strong( a... ); } );
  line_searches.emplace_back( "Goldstein", [&]( auto... a ) { return gold( a... ); } );
  line_searches.emplace_back( "HagerZhang", [&]( auto... a ) { return HZ( a... ); } );
  line_searches.emplace_back( "MoreThuente", [&]( auto... a ) { return More( a... ); } );

  for ( const auto & [ls_name, line_search] : line_searches )
  {
    MINIMIZER minimizer( opts );
    minimizer.set_bounds( tp.lower(), tp.upper() );

    minimizer.minimize( x0, cb, line_search );

    // ===========================================================================
    // MIGLIORAMENTO: Calcola la norma del gradiente finale
    // ===========================================================================
    Vector final_solution = minimizer.solution();

    // Salva il risultato con la norma del gradiente
    TestResult result;
    result.problem_name           = problem_name;
    result.linesearch_name        = ls_name;
    result.final_solution         = final_solution;
    result.dimension              = final_solution.size();
    result.status                 = minimizer.status();
    result.total_iterations       = minimizer.total_iterations();
    result.total_evaluations      = minimizer.total_evaluations();
    result.final_function_value   = minimizer.final_function_value();
    result.initial_function_value = minimizer.initial_function_value();
    result.final_gradient_norm    = minimizer.final_gradient_norm();

    global_test_results.push_back( result );
    update_line_search_statistics( result );

    string status_str = MINIMIZER::to_string( result.status );

    fmt::print( "{} - {}: {} after {} iterations\n", problem_name, ls_name, status_str, result.total_iterations );
    fmt::print( "   f = {:.6e}, ‖g‖ = {:.6e}\n", result.final_function_value, result.final_gradient_norm );
    fmt::print( "   Solution: {}\n\n", Utils::format_reduced_vector( result.final_solution, 10 ) );
  }
  fmt::print( "\n" );
}

// ===========================================================================
// MIGLIORAMENTO: Tabella riassuntiva raggruppata per line search
// ===========================================================================
void print_summary_table_by_linesearch()
{
  // Raggruppa per line search
  map<string, vector<TestResult const *>> grouped;

  for ( auto const & r : global_test_results ) grouped[r.linesearch_name].push_back( &r );

  for ( auto const & [ls_name, results] : grouped )
  {
    fmt::print(
      fmt::fg( fmt::color::light_blue ),
      "\n\n"
      "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
      "║  LINE SEARCH: {:<82} ║\n"
      "╠════════════════════════╤════════╤══════════╤════════════════╤═══════════════╤════════════════════╣\n"
      "║ Function               │ Dim    │ Iter     │ Final Value    │ ‖g_final‖     │ Status             ║\n"
      "╠════════════════════════╪════════╪══════════╪════════════════╪═══════════════╪════════════════════╣\n",
      ls_name );

    for ( auto const * rp : results )
    {
      auto const & r = *rp;

      string status_str = MINIMIZER::to_string( r.status );
      bool   converged  = r.status == Status::CONVERGED;

      // Colori per lo status
      auto status_color = converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );

      // Colori per la norma del gradiente
      auto grad_color = ( r.final_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                        : ( r.final_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                           : fmt::fg( fmt::color::red );

      string fname = r.problem_name;
      if ( fname.size() > 22 ) fname = fname.substr( 0, 19 ) + "...";

      fmt::print(
        "║ {:<22} │ {:>6} │ {:>8} │ {:<14.4e} │ ",
        fname,
        r.dimension,
        r.total_iterations,
        r.final_function_value );

      // Norma gradiente colorata
      fmt::print( grad_color, "{:<13.2e}", r.final_gradient_norm );
      fmt::print( " │ " );

      // Status colorato
      fmt::print( status_color, "{:<18}", status_str );
      fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
    }

    fmt::print(
      fmt::fg( fmt::color::light_blue ),
      "╚════════════════════════╧════════╧══════════╧════════════════╧═══════════════╧════════════════════╝\n" );
  }
}

// ===========================================================================
// MIGLIORAMENTO: Statistiche delle line search con norma gradiente media
// ===========================================================================
void print_line_search_statistics()
{
  // Calcola medie finali prima di stampare
  for ( auto & [name, stats] : line_search_statistics )
  {
    if ( stats.successful_tests > 0 )
    {
      stats.average_iterations =
        static_cast<Scalar>( stats.total_iterations ) / static_cast<Scalar>( stats.successful_tests );

      stats.avg_gradient_norm = stats.total_gradient_norm / static_cast<Scalar>( stats.successful_tests );

      stats.success_rate =
        ( 100.0 * static_cast<double>( stats.successful_tests ) ) / static_cast<double>( stats.total_tests );
    }
    else
    {
      stats.average_iterations = 0.0;
      stats.avg_gradient_norm  = 0.0;
      stats.success_rate       = 0.0;
    }
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔═══════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                            L-BFGS LINE SEARCH SUMMARY                                         ║\n"
    "╠═══════════════════╤══════════╤═════════════╤════════════╤══════════════╤══════════════════════╣\n"
    "║ LineSearch        │ Tests    │ Success %   │ Avg Iter   │ Avg Eval     │ Avg ‖g_final‖        ║\n"
    "╠═══════════════════╪══════════╪═════════════╪════════════╪══════════════╪══════════════════════╣\n" );

  for ( auto const & [_, s] : line_search_statistics )
  {
    auto color = ( s.success_rate >= 80.0 )   ? fmt::fg( fmt::color::green )
                 : ( s.success_rate >= 60.0 ) ? fmt::fg( fmt::color::yellow )
                                              : fmt::fg( fmt::color::red );

    auto grad_color = ( s.avg_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                      : ( s.avg_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                       : fmt::fg( fmt::color::red );

    fmt::print( "║ {:<17} │ {:>8} │ ", s.name, s.total_tests );
    fmt::print( color, "{:>10.1f}% ", s.success_rate );
    fmt::print(
      "│ {:>10.1f} │ {:>12.1f} │ ",
      s.average_iterations,
      Scalar( s.total_evaluations ) / static_cast<Scalar>( std::max<integer>( s.successful_tests, 1 ) ) );
    fmt::print( grad_color, "{:>20.2e}", s.avg_gradient_norm );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚═══════════════════╧══════════╧═════════════╧════════════╧══════════════╧══════════════════════╝\n" );
}

// ===========================================================================
// MIGLIORAMENTO: Tabella riassuntiva globale
// ===========================================================================
void print_summary_table()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                           L-BFGS GLOBAL SUMMARY                                        ║\n"
    "╠════════════════════════╤════════╤══════════════╤══════════╤════════════════╤═══════════════╤═══════════╣\n"
    "║ Function               │ Dim    │ LineSearch   │ Iter     │ Final Value    │ ‖g_final‖     │ Status    ║\n"
    "╠════════════════════════╪════════╪══════════════╪══════════╪════════════════╪═══════════════╪═══════════╣\n" );

  for ( auto const & result : global_test_results )
  {
    string status_str = MINIMIZER::to_string( result.status );
    bool   converged  = result.status == Status::CONVERGED;

    // Colori per lo status
    auto status_color = converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );

    // Colori per la norma del gradiente
    auto grad_color = ( result.final_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                      : ( result.final_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                              : fmt::fg( fmt::color::red );

    string problem_name = result.problem_name;
    if ( problem_name.length() > 22 ) { problem_name = problem_name.substr( 0, 19 ) + "..."; }

    fmt::print(
      "║ {:<22} │ {:>6} │ {:<12} │ {:>8} │ {:<14.4e} │ ",
      problem_name,
      result.dimension,
      result.linesearch_name,
      result.total_iterations,
      result.final_function_value );

    // Norma gradiente colorata
    fmt::print( grad_color, "{:<13.2e}", result.final_gradient_norm );
    fmt::print( " │ " );

    // Status colorato
    fmt::print( status_color, "{:<9}", status_str );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚════════════════════════╧════════╧══════════════╧══════════╧════════════════╧═══════════════╧═══════════╝\n" );

  // Statistiche globali
  integer total_tests     = global_test_results.size();
  integer converged_tests = std::count_if(
    global_test_results.begin(),
    global_test_results.end(),
    []( const TestResult & r ) { return r.status == Status::CONVERGED; } );

  integer accumulated_iter{ 0 };
  integer accumulated_evals{ 0 };
  Scalar  total_grad_norm{ 0.0 };
  integer grad_count{ 0 };

  for ( auto const & r : global_test_results )
  {
    if ( r.status == Status::CONVERGED )
    {
      accumulated_iter += r.total_iterations;
      accumulated_evals += r.total_evaluations;
      total_grad_norm += r.final_gradient_norm;
      grad_count++;
    }
  }

  fmt::print( fmt::fg( fmt::color::light_blue ), "\n📊 Global Statistics:\n" );
  fmt::print( "   • Total problems: {}\n", total_tests );
  fmt::print(
    "   • Converged: {} ({:.1f}%)\n",
    converged_tests,
    ( 100.0 * static_cast<double>( converged_tests ) / static_cast<double>( total_tests ) ) );
  fmt::print( "   • Total iterations: {}\n", accumulated_iter );
  fmt::print( "   • Total function evaluations: {}\n", accumulated_evals );
  if ( grad_count > 0 ) {
    fmt::print( "   • Average final ‖g‖: {:.2e}\n", total_grad_norm / static_cast<Scalar>( grad_count ) );
  }
}
// ===========================================================================
// MAIN
// ===========================================================================
int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║              L-BFGS Optimization Test Suite                    ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  for ( auto [ptr, name] : NL_list ) test( *ptr, name );

  // Stampa dei risultati
  // print_summary_table();
  print_summary_table_by_linesearch();
  print_line_search_statistics();

  return 0;
}
