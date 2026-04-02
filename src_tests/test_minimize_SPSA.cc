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

#include "Utils_minimize_SPSA.hh"

using Scalar  = double;
using integer = Eigen::Index;
using Vector  = Utils::SPSA_minimizer<Scalar>::Vector;

// Struttura per raccogliere i risultati dei test
struct TestResult
{
  std::string problem_name;
  std::string linesearch_name;
  Scalar      final_value;
  Vector      final_solution;
  integer     dimension;
  integer     iterations;
  bool        converged;
  integer     f_eval_count;
  std::string message;
};

// Statistiche line search
struct LineSearchStats
{
  std::string name;
  integer     total_tests{ 0 };
  integer     successful_tests{ 0 };
  integer     total_iterations{ 0 };
  integer     total_function_evals{ 0 };
};

// Collettore globale
std::vector<TestResult>                global_test_results;
std::map<std::string, LineSearchStats> line_search_statistics;

// -------------------------------------------------------------------
// Aggiorna statistiche
// -------------------------------------------------------------------
void update_line_search_statistics( const TestResult & result )
{
  auto & stats = line_search_statistics[result.linesearch_name];
  stats.name   = result.linesearch_name;
  stats.total_tests++;
  if ( result.converged )
  {
    stats.successful_tests++;
    stats.total_iterations += result.iterations;
  }
}

// -------------------------------------------------------------------
// Stampa statistiche
// -------------------------------------------------------------------
void print_line_search_statistics()
{
  fmt::print( "\n\n{:=^80}\n", " SPSA STATISTICS " );
  fmt::print( "{:<15} {:<8} {:<8} {:<12} {:<10}\n", "Optimizer", "Tests", "Success", "Success%", "AvgIter" );
  fmt::print( "{:-<80}\n", "" );

  for ( const auto & [name, stats] : line_search_statistics )
  {
    Scalar success_rate   = ( stats.total_tests > 0 )
                              ? 100.0 * static_cast<Scalar>( stats.successful_tests ) / static_cast<Scalar>( stats.total_tests )
                              : 0.0;
    Scalar avg_iterations = ( stats.successful_tests > 0 )
                              ? static_cast<Scalar>( stats.total_iterations ) / static_cast<Scalar>( stats.successful_tests )
                              : 0.0;
    auto   color          = ( success_rate >= 80.0 )   ? fmt::fg( fmt::color::green )
                            : ( success_rate >= 60.0 ) ? fmt::fg( fmt::color::yellow )
                                                       : fmt::fg( fmt::color::red );

    fmt::print( "{:<15} {:<8} {:<8} ", stats.name, stats.total_tests, stats.successful_tests );
    fmt::print( color, "{:<12.1f}", success_rate );
    fmt::print( " {:<10.1f}\n", avg_iterations );
  }
  fmt::print( "{:=^80}\n", "" );
}

// -------------------------------------------------------------------
// Stampa tabella riassuntiva
// -------------------------------------------------------------------
void print_summary_table()
{
  fmt::print( "\n\n{:=^80}\n", " SUMMARY TEST RESULTS " );
  fmt::print(
    "{:<28} {:<12} {:<8} {:<12} {:<15} {:<10}\n",
    "Problem",
    "Optimizer",
    "Dimension",
    "Iterations",
    "final f(x)",
    "Status" );
  fmt::print( "{:-<80}\n", "" );

  for ( auto const & result : global_test_results )
  {
    std::string  status_str = result.converged ? "CONVERGED" : "MAX_ITER";
    auto const & GREEN{ fmt::fg( fmt::color::green ) };
    auto const & RED{ fmt::fg( fmt::color::red ) };

    fmt::print(
      "{:<28} {:<12} {:<8} {:<12} {:<15.6e}",
      result.problem_name,
      result.linesearch_name,
      result.dimension,
      result.iterations,
      result.final_value );

    if ( result.converged )
      fmt::print( GREEN, "{}\n", status_str );
    else
      fmt::print( RED, "{}\n", status_str );
  }

  fmt::print( "{:=^80}\n", "" );
}

#include "ND_func.cxx"

// -------------------------------------------------------------------
// Funzione di test SPSA verbosa
// -------------------------------------------------------------------
void test( NDbase<double> * prob, std::string const & problem_name )
{
  fmt::print( "\n\nSTART: {}\n", problem_name );

  // Parametri SPSA più robusti
  Utils::SPSA_minimizer<Scalar>::Options opts;
  opts.max_iter     = 500;  // più iterazioni
  opts.a0           = 0.1;  // learning rate più piccolo
  opts.c0           = 0.5;  // perturbazioni più piccole
  opts.alpha        = 0.602;
  opts.gamma        = 0.101;
  opts.gradient_avg = 1;  // più medie per il gradiente
  opts.verbose      = true;

  Utils::SPSA_minimizer<Scalar> optimizer( opts );
  optimizer.set_bounds( prob->lower(), prob->upper() );

  Vector x0 = prob->init();

  // Run minimization
  bool ok = optimizer.minimize( x0, [&prob]( Vector const & x ) -> double { return prob->operator()( x ); } );

  // Collect results using getter methods
  TestResult result;
  result.problem_name    = problem_name;
  result.linesearch_name = "SPSA";
  result.final_value     = optimizer.final_f();
  result.final_solution  = optimizer.final_x();
  result.dimension       = static_cast<integer>( x0.size() );
  result.iterations      = optimizer.iterations();
  result.converged       = optimizer.converged();
  result.f_eval_count    = optimizer.f_eval_count();
  result.message         = optimizer.message();

  global_test_results.push_back( result );
  update_line_search_statistics( result );

  fmt::print(
    "{}: final f = {:.6e}, iterations = {}, ok = {}\n{}\n\n\n",
    problem_name,
    optimizer.final_f(),
    optimizer.iterations(),
    optimizer.final_x().transpose(),
    ok );
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main()
{
  fmt::print( "Esecuzione test SPSA_minimizer...\n" );

  for ( auto [ptr, name] : NL_list ) test( ptr.get(), name );

  print_summary_table();
  print_line_search_statistics();

  return 0;
}
