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
// File: test_NewtonDumped.cc
// -----------------------------------------------------------------------------

/*--------------------------------------------------------------------------*\
 |  Driver program for testing Newton's method with various damping          |
 |  strategies on all nonlinear system test problems.                        |
\*--------------------------------------------------------------------------*/

#include "Utils_CLI11.hh"  // CLI11 per il parsing degli argomenti
#include "Utils_fmt.hh"
#include "Utils_nonlinear_NewtonDumped.hh"
#include "Utils_nonlinear_system.hh"
#include "Utils_TicToc.hh"
#include "Utils_progress_bar.hh"

using namespace Utils;
using namespace std;

// Parametri configurabili (spostato fuori da main per accesso globale)
struct Config
{
  double tolerance              = 1e-8;
  int    max_iterations         = 100;
  int    max_damping_iterations = 20;
  bool   verbose                = false;
  bool   quiet                  = false;
  bool   test_all               = false;  // NUOVO: flag per testare tutte le strategie
  string output_file            = "";
  string strategy               = "deuflhard";  // Default

  // Deuflhard parameters
  double min_lambda = 1e-6;

  // L2 parameters
  double mu_init            = 0.01;
  double mu_min             = 1e-8;
  double mu_max             = 1e4;
  double mu_increase_factor = 10.0;
  double mu_decrease_factor = 0.1;

  // Adaptive L2 parameters
  double trust_region_radius   = 1.0;
  double trust_region_min      = 1e-6;
  double trust_region_max      = 100.0;
  double acceptance_ratio_good = 0.75;
  double acceptance_ratio_bad  = 0.25;

  // Dogleg parameters
  double dogleg_delta     = 1.0;
  double dogleg_delta_min = 1e-6;
  double dogleg_delta_max = 100.0;
  double dogleg_eta1      = 0.1;
  double dogleg_eta2      = 0.75;
  double dogleg_gamma1    = 0.5;
  double dogleg_gamma2    = 2.0;

  // Wolfe parameters
  double wolfe_c1         = 1e-4;
  double wolfe_c2         = 0.9;
  double wolfe_alpha_init = 1.0;
  double wolfe_alpha_min  = 1e-10;
  double wolfe_alpha_max  = 1e10;
  double wolfe_rho        = 0.5;

  // Cubic regularization parameters
  double cubic_sigma          = 0.1;
  double cubic_sigma_min      = 1e-8;
  double cubic_sigma_max      = 1e8;
  double cubic_gamma_decrease = 0.5;
  double cubic_gamma_increase = 2.0;
  double cubic_eta1           = 0.1;
  double cubic_eta2           = 0.75;

  // Quadratic backtracking parameters
  double quadratic_c1         = 1e-4;
  double quadratic_c2         = 0.5;
  int    quadratic_max_interp = 5;

  // Bank & Rose parameters (NUOVE)
  double bank_rose_alpha     = 0.5;
  double bank_rose_beta      = 0.1;
  double bank_rose_gamma     = 0.9;
  double bank_rose_theta_min = 1e-4;
  double bank_rose_theta_max = 1.0;

  // Griewank parameters (NUOVE)
  double griewank_eta   = 0.1;
  double griewank_omega = 0.5;
  double griewank_tau   = 1e-4;
  double griewank_zeta  = 0.9;

  // Filter method parameters (NUOVE)
  double filter_theta_min   = 1e-6;
  double filter_gamma_theta = 0.01;
  double filter_gamma_f     = 0.01;
  double filter_alpha       = 0.5;
  double filter_beta        = 0.8;

  // Cubic trust region parameters (NUOVE)
  double ctr_delta     = 1.0;
  double ctr_delta_min = 1e-6;
  double ctr_delta_max = 100.0;
  double ctr_eta1      = 0.1;
  double ctr_eta2      = 0.75;
  double ctr_gamma1    = 0.5;
  double ctr_gamma2    = 2.0;
  double ctr_sigma     = 0.1;

  // Test selection
  vector<int> test_indices;
  string      test_name_filter         = "";
  int         max_tests                = 0;  // 0 = tutti
  bool        run_single_initial_point = false;
  int         initial_point_index      = 0;
};

// Struttura per memorizzare i risultati di un test
struct TestResult
{
  string test_name;
  int    num_equations;
  bool   converged;
  int    iterations;
  int    function_evals;
  int    jacobian_evals;
  double final_residual;
  double elapsed_time_ms;
  int    initial_point_index;

  TestResult()
    : num_equations( 0 )
    , converged( false )
    , iterations( 0 )
    , function_evals( 0 )
    , jacobian_evals( 0 )
    , final_residual( 0.0 )
    , elapsed_time_ms( 0.0 )
    , initial_point_index( -1 )
  {
  }
};

// Struttura per statistiche aggregate per strategia
struct StrategyStatistics
{
  string             strategy_name;
  int                total_tests;
  int                converged_tests;
  int                failed_tests;
  double             success_rate;
  double             avg_iterations;
  double             avg_function_evals;
  double             avg_jacobian_evals;
  double             avg_time_ms;
  double             total_time_ms;
  double             max_time_ms;
  double             min_time_ms;
  vector<TestResult> detailed_results;

  StrategyStatistics()
    : total_tests( 0 )
    , converged_tests( 0 )
    , failed_tests( 0 )
    , success_rate( 0.0 )
    , avg_iterations( 0.0 )
    , avg_function_evals( 0.0 )
    , avg_jacobian_evals( 0.0 )
    , avg_time_ms( 0.0 )
    , total_time_ms( 0.0 )
    , max_time_ms( 0.0 )
    , min_time_ms( numeric_limits<double>::max() )
  {
  }
};

// Struttura per statistiche generali
struct OverallStatistics
{
  int    total_tests;
  int    converged_tests;
  int    failed_tests;
  double success_rate;
  double avg_iterations;
  double avg_function_evals;
  double avg_jacobian_evals;
  double avg_time_ms;
  double total_time_ms;
  double max_time_ms;
  double min_time_ms;

  OverallStatistics()
    : total_tests( 0 )
    , converged_tests( 0 )
    , failed_tests( 0 )
    , success_rate( 0.0 )
    , avg_iterations( 0.0 )
    , avg_function_evals( 0.0 )
    , avg_jacobian_evals( 0.0 )
    , avg_time_ms( 0.0 )
    , total_time_ms( 0.0 )
    , max_time_ms( 0.0 )
    , min_time_ms( numeric_limits<double>::max() )
  {
  }
};

// Mappa per la strategia di damping - TUTTE LE STRATEGIE (AGGIORNATA)
static const map<string, NewtonDumped::DampingStrategy> damping_strategy_map = {
  { "deuflhard", NewtonDumped::DEUFLHARD },
  { "l2_classic", NewtonDumped::L2_CLASSIC },
  { "l2_adaptive", NewtonDumped::L2_ADAPTIVE },
  { "l2_hybrid", NewtonDumped::L2_HYBRID },
  { "dogleg", NewtonDumped::DOGLEG },
  { "wolfe", NewtonDumped::WOLFE_LINE_SEARCH },
  { "cubic", NewtonDumped::CUBIC_REGULARIZATION },
  { "quadratic", NewtonDumped::BACKTRACKING_QUADRATIC },
  { "bank_rose", NewtonDumped::BANK_ROSE },
  { "griewank", NewtonDumped::GRIEWANK },
  { "filter", NewtonDumped::FILTER_METHODS },
  { "cubic_trust_region", NewtonDumped::CUBIC_TRUST_REGION }
};

// Helper function to join vector of ints into string
string join_ints( const vector<int> & vec, const string & delimiter = ", " )
{
  ostringstream oss;
  for ( size_t i = 0; i < vec.size(); ++i )
  {
    if ( i > 0 ) oss << delimiter;
    oss << vec[i];
  }
  return oss.str();
}

// Funzione per troncare una stringa se troppo lunga
string truncate_string( string const & str, size_t max_length )
{
  if ( str.length() <= max_length ) return str;
  return str.substr( 0, max_length - 3 ) + "...";
}

// Funzione per stampare una barra di progresso
void print_progress( int current, int total )
{
  double progress = static_cast<double>( current ) / static_cast<double>( total );
  Utils::progress_bar( std::cout, progress, 50, "Progress:" );
}

// Funzione per stampare la tabella riassuntiva usando fmt con allineamento
// perfetto
void print_summary_table( const vector<TestResult> & results )
{
  // Dimensioni delle colonne
  constexpr int col_idx      = 5;   // # (indice)
  constexpr int col_status   = 8;   // Status (incluso spazio per i colori)
  constexpr int col_neq      = 4;   // NEQ
  constexpr int col_iter     = 5;   // Iter
  constexpr int col_feval    = 5;   // F-Eval
  constexpr int col_jeval    = 5;   // J-Eval
  constexpr int col_residual = 10;  // Residual
  constexpr int col_time     = 10;  // Time(ms)
  constexpr int col_name     = 35;  // Test Name

  // Calcola la larghezza totale della tabella
  constexpr int total_width = 2 + col_idx + 3 + col_status + 3 + col_neq + 3 + col_iter + 3 + col_feval + 3 +
                              col_jeval + 3 + col_residual + 3 + col_time + 3 + col_name + 2;

  // Intestazione della tabella
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n\n"
    "╔{:═^{}}╗\n",
    " NEWTON METHOD TEST RESULTS ",
    total_width - 2 );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "╠{:═^{}}╣\n", "", total_width - 2 );

  // Intestazione delle colonne
  fmt::print( fg( fmt::color::cyan ), "║ " );
  fmt::print( "{:>{}} ║ ", "#", col_idx );
  fmt::print( "{:^{}} ║ ", "Status", col_status );
  fmt::print( "{:>{}} ║ ", "NEQ", col_neq );
  fmt::print( "{:>{}} ║ ", "Iter", col_iter );
  fmt::print( "{:>{}} ║ ", "F-Eval", col_feval );
  fmt::print( "{:>{}} ║ ", "J-Eval", col_jeval );
  fmt::print( "{:>{}} ║ ", "Residual", col_residual );
  fmt::print( "{:>{}} ║ ", "Time(ms)", col_time );
  fmt::print( "{:<{}} ", "Test Name", col_name );
  fmt::print( fg( fmt::color::cyan ), "║\n" );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "╠{:═^{}}╣\n", "", total_width - 2 );

  // Dati
  for ( size_t i = 0; i < results.size(); ++i )
  {
    auto const & r = results[i];

    // Colonna indice
    fmt::print( fg( fmt::color::cyan ), "║ {:>{}} ║ ", i + 1, col_idx );

    // Colonna status
    if ( r.converged ) { fmt::print( fg( fmt::color::green ), "{:^{}}", "OK", col_status ); }
    else
    {
      fmt::print( fg( fmt::color::red ), "{:^{}}", "FAIL", col_status );
    }
    fmt::print( fg( fmt::color::cyan ), " ║ " );

    // Colonna NEQ
    fmt::print( "{:>{}} ║ ", r.num_equations, col_neq );

    // Colonna Iter
    fmt::print( "{:>{}} ║ ", r.iterations, col_iter );

    // Colonna F-Eval
    fmt::print( "{:>{}} ║ ", r.function_evals, col_feval );

    // Colonna J-Eval
    fmt::print( "{:>{}} ║ ", r.jacobian_evals, col_jeval );

    // Colonna Residual
    if ( r.final_residual == 0.0 ) { fmt::print( "{:>12} ║ ", "0.00e+00" ); }
    else
    {
      fmt::print( "{:>12.2e} ║ ", r.final_residual );
    }

    // Colonna Time
    if ( r.elapsed_time_ms < 0.01 ) { fmt::print( "{:>10.3f} ║ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 1.0 ) { fmt::print( "{:>10.2f} ║ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 10.0 ) { fmt::print( "{:>10.2f} ║ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 100.0 ) { fmt::print( "{:>10.2f} ║ ", r.elapsed_time_ms ); }
    else
    {
      fmt::print( "{:>10.0f} ║ ", r.elapsed_time_ms );
    }

    // Colonna Test Name
    string test_name = truncate_string( r.test_name, col_name );
    fmt::print( "{:<{}} ", test_name, col_name );

    fmt::print( fg( fmt::color::cyan ), "║\n" );
  }

  // Linea finale
  fmt::print( fg( fmt::color::cyan ), "╚{:═^{}}╝\n", "═", total_width - 2 );
}

// Funzione per calcolare e stampare le statistiche con allineamento perfetto
void print_statistics( const vector<TestResult> & results )
{
  OverallStatistics stats;
  stats.total_tests = static_cast<int>( results.size() );

  double total_iterations     = 0.0;
  double total_function_evals = 0.0;
  double total_jacobian_evals = 0.0;
  double total_time           = 0.0;

  for ( auto const & r : results )
  {
    if ( r.converged )
    {
      stats.converged_tests++;
      total_iterations += r.iterations;
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
    stats.avg_iterations     = total_iterations / stats.converged_tests;
    stats.avg_function_evals = total_function_evals / stats.converged_tests;
    stats.avg_jacobian_evals = total_jacobian_evals / stats.converged_tests;
    stats.avg_time_ms        = total_time / stats.converged_tests;
    stats.total_time_ms      = total_time;
  }

  // Dimensioni per la tabella delle statistiche
  constexpr int stat_col_label   = 25;
  constexpr int stat_col_value   = 12;
  constexpr int stat_total_width = stat_col_label + stat_col_value + 4;  // +6 per bordi e spazi

  // Stampa delle statistiche
  fmt::print( "\n" );
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "╭{:─^{}}╮\n",
    " STATISTICAL SUMMARY ",
    stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "├{:─^{}}┤\n", "", stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "│" );
  fmt::print( "{:^{}}", "", stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "│\n" );

  // Total Tests
  fmt::print( fg( fmt::color::cyan ), "│ " );
  fmt::print( "{:<{}}", "Total Tests:", stat_col_label );
  fmt::print( fg( fmt::color::white ), "{:>{}}", stats.total_tests, stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " │\n" );

  // Converged Tests
  fmt::print( fg( fmt::color::cyan ), "│ " );
  fmt::print( "{:<{}}", "Converged Tests:", stat_col_label );
  fmt::print(
    fg( fmt::color::green ),
    "{:>{}}",
    fmt::format( "{} ({:.1f}%)", stats.converged_tests, stats.success_rate ),
    stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " │\n" );

  // Failed Tests
  fmt::print( fg( fmt::color::cyan ), "│ " );
  fmt::print( "{:<{}}", "Failed Tests:", stat_col_label );
  fmt::print(
    fg( fmt::color::red ),
    "{:>{}}",
    fmt::format( "{} ({:.1f}%)", stats.failed_tests, 100.0 - stats.success_rate ),
    stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " │\n" );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "├{:─^{}}┤\n", "", stat_total_width - 2 );

  if ( stats.converged_tests > 0 )
  {
    // Average Iterations
    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Average Iterations:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_iterations, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    // Average Function Evals
    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Average Function Evals:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_function_evals, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    // Average Jacobian Evals
    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Average Jacobian Evals:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_jacobian_evals, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    // Average Time
    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Average Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    // Min Time
    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Min Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.min_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );

    // Max Time
    fmt::print( fg( fmt::color::cyan ), "│ " );
    fmt::print( "{:<{}}", "Max Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.max_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " │\n" );
  }

  fmt::print( fg( fmt::color::cyan ), "│{:^{}}│\n", "", stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "╰{:─^{}}╯\n", "", stat_total_width - 2 );
}

// Funzione per stampare la tabella comparativa delle strategie
void print_strategy_comparison_table( const vector<StrategyStatistics> & strategy_stats )
{
  // Dimensioni delle colonne
  constexpr int col_strategy  = 25;  // Nome strategia
  constexpr int col_tests     = 10;  // Test
  constexpr int col_converged = 12;  // Convergiti
  constexpr int col_success   = 12;  // % Successo
  constexpr int col_iter      = 10;  // Iterazioni medie
  constexpr int col_feval     = 12;  // F-Eval medie
  constexpr int col_jeval     = 12;  // J-Eval medie
  constexpr int col_time      = 12;  // Tempo medio (ms)
  constexpr int col_rank      = 8;   // Rank

  // Calcola la larghezza totale
  constexpr int total_width = 2 + col_strategy + 3 + col_tests + 3 + col_converged + 3 + col_success + 3 + col_iter +
                              3 + col_feval + 3 + col_jeval + 3 + col_time + 3 + col_rank + 2;

  // Intestazione della tabella
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n\n"
    "╔{:═^{}}╗\n",
    " STRATEGY COMPARISON ",
    total_width - 2 );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "╠{:═^{}}╣\n", "", total_width - 2 );

  // Intestazione delle colonne
  fmt::print( fg( fmt::color::cyan ), "║ " );
  fmt::print( "{:<{}} ║ ", "Strategy", col_strategy );
  fmt::print( "{:>{}} ║ ", "# Tests", col_tests );
  fmt::print( "{:>{}} ║ ", "Converged", col_converged );
  fmt::print( "{:>{}} ║ ", "Success %", col_success );
  fmt::print( "{:>{}} ║ ", "Avg Iter", col_iter );
  fmt::print( "{:>{}} ║ ", "Avg F-Eval", col_feval );
  fmt::print( "{:>{}} ║ ", "Avg J-Eval", col_jeval );
  fmt::print( "{:>{}} ║ ", "Avg Time(ms)", col_time );
  fmt::print( "{:>{}} ", "Rank", col_rank );
  fmt::print( fg( fmt::color::cyan ), "║\n" );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "╠{:═^{}}╣\n", "", total_width - 2 );

  // Calcola il ranking delle strategie basato su success rate e tempo
  vector<pair<double, size_t>> rankings;  // (score, index)
  for ( size_t i = 0; i < strategy_stats.size(); ++i )
  {
    const auto & stats = strategy_stats[i];
    // Score: success rate (peso 0.7) + inverso del tempo normalizzato (peso
    // 0.3)
    double time_score = 0.0;
    if ( stats.avg_time_ms > 0 )
    {
      // Trova il tempo massimo per normalizzare
      double max_time = 0.0;
      for ( const auto & s : strategy_stats )
      {
        if ( s.avg_time_ms > max_time ) max_time = s.avg_time_ms;
      }
      time_score = 1.0 - ( stats.avg_time_ms / max_time );
    }
    double score = 0.7 * ( stats.success_rate / 100.0 ) + 0.3 * time_score;
    rankings.emplace_back( score, i );
  }

  // Ordina per score decrescente
  sort(
    rankings.begin(),
    rankings.end(),
    []( const pair<double, size_t> & a, const pair<double, size_t> & b ) { return a.first > b.first; } );

  // Assegna rank
  vector<int> ranks( strategy_stats.size(), 0 );
  for ( size_t i = 0; i < rankings.size(); ++i ) { ranks[rankings[i].second] = static_cast<int>( i + 1 ); }

  // Dati delle strategie
  for ( size_t i = 0; i < strategy_stats.size(); ++i )
  {
    const auto & stats = strategy_stats[i];
    int          rank  = ranks[i];

    fmt::print( fg( fmt::color::cyan ), "║ " );

    // Colonna Strategy
    string strategy_name = truncate_string( stats.strategy_name, col_strategy );
    fmt::print( "{:<{}} ║ ", strategy_name, col_strategy );

    // Colonna # Tests
    fmt::print( "{:>{}} ║ ", stats.total_tests, col_tests );

    // Colonna Converged
    fmt::print( "{:>{}} ║ ", stats.converged_tests, col_converged );

    // Colonna Success %
    fmt::color success_color;
    if ( stats.success_rate >= 90.0 ) { success_color = fmt::color::green; }
    else if ( stats.success_rate >= 70.0 ) { success_color = fmt::color::yellow; }
    else
    {
      success_color = fmt::color::red;
    }
    fmt::print( fg( success_color ), "{:>{}.1f}", stats.success_rate, col_success );
    fmt::print( fg( fmt::color::cyan ), " ║ " );

    // Colonna Avg Iter
    if ( stats.converged_tests > 0 ) { fmt::print( "{:>{}.1f} ║ ", stats.avg_iterations, col_iter ); }
    else
    {
      fmt::print( "{:>{}} ║ ", "N/A", col_iter );
    }

    // Colonna Avg F-Eval
    if ( stats.converged_tests > 0 ) { fmt::print( "{:>{}.1f} ║ ", stats.avg_function_evals, col_feval ); }
    else
    {
      fmt::print( "{:>{}} ║ ", "N/A", col_feval );
    }

    // Colonna Avg J-Eval
    if ( stats.converged_tests > 0 ) { fmt::print( "{:>{}.1f} ║ ", stats.avg_jacobian_evals, col_jeval ); }
    else
    {
      fmt::print( "{:>{}} ║ ", "N/A", col_jeval );
    }

    // Colonna Avg Time
    if ( stats.converged_tests > 0 )
    {
      fmt::color time_color;
      if ( stats.avg_time_ms < 10.0 ) { time_color = fmt::color::green; }
      else if ( stats.avg_time_ms < 100.0 ) { time_color = fmt::color::yellow; }
      else
      {
        time_color = fmt::color::red;
      }
      fmt::print( fg( time_color ), "{:>{}.1f}", stats.avg_time_ms, col_time );
      fmt::print( fg( fmt::color::cyan ), " ║ " );
    }
    else
    {
      fmt::print( "{:>{}} ║ ", "N/A", col_time );
    }

    // Colonna Rank
    fmt::color rank_color;
    if ( rank == 1 ) { rank_color = fmt::color::green; }
    else if ( rank <= 3 ) { rank_color = fmt::color::yellow; }
    else if ( rank <= 6 ) { rank_color = fmt::color::light_blue; }
    else
    {
      rank_color = fmt::color::white;
    }
    fmt::print( fg( rank_color ), "{:>{}}", rank, col_rank );

    fmt::print( fg( fmt::color::cyan ), " ║\n" );
  }

  // Linea finale
  fmt::print( fg( fmt::color::cyan ), "╚{:═^{}}╝\n", "═", total_width - 2 );

  // Statistiche aggregate
  OverallStatistics overall;
  for ( const auto & stats : strategy_stats )
  {
    overall.total_tests += stats.total_tests;
    overall.converged_tests += stats.converged_tests;
    overall.failed_tests += stats.failed_tests;
    overall.total_time_ms += stats.total_time_ms;

    if ( stats.max_time_ms > overall.max_time_ms ) overall.max_time_ms = stats.max_time_ms;
    if ( stats.min_time_ms < overall.min_time_ms ) overall.min_time_ms = stats.min_time_ms;
  }

  overall.success_rate = 100.0 * overall.converged_tests / overall.total_tests;

  if ( overall.converged_tests > 0 )
  {
    // Calcola medie ponderate
    double total_weighted_iter  = 0.0;
    double total_weighted_feval = 0.0;
    double total_weighted_jeval = 0.0;

    for ( const auto & stats : strategy_stats )
    {
      if ( stats.converged_tests > 0 )
      {
        total_weighted_iter += stats.avg_iterations * stats.converged_tests;
        total_weighted_feval += stats.avg_function_evals * stats.converged_tests;
        total_weighted_jeval += stats.avg_jacobian_evals * stats.converged_tests;
      }
    }

    overall.avg_iterations     = total_weighted_iter / overall.converged_tests;
    overall.avg_function_evals = total_weighted_feval / overall.converged_tests;
    overall.avg_jacobian_evals = total_weighted_jeval / overall.converged_tests;
    overall.avg_time_ms        = overall.total_time_ms / overall.converged_tests;
  }

  // Stampa statistiche aggregate
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n╭{:─^{}}╮\n", " OVERALL SUMMARY ", 50 );
  fmt::print( fg( fmt::color::cyan ), "│" );
  fmt::print( fg( fmt::color::white ), "{:^50}", fmt::format( "Total Strategies: {}", strategy_stats.size() ) );
  fmt::print( fg( fmt::color::cyan ), "│\n" );

  fmt::print( fg( fmt::color::cyan ), "│" );
  fmt::print( fg( fmt::color::white ), "{:^50}", fmt::format( "Total Tests: {}", overall.total_tests ) );
  fmt::print( fg( fmt::color::cyan ), "│\n" );

  fmt::print( fg( fmt::color::cyan ), "│" );
  fmt::print( fg( fmt::color::green ), "{:^50}", fmt::format( "Overall Success Rate: {:.1f}%", overall.success_rate ) );
  fmt::print( fg( fmt::color::cyan ), "│\n" );

  fmt::print( fg( fmt::color::cyan ), "│" );
  fmt::print(
    fg( fmt::color::white ),
    "{:^50}",
    fmt::format( "Average Time per Test: {:.1f} ms", overall.avg_time_ms ) );
  fmt::print( fg( fmt::color::cyan ), "│\n" );

  fmt::print( fg( fmt::color::cyan ), "╰{:─^{}}╯\n", "", 50 );
}

// Funzione per eseguire i test per una specifica strategia
StrategyStatistics run_tests_for_strategy(
  const string &                    strategy_name,
  NewtonDumped::DampingStrategy     strategy,
  const vector<NonlinearSystem *> & selected_tests,
  const Config &                    config,
  bool                              quiet_mode )
{
  StrategyStatistics stats;
  stats.strategy_name = strategy_name;

  // Setup solver
  NewtonDumped solver;
  solver.set_tolerance( config.tolerance );
  solver.set_max_iterations( config.max_iterations );
  solver.set_max_damping_iterations( config.max_damping_iterations );
  solver.set_verbose( config.verbose && !quiet_mode );
  solver.set_damping_strategy( strategy );

  // Imposta parametri comuni a tutte le strategie
  solver.set_min_lambda( config.min_lambda );

  // Imposta parametri specifici per strategia
  solver.set_mu_init( config.mu_init );
  solver.set_mu_min( config.mu_min );
  solver.set_mu_max( config.mu_max );
  solver.set_mu_increase_factor( config.mu_increase_factor );
  solver.set_mu_decrease_factor( config.mu_decrease_factor );
  solver.set_trust_region_radius( config.trust_region_radius );
  solver.set_trust_region_min( config.trust_region_min );
  solver.set_trust_region_max( config.trust_region_max );

  solver.set_dogleg_delta( config.dogleg_delta );
  solver.set_dogleg_delta_min( config.dogleg_delta_min );
  solver.set_dogleg_delta_max( config.dogleg_delta_max );
  solver.set_dogleg_eta1( config.dogleg_eta1 );
  solver.set_dogleg_eta2( config.dogleg_eta2 );
  solver.set_dogleg_gamma1( config.dogleg_gamma1 );
  solver.set_dogleg_gamma2( config.dogleg_gamma2 );

  solver.set_wolfe_c1( config.wolfe_c1 );
  solver.set_wolfe_c2( config.wolfe_c2 );
  solver.set_wolfe_alpha_init( config.wolfe_alpha_init );
  solver.set_wolfe_alpha_min( config.wolfe_alpha_min );
  solver.set_wolfe_alpha_max( config.wolfe_alpha_max );
  solver.set_wolfe_rho( config.wolfe_rho );

  solver.set_cubic_sigma( config.cubic_sigma );
  solver.set_cubic_sigma_min( config.cubic_sigma_min );
  solver.set_cubic_sigma_max( config.cubic_sigma_max );
  solver.set_cubic_gamma_decrease( config.cubic_gamma_decrease );
  solver.set_cubic_gamma_increase( config.cubic_gamma_increase );
  solver.set_cubic_eta1( config.cubic_eta1 );
  solver.set_cubic_eta2( config.cubic_eta2 );

  solver.set_quadratic_c1( config.quadratic_c1 );
  solver.set_quadratic_c2( config.quadratic_c2 );
  solver.set_quadratic_max_interp( config.quadratic_max_interp );

  solver.set_bank_rose_alpha( config.bank_rose_alpha );
  solver.set_bank_rose_beta( config.bank_rose_beta );
  solver.set_bank_rose_gamma( config.bank_rose_gamma );
  solver.set_bank_rose_theta_min( config.bank_rose_theta_min );
  solver.set_bank_rose_theta_max( config.bank_rose_theta_max );

  solver.set_griewank_eta( config.griewank_eta );
  solver.set_griewank_omega( config.griewank_omega );
  solver.set_griewank_tau( config.griewank_tau );
  solver.set_griewank_zeta( config.griewank_zeta );

  solver.set_filter_theta_min( config.filter_theta_min );
  solver.set_filter_gamma_theta( config.filter_gamma_theta );
  solver.set_filter_gamma_f( config.filter_gamma_f );
  solver.set_filter_alpha( config.filter_alpha );
  solver.set_filter_beta( config.filter_beta );

  solver.set_ctr_delta( config.ctr_delta );
  solver.set_ctr_delta_min( config.ctr_delta_min );
  solver.set_ctr_delta_max( config.ctr_delta_max );
  solver.set_ctr_eta1( config.ctr_eta1 );
  solver.set_ctr_eta2( config.ctr_eta2 );
  solver.set_ctr_gamma1( config.ctr_gamma1 );
  solver.set_ctr_gamma2( config.ctr_gamma2 );
  solver.set_ctr_sigma( config.ctr_sigma );

  // Loop sui test selezionati
  for ( size_t test_idx = 0; test_idx < selected_tests.size(); ++test_idx )
  {
    if ( !quiet_mode && !config.verbose )
    {
      print_progress( static_cast<int>( test_idx ), static_cast<int>( selected_tests.size() ) );
    }

    NonlinearSystem * system = selected_tests[test_idx];

    // Ottieni i punti iniziali
    vector<NewtonDumped::Vector> initial_points;
    system->initial_points( initial_points );

    // Se non ci sono punti iniziali, skippa il test
    if ( initial_points.empty() )
    {
      TestResult result;
      result.test_name     = system->title();
      result.num_equations = static_cast<int>( system->num_equations() );
      result.converged     = false;
      stats.detailed_results.push_back( result );
      stats.total_tests++;
      stats.failed_tests++;
      continue;
    }

    // Determina quali punti iniziali testare
    vector<size_t> points_to_test;
    if ( config.run_single_initial_point )
    {
      if ( config.initial_point_index < static_cast<int>( initial_points.size() ) )
      {
        points_to_test.push_back( static_cast<size_t>( config.initial_point_index ) );
      }
    }
    else
    {
      for ( size_t i = 0; i < initial_points.size(); ++i ) { points_to_test.push_back( i ); }
    }

    // Prova i punti iniziali selezionati
    for ( size_t ip_idx : points_to_test )
    {
      if ( ip_idx >= initial_points.size() )
      {
        continue;  // Indice fuori range
      }

      NewtonDumped::Vector x = initial_points[ip_idx];

      // Risolvi
      Utils::TicToc tm;
      tm.tic();
      bool converged = solver.solve( *system, x );
      tm.toc();

      // Salva risultati
      TestResult result;
      result.test_name           = system->title();
      result.num_equations       = static_cast<int>( system->num_equations() );
      result.converged           = converged;
      result.iterations          = static_cast<int>( solver.get_num_iterations() );
      result.function_evals      = static_cast<int>( solver.get_num_function_evals() );
      result.jacobian_evals      = static_cast<int>( solver.get_num_jacobian_evals() );
      result.final_residual      = solver.get_final_residual();
      result.elapsed_time_ms     = tm.elapsed_ms();
      result.initial_point_index = static_cast<int>( ip_idx );

      stats.detailed_results.push_back( result );
      stats.total_tests++;

      if ( converged )
      {
        stats.converged_tests++;
        stats.total_time_ms += result.elapsed_time_ms;

        if ( result.elapsed_time_ms > stats.max_time_ms ) stats.max_time_ms = result.elapsed_time_ms;
        if ( result.elapsed_time_ms < stats.min_time_ms ) stats.min_time_ms = result.elapsed_time_ms;
      }
      else
      {
        stats.failed_tests++;
      }
    }
  }

  // Calcola statistiche
  if ( stats.converged_tests > 0 )
  {
    stats.success_rate = 100.0 * stats.converged_tests / stats.total_tests;

    double total_iterations     = 0.0;
    double total_function_evals = 0.0;
    double total_jacobian_evals = 0.0;

    for ( const auto & result : stats.detailed_results )
    {
      if ( result.converged )
      {
        total_iterations += result.iterations;
        total_function_evals += result.function_evals;
        total_jacobian_evals += result.jacobian_evals;
      }
    }

    stats.avg_iterations     = total_iterations / stats.converged_tests;
    stats.avg_function_evals = total_function_evals / stats.converged_tests;
    stats.avg_jacobian_evals = total_jacobian_evals / stats.converged_tests;
    stats.avg_time_ms        = stats.total_time_ms / stats.converged_tests;
  }
  else
  {
    stats.success_rate = 0.0;
  }

  return stats;
}

int main( int argc, char * argv[] )
{
  Config config;  // Configurazione locale

  // Setup CLI11
  CLI::App app{ "Newton Method with Damping - Comprehensive Test Suite" };

  // Basic options
  app.add_option( "-t,--tolerance", config.tolerance, "Stopping tolerance on ||f|| (default: 1e-8)" )
    ->check( CLI::PositiveNumber );

  app.add_option( "-i,--max-iterations", config.max_iterations, "Maximum Newton iterations (default: 100)" )
    ->check( CLI::PositiveNumber );

  app
    .add_option(
      "-d,--max-damping-iterations",
      config.max_damping_iterations,
      "Maximum damping iterations per step (default: 20)" )
    ->check( CLI::PositiveNumber );

  app.add_flag( "-v,--verbose", config.verbose, "Enable verbose output during solving" );

  app.add_flag( "-q,--quiet", config.quiet, "Suppress all output except errors" );

  app.add_flag( "--test-all", config.test_all, "Test all damping strategies and print comparative table" );

  app.add_option( "-o,--output", config.output_file, "Output file for results (default: stdout)" );

  // Strategy selection - TUTTE LE STRATEGIE (AGGIORNATA)
  vector<string> strategy_choices = { "deuflhard", "l2_classic", "l2_adaptive", "l2_hybrid",
                                      "dogleg",    "wolfe",      "cubic",       "quadratic",
                                      "bank_rose", "griewank",   "filter",      "cubic_trust_region" };
  app.add_option( "-s,--strategy", config.strategy, "Damping strategy (default: deuflhard)" )
    ->check( CLI::IsMember( strategy_choices ) )
    ->ignore_case();

  // Deuflhard parameters
  auto deuflhard_group = app.add_option_group( "Deuflhard Parameters", "Parameters for Deuflhard damping strategy" );
  deuflhard_group->add_option( "--min-lambda", config.min_lambda, "Minimum allowed damping factor (default: 1e-6)" )
    ->check( CLI::PositiveNumber );

  // L2 parameters
  auto l2_group = app.add_option_group( "L2 Damping Parameters", "Parameters for L2 damping strategies" );
  l2_group->add_option( "--mu-init", config.mu_init, "Initial damping parameter mu (default: 0.01)" )
    ->check( CLI::PositiveNumber );

  l2_group->add_option( "--mu-min", config.mu_min, "Minimum mu value (default: 1e-8)" )->check( CLI::PositiveNumber );

  l2_group->add_option( "--mu-max", config.mu_max, "Maximum mu value (default: 1e4)" )->check( CLI::PositiveNumber );

  l2_group->add_option( "--mu-increase", config.mu_increase_factor, "Factor to increase mu (default: 10.0)" )
    ->check( CLI::PositiveNumber );

  l2_group->add_option( "--mu-decrease", config.mu_decrease_factor, "Factor to decrease mu (default: 0.1)" )
    ->check( CLI::PositiveNumber );

  // Adaptive L2 parameters
  auto adaptive_group = app.add_option_group( "Adaptive L2 Parameters", "Parameters for adaptive L2 strategy" );
  adaptive_group
    ->add_option( "--trust-radius", config.trust_region_radius, "Initial trust region radius (default: 1.0)" )
    ->check( CLI::PositiveNumber );

  adaptive_group->add_option( "--trust-min", config.trust_region_min, "Minimum trust region radius (default: 1e-6)" )
    ->check( CLI::PositiveNumber );

  adaptive_group->add_option( "--trust-max", config.trust_region_max, "Maximum trust region radius (default: 100.0)" )
    ->check( CLI::PositiveNumber );

  adaptive_group->add_option( "--accept-good", config.acceptance_ratio_good, "Good acceptance ratio (default: 0.75)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  adaptive_group->add_option( "--accept-bad", config.acceptance_ratio_bad, "Bad acceptance ratio (default: 0.25)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  // Dogleg parameters
  auto dogleg_group = app.add_option_group( "Dogleg Parameters", "Parameters for Dogleg trust region strategy" );
  dogleg_group->add_option( "--dogleg-delta", config.dogleg_delta, "Initial trust region radius (default: 1.0)" )
    ->check( CLI::PositiveNumber );

  dogleg_group
    ->add_option( "--dogleg-delta-min", config.dogleg_delta_min, "Minimum trust region radius (default: 1e-6)" )
    ->check( CLI::PositiveNumber );

  dogleg_group
    ->add_option( "--dogleg-delta-max", config.dogleg_delta_max, "Maximum trust region radius (default: 100.0)" )
    ->check( CLI::PositiveNumber );

  dogleg_group->add_option( "--dogleg-eta1", config.dogleg_eta1, "Shrink threshold (default: 0.1)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  dogleg_group->add_option( "--dogleg-eta2", config.dogleg_eta2, "Expand threshold (default: 0.75)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  dogleg_group->add_option( "--dogleg-gamma1", config.dogleg_gamma1, "Shrink factor (default: 0.5)" )
    ->check( CLI::PositiveNumber );

  dogleg_group->add_option( "--dogleg-gamma2", config.dogleg_gamma2, "Expand factor (default: 2.0)" )
    ->check( CLI::PositiveNumber );

  // Wolfe parameters
  auto wolfe_group = app.add_option_group(
    "Wolfe Line Search Parameters",
    "Parameters for Wolfe line search strategy" );
  wolfe_group->add_option( "--wolfe-c1", config.wolfe_c1, "Armijo constant (default: 1e-4)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  wolfe_group->add_option( "--wolfe-c2", config.wolfe_c2, "Curvature constant (default: 0.9)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  wolfe_group->add_option( "--wolfe-alpha-init", config.wolfe_alpha_init, "Initial step length (default: 1.0)" )
    ->check( CLI::PositiveNumber );

  wolfe_group->add_option( "--wolfe-alpha-min", config.wolfe_alpha_min, "Minimum step length (default: 1e-10)" )
    ->check( CLI::PositiveNumber );

  wolfe_group->add_option( "--wolfe-alpha-max", config.wolfe_alpha_max, "Maximum step length (default: 1e10)" )
    ->check( CLI::PositiveNumber );

  wolfe_group->add_option( "--wolfe-rho", config.wolfe_rho, "Backtracking factor (default: 0.5)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  // Cubic regularization parameters
  auto cubic_group = app.add_option_group(
    "Cubic Regularization Parameters",
    "Parameters for adaptive cubic regularization" );
  cubic_group->add_option( "--cubic-sigma", config.cubic_sigma, "Initial regularization parameter (default: 0.1)" )
    ->check( CLI::PositiveNumber );

  cubic_group->add_option( "--cubic-sigma-min", config.cubic_sigma_min, "Minimum sigma (default: 1e-8)" )
    ->check( CLI::PositiveNumber );

  cubic_group->add_option( "--cubic-sigma-max", config.cubic_sigma_max, "Maximum sigma (default: 1e8)" )
    ->check( CLI::PositiveNumber );

  cubic_group->add_option( "--cubic-gamma-decrease", config.cubic_gamma_decrease, "Decrease factor (default: 0.5)" )
    ->check( CLI::PositiveNumber );

  cubic_group->add_option( "--cubic-gamma-increase", config.cubic_gamma_increase, "Increase factor (default: 2.0)" )
    ->check( CLI::PositiveNumber );

  cubic_group->add_option( "--cubic-eta1", config.cubic_eta1, "Successful step threshold (default: 0.1)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  cubic_group->add_option( "--cubic-eta2", config.cubic_eta2, "Very successful step threshold (default: 0.75)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  // Quadratic backtracking parameters
  auto quadratic_group = app.add_option_group(
    "Quadratic Backtracking Parameters",
    "Parameters for quadratic/cubic backtracking" );
  quadratic_group->add_option( "--quadratic-c1", config.quadratic_c1, "Armijo constant (default: 1e-4)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  quadratic_group->add_option( "--quadratic-c2", config.quadratic_c2, "Backtracking factor (default: 0.5)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  quadratic_group
    ->add_option( "--quadratic-max-interp", config.quadratic_max_interp, "Maximum interpolation attempts (default: 5)" )
    ->check( CLI::PositiveNumber );

  // Bank & Rose parameters (NUOVO GRUPPO)
  auto bank_rose_group = app.add_option_group( "Bank & Rose Parameters", "Parameters for Bank & Rose (1981) strategy" );
  bank_rose_group->add_option( "--bank-rose-alpha", config.bank_rose_alpha, "Damping reduction factor (default: 0.5)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  bank_rose_group
    ->add_option( "--bank-rose-beta", config.bank_rose_beta, "Sufficient decrease constant (default: 0.1)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  bank_rose_group->add_option( "--bank-rose-gamma", config.bank_rose_gamma, "Contraction factor (default: 0.9)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  bank_rose_group
    ->add_option( "--bank-rose-theta-min", config.bank_rose_theta_min, "Minimum damping factor (default: 1e-4)" )
    ->check( CLI::PositiveNumber );

  bank_rose_group
    ->add_option( "--bank-rose-theta-max", config.bank_rose_theta_max, "Maximum damping factor (default: 1.0)" )
    ->check( CLI::PositiveNumber );

  // Griewank parameters (NUOVO GRUPPO)
  auto griewank_group = app.add_option_group( "Griewank Parameters", "Parameters for Griewank (1980) strategy" );
  griewank_group->add_option( "--griewank-eta", config.griewank_eta, "Acceptance threshold (default: 0.1)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  griewank_group->add_option( "--griewank-omega", config.griewank_omega, "Reduction factor (default: 0.5)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  griewank_group->add_option( "--griewank-tau", config.griewank_tau, "Minimum step size (default: 1e-4)" )
    ->check( CLI::PositiveNumber );

  griewank_group->add_option( "--griewank-zeta", config.griewank_zeta, "Contraction factor (default: 0.9)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  // Filter method parameters (NUOVO GRUPPO)
  auto filter_group = app.add_option_group( "Filter Method Parameters", "Parameters for Filter Methods strategy" );
  filter_group
    ->add_option( "--filter-theta-min", config.filter_theta_min, "Minimum constraint violation (default: 1e-6)" )
    ->check( CLI::PositiveNumber );

  filter_group
    ->add_option( "--filter-gamma-theta", config.filter_gamma_theta, "Filter parameter for theta (default: 0.01)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  filter_group->add_option( "--filter-gamma-f", config.filter_gamma_f, "Filter parameter for f (default: 0.01)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  filter_group->add_option( "--filter-alpha", config.filter_alpha, "Armijo constant (default: 0.5)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  filter_group->add_option( "--filter-beta", config.filter_beta, "Backtracking factor (default: 0.8)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  // Cubic trust region parameters (NUOVO GRUPPO)
  auto ctr_group = app.add_option_group(
    "Cubic Trust Region Parameters",
    "Parameters for Cubic Trust Region strategy" );
  ctr_group->add_option( "--ctr-delta", config.ctr_delta, "Initial trust region radius (default: 1.0)" )
    ->check( CLI::PositiveNumber );

  ctr_group->add_option( "--ctr-delta-min", config.ctr_delta_min, "Minimum trust region radius (default: 1e-6)" )
    ->check( CLI::PositiveNumber );

  ctr_group->add_option( "--ctr-delta-max", config.ctr_delta_max, "Maximum trust region radius (default: 100.0)" )
    ->check( CLI::PositiveNumber );

  ctr_group->add_option( "--ctr-eta1", config.ctr_eta1, "Unsuccessful step threshold (default: 0.1)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  ctr_group->add_option( "--ctr-eta2", config.ctr_eta2, "Very successful step threshold (default: 0.75)" )
    ->check( CLI::Range( 0.0, 1.0 ) );

  ctr_group->add_option( "--ctr-gamma1", config.ctr_gamma1, "Shrink factor (default: 0.5)" )
    ->check( CLI::PositiveNumber );

  ctr_group->add_option( "--ctr-gamma2", config.ctr_gamma2, "Expand factor (default: 2.0)" )
    ->check( CLI::PositiveNumber );

  ctr_group->add_option( "--ctr-sigma", config.ctr_sigma, "Cubic regularization parameter (default: 0.1)" )
    ->check( CLI::PositiveNumber );

  // Test selection options
  auto test_group = app.add_option_group( "Test Selection", "Control which tests to run" );

  test_group
    ->add_option( "--test-indices", config.test_indices, "Run specific test indices (comma-separated or range)" )
    ->expected( -1 );

  test_group->add_option( "--test-name", config.test_name_filter, "Filter tests by name (substring match)" );

  test_group->add_option( "--max-tests", config.max_tests, "Maximum number of tests to run (0 = all)" )
    ->check( CLI::NonNegativeNumber );

  test_group->add_flag(
    "--single-initial-point",
    config.run_single_initial_point,
    "Run only the first initial point for each test" );

  test_group
    ->add_option( "--initial-point", config.initial_point_index, "Specific initial point index to use (default: 0)" )
    ->check( CLI::NonNegativeNumber );

  // Parse command line
  try
  {
    app.parse( argc, argv );
  }
  catch ( const CLI::ParseError & e )
  {
    return app.exit( e );
  }

  // Inizializza i test
  init_nonlinear_system_tests();

  // Banner (solo se non in modalità quiet)
  if ( !config.quiet )
  {
    fmt::print(
      fg( fmt::color::cyan ) | fmt::emphasis::bold,
      "\n\n"
      "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      "━━━━━━━━━━━━━┓\n"
      "┃         NEWTON METHOD WITH DAMPING - COMPREHENSIVE TEST "
      "SUITE            ┃\n"
      "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      "━━━━━━━━━━━━━┛\n"
      "\n" );

    fmt::print( fg( fmt::color::yellow ), "Total number of test problems: " );
    fmt::print( fg( fmt::color::white ), "{}\n", nonlinear_system_tests.size() );

    if ( !config.test_all )
    {
      fmt::print( fg( fmt::color::yellow ), "Damping strategy: " );
      fmt::print( fg( fmt::color::white ), "{}\n", config.strategy );
    }
    else
    {
      fmt::print( fg( fmt::color::yellow ), "Mode: " );
      fmt::print( fg( fmt::color::white ), "Testing ALL strategies\n" );
      fmt::print( fg( fmt::color::yellow ), "Number of strategies: " );
      fmt::print( fg( fmt::color::white ), "{}\n", damping_strategy_map.size() );
    }

    fmt::print( fg( fmt::color::yellow ), "Tolerance: " );
    fmt::print( fg( fmt::color::white ), "{:e}\n", config.tolerance );

    fmt::print( fg( fmt::color::yellow ), "Max iterations: " );
    fmt::print( fg( fmt::color::white ), "{}\n", config.max_iterations );

    fmt::print( fg( fmt::color::yellow ), "Max damping iterations: " );
    fmt::print( fg( fmt::color::white ), "{}\n", config.max_damping_iterations );

    fmt::print( fg( fmt::color::yellow ), "Verbose mode: " );
    fmt::print( fg( fmt::color::white ), "{}\n", config.verbose ? "ON" : "OFF" );

    if ( !config.test_name_filter.empty() )
    {
      fmt::print( fg( fmt::color::yellow ), "Test name filter: " );
      fmt::print( fg( fmt::color::white ), "{}\n", config.test_name_filter );
    }

    if ( !config.test_indices.empty() )
    {
      fmt::print( fg( fmt::color::yellow ), "Selected test indices: " );
      fmt::print( fg( fmt::color::white ), "{}\n", join_ints( config.test_indices, ", " ) );
    }

    if ( config.max_tests > 0 )
    {
      fmt::print( fg( fmt::color::yellow ), "Maximum tests to run: " );
      fmt::print( fg( fmt::color::white ), "{}\n", config.max_tests );
    }

    fmt::print( fg( fmt::color::blue ), "\nStarting tests...\n\n" );
  }

  // Filtra i test da eseguire
  vector<NonlinearSystem *> selected_tests;
  for ( size_t i = 0; i < nonlinear_system_tests.size(); ++i )
  {
    // Controlla se il test è selezionato per indice
    if ( !config.test_indices.empty() )
    {
      bool index_match = false;
      for ( int idx : config.test_indices )
      {
        if ( static_cast<size_t>( idx ) == i )
        {
          index_match = true;
          break;
        }
      }
      if ( !index_match ) continue;
    }

    // Controlla filtro per nome
    if ( !config.test_name_filter.empty() )
    {
      string title = nonlinear_system_tests[i]->title();
      if ( title.find( config.test_name_filter ) == string::npos ) { continue; }
    }

    selected_tests.push_back( nonlinear_system_tests[i] );

    // Limita il numero di test se richiesto
    if ( config.max_tests > 0 && selected_tests.size() >= static_cast<size_t>( config.max_tests ) ) { break; }
  }

  if ( !config.quiet )
  {
    fmt::print( fg( fmt::color::yellow ), "Selected tests to run: " );
    fmt::print( fg( fmt::color::white ), "{}\n\n", selected_tests.size() );
  }

  // MODALITÀ: TEST TUTTE LE STRATEGIE
  if ( config.test_all )
  {
    if ( !config.quiet )
    {
      fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "Testing ALL damping strategies...\n\n" );
    }

    vector<StrategyStatistics> all_strategy_stats;

    // Testa ogni strategia
    for ( const auto & strat_pair : damping_strategy_map )
    {
      const string &                strat_name = strat_pair.first;
      NewtonDumped::DampingStrategy strategy   = strat_pair.second;

      if ( !config.quiet )
      {
        fmt::print( fg( fmt::color::yellow ), "Testing strategy: " );
        fmt::print( fg( fmt::color::white ), "{}", strat_name );
        fmt::print( "... " );
      }

      // Esegui i test per questa strategia
      StrategyStatistics stats = run_tests_for_strategy(
        strat_name,
        strategy,
        selected_tests,
        config,
        true  // true = quiet mode per singola strategia
      );

      all_strategy_stats.push_back( stats );

      if ( !config.quiet )
      {
        if ( stats.converged_tests > 0 )
        {
          fmt::print( fg( fmt::color::green ), "✓ " );
          fmt::print(
            fg( fmt::color::white ),
            "({}/{}, {:.1f}% success, avg {:.1f} ms)\n",
            stats.converged_tests,
            stats.total_tests,
            stats.success_rate,
            stats.avg_time_ms );
        }
        else
        {
          fmt::print( fg( fmt::color::red ), "✗ " );
          fmt::print( fg( fmt::color::white ), "(0/{}, 0.0% success)\n", stats.total_tests );
        }
      }
    }

    if ( !config.quiet )
    {
      fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n✓ All strategies tested!\n\n" );

      // Stampa la tabella comparativa
      print_strategy_comparison_table( all_strategy_stats );

      // Output su file se richiesto
      if ( !config.output_file.empty() )
      {
        ofstream out_file( config.output_file );
        if ( out_file )
        {
          // Salva i dati in formato CSV
          out_file << "Strategy,TotalTests,Converged,SuccessRate,AvgIter,"
                      "AvgFEval,AvgJEval,AvgTime(ms),TotalTime(ms)\n";
          for ( const auto & stats : all_strategy_stats )
          {
            out_file << stats.strategy_name << "," << stats.total_tests << "," << stats.converged_tests << ","
                     << stats.success_rate << "," << stats.avg_iterations << "," << stats.avg_function_evals << ","
                     << stats.avg_jacobian_evals << "," << stats.avg_time_ms << "," << stats.total_time_ms << "\n";
          }
          out_file.close();
          fmt::print( fg( fmt::color::green ), "Results saved to: {}\n", config.output_file );
        }
      }
    }
  }
  // MODALITÀ: TEST SINGOLA STRATEGIA (comportamento originale)
  else
  {
    // Setup solver per singola strategia
    auto it = damping_strategy_map.find( config.strategy );
    if ( it != damping_strategy_map.end() )
    {
      // Esegui i test per la strategia selezionata
      StrategyStatistics stats =
        run_tests_for_strategy( config.strategy, it->second, selected_tests, config, config.quiet );

      // Output dei risultati
      ofstream out_file;

      if ( !config.output_file.empty() )
      {
        out_file.open( config.output_file );
        if ( !out_file )
        {
          fmt::print( fg( fmt::color::red ), "Cannot open output file: {}\n", config.output_file );
          return 1;
        }
      }

      if ( !config.quiet )
      {
        // Output tabellare normale
        print_summary_table( stats.detailed_results );
        print_statistics( stats.detailed_results );

        // Stampa un riepilogo finale
        auto msg = fmt::format(
          " FINAL SUMMARY: {}/{} tests converged ({:.1f}%) ",
          stats.converged_tests,
          stats.total_tests,
          stats.success_rate );

        fmt::print( fg( fmt::color::cyan ), "\n╭{:─^{}}╮\n", "", 58 );
        fmt::print( fg( fmt::color::cyan ), "│" );
        fmt::print( fg( fmt::color::white ) | fmt::emphasis::bold, "{:^58}", msg );
        fmt::print( fg( fmt::color::cyan ), "│" );
        fmt::print( fg( fmt::color::cyan ), "\n╰{:─^{}}╯\n", "", 58 );
      }

      if ( !config.output_file.empty() && out_file.is_open() )
      {
        // Salva i risultati dettagliati in CSV
        out_file << "TestName,NumEquations,Converged,Iterations,FunctionEvals,"
                    "JacobianEvals,FinalResidual,Time(ms),InitialPointIndex\n";
        for ( const auto & result : stats.detailed_results )
        {
          out_file << "\"" << result.test_name << "\"," << result.num_equations << ","
                   << ( result.converged ? "true" : "false" ) << "," << result.iterations << ","
                   << result.function_evals << "," << result.jacobian_evals << "," << result.final_residual << ","
                   << result.elapsed_time_ms << "," << result.initial_point_index << "\n";
        }
        out_file.close();

        if ( !config.quiet )
        {
          fmt::print( fg( fmt::color::green ), "Detailed results saved to: {}\n", config.output_file );
        }
      }

      if ( !config.quiet ) { fmt::print( fg( fmt::color::cyan ), "\nTest suite completed successfully.\n\n" ); }
    }
    else
    {
      fmt::print( fg( fmt::color::red ), "Unknown damping strategy: {}\n", config.strategy );
      return 1;
    }
  }

  // Cleanup
  for ( auto * system : nonlinear_system_tests ) { delete system; }

  return 0;
}
