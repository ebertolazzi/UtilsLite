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

//
// File: test_DifferentialEvolution.cc
//

/*--------------------------------------------------------------------------*\
 |  Driver program for testing Differential Evolution                       |
 |  on all nonlinear system test problems.                                  |
\*--------------------------------------------------------------------------*/

#include "Utils_fmt.hh"
#include "Utils_minimize_DifferentialEvolution.hh"
#include "Utils_nonlinear_system.hh"
#include "Utils_TicToc.hh"
#include "Utils_progress_bar.hh"

using namespace Utils;
using namespace std;

using real_type = double;
using integer   = int;
using DE        = DifferentialEvolution<real_type>;
using Vector    = DifferentialEvolution<real_type>::Vector;

// Struttura per memorizzare i risultati di un test
struct DETestResult
{
  string    test_name;
  integer   num_equations;
  bool      converged;
  integer   iterations;
  integer   function_evals;
  integer   population_size;
  real_type final_residual;
  real_type best_fitness;
  real_type elapsed_time_ms;
  integer   initial_point_index;

  DETestResult()
    : num_equations( 0 )
    , converged( false )
    , iterations( 0 )
    , function_evals( 0 )
    , population_size( 0 )
    , final_residual( 0.0 )
    , best_fitness( 0.0 )
    , elapsed_time_ms( 0.0 )
    , initial_point_index( -1 )
  {
  }
};

// Struttura per statistiche aggregate
struct DEStatistics
{
  integer   total_tests;
  integer   converged_tests;
  integer   failed_tests;
  real_type success_rate;
  real_type avg_iterations;
  real_type avg_function_evals;
  real_type avg_time_ms;
  real_type max_time_ms;
  real_type min_time_ms;
  real_type avg_population_size;

  DEStatistics()
    : total_tests( 0 )
    , converged_tests( 0 )
    , failed_tests( 0 )
    , success_rate( 0.0 )
    , avg_iterations( 0.0 )
    , avg_function_evals( 0.0 )
    , avg_time_ms( 0.0 )
    , max_time_ms( 0.0 )
    , min_time_ms( numeric_limits<real_type>::max() )
    , avg_population_size( 0.0 )
  {
  }
};

// Funzione per troncare una stringa se troppo lunga
string truncate_string( string const & str, size_t max_length )
{
  if ( str.length() <= max_length ) return str;
  return str.substr( 0, max_length - 3 ) + "...";
}

// Funzione per stampare una barra di progresso
void print_progress( integer current, integer total )
{
  real_type progress = static_cast<real_type>( current ) / static_cast<real_type>( total );
  Utils::progress_bar( std::cout, progress, 50, "Progress:" );
}

// Funzione per stampare la tabella riassuntiva
void print_summary_table( const vector<DETestResult> & results )
{
  // Dimensioni delle colonne
  constexpr integer col_idx      = 5;   // # (indice)
  constexpr integer col_status   = 8;   // Status
  constexpr integer col_neq      = 7;   // NEQ
  constexpr integer col_pop      = 7;   // Pop
  constexpr integer col_iter     = 5;   // Iter
  constexpr integer col_feval    = 8;   // F-Eval
  constexpr integer col_fitness  = 12;  // Fitness
  constexpr integer col_residual = 12;  // Residual
  constexpr integer col_time     = 10;  // Time(ms)
  constexpr integer col_name     = 40;  // Test Name

  // Calcola la larghezza totale della tabella
  constexpr integer total_width = 2 + col_idx + 3 + col_status + 3 + col_neq + 3 + col_pop + 3 + col_iter + 3 +
                                  col_feval + 3 + col_fitness + 3 + col_residual + 3 + col_time + 3 + col_name + 2;

  // Intestazione della tabella
  fmt::print( "\n\n" );
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "{:━^{}}\n",
    " DIFFERENTIAL EVOLUTION TEST RESULTS ",
    total_width );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "┏{}┓\n", fmt::format( "{:━^{}}", "", total_width - 2 ) );

  // Intestazione delle colonne
  fmt::print( fg( fmt::color::cyan ), "┃ " );
  fmt::print( "{:>{}} │ ", "#", col_idx );
  fmt::print( "{:^{}} │ ", "Status", col_status );
  fmt::print( "{:>{}} │ ", "NEQ", col_neq );
  fmt::print( "{:>{}} │ ", "Pop", col_pop );
  fmt::print( "{:>{}} │ ", "Iter", col_iter );
  fmt::print( "{:>{}} │ ", "F-Eval", col_feval );
  fmt::print( "{:>{}} │ ", "Fitness", col_fitness );
  fmt::print( "{:>{}} │ ", "Residual", col_residual );
  fmt::print( "{:>{}} │ ", "Time(ms)", col_time );
  fmt::print( "{:<{}} ", "Test Name", col_name );
  fmt::print( fg( fmt::color::cyan ), "┃\n" );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "┠{}┨\n", fmt::format( "{:─^{}}", "", total_width - 2 ) );

  // Dati
  for ( size_t i = 0; i < results.size(); ++i )
  {
    auto const & r = results[i];

    // Colonna indice
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:>{}} │ ", i + 1, col_idx );

    // Colonna status
    if ( r.converged ) { fmt::print( fg( fmt::color::green ), "{:^{}}", "✓ OK", col_status ); }
    else
    {
      fmt::print( fg( fmt::color::red ), "{:^{}}", "✗ FAIL", col_status );
    }
    fmt::print( fg( fmt::color::cyan ), " │ " );

    // Colonna NEQ
    fmt::print( "{:>{}} │ ", r.num_equations, col_neq );

    // Colonna Pop
    fmt::print( "{:>{}} │ ", r.population_size, col_pop );

    // Colonna Iter
    fmt::print( "{:>{}} │ ", r.iterations, col_iter );

    // Colonna F-Eval
    fmt::print( "{:>{}} │ ", r.function_evals, col_feval );

    // Colonna Fitness
    if ( r.best_fitness == 0.0 ) { fmt::print( "{:>12} │ ", "0.00e+00" ); }
    else
    {
      fmt::print( "{:>12.2e} │ ", r.best_fitness );
    }

    // Colonna Residual
    if ( r.final_residual == 0.0 ) { fmt::print( "{:>12} │ ", "0.00e+00" ); }
    else
    {
      fmt::print( "{:>12.2e} │ ", r.final_residual );
    }

    // Colonna Time
    if ( r.elapsed_time_ms < 0.01 ) { fmt::print( "{:>10.3f} │ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 1.0 ) { fmt::print( "{:>10.2f} │ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 10.0 ) { fmt::print( "{:>10.2f} │ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 100.0 ) { fmt::print( "{:>10.2f} │ ", r.elapsed_time_ms ); }
    else
    {
      fmt::print( "{:>10.0f} │ ", r.elapsed_time_ms );
    }

    // Colonna Test Name
    string test_name = truncate_string( r.test_name, col_name );
    fmt::print( "{:<{}} ", test_name, col_name );

    fmt::print( fg( fmt::color::cyan ), "┃\n" );
  }

  // Linea finale
  fmt::print( fg( fmt::color::cyan ), "┗{}┛\n", fmt::format( "{:━^{}}", "", total_width - 2 ) );
}

// Funzione per calcolare e stampare le statistiche
void print_statistics( const vector<DETestResult> & results )
{
  DEStatistics stats;
  stats.total_tests = static_cast<integer>( results.size() );

  real_type total_iterations     = 0.0;
  real_type total_function_evals = 0.0;
  real_type total_time           = 0.0;
  real_type total_population     = 0.0;

  for ( auto const & r : results )
  {
    if ( r.converged )
    {
      stats.converged_tests++;
      total_iterations += r.iterations;
      total_function_evals += r.function_evals;
      total_time += r.elapsed_time_ms;
      total_population += r.population_size;

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
    stats.avg_iterations      = total_iterations / stats.converged_tests;
    stats.avg_function_evals  = total_function_evals / stats.converged_tests;
    stats.avg_time_ms         = total_time / stats.converged_tests;
    stats.avg_population_size = total_population / stats.converged_tests;
  }

  // Dimensioni per la tabella delle statistiche
  constexpr integer stat_col_label   = 25;
  constexpr integer stat_col_value   = 12;
  constexpr integer stat_total_width = stat_col_label + stat_col_value + 4;

  // Stampa delle statistiche
  fmt::print( "\n" );
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "{:━^{}}\n", " STATISTICAL SUMMARY ", stat_total_width );
  fmt::print( fg( fmt::color::cyan ), "┏{}┓\n", fmt::format( "{:━^{}}", "", stat_total_width - 2 ) );
  fmt::print( fg( fmt::color::cyan ), "┃" );
  fmt::print( "{:^{}}", "", stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "┃\n" );

  // Total Tests
  fmt::print( fg( fmt::color::cyan ), "┃ " );
  fmt::print( "{:<{}}", "Total Tests:", stat_col_label );
  fmt::print( fg( fmt::color::white ), "{:>{}}", stats.total_tests, stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " ┃\n" );

  // Converged Tests
  fmt::print( fg( fmt::color::cyan ), "┃ " );
  fmt::print( "{:<{}}", "Converged Tests:", stat_col_label );
  fmt::print(
    fg( fmt::color::green ),
    "{:>{}}",
    fmt::format( "{} ({:.1f}%)", stats.converged_tests, stats.success_rate ),
    stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " ┃\n" );

  // Failed Tests
  fmt::print( fg( fmt::color::cyan ), "┃ " );
  fmt::print( "{:<{}}", "Failed Tests:", stat_col_label );
  fmt::print(
    fg( fmt::color::red ),
    "{:>{}}",
    fmt::format( "{} ({:.1f}%)", stats.failed_tests, 100.0 - stats.success_rate ),
    stat_col_value );
  fmt::print( fg( fmt::color::cyan ), " ┃\n" );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "┠{}┨\n", fmt::format( "{:─^{}}", "", stat_total_width - 2 ) );

  if ( stats.converged_tests > 0 )
  {
    // Average Population Size
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Average Population:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_population_size, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " ┃\n" );

    // Average Iterations
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Average Iterations:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_iterations, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " ┃\n" );

    // Average Function Evals
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Average Function Evals:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_function_evals, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " ┃\n" );

    // Average Time
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Average Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " ┃\n" );

    // Min Time
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Min Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.min_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " ┃\n" );

    // Max Time
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Max Time (ms):", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.max_time_ms, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " ┃\n" );
  }

  fmt::print( fg( fmt::color::cyan ), "┃" );
  fmt::print( "{:^{}}", "", stat_total_width - 2 );
  fmt::print( fg( fmt::color::cyan ), "┃\n" );
  fmt::print( fg( fmt::color::cyan ), "┗{}┛\n", fmt::format( "{:━^{}}", "", stat_total_width - 2 ) );
}

// Funzione per determinare i bound basati sui punti iniziali
void determine_bounds( NonlinearSystem * system, Vector & lower, Vector & upper )
{
  integer n = static_cast<integer>( system->num_equations() );
  lower.resize( n );
  upper.resize( n );

  vector<Vector> initial_points;
  system->initial_points( initial_points );

  if ( initial_points.empty() )
  {
    // Se non ci sono punti iniziali, usa bounds predefiniti
    for ( integer i = 0; i < n; ++i )
    {
      lower[i] = -10.0;
      upper[i] = 10.0;
    }
  }
  else
  {
    // Calcola min e max per ogni dimensione dai punti iniziali
    lower.setConstant( numeric_limits<real_type>::max() );
    upper.setConstant( -numeric_limits<real_type>::max() );

    for ( const auto & point : initial_points )
    {
      for ( integer i = 0; i < n; ++i )
      {
        if ( point[i] < lower[i] ) lower[i] = point[i];
        if ( point[i] > upper[i] ) upper[i] = point[i];
      }
    }

    // Espandi i bounds del 50% in ogni direzione
    for ( integer i = 0; i < n; ++i )
    {
      real_type width = upper[i] - lower[i];
      if ( width < 1e-6 )
      {
        lower[i] -= 5.0;
        upper[i] += 5.0;
      }
      else
      {
        real_type expansion = 0.5 * width;
        lower[i] -= expansion;
        upper[i] += expansion;
      }
    }
  }
}

// Helper: print usage
void print_usage( const char * prog_name )
{
  fmt::print( "Usage: {} [options]\n", prog_name );
  fmt::print( "Options:\n" );
  fmt::print( "  --help                 Show this help and exit\n" );
  fmt::print( "  --verbose              Enable verbose output\n" );
  fmt::print(
    "  --max-iter=N           Set maximum number of DE iterations "
    "(integer)\n" );
  fmt::print(
    "  --tolerance=VAL        Set stopping tolerance (floating, e.g. "
    "1e-8)\n" );
  fmt::print(
    "  --pop-size=N           Set population size (integer, default: "
    "10*Dim)\n" );
  fmt::print( "  --strategy=N           Set DE strategy (1-7, default: 1=RAND_1)\n" );
  fmt::print( "  --weight=F             Set differential weight F (default: 0.8)\n" );
  fmt::print( "  --cr=CR                Set crossover rate (default: 0.9)\n" );
  fmt::print( "  --norm=N               Set norm type (1, 2, or inf, default: 1)\n" );
  fmt::print( "Examples:\n" );
  fmt::print( "  {} --verbose --max-iter=2000 --tolerance=1e-6\n", prog_name );
  fmt::print( "  {} --strategy=2 --weight=0.5 --cr=0.7 --norm=2\n", prog_name );
}

int main( int argc, char * argv[] )
{
  Utils::TicToc tm;

  // Banner
  fmt::print( "\n" );
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    "┃ DIFFERENTIAL EVOLUTION - NONLINEAR SYSTEM TEST SUITE                     ┃\n"
    "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
    "\n" );

  // Default parameters
  real_type tolerance             = 1e-6;
  integer   max_iterations        = 1000;
  integer   population_multiplier = 10;  // NP = population_multiplier * Dim
  integer   strategy              = 1;   // RAND_1
  real_type weight                = 0.8;
  real_type crossover_rate        = 0.9;
  integer   norm_type             = 1;  // 1=norm1, 2=norm2, 3=norm_inf
  bool      verbose_mode          = false;
  bool      custom_pop_size       = false;
  integer   custom_pop_value      = 0;

  // Parse command line arguments
  for ( integer i = 1; i < argc; ++i )
  {
    string arg = argv[i];
    if ( arg == "--help" || arg == "-h" )
    {
      print_usage( argv[0] );
      return 0;
    }
    else if ( arg == "--verbose" ) { verbose_mode = true; }
    else if ( arg.rfind( "--max-iter=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--max-iter=" ).size() );
      try
      {
        max_iterations = stoi( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --max-iter: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--tolerance=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--tolerance=" ).size() );
      try
      {
        tolerance = stod( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --tolerance: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--pop-size=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--pop-size=" ).size() );
      try
      {
        custom_pop_value = stoi( val );
        custom_pop_size  = true;
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --pop-size: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--strategy=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--strategy=" ).size() );
      try
      {
        strategy = stoi( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --strategy: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--weight=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--weight=" ).size() );
      try
      {
        weight = stod( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --weight: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--cr=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--cr=" ).size() );
      try
      {
        crossover_rate = stod( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --cr: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--norm=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--norm=" ).size() );
      try
      {
        norm_type = stoi( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --norm: {}\n", val );
        return 1;
      }
    }
    else
    {
      fmt::print( fg( fmt::color::yellow ), "Unknown option: {}\n", arg );
      print_usage( argv[0] );
      return 1;
    }
  }

  // Print configuration
  fmt::print( fg( fmt::color::yellow ), "Configuration:\n" );
  fmt::print( fg( fmt::color::white ), "  Tolerance:       {:e}\n", tolerance );
  fmt::print( fg( fmt::color::white ), "  Max iterations:  {}\n", max_iterations );
  if ( custom_pop_size ) { fmt::print( fg( fmt::color::white ), "  Population size: {}\n", custom_pop_value ); }
  else
  {
    fmt::print( fg( fmt::color::white ), "  Population:      {} * Dim\n", population_multiplier );
  }
  fmt::print( fg( fmt::color::white ), "  Strategy:        {}\n", strategy );
  fmt::print( fg( fmt::color::white ), "  Weight (F):      {}\n", weight );
  fmt::print( fg( fmt::color::white ), "  Crossover (CR):  {}\n", crossover_rate );

  string norm_name;
  switch ( norm_type )
  {
    case 1: norm_name = "norm1"; break;
    case 2: norm_name = "norm2"; break;
    case 3: norm_name = "norm_inf"; break;
    default: norm_name = "norm1"; break;
  }
  fmt::print( fg( fmt::color::white ), "  Norm type:       {}\n", norm_name );
  fmt::print( fg( fmt::color::white ), "  Verbose mode:    {}\n", verbose_mode ? "ON" : "OFF" );

  // Inizializza i test
  init_nonlinear_system_tests();

  fmt::print( fg( fmt::color::yellow ), "\nTotal number of test problems: " );
  fmt::print( fg( fmt::color::white ), "{}\n\n", nonlinear_system_tests.size() );

  fmt::print( fg( fmt::color::blue ), "Starting Differential Evolution tests...\n\n" );

  // Vettore per memorizzare tutti i risultati
  vector<DETestResult> all_results;

  // Loop su tutti i test
  for ( size_t test_idx = 0; test_idx < nonlinear_system_tests.size(); ++test_idx )
  {
    if ( !verbose_mode ) print_progress( static_cast<integer>( test_idx ), static_cast<integer>( nonlinear_system_tests.size() ) );

    NonlinearSystem * system = nonlinear_system_tests[test_idx];
    integer           n      = static_cast<integer>( system->num_equations() );

    // Determina i bounds
    Eigen::VectorXd lower, upper;
    determine_bounds( system, lower, upper );

    // Crea la funzione obiettivo (norma del residuo)
    auto objective = [system, norm_type]( const Eigen::VectorXd & x ) -> real_type
    {
      Vector f( system->num_equations() );
      try
      {
        system->evaluate( x, f );
      }
      catch ( ... )
      {
        return 1e100;
      }

      switch ( norm_type )
      {
        case 1: return f.lpNorm<1>();
        case 2: return f.norm();
        case 3: return f.lpNorm<Eigen::Infinity>();
        default: return f.lpNorm<1>();
      }
    };

    // Configura DE
    DifferentialEvolution de( n );

    // Imposta la popolazione
    integer npop = population_multiplier * n;
    if ( custom_pop_size ) npop = custom_pop_value;
    if ( npop < 5 )
      npop = 5;
    else if ( npop > 100 )
      npop = 100;

    de.set_population_size( npop );

    // Imposta altri parametri
    de.set_max_iterations( max_iterations );
    de.set_strategy( static_cast<DE::Strategy>( strategy ) );
    de.set_weight( weight );
    de.set_crossover_rate( crossover_rate );
    de.set_bounds( lower, upper );
    de.set_tolerance( tolerance );
    de.set_value_to_reach( tolerance );
    de.set_verbose( verbose_mode );
    de.set_print_interval( verbose_mode ? 100 : max_iterations + 1 );  // Stampa solo alla fine se non verbose

    // Esegui DE
    tm.tic();
    bool converged = de.minimize( objective );
    tm.toc();

    // Calcola il residuo finale con la stessa norma
    Vector final_solution = de.get_best_solution();
    Vector final_residual( n );
    system->evaluate( final_solution, final_residual );

    real_type final_norm;
    switch ( norm_type )
    {
      case 1: final_norm = final_residual.lpNorm<1>(); break;
      case 2: final_norm = final_residual.norm(); break;
      case 3: final_norm = final_residual.lpNorm<Eigen::Infinity>(); break;
      default: final_norm = final_residual.lpNorm<1>(); break;
    }

    // Salva risultati
    DETestResult result;
    result.test_name           = system->title();
    result.num_equations       = n;
    result.converged           = converged;
    result.iterations          = static_cast<integer>( de.get_iteration() );
    result.function_evals      = static_cast<integer>( de.get_function_evaluations() );
    result.population_size     = static_cast<integer>( de.get_population_size() );
    result.best_fitness        = de.get_best_fitness();
    result.final_residual      = final_norm;
    result.elapsed_time_ms     = tm.elapsed_ms();
    result.initial_point_index = 0;  // DE usa popolazione casuale, non punti iniziali specifici

    all_results.push_back( result );

    // Stampa verbose per ogni test
    if ( verbose_mode )
    {
      fmt::print( fg( fmt::color::cyan ), "\nTest: {} (Dim={})\n", system->title(), n );
      fmt::print( fg( fmt::color::white ), "  Result: {}\n", converged ? "Converged" : "Failed" );
      fmt::print( fg( fmt::color::white ), "  Iterations: {}\n", de.get_iteration() );
      fmt::print( fg( fmt::color::white ), "  Function evaluations: {}\n", de.get_function_evaluations() );
      fmt::print( fg( fmt::color::white ), "  Best fitness: {:.2e}\n", de.get_best_fitness() );
      fmt::print( fg( fmt::color::white ), "  Final residual norm: {:.2e}\n", final_norm );
      fmt::print( fg( fmt::color::white ), "  Time: {:.2f} ms\n\n", tm.elapsed_ms() );
    }
  }

  if ( !verbose_mode ) {
    print_progress(
      static_cast<integer>( nonlinear_system_tests.size() ),
      static_cast<integer>( nonlinear_system_tests.size() )
    );
  }
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n\nAll tests completed!\n" );

  // Stampa tabella riassuntiva
  print_summary_table( all_results );

  // Stampa statistiche
  print_statistics( all_results );

  // Stampare i migliori e peggiori risultati
  if ( !all_results.empty() )
  {
    // Trova il test con il miglior fitness
    auto best_it = min_element(
      all_results.begin(),
      all_results.end(),
      []( const DETestResult & a, const DETestResult & b ) { return a.best_fitness < b.best_fitness; } );

    // Trova il test con il peggior fitness (tra quelli convergati)
    auto worst_it = max_element(
      all_results.begin(),
      all_results.end(),
      []( const DETestResult & a, const DETestResult & b )
      {
        real_type a_score = a.converged ? a.best_fitness : -1e100;
        real_type b_score = b.converged ? b.best_fitness : -1e100;
        return a_score < b_score;
      } );

    // Trova il test più veloce
    auto fastest_it = min_element(
      all_results.begin(),
      all_results.end(),
      []( const DETestResult & a, const DETestResult & b ) { return a.elapsed_time_ms < b.elapsed_time_ms; } );

    // Trova il test più lento
    auto slowest_it = max_element(
      all_results.begin(),
      all_results.end(),
      []( const DETestResult & a, const DETestResult & b ) { return a.elapsed_time_ms < b.elapsed_time_ms; } );

    fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{:━^{}}\n", " PERFORMANCE HIGHLIGHTS ", 70 );

    fmt::print( fg( fmt::color::yellow ), "\nBest accuracy: " );
    fmt::print( fg( fmt::color::white ), "{} (fitness={:.2e})\n", best_it->test_name, best_it->best_fitness );

    if ( worst_it->converged )
    {
      fmt::print( fg( fmt::color::yellow ), "Worst accuracy (converged): " );
      fmt::print( fg( fmt::color::white ), "{} (fitness={:.2e})\n", worst_it->test_name, worst_it->best_fitness );
    }

    fmt::print( fg( fmt::color::yellow ), "Fastest: " );
    fmt::print( fg( fmt::color::white ), "{} ({:.2f} ms)\n", fastest_it->test_name, fastest_it->elapsed_time_ms );

    fmt::print( fg( fmt::color::yellow ), "Slowest: " );
    fmt::print( fg( fmt::color::white ), "{} ({:.2f} ms)\n", slowest_it->test_name, slowest_it->elapsed_time_ms );
  }

  // Cleanup
  for ( auto * system : nonlinear_system_tests ) { delete system; }

  fmt::print( fg( fmt::color::cyan ), "\nDifferential Evolution test suite completed.\n\n" );

  return 0;
}
