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
 |  Driver program for testing Nonlinear Kaczmarz method                    |
 |  on all nonlinear system test problems.                                 |
\*--------------------------------------------------------------------------*/

#include "Utils_fmt.hh"
#include "Utils_nonlinear_Kaczmarz.hh"
#include "Utils_nonlinear_system.hh"
#include "Utils_progress_bar.hh"

using namespace Utils;
using namespace std;

// Struttura per memorizzare i risultati di un test
struct TestResult
{
  string test_name;
  int    num_equations;
  bool   converged;
  int    iterations;
  int    function_evals;
  int    jacobian_evals;
  int    line_searches;
  double final_residual;
  double elapsed_time_ms;
  int    initial_point_index;

  TestResult()
    : num_equations( 0 )
    , converged( false )
    , iterations( 0 )
    , function_evals( 0 )
    , jacobian_evals( 0 )
    , line_searches( 0 )
    , final_residual( 0.0 )
    , elapsed_time_ms( 0.0 )
    , initial_point_index( -1 )
  {
  }
};

// Struttura per statistiche aggregate
struct Statistics
{
  int    total_tests;
  int    converged_tests;
  int    failed_tests;
  double success_rate;
  double avg_iterations;
  double avg_function_evals;
  double avg_jacobian_evals;
  double avg_line_searches;
  double avg_time_ms;
  double max_time_ms;
  double min_time_ms;

  Statistics()
    : total_tests( 0 )
    , converged_tests( 0 )
    , failed_tests( 0 )
    , success_rate( 0.0 )
    , avg_iterations( 0.0 )
    , avg_function_evals( 0.0 )
    , avg_jacobian_evals( 0.0 )
    , avg_line_searches( 0.0 )
    , avg_time_ms( 0.0 )
    , max_time_ms( 0.0 )
    , min_time_ms( numeric_limits<double>::max() )
  {
  }
};

// Funzione per troncare una stringa se troppo lunga
string truncate_string( const string & str, size_t max_length )
{
  if ( str.length() <= max_length ) return str;
  return str.substr( 0, max_length - 3 ) + "...";
}

// Funzione per stampare una barra di progresso
void print_progress( int current, int total )
{
  double progress = static_cast<double>( current ) / static_cast<double>( total );
  Utils::progress_bar( std::cout, progress, 50, "Progess:" );
}

// Funzione per stampare la tabella riassuntiva
void print_summary_table( const vector<TestResult> & results )
{
  // Dimensioni delle colonne
  constexpr int col_idx      = 4;   // # (indice)
  constexpr int col_status   = 8;   // Status
  constexpr int col_neq      = 4;   // NEQ
  constexpr int col_iter     = 5;   // Iter (Kaczmarz usa più iterazioni)
  constexpr int col_feval    = 5;   // F-Eval
  constexpr int col_jeval    = 5;   // J-Eval
  constexpr int col_ls       = 8;   // Line Search
  constexpr int col_residual = 10;  // Residual
  constexpr int col_time     = 10;  // Time(ms)
  constexpr int col_name     = 30;  // Test Name

  // Calcola la larghezza totale della tabella
  constexpr int total_width = 2 + col_idx + 3 + col_status + 3 + col_neq + 3 + col_iter + 3 + col_feval + 3 +
                              col_jeval + 3 + col_ls + 3 + col_residual + 3 + col_time + 3 + col_name + 2;

  // Intestazione della tabella
  fmt::print( "\n\n" );
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "{:━^{}}\n",
    " NONLINEAR KACZMARZ TEST RESULTS ",
    total_width );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "┏{}┓\n", fmt::format( "{:━^{}}", "", total_width - 2 ) );

  // Intestazione delle colonne
  fmt::print( fg( fmt::color::cyan ), "┃ " );
  fmt::print( "{:>{}} │ ", "#", col_idx );
  fmt::print( "{:^{}} │ ", "Status", col_status );
  fmt::print( "{:>{}} │ ", "NEQ", col_neq );
  fmt::print( "{:>{}} │ ", "Iter", col_iter );
  fmt::print( "{:>{}} │ ", "F-Eval", col_feval );
  fmt::print( "{:>{}} │ ", "J-Eval", col_jeval );
  fmt::print( "{:>{}} │ ", "LS", col_ls );
  fmt::print( "{:>{}} │ ", "Residual", col_residual );
  fmt::print( "{:>{}} │ ", "Time(ms)", col_time );
  fmt::print( "{:<{}} ", "Test Name", col_name );
  fmt::print( fg( fmt::color::cyan ), "┃\n" );

  // Linea divisoria
  fmt::print( fg( fmt::color::cyan ), "┠{}┨\n", fmt::format( "{:─^{}}", "", total_width - 2 ) );

  // Dati
  for ( size_t i = 0; i < results.size(); ++i )
  {
    const auto & r = results[i];

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

    // Colonna Iter
    fmt::print( "{:>{}} │ ", r.iterations, col_iter );

    // Colonna F-Eval
    fmt::print( "{:>{}} │ ", r.function_evals, col_feval );

    // Colonna J-Eval
    fmt::print( "{:>{}} │ ", r.jacobian_evals, col_jeval );

    // Colonna LS
    fmt::print( "{:>{}} │ ", r.line_searches, col_ls );

    // Colonna Residual
    if ( r.final_residual == 0.0 ) { fmt::print( "{:>12} │ ", "0.00e+00" ); }
    else
    {
      fmt::print( "{:>12.2e} │ ", r.final_residual );
    }

    // Colonna Time
    if ( r.elapsed_time_ms < 0.01 ) { fmt::print( "{:>12.3f} │ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 1.0 ) { fmt::print( "{:>12.2f} │ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 10.0 ) { fmt::print( "{:>12.2f} │ ", r.elapsed_time_ms ); }
    else if ( r.elapsed_time_ms < 100.0 ) { fmt::print( "{:>12.2f} │ ", r.elapsed_time_ms ); }
    else
    {
      fmt::print( "{:>12.0f} │ ", r.elapsed_time_ms );
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
void print_statistics( const vector<TestResult> & results )
{
  Statistics stats;
  stats.total_tests = static_cast<int>( results.size() );

  double total_iterations     = 0.0;
  double total_function_evals = 0.0;
  double total_jacobian_evals = 0.0;
  double total_line_searches  = 0.0;
  double total_time           = 0.0;

  for ( const auto & r : results )
  {
    if ( r.converged )
    {
      stats.converged_tests++;
      total_iterations += r.iterations;
      total_function_evals += r.function_evals;
      total_jacobian_evals += r.jacobian_evals;
      total_line_searches += r.line_searches;
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
    stats.avg_line_searches  = total_line_searches / stats.converged_tests;
    stats.avg_time_ms        = total_time / stats.converged_tests;
  }

  // Dimensioni per la tabella delle statistiche
  constexpr int stat_col_label   = 25;
  constexpr int stat_col_value   = 12;
  constexpr int stat_total_width = stat_col_label + stat_col_value + 4;

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

    // Average Jacobian Evals
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Average Jacobian Evals:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_jacobian_evals, stat_col_value );
    fmt::print( fg( fmt::color::cyan ), " ┃\n" );

    // Average Line Searches
    fmt::print( fg( fmt::color::cyan ), "┃ " );
    fmt::print( "{:<{}}", "Average Line Searches:", stat_col_label );
    fmt::print( fg( fmt::color::white ), "{:>{}.2f}", stats.avg_line_searches, stat_col_value );
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

int main( int argc, char * argv[] )
{
  // Banner
  fmt::print( "\n" );
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    "┃                  NONLINEAR KACZMARZ - COMPREHENSIVE TEST SUITE                       ┃\n"
    "┃                (Randomized Kaczmarz Method for Nonlinear Systems)                    ┃\n"
    "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
    "\n" );

  // Inizializza i test
  init_nonlinear_system_tests();

  fmt::print( fg( fmt::color::yellow ), "Total number of test problems: " );
  fmt::print( fg( fmt::color::white ), "{}\n\n", nonlinear_system_tests.size() );

  // Parametri del metodo con valori di default
  double       tolerance          = 1e-8;
  double       rel_tolerance      = 1e-8;
  int          max_iterations     = 1000;
  int          max_function_evals = 10000;
  double       relaxation         = 1.0;
  int          strategy           = 1;  // 0: CYCLIC, 1: RANDOM_UNIFORM, 2: RANDOM_WEIGHTED, 3: GREEDY
  bool         use_line_search    = false;
  int          block_size         = 1;  // 1 = Kaczmarz classico, >1 = Block Kaczmarz
  int          verbose_level      = 1;  // 0=silent, 1=summary, 2=detailed
  unsigned int random_seed        = 0;  // 0 significa usare time

  // Helper: print usage
  auto print_usage = [&]()
  {
    fmt::print( "Usage: {} [options]\n", argv[0] );
    fmt::print( "Options:\n" );
    fmt::print( "  --help                 Show this help and exit\n" );
    fmt::print( "  --verbose-level=N      Set verbose level (0,1,2, default=1)\n" );
    fmt::print( "  --max-iter=N           Set maximum number of iterations (integer)\n" );
    fmt::print( "  --max-feval=N          Set maximum number of function evaluations (integer)\n" );
    fmt::print( "  --tolerance=VAL        Set absolute tolerance (floating, e.g. 1e-8)\n" );
    fmt::print( "  --rel-tolerance=VAL    Set relative tolerance (floating, e.g. 1e-8)\n" );
    fmt::print( "  --relaxation=VAL       Set relaxation parameter (floating, efault=1.0)\n" );
    fmt::print(
      "  --strategy=N           Set selection strategy "
      "(0:CYCLIC,1:RANDOM_UNIFORM,2:RANDOM_WEIGHTED,3:GREEDY, default=1)\n" );
    fmt::print( "  --line-search=N        Enable line search (0:OFF, 1:ON, default=0)\n" );
    fmt::print( "  --block-size=N         Set block size for Block Kaczmarz (integer, default=1)\n" );
    fmt::print( "  --print-freq=N         Set print frequency (integer, default=1000)\n" );
    fmt::print( "  --seed=N               Set random seed (unsigned integer, default=0 means time)\n" );
    fmt::print( "\n" );
    fmt::print( "You can also use positional arguments (for backward compatibility):\n" );
    fmt::print( "  {} [tolerance] [max_iter] [relaxation] [strategy] [line_search] [block_size]\n", argv[0] );
    fmt::print( "Examples:\n" );
    fmt::print( "  {} --verbose-level=2 --max-iter=5000 --tolerance=1e-10 --strategy=3\n", argv[0] );
    fmt::print( "  {} 1e-10 5000 1.0 3 0 1\n", argv[0] );
  };

  // Parse command line arguments; support long options and positional args
  int positional_count = 0;
  for ( int i = 1; i < argc; ++i )
  {
    string arg = argv[i];
    if ( arg == "--help" || arg == "-h" )
    {
      print_usage();
      return 0;
    }
    else if ( arg.rfind( "--verbose-level=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--verbose-level=" ).size() );
      try
      {
        verbose_level = stoi( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --verbose-level: {}\n", val );
        return 1;
      }
    }
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
    else if ( arg.rfind( "--max-feval=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--max-feval=" ).size() );
      try
      {
        max_function_evals = stoi( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --max-feval: {}\n", val );
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
    else if ( arg.rfind( "--rel-tolerance=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--rel-tolerance=" ).size() );
      try
      {
        rel_tolerance = stod( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --rel-tolerance: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--relaxation=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--relaxation=" ).size() );
      try
      {
        relaxation = stod( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --relaxation: {}\n", val );
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
    else if ( arg.rfind( "--line-search=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--line-search=" ).size() );
      try
      {
        use_line_search = ( stoi( val ) != 0 );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --line-search: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--block-size=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--block-size=" ).size() );
      try
      {
        block_size = stoi( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --block-size: {}\n", val );
        return 1;
      }
    }
    else if ( arg.rfind( "--seed=", 0 ) == 0 )
    {
      string val = arg.substr( string( "--seed=" ).size() );
      try
      {
        random_seed = stoul( val );
      }
      catch ( ... )
      {
        fmt::print( fg( fmt::color::red ), "Invalid value for --seed: {}\n", val );
        return 1;
      }
    }
    else if ( arg.size() > 0 && arg[0] == '-' )
    {
      fmt::print( fg( fmt::color::yellow ), "Unknown option: {}\n", arg );
      print_usage();
      return 1;
    }
    else
    {
      // positional args
      ++positional_count;
      if ( positional_count == 1 )
      {
        try
        {
          tolerance = stod( arg );
        }
        catch ( ... )
        {
          fmt::print( fg( fmt::color::red ), "Invalid positional tolerance: {}\n", arg );
          return 1;
        }
      }
      else if ( positional_count == 2 )
      {
        try
        {
          max_iterations = stoi( arg );
        }
        catch ( ... )
        {
          fmt::print( fg( fmt::color::red ), "Invalid positional max_iter: {}\n", arg );
          return 1;
        }
      }
      else if ( positional_count == 3 )
      {
        try
        {
          relaxation = stod( arg );
        }
        catch ( ... )
        {
          fmt::print( fg( fmt::color::red ), "Invalid positional relaxation: {}\n", arg );
          return 1;
        }
      }
      else if ( positional_count == 4 )
      {
        try
        {
          strategy = stoi( arg );
        }
        catch ( ... )
        {
          fmt::print( fg( fmt::color::red ), "Invalid positional strategy: {}\n", arg );
          return 1;
        }
      }
      else if ( positional_count == 5 )
      {
        try
        {
          use_line_search = ( stoi( arg ) != 0 );
        }
        catch ( ... )
        {
          fmt::print( fg( fmt::color::red ), "Invalid positional line_search: {}\n", arg );
          return 1;
        }
      }
      else if ( positional_count == 6 )
      {
        try
        {
          block_size = stoi( arg );
        }
        catch ( ... )
        {
          fmt::print( fg( fmt::color::red ), "Invalid positional block_size: {}\n", arg );
          return 1;
        }
      }
    }
  }

  // Stampa i parametri utilizzati
  fmt::print( fg( fmt::color::yellow ), "Using absolute tolerance: " );
  fmt::print( fg( fmt::color::white ), "{:e}\n", tolerance );
  fmt::print( fg( fmt::color::yellow ), "Using relative tolerance: " );
  fmt::print( fg( fmt::color::white ), "{:e}\n", rel_tolerance );
  fmt::print( fg( fmt::color::yellow ), "Using max iterations: " );
  fmt::print( fg( fmt::color::white ), "{}\n", max_iterations );
  fmt::print( fg( fmt::color::yellow ), "Using max function evaluations: " );
  fmt::print( fg( fmt::color::white ), "{}\n", max_function_evals );
  fmt::print( fg( fmt::color::yellow ), "Using relaxation parameter: " );
  fmt::print( fg( fmt::color::white ), "{}\n", relaxation );
  fmt::print( fg( fmt::color::yellow ), "Using strategy: " );
  string strategy_name;
  switch ( strategy )
  {
    case 0: strategy_name = "CYCLIC"; break;
    case 1: strategy_name = "RANDOM_UNIFORM"; break;
    case 2: strategy_name = "RANDOM_WEIGHTED"; break;
    case 3: strategy_name = "GREEDY"; break;
    default: strategy_name = "RANDOM_UNIFORM";
  }
  fmt::print( fg( fmt::color::white ), "{}\n", strategy_name );
  fmt::print( fg( fmt::color::yellow ), "Using line search: " );
  fmt::print( fg( fmt::color::white ), "{}\n", use_line_search );
  fmt::print( fg( fmt::color::yellow ), "Using block size: " );
  fmt::print( fg( fmt::color::white ), "{}\n", block_size );
  fmt::print( fg( fmt::color::yellow ), "Using verbose level: " );
  fmt::print( fg( fmt::color::white ), "{}\n", verbose_level );
  fmt::print( fg( fmt::color::yellow ), "Using print frequency: " );
  if ( random_seed == 0 )
  {
    fmt::print( fg( fmt::color::yellow ), "Random seed: " );
    fmt::print( fg( fmt::color::white ), "time-based\n" );
  }
  else
  {
    fmt::print( fg( fmt::color::yellow ), "Using random seed: " );
    fmt::print( fg( fmt::color::white ), "{}\n", random_seed );
  }

  fmt::print( fg( fmt::color::blue ), "\nStarting tests...\n\n" );

  // Vettore per memorizzare tutti i risultati
  vector<TestResult> all_results;

  // Loop su tutti i test
  for ( size_t test_idx = 0; test_idx < nonlinear_system_tests.size(); ++test_idx )
  {
    print_progress( static_cast<int>( test_idx ), static_cast<int>( nonlinear_system_tests.size() ) );

    NonlinearSystem * system = nonlinear_system_tests[test_idx];

    // Ottieni i punti iniziali
    vector<NonlinearKaczmarz::Vector> initial_points;
    system->initial_points( initial_points );

    // Se non ci sono punti iniziali, skippa il test
    if ( initial_points.empty() )
    {
      TestResult result;
      result.test_name     = system->title();
      result.num_equations = static_cast<int>( system->num_equations() );
      result.converged     = false;
      all_results.push_back( result );
      continue;
    }

    // Prova con il primo punto iniziale (come nel codice originale)
    for ( size_t ip_idx = 0; ip_idx < min( initial_points.size(), size_t( 1 ) ); ++ip_idx )
    {
      NonlinearKaczmarz solver;
      solver.set_tolerance( tolerance );
      solver.set_relative_tolerance( rel_tolerance );
      solver.set_max_iterations( max_iterations );
      solver.set_max_function_evals( max_function_evals );
      solver.set_relaxation( relaxation );
      solver.set_use_line_search( use_line_search );
      solver.set_strategy( static_cast<NonlinearKaczmarz::SelectionStrategy>( strategy ) );
      solver.set_verbose_level( verbose_level );
      if ( random_seed != 0 )
      {
        solver.set_random_seed( random_seed + static_cast<unsigned>( test_idx ) );  // diverso per ogni test
      }

      NonlinearKaczmarz::Vector x = initial_points[ip_idx];

      // Tempo di esecuzione
      auto start = chrono::high_resolution_clock::now();

      // Risolvi (scegli tra metodo classico o block)
      bool converged;
      if ( block_size == 1 ) { converged = solver.solve( *system, x ); }
      else
      {
        converged = solver.solve( *system, x );  // Usiamo solve normale per ora
        // Nota: solve_block non è implementato con tutte le feature
      }

      auto                            end     = chrono::high_resolution_clock::now();
      chrono::duration<double, milli> elapsed = end - start;

      // Salva risultati
      TestResult result;
      result.test_name           = system->title();
      result.num_equations       = static_cast<int>( system->num_equations() );
      result.converged           = converged;
      result.iterations          = static_cast<int>( solver.get_num_iterations() );
      result.function_evals      = static_cast<int>( solver.get_num_function_evals() );
      result.jacobian_evals      = static_cast<int>( solver.get_num_jacobian_evals() );
      result.line_searches       = static_cast<int>( solver.get_num_line_searches() );
      result.final_residual      = solver.get_final_residual();
      result.elapsed_time_ms     = elapsed.count();
      result.initial_point_index = static_cast<int>( ip_idx );

      all_results.push_back( result );
    }
  }

  print_progress( static_cast<int>( nonlinear_system_tests.size() ), static_cast<int>( nonlinear_system_tests.size() ) );
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n\nAll tests completed!\n" );

  // Stampa tabella riassuntiva
  print_summary_table( all_results );

  // Stampa statistiche
  print_statistics( all_results );

  // Cleanup
  for ( auto * system : nonlinear_system_tests ) { delete system; }

  fmt::print( fg( fmt::color::cyan ), "\nTest suite completed successfully.\n\n" );

  return 0;
}
