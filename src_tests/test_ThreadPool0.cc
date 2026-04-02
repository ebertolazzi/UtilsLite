/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2017                                                      |
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

#include "Utils.hh"
#include "Utils_fmt.hh"
#include "Utils_TicToc.hh"

// #include <fstream>
// #include <algorithm>
// #include <numeric>

using std::cout;

// ===========================================================================
// Configuration
// ===========================================================================

enum class RunMode
{
  QUICK,
  FULL,
  SCALING,
  HELP
};

struct Config
{
  int     num_threads{ 16 };
  int     task_size{ 200 };
  RunMode mode{ RunMode::QUICK };
  bool    export_csv{ false };
};

// ===========================================================================
// Simple Test Function
// ===========================================================================

static void do_test( int, int )
{
  // Simulate some work
  std::this_thread::sleep_for( std::chrono::microseconds( 10 ) );
}

// ===========================================================================
// Simple Testing Implementation
// ===========================================================================

template <class TP> std::pair<double, double> test_pool_simple( int NN, int nt, int sz, int num_runs = 3 )
{
  Utils::TicToc       tm;
  std::vector<double> times;

  // Warm-up
  {
    TP pool( 1 );
    for ( int i = 0; i < 10; ++i ) pool.run( do_test, i, sz );
    pool.wait();
  }

  for ( int run = 0; run < num_runs; ++run )
  {
    tm.tic();
    {
      TP pool( nt );
      for ( int i = 0; i < NN; ++i ) pool.run( do_test, i, sz );
      pool.wait();
    }
    tm.toc();

    if ( run > 0 )
    {  // Skip first run (warm-up)
      times.push_back( tm.elapsed_mus() );
    }

    if ( run < num_runs - 1 ) { std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) ); }
  }

  if ( times.empty() ) return { 0.0, 0.0 };

  double sum    = std::accumulate( times.begin(), times.end(), 0.0 );
  double count  = static_cast<double>( times.size() );
  double avg    = sum / count;
  double sq_sum = std::inner_product( times.begin(), times.end(), times.begin(), 0.0 );
  double stddev = std::sqrt( sq_sum / count - avg * avg );

  return { avg, stddev };
}

// ===========================================================================
// Results Presentation
// ===========================================================================

void print_simple_table( const std::vector<std::tuple<std::string, double, double>> & results )
{
  // Find best time
  double best_time = std::numeric_limits<double>::max();
  for ( const auto & [name, time, stddev] : results )
  {
    if ( time < best_time ) best_time = time;
  }

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "\n"
    "┌───────────────────────────┬────────────────┬────────────────┬────────────────┐\n"
    "│ Thread Pool Type          │   Total Time   │     Std Dev    │   Efficiency   │\n"
    "├───────────────────────────┼────────────────┼────────────────┼────────────────┤\n" );

  for ( const auto & [name, time, stddev] : results )
  {
    std::string time_str;
    std::string stddev_str = fmt::format( "{:>8.1f}%", ( stddev / time ) * 100.0 );

    if ( time < 1000 ) { time_str = fmt::format( "{:>8.2f} µs", time ); }
    else if ( time < 1'000'000 ) { time_str = fmt::format( "{:>8.3f} ms", time / 1000 ); }
    else
    {
      time_str = fmt::format( "{:>8.3f} s", time / 1'000'000 );
    }

    // Color code based on performance
    if ( time == best_time )
    {
      time_str = fmt::format( fg( fmt::color::green ) | fmt::emphasis::bold, "{:>14}", time_str );
    }
    else if ( time < best_time * 1.1 ) { time_str = fmt::format( fg( fmt::color::yellow ), "{:>14}", time_str ); }
    else
    {
      time_str = fmt::format( fg( fmt::color::orange_red ), "{:>14}", time_str );
    }

    double      efficiency = ( best_time > 0 ) ? best_time / time * 100.0 : 0.0;
    std::string eff_str    = fmt::format( "{:>8.1f}%", efficiency );

    fmt::print( "│ {:<25} │ {} │ {:>14} │ {:>14} │\n", name, time_str, stddev_str, eff_str );
  }

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "└───────────────────────────┴────────────────┴────────────────┴────────────────┘\n" );
}

void print_scaling_results( const std::vector<std::tuple<int, double, double>> & results )
{
  if ( results.empty() ) return;

  double single_thread_time = std::get<1>( results[0] );

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "\n"
    "┌──────────┬────────────────┬─────────┬────────────┐\n"
    "│ Threads  │   Total Time   │ Speedup │ Efficiency │\n"
    "├──────────┼────────────────┼─────────┼────────────┤\n" );

  for ( const auto & [threads, time, stddev] : results )
  {
    double speedup    = single_thread_time / time;
    double efficiency = ( speedup / threads ) * 100.0;

    std::string time_str;
    if ( time < 1000 ) { time_str = fmt::format( "{:>8.2f} µs", time ); }
    else if ( time < 1'000'000 ) { time_str = fmt::format( "{:>8.3f} ms", time / 1000 ); }
    else
    {
      time_str = fmt::format( "{:>8.3f} s", time / 1'000'000 );
    }

    std::string efficiency_str;
    if ( efficiency >= 90.0 ) { efficiency_str = fmt::format( fg( fmt::color::green ), "{:>9.2f}%", efficiency ); }
    else if ( efficiency >= 70.0 )
    {
      efficiency_str = fmt::format( fg( fmt::color::yellow ), "{:>9.2f}%", efficiency );
    }
    else
    {
      efficiency_str = fmt::format( fg( fmt::color::orange_red ), "{:>9.2f}%", efficiency );
    }

    fmt::print( "│ {:8d} │ {:>14} │ {:>7.2f} │ {} │\n", threads, time_str, speedup, efficiency_str );
  }

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "└──────────┴────────────────┴─────────┴────────────┘\n" );
}

// ===========================================================================
// Test Functions
// ===========================================================================

void run_quick_test( int nt, int sz )
{
  const int NN = 1000;

  fmt::print(
    fg( fmt::color::light_blue ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                         QUICK THREAD POOL BENCHMARK                          ║\n"
    "╠══════════════════════════════════════════════════════════════════════════════╣\n"
    "║  Configuration: {:<5d} tasks | {:<2d} threads | size={:<5d}                        ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n",
    NN,
    nt,
    sz );

  fmt::print( fg( fmt::color::light_gray ), "Running quick benchmark (3 runs per pool)...\n\n" );

  std::vector<std::tuple<std::string, double, double>> results;

  auto test_and_add = [&]( auto pool, const std::string & name )
  {
    fmt::print( fg( fmt::color::light_gray ), "  Testing {}...", name );
    auto [avg, stddev] = test_pool_simple<decltype( pool )>( NN, nt, sz, 3 );
    results.emplace_back( name, avg, stddev );
    fmt::print( fg( fmt::color::green ), " ✓\n" );
  };

  test_and_add( Utils::ThreadPool0( 1 ), "ThreadPool0" );
  test_and_add( Utils::ThreadPool1( 1 ), "ThreadPool1" );
  test_and_add( Utils::ThreadPool2( 1 ), "ThreadPool2" );
  test_and_add( Utils::ThreadPool3( 1 ), "ThreadPool3" );
  test_and_add( Utils::ThreadPool4( 1 ), "ThreadPool4" );
  test_and_add( Utils::ThreadPool5( 1 ), "ThreadPool5" );
  test_and_add( Utils::ThreadPoolEigen( 1 ), "ThreadPoolEigen" );

  print_simple_table( results );
}

void run_full_test( int nt, int sz )
{
  fmt::print(
    fg( fmt::color::light_blue ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                        FULL THREAD POOL BENCHMARK                            ║\n"
    "╠══════════════════════════════════════════════════════════════════════════════╣\n"
    "║  Configuration: {:<2d} threads | size={:<5d}                                      ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n",
    nt,
    sz );

  std::vector<int> test_sizes = { 16, 100, 1000, 10000 };

  for ( int NN : test_sizes )
  {
    fmt::print(
      fg( fmt::color::light_sea_green ) | fmt::emphasis::bold,
      "\n"
      "══════════════════════════════════════════════════════════════════════════════════\n"
      "  Testing with {} tasks\n"
      "══════════════════════════════════════════════════════════════════════════════════\n",
      NN );

    std::vector<std::tuple<std::string, double, double>> results;

    auto test_and_add = [&]( auto pool, const std::string & name )
    {
      auto [avg, stddev] = test_pool_simple<decltype( pool )>( NN, nt, sz, 3 );
      results.emplace_back( name, avg, stddev );
    };

    test_and_add( Utils::ThreadPool0( 1 ), "ThreadPool0" );
    test_and_add( Utils::ThreadPool1( 1 ), "ThreadPool1" );
    test_and_add( Utils::ThreadPool2( 1 ), "ThreadPool2" );
    test_and_add( Utils::ThreadPool3( 1 ), "ThreadPool3" );
    test_and_add( Utils::ThreadPool4( 1 ), "ThreadPool4" );
    test_and_add( Utils::ThreadPool5( 1 ), "ThreadPool5" );
    test_and_add( Utils::ThreadPoolEigen( 1 ), "ThreadPoolEigen" );

    print_simple_table( results );

    if ( NN != test_sizes.back() ) { std::this_thread::sleep_for( std::chrono::milliseconds( 200 ) ); }
  }
}

void run_scaling_test( int max_threads, int sz )
{
  const int NN = 10000;

  fmt::print(
    fg( fmt::color::light_blue ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                       SCALABILITY BENCHMARK                                  ║\n"
    "╠══════════════════════════════════════════════════════════════════════════════╣\n"
    "║  Configuration: {:<5d} tasks | up to {:<2d} threads | size={:<5d}                 ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n",
    NN,
    max_threads,
    sz );

  std::vector<int> thread_counts = { 1, 2, 4, 8, 16, 32 };
  thread_counts.erase(
    std::remove_if( thread_counts.begin(), thread_counts.end(), [max_threads]( int t ) { return t > max_threads; } ),
    thread_counts.end() );

  std::vector<std::tuple<int, double, double>> results;

  for ( int nt : thread_counts )
  {
    fmt::print( fg( fmt::color::light_gray ), "  Testing with {} threads...", nt );
    auto [avg, stddev] = test_pool_simple<Utils::ThreadPool4>( NN, nt, sz, 3 );
    results.emplace_back( nt, avg, stddev );
    fmt::print( fg( fmt::color::green ), " ✓\n" );

    if ( nt != thread_counts.back() ) { std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) ); }
  }

  print_scaling_results( results );
}

// ===========================================================================
// Command Line Parsing
// ===========================================================================

Config parse_config( int argc, char * argv[] )
{
  Config config;

  for ( int i = 1; i < argc; ++i )
  {
    std::string arg = argv[i];

    if ( arg == "-t" || arg == "--threads" )
    {
      if ( i + 1 < argc ) config.num_threads = std::atoi( argv[++i] );
    }
    else if ( arg == "-s" || arg == "--size" )
    {
      if ( i + 1 < argc ) config.task_size = std::atoi( argv[++i] );
    }
    else if ( arg == "--full" ) { config.mode = RunMode::FULL; }
    else if ( arg == "--scaling" )
    {
      config.mode = RunMode::SCALING;
      if ( i + 1 < argc )
      {
        int max_threads = std::atoi( argv[++i] );
        if ( max_threads > 0 ) config.num_threads = max_threads;
      }
    }
    else if ( arg == "--csv" ) { config.export_csv = true; }
    else if ( arg == "-h" || arg == "--help" ) { config.mode = RunMode::HELP; }
  }

  return config;
}

void print_help( const char * program_name )
{
  fmt::print(
    "Thread Pool Benchmark Tool\n"
    "Usage: {} [options]\n\n"
    "Options:\n"
    "  -t, --threads N      Set number of threads (default: 16)\n"
    "  -s, --size N         Set task size (default: 200)\n"
    "  --full               Run full benchmark (multiple task sizes)\n"
    "  --scaling [MAX]      Run scalability test (optional max threads)\n"
    "  --csv                Export results to CSV\n"
    "  -h, --help           Show this help message\n\n"
    "Examples:\n"
    "  {}                    # Quick test with default settings\n"
    "  {} --full            # Full test with multiple task sizes\n"
    "  {} --scaling 32      # Scalability test up to 32 threads\n"
    "  {} -t 8 -s 500       # Test with 8 threads and task size 500\n",
    program_name,
    program_name,
    program_name,
    program_name,
    program_name );
}

// ===========================================================================
// Main Function
// ===========================================================================

int main( int const argc, char * argv[] )
{
  auto config = parse_config( argc, argv );

  if ( config.mode == RunMode::HELP )
  {
    print_help( argv[0] );
    return 0;
  }

  // Print header
  fmt::print(
    fg( fmt::color::light_blue ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                        SIMPLE THREAD POOL BENCHMARK                          ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  switch ( config.mode )
  {
    case RunMode::QUICK: run_quick_test( config.num_threads, config.task_size ); break;
    case RunMode::FULL: run_full_test( config.num_threads, config.task_size ); break;
    case RunMode::SCALING: run_scaling_test( config.num_threads, config.task_size ); break;
    default: break;
  }

  fmt::print(
    fg( fmt::color::lime_green ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                             BENCHMARK COMPLETE                               ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  fmt::print( fg( fmt::color::light_gray ), "All done folks!\n\n" );

  return 0;
}
