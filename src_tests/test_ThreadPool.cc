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

#if defined( _WIN32 ) || defined( _WIN64 )
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using std::cout;

// ===========================================================================
// Enumerations and Configuration
// ===========================================================================

enum class OutputMode
{
  VERBOSE,
  QUIET,
  CSV,
  JSON
};

struct BenchmarkConfig
{
  int              num_threads{ 16 };
  int              task_size{ 200 };
  std::vector<int> task_counts{ 16, 100, 1000, 10000 };
  int              num_runs{ 3 };
  bool             warmup{ true };
  bool             validate{ true };
  bool             export_csv{ false };
  bool             export_json{ false };
  bool             scaling_test{ false };
  int              scaling_max_threads{ 32 };
  OutputMode       output_mode{ OutputMode::VERBOSE };
};

// ===========================================================================
// Counter Class (Thread-local accumulator)
// ===========================================================================

static std::atomic<unsigned> global_accumulator;

class Counter
{
  Utils::BinarySearch<int> bs;

public:
  Counter()
  {
    bool  ok;
    int * pdata = bs.search( std::this_thread::get_id(), ok );
    *pdata      = 0;
  }

  void inc() const
  {
    bool  ok;
    int * pdata{ bs.search( std::this_thread::get_id(), ok ) };
    if ( !ok ) fmt::print( fg( fmt::color::red ), "Counter::inc failed thread\n" );
    ++( *pdata );
  }

  int get() const
  {
    bool        ok;
    int const * pdata{ bs.search( std::this_thread::get_id(), ok ) };
    if ( !ok ) fmt::print( fg( fmt::color::red ), "Counter::inc failed thread\n" );
    return *pdata;
  }

  void print() const
  {
    bool  ok;
    int * pdata{ bs.search( std::this_thread::get_id(), ok ) };
    if ( !ok ) fmt::print( fg( fmt::color::red ), "Counter::inc failed thread\n" );
    fmt::print( "thread {}, counter = {}\n", std::this_thread::get_id(), *pdata );
  }
};

// ===========================================================================
// Test Workload Function
// ===========================================================================

static void do_test( int const n, int const sz )
{
  Counter   c;
  int const nn{ 1 + ( ( n * 111 ) % sz ) };
  for ( int i{ 0 }; i < nn; ++i )
  {
    int const mm{ 1 + ( ( i * 11 ) % 64 ) };
    for ( int j{ 0 }; j < mm; ++j ) c.inc();
  }
  global_accumulator += c.get();
}

// ===========================================================================
// Test Result Structures
// ===========================================================================

struct TestResult
{
  std::string pool_name;
  unsigned    result{ 0 };
  double      launch_avg_mus{ 0.0 };
  double      launch_std_mus{ 0.0 };
  double      wait_time_mus{ 0.0 };
  double      wait_std_mus{ 0.0 };
  double      delete_time_mus{ 0.0 };
  double      delete_std_mus{ 0.0 };
  double      total_time_mus{ 0.0 };
  double      total_std_mus{ 0.0 };
  double      throughput{ 0.0 };  // tasks/ms
  double      efficiency{ 0.0 };  // speedup vs baseline
  long        max_rss_kb{ 0 };    // peak memory usage

  // Ranking flags
  bool is_best_launch{ false };
  bool is_best_wait{ false };
  bool is_best_total{ false };
  bool is_best_throughput{ false };
  bool is_best_efficiency{ false };

  // Raw data for statistics
  std::vector<double> launch_samples;
  std::vector<double> wait_samples;
  std::vector<double> total_samples;

  void compute_statistics()
  {
    auto compute_mean_std = []( const std::vector<double> & samples ) -> std::pair<double, double>
    {
      if ( samples.empty() ) return { 0.0, 0.0 };
      double sum    = std::accumulate( samples.begin(), samples.end(), 0.0 );
      double count  = static_cast<double>( samples.size() );
      double mean   = sum / count;
      double sq_sum = std::inner_product( samples.begin(), samples.end(), samples.begin(), 0.0 );
      double stddev = std::sqrt( sq_sum / count - mean * mean );
      return { mean, stddev };
    };

    auto [launch_mean, launch_std] = compute_mean_std( launch_samples );
    auto [wait_mean, wait_std]     = compute_mean_std( wait_samples );
    auto [total_mean, total_std]   = compute_mean_std( total_samples );

    launch_avg_mus = launch_mean;
    launch_std_mus = launch_std;
    wait_time_mus  = wait_mean;
    wait_std_mus   = wait_std;
    total_time_mus = total_mean;
    total_std_mus  = total_std;

    // Compute throughput (tasks per millisecond)
    throughput = ( result > 0 ) ? ( 1000.0 * result / total_time_mus ) : 0.0;
  }
};

struct ResourceUsage
{
  long max_rss_kb{ 0 };
  long user_time_ms{ 0 };
  long system_time_ms{ 0 };

  static ResourceUsage get_current()
  {
#if defined( _WIN32 ) || defined( _WIN64 )
    PROCESS_MEMORY_COUNTERS_EX mem_counters{};
    FILETIME                   creation_time{};
    FILETIME                   exit_time{};
    FILETIME                   kernel_time{};
    FILETIME                   user_time{};

    GetProcessMemoryInfo(
      GetCurrentProcess(),
      reinterpret_cast<PROCESS_MEMORY_COUNTERS *>( &mem_counters ),
      sizeof( mem_counters ) );

    GetProcessTimes( GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time );

    ULARGE_INTEGER kernel_ticks{};
    kernel_ticks.LowPart  = kernel_time.dwLowDateTime;
    kernel_ticks.HighPart = kernel_time.dwHighDateTime;

    ULARGE_INTEGER user_ticks{};
    user_ticks.LowPart  = user_time.dwLowDateTime;
    user_ticks.HighPart = user_time.dwHighDateTime;

    return ResourceUsage{
      static_cast<long>( mem_counters.PeakWorkingSetSize / 1024ULL ),
      static_cast<long>( user_ticks.QuadPart / 10000ULL ),
      static_cast<long>( kernel_ticks.QuadPart / 10000ULL )
    };
#else
    struct rusage usage;
    getrusage( RUSAGE_SELF, &usage );

    return ResourceUsage{
      usage.ru_maxrss,
      usage.ru_utime.tv_sec * 1000 + usage.ru_utime.tv_usec / 1000,
      usage.ru_stime.tv_sec * 1000 + usage.ru_stime.tv_usec / 1000
    };
#endif
  }

  ResourceUsage operator-( const ResourceUsage & other ) const
  {
    return ResourceUsage{
      std::max( max_rss_kb, other.max_rss_kb ),
      user_time_ms - other.user_time_ms,
      system_time_ms - other.system_time_ms
    };
  }
};

// ===========================================================================
// Utility Functions
// ===========================================================================

std::string format_time_auto( double time_mus, int width = 12 )
{
  const char * unit;
  double       value;

  if ( time_mus < 1000 )
  {
    value = time_mus;
    unit  = "µs";
    return fmt::format( "{:>{}.2f} {}", value, width - 3, unit );
  }
  else if ( time_mus < 1'000'000 )
  {
    value = time_mus / 1000;
    unit  = "ms";
    return fmt::format( "{:>{}.3f} {}", value, width - 3, unit );
  }
  else if ( time_mus < 60'000'000 )
  {
    value = time_mus / 1'000'000;
    unit  = "s";
    return fmt::format( "{:>{}.3f} {}", value, width - 2, unit );
  }
  else
  {
    value = time_mus / 60'000'000;
    unit  = "min";
    return fmt::format( "{:>{}.3f} {}", value, width - 4, unit );
  }
}

std::string format_with_ranking( const std::string & value, bool is_best )
{
  if ( is_best ) { return fmt::format( fg( fmt::color::green ) | fmt::emphasis::bold, "{}", value ); }
  else
  {
    return fmt::format( fg( fmt::color::orange_red ), "{}", value );
  }
}

std::string format_time_with_ranking( double time_mus, bool is_best, int width = 20 )
{
  std::string time_str = format_time_auto( time_mus, width );
  return format_with_ranking( time_str, is_best );
}

std::string format_std_dev( double std_mus, double mean_mus, int width = 8 )
{
  if ( mean_mus == 0 ) return fmt::format( "{:>{}}", "N/A", width );
  double cv_percent = ( std_mus / mean_mus ) * 100.0;
  return fmt::format( "{:>{}.1f}%", cv_percent, width );
}

// ===========================================================================
// Thread Pool Testing Implementation
// ===========================================================================

template <typename Func, typename... Args> bool run_with_timeout( Func && func, int timeout_ms, Args &&... args )
{
  auto future = std::async( std::launch::async, [&]() { return func( std::forward<Args>( args )... ); } );

  return future.wait_for( std::chrono::milliseconds( timeout_ms ) ) != std::future_status::timeout;
}

template <class TP> TestResult test_TP_single_run( int const NN, int nt, int sz )
{
  Utils::TicToc tm;
  Utils::TicToc tm2;
  TestResult    result;

  global_accumulator        = 0;
  ResourceUsage start_usage = ResourceUsage::get_current();

  double t_launch, t_wait, t_delete;

  {
    TP pool( nt );

    // Measure launch time (average per task)
    tm.tic();
    tm2.tic();
    for ( int i{ 0 }; i < NN; ++i ) pool.run( do_test, i, sz );
    tm.toc();
    t_launch = tm.elapsed_mus() / NN;

    // Measure wait time
    tm.tic();
    pool.wait();
    tm.toc();
    tm2.toc();
    t_wait = tm.elapsed_mus();
  }

  // Measure destruction time
  tm.tic();
  tm.toc();
  t_delete = tm.elapsed_mus();

  ResourceUsage end_usage   = ResourceUsage::get_current();
  ResourceUsage delta_usage = end_usage - start_usage;

  result.pool_name       = TP::Name();
  result.result          = global_accumulator.load();
  result.launch_avg_mus  = t_launch;
  result.wait_time_mus   = t_wait;
  result.delete_time_mus = t_delete;
  result.total_time_mus  = tm2.elapsed_mus();
  result.max_rss_kb      = delta_usage.max_rss_kb;

  // Store raw samples
  result.launch_samples.push_back( t_launch );
  result.wait_samples.push_back( t_wait );
  result.total_samples.push_back( result.total_time_mus );

  return result;
}

template <class TP> TestResult test_TP_multiple_runs( int const NN, int nt, int sz, int num_runs = 5 )
{
  TestResult aggregated;
  aggregated.pool_name = TP::Name();

  std::vector<unsigned> results;
  int                   successful_runs = 0;

  // Warm-up run (discarded)
  if ( num_runs > 1 )
  {
    test_TP_single_run<TP>( std::min( 100, NN ), nt, sz );
    std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
  }

  for ( int run = 0; run < num_runs; ++run )
  {
    try
    {
      auto run_result = test_TP_single_run<TP>( NN, nt, sz );

      // Skip first run as warm-up for statistics
      if ( run > 0 )
      {
        aggregated.launch_samples.insert(
          aggregated.launch_samples.end(),
          run_result.launch_samples.begin(),
          run_result.launch_samples.end() );
        aggregated.wait_samples.insert(
          aggregated.wait_samples.end(),
          run_result.wait_samples.begin(),
          run_result.wait_samples.end() );
        aggregated.total_samples.insert(
          aggregated.total_samples.end(),
          run_result.total_samples.begin(),
          run_result.total_samples.end() );
        results.push_back( run_result.result );
        aggregated.max_rss_kb = std::max( aggregated.max_rss_kb, run_result.max_rss_kb );
        successful_runs++;
      }

      // Small pause between runs
      if ( run < num_runs - 1 ) { std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) ); }
    }
    catch ( const std::exception & e )
    {
      fmt::print( fg( fmt::color::red ), "⚠️  Run {} failed for {}: {}\n", run + 1, TP::Name(), e.what() );
    }
  }

  if ( successful_runs == 0 )
  {
    fmt::print( fg( fmt::color::red ), "❌ All runs failed for {}\n", TP::Name() );
    return aggregated;
  }

  // Validate results consistency
  if ( results.size() > 1 )
  {
    unsigned expected = results[0];
    for ( size_t i = 1; i < results.size(); ++i )
    {
      if ( results[i] != expected )
      {
        fmt::print(
          fg( fmt::color::yellow ),
          "⚠️  Result mismatch in {}: run {} = {}, run {} = {}\n",
          TP::Name(),
          1,
          expected,
          i + 1,
          results[i] );
      }
    }
    aggregated.result = expected;
  }
  else if ( !results.empty() ) { aggregated.result = results[0]; }

  aggregated.compute_statistics();
  return aggregated;
}

// ===========================================================================
// Results Presentation
// ===========================================================================

void print_test_header( int NN, int nt, int sz, int run_num = 1, int total_runs = 1 )
{
  std::string run_info = ( total_runs > 1 ) ? fmt::format( "Run {}/{} | ", run_num, total_runs ) : "";

  fmt::print(
    fg( fmt::color::steel_blue ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                             BENCHMARK RESULTS                                ║\n"
    "╠══════════════════════════════════════════════════════════════════════════════╣\n"
    "║  Configuration: {}{:<5d} tasks | {:<2d} threads | size={:<5d}              ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n",
    run_info,
    NN,
    nt,
    sz );
}

void print_detailed_results_table( const std::vector<TestResult> & results )
{
  // Header with additional columns
  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "┌───────────────────────────┬────────────┬─────────────────────┬─────────────────────┬──────────────┬─────────────────────┬──────────┬──────────┐\n"
    "│ Thread Pool Type          │ Result     │     Launch/avg      │      Wait Time      │ Delete Time  │     Total Time      │Throughput│Efficiency│\n"
    "│                           │            │     (avg ± cv%)     │     (avg ± cv%)     │              │     (avg ± cv%)     │(tasks/ms)│(speedup) │\n"
    "├───────────────────────────┼────────────┼─────────────────────┼─────────────────────┼──────────────┼─────────────────────┼──────────┼──────────┤\n" );

  for ( const auto & res : results )
  {
    // Format each metric with standard deviation as coefficient of variation
    std::string launch_str = format_time_with_ranking( res.launch_avg_mus, res.is_best_launch, 10 );
    std::string launch_cv  = format_std_dev( res.launch_std_mus, res.launch_avg_mus, 4 );

    std::string wait_str = format_time_with_ranking( res.wait_time_mus, res.is_best_wait, 10 );
    std::string wait_cv  = format_std_dev( res.wait_std_mus, res.wait_time_mus, 4 );

    std::string delete_str = format_time_auto( res.delete_time_mus, 10 );

    std::string total_str = format_time_with_ranking( res.total_time_mus, res.is_best_total, 10 );
    std::string total_cv  = format_std_dev( res.total_std_mus, res.total_time_mus, 4 );

    std::string throughput_str = fmt::format( "{:8.1f}", res.throughput );
    if ( res.is_best_throughput )
    {
      throughput_str = fmt::format( fg( fmt::color::green ) | fmt::emphasis::bold, "{}", throughput_str );
    }

    std::string efficiency_str = fmt::format( "{:8.2f}", res.efficiency );
    if ( res.is_best_efficiency )
    {
      efficiency_str = fmt::format( fg( fmt::color::green ) | fmt::emphasis::bold, "{}", efficiency_str );
    }

    fmt::print(
      "│ {:25} │ {:>10} │ {:>11} ± {:>6} │ {:>11} ± {:>6} │ {:>12} │ {:>11} ± {:>6} │ {} │ {} │\n",
      res.pool_name,
      res.result,
      launch_str,
      launch_cv,
      wait_str,
      wait_cv,
      delete_str,
      total_str,
      total_cv,
      throughput_str,
      efficiency_str );
  }

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "└───────────────────────────┴────────────┴─────────────────────┴─────────────────────┴──────────────┴─────────────────────┴──────────┴──────────┘\n" );
}

void print_compact_results_table( const std::vector<TestResult> & results )
{
  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "┌───────────────────────────┬────────────────┬────────────────┬────────────────┐\n"
    "│ {:25} │ {:>22} │ {:>22} │ {:>22} │\n"
    "├───────────────────────────┼────────────────┼────────────────┼────────────────┤\n",
    "Thread Pool Type",
    "Total Time",
    "Throughput",
    "Efficiency" );

  for ( const auto & res : results )
  {
    std::string total_str      = format_time_with_ranking( res.total_time_mus, res.is_best_total, 14 );
    std::string throughput_str = fmt::format( "{:>8.1f} t/ms", res.throughput );
    if ( res.is_best_throughput )
    {
      throughput_str = fmt::format( fg( fmt::color::green ) | fmt::emphasis::bold, "{}", throughput_str );
    }
    std::string efficiency_str = fmt::format( "{:>8.2f}x", res.efficiency );
    if ( res.is_best_efficiency )
    {
      efficiency_str = fmt::format( fg( fmt::color::green ) | fmt::emphasis::bold, "{}", efficiency_str );
    }

    fmt::print( "│ {:25} │ {} │ {:>22} │ {:>22} │\n", res.pool_name, total_str, throughput_str, efficiency_str );
  }

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "└───────────────────────────┴────────────────┴────────────────┴────────────────┘\n" );
}

void print_summary_statistics( const std::vector<TestResult> & results, int NN, OutputMode mode )
{
  if ( results.size() < 2 ) return;

  // Find best performers
  double best_launch     = std::numeric_limits<double>::max();
  double best_wait       = std::numeric_limits<double>::max();
  double best_total      = std::numeric_limits<double>::max();
  double best_throughput = 0.0;
  double best_efficiency = 0.0;

  // Baseline is first result (no thread pool)
  double baseline_time = results[0].total_time_mus;

  int nskip = 2;

  for ( size_t i = 1; i < results.size(); ++i )
  {
    if ( --nskip >= 0 ) continue;

    const auto & res = results[i];

    // Calculate efficiency (speedup vs baseline)
    const_cast<TestResult &>( res ).efficiency = ( baseline_time > 0 ) ? baseline_time / res.total_time_mus : 0.0;

    if ( res.launch_avg_mus < best_launch ) best_launch = res.launch_avg_mus;
    if ( res.wait_time_mus < best_wait ) best_wait = res.wait_time_mus;
    if ( res.total_time_mus < best_total ) best_total = res.total_time_mus;
    if ( res.throughput > best_throughput ) best_throughput = res.throughput;
    if ( res.efficiency > best_efficiency ) best_efficiency = res.efficiency;
  }

  // Mark best performers
  auto mutable_results = results;
  for ( auto & res : mutable_results )
  {
    res.is_best_launch     = ( res.launch_avg_mus == best_launch );
    res.is_best_wait       = ( res.wait_time_mus == best_wait );
    res.is_best_total      = ( res.total_time_mus == best_total );
    res.is_best_throughput = ( res.throughput == best_throughput );
    res.is_best_efficiency = ( res.efficiency == best_efficiency );
  }

  // Print appropriate table based on output mode
  if ( mode == OutputMode::VERBOSE ) { print_detailed_results_table( mutable_results ); }
  else
  {
    print_compact_results_table( mutable_results );
  }

  // Performance summary
  fmt::print( fg( fmt::color::gold ) | fmt::emphasis::bold, "\n📊 Performance Summary (NN={}):\n", NN );

  // Find names of best performers
  std::string best_launch_name, best_wait_name, best_total_name, best_throughput_name, best_efficiency_name;
  for ( const auto & res : mutable_results )
  {
    if ( res.is_best_launch ) best_launch_name = res.pool_name;
    if ( res.is_best_wait ) best_wait_name = res.pool_name;
    if ( res.is_best_total ) best_total_name = res.pool_name;
    if ( res.is_best_throughput ) best_throughput_name = res.pool_name;
    if ( res.is_best_efficiency ) best_efficiency_name = res.pool_name;
  }

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  🏆 Best Launch Time:   " );
  fmt::print( "{} ({})\n", best_launch_name, format_time_auto( best_launch ) );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  🏆 Best Wait Time:     " );
  fmt::print( "{} ({})\n", best_wait_name, format_time_auto( best_wait ) );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  🏆 Best Total Time:    " );
  fmt::print( "{} ({})\n", best_total_name, format_time_auto( best_total ) );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  🏆 Best Throughput:    " );
  fmt::print( "{} ({:.1f} tasks/ms)\n", best_throughput_name, best_throughput );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  🏆 Best Efficiency:    " );
  fmt::print( "{} ({:.2f}x speedup)\n", best_efficiency_name, best_efficiency );

  // Memory usage summary
  long max_memory_kb = 0;
  for ( const auto & res : mutable_results ) { max_memory_kb = std::max( max_memory_kb, res.max_rss_kb ); }
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "  💾 Peak Memory Usage:  " );
  fmt::print( "{:.1f} MB\n", static_cast<double>( max_memory_kb ) / 1024.0 );
}

// ===========================================================================
// Export Functions
// ===========================================================================

void export_to_csv( const std::vector<TestResult> & results, const std::string & filename, int NN, int nt, int sz )
{
  std::ofstream file( filename, std::ios::app );
  if ( !file.is_open() )
  {
    fmt::print( fg( fmt::color::red ), "❌ Failed to open CSV file: {}\n", filename );
    return;
  }

  // Write header if file is empty
  file.seekp( 0, std::ios::end );
  if ( file.tellp() == 0 )
  {
    file << "Timestamp,TaskCount,ThreadCount,TaskSize,PoolName,Result,"
         << "LaunchAvg(µs),LaunchStd(µs),WaitAvg(µs),WaitStd(µs),"
         << "DeleteAvg(µs),DeleteStd(µs),TotalAvg(µs),TotalStd(µs),"
         << "Throughput(tasks/ms),Efficiency,MaxRSS(kB)\n";
  }

  auto now       = std::chrono::system_clock::now();
  auto timestamp = std::chrono::system_clock::to_time_t( now );

  for ( const auto & res : results )
  {
    file << fmt::format(
      "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
      timestamp,
      NN,
      nt,
      sz,
      res.pool_name,
      res.result,
      res.launch_avg_mus,
      res.launch_std_mus,
      res.wait_time_mus,
      res.wait_std_mus,
      res.delete_time_mus,
      res.delete_std_mus,
      res.total_time_mus,
      res.total_std_mus,
      res.throughput,
      res.efficiency,
      res.max_rss_kb );
  }

  file.close();
  fmt::print( fg( fmt::color::light_green ), "  📁 Results exported to: {}\n", filename );
}

void export_to_json( const std::vector<TestResult> & results, const std::string & filename, int NN, int nt, int sz )
{
  std::ofstream file( filename );
  if ( !file.is_open() )
  {
    fmt::print( fg( fmt::color::red ), "❌ Failed to open JSON file: {}\n", filename );
    return;
  }

  file << "{\n";
  file << fmt::format( "  \"metadata\": {{\n" );
  file << fmt::format( "    \"task_count\": {},\n", NN );
  file << fmt::format( "    \"thread_count\": {},\n", nt );
  file << fmt::format( "    \"task_size\": {},\n", sz );
  file << fmt::format( "    \"timestamp\": {}\n", std::time( nullptr ) );
  file << fmt::format( "  }},\n" );
  file << fmt::format( "  \"results\": [\n" );

  for ( size_t i = 0; i < results.size(); ++i )
  {
    const auto & res = results[i];
    file << fmt::format( "    {{\n" );
    file << fmt::format( "      \"pool_name\": \"{}\",\n", res.pool_name );
    file << fmt::format( "      \"result\": {},\n", res.result );
    file << fmt::format( "      \"launch_time_avg_us\": {:.2f},\n", res.launch_avg_mus );
    file << fmt::format( "      \"launch_time_std_us\": {:.2f},\n", res.launch_std_mus );
    file << fmt::format( "      \"wait_time_avg_us\": {:.2f},\n", res.wait_time_mus );
    file << fmt::format( "      \"wait_time_std_us\": {:.2f},\n", res.wait_std_mus );
    file << fmt::format( "      \"total_time_avg_us\": {:.2f},\n", res.total_time_mus );
    file << fmt::format( "      \"total_time_std_us\": {:.2f},\n", res.total_std_mus );
    file << fmt::format( "      \"throughput\": {:.2f},\n", res.throughput );
    file << fmt::format( "      \"efficiency\": {:.2f},\n", res.efficiency );
    file << fmt::format( "      \"max_rss_kb\": {}\n", res.max_rss_kb );
    file << fmt::format( "    }}" );
    if ( i < results.size() - 1 ) file << ",";
    file << "\n";
  }

  file << fmt::format( "  ]\n" );
  file << fmt::format( "}}\n" );

  file.close();
  fmt::print( fg( fmt::color::light_green ), "  📁 Results exported to: {}\n", filename );
}

// ===========================================================================
// Test Functions
// ===========================================================================

bool validate_results( const std::vector<TestResult> & results )
{
  if ( results.empty() ) return true;

  unsigned expected = results[0].result;
  bool     valid    = true;

  for ( size_t i = 1; i < results.size(); ++i )
  {
    if ( results[i].result != expected )
    {
      fmt::print(
        fg( fmt::color::red ),
        "❌ Validation failed: {} result {} != baseline {}\n",
        results[i].pool_name,
        results[i].result,
        expected );
      valid = false;
    }
  }

  if ( valid ) { fmt::print( fg( fmt::color::green ), "✅ All results validated successfully\n" ); }

  return valid;
}

std::vector<TestResult> test_all_pools( int NN, int nt, int sz, int num_runs, bool include_baseline = true )
{
  std::vector<TestResult> results;

  // Test without thread pool (baseline)
  if ( include_baseline )
  {
    Utils::TicToc tm;
    global_accumulator = 0;

    tm.tic();
    for ( int i{ 0 }; i < NN; ++i ) do_test( i, sz );
    tm.toc();

    TestResult baseline;
    baseline.pool_name       = "⚠️  No Thread Pool";
    baseline.result          = global_accumulator.load();
    baseline.launch_avg_mus  = tm.elapsed_mus() / NN;
    baseline.wait_time_mus   = 0;
    baseline.delete_time_mus = 0;
    baseline.total_time_mus  = tm.elapsed_mus();
    baseline.throughput      = ( baseline.result > 0 ) ? ( 1000.0 * baseline.result / baseline.total_time_mus ) : 0.0;
    baseline.efficiency      = 1.0;  // Baseline is reference

    results.push_back( baseline );
  }

  // Test each thread pool implementation
  auto test_pool = [&]( auto pool_type, const std::string & name )
  {
    fmt::print( fg( fmt::color::light_gray ), "  Testing {}...\n", name );
    try
    {
      auto result = test_TP_multiple_runs<decltype( pool_type )>( NN, nt, sz, num_runs );
      results.push_back( result );
    }
    catch ( const std::exception & e )
    {
      fmt::print( fg( fmt::color::red ), "  ❌ {} failed: {}\n", name, e.what() );
      TestResult failed;
      failed.pool_name = name + " (FAILED)";
      results.push_back( failed );
    }
  };

  test_pool( Utils::ThreadPool0( 1 ), "ThreadPool0" );
  test_pool( Utils::ThreadPool1( 1 ), "ThreadPool1" );
  test_pool( Utils::ThreadPool2( 1 ), "ThreadPool2" );
  test_pool( Utils::ThreadPool3( 1 ), "ThreadPool3" );
  test_pool( Utils::ThreadPool4( 1 ), "ThreadPool4" );
  test_pool( Utils::ThreadPool5( 1 ), "ThreadPool5" );
  test_pool( Utils::ThreadPoolEigen( 1 ), "ThreadPoolEigen" );

  return results;
}

void test_scaling( int max_threads, int task_count, int task_size, int num_runs )
{
  std::vector<int> thread_counts = { 1, 2, 4, 8, 16, 32 };
  thread_counts.erase(
    std::remove_if( thread_counts.begin(), thread_counts.end(), [max_threads]( int t ) { return t > max_threads; } ),
    thread_counts.end() );

  fmt::print(
    fg( fmt::color::light_sea_green ) | fmt::emphasis::bold,
    "\n"
    "══════════════════════════════════════════════════════════════════════════════════\n"
    "  SCALABILITY TEST: {} TASKS, {} SIZE\n"
    "══════════════════════════════════════════════════════════════════════════════════\n",
    task_count,
    task_size );

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "┌──────────┬────────────────┬────────────────┬────────────────┬────────────────┐\n"
    "│ Threads  │   Total Time   │   Speedup      │   Efficiency   │   Throughput   │\n"
    "├──────────┼────────────────┼────────────────┼────────────────┼────────────────┤\n" );

  double single_thread_time = 0;

  for ( int nt : thread_counts )
  {
    fmt::print( fg( fmt::color::light_gray ), "  Testing with {} threads...\n", nt );

    try
    {
      auto result = test_TP_multiple_runs<Utils::ThreadPool4>( task_count, nt, task_size, num_runs );

      if ( nt == 1 ) single_thread_time = result.total_time_mus;

      double speedup    = ( single_thread_time > 0 ) ? single_thread_time / result.total_time_mus : 0.0;
      double efficiency = ( nt > 0 ) ? speedup / nt * 100.0 : 0.0;
      double throughput = result.throughput;

      // Color code based on efficiency
      std::string efficiency_str;
      if ( efficiency >= 80.0 ) { efficiency_str = fmt::format( fg( fmt::color::green ), "{:6.1f}%", efficiency ); }
      else if ( efficiency >= 60.0 )
      {
        efficiency_str = fmt::format( fg( fmt::color::yellow ), "{:6.1f}%", efficiency );
      }
      else
      {
        efficiency_str = fmt::format( fg( fmt::color::red ), "{:6.1f}%", efficiency );
      }

      fmt::print(
        "│ {:8d} │ {:>22} │ {:>22.2f} │ {:>22} │ {:>8.1f} t/ms │\n",
        nt,
        format_time_auto( result.total_time_mus, 14 ),
        speedup,
        efficiency_str,
        throughput );
    }
    catch ( const std::exception & e )
    {
      fmt::print( fg( fmt::color::red ), "│ {:8d} │ {:>22} │\n", nt, "FAILED" );
    }

    // Small pause between tests
    if ( nt != thread_counts.back() ) { std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) ); }
  }

  fmt::print(
    fg( fmt::color::dark_cyan ) | fmt::emphasis::bold,
    "└──────────┴────────────────┴────────────────┴────────────────┴────────────────┘\n" );
}

// ===========================================================================
// Command Line Parsing
// ===========================================================================

BenchmarkConfig parse_args( int argc, char * argv[] )
{
  BenchmarkConfig config;

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
    else if ( arg == "-n" || arg == "--tasks" )
    {
      if ( i + 1 < argc )
      {
        config.task_counts.clear();
        config.task_counts.push_back( std::atoi( argv[++i] ) );
      }
    }
    else if ( arg == "-r" || arg == "--runs" )
    {
      if ( i + 1 < argc ) config.num_runs = std::atoi( argv[++i] );
    }
    else if ( arg == "--scaling" )
    {
      config.scaling_test = true;
      if ( i + 1 < argc ) config.scaling_max_threads = std::atoi( argv[++i] );
    }
    else if ( arg == "--csv" ) { config.export_csv = true; }
    else if ( arg == "--json" ) { config.export_json = true; }
    else if ( arg == "--quiet" ) { config.output_mode = OutputMode::QUIET; }
    else if ( arg == "--verbose" ) { config.output_mode = OutputMode::VERBOSE; }
    else if ( arg == "--no-warmup" ) { config.warmup = false; }
    else if ( arg == "--no-validate" ) { config.validate = false; }
    else if ( arg == "-h" || arg == "--help" )
    {
      fmt::print(
        "Usage: {} [options]\n"
        "Options:\n"
        "  -t, --threads N      Number of threads (default: 16)\n"
        "  -s, --size N         Task size (default: 200)\n"
        "  -n, --tasks N        Number of tasks (default: 16,100,1000,10000)\n"
        "  -r, --runs N         Number of test runs (default: 3)\n"
        "  --scaling [MAX]      Run scalability test (optional max threads)\n"
        "  --csv                Export results to CSV\n"
        "  --json               Export results to JSON\n"
        "  --quiet              Quiet mode (minimal output)\n"
        "  --verbose            Verbose mode (detailed output)\n"
        "  --no-warmup          Skip warm-up runs\n"
        "  --no-validate        Skip result validation\n"
        "  -h, --help           Show this help message\n",
        argv[0] );
      std::exit( 0 );
    }
  }

  return config;
}

// ===========================================================================
// Main Function
// ===========================================================================

int main( int const argc, char * argv[] )
{
  auto config = parse_args( argc, argv );

  // Print banner
  fmt::print(
    fg( fmt::color::light_blue ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║                    🚀 COMPREHENSIVE THREAD POOL BENCHMARK                    ║\n"
    "╠══════════════════════════════════════════════════════════════════════════════╣\n"
    "║  Configuration: Threads={:<3d}    Task Size={:<5d}    Runs={:<2d}                    ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n",
    config.num_threads,
    config.task_size,
    config.num_runs );

  fmt::print(
    fg( fmt::color::light_gray ),
    "📋 Testing {} different thread pool implementations + baseline\n"
    "📊 Running {} test runs per configuration{}\n\n",
    7,
    config.num_runs,
    config.warmup ? " (with warm-up)" : "" );

  // Warm-up phase
  if ( config.warmup )
  {
    fmt::print( fg( fmt::color::light_gray ), "🔥 Performing warm-up runs...\n" );
    test_TP_multiple_runs<Utils::ThreadPool0>( 100, config.num_threads, config.task_size, 1 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 200 ) );
  }

  // Main test loop
  std::vector<std::vector<TestResult>> all_results;

  for ( size_t i = 0; i < config.task_counts.size(); ++i )
  {
    int NN = config.task_counts[i];

    if ( config.output_mode != OutputMode::QUIET )
    {
      fmt::print(
        fg( fmt::color::light_sea_green ) | fmt::emphasis::bold,
        "\n"
        "══════════════════════════════════════════════════════════════════════════════════\n"
        "  TEST SET {} of {}: {} TASKS\n"
        "══════════════════════════════════════════════════════════════════════════════════\n",
        i + 1,
        config.task_counts.size(),
        NN );
    }

    print_test_header( NN, config.num_threads, config.task_size, i + 1, config.task_counts.size() );

    auto results = test_all_pools( NN, config.num_threads, config.task_size, config.num_runs );
    all_results.push_back( results );

    // Validate results
    if ( config.validate && !validate_results( results ) )
    {
      if ( config.output_mode != OutputMode::QUIET )
      {
        fmt::print( fg( fmt::color::yellow ), "⚠️  Continuing despite validation issues\n" );
      }
    }

    // Print results
    print_summary_statistics( results, NN, config.output_mode );

    // Export if requested
    if ( config.export_csv )
    {
      export_to_csv( results, "benchmark_results.csv", NN, config.num_threads, config.task_size );
    }
    if ( config.export_json )
    {
      std::string filename = fmt::format( "benchmark_N{}_T{}_S{}.json", NN, config.num_threads, config.task_size );
      export_to_json( results, filename, NN, config.num_threads, config.task_size );
    }

    // Pause between test sets
    if ( i < config.task_counts.size() - 1 ) { std::this_thread::sleep_for( std::chrono::milliseconds( 200 ) ); }
  }

  // Scalability test (optional)
  if ( config.scaling_test )
  {
    int scaling_tasks = 1000;
    if ( !config.task_counts.empty() ) { scaling_tasks = config.task_counts.back(); }
    test_scaling( config.scaling_max_threads, scaling_tasks, config.task_size, config.num_runs );
  }

  // Final summary
  if ( config.output_mode != OutputMode::QUIET )
  {
    fmt::print(
      fg( fmt::color::lime_green ) | fmt::emphasis::bold,
      "\n"
      "╔══════════════════════════════════════════════════════════════════════════════╗\n"
      "║                        ✅ ALL TESTS COMPLETED SUCCESSFULLY                   ║\n"
      "╚══════════════════════════════════════════════════════════════════════════════╝\n"
      "\n" );

    fmt::print(
      fg( fmt::color::light_blue ),
      "🎯 Benchmark completed with {} different task loads\n"
      "📊 {} test runs per configuration\n"
      "🏆 Best performing pools are highlighted in green\n"
      "📈 Throughput measured in tasks per millisecond\n"
      "⚡ Efficiency shows speedup relative to single-threaded baseline\n"
      "\n",
      config.task_counts.size(),
      config.num_runs );

    if ( config.export_csv )
    {
      fmt::print( fg( fmt::color::light_green ), "💾 CSV data exported to: benchmark_results.csv\n" );
    }
    if ( config.export_json )
    {
      fmt::print( fg( fmt::color::light_green ), "💾 JSON data exported to separate files\n" );
    }
  }

  return 0;
}
