/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2026                                                      |
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

#include "Utils_search_intervals.hh"
#include "Utils_search_intervals2.hh"
#include "Utils_TicToc.hh"
#include "Utils_fmt.hh"
#include "Utils_string.hh"

#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>
#include <numeric>
#include <set>
#include <map>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wc++98-compat"
#endif

using namespace std;
using namespace Utils;

using integer   = int;
using real_type = double;

using Utils::SearchInterval;

// ============================================================================
// OUTPUT UTILITIES
// ============================================================================

void print_header( const string & title, fmt::color color = fmt::color::cyan )
{
  fmt::print(
    fg( color ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║{:^78}║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n",
    title );
}

void print_section( const string & subtitle )
{
  fmt::print( fg( fmt::color::blue ) | fmt::emphasis::bold, "\n┌─ {} ─\n", subtitle );
}

void print_success( const string & message )
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "✅  {}\n", message );
}

void print_error( const string & message )
{
  fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "❌  {}\n", message );
}

void print_warning( const string & message )
{
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "⚠️  {}\n", message );
}

void print_info( const string & message )
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "ℹ️  {}\n", message );
}

void print_progress( const string & message )
{
  fmt::print( fg( fmt::color::light_blue ), "   [{}]\n", message );
}

// ============================================================================
// ENHANCED TEST RUNNER
// ============================================================================

class EnhancedTestRunner
{
private:
  struct TestResult
  {
    string test_name;
    bool   function_correct = false;
    bool   class_correct    = false;
    double time_function_us = 0.0;
    double time_class_us    = 0.0;
    string function_error;
    string class_error;
    string note;
  };

  vector<TestResult> test_results;
  atomic<int>        total_tests{ 0 };
  atomic<int>        passed_tests{ 0 };
  atomic<int>        failed_tests{ 0 };

public:
  void printHeader( const string & title ) { print_header( title, fmt::color::light_blue ); }

  void printTestStart( const string & test_name ) { fmt::print( "  ├─ {} ... ", test_name ); }

  TestResult & runDualTest(
    const string & test_name,
    function<void( integer npts, const real_type * X, real_type x, integer & last, bool closed, bool can_extend )>
                                                                                                  test_function,
    function<void( SearchInterval<real_type, integer> & search, pair<integer, real_type> & res )> test_class,
    const string &                                                                                note = "" )
  {
    total_tests++;
    TestResult result;
    result.test_name = test_name;
    result.note      = note;

    try
    {
      // Common setup
      vector<real_type> X          = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      integer           n          = static_cast<integer>( X.size() );
      bool              closed     = false;
      bool              can_extend = true;

      // Test function implementation
      integer function_last = 0;
      TicToc  tm;
      tm.tic();
      real_type x_function = 2.5;
      test_function( n, X.data(), x_function, function_last, closed, can_extend );
      tm.toc();
      result.time_function_us = tm.elapsed_ns();
      result.function_correct = true;

      // Test class implementation
      string                             name  = "test";
      real_type *                        X_ptr = X.data();
      SearchInterval<real_type, integer> search_class;
      search_class.setup( &name, &n, &X_ptr, &closed, &can_extend );

      tm.tic();
      pair<integer, real_type> class_result = { 0, 2.5 };
      test_class( search_class, class_result );
      tm.toc();
      result.time_class_us = tm.elapsed_ns();
      result.class_correct = true;

      // Verify consistency
      if ( function_last != class_result.first )
      {
        result.note +=
          fmt::format( " | Discrepancy: function_idx={}, class_idx={}", function_last, class_result.first );
        result.function_correct = false;
        result.class_correct    = false;
      }
    }
    catch ( const exception & e )
    {
      result.note = string( "Error: " ) + e.what();
    }

    test_results.push_back( result );

    // Print immediate result
    if ( result.function_correct && result.class_correct )
    {
      fmt::print( fg( fmt::color::green ), "✓✓\n" );
      passed_tests++;
    }
    else
    {
      fmt::print( fg( fmt::color::red ), "✗\n" );
      failed_tests++;
      if ( !result.note.empty() ) { fmt::print( fg( fmt::color::yellow ), "      Note: {}\n", result.note ); }
    }

    return test_results.back();
  }

  void printSummaryTable()
  {
    fmt::print( "\n" );
    fmt::print(
      fg( fmt::color::light_blue ),
      "┌────────────────────────────────────────────────────────────────────────────────────┐\n"
      "│                                 TEST SUMMARY                                       │\n"
      "├────────────────────────────────────────────────────────────────────────────────────┤\n"
      "│ {:<30} │ {:<10} │ {:<10} │ {:<12} │ {:<12} │\n"
      "├────────────────────────────────────────────────────────────────────────────────────┤\n",
      "Test",
      "Function",
      "Class",
      "Time Func",
      "Time Class" );

    for ( const auto & res : test_results )
    {
      auto function_status = res.function_correct ? fmt::format( fg( fmt::color::green ), "      ✓" )
                                                  : fmt::format( fg( fmt::color::red ), "      ✗" );

      auto class_status = res.class_correct ? fmt::format( fg( fmt::color::green ), "      ✓" )
                                            : fmt::format( fg( fmt::color::red ), "      ✗" );

      fmt::print(
        "│ {:<30} │ {} │ {} │ {:>9.2f} μs │ {:>9.2f} μs │\n",
        res.test_name,
        function_status,
        class_status,
        res.time_function_us,
        res.time_class_us );
    }

    fmt::print(
      fg( fmt::color::light_blue ),
      "└────────────────────────────────────────────────────────────────────────────────────┘\n" );
  }

  void printPerformanceComparison()
  {
    fmt::print( "\n" );
    fmt::print(
      fg( fmt::color::cyan ),
      "╔══════════════════════════════════════════════════════════════════════════════════╗\n"
      "║                           PERFORMANCE COMPARISON                                 ║\n"
      "╠══════════════════════════════════════════════════════════════════════════════════╣\n" );

    double total_time_function = 0.0;
    double total_time_class    = 0.0;
    int    correct_function    = 0;
    int    correct_class       = 0;

    for ( const auto & res : test_results )
    {
      total_time_function += res.time_function_us;
      total_time_class += res.time_class_us;
      if ( res.function_correct ) correct_function++;
      if ( res.class_correct ) correct_class++;
    }

    double avg_speedup       = total_time_function / total_time_class;
    double accuracy_function = 100.0 * correct_function / test_results.size();
    double accuracy_class    = 100.0 * correct_class / test_results.size();

    fmt::print( "  Total time function implementation: {:>10.2f} μs\n", total_time_function );
    fmt::print( "  Total time class implementation: {:>10.2f} μs\n", total_time_class );
    fmt::print( "  Average speedup: {:>10.2f}x\n", avg_speedup );

    if ( avg_speedup > 3.0 ) { fmt::print( fg( fmt::color::green ), "  🚀 EXCELLENT PERFORMANCE!\n" ); }
    else if ( avg_speedup > 1.5 ) { fmt::print( fg( fmt::color::yellow ), "  ⚡ Significant improvement\n" ); }
    else if ( avg_speedup > 1.0 ) { fmt::print( "  ↗️  Slight improvement\n" ); }
    else
    {
      fmt::print( fg( fmt::color::red ), "  ⚠️  Performance degradation\n" );
    }

    fmt::print( "\n  Function accuracy: {:>6.1f}%\n", accuracy_function );
    fmt::print( "  Class accuracy: {:>6.1f}%\n", accuracy_class );

    if ( accuracy_class >= accuracy_function )
    {
      fmt::print( fg( fmt::color::green ), "  ✅ Class implementation maintains or improves correctness\n" );
    }
    else
    {
      fmt::print( fg( fmt::color::yellow ), "  ⚠️  Warning: class implementation has lower accuracy\n" );
    }

    fmt::print(
      fg( fmt::color::cyan ),
      "╚══════════════════════════════════════════════════════════════════════════════════╝\n" );
  }

  int getExitCode() const { return failed_tests > 0 ? 1 : 0; }
  int getTotalTests() const { return total_tests; }
  int getPassedTests() const { return passed_tests; }
  int getFailedTests() const { return failed_tests; }
};

// ============================================================================
// COMPREHENSIVE TESTER
// ============================================================================

class ComprehensiveSearchIntervalTester
{
private:
  struct DiscrepancyDetail
  {
    string    test_name;
    real_type x;
    integer   func_interval;
    integer   class_interval;
    real_type func_x_mod;
    real_type class_x_mod;
    string    x_vector_str;
    string    analysis;
    // Nuovi campi per i valori effettivi degli intervalli
    real_type func_XL;
    real_type func_XR;
    real_type class_XL;
    real_type class_XR;
  };

  struct TestPerformance
  {
    double  func_time_us  = 0.0;
    double  class_time_us = 0.0;
    double  speedup       = 0.0;
    integer queries       = 0;
  };

  struct TestResult
  {
    string                    name;
    integer                   total_points   = 0;
    integer                   matches        = 0;
    integer                   mismatches     = 0;
    integer                   func_failures  = 0;
    integer                   class_failures = 0;
    vector<real_type>         X;
    bool                      closed;
    bool                      can_extend;
    TestPerformance           performance;
    vector<DiscrepancyDetail> discrepancies;
  };

  vector<TestResult> test_results;
  integer            total_tests_run  = 0;
  integer            total_mismatches = 0;

  // Analyze discrepancy and provide explanation
  string analyze_discrepancy(
    const vector<real_type> & X,
    real_type                 x,
    integer                   func_int,
    integer                   class_int,
    real_type                 func_x_mod,
    real_type                 class_x_mod,
    bool                      closed,
    bool /*can_extend*/,
    // Nuovi parametri per restituire i valori degli intervalli
    real_type & func_XL,
    real_type & func_XR,
    real_type & class_XL,
    real_type & class_XR )
  {
    stringstream analysis;

    // Inizializza i valori degli intervalli
    func_XL = func_XR = class_XL = class_XR = numeric_limits<real_type>::quiet_NaN();

    if ( func_int != class_int )
    {
      // Ottieni i valori reali degli intervalli se gli indici sono validi
      if ( func_int >= 0 && func_int + 1 < static_cast<integer>( X.size() ) )
      {
        func_XL = X[func_int];
        func_XR = X[func_int + 1];
      }

      if ( class_int >= 0 && class_int + 1 < static_cast<integer>( X.size() ) )
      {
        class_XL = X[class_int];
        class_XR = X[class_int + 1];
      }

      analysis << "Different intervals: function returns [" << func_int << "," << func_int + 1 << "] (XL=" << func_XL
               << ", XR=" << func_XR << "), class returns [" << class_int << "," << class_int + 1
               << "] (XL=" << class_XL << ", XR=" << class_XR << ")";

      // Check for duplicate nodes
      if ( func_int >= 0 && func_int + 1 < (integer) X.size() && class_int >= 0 && class_int + 1 < (integer) X.size() )
      {
        if ( abs( X[func_int] - X[func_int + 1] ) < 1e-12 ) analysis << " | Function interval has zero width";
        if ( abs( X[class_int] - X[class_int + 1] ) < 1e-12 ) analysis << " | Class interval has zero width";

        // Check if values are at boundaries
        if ( abs( x - X[func_int] ) < 1e-12 ) analysis << " | x at left boundary of function interval";
        if ( abs( x - X[func_int + 1] ) < 1e-12 ) analysis << " | x at right boundary of function interval";
        if ( abs( x - X[class_int] ) < 1e-12 ) analysis << " | x at left boundary of class interval";
        if ( abs( x - X[class_int + 1] ) < 1e-12 ) analysis << " | x at right boundary of class interval";
      }
    }
    else
    {
      // Anche se gli intervalli sono gli stessi, otteniamo i valori
      if ( func_int >= 0 && func_int + 1 < static_cast<integer>( X.size() ) )
      {
        func_XL = class_XL = X[func_int];
        func_XR = class_XR = X[func_int + 1];
      }
    }

    if ( abs( func_x_mod - class_x_mod ) > 1e-12 )
    {
      if ( !analysis.str().empty() ) analysis << " | ";
      analysis << "Different x modifications: function=" << fixed << setprecision( 12 ) << func_x_mod
               << ", class=" << class_x_mod;

      if ( closed && ( x < X.front() || x > X.back() ) ) analysis << " | Closed curve wrap-around handling differs";
    }

    return analysis.str();
  }

public:
  // CORE TESTING FUNCTIONS
  TestResult run_single_test_case(
    const string &            test_name,
    const vector<real_type> & X,
    bool                      is_closed  = false,
    bool                      can_extend = true,
    bool                      run_perf   = true )
  {
    TestResult result;
    result.name       = test_name;
    result.X          = X;
    result.closed     = is_closed;
    result.can_extend = can_extend;

    print_progress( "Running test: " + test_name );

    if ( X.size() < 2 )
    {
      print_warning( "Skipping test - need at least 2 points" );
      return result;
    }

    integer   n     = static_cast<integer>( X.size() );
    real_type x_min = X[0];
    real_type x_max = X[n - 1];
    real_type range = x_max - x_min;

    // Setup class implementation
    string                             name  = test_name;
    real_type *                        X_ptr = const_cast<real_type *>( X.data() );
    SearchInterval<real_type, integer> search_class;
    search_class.setup( &name, &n, &X_ptr, &is_closed, &can_extend );
    search_class.must_reset();

    // CORRECTNESS TESTS
    vector<real_type> test_points;

    // 1. Test all knots
    for ( size_t i = 0; i < X.size(); ++i )
    {
      test_points.push_back( X[i] );
      // Add small epsilon around knots
      if ( i > 0 ) test_points.push_back( X[i] - 1e-12 );
      if ( i < X.size() - 1 ) test_points.push_back( X[i] + 1e-12 );
    }

    // 2. Test midpoints
    for ( size_t i = 0; i < X.size() - 1; ++i ) { test_points.push_back( ( X[i] + X[i + 1] ) / 2.0 ); }

    // 3. Random points within range
    {
      random_device                        rd;
      mt19937                              gen( rd() );
      uniform_real_distribution<real_type> dist( x_min, x_max );
      for ( integer i = 0; i < 15; ++i ) test_points.push_back( dist( gen ) );
    }

    // 4. Out-of-bounds points (if allowed)
    if ( can_extend || is_closed )
    {
      real_type margin = max( 1.0, range * 0.5 );
      test_points.push_back( x_min - margin );
      test_points.push_back( x_max + margin );
      if ( range > 0 )
      {
        test_points.push_back( x_min - 2.0 * margin );
        test_points.push_back( x_max + 2.0 * margin );
      }
    }

    // Remove duplicates while preserving order
    sort( test_points.begin(), test_points.end() );
    test_points.erase( unique( test_points.begin(), test_points.end() ), test_points.end() );

    // Run correctness comparison
    for ( real_type x : test_points )
    {
      result.total_points++;

      integer   interval_func = -1, interval_class = -1;
      real_type x_mod_func = x, x_mod_class = x;
      bool      func_ok = false, class_ok = false;

      // Function implementation (search_interval)
      try
      {
        integer   last_interval = 0;
        real_type x_copy        = x;
        Utils::search_interval( n, X.data(), x_copy, last_interval, is_closed, can_extend );
        interval_func = last_interval;
        x_mod_func    = x_copy;
        func_ok       = true;
      }
      catch ( const exception & e )
      {
        result.func_failures++;
        func_ok = false;
      }

      // Class implementation (SearchInterval)
      try
      {
        pair<integer, real_type> query{ 0, x };
        search_class.find( query );
        interval_class = query.first;
        x_mod_class    = query.second;
        class_ok       = true;
      }
      catch ( const exception & e )
      {
        result.class_failures++;
        class_ok = false;
      }

      // Compare results
      if ( func_ok && class_ok )
      {
        bool interval_match = ( interval_func == interval_class );
        bool x_mod_match    = ( abs( x_mod_func - x_mod_class ) < 1e-12 );

        if ( interval_match && x_mod_match ) { result.matches++; }
        else
        {
          result.mismatches++;
          total_mismatches++;

          DiscrepancyDetail disc;
          disc.test_name      = test_name;
          disc.x              = x;
          disc.func_interval  = interval_func;
          disc.class_interval = interval_class;
          disc.func_x_mod     = x_mod_func;
          disc.class_x_mod    = x_mod_class;

          // Aggiungi i valori effettivi degli intervalli
          disc.analysis = analyze_discrepancy(
            X,
            x,
            interval_func,
            interval_class,
            x_mod_func,
            x_mod_class,
            is_closed,
            can_extend,
            disc.func_XL,
            disc.func_XR,
            disc.class_XL,
            disc.class_XR );
          result.discrepancies.push_back( disc );
        }
      }
    }

    // PERFORMANCE TESTS
    if ( run_perf && n >= 10 )
    {
      const integer NUM_QUERIES = min<integer>( 100000, 10000 * n );

      // Generate random queries
      vector<real_type> queries( NUM_QUERIES );
      {
        random_device                        rd;
        mt19937                              gen( rd() );
        real_type                            margin = can_extend ? max( 1.0, range * 0.1 ) : 0.0;
        uniform_real_distribution<real_type> dist( x_min - margin, x_max + margin );
        for ( auto & q : queries ) q = dist( gen );
      }

      // Benchmark function implementation
      {
        integer last_interval = 0;
        TicToc  timer;
        timer.tic();
        for ( auto x : queries )
        {
          real_type x_copy = x;
          try
          {
            Utils::search_interval( n, X.data(), x_copy, last_interval, is_closed, can_extend );
          }
          catch ( ... )
          {
          }
        }
        timer.toc();
        result.performance.func_time_us = timer.elapsed_mus();
      }

      // Benchmark class implementation
      {
        TicToc timer;
        timer.tic();
        for ( auto x : queries )
        {
          pair<integer, real_type> query{ 0, x };
          try
          {
            search_class.find( query );
          }
          catch ( ... )
          {
          }
        }
        timer.toc();
        result.performance.class_time_us = timer.elapsed_mus();
        result.performance.queries       = NUM_QUERIES;
      }

      // Calculate speedup
      if ( result.performance.class_time_us > 0 )
      {
        result.performance.speedup = result.performance.func_time_us / result.performance.class_time_us;
      }
    }

    test_results.push_back( result );
    total_tests_run++;

    return result;
  }

  // ============================================================================
  // SPECIALIZED TESTS
  // ============================================================================

  void run_edge_case_tests()
  {
    print_section( "EDGE CASE TESTS" );

    struct EdgeCase
    {
      string            name;
      vector<real_type> X;
      bool              closed;
      bool              can_extend;
    };

    vector<EdgeCase> edge_cases = {
      { "Single duplicate", { 1.0, 1.0 }, false, true },
      { "All duplicates", { 1.0, 1.0, 1.0, 1.0, 1.0 }, false, true },
      { "Very small range", { 0.0, 1e-15, 2e-15, 3e-15 }, false, true },
      { "Large range", { -1e10, 0.0, 1e10 }, false, true },
      { "Non-monotonic (should fail)", { 3.0, 2.0, 1.0 }, false, true },
      { "Single point (invalid)", { 1.0 }, false, true },
      { "Two points reverse", { 2.0, 1.0 }, false, true },
      { "Closed empty range", { 0.0, 0.0 }, true, false },
      { "Closed with wrap", { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 }, true, false },
      { "Cannot extend", { 0.0, 1.0, 2.0, 3.0 }, false, false },
      { "Negative values", { -10.0, -5.0, 0.0, 5.0, 10.0 }, false, true },
      { "Mixed signs", { -100.0, -1.0, 0.0, 1.0, 100.0 }, false, true },
      { "Exact zero intervals", { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 }, false, true },
    };

    for ( const auto & ec : edge_cases )
    {
      run_single_test_case( "[EDGE] " + ec.name, ec.X, ec.closed, ec.can_extend, false );
    }
  }

  void run_scaling_performance_tests()
  {
    print_section( "SCALING PERFORMANCE TESTS" );

    vector<integer> sizes       = { 10, 100, 1000, 10000, 100000 };
    const integer   NUM_QUERIES = 100000;

    fmt::print(
      "\n{:<12} {:<12} {:<14} {:<14} {:<14} {:<12}\n",
      "N points",
      "Queries",
      "Function (µs)",
      "Class (µs)",
      "µs/query",
      "Speedup" );
    fmt::print( "{}\n", Utils::repeat( "─", 78 ) );

    for ( integer n : sizes )
    {
      // Create dataset with some non-uniformity
      vector<real_type> X( n );
      X[0] = 0.0;
      for ( integer i = 1; i < n; ++i )
      {
        // Add some non-linearity
        real_type t = real_type( i ) / ( n - 1 );
        X[i]        = X[i - 1] + 1.0 + 0.5 * sin( 10.0 * t );
      }

      // Generate queries
      vector<real_type> queries( NUM_QUERIES );
      {
        random_device                        rd;
        mt19937                              gen( rd() );
        real_type                            margin = 0.1 * ( X.back() - X.front() );
        uniform_real_distribution<real_type> dist( X.front() - margin, X.back() + margin );
        for ( auto & q : queries ) q = dist( gen );
      }

      // Setup
      string                             name   = "scaling_test";
      integer                            n_pts  = n;
      real_type *                        X_ptr  = X.data();
      bool                               closed = false, can_extend = true;
      SearchInterval<real_type, integer> search_class;
      search_class.setup( &name, &n_pts, &X_ptr, &closed, &can_extend );

      // Warm-up
      for ( integer i = 0; i < 1000; ++i )
      {
        integer   dummy  = 0;
        real_type x_copy = queries[i % queries.size()];
        Utils::search_interval( n, X.data(), x_copy, dummy, closed, can_extend );
      }

      // Benchmark function implementation
      double func_time_us = 0.0;
      {
        integer last_interval = 0;
        TicToc  timer;
        timer.tic();
        for ( auto x : queries )
        {
          real_type x_copy = x;
          try
          {
            Utils::search_interval( n, X.data(), x_copy, last_interval, closed, can_extend );
          }
          catch ( ... )
          {
          }
        }
        timer.toc();
        func_time_us = timer.elapsed_mus();
      }

      // Benchmark class implementation
      double class_time_us = 0.0;
      {
        TicToc timer;
        timer.tic();
        for ( auto x : queries )
        {
          pair<integer, real_type> query{ 0, x };
          try
          {
            search_class.find( query );
          }
          catch ( ... )
          {
          }
        }
        timer.toc();
        class_time_us = timer.elapsed_mus();
      }

      double speedup            = func_time_us / class_time_us;
      double us_per_query_class = class_time_us / NUM_QUERIES;

      // Color code based on speedup
      string speedup_str;
      if ( speedup > 5.0 )
        speedup_str = fmt::format( fg( fmt::color::green ), "{:.4g}x 🚀", speedup );
      else if ( speedup > 2.0 )
        speedup_str = fmt::format( fg( fmt::color::green ), "{:.4g}x ⚡", speedup );
      else if ( speedup > 1.0 )
        speedup_str = fmt::format( "{:.4g}x ↗", speedup );
      else if ( speedup < 1.0 )
        speedup_str = fmt::format( fg( fmt::color::yellow ), "{:.4g}x ↘", speedup );
      else
        speedup_str = "1.00x";

      fmt::print(
        "{:<12} {:<12} {:<14.4g} {:<14.4g} {:<12.4g} {:<12} \n",
        n,
        NUM_QUERIES,
        func_time_us,
        class_time_us,
        us_per_query_class,
        speedup_str );
    }
  }

  // ============================================================================
  // RESULTS REPORTING
  // ============================================================================

  void print_detailed_discrepancies()
  {
    if ( total_mismatches == 0 )
    {
      print_success( "No discrepancies found between function and class implementations" );
      return;
    }

    print_header( "DETAILED DISCREPANCY ANALYSIS", fmt::color::magenta );

    // integer total_discrepancies_shown = 0; // Variable non usata, commentata
    const integer MAX_DISCREPANCIES_PER_TEST = 5;

    for ( const auto & result : test_results )
    {
      if ( !result.discrepancies.empty() )
      {
        fmt::print( fg( fmt::color::yellow ), "\n📋 Test: {}\n", result.name );
        fmt::print(
          "   Mismatches: {} out of {} points ({:.1g}%)\n",
          result.mismatches,
          result.total_points,
          ( 100.0 * result.mismatches ) / result.total_points );

        // Show representative discrepancies
        integer shown = 0;
        for ( const auto & disc : result.discrepancies )
        {
          if ( shown >= MAX_DISCREPANCIES_PER_TEST )
          {
            fmt::print(
              fg( fmt::color::yellow ),
              "   ... and {} more discrepancies for this test\n",
              result.discrepancies.size() - shown );
            break;
          }

          // Aggiornata la stampa per includere i valori degli intervalli
          fmt::print(
            fg( fmt::color::magenta ),
            "   ✗ x={:.12g}: function→[{},{}] (XL={:.12g}, XR={:.12g}) (x_mod={:.12g}), "
            "class→[{},{}] (XL={:.12g}, XR={:.12g}) (x_mod={:.12g})\n",
            disc.x,
            disc.func_interval,
            disc.func_interval + 1,
            disc.func_XL,
            disc.func_XR,
            disc.func_x_mod,
            disc.class_interval,
            disc.class_interval + 1,
            disc.class_XL,
            disc.class_XR,
            disc.class_x_mod );

          if ( !disc.analysis.empty() ) fmt::print( fg( fmt::color::cyan ), "      Analysis: {}\n", disc.analysis );

          shown++;
          // total_discrepancies_shown++; // Variable non usata, commentata
        }
      }
    }

    fmt::print( "\n" );
    fmt::print( fg( fmt::color::yellow ), "📊 Total discrepancies found: {}\n", total_mismatches );
  }

  void print_summary_statistics()
  {
    integer total_points           = 0;
    integer total_matches          = 0;
    integer total_mismatches_local = 0;
    integer total_func_fails       = 0;
    integer total_class_fails      = 0;

    for ( const auto & result : test_results )
    {
      total_points += result.total_points;
      total_matches += result.matches;
      total_mismatches_local += result.mismatches;
      total_func_fails += result.func_failures;
      total_class_fails += result.class_failures;
    }

    double match_percentage = ( total_points > 0 ) ? ( 100.0 * total_matches / total_points ) : 0.0;

    fmt::print(
      "\n"
      "📊 Overall Statistics:\n"
      "   ┌───────────────────────────────────────────┐\n"
      "   │ {:<20} {:>20} │\n"
      "   ├───────────────────────────────────────────┤\n"
      "   │ {:<20} {:>20} │\n"
      "   │ {:<20} {:>20} │\n"
      "   │ {:<20} {:>19.2f}% │\n"
      "   │ {:<20} {:>20} │\n"
      "   │ {:<20} {:>20} │\n"
      "   │ {:<20} {:>20} │\n"
      "   └───────────────────────────────────────────┘\n",
      "Metric",
      "Value",
      "Total Tests:",
      total_tests_run,
      "Total Points Tested:",
      total_points,
      "Match Rate:",
      match_percentage,
      "Mismatches:",
      total_mismatches_local,
      "Function Failures:",
      total_func_fails,
      "Class Failures:",
      total_class_fails );
  }

  integer get_total_mismatches() const { return total_mismatches; }
  integer get_total_tests() const { return total_tests_run; }
};

// ============================================================================
// ADVANCED PERFORMANCE TESTER
// ============================================================================

class AdvancedPerformanceTester
{
private:
  struct PerformanceResult
  {
    string              test_name;
    size_t              num_intervals;
    size_t              num_queries;
    double              function_time_us;
    double              class_time_us;
    double              speedup;
    double              time_per_query_func_ns;
    double              time_per_query_class_ns;
    double              memory_overhead_kb;
    bool                correctness_passed;
    bool                is_closed;
    map<string, double> detailed_metrics;
  };

  vector<PerformanceResult> results;

  // Generate uniform intervals
  vector<real_type> generate_uniform_intervals( size_t num_intervals, real_type start = 0.0, real_type end = 100.0 )
  {
    vector<real_type> X( num_intervals + 1 );
    real_type         step = ( end - start ) / num_intervals;
    for ( size_t i = 0; i <= num_intervals; ++i ) { X[i] = start + i * step; }
    return X;
  }

  // Generate non-uniform intervals (logarithmic distribution)
  vector<real_type> generate_nonuniform_intervals( size_t num_intervals, real_type start = 0.1, real_type end = 100.0 )
  {
    vector<real_type> X( num_intervals + 1 );
    real_type         log_start = log10( start );
    real_type         log_end   = log10( end );
    real_type         log_step  = ( log_end - log_start ) / num_intervals;

    for ( size_t i = 0; i <= num_intervals; ++i ) { X[i] = pow( 10.0, log_start + i * log_step ); }
    // Add some randomness to make it truly non-uniform
    random_device                        rd;
    mt19937                              gen( rd() );
    uniform_real_distribution<real_type> dist( -0.05, 0.05 );

    for ( size_t i = 1; i < num_intervals; ++i ) { X[i] *= ( 1.0 + dist( gen ) ); }
    sort( X.begin(), X.end() );
    return X;
  }

  // Generate sequential query pattern (A to B and back)
  vector<real_type> generate_sequential_queries(
    const vector<real_type> & X,
    size_t                    num_queries,
    bool                      forward_backward = true )
  {
    vector<real_type> queries;
    real_type         x_min = X.front();
    real_type         x_max = X.back();
    real_type         range = x_max - x_min;

    if ( forward_backward )
    {
      // Forward
      for ( size_t i = 0; i < num_queries / 2; ++i )
      {
        real_type t = static_cast<real_type>( i ) / ( num_queries / 2 - 1 );
        queries.push_back( x_min - range * 0.1 + t * ( range * 1.2 ) );
      }
      // Backward
      for ( size_t i = 0; i < num_queries / 2; ++i )
      {
        real_type t = static_cast<real_type>( i ) / ( num_queries / 2 - 1 );
        queries.push_back( x_max + range * 0.1 - t * ( range * 1.2 ) );
      }
    }
    else
    {
      // Forward only
      for ( size_t i = 0; i < num_queries; ++i )
      {
        real_type t = static_cast<real_type>( i ) / ( num_queries - 1 );
        queries.push_back( x_min - range * 0.1 + t * ( range * 1.2 ) );
      }
    }
    return queries;
  }

  // Generate random query pattern
  vector<real_type> generate_random_queries( const vector<real_type> & X, size_t num_queries )
  {
    random_device rd;
    mt19937       gen( rd() );

    real_type x_min = X.front();
    real_type x_max = X.back();
    real_type range = x_max - x_min;

    uniform_real_distribution<real_type> dist( x_min - range * 0.1, x_max + range * 0.1 );

    vector<real_type> queries( num_queries );
    for ( size_t i = 0; i < num_queries; ++i ) { queries[i] = dist( gen ); }
    return queries;
  }

  // Run single performance test
  PerformanceResult run_performance_test(
    const string &            test_name,
    const vector<real_type> & X,
    const vector<real_type> & queries,
    bool                      is_closed  = false,
    bool                      can_extend = true )
  {
    PerformanceResult result;
    result.test_name     = test_name;
    result.num_intervals = X.size() - 1;
    result.num_queries   = queries.size();
    result.is_closed     = is_closed;

    // Setup implementations
    integer     n     = static_cast<integer>( X.size() );
    real_type * X_ptr = const_cast<real_type *>( X.data() );
    string      name  = test_name;

    // Class implementation
    SearchInterval<real_type, integer> search_class;
    search_class.setup( &name, &n, &X_ptr, &is_closed, &can_extend );
    search_class.must_reset();

    // Verify correctness on a subset
    bool           correctness_passed  = true;
    const size_t   correctness_samples = min<size_t>( 1000, queries.size() );
    vector<size_t> sample_indices( correctness_samples );

    random_device                    rd;
    mt19937                          gen( rd() );
    uniform_int_distribution<size_t> dist_idx( 0, queries.size() - 1 );

    for ( size_t i = 0; i < correctness_samples; ++i )
    {
      size_t idx        = dist_idx( gen );
      sample_indices[i] = idx;

      real_type x               = queries[idx];
      real_type x_func          = x;
      integer   last_func_local = 0;

      // Function
      Utils::search_interval( n, X.data(), x_func, last_func_local, is_closed, can_extend );

      // Class
      pair<integer, real_type> query_class = { 0, x };
      search_class.find( query_class );

      // Compare
      if ( last_func_local != query_class.first || abs( x_func - query_class.second ) > 1e-12 )
      {
        correctness_passed = false;
        break;
      }
    }
    result.correctness_passed = correctness_passed;

    // Warm-up runs
    for ( size_t i = 0; i < min<size_t>( 1000, queries.size() ); ++i )
    {
      real_type x               = queries[i];
      real_type x_func          = x;
      integer   last_func_local = 0;
      Utils::search_interval( n, X.data(), x_func, last_func_local, is_closed, can_extend );

      pair<integer, real_type> query_class = { 0, x };
      search_class.find( query_class );
    }

    // Benchmark function implementation
    {
      integer last_local = 0;
      TicToc  timer;
      timer.tic();

      for ( const auto & x : queries )
      {
        real_type x_copy = x;
        Utils::search_interval( n, X.data(), x_copy, last_local, is_closed, can_extend );
      }

      timer.toc();
      result.function_time_us       = timer.elapsed_mus();
      result.time_per_query_func_ns = timer.elapsed_ns() / queries.size();
    }

    // Benchmark class implementation
    {
      TicToc timer;
      timer.tic();

      for ( const auto & x : queries )
      {
        pair<integer, real_type> query = { 0, x };
        search_class.find( query );
      }

      timer.toc();
      result.class_time_us           = timer.elapsed_mus();
      result.time_per_query_class_ns = timer.elapsed_ns() / queries.size();
    }

    // Calculate speedup
    if ( result.class_time_us > 0 ) { result.speedup = result.function_time_us / result.class_time_us; }
    else
    {
      result.speedup = 0.0;
    }

    // Estimate memory overhead (for class implementation)
    size_t table_size         = static_cast<size_t>( sqrt( result.num_intervals ) ) + 2;
    result.memory_overhead_kb = ( table_size * 2 * sizeof( integer ) ) / 1024.0;

    // Store detailed metrics
    result.detailed_metrics["function_total_us"]  = result.function_time_us;
    result.detailed_metrics["class_total_us"]     = result.class_time_us;
    result.detailed_metrics["speedup"]            = result.speedup;
    result.detailed_metrics["func_ns_per_query"]  = result.time_per_query_func_ns;
    result.detailed_metrics["class_ns_per_query"] = result.time_per_query_class_ns;
    result.detailed_metrics["memory_kb"]          = result.memory_overhead_kb;

    return result;
  }

  // Calculate R-squared for linear regression
  double calculate_r_squared( const vector<pair<double, double>> & data, double slope, double intercept )
  {
    double ss_total = 0, ss_residual = 0;
    double mean_y = 0;

    for ( const auto & [x, y] : data ) { mean_y += y; }
    mean_y /= data.size();

    for ( const auto & [x, y] : data )
    {
      double y_pred = slope * x + intercept;
      ss_total += pow( y - mean_y, 2 );
      ss_residual += pow( y - y_pred, 2 );
    }

    return 1.0 - ( ss_residual / ss_total );
  }

public:
  // Run specific scenario tests for both closed and non-closed cases
  void run_requested_scenarios()
  {
    print_header( "REQUESTED SCENARIO TESTS", fmt::color::green );

    // Define test scenarios: 1A, 1B, 2A, 2B
    struct Scenario
    {
      string         name;
      bool           uniform;
      bool           sequential;
      vector<size_t> interval_counts;
    };

    vector<Scenario> scenarios = { { "1A - Uniform + Sequential", true, true, { 1, 10, 100, 1000, 10000 } },
                                   { "1B - Uniform + Random", true, false, { 1, 10, 100, 1000, 10000 } },
                                   { "2A - NonUniform + Sequential", false, true, { 1, 10, 100, 1000, 10000 } },
                                   { "2B - NonUniform + Random", false, false, { 1, 10, 100, 1000, 10000 } } };

    const size_t NUM_QUERIES = 10000;

    // Test both non-closed and closed cases
    vector<pair<bool, string>> closed_cases = { { false, "Non-Closed" }, { true, "Closed" } };

    for ( const auto & [is_closed, closed_name] : closed_cases )
    {
      print_section( fmt::format( "{} CASE", closed_name ) );

      for ( const auto & scenario : scenarios )
      {
        print_progress( fmt::format( "Running {}", scenario.name ) );

        fmt::print(
          "\n{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n",
          "N Intervals",
          "Func Total",
          "Class Total",
          "Speedup",
          "Func/q (ns)",
          "Class/q (ns)",
          "Correct" );
        fmt::print( "{}\n", Utils::repeat( "─", 85 ) );

        for ( size_t num_intervals : scenario.interval_counts )
        {
          // Generate intervals
          vector<real_type> X;
          if ( scenario.uniform ) { X = generate_uniform_intervals( num_intervals ); }
          else
          {
            X = generate_nonuniform_intervals( num_intervals );
          }

          // Generate queries
          vector<real_type> queries;
          if ( scenario.sequential ) { queries = generate_sequential_queries( X, NUM_QUERIES, true ); }
          else
          {
            queries = generate_random_queries( X, NUM_QUERIES );
          }

          // Run test
          string test_name =
            fmt::format( "{}_{}_{}_{}", closed_name, scenario.name, num_intervals, is_closed ? "closed" : "open" );

          auto result = run_performance_test( test_name, X, queries, is_closed, true );
          results.push_back( result );

          // Format output
          string speedup_color;
          if ( result.speedup > 2.0 )
            speedup_color = "🟢";
          else if ( result.speedup > 1.2 )
            speedup_color = "🟡";
          else if ( result.speedup > 0.8 )
            speedup_color = "🟠";
          else
            speedup_color = "🔴";

          string correct_symbol = result.correctness_passed ? "✅" : "❌";

          fmt::print(
            "{:<12} {:<12.0f} {:<12.0f} {} {:<8.2f}x {:<12.2f} {:<12.2f} {:<12}\n",
            num_intervals,
            result.function_time_us,
            result.class_time_us,
            speedup_color,
            result.speedup,
            result.time_per_query_func_ns,
            result.time_per_query_class_ns,
            correct_symbol );
        }

        // Print scenario summary
        print_scenario_summary( scenario.name, is_closed );
      }
    }
  }

  void print_scenario_summary( const string & scenario_name, bool is_closed )
  {
    // Collect results for this scenario
    vector<const PerformanceResult *> scenario_results;
    for ( const auto & result : results )
    {
      if ( result.test_name.find( scenario_name ) != string::npos && result.is_closed == is_closed )
      {
        scenario_results.push_back( &result );
      }
    }

    if ( scenario_results.empty() ) return;

    // Calculate statistics
    double avg_speedup       = 0;
    double avg_func_time     = 0;
    double avg_class_time    = 0;
    int    correctness_count = 0;

    for ( const auto * res : scenario_results )
    {
      avg_speedup += res->speedup;
      avg_func_time += res->function_time_us;
      avg_class_time += res->class_time_us;
      if ( res->correctness_passed ) correctness_count++;
    }

    avg_speedup /= scenario_results.size();
    avg_func_time /= scenario_results.size();
    avg_class_time /= scenario_results.size();

    // Print summary
    fmt::print( "\n📊 {} Summary ({}):\n", scenario_name, is_closed ? "Closed" : "Non-Closed" );
    fmt::print( "   Average Speedup: {:.2f}x\n", avg_speedup );
    fmt::print( "   Average Function Time: {:.0f} µs\n", avg_func_time );
    fmt::print( "   Average Class Time: {:.0f} µs\n", avg_class_time );
    fmt::print( "   Correctness: {}/{} tests passed\n", correctness_count, scenario_results.size() );

    // Recommendation
    string recommendation;
    if ( avg_speedup > 3.0 ) { recommendation = "🚀 CLASS HIGHLY RECOMMENDED"; }
    else if ( avg_speedup > 2.0 ) { recommendation = "⚡ CLASS RECOMMENDED"; }
    else if ( avg_speedup > 1.2 ) { recommendation = "✓ CLASS PREFERRED"; }
    else if ( avg_speedup > 1.0 ) { recommendation = "↗️ MARGINAL GAIN"; }
    else if ( avg_speedup > 0.8 ) { recommendation = "⚠️ SIMILAR PERFORMANCE"; }
    else
    {
      recommendation = "🔴 FUNCTION PREFERRED";
    }

    fmt::print( "   Recommendation: {}\n\n", recommendation );
  }

  void print_comprehensive_analysis()
  {
    print_header( "COMPREHENSIVE PERFORMANCE ANALYSIS", fmt::color::magenta );

    // Group by closed status and interval count
    map<pair<bool, size_t>, vector<double>> speedups_by_category;

    for ( const auto & result : results )
    {
      auto key = make_pair( result.is_closed, result.num_intervals );
      speedups_by_category[key].push_back( result.speedup );
    }

    // Print comparison between closed and non-closed
    fmt::print(
      "\n{:<12} {:<15} {:<15} {:<15} {:<15}\n",
      "N Intvls",
      "Non-Closed Avg",
      "Non-Closed Min",
      "Closed Avg",
      "Closed Min" );
    fmt::print( "{}\n", Utils::repeat( "─", 85 ) );

    std::set<size_t> interval_sizes;
    for ( const auto & [key, _] : speedups_by_category ) { interval_sizes.insert( key.second ); }

    for ( size_t n : interval_sizes )
    {
      double non_closed_avg = 0, non_closed_min = 1000;
      double closed_avg = 0, closed_min = 1000;
      int    non_closed_count = 0, closed_count = 0;

      for ( const auto & [key, speedups] : speedups_by_category )
      {
        if ( key.second == n )
        {
          double avg     = accumulate( speedups.begin(), speedups.end(), 0.0 ) / speedups.size();
          double min_val = *min_element( speedups.begin(), speedups.end() );

          if ( key.first == false )
          {  // non-closed
            non_closed_avg   = avg;
            non_closed_min   = min_val;
            non_closed_count = speedups.size();
          }
          else
          {  // closed
            closed_avg   = avg;
            closed_min   = min_val;
            closed_count = speedups.size();
          }
        }
      }

      if ( non_closed_count > 0 && closed_count > 0 )
      {
        fmt::print(
          "{:<12} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}\n",
          n,
          non_closed_avg,
          non_closed_min,
          closed_avg,
          closed_min );
      }
    }

    // Performance trends analysis
    print_performance_trends();

    // Memory-performance tradeoff analysis
    print_memory_tradeoff_analysis();
  }

  void print_performance_trends()
  {
    print_section( "PERFORMANCE TRENDS ANALYSIS" );

    // Group by number of intervals
    map<size_t, vector<double>> speedups_by_size;
    map<size_t, vector<double>> times_by_size;

    for ( const auto & result : results )
    {
      speedups_by_size[result.num_intervals].push_back( result.speedup );
      times_by_size[result.num_intervals].push_back( result.time_per_query_class_ns );
    }

    fmt::print(
      "\n{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n",
      "N Intvls",
      "Avg Speedup",
      "Min Speedup",
      "Max Speedup",
      "Avg Time/query",
      "Theoretical O()" );
    fmt::print( "{}\n", Utils::repeat( "─", 85 ) );

    for ( const auto & [num_intvls, speedups] : speedups_by_size )
    {
      double avg_speedup = accumulate( speedups.begin(), speedups.end(), 0.0 ) / speedups.size();
      double min_speedup = *min_element( speedups.begin(), speedups.end() );
      double max_speedup = *max_element( speedups.begin(), speedups.end() );

      const vector<double> & times    = times_by_size[num_intvls];
      double                 avg_time = accumulate( times.begin(), times.end(), 0.0 ) / times.size();

      // Theoretical complexities
      // double log_n = log2(num_intvls); // Variable non usata, commentata
      double log_sqrt_n = log2( sqrt( num_intvls ) );
      string theory     = fmt::format( "O(log √n)≈{:.1f}", log_sqrt_n );

      fmt::print(
        "{:<12} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12}\n",
        num_intvls,
        avg_speedup,
        min_speedup,
        max_speedup,
        avg_time,
        theory );
    }
  }

  void print_memory_tradeoff_analysis()
  {
    print_section( "MEMORY-PERFORMANCE TRADEOFF" );

    // Analyze correlation between memory usage and performance
    vector<pair<double, double>> memory_speedup_pairs;

    for ( const auto & result : results )
    {
      if ( result.correctness_passed )
      {
        memory_speedup_pairs.emplace_back( result.memory_overhead_kb, result.speedup );
      }
    }

    if ( memory_speedup_pairs.size() < 2 ) return;

    // Simple linear regression
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    size_t n = memory_speedup_pairs.size();

    for ( const auto & [x, y] : memory_speedup_pairs )
    {
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_x2 += x * x;
    }

    double slope     = ( n * sum_xy - sum_x * sum_y ) / ( n * sum_x2 - sum_x * sum_x );
    double intercept = ( sum_y - slope * sum_x ) / n;

    fmt::print( "\nMemory-Performance Correlation:\n" );
    fmt::print( "   Regression: Speedup = {:.4f} * Memory(KB) + {:.4f}\n", slope, intercept );
    fmt::print( "   R² = {:.3f}\n", calculate_r_squared( memory_speedup_pairs, slope, intercept ) );
  }

  void run_all_tests()
  {
    print_header( "ADVANCED PERFORMANCE TESTING", fmt::color::magenta );
    run_requested_scenarios();
    print_comprehensive_analysis();
  }
};

// ============================================================================
// UNIFIED TEST SUITE WITH ADDITIONAL STATISTICS
// ============================================================================

class UnifiedTestSuite
{
private:
  struct GlobalStatistics
  {
    int              total_correctness_tests = 0;
    int              total_performance_tests = 0;
    int              total_edge_cases        = 0;
    int              discrepancies_found     = 0;
    double           avg_speedup             = 0.0;
    double           max_speedup             = 0.0;
    double           min_speedup             = 1000000.0;
    double           total_execution_time_ms = 0.0;
    map<string, int> test_categories;
  };

  GlobalStatistics stats;
  TicToc           global_timer;

public:
  void run_all_tests()
  {
    global_timer.tic();

    print_header( "UNIFIED SEARCH INTERVAL TEST SUITE", fmt::color::cyan );
    print_info(
      "Comparing: Utils::search_interval (Function Implementation) vs Utils::SearchInterval (Class Implementation)" );

    // Phase 1: Comprehensive Correctness Testing
    {
      print_section( "PHASE 1: COMPREHENSIVE CORRECTNESS TESTING" );

      ComprehensiveSearchIntervalTester comp_tester;

      // Basic distributions
      print_progress( "Testing basic distributions" );
      vector<real_type> uniform_X = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
      comp_tester.run_single_test_case( "Uniform-11", uniform_X, false, true, false );
      stats.total_correctness_tests++;

      vector<real_type> nonuniform_X = { 0.0, 0.1, 0.5, 1.2, 2.0, 3.5, 5.0, 6.1, 6.2, 6.3, 7.0, 8.5, 10.0 };
      comp_tester.run_single_test_case( "NonUniform-13", nonuniform_X, false, true, false );
      stats.total_correctness_tests++;

      // Edge cases
      print_progress( "Testing edge cases" );
      comp_tester.run_edge_case_tests();
      stats.total_edge_cases += 13;  // based on edge_cases vector size

      stats.discrepancies_found = comp_tester.get_total_mismatches();

      // Print summary
      comp_tester.print_summary_statistics();
      if ( stats.discrepancies_found > 0 ) { comp_tester.print_detailed_discrepancies(); }
    }

    // Phase 2: Performance Benchmarking
    {
      print_section( "PHASE 2: PERFORMANCE BENCHMARKING" );

      EnhancedTestRunner enh_runner;

      // Performance scaling test
      print_progress( "Running performance scaling tests" );

      vector<int>    sizes = { 10, 100, 1000, 10000, 100000 };
      vector<double> speedups;

      for ( int n : sizes )
      {
        vector<real_type> X( n );
        for ( int i = 0; i < n; ++i ) X[i] = static_cast<real_type>( i ) * 100.0 / ( n - 1 );

        const int         NUM_QUERIES = 100000;
        vector<real_type> queries( NUM_QUERIES );

        random_device               rd;
        mt19937                     gen( rd() );
        uniform_real_distribution<> dis( -50.0, 150.0 );
        for ( int i = 0; i < NUM_QUERIES; ++i ) queries[i] = dis( gen );

        // Setup
        string      name      = "perf_test";
        integer     n_new     = n;
        real_type * X_ptr     = X.data();
        bool        is_closed = false, can_extend = true;

        SearchInterval<real_type, integer> search_class;
        search_class.setup( &name, &n_new, &X_ptr, &is_closed, &can_extend );

        // Benchmark function implementation
        TicToc tm;
        tm.tic();
        for ( int i = 0; i < NUM_QUERIES; ++i )
        {
          integer last = 0;
          Utils::search_interval( n, X.data(), queries[i], last, is_closed, can_extend );
        }
        tm.toc();
        double time_function = tm.elapsed_ns();

        // Benchmark class implementation
        tm.tic();
        for ( int i = 0; i < NUM_QUERIES; ++i )
        {
          pair<integer, real_type> res = { 0, queries[i] };
          search_class.find( res );
        }
        tm.toc();
        double time_class = tm.elapsed_ns();

        double speedup = time_function / time_class;
        speedups.push_back( speedup );

        stats.max_speedup = max( stats.max_speedup, speedup );
        stats.min_speedup = min( stats.min_speedup, speedup );

        string status;
        if ( speedup > 5.0 )
          status = "🚀 EXCELLENT";
        else if ( speedup > 2.0 )
          status = "⚡ GOOD";
        else if ( speedup > 1.0 )
          status = "✓ OK";
        else
          status = "⚠️  SLOWER";

        fmt::print( "  n={:<7}: speedup={:.2f}x {}\n", n, speedup, status );
        stats.total_performance_tests++;
      }

      // Calculate average speedup
      if ( !speedups.empty() )
      {
        stats.avg_speedup = accumulate( speedups.begin(), speedups.end(), 0.0 ) / speedups.size();
      }

      enh_runner.printPerformanceComparison();
    }

    // Phase 3: Additional Statistics
    {
      print_section( "PHASE 3: ADDITIONAL STATISTICS & ANALYSIS" );

      // Memory usage analysis
      print_progress( "Analyzing memory usage" );
      fmt::print( "\n💾 MEMORY USAGE ESTIMATION:\n" );
      fmt::print( "   {:<10} {:<15} {:<15}\n", "N points", "Table Size", "Memory (KB)" );
      fmt::print( "   {}\n", Utils::repeat( "─", 45 ) );

      vector<int> mem_sizes = { 10, 100, 1000, 10000, 100000, 1000000 };
      for ( int n : mem_sizes )
      {
        size_t table_entries = static_cast<size_t>( sqrt( n ) ) + 2;
        size_t memory_bytes  = table_entries * 2 * sizeof( integer );
        fmt::print( "   {:<10} {:<15} {:<15.2f}\n", n, table_entries, memory_bytes / 1024.0 );
      }

      // Performance prediction
      fmt::print( "\n📈 PERFORMANCE PREDICTION:\n" );
      fmt::print( "   For N=1,000,000 points:\n" );
      fmt::print( "   - Function implementation (binary search): ~log2(N) = 20 comparisons per query\n" );
      fmt::print( "   - Class implementation (hybrid search): ~log2(√N) = 10 comparisons + 1 table lookup\n" );
      fmt::print( "   - Expected speedup: 1.5x - 3.0x\n" );

      // Implementation characteristics
      fmt::print( "\n🔧 IMPLEMENTATION CHARACTERISTICS:\n" );
      fmt::print( "   Function Implementation:\n" );
      fmt::print( "   - Pure binary search algorithm\n" );
      fmt::print( "   - No memory overhead\n" );
      fmt::print( "   - O(log n) time complexity\n" );
      fmt::print( "   - Stateless, thread-safe by design\n\n" );

      fmt::print( "   Class Implementation:\n" );
      fmt::print( "   - Hybrid approach: lookup table + binary search\n" );
      fmt::print( "   - Memory overhead: O(√n)\n" );
      fmt::print( "   - O(log √n) average time complexity\n" );
      fmt::print( "   - Stateful, uses mutex for thread safety\n" );
    }

    global_timer.toc();
    stats.total_execution_time_ms = global_timer.elapsed_ms();
  }

  void print_final_report()
  {
    print_header( "FINAL TEST REPORT", fmt::color::green );

    fmt::print(
      "\n"
      "╔══════════════════════════════════════════════════════════════════════════════════╗\n"
      "║                              SUMMARY STATISTICS                                  ║\n"
      "╠══════════════════════════════════════════════════════════════════════════════════╣\n"
      "║                                                                                  ║\n"
      "║  🧪 Tests Executed:        {:>10}                                            ║\n"
      "║     • Correctness tests:   {:>10}                                            ║\n"
      "║     • Performance tests:   {:>10}                                            ║\n"
      "║     • Edge cases:          {:>10}                                            ║\n"
      "║                                                                                  ║\n"
      "║  ⚡ Performance Results:                                                         ║\n"
      "║     • Average speedup:     {:>10.2f}x                                           ║\n"
      "║     • Maximum speedup:     {:>10.2f}x                                           ║\n"
      "║     • Minimum speedup:     {:>10.2f}x                                           ║\n"
      "║                                                                                  ║\n"
      "║  🔍 Correctness Results:                                                         ║\n"
      "║     • Discrepancies found: {:>10}                                            ║\n"
      "║                                                                                  ║\n"
      "║  ⏱️  Execution Time:        {:>10.2f} ms                                        ║\n"
      "║                                                                                  ║\n"
      "╚══════════════════════════════════════════════════════════════════════════════════╝\n",
      stats.total_correctness_tests + stats.total_performance_tests + stats.total_edge_cases,
      stats.total_correctness_tests,
      stats.total_performance_tests,
      stats.total_edge_cases,
      stats.avg_speedup,
      stats.max_speedup,
      stats.min_speedup,
      stats.discrepancies_found,
      stats.total_execution_time_ms );

    // Recommendations
    fmt::print( "\n💡 RECOMMENDATIONS:\n" );

    if ( stats.discrepancies_found == 0 ) { print_success( "Both implementations are functionally equivalent" ); }
    else
    {
      print_warning( fmt::format( "Found {} discrepancies - needs investigation", stats.discrepancies_found ) );
    }

    if ( stats.avg_speedup > 2.0 )
    {
      print_success( "Class implementation shows significant performance improvement" );
      fmt::print( "   Recommendation: CONSIDER class implementation for performance-critical applications\n" );
    }
    else if ( stats.avg_speedup > 1.0 )
    {
      print_success( "Class implementation shows moderate performance improvement" );
      fmt::print( "   Recommendation: EVALUATE based on specific use case requirements\n" );
    }
    else
    {
      print_warning( "Class implementation shows no performance improvement" );
      fmt::print( "   Recommendation: PREFER function implementation for simplicity\n" );
    }

    // Memory considerations
    fmt::print( "\n💾 MEMORY CONSIDERATIONS:\n" );
    fmt::print(
      "   Class implementation uses ~{} KB additional memory for 1M points\n",
      ( static_cast<size_t>( sqrt( 1000000 ) ) * 2 * sizeof( integer ) ) / 1024 );
    fmt::print( "   Function implementation: 0 KB additional memory\n" );
    fmt::print( "   Trade-off: Memory vs Performance - choose based on application constraints\n" );

    // Use case guidance
    fmt::print( "\n🎯 USE CASE GUIDANCE:\n" );
    fmt::print( "   Choose Function Implementation when:\n" );
    fmt::print( "   - Memory is constrained\n" );
    fmt::print( "   - Simple deployment is preferred\n" );
    fmt::print( "   - Thread safety without synchronization is needed\n\n" );

    fmt::print( "   Choose Class Implementation when:\n" );
    fmt::print( "   - Performance is critical\n" );
    fmt::print( "   - Many queries on same dataset are expected\n" );
    fmt::print( "   - Memory overhead is acceptable\n" );

    // Final verdict
    fmt::print( "\n🏁 FINAL VERDICT:\n" );
    if ( stats.discrepancies_found == 0 && stats.avg_speedup > 1.5 )
    {
      print_success( "✅ CLASS IMPLEMENTATION PASSES ALL TESTS - RECOMMENDED FOR PERFORMANCE-CRITICAL APPLICATIONS" );
    }
    else if ( stats.discrepancies_found == 0 && stats.avg_speedup > 1.0 )
    {
      print_success( "✅ BOTH IMPLEMENTATIONS VALID - CHOOSE BASED ON SPECIFIC REQUIREMENTS" );
    }
    else if ( stats.discrepancies_found > 0 )
    {
      print_error( "❌ IMPLEMENTATIONS HAVE DISCREPANCIES - FIX BEFORE PRODUCTION USE" );
    }
    else
    {
      print_success( "✅ FUNCTION IMPLEMENTATION PREFERRED FOR SIMPLICITY AND MEMORY EFFICIENCY" );
    }
  }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main()
{
  try
  {
    // Run unified test suite
    UnifiedTestSuite test_suite;
    test_suite.run_all_tests();
    test_suite.print_final_report();

    // Run advanced performance tests (for both closed and non-closed cases)
    AdvancedPerformanceTester perf_tester;
    perf_tester.run_all_tests();
  }
  catch ( const exception & e )
  {
    print_error( fmt::format( "Fatal error during tests: {}", e.what() ) );
    return 1;
  }
  print_header( "ALL DONE", fmt::color::green );
  return 0;
}
