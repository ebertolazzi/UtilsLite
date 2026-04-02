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
\*--------------------------------------------------------------------------*/

#include <random>
#include <vector>
#include <string>
#include <algorithm>

#include "Utils.hh"
#include "Utils_eigen.hh"
#include "Utils_TicToc.hh"

using namespace std;
using integer   = int;
using real_type = double;

// Unicode symbols
const std::string CHECK_MARK  = "✓";
const std::string CROSS_MARK  = "✗";
const std::string ARROW_RIGHT = "→";
const std::string MICROSECOND = "μs";
const std::string NANOSECOND  = "ns";

static unsigned     seed1 = 2;
static std::mt19937 generator( seed1 );

static real_type rand( real_type const xmin, real_type const xmax )
{
  real_type const random{ static_cast<real_type>( generator() ) / generator.max() };
  return xmin + ( xmax - xmin ) * random;
}

using dmat_t = Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic>;
using dvec_t = Eigen::Matrix<real_type, Eigen::Dynamic, 1>;

struct BenchmarkResult
{
  std::string test_name;
  int         size;
  double      time_dynamic;
  double      time_map_dynamic;
  double      time_fixed;
  double      time_fixed_map;
  std::string unit;
};

class BenchmarkRunner
{
private:
  std::vector<BenchmarkResult> results;

public:
  template <int N> BenchmarkResult runVV()
  {
    int const    N_TIMES{ 1000000 / N };
    double const to_ps{ 1000000.0 / N_TIMES };

    BenchmarkResult result;
    result.test_name = "Vector-Vector AXPY";
    result.size      = N;
    result.unit      = "ps/op";

    using vecN_t = Eigen::Matrix<real_type, N, 1>;

    Utils::Malloc<real_type> baseValue( "real" );
    baseValue.allocate( N * 10 );

    real_type * V1{ baseValue( N ) };
    real_type * V2{ baseValue( N ) };
    real_type * V3{ baseValue( N ) };

    // Initialize data
    for ( int i = 0; i < N; ++i )
    {
      V1[i] = rand( -1, 1 );
      V2[i] = rand( -1, 1 );
      V3[i] = rand( -1, 1 );
    }

    Utils::TicToc tm;

    // Test 1: Eigen dynamic
    {
      dvec_t dv1( N ), dv2( N ), dv3( N );
      for ( int i = 0; i < N; ++i )
      {
        dv1( i ) = V1[i];
        dv2( i ) = V2[i];
        dv3( i ) = V3[i];
      }

      tm.tic();
      for ( int i = 1; i <= N_TIMES; ++i )
      {
        real_type alpha = 1.0 / i;
        dv3.noalias()   = dv2 + alpha * dv1;
        dv1             = dv3;
      }
      tm.toc();
      result.time_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 2: Eigen map dynamic
    {
      Eigen::Map<dvec_t> vv1( V1, N );
      Eigen::Map<dvec_t> vv2( V2, N );
      Eigen::Map<dvec_t> vv3( V3, N );

      tm.tic();
      for ( int i = 1; i <= N_TIMES; ++i )
      {
        real_type alpha = 1.0 / i;
        vv3.noalias()   = vv2 + alpha * vv1;
        vv1             = vv3;
      }
      tm.toc();
      result.time_map_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 3: Eigen fixed
    {
      vecN_t v1, v2, v3;
      for ( int i = 0; i < N; ++i )
      {
        v1( i ) = V1[i];
        v2( i ) = V2[i];
        v3( i ) = V3[i];
      }

      tm.tic();
      for ( int i = 1; i <= N_TIMES; ++i )
      {
        real_type alpha = 1.0 / i;
        v3.noalias()    = v2 + alpha * v1;
        v2              = v3;
      }
      tm.toc();
      result.time_fixed = to_ps * tm.elapsed_ms();
    }

    // Test 4: Eigen fixed map
    {
      Eigen::Map<vecN_t> vv1( V1 );
      Eigen::Map<vecN_t> vv2( V2 );
      Eigen::Map<vecN_t> vv3( V3 );

      tm.tic();
      for ( int i = 1; i <= N_TIMES; ++i )
      {
        real_type alpha = 1.0 / i;
        vv3.noalias()   = vv2 + alpha * vv1;
        vv1             = vv3;
      }
      tm.toc();
      result.time_fixed_map = to_ps * tm.elapsed_ms();
    }

    return result;
  }

  template <int N> BenchmarkResult runMV()
  {
    int    N_TIMES = 1000000 / N;
    double to_ps   = 1000000.0 / N_TIMES;

    BenchmarkResult result;
    result.test_name = "Matrix-Vector Mult";
    result.size      = N;
    result.unit      = "ps/op";

    using matN_t = Eigen::Matrix<real_type, N, N>;
    using vecN_t = Eigen::Matrix<real_type, N, 1>;

    Utils::Malloc<real_type> baseValue( "real" );
    baseValue.allocate( N * N * 10 );

    real_type * M = baseValue( N * N );
    real_type * V = baseValue( N );
    real_type * R = baseValue( N );

    // Initialize matrix
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j ) { M[i + j * N] = rand( -1, 1 ); }
    }
    for ( int i = 0; i < N; ++i )
    {
      V[i] = rand( -1, 1 );
      R[i] = rand( -1, 1 );
    }

    Utils::TicToc tm;

    // Test 1: Eigen dynamic
    {
      dmat_t dm( N, N );
      dvec_t dv( N ), dr( N );

      for ( int i = 0; i < N; ++i )
      {
        dv( i ) = V[i];
        dr( i ) = R[i];
        for ( int j = 0; j < N; ++j ) { dm( i, j ) = M[i + j * N]; }
      }

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        dr.noalias() -= dm * dv;
        dv = dr;
      }
      tm.toc();
      result.time_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 2: Eigen map dynamic
    {
      Eigen::Map<dmat_t> mm( M, N, N );
      Eigen::Map<dvec_t> vv( V, N );
      Eigen::Map<dvec_t> rr( R, N );

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        rr.noalias() -= mm * vv;
        vv = rr;
      }
      tm.toc();
      result.time_map_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 3: Eigen fixed
    {
      matN_t m;
      vecN_t v, r;

      for ( int i = 0; i < N; ++i )
      {
        v( i ) = V[i];
        r( i ) = R[i];
        for ( int j = 0; j < N; ++j ) { m( i, j ) = M[i + j * N]; }
      }

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        r.noalias() -= m * v;
        v = r;
      }
      tm.toc();
      result.time_fixed = to_ps * tm.elapsed_ms();
    }

    // Test 4: Eigen fixed map
    {
      Eigen::Map<matN_t> mm( M );
      Eigen::Map<vecN_t> vv( V );
      Eigen::Map<vecN_t> rr( R );

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        rr.noalias() -= mm * vv;
        vv = rr;
      }
      tm.toc();
      result.time_fixed_map = to_ps * tm.elapsed_ms();
    }

    return result;
  }

  template <int N> BenchmarkResult runMM()
  {
    int    N_TIMES = 10000 / N;
    double to_ps   = 10000.0 / N_TIMES;

    BenchmarkResult result;
    result.test_name = "Matrix-Matrix Mult";
    result.size      = N;
    result.unit      = "ps/op";

    using matN_t = Eigen::Matrix<real_type, N, N>;

    Utils::Malloc<real_type> baseValue( "real" );
    baseValue.allocate( N * N * 10 );

    real_type * M1 = baseValue( N * N );
    real_type * M2 = baseValue( N * N );
    real_type * M3 = baseValue( N * N );

    // Initialize matrices
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        int idx = i + j * N;
        M1[idx] = rand( -1, 1 );
        M2[idx] = rand( -1, 1 );
        M3[idx] = rand( -1, 1 );
      }
    }

    Utils::TicToc tm;

    // Test 1: Eigen dynamic
    {
      dmat_t dm1( N, N ), dm2( N, N ), dm3( N, N );

      for ( int i = 0; i < N; ++i )
      {
        for ( int j = 0; j < N; ++j )
        {
          int idx     = i + j * N;
          dm1( i, j ) = M1[idx];
          dm2( i, j ) = M2[idx];
          dm3( i, j ) = M3[idx];
        }
      }

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        dm3.noalias() -= dm1 * dm2;
        dm2 = dm3;
      }
      tm.toc();
      result.time_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 2: Eigen map dynamic
    {
      Eigen::Map<dmat_t> mm1( M1, N, N );
      Eigen::Map<dmat_t> mm2( M2, N, N );
      Eigen::Map<dmat_t> mm3( M3, N, N );

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        mm3.noalias() -= mm1 * mm2;
        mm2 = mm3;
      }
      tm.toc();
      result.time_map_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 3: Eigen fixed
    {
      matN_t m1, m2, m3;

      for ( int i = 0; i < N; ++i )
      {
        for ( int j = 0; j < N; ++j )
        {
          int idx    = i + j * N;
          m1( i, j ) = M1[idx];
          m2( i, j ) = M2[idx];
          m3( i, j ) = M3[idx];
        }
      }

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        m3.noalias() -= m1 * m2;
        m2 = m3;
      }
      tm.toc();
      result.time_fixed = to_ps * tm.elapsed_ms();
    }

    // Test 4: Eigen fixed map
    {
      Eigen::Map<matN_t> mm1( M1 );
      Eigen::Map<matN_t> mm2( M2 );
      Eigen::Map<matN_t> mm3( M3 );

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        mm3.noalias() -= mm1 * mm2;
        mm2 = mm3;
      }
      tm.toc();
      result.time_fixed_map = to_ps * tm.elapsed_ms();
    }

    return result;
  }

  template <int N> BenchmarkResult runCopy()
  {
    int    N_TIMES = 1000000 / N;
    double to_ps   = 1000000.0 / N_TIMES;

    BenchmarkResult result;
    result.test_name = "Matrix Copy";
    result.size      = N;
    result.unit      = "ps/op";

    using matN_t = Eigen::Matrix<real_type, N, N>;

    Utils::Malloc<real_type> baseValue( "real" );
    baseValue.allocate( N * N * 10 );

    real_type * M1 = baseValue( N * N );
    real_type * M2 = baseValue( N * N );

    // Initialize matrix
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        M1[i + j * N] = rand( -1, 1 );
        M2[i + j * N] = 0.0;
      }
    }

    Utils::TicToc tm;

    // Test 1: Eigen dynamic
    {
      dmat_t dm1( N, N ), dm2( N, N );

      for ( int i = 0; i < N; ++i )
      {
        for ( int j = 0; j < N; ++j ) { dm1( i, j ) = M1[i + j * N]; }
      }

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        dm2 = dm1;
        dm1( 0, 0 ) += dm2( 0, 0 );
        dm2 = dm1;
        dm1( 0, 0 ) += dm2( 0, 0 );
        dm2 = dm1;
        dm1( 0, 0 ) += dm2( 0, 0 );
        dm2 = dm1;
        dm1( 0, 0 ) += dm2( 0, 0 );
        dm2 = dm1;
        dm1( 0, 0 ) += dm2( 0, 0 );
      }
      tm.toc();
      result.time_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 2: Eigen map dynamic
    {
      Eigen::Map<dmat_t> mm1( M1, N, N );
      Eigen::Map<dmat_t> mm2( M2, N, N );

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
      }
      tm.toc();
      result.time_map_dynamic = to_ps * tm.elapsed_ms();
    }

    // Test 3: Eigen fixed
    {
      matN_t m1, m2;

      for ( int i = 0; i < N; ++i )
      {
        for ( int j = 0; j < N; ++j ) { m1( i, j ) = M1[i + j * N]; }
      }

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        m2 = m1;
        m1( 0, 0 ) += m2( 0, 0 );
        m2 = m1;
        m1( 0, 0 ) += m2( 0, 0 );
        m2 = m1;
        m1( 0, 0 ) += m2( 0, 0 );
        m2 = m1;
        m1( 0, 0 ) += m2( 0, 0 );
        m2 = m1;
        m1( 0, 0 ) += m2( 0, 0 );
      }
      tm.toc();
      result.time_fixed = to_ps * tm.elapsed_ms();
    }

    // Test 4: Eigen fixed map
    {
      Eigen::Map<matN_t> mm1( M1 );
      Eigen::Map<matN_t> mm2( M2 );

      tm.tic();
      for ( int i = 0; i < N_TIMES; ++i )
      {
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
        mm2 = mm1;
        mm1( 0, 0 ) += mm2( 0, 0 );
      }
      tm.toc();
      result.time_fixed_map = to_ps * tm.elapsed_ms();
    }

    return result;
  }

  void addResult( const BenchmarkResult & result ) { results.push_back( result ); }

  void printHeader( const std::string & title )
  {
    fmt::print( fg( fmt::color::steel_blue ) | fmt::emphasis::bold, "\n{:=^80}\n", " " + title + " " );
    fmt::print( "\n" );
  }

  void printTable()
  {
    // Group results by test type
    std::vector<std::string> test_types = { "Vector-Vector AXPY",
                                            "Matrix-Vector Mult",
                                            "Matrix-Matrix Mult",
                                            "Matrix Copy" };

    for ( const auto & test_type : test_types )
    {
      auto it = std::find_if(
        results.begin(),
        results.end(),
        [&]( const BenchmarkResult & r ) { return r.test_name == test_type; } );
      if ( it == results.end() ) continue;

      printHeader( test_type );

      // Print table header
      fmt::print(
        fg( fmt::color::cyan ) | fmt::emphasis::bold,
        "{:<8} {:<15} {:<15} {:<15} {:<15}\n",
        "Size",
        "Dynamic",
        "Map Dynamic",
        "Fixed",
        "Fixed Map" );
      fmt::print( "{:-<80}\n", "" );

      // Find all results for this test type
      std::vector<BenchmarkResult> filtered;
      std::copy_if(
        results.begin(),
        results.end(),
        std::back_inserter( filtered ),
        [&]( const BenchmarkResult & r ) { return r.test_name == test_type; } );

      // Sort by size
      std::sort(
        filtered.begin(),
        filtered.end(),
        []( const BenchmarkResult & a, const BenchmarkResult & b ) { return a.size < b.size; } );

      // Print rows
      for ( const auto & res : filtered )
      {
        // Find the best (lowest) time
        std::vector<double> times     = { res.time_dynamic, res.time_map_dynamic, res.time_fixed, res.time_fixed_map };
        double              best_time = *std::min_element( times.begin(), times.end() );

        auto format_time = [&]( double time, const std::string & unit )
        {
          std::string formatted = fmt::format( "{:8.2f} {}", time, unit );
          if ( std::abs( time - best_time ) < 1e-6 )
          {
            return fmt::format( fg( fmt::color::green ) | fmt::emphasis::bold, "{}", formatted );
          }
          return formatted;
        };

        fmt::print(
          "{:<8} {:<15} {:<15} {:<15} {:<15}\n",
          res.size,
          format_time( res.time_dynamic, res.unit ),
          format_time( res.time_map_dynamic, res.unit ),
          format_time( res.time_fixed, res.unit ),
          format_time( res.time_fixed_map, res.unit ) );
      }
    }
  }

  void printSummary()
  {
    printHeader( "Benchmark Summary" );

    // Calculate average speedup for fixed vs dynamic
    double total_speedup_fixed     = 0.0;
    double total_speedup_fixed_map = 0.0;
    int    count                   = 0;

    for ( const auto & res : results )
    {
      if ( res.time_dynamic > 0 )
      {
        total_speedup_fixed += res.time_dynamic / res.time_fixed;
        total_speedup_fixed_map += res.time_dynamic / res.time_fixed_map;
        count++;
      }
    }

    double avg_speedup_fixed     = total_speedup_fixed / count;
    double avg_speedup_fixed_map = total_speedup_fixed_map / count;

    fmt::print( fg( fmt::color::gold ) | fmt::emphasis::bold, "Average Speedup (vs Dynamic):\n" );
    fmt::print( "  Fixed Size:      {:.2f}x\n", avg_speedup_fixed );
    fmt::print( "  Fixed Size Map:  {:.2f}x\n", avg_speedup_fixed_map );

    // Find best performing configuration
    std::vector<std::string> configs = { "Dynamic", "Map Dynamic", "Fixed", "Fixed Map" };
    std::vector<int>         wins( configs.size(), 0 );

    for ( const auto & res : results )
    {
      std::vector<double> times  = { res.time_dynamic, res.time_map_dynamic, res.time_fixed, res.time_fixed_map };
      auto                min_it = std::min_element( times.begin(), times.end() );
      int                 idx    = static_cast<int>( std::distance( times.begin(), min_it ) );
      wins[idx]++;
    }

    fmt::print( fg( fmt::color::gold ) | fmt::emphasis::bold, "\nBest Performance Count:\n" );
    for ( size_t i = 0; i < configs.size(); ++i )
    {
      fmt::print(
        "  {}: {} {}\n",
        configs[i],
        wins[i],
        wins[i] == *std::max_element( wins.begin(), wins.end() ) ? fmt::format( fg( fmt::color::green ), "🏆" ) : "" );
    }
  }
};

void runAllBenchmarks()
{
  fmt::print( fg( fmt::color::orange ) | fmt::emphasis::bold, "\n🚀 Eigen Performance Benchmark Suite\n" );
  fmt::print( "Threads: {}\n", Eigen::nbThreads() );

  BenchmarkRunner runner;

  // Run VV benchmarks
  std::vector<int> vv_sizes = { 2, 4, 6, 8, 12, 16, 100, 1000 };
  for ( int size : vv_sizes )
  {
    switch ( size )
    {
      case 2: runner.addResult( runner.runVV<2>() ); break;
      case 4: runner.addResult( runner.runVV<4>() ); break;
      case 6: runner.addResult( runner.runVV<6>() ); break;
      case 8: runner.addResult( runner.runVV<8>() ); break;
      case 12: runner.addResult( runner.runVV<12>() ); break;
      case 16: runner.addResult( runner.runVV<16>() ); break;
      case 100: runner.addResult( runner.runVV<100>() ); break;
      case 1000: runner.addResult( runner.runVV<1000>() ); break;
    }
  }

  // Run MV benchmarks
  std::vector<int> mv_sizes = { 2, 4, 6, 8, 12, 16, 100 };
  for ( int size : mv_sizes )
  {
    switch ( size )
    {
      case 2: runner.addResult( runner.runMV<2>() ); break;
      case 4: runner.addResult( runner.runMV<4>() ); break;
      case 6: runner.addResult( runner.runMV<6>() ); break;
      case 8: runner.addResult( runner.runMV<8>() ); break;
      case 12: runner.addResult( runner.runMV<12>() ); break;
      case 16: runner.addResult( runner.runMV<16>() ); break;
      case 100: runner.addResult( runner.runMV<100>() ); break;
    }
  }

  // Run MM benchmarks
  std::vector<int> mm_sizes = { 2, 4, 6, 8, 12, 16, 100 };
  for ( int size : mm_sizes )
  {
    switch ( size )
    {
      case 2: runner.addResult( runner.runMM<2>() ); break;
      case 4: runner.addResult( runner.runMM<4>() ); break;
      case 6: runner.addResult( runner.runMM<6>() ); break;
      case 8: runner.addResult( runner.runMM<8>() ); break;
      case 12: runner.addResult( runner.runMM<12>() ); break;
      case 16: runner.addResult( runner.runMM<16>() ); break;
      case 100: runner.addResult( runner.runMM<100>() ); break;
    }
  }

  // Run Copy benchmarks
  std::vector<int> copy_sizes = { 2, 4, 6, 8, 12, 16, 100 };
  for ( int size : copy_sizes )
  {
    switch ( size )
    {
      case 2: runner.addResult( runner.runCopy<2>() ); break;
      case 4: runner.addResult( runner.runCopy<4>() ); break;
      case 6: runner.addResult( runner.runCopy<6>() ); break;
      case 8: runner.addResult( runner.runCopy<8>() ); break;
      case 12: runner.addResult( runner.runCopy<12>() ); break;
      case 16: runner.addResult( runner.runCopy<16>() ); break;
      case 100: runner.addResult( runner.runCopy<100>() ); break;
    }
  }

  // Print results
  runner.printTable();
  runner.printSummary();
}

int main()
{
  // Set Eigen threading
  Eigen::setNbThreads( 4 );

  try
  {
    runAllBenchmarks();

    fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n{} Benchmark completed successfully!\n", CHECK_MARK );
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n{} Error: {}\n", CROSS_MARK, e.what() );
    return 1;
  }

  return 0;
}
