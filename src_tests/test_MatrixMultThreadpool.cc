// ================================================
// MATRIX MULTIPLICATION WITH BLOCK PARTITIONING
// ================================================
//
// Write a C++11 or later program that utilizes the Eigen3 library to perform
// matrix-matrix multiplication using block partitioning. Given matrices A and
// B, compute the matrix C = A * B. Matrices A and B must be compatible for
// multiplication. Given the integers N, P, M, partition the matrices as
// follows:
//  - Matrix A into N x P blocks
//  - Matrix B into P x M blocks
//  - Matrix C into N x M blocks
// Ensure that the partitioning is compatible. If matrices A and B are
// incompatible, or if the required partitioning (N, P, M) is not possible,
// throw an exception. Each (i, j) block of matrix C must be computed on a
// separate thread if available, enabling parallel code execution. Finally,
// compare the execution speed of your block partitioning matrix multiplication
// with the timing of the standard Eigen3 matrix multiplication command. Use the
// proposed ThreadPool to find a better one to perform parallel tasks.

#include "Utils_eigen.hh"
#include "Utils_fmt.hh"
#include "Utils_TicToc.hh"

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <mutex>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

using mat     = Eigen::MatrixXd;
using integer = Eigen::Index;

std::random_device              rd;
std::mt19937                    gen( rd() );
std::uniform_int_distribution<> distrib( 2, 25 );

// Structure to store benchmark results
struct BenchmarkResult
{
  std::string name;
  double      mean_time;
  double      std_dev;
  double      error_norm;
  bool        is_standard;
  double      speedup;
  double      flop_rate;      // FLOP per millisecond (GFLOP/s * 1e6)
  double      time_per_cost;  // Tempo medio / Costo (ms / GFLOP)
};

class BlockMult
{
  std::mutex             mtx;
  Utils::ThreadPool0     Pool0{ 5 };
  Utils::ThreadPool1     Pool1{ 5 };
  Utils::ThreadPool2     Pool2{ 5 };
  Utils::ThreadPool3     Pool3{ 5 };
  Utils::ThreadPool4     Pool4{ 5 };
  Utils::ThreadPool5     Pool5{ 5 };
  Utils::ThreadPoolEigen PoolEigen{ 5 };

  const std::vector<integer> * m_i_block{ nullptr };
  const std::vector<integer> * m_j_block{ nullptr };
  const std::vector<integer> * m_k_block{ nullptr };

  void Compute_C_block( const mat & A, const mat & B, mat & C, integer i, integer j ) const;

public:
  BlockMult() = default;

  bool multiply(
    integer                      ntp,
    const mat &                  A,
    const mat &                  B,
    mat &                        C,
    const std::vector<integer> & i_block,
    const std::vector<integer> & j_block,
    const std::vector<integer> & k_block );

  ~BlockMult() = default;
};

void BlockMult::Compute_C_block( const mat & A, const mat & B, mat & C, integer const i, integer const j ) const
{
  auto const II = Eigen::seqN( ( *m_i_block )[i - 1], ( *m_i_block )[i] - ( *m_i_block )[i - 1] );
  auto const JJ = Eigen::seqN( ( *m_j_block )[j - 1], ( *m_j_block )[j] - ( *m_j_block )[j - 1] );

  for ( size_t k{ 1 }; k < m_k_block->size(); ++k )
  {
    auto KK = Eigen::seqN( ( *m_k_block )[k - 1], ( *m_k_block )[k] - ( *m_k_block )[k - 1] );
    C( II, JJ ) += A( II, KK ) * B( KK, JJ );
  }
}

bool BlockMult::multiply(
  integer const                ntp,
  const mat &                  A,
  const mat &                  B,
  mat &                        C,
  const std::vector<integer> & i_block,
  const std::vector<integer> & j_block,
  const std::vector<integer> & k_block )
{
  if ( A.cols() != B.rows() )
  {
    fmt::print(
      fg( fmt::terminal_color::red ) | fmt::emphasis::bold,
      "[ERROR] Invalid matrix multiplication. Found {} x {} times {} x {}\n",
      A.rows(),
      A.cols(),
      B.rows(),
      B.cols() );
    return false;
  }

  m_i_block = &i_block;
  m_j_block = &j_block;
  m_k_block = &k_block;

  C.setZero();

#define THE_TASK( POOL )                                                                                \
  for ( integer i{ 1 }; i < static_cast<integer>( i_block.size() ); ++i )                               \
  {                                                                                                     \
    for ( integer j{ 1 }; j < static_cast<integer>( j_block.size() ); ++j )                             \
    {                                                                                                   \
      POOL.run( &BlockMult::Compute_C_block, this, std::ref( A ), std::ref( B ), std::ref( C ), i, j ); \
    }                                                                                                   \
  }                                                                                                     \
  POOL.wait()

  switch ( ntp )
  {
    case 0: THE_TASK( Pool0 ); break;
    case 1: THE_TASK( Pool1 ); break;
    case 2: THE_TASK( Pool2 ); break;
    case 3: THE_TASK( Pool3 ); break;
    case 4: THE_TASK( Pool4 ); break;
    case 5: THE_TASK( Pool5 ); break;
    case 6: THE_TASK( PoolEigen ); break;
    default: fmt::print( fg( fmt::terminal_color::red ), "ERROR: Invalid thread pool ID\n" ); return false;
  }

  return true;
}

// Calculate FLOP cost for matrix multiplication
double calculate_flop_cost( int N, int P, int M )
{
  // Matrix multiplication A[N x P] * B[P x M] = C[N x M]
  // Each element of C requires P multiplications and P-1 additions
  // So total FLOPs = N * M * (2P - 1) ≈ 2 * N * P * M for large P
  return 2.0 * N * P * M;
}

// Function to print a table row
void print_table_row( const BenchmarkResult & result, bool is_fastest = false )
{
  // Colors for different fields
  auto name_style = result.is_standard ? fmt::emphasis::bold | fg( fmt::terminal_color::bright_white )
                                       : fg( fmt::terminal_color::white );

  if ( is_fastest ) { name_style = fmt::emphasis::bold | fg( fmt::terminal_color::bright_green ); }

  // Format values
  std::string mean_str          = fmt::format( "{:>9.3g}", result.mean_time );
  std::string std_str           = fmt::format( "{:>9.3g}", result.std_dev );
  std::string error_str         = result.is_standard ? "      N/A" : fmt::format( "{:>9.2g}", result.error_norm );
  std::string speedup_str       = result.is_standard ? "     1.00x" : fmt::format( "{:>9.2g}x", result.speedup );
  std::string flop_rate_str     = fmt::format( "{:>10.4g}", result.flop_rate );
  std::string time_per_cost_str = fmt::format( "{:>10.4g}", result.time_per_cost );

  // Print row
  fmt::print( "│ " );
  fmt::print( name_style, "{:<25}", result.name );
  fmt::print( " │ " );
  fmt::print( fg( fmt::terminal_color::cyan ), "{}", mean_str );
  fmt::print( " │ " );
  fmt::print( fg( fmt::terminal_color::yellow ), "{}", std_str );
  fmt::print( " │ " );
  fmt::print( fg( fmt::terminal_color::magenta ), "{}", error_str );
  fmt::print( " │ " );
  fmt::print( fg( fmt::terminal_color::green ), "{}", speedup_str );
  fmt::print( " │ " );
  fmt::print( fg( fmt::terminal_color::bright_cyan ), "{}", flop_rate_str );
  fmt::print( " │ " );
  fmt::print( fg( fmt::terminal_color::bright_magenta ), "{}", time_per_cost_str );
  fmt::print( " │\n" );
}

// Function to print table header
void print_table_header()
{
  auto header_style = fg( fmt::terminal_color::bright_blue ) | fmt::emphasis::bold;

  fmt::print(
    header_style,
    "┌───────────────────────────┬───────────┬───────────┬───────────┬────────────┬────────────┬────────────┐\n" );
  fmt::print(
    header_style,
    "│ {:<25} │ {:>9} │ {:>9} │ {:>9} │ {:>10} │ {:>10} │ {:>10} │\n",
    "Method",
    "Mean (ms)",
    "Std Dev",
    "Error",
    "Speedup",
    "GFLOP/s",
    "Time/Cost" );
  fmt::print(
    header_style,
    "├───────────────────────────┼───────────┼───────────┼───────────┼────────────┼────────────┼────────────┤\n" );
}

// Function to print table footer
void print_table_footer()
{
  auto footer_style = fg( fmt::terminal_color::bright_blue ) | fmt::emphasis::bold;
  fmt::print(
    footer_style,
    "└───────────────────────────┴───────────┴───────────┴───────────┴────────────┴────────────┴────────────┘\n" );
}

// Function to run benchmark for a specific matrix size
void run_benchmark_for_size( int N, int P, int M, int n_runs = 5 )
{
  fmt::print(
    fg( fmt::terminal_color::bright_cyan ) | fmt::emphasis::bold,
    "\n\n\n"
    "════════════════════ Matrix Size: A[{}x{}] x B[{}x{}] = C[{}x{}] ════════════════════\n",
    N,
    P,
    P,
    M,
    N,
    M );

  // Calculate theoretical FLOP cost
  double flop_cost  = calculate_flop_cost( N, P, M );
  double gflop_cost = flop_cost / 1e9;  // Convert to GFLOP

  fmt::print(
    fg( fmt::terminal_color::bright_yellow ),
    "Theoretical FLOP cost: {:.2e} FLOP ({:.2f} GFLOP)\n",
    flop_cost,
    gflop_cost );

  // Allocate and initialize matrices
  fmt::print( fg( fmt::terminal_color::bright_cyan ), "Initializing matrices...\n" );
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random( N, P );
  Eigen::MatrixXd M2 = Eigen::MatrixXd::Random( P, M );
  Eigen::MatrixXd M3a( N, M ), M3b( N, M );

  // Vector for results
  std::vector<BenchmarkResult> results;

  // 1. Benchmark: Standard Eigen multiplication
  fmt::print( fg( fmt::terminal_color::bright_yellow ), "Running Standard Eigen Multiplication...\n" );

  Eigen::VectorXd times( n_runs );
  Utils::TicToc   tm;

  for ( int i = 0; i < n_runs; i++ )
  {
    tm.tic();
    M3a = M1 * M2;
    tm.toc();
    times( i ) = tm.elapsed_ms();
  }

  double mean          = times.mean();
  double stdDev        = std::sqrt( ( times.array() - mean ).square().sum() / ( n_runs - 1 ) );
  double flop_rate     = gflop_cost / ( mean / 1000.0 );  // GFLOP per second
  double time_per_cost = mean / gflop_cost;               // ms per GFLOP

  results.push_back( { "Standard Eigen", mean, stdDev, 0.0, true, 1.0, flop_rate, time_per_cost } );

  // Prepare block partitions
  fmt::print( fg( fmt::terminal_color::bright_cyan ), "Generating block partitions...\n" );

  std::vector<integer> i_block, j_block, k_block;

  // Blocks for rows (N)
  i_block.push_back( 0 );
  while ( i_block.back() < N ) { i_block.push_back( i_block.back() + distrib( gen ) ); }
  i_block.back() = N;

  // Blocks for columns (M)
  j_block.push_back( 0 );
  while ( j_block.back() < M ) { j_block.push_back( j_block.back() + distrib( gen ) ); }
  j_block.back() = M;

  // Blocks for inner dimension (P)
  k_block.push_back( 0 );
  while ( k_block.back() < P ) { k_block.push_back( k_block.back() + distrib( gen ) ); }
  k_block.back() = P;

  fmt::print(
    "  Blocks: {} x {} x {} (Total C blocks: {})\n",
    i_block.size() - 1,
    j_block.size() - 1,
    k_block.size() - 1,
    ( i_block.size() - 1 ) * ( j_block.size() - 1 ) );

  // 2. Benchmark: Multiplication with various ThreadPools
  fmt::print( fg( fmt::terminal_color::bright_yellow ), "Running Block Partitioning Tests...\n" );

  std::vector<std::string> pool_names = { Utils::ThreadPool0::Name(),    Utils::ThreadPool1::Name(),
                                          Utils::ThreadPool2::Name(),    Utils::ThreadPool3::Name(),
                                          Utils::ThreadPool4::Name(),    Utils::ThreadPool5::Name(),
                                          Utils::ThreadPoolEigen::Name() };

  for ( int nptp = 0; nptp <= 6; ++nptp )
  {
    fmt::print( "  Testing {}...", pool_names[nptp] );

    BlockMult BM;
    bool      success = true;
    for ( int i = 0; i < n_runs; ++i )
    {
      tm.tic();
      success = BM.multiply( nptp, M1, M2, M3b, i_block, j_block, k_block );
      tm.toc();

      if ( !success )
      {
        fmt::print( fg( fmt::terminal_color::red ), " Failed!\n" );
        break;
      }

      times( i ) = tm.elapsed_ms();
    }

    if ( success )
    {
      mean              = times.mean();
      stdDev            = std::sqrt( ( times.array() - mean ).square().sum() / ( n_runs - 1 ) );
      double error_norm = ( M3a - M3b ).norm();
      double speedup    = results[0].mean_time / mean;
      flop_rate         = gflop_cost / ( mean / 1000.0 );  // GFLOP per second
      time_per_cost     = mean / gflop_cost;               // ms per GFLOP

      results.push_back( { pool_names[nptp], mean, stdDev, error_norm, false, speedup, flop_rate, time_per_cost } );
      fmt::print(
        fg( fmt::terminal_color::green ),
        " Done ({:.2f} ms, {:.2f}x, {:.2f} GFLOP/s)\n",
        mean,
        speedup,
        flop_rate );
    }
  }

  // Find the fastest method (excluding standard)
  double fastest_time  = std::numeric_limits<double>::max();
  int    fastest_index = -1;

  for ( size_t i = 1; i < results.size(); ++i )
  {
    if ( results[i].mean_time < fastest_time )
    {
      fastest_time  = results[i].mean_time;
      fastest_index = static_cast<int>( i );
    }
  }

  // Find the most efficient method (highest GFLOP/s)
  double highest_flop_rate    = 0.0;
  int    most_efficient_index = -1;

  for ( size_t i = 1; i < results.size(); ++i )
  {
    if ( results[i].flop_rate > highest_flop_rate )
    {
      highest_flop_rate    = results[i].flop_rate;
      most_efficient_index = static_cast<int>( i );
    }
  }

  // Print results table
  fmt::print( fg( fmt::terminal_color::bright_green ) | fmt::emphasis::bold, "\nBenchmark Results:\n" );

  print_table_header();

  for ( size_t i = 0; i < results.size(); ++i )
  {
    bool is_fastest        = ( i == static_cast<size_t>( fastest_index ) );
    bool is_most_efficient = ( i == static_cast<size_t>( most_efficient_index ) );

    // Highlight if fastest OR most efficient
    print_table_row( results[i], is_fastest || is_most_efficient );
  }

  print_table_footer();

  // Additional information
  if ( fastest_index != -1 )
  {
    fmt::print(
      fg( fmt::terminal_color::bright_green ) | fmt::emphasis::bold,
      "  Fastest Method: {} ({:.2f}x speedup, {:.2f} GFLOP/s)\n",
      results[fastest_index].name,
      results[fastest_index].speedup,
      results[fastest_index].flop_rate );
  }

  if ( most_efficient_index != -1 && most_efficient_index != fastest_index )
  {
    fmt::print(
      fg( fmt::terminal_color::bright_cyan ) | fmt::emphasis::bold,
      "  Most Efficient: {} ({:.2f} GFLOP/s, {:.2f}x speedup)\n",
      results[most_efficient_index].name,
      results[most_efficient_index].flop_rate,
      results[most_efficient_index].speedup );
  }

  // Validation of results
  fmt::print( fg( fmt::terminal_color::bright_yellow ), "\nVerification:\n" );

  bool all_valid = true;
  for ( size_t i = 1; i < results.size(); ++i )
  {
    if ( results[i].error_norm > 1e-6 )
    {
      fmt::print( fg( fmt::terminal_color::red ), "  {}: Error = {:.2e}\n", results[i].name, results[i].error_norm );
      all_valid = false;
    }
    else
    {
      fmt::print( fg( fmt::terminal_color::green ), "  {}: Error = {:.2e}\n", results[i].name, results[i].error_norm );
    }
  }

  if ( all_valid ) { fmt::print( fg( fmt::terminal_color::bright_green ), "  All tests passed successfully!\n" ); }
  else
  {
    fmt::print(
      fg( fmt::terminal_color::red ) | fmt::emphasis::bold,
      "  Some tests show significant numerical errors!\n" );
  }
}

int main()
{
  // Program header
  fmt::print(
    fg( fmt::terminal_color::bright_green ) | fmt::emphasis::bold,
    "\n"
    "═════════════════════════════════════════════════════════════════\n"
    "              MATRIX MULTIPLICATION BENCHMARK TOOL               \n"
    "                  Eigen + ThreadPool Performance                 \n"
    "═════════════════════════════════════════════════════════════════\n"
    "\n" );

  // Configuration
  const int n_runs = 5;

  // Define various matrix sizes to test
  std::vector<std::tuple<int, int, int>> test_sizes = { // Very small matrices (for correctness testing)
                                                        { 2, 2, 2 },
                                                        { 4, 4, 4 },
                                                        { 10, 10, 10 },

                                                        // Small matrices
                                                        { 50, 50, 50 },
                                                        { 100, 100, 100 },

                                                        // Medium matrices
                                                        { 200, 200, 200 },
                                                        { 300, 300, 300 },

                                                        // Larger matrices (performance testing)
                                                        { 500, 500, 500 },
                                                        { 800, 400, 1200 },  // Original test size
                                                        { 1000, 500, 1500 },

                                                        // Very large matrices (may be memory intensive)
                                                        { 2000, 500, 1000 },
                                                        { 1500, 1500, 1500 }
  };

  fmt::print( fg( fmt::terminal_color::bright_cyan ), "Test Configuration:\n" );
  fmt::print( "  Number of runs per test: {}\n", n_runs );
  fmt::print( "  Threads per pool: {}\n", 5 );
  fmt::print( "  Number of different matrix sizes: {}\n\n", test_sizes.size() );

  // Statistics for summary
  std::vector<std::string> best_methods;
  std::vector<std::string> best_efficiency_methods;
  std::vector<double>      best_speedups;
  std::vector<double>      best_gflops;

  // Run benchmarks for each matrix size
  for ( const auto & [N, P, M] : test_sizes )
  {
    try
    {
      run_benchmark_for_size( N, P, M, n_runs );

      // Store best results for summary
      best_methods.push_back( fmt::format( "{}x{}x{}", N, P, M ) );

      // We could collect more statistics here if needed
    }
    catch ( const std::exception & e )
    {
      fmt::print( fg( fmt::terminal_color::red ), "Error testing size {}x{}x{}: {}\n", N, P, M, e.what() );
    }
  }

  // Final summary
  fmt::print(
    fg( fmt::terminal_color::bright_green ) | fmt::emphasis::bold,
    "\n"
    "════════════════════════════════════════════════════════════════════════════════\n" );
  fmt::print( fg( fmt::terminal_color::bright_green ) | fmt::emphasis::bold, "Benchmark Summary:\n" );
  fmt::print( "  Tested {} different matrix sizes\n", test_sizes.size() );
  fmt::print( "  Each test run {} times for statistical significance\n", n_runs );
  fmt::print( "  Performance metrics:\n" );
  fmt::print( "    - Speedup: Compared to standard Eigen multiplication\n" );
  fmt::print( "    - GFLOP/s: Billions of floating point operations per second\n" );
  fmt::print( "    - Time/Cost: Milliseconds per GFLOP (lower is better)\n" );
  fmt::print(
    fg( fmt::terminal_color::bright_green ) | fmt::emphasis::bold,
    "════════════════════════════════════════════════════════════════════════════════\n" );

  return 0;
}
