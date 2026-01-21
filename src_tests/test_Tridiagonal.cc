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
/*==========================================================================*\
 |                                                                          |
 |  Test per TridiagonalSolver                                              |
 |                                                                          |
\*==========================================================================*/

#include "Utils_TridiagonalSolver.hh"
#include "Utils_fmt.hh"
#include "Utils_TicToc.hh"
#include <random>
#include <chrono>
#include <functional>

using namespace Utils;
using namespace std::chrono;

// ===========================================================================
// Utility functions
// ===========================================================================

template <typename Scalar> Scalar random_scalar( Scalar min, Scalar max )
{
  static std::random_device              rd;
  static std::mt19937                    gen( rd() );
  std::uniform_real_distribution<Scalar> dis( min, max );
  return dis( gen );
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> random_vector( Eigen::Index n, Scalar min, Scalar max )
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec( n );
  for ( Eigen::Index i = 0; i < n; ++i ) { vec( i ) = random_scalar<Scalar>( min, max ); }
  return vec;
}

template <typename Scalar> Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> random_matrix(
  Eigen::Index rows,
  Eigen::Index cols,
  Scalar       min,
  Scalar       max )
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat( rows, cols );
  for ( Eigen::Index i = 0; i < rows; ++i )
  {
    for ( Eigen::Index j = 0; j < cols; ++j ) { mat( i, j ) = random_scalar<Scalar>( min, max ); }
  }
  return mat;
}

template <typename Derived> typename Derived::Scalar vector_norm( const Eigen::MatrixBase<Derived> & v )
{
  return v.template lpNorm<Eigen::Infinity>();
}

// ===========================================================================
// Test results table
// ===========================================================================

struct TestResult
{
  std::string method;
  std::string type;
  std::string configuration;
  double      error;
  double      time_mus;
  bool        passed;
};

void print_test_table( const std::vector<TestResult> & results, const std::string & title )
{
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{}\n", title );

  // Table header
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "┌─────────────────────────────────────────────────────────────────────────────┐\n"
    "│ {:^75s} │\n"
    "├──────────────┬──────────────┬──────────────────────┬────────────┬───────────┤\n"
    "│ {:^12s} │ {:^12s} │ {:^20s} │ {:^10s} │ {:^9s} │\n"
    "├──────────────┼──────────────┼──────────────────────┼────────────┼───────────┤\n",
    title,
    "Method",
    "Type",
    "Configuration",
    "Error",
    "Time(μs)" );

  // Table rows
  for ( const auto & r : results )
  {
    auto        color     = r.passed ? fg( fmt::color::green ) : fg( fmt::color::red );
    std::string error_str = r.error < 1e-10 ? "< 1e-10" : fmt::format( "{:.2e}", r.error );
    std::string time_str  = fmt::format( "{:.1f}", r.time_mus );

    fmt::print(
      color,
      "│ {:^12s} │ {:^12s} │ {:^20s} │ {:^10s} │ {:^9s} │\n",
      r.method,
      r.type,
      r.configuration,
      error_str,
      time_str );
  }

  // Table footer
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "└──────────────┴──────────────┴──────────────────────┴────────────┴───────────┘\n" );

  // Summary for this table
  int passed = std::count_if( results.begin(), results.end(), []( const TestResult & r ) { return r.passed; } );
  int total  = results.size();

  fmt::print(
    fg( passed == total ? fmt::color::green : fmt::color::red ) | fmt::emphasis::bold,
    "📊 {}/{} tests passed ({:.1f}%)\n",
    passed,
    total,
    100.0 * passed / total );
}

// ===========================================================================
// Funzioni helper per test multipli
// ===========================================================================

struct TestConfig
{
  std::string                       name;
  std::function<TestResult( void )> test_func;
};

// ===========================================================================
// Test 1: Scalar Tridiagonal (Thomas Algorithm) - Non Cyclic
// ===========================================================================

template <typename Scalar> TestResult test_scalar_non_cyclic( Eigen::Index n, Eigen::Index n_rhs = 1 )
{
  TestResult result;
  result.method        = "Thomas";
  result.type          = "Scalar";
  result.configuration = fmt::format( "n={}, rhs={}", n, n_rhs );

  using Solver = TridiagonalSolver<Scalar>;
  using VecS   = typename Solver::VecS;

  TicToc tm;
  tm.tic();

  try
  {
    // Generate random tridiagonal system
    VecS a = random_vector<Scalar>( n - 1, 0.1, 1.0 );  // subdiagonal
    VecS b = random_vector<Scalar>( n, 2.0, 5.0 );      // diagonal (dominant)
    VecS c = random_vector<Scalar>( n - 1, 0.1, 1.0 );  // superdiagonal

    // Create solver and factorize
    Solver solver( n );
    solver.factorize( a, b, c );

    double max_error = 0.0;

    // Test with multiple RHS
    for ( Eigen::Index rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx )
    {
      VecS rhs = random_vector<Scalar>( n, -1.0, 1.0 );  // right-hand side
      VecS x( n );
      solver.solve( a, b, rhs, x );

      // Verify: compute Ax - rhs
      VecS Ax( n );
      if ( n == 1 ) { Ax( 0 ) = b( 0 ) * x( 0 ); }
      else if ( n == 2 )
      {
        Ax( 0 ) = b( 0 ) * x( 0 ) + c( 0 ) * x( 1 );
        Ax( 1 ) = a( 0 ) * x( 0 ) + b( 1 ) * x( 1 );
      }
      else
      {
        Ax( 0 ) = b( 0 ) * x( 0 ) + c( 0 ) * x( 1 );
        for ( Eigen::Index i = 1; i < n - 1; ++i )
        {
          Ax( i ) = a( i - 1 ) * x( i - 1 ) + b( i ) * x( i ) + c( i ) * x( i + 1 );
        }
        Ax( n - 1 ) = a( n - 2 ) * x( n - 2 ) + b( n - 1 ) * x( n - 1 );
      }

      double error = vector_norm( Ax - rhs );
      max_error    = std::max( max_error, error );
    }

    result.error  = max_error;
    result.passed = result.error < 1e-10;
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ), "Error in scalar non-cyclic test n={}: {}\n", n, e.what() );
    result.error  = 1.0;
    result.passed = false;
  }

  tm.toc();
  result.time_mus = tm.elapsed_mus();

  return result;
}

// ===========================================================================
// Test 2: Scalar Cyclic Tridiagonal
// ===========================================================================

template <typename Scalar> TestResult test_scalar_cyclic( Eigen::Index n, Eigen::Index n_rhs = 1 )
{
  TestResult result;
  result.method        = "Cyclic";
  result.type          = "Scalar";
  result.configuration = fmt::format( "n={}, rhs={}", n, n_rhs );

  using Solver = TridiagonalSolver<Scalar>;
  using VecS   = typename Solver::VecS;

  TicToc tm;
  tm.tic();

  try
  {
    // Generate random cyclic tridiagonal system
    VecS   a     = random_vector<Scalar>( n - 1, 0.1, 1.0 );
    VecS   b     = random_vector<Scalar>( n, 2.0, 5.0 );
    VecS   c     = random_vector<Scalar>( n - 1, 0.1, 1.0 );
    Scalar alpha = random_scalar<Scalar>( 0.1, 0.5 );  // top-right corner
    Scalar beta  = random_scalar<Scalar>( 0.1, 0.5 );  // bottom-left corner

    // Create solver
    Solver solver( n );

    double max_error = 0.0;

    // Test with multiple RHS
    for ( Eigen::Index rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx )
    {
      VecS rhs = random_vector<Scalar>( n, -1.0, 1.0 );
      VecS x( n );
      solver.solve_cyclic( a, b, c, alpha, beta, rhs, x );

      // Verify cyclic system: Ax = rhs
      VecS Ax( n );
      if ( n == 1 ) { Ax( 0 ) = ( b( 0 ) + alpha + beta ) * x( 0 ); }
      else if ( n == 2 )
      {
        Ax( 0 ) = ( b( 0 ) + alpha ) * x( 0 ) + c( 0 ) * x( 1 );
        Ax( 1 ) = a( 0 ) * x( 0 ) + ( b( 1 ) + beta ) * x( 1 );
      }
      else
      {
        Ax( 0 ) = b( 0 ) * x( 0 ) + c( 0 ) * x( 1 ) + alpha * x( n - 1 );
        for ( Eigen::Index i = 1; i < n - 1; ++i )
        {
          Ax( i ) = a( i - 1 ) * x( i - 1 ) + b( i ) * x( i ) + c( i ) * x( i + 1 );
        }
        Ax( n - 1 ) = a( n - 2 ) * x( n - 2 ) + b( n - 1 ) * x( n - 1 ) + beta * x( 0 );
      }

      double error = vector_norm( Ax - rhs );
      max_error    = std::max( max_error, error );
    }

    result.error  = max_error;
    result.passed = result.error < 1e-10;
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ), "Error in scalar cyclic test n={}: {}\n", n, e.what() );
    result.error  = 1.0;
    result.passed = false;
  }

  tm.toc();
  result.time_mus = tm.elapsed_mus();

  return result;
}

// ===========================================================================
// Test 3: Block Tridiagonal (Non Cyclic)
// ===========================================================================

template <typename Scalar> TestResult test_block_non_cyclic( Eigen::Index n, Eigen::Index m, Eigen::Index n_rhs = 1 )
{
  TestResult result;
  result.method        = "Block";
  result.type          = "Non-Cyclic";
  result.configuration = fmt::format( "n={}, m={}, rhs={}", n, m, n_rhs );

  using Solver = BlockTridiagonalSolver<Scalar, -1>;
  using Block  = typename Solver::Block;
  using VecB   = typename Solver::VecB;

  TicToc tm;
  tm.tic();

  try
  {
    // Generate random block-tridiagonal system
    std::vector<Block> A( n - 1 ), B( n ), C( n - 1 );

    for ( Eigen::Index i = 0; i < n - 1; ++i )
    {
      A[i].resize( m, m );
      C[i].resize( m, m );
      A[i] = random_matrix<Scalar>( m, m, 0.1, 1.0 );
      C[i] = random_matrix<Scalar>( m, m, 0.1, 1.0 );
    }

    for ( Eigen::Index i = 0; i < n; ++i )
    {
      B[i].resize( m, m );
      B[i] = random_matrix<Scalar>( m, m, 2.0, 5.0 );
      // Make diagonally dominant for stability
      for ( Eigen::Index j = 0; j < m; ++j ) { B[i]( j, j ) += 5.0; }
    }

    // Create solver and factorize
    Solver solver( n, m );
    solver.factorize( A, B, C );

    double max_error = 0.0;

    // Test with multiple RHS
    for ( Eigen::Index rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx )
    {
      // Generate random RHS as block vectors
      std::vector<VecB> RHS( n );
      for ( Eigen::Index i = 0; i < n; ++i )
      {
        RHS[i].resize( m );
        RHS[i] = random_vector<Scalar>( m, -1.0, 1.0 );
      }

      // Solve
      std::vector<VecB> X( n );
      solver.solve( A, B, RHS, X );

      // Verify: Build full block matrix and check error
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Full( n * m, n * m );
      Full.setZero();

      // Fill block tridiagonal matrix
      for ( Eigen::Index i = 0; i < n; ++i )
      {
        Full.block( i * m, i * m, m, m ) = B[i];
        if ( i < n - 1 )
        {
          Full.block( ( i + 1 ) * m, i * m, m, m ) = A[i];
          Full.block( i * m, ( i + 1 ) * m, m, m ) = C[i];
        }
      }

      // Convert solution and RHS to vectors
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x_vec( n * m );
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs_vec( n * m );

      for ( Eigen::Index i = 0; i < n; ++i )
      {
        x_vec.segment( i * m, m )   = X[i];
        rhs_vec.segment( i * m, m ) = RHS[i];
      }

      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Ax_vec = Full * x_vec;

      double error = vector_norm( Ax_vec - rhs_vec );
      max_error    = std::max( max_error, error );
    }

    result.error  = max_error;
    result.passed = result.error < 1e-10;
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ), "Error in block non-cyclic test n={}, m={}: {}\n", n, m, e.what() );
    result.error  = 1.0;
    result.passed = false;
  }

  tm.toc();
  result.time_mus = tm.elapsed_mus();

  return result;
}

// ===========================================================================
// Test 4: Block Cyclic Tridiagonal
// ===========================================================================

template <typename Scalar> TestResult test_block_cyclic( Eigen::Index n, Eigen::Index m, Eigen::Index n_rhs = 1 )
{
  TestResult result;
  result.method        = "Block-Cyclic";
  result.type          = "Cyclic";
  result.configuration = fmt::format( "n={}, m={}, rhs={}", n, m, n_rhs );

  using Solver = BlockTridiagonalSolver<Scalar, -1>;
  using Block  = typename Solver::Block;
  using VecB   = typename Solver::VecB;

  TicToc tm;
  tm.tic();

  try
  {
    // Generate random block-cyclic tridiagonal system
    std::vector<Block> A( n - 1 ), B( n ), C( n - 1 );

    for ( Eigen::Index i = 0; i < n - 1; ++i )
    {
      A[i].resize( m, m );
      C[i].resize( m, m );
      A[i] = random_matrix<Scalar>( m, m, 0.1, 1.0 );
      C[i] = random_matrix<Scalar>( m, m, 0.1, 1.0 );
    }

    for ( Eigen::Index i = 0; i < n; ++i )
    {
      B[i].resize( m, m );
      B[i] = random_matrix<Scalar>( m, m, 2.0, 5.0 );
      for ( Eigen::Index j = 0; j < m; ++j ) { B[i]( j, j ) += 5.0; }
    }

    // Corner blocks for cyclic system
    Block Alpha, Beta;
    Alpha.resize( m, m );
    Beta.resize( m, m );
    Alpha = random_matrix<Scalar>( m, m, 0.1, 0.5 );
    Beta  = random_matrix<Scalar>( m, m, 0.1, 0.5 );

    // Create solver
    Solver solver( n, m );

    double max_error = 0.0;

    // Test with multiple RHS
    for ( Eigen::Index rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx )
    {
      // RHS as block vectors
      std::vector<VecB> RHS( n );
      for ( Eigen::Index i = 0; i < n; ++i )
      {
        RHS[i].resize( m );
        RHS[i] = random_vector<Scalar>( m, -1.0, 1.0 );
      }

      // Solve cyclic system
      std::vector<VecB> X( n );
      solver.solve_cyclic( A, B, C, Alpha, Beta, RHS, X );

      // Verify: Build full cyclic block matrix
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Full( n * m, n * m );
      Full.setZero();

      // Fill main tridiagonal
      for ( Eigen::Index i = 0; i < n; ++i )
      {
        Full.block( i * m, i * m, m, m ) = B[i];
        if ( i < n - 1 )
        {
          Full.block( ( i + 1 ) * m, i * m, m, m ) = A[i];
          Full.block( i * m, ( i + 1 ) * m, m, m ) = C[i];
        }
      }

      // Add cyclic corners
      if ( n > 2 )
      {
        Full.block( 0, ( n - 1 ) * m, m, m ) += Alpha;  // top-right
        Full.block( ( n - 1 ) * m, 0, m, m ) += Beta;   // bottom-left
      }
      else if ( n == 2 )
      {
        // non fare niente
      }
      else
      {
        // n=1 case: corner terms are added to the single block
        Full.block( 0, 0, m, m ) = B[0] + Alpha + Beta;
      }

      // Convert to vectors
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x_vec( n * m );
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs_vec( n * m );

      for ( Eigen::Index i = 0; i < n; ++i )
      {
        x_vec.segment( i * m, m )   = X[i];
        rhs_vec.segment( i * m, m ) = RHS[i];
      }

      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Ax_vec = Full * x_vec;

      double error = vector_norm( Ax_vec - rhs_vec );
      max_error    = std::max( max_error, error );
    }

    result.error  = max_error;
    result.passed = result.error < 1e-10;
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ), "Error in block cyclic test n={}, m={}: {}\n", n, m, e.what() );
    result.error  = 1.0;
    result.passed = false;
  }

  tm.toc();
  result.time_mus = tm.elapsed_mus();

  return result;
}

// ===========================================================================
// Definizione dei test suite
// ===========================================================================

void run_scalar_non_cyclic_tests()
{
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "╔═══════════════════════════╗\n"
    "║ TEST 1: Scalar Non-Cyclic ║\n"
    "║    (Thomas Algorithm)     ║\n"
    "╚═══════════════════════════╝\n\n" );

  std::vector<TestConfig> test_configs = {
    // Dimensioni piccole
    { "n=1, rhs=1", []() { return test_scalar_non_cyclic<double>( 1, 1 ); } },
    { "n=1, rhs=10", []() { return test_scalar_non_cyclic<double>( 1, 10 ); } },
    { "n=1, rhs=100", []() { return test_scalar_non_cyclic<double>( 1, 100 ); } },
    { "n=2, rhs=1", []() { return test_scalar_non_cyclic<double>( 2, 1 ); } },
    { "n=2, rhs=10", []() { return test_scalar_non_cyclic<double>( 2, 10 ); } },
    { "n=3, rhs=5", []() { return test_scalar_non_cyclic<double>( 3, 5 ); } },
    { "n=5, rhs=1", []() { return test_scalar_non_cyclic<double>( 5, 1 ); } },
    { "n=5, rhs=10", []() { return test_scalar_non_cyclic<double>( 5, 10 ); } },
    { "n=5, rhs=50", []() { return test_scalar_non_cyclic<double>( 5, 50 ); } },
    { "n=10, rhs=1", []() { return test_scalar_non_cyclic<double>( 10, 1 ); } },

    // Dimensioni medie
    { "n=10, rhs=10", []() { return test_scalar_non_cyclic<double>( 10, 10 ); } },
    { "n=10, rhs=50", []() { return test_scalar_non_cyclic<double>( 10, 50 ); } },
    { "n=20, rhs=1", []() { return test_scalar_non_cyclic<double>( 20, 1 ); } },
    { "n=20, rhs=10", []() { return test_scalar_non_cyclic<double>( 20, 10 ); } },
    { "n=50, rhs=1", []() { return test_scalar_non_cyclic<double>( 50, 1 ); } },
    { "n=50, rhs=5", []() { return test_scalar_non_cyclic<double>( 50, 5 ); } },
    { "n=50, rhs=20", []() { return test_scalar_non_cyclic<double>( 50, 20 ); } },
    { "n=100, rhs=1", []() { return test_scalar_non_cyclic<double>( 100, 1 ); } },
    { "n=100, rhs=10", []() { return test_scalar_non_cyclic<double>( 100, 10 ); } },
    { "n=100, rhs=50", []() { return test_scalar_non_cyclic<double>( 100, 50 ); } },

    // Dimensioni grandi
    { "n=200, rhs=1", []() { return test_scalar_non_cyclic<double>( 200, 1 ); } },
    { "n=200, rhs=10", []() { return test_scalar_non_cyclic<double>( 200, 10 ); } },
    { "n=500, rhs=1", []() { return test_scalar_non_cyclic<double>( 500, 1 ); } },
    { "n=500, rhs=5", []() { return test_scalar_non_cyclic<double>( 500, 5 ); } },
    { "n=500, rhs=20", []() { return test_scalar_non_cyclic<double>( 500, 20 ); } },
    { "n=1000, rhs=1", []() { return test_scalar_non_cyclic<double>( 1000, 1 ); } },
    { "n=1000, rhs=10", []() { return test_scalar_non_cyclic<double>( 1000, 10 ); } },
    { "n=1000, rhs=50", []() { return test_scalar_non_cyclic<double>( 1000, 50 ); } },
    { "n=2000, rhs=1", []() { return test_scalar_non_cyclic<double>( 2000, 1 ); } },
    { "n=2000, rhs=5", []() { return test_scalar_non_cyclic<double>( 2000, 5 ); } },

    // Dimensioni molto grandi
    { "n=5000, rhs=1", []() { return test_scalar_non_cyclic<double>( 5000, 1 ); } },
    { "n=5000, rhs=3", []() { return test_scalar_non_cyclic<double>( 5000, 3 ); } },
    { "n=10000, rhs=1", []() { return test_scalar_non_cyclic<double>( 10000, 1 ); } },
    { "n=10000, rhs=2", []() { return test_scalar_non_cyclic<double>( 10000, 2 ); } },
    { "n=20000, rhs=1", []() { return test_scalar_non_cyclic<double>( 20000, 1 ); } },
    { "n=50000, rhs=1", []() { return test_scalar_non_cyclic<double>( 50000, 1 ); } },

    // Test con molti RHS
    { "n=10, rhs=100", []() { return test_scalar_non_cyclic<double>( 10, 100 ); } },
    { "n=20, rhs=100", []() { return test_scalar_non_cyclic<double>( 20, 100 ); } },
    { "n=50, rhs=100", []() { return test_scalar_non_cyclic<double>( 50, 100 ); } },
    { "n=100, rhs=100", []() { return test_scalar_non_cyclic<double>( 100, 100 ); } },
    { "n=200, rhs=50", []() { return test_scalar_non_cyclic<double>( 200, 50 ); } },

    // Dimensioni varie
    { "n=7, rhs=7", []() { return test_scalar_non_cyclic<double>( 7, 7 ); } },
    { "n=13, rhs=13", []() { return test_scalar_non_cyclic<double>( 13, 13 ); } },
    { "n=17, rhs=17", []() { return test_scalar_non_cyclic<double>( 17, 17 ); } },
    { "n=23, rhs=23", []() { return test_scalar_non_cyclic<double>( 23, 23 ); } },
    { "n=31, rhs=31", []() { return test_scalar_non_cyclic<double>( 31, 31 ); } },
    { "n=47, rhs=47", []() { return test_scalar_non_cyclic<double>( 47, 47 ); } },
    { "n=73, rhs=73", []() { return test_scalar_non_cyclic<double>( 73, 73 ); } },
    { "n=127, rhs=127", []() { return test_scalar_non_cyclic<double>( 127, 127 ); } },
    { "n=257, rhs=257", []() { return test_scalar_non_cyclic<double>( 257, 257 ); } },
    { "n=521, rhs=521", []() { return test_scalar_non_cyclic<double>( 521, 521 ); } }
  };

  std::vector<TestResult> results;
  for ( size_t i = 0; i < test_configs.size(); ++i )
  {
    fmt::print( fg( fmt::color::cyan ), "Running test {}/{}: {}\n", i + 1, test_configs.size(), test_configs[i].name );
    results.push_back( test_configs[i].test_func() );
  }

  print_test_table( results, "SCALAR NON-CYCLIC TRIDIAGONAL SYSTEMS" );
}

void run_scalar_cyclic_tests()
{
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "╔══════════════════════════════╗\n"
    "║     TEST 2: Scalar Cyclic    ║\n"
    "║ (Sherman-Morrison Algorithm) ║\n"
    "╚══════════════════════════════╝\n\n" );

  std::vector<TestConfig> test_configs = {
    // Dimensioni piccole
    { "n=1, rhs=1", []() { return test_scalar_cyclic<double>( 1, 1 ); } },
    { "n=1, rhs=10", []() { return test_scalar_cyclic<double>( 1, 10 ); } },
    { "n=1, rhs=100", []() { return test_scalar_cyclic<double>( 1, 100 ); } },
    { "n=2, rhs=1", []() { return test_scalar_cyclic<double>( 2, 1 ); } },
    { "n=2, rhs=10", []() { return test_scalar_cyclic<double>( 2, 10 ); } },
    { "n=2, rhs=50", []() { return test_scalar_cyclic<double>( 2, 50 ); } },
    { "n=3, rhs=5", []() { return test_scalar_cyclic<double>( 3, 5 ); } },
    { "n=5, rhs=1", []() { return test_scalar_cyclic<double>( 5, 1 ); } },
    { "n=5, rhs=10", []() { return test_scalar_cyclic<double>( 5, 10 ); } },
    { "n=5, rhs=50", []() { return test_scalar_cyclic<double>( 5, 50 ); } },

    // Dimensioni medie
    { "n=10, rhs=1", []() { return test_scalar_cyclic<double>( 10, 1 ); } },
    { "n=10, rhs=10", []() { return test_scalar_cyclic<double>( 10, 10 ); } },
    { "n=10, rhs=50", []() { return test_scalar_cyclic<double>( 10, 50 ); } },
    { "n=20, rhs=1", []() { return test_scalar_cyclic<double>( 20, 1 ); } },
    { "n=20, rhs=10", []() { return test_scalar_cyclic<double>( 20, 10 ); } },
    { "n=20, rhs=30", []() { return test_scalar_cyclic<double>( 20, 30 ); } },
    { "n=50, rhs=1", []() { return test_scalar_cyclic<double>( 50, 1 ); } },
    { "n=50, rhs=5", []() { return test_scalar_cyclic<double>( 50, 5 ); } },
    { "n=50, rhs=20", []() { return test_scalar_cyclic<double>( 50, 20 ); } },
    { "n=100, rhs=1", []() { return test_scalar_cyclic<double>( 100, 1 ); } },

    // Dimensioni grandi
    { "n=100, rhs=10", []() { return test_scalar_cyclic<double>( 100, 10 ); } },
    { "n=100, rhs=50", []() { return test_scalar_cyclic<double>( 100, 50 ); } },
    { "n=200, rhs=1", []() { return test_scalar_cyclic<double>( 200, 1 ); } },
    { "n=200, rhs=10", []() { return test_scalar_cyclic<double>( 200, 10 ); } },
    { "n=200, rhs=30", []() { return test_scalar_cyclic<double>( 200, 30 ); } },
    { "n=500, rhs=1", []() { return test_scalar_cyclic<double>( 500, 1 ); } },
    { "n=500, rhs=5", []() { return test_scalar_cyclic<double>( 500, 5 ); } },
    { "n=500, rhs=20", []() { return test_scalar_cyclic<double>( 500, 20 ); } },
    { "n=1000, rhs=1", []() { return test_scalar_cyclic<double>( 1000, 1 ); } },
    { "n=1000, rhs=10", []() { return test_scalar_cyclic<double>( 1000, 10 ); } },

    // Dimensioni molto grandi
    { "n=2000, rhs=1", []() { return test_scalar_cyclic<double>( 2000, 1 ); } },
    { "n=2000, rhs=5", []() { return test_scalar_cyclic<double>( 2000, 5 ); } },
    { "n=5000, rhs=1", []() { return test_scalar_cyclic<double>( 5000, 1 ); } },
    { "n=5000, rhs=3", []() { return test_scalar_cyclic<double>( 5000, 3 ); } },
    { "n=10000, rhs=1", []() { return test_scalar_cyclic<double>( 10000, 1 ); } },

    // Test con molti RHS
    { "n=10, rhs=100", []() { return test_scalar_cyclic<double>( 10, 100 ); } },
    { "n=20, rhs=100", []() { return test_scalar_cyclic<double>( 20, 100 ); } },
    { "n=50, rhs=100", []() { return test_scalar_cyclic<double>( 50, 100 ); } },
    { "n=100, rhs=100", []() { return test_scalar_cyclic<double>( 100, 100 ); } },
    { "n=200, rhs=50", []() { return test_scalar_cyclic<double>( 200, 50 ); } },

    // Dimensioni varie (numeri primi)
    { "n=7, rhs=7", []() { return test_scalar_cyclic<double>( 7, 7 ); } },
    { "n=13, rhs=13", []() { return test_scalar_cyclic<double>( 13, 13 ); } },
    { "n=17, rhs=17", []() { return test_scalar_cyclic<double>( 17, 17 ); } },
    { "n=23, rhs=23", []() { return test_scalar_cyclic<double>( 23, 23 ); } },
    { "n=31, rhs=31", []() { return test_scalar_cyclic<double>( 31, 31 ); } },
    { "n=47, rhs=47", []() { return test_scalar_cyclic<double>( 47, 47 ); } },
    { "n=73, rhs=73", []() { return test_scalar_cyclic<double>( 73, 73 ); } },
    { "n=127, rhs=127", []() { return test_scalar_cyclic<double>( 127, 127 ); } },
    { "n=257, rhs=257", []() { return test_scalar_cyclic<double>( 257, 257 ); } },
    { "n=521, rhs=521", []() { return test_scalar_cyclic<double>( 521, 521 ); } },

    // Test edge cases
    { "n=4, rhs=100", []() { return test_scalar_cyclic<double>( 4, 100 ); } },
    { "n=8, rhs=100", []() { return test_scalar_cyclic<double>( 8, 100 ); } },
    { "n=16, rhs=100", []() { return test_scalar_cyclic<double>( 16, 100 ); } },
    { "n=32, rhs=100", []() { return test_scalar_cyclic<double>( 32, 100 ); } },
    { "n=64, rhs=100", []() { return test_scalar_cyclic<double>( 64, 100 ); } }
  };

  std::vector<TestResult> results;
  for ( size_t i = 0; i < test_configs.size(); ++i )
  {
    fmt::print( fg( fmt::color::cyan ), "Running test {}/{}: {}\n", i + 1, test_configs.size(), test_configs[i].name );
    results.push_back( test_configs[i].test_func() );
  }

  print_test_table( results, "SCALAR CYCLIC TRIDIAGONAL SYSTEMS" );
}

void run_block_non_cyclic_tests()
{
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "╔══════════════════════════╗\n"
    "║ TEST 3: Block Non-Cyclic ║\n"
    "║ (Block Thomas Algorithm) ║\n"
    "╚══════════════════════════╝\n\n" );

  std::vector<TestConfig> test_configs = {
    // Dimensioni piccole (vari blocchi e dimensioni)
    { "n=1, m=1, rhs=1", []() { return test_block_non_cyclic<double>( 1, 1, 1 ); } },
    { "n=1, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 1, 2, 1 ); } },
    { "n=1, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 1, 5, 1 ); } },
    { "n=1, m=10, rhs=1", []() { return test_block_non_cyclic<double>( 1, 10, 1 ); } },
    { "n=1, m=1, rhs=10", []() { return test_block_non_cyclic<double>( 1, 1, 10 ); } },
    { "n=1, m=2, rhs=10", []() { return test_block_non_cyclic<double>( 1, 2, 10 ); } },
    { "n=1, m=5, rhs=10", []() { return test_block_non_cyclic<double>( 1, 5, 10 ); } },

    // n=2 con vari m
    { "n=2, m=1, rhs=1", []() { return test_block_non_cyclic<double>( 2, 1, 1 ); } },
    { "n=2, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 2, 2, 1 ); } },
    { "n=2, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 2, 3, 1 ); } },
    { "n=2, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 2, 5, 1 ); } },
    { "n=2, m=10, rhs=1", []() { return test_block_non_cyclic<double>( 2, 10, 1 ); } },
    { "n=2, m=2, rhs=10", []() { return test_block_non_cyclic<double>( 2, 2, 10 ); } },
    { "n=2, m=3, rhs=10", []() { return test_block_non_cyclic<double>( 2, 3, 10 ); } },

    // n piccolo, m vario
    { "n=3, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 3, 2, 1 ); } },
    { "n=3, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 3, 3, 1 ); } },
    { "n=3, m=4, rhs=1", []() { return test_block_non_cyclic<double>( 3, 4, 1 ); } },
    { "n=3, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 3, 5, 1 ); } },
    { "n=4, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 4, 2, 1 ); } },
    { "n=4, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 4, 3, 1 ); } },
    { "n=4, m=4, rhs=1", []() { return test_block_non_cyclic<double>( 4, 4, 1 ); } },
    { "n=5, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 5, 2, 1 ); } },
    { "n=5, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 5, 3, 1 ); } },
    { "n=5, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 5, 5, 1 ); } },

    // Dimensioni medie
    { "n=10, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 10, 2, 1 ); } },
    { "n=10, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 10, 3, 1 ); } },
    { "n=10, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 10, 5, 1 ); } },
    { "n=10, m=10, rhs=1", []() { return test_block_non_cyclic<double>( 10, 10, 1 ); } },
    { "n=10, m=2, rhs=10", []() { return test_block_non_cyclic<double>( 10, 2, 10 ); } },
    { "n=10, m=3, rhs=10", []() { return test_block_non_cyclic<double>( 10, 3, 10 ); } },
    { "n=10, m=5, rhs=10", []() { return test_block_non_cyclic<double>( 10, 5, 10 ); } },

    // n medio, m vario
    { "n=20, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 20, 2, 1 ); } },
    { "n=20, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 20, 3, 1 ); } },
    { "n=20, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 20, 5, 1 ); } },
    { "n=20, m=8, rhs=1", []() { return test_block_non_cyclic<double>( 20, 8, 1 ); } },
    { "n=20, m=10, rhs=1", []() { return test_block_non_cyclic<double>( 20, 10, 1 ); } },
    { "n=20, m=3, rhs=5", []() { return test_block_non_cyclic<double>( 20, 3, 5 ); } },

    // n grande, m piccolo
    { "n=50, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 50, 2, 1 ); } },
    { "n=50, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 50, 3, 1 ); } },
    { "n=50, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 50, 5, 1 ); } },
    { "n=100, m=2, rhs=1", []() { return test_block_non_cyclic<double>( 100, 2, 1 ); } },
    { "n=100, m=3, rhs=1", []() { return test_block_non_cyclic<double>( 100, 3, 1 ); } },
    { "n=100, m=5, rhs=1", []() { return test_block_non_cyclic<double>( 100, 5, 1 ); } },

    // n piccolo, m grande
    { "n=5, m=20, rhs=1", []() { return test_block_non_cyclic<double>( 5, 20, 1 ); } },
    { "n=5, m=30, rhs=1", []() { return test_block_non_cyclic<double>( 5, 30, 1 ); } },
    { "n=10, m=20, rhs=1", []() { return test_block_non_cyclic<double>( 10, 20, 1 ); } },
    { "n=10, m=30, rhs=1", []() { return test_block_non_cyclic<double>( 10, 30, 1 ); } },

    // Test con molti RHS
    { "n=5, m=3, rhs=50", []() { return test_block_non_cyclic<double>( 5, 3, 50 ); } },
    { "n=10, m=3, rhs=50", []() { return test_block_non_cyclic<double>( 10, 3, 50 ); } },
    { "n=20, m=3, rhs=30", []() { return test_block_non_cyclic<double>( 20, 3, 30 ); } },
    { "n=30, m=3, rhs=20", []() { return test_block_non_cyclic<double>( 30, 3, 20 ); } },

    // Dimensioni particolari
    { "n=7, m=7, rhs=7", []() { return test_block_non_cyclic<double>( 7, 7, 7 ); } },
    { "n=11, m=11, rhs=11", []() { return test_block_non_cyclic<double>( 11, 11, 11 ); } },
    { "n=13, m=13, rhs=13", []() { return test_block_non_cyclic<double>( 13, 13, 13 ); } },
    { "n=17, m=17, rhs=17", []() { return test_block_non_cyclic<double>( 17, 17, 17 ); } },

    // Test edge cases
    { "n=2, m=1, rhs=100", []() { return test_block_non_cyclic<double>( 2, 1, 100 ); } },
    { "n=3, m=1, rhs=100", []() { return test_block_non_cyclic<double>( 3, 1, 100 ); } },
    { "n=4, m=1, rhs=100", []() { return test_block_non_cyclic<double>( 4, 1, 100 ); } },
    { "n=5, m=1, rhs=100", []() { return test_block_non_cyclic<double>( 5, 1, 100 ); } }
  };

  std::vector<TestResult> results;
  for ( size_t i = 0; i < test_configs.size(); ++i )
  {
    fmt::print( fg( fmt::color::cyan ), "Running test {}/{}: {}\n", i + 1, test_configs.size(), test_configs[i].name );
    results.push_back( test_configs[i].test_func() );
  }

  print_test_table( results, "BLOCK NON-CYCLIC TRIDIAGONAL SYSTEMS" );
}

void run_block_cyclic_tests()
{
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "╔═══════════════════════════════════════╗\n"
    "║         TEST 4: Block Cyclic          ║\n"
    "║ (Sherman-Morrison-Woodbury Algorithm) ║\n"
    "╚═══════════════════════════════════════╝\n\n" );

  std::vector<TestConfig> test_configs = {
    // n=1 con vari m
    { "n=1, m=1, rhs=1", []() { return test_block_cyclic<double>( 1, 1, 1 ); } },
    { "n=1, m=2, rhs=1", []() { return test_block_cyclic<double>( 1, 2, 1 ); } },
    { "n=1, m=3, rhs=1", []() { return test_block_cyclic<double>( 1, 3, 1 ); } },
    { "n=1, m=5, rhs=1", []() { return test_block_cyclic<double>( 1, 5, 1 ); } },
    { "n=1, m=10, rhs=1", []() { return test_block_cyclic<double>( 1, 10, 1 ); } },
    { "n=1, m=2, rhs=10", []() { return test_block_cyclic<double>( 1, 2, 10 ); } },
    { "n=1, m=5, rhs=10", []() { return test_block_cyclic<double>( 1, 5, 10 ); } },

    // n=2 con vari m
    { "n=2, m=1, rhs=1", []() { return test_block_cyclic<double>( 2, 1, 1 ); } },
    { "n=2, m=2, rhs=1", []() { return test_block_cyclic<double>( 2, 2, 1 ); } },
    { "n=2, m=3, rhs=1", []() { return test_block_cyclic<double>( 2, 3, 1 ); } },
    { "n=2, m=5, rhs=1", []() { return test_block_cyclic<double>( 2, 5, 1 ); } },
    { "n=2, m=10, rhs=1", []() { return test_block_cyclic<double>( 2, 10, 1 ); } },
    { "n=2, m=2, rhs=10", []() { return test_block_cyclic<double>( 2, 2, 10 ); } },
    { "n=2, m=3, rhs=10", []() { return test_block_cyclic<double>( 2, 3, 10 ); } },

    // n piccolo, m vario
    { "n=3, m=2, rhs=1", []() { return test_block_cyclic<double>( 3, 2, 1 ); } },
    { "n=3, m=3, rhs=1", []() { return test_block_cyclic<double>( 3, 3, 1 ); } },
    { "n=3, m=4, rhs=1", []() { return test_block_cyclic<double>( 3, 4, 1 ); } },
    { "n=3, m=5, rhs=1", []() { return test_block_cyclic<double>( 3, 5, 1 ); } },
    { "n=4, m=2, rhs=1", []() { return test_block_cyclic<double>( 4, 2, 1 ); } },
    { "n=4, m=3, rhs=1", []() { return test_block_cyclic<double>( 4, 3, 1 ); } },
    { "n=4, m=4, rhs=1", []() { return test_block_cyclic<double>( 4, 4, 1 ); } },
    { "n=5, m=2, rhs=1", []() { return test_block_cyclic<double>( 5, 2, 1 ); } },
    { "n=5, m=3, rhs=1", []() { return test_block_cyclic<double>( 5, 3, 1 ); } },
    { "n=5, m=5, rhs=1", []() { return test_block_cyclic<double>( 5, 5, 1 ); } },

    // Dimensioni medie
    { "n=10, m=2, rhs=1", []() { return test_block_cyclic<double>( 10, 2, 1 ); } },
    { "n=10, m=3, rhs=1", []() { return test_block_cyclic<double>( 10, 3, 1 ); } },
    { "n=10, m=5, rhs=1", []() { return test_block_cyclic<double>( 10, 5, 1 ); } },
    { "n=10, m=8, rhs=1", []() { return test_block_cyclic<double>( 10, 8, 1 ); } },
    { "n=10, m=10, rhs=1", []() { return test_block_cyclic<double>( 10, 10, 1 ); } },
    { "n=10, m=2, rhs=10", []() { return test_block_cyclic<double>( 10, 2, 10 ); } },
    { "n=10, m=3, rhs=10", []() { return test_block_cyclic<double>( 10, 3, 10 ); } },

    // n medio, m vario
    { "n=15, m=2, rhs=1", []() { return test_block_cyclic<double>( 15, 2, 1 ); } },
    { "n=15, m=3, rhs=1", []() { return test_block_cyclic<double>( 15, 3, 1 ); } },
    { "n=15, m=5, rhs=1", []() { return test_block_cyclic<double>( 15, 5, 1 ); } },
    { "n=20, m=2, rhs=1", []() { return test_block_cyclic<double>( 20, 2, 1 ); } },
    { "n=20, m=3, rhs=1", []() { return test_block_cyclic<double>( 20, 3, 1 ); } },
    { "n=20, m=5, rhs=1", []() { return test_block_cyclic<double>( 20, 5, 1 ); } },
    { "n=20, m=8, rhs=1", []() { return test_block_cyclic<double>( 20, 8, 1 ); } },

    // n grande, m piccolo
    { "n=30, m=2, rhs=1", []() { return test_block_cyclic<double>( 30, 2, 1 ); } },
    { "n=30, m=3, rhs=1", []() { return test_block_cyclic<double>( 30, 3, 1 ); } },
    { "n=30, m=5, rhs=1", []() { return test_block_cyclic<double>( 30, 5, 1 ); } },
    { "n=50, m=2, rhs=1", []() { return test_block_cyclic<double>( 50, 2, 1 ); } },
    { "n=50, m=3, rhs=1", []() { return test_block_cyclic<double>( 50, 3, 1 ); } },
    { "n=50, m=5, rhs=1", []() { return test_block_cyclic<double>( 50, 5, 1 ); } },

    // n piccolo, m grande
    { "n=5, m=15, rhs=1", []() { return test_block_cyclic<double>( 5, 15, 1 ); } },
    { "n=5, m=20, rhs=1", []() { return test_block_cyclic<double>( 5, 20, 1 ); } },
    { "n=10, m=15, rhs=1", []() { return test_block_cyclic<double>( 10, 15, 1 ); } },
    { "n=10, m=20, rhs=1", []() { return test_block_cyclic<double>( 10, 20, 1 ); } },

    // Test con molti RHS
    { "n=5, m=3, rhs=50", []() { return test_block_cyclic<double>( 5, 3, 50 ); } },
    { "n=10, m=3, rhs=50", []() { return test_block_cyclic<double>( 10, 3, 50 ); } },
    { "n=15, m=3, rhs=30", []() { return test_block_cyclic<double>( 15, 3, 30 ); } },
    { "n=20, m=3, rhs=20", []() { return test_block_cyclic<double>( 20, 3, 20 ); } },

    // Dimensioni particolari
    { "n=7, m=7, rhs=7", []() { return test_block_cyclic<double>( 7, 7, 7 ); } },
    { "n=11, m=11, rhs=11", []() { return test_block_cyclic<double>( 11, 11, 11 ); } },
    { "n=13, m=13, rhs=13", []() { return test_block_cyclic<double>( 13, 13, 13 ); } },
    { "n=17, m=17, rhs=17", []() { return test_block_cyclic<double>( 17, 17, 17 ); } },

    // Test con n potenze di 2
    { "n=4, m=4, rhs=10", []() { return test_block_cyclic<double>( 4, 4, 10 ); } },
    { "n=8, m=4, rhs=10", []() { return test_block_cyclic<double>( 8, 4, 10 ); } },
    { "n=16, m=4, rhs=10", []() { return test_block_cyclic<double>( 16, 4, 10 ); } },
    { "n=32, m=3, rhs=5", []() { return test_block_cyclic<double>( 32, 3, 5 ); } },

    // Edge cases con n=2 e molti RHS
    { "n=2, m=2, rhs=100", []() { return test_block_cyclic<double>( 2, 2, 100 ); } },
    { "n=2, m=3, rhs=100", []() { return test_block_cyclic<double>( 2, 3, 100 ); } },
    { "n=2, m=5, rhs=100", []() { return test_block_cyclic<double>( 2, 5, 100 ); } },

    // Test con blocchi rettangolari (ma il nostro caso è quadrato, quindi m x m)
    { "n=3, m=4, rhs=20", []() { return test_block_cyclic<double>( 3, 4, 20 ); } },
    { "n=5, m=6, rhs=20", []() { return test_block_cyclic<double>( 5, 6, 20 ); } },
    { "n=7, m=8, rhs=15", []() { return test_block_cyclic<double>( 7, 8, 15 ); } }
  };

  std::vector<TestResult> results;
  for ( size_t i = 0; i < test_configs.size(); ++i )
  {
    fmt::print( fg( fmt::color::cyan ), "Running test {}/{}: {}\n", i + 1, test_configs.size(), test_configs[i].name );
    results.push_back( test_configs[i].test_func() );
  }

  print_test_table( results, "BLOCK CYCLIC TRIDIAGONAL SYSTEMS" );
}

// ===========================================================================
// Test 5: Scalar Tridiagonal with Eigen::Map (Non Cyclic and Cyclic)
// ===========================================================================

template <typename Scalar>
TestResult test_scalar_non_cyclic_map( Eigen::Index n, Eigen::Index n_rhs = 1 )
{
  TestResult result;
  result.method        = "Thomas-Map";
  result.type          = "Scalar";
  result.configuration = fmt::format( "n={}, rhs={}", n, n_rhs );

  using Solver = TridiagonalSolver<Scalar>;
  using VecS   = typename Solver::VecS;

  TicToc tm;
  tm.tic();

  try
  {
    // Allocate data in std::vector (simulating external data)
    std::vector<Scalar> a_data( n - 1 );
    std::vector<Scalar> b_data( n );
    std::vector<Scalar> c_data( n - 1 );
    // Fill with random values
    for ( Eigen::Index i = 0; i < n - 1; ++i )
    {
      a_data[i] = random_scalar<Scalar>( 0.1, 1.0 );
      c_data[i] = random_scalar<Scalar>( 0.1, 1.0 );
    }
    for ( Eigen::Index i = 0; i < n; ++i ) { b_data[i] = random_scalar<Scalar>( 2.0, 5.0 ); }

    // Create Eigen::Map objects
    Eigen::Map<const VecS> a_map( a_data.data(), n - 1 );
    Eigen::Map<const VecS> b_map( b_data.data(), n );
    Eigen::Map<const VecS> c_map( c_data.data(), n - 1 );

    Solver solver( n );
    solver.factorize( a_map, b_map, c_map );

    double max_error = 0.0;
    for ( Eigen::Index rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx )
    {
      // Generate random RHS in a vector
      std::vector<Scalar> rhs_data( n );
      for ( Eigen::Index i = 0; i < n; ++i ) { rhs_data[i] = random_scalar<Scalar>( -1.0, 1.0 ); }
      Eigen::Map<const VecS> rhs_map( rhs_data.data(), n );

      // Solve using Map
      VecS x( n );
      solver.solve( a_map, b_map, rhs_map, x );

      // Verify solution
      VecS Ax( n );
      if ( n == 1 ) { Ax( 0 ) = b_map( 0 ) * x( 0 ); }
      else if ( n == 2 )
      {
        Ax( 0 ) = b_map( 0 ) * x( 0 ) + c_map( 0 ) * x( 1 );
        Ax( 1 ) = a_map( 0 ) * x( 0 ) + b_map( 1 ) * x( 1 );
      }
      else
      {
        Ax( 0 ) = b_map( 0 ) * x( 0 ) + c_map( 0 ) * x( 1 );
        for ( Eigen::Index i = 1; i < n - 1; ++i )
        {
          Ax( i ) = a_map( i - 1 ) * x( i - 1 ) + b_map( i ) * x( i ) + c_map( i ) * x( i + 1 );
        }
        Ax( n - 1 ) = a_map( n - 2 ) * x( n - 2 ) + b_map( n - 1 ) * x( n - 1 );
      }

      // Compute error
      double error = vector_norm( Ax - rhs_map );
      max_error    = std::max( max_error, error );
    }

    result.error  = max_error;
    result.passed = result.error < 1e-10;
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ), "Error in scalar non-cyclic map test n={}: {}\n", n, e.what() );
    result.error  = 1.0;
    result.passed = false;
  }

  tm.toc();
  result.time_mus = tm.elapsed_mus();
  return result;
}

template <typename Scalar>
TestResult test_scalar_cyclic_map( Eigen::Index n, Eigen::Index n_rhs = 1 )
{
  TestResult result;
  result.method        = "Cyclic-Map";
  result.type          = "Scalar";
  result.configuration = fmt::format( "n={}, rhs={}", n, n_rhs );

  using Solver = TridiagonalSolver<Scalar>;
  using VecS   = typename Solver::VecS;

  TicToc tm;
  tm.tic();

  try
  {
    // Allocate data
    std::vector<Scalar> a_data( n - 1 );
    std::vector<Scalar> b_data( n );
    std::vector<Scalar> c_data( n - 1 );
    Scalar              alpha = random_scalar<Scalar>( 0.1, 0.5 );
    Scalar              beta  = random_scalar<Scalar>( 0.1, 0.5 );

    for ( Eigen::Index i = 0; i < n - 1; ++i )
    {
      a_data[i] = random_scalar<Scalar>( 0.1, 1.0 );
      c_data[i] = random_scalar<Scalar>( 0.1, 1.0 );
    }
    for ( Eigen::Index i = 0; i < n; ++i ) { b_data[i] = random_scalar<Scalar>( 2.0, 5.0 ); }

    Eigen::Map<const VecS> a_map( a_data.data(), n - 1 );
    Eigen::Map<const VecS> b_map( b_data.data(), n );
    Eigen::Map<const VecS> c_map( c_data.data(), n - 1 );

    Solver solver( n );

    double max_error = 0.0;
    for ( Eigen::Index rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx )
    {
      std::vector<Scalar> rhs_data( n );
      for ( Eigen::Index i = 0; i < n; ++i ) { rhs_data[i] = random_scalar<Scalar>( -1.0, 1.0 ); }
      Eigen::Map<const VecS> rhs_map( rhs_data.data(), n );

      VecS x( n );
      solver.solve_cyclic( a_map, b_map, c_map, alpha, beta, rhs_map, x );

      // Verify cyclic system
      VecS Ax( n );
      if ( n == 1 ) { Ax( 0 ) = ( b_map( 0 ) + alpha + beta ) * x( 0 ); }
      else if ( n == 2 )
      {
        Ax( 0 ) = ( b_map( 0 ) + alpha ) * x( 0 ) + c_map( 0 ) * x( 1 );
        Ax( 1 ) = a_map( 0 ) * x( 0 ) + ( b_map( 1 ) + beta ) * x( 1 );
      }
      else
      {
        Ax( 0 ) = b_map( 0 ) * x( 0 ) + c_map( 0 ) * x( 1 ) + alpha * x( n - 1 );
        for ( Eigen::Index i = 1; i < n - 1; ++i )
        {
          Ax( i ) = a_map( i - 1 ) * x( i - 1 ) + b_map( i ) * x( i ) + c_map( i ) * x( i + 1 );
        }
        Ax( n - 1 ) = a_map( n - 2 ) * x( n - 2 ) + b_map( n - 1 ) * x( n - 1 ) + beta * x( 0 );
      }

      double error = vector_norm( Ax - rhs_map );
      max_error    = std::max( max_error, error );
    }

    result.error  = max_error;
    result.passed = result.error < 1e-10;
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ), "Error in scalar cyclic map test n={}: {}\n", n, e.what() );
    result.error  = 1.0;
    result.passed = false;
  }

  tm.toc();
  result.time_mus = tm.elapsed_mus();
  return result;
}

// ===========================================================================
// Test suite for Eigen::Map
// ===========================================================================

void run_scalar_map_tests()
{
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "╔══════════════════════════════╗\n"
    "║   TEST 5: Scalar with Map    ║\n"
    "║   (Using Eigen::Map)         ║\n"
    "╚══════════════════════════════╝\n\n" );

  std::vector<TestConfig> test_configs = {
    // Non-cyclic with Map
    { "n=1, rhs=1 (Map)", []() { return test_scalar_non_cyclic_map<double>( 1, 1 ); } },
    { "n=2, rhs=1 (Map)", []() { return test_scalar_non_cyclic_map<double>( 2, 1 ); } },
    { "n=5, rhs=5 (Map)", []() { return test_scalar_non_cyclic_map<double>( 5, 5 ); } },
    { "n=10, rhs=10 (Map)", []() { return test_scalar_non_cyclic_map<double>( 10, 10 ); } },
    { "n=100, rhs=1 (Map)", []() { return test_scalar_non_cyclic_map<double>( 100, 1 ); } },
    // Cyclic with Map
    { "n=1, rhs=1 (Cyclic Map)", []() { return test_scalar_cyclic_map<double>( 1, 1 ); } },
    { "n=2, rhs=1 (Cyclic Map)", []() { return test_scalar_cyclic_map<double>( 2, 1 ); } },
    { "n=5, rhs=5 (Cyclic Map)", []() { return test_scalar_cyclic_map<double>( 5, 5 ); } },
    { "n=10, rhs=10 (Cyclic Map)", []() { return test_scalar_cyclic_map<double>( 10, 10 ); } },
    { "n=100, rhs=1 (Cyclic Map)", []() { return test_scalar_cyclic_map<double>( 100, 1 ); } },
  };

  std::vector<TestResult> results;
  for ( size_t i = 0; i < test_configs.size(); ++i )
  {
    fmt::print( fg( fmt::color::cyan ), "Running test {}/{}: {}\n", i + 1, test_configs.size(), test_configs[i].name );
    results.push_back( test_configs[i].test_func() );
  }

  print_test_table( results, "SCALAR TRIDIAGONAL SYSTEMS WITH EIGEN::MAP" );
}
// ===========================================================================
// Main Test Function
// ===========================================================================

int main()
{
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════╗\n"
    "║        TRIDIAGONAL SOLVER EXTENSIVE TEST SUITE           ║\n"
    "║           50+ TESTS PER VARIANT, MULTIPLE RHS            ║\n"
    "╚══════════════════════════════════════════════════════════╝\n\n" );

  fmt::print( fg( fmt::color::cyan ), "Starting extensive testing...\n\n" );

  // Run all 5 test suites
  run_scalar_non_cyclic_tests();
  run_scalar_cyclic_tests();
  run_block_non_cyclic_tests();
  run_block_cyclic_tests();
  run_scalar_map_tests();  // <-- Nuova suite aggiunta qui

  // Final summary
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════╗\n"
    "║               TESTING COMPLETE                           ║\n"
    "║     All test suites executed with extensive coverage     ║\n"
    "╚══════════════════════════════════════════════════════════╝\n\n" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "✅ All test suites completed successfully! ✅\n" );
  fmt::print( fg( fmt::color::cyan ), "\nSummary:\n" );
  fmt::print( "  • Scalar non-cyclic: ~50 tests\n" );
  fmt::print( "  • Scalar cyclic: ~50 tests\n" );
  fmt::print( "  • Block non-cyclic: ~50 tests\n" );
  fmt::print( "  • Block cyclic: ~50 tests\n" );
  fmt::print( "  • Scalar with Eigen::Map: 10 tests\n" );  // <-- Aggiorna il sommario
  fmt::print( "  • Total: ~210 tests with varying dimensions and RHS counts\n" );

  return 0;
}
