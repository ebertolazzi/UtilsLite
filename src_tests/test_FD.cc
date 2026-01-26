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

#if defined( __llvm__ ) || defined( __clang__ )
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include "Utils_FD.hh"
#include "Utils_fmt.hh"

using namespace Utils;
using Real = double;

// Test functions
Real test_function( Real x )
{
  return std::sin( 2.0 * x ) + std::exp( -0.5 * x ) * std::cos( 3.0 * x );
}

Real exact_first_derivative( Real x )
{
  return 2 * std::cos( 2 * x ) - std::exp( -x / 2.0 ) * std::cos( 3 * x ) / 2 -
         3 * std::exp( -x / 2 ) * std::sin( 3 * x );
}

Real exact_second_derivative( Real x )
{
  return ( -35 * std::cos( 3 * x ) + 12 * std::sin( 3 * x ) ) * std::exp( -x / 2 ) / 4.0 - 4.0 * std::sin( 2 * x );
}

// Grid generator
enum GridType
{
  UNIFORM,
  RANDOM_PERTURBED,
  GEOMETRIC,
  CHEBYSHEV
};

std::vector<Real> generate_grid( Real x0, int n, Real h, GridType type = UNIFORM )
{
  std::vector<Real> grid( n );
  grid[0] = x0;

  std::random_device rd;
  std::mt19937       gen( rd() );

  switch ( type )
  {
    case UNIFORM:
      for ( int i = 1; i < n; ++i ) { grid[i] = grid[i - 1] + h; }
      break;

    case RANDOM_PERTURBED:
    {
      std::uniform_real_distribution<Real> dist( -0.3 * h, 0.3 * h );
      for ( int i = 1; i < n; ++i ) { grid[i] = grid[i - 1] + h + dist( gen ); }
      // Ensure strictly increasing
      for ( int i = 1; i < n; ++i )
      {
        if ( grid[i] <= grid[i - 1] ) { grid[i] = grid[i - 1] + 0.1 * h; }
      }
      break;
    }

    case GEOMETRIC:
      for ( int i = 1; i < n; ++i ) { grid[i] = grid[i - 1] + h * std::pow( 1.1, i - 1 ); }
      break;

    case CHEBYSHEV:
      // Chebyshev points on [x0, x0 + (n-1)*h]
      Real L = ( n - 1 ) * h;
      for ( int i = 0; i < n; ++i ) { grid[i] = x0 + 0.5 * L * ( 1 - std::cos( M_PI * i / ( n - 1 ) ) ); }
      break;
  }

  return grid;
}

// Convergence rate calculation
Real compute_convergence_rate( const std::vector<Real> & errors, const std::vector<Real> & steps )
{
  int n = std::min( errors.size(), steps.size() );
  if ( n < 2 ) return 0.0;

  Real sum_log = 0.0;
  int  count   = 0;

  for ( int i = 1; i < n; ++i )
  {
    if ( errors[i - 1] > 0 && errors[i] > 0 && steps[i - 1] > 0 && steps[i] > 0 )
    {
      sum_log += std::log( errors[i - 1] / errors[i] ) / std::log( steps[i - 1] / steps[i] );
      count++;
    }
  }

  return count > 0 ? sum_log / count : 0.0;
}

// Test first derivatives
void test_first_derivatives( Real x0, const std::vector<Real> & steps, GridType grid_type )
{
  fmt::print(
    fmt::fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "┌─────────────────────────────────────────────────────────┐\n"
    "│           TEST FIRST DERIVATIVES {:<22} │\n"
    "└─────────────────────────────────────────────────────────┘\n\n",
    grid_type == UNIFORM            ? "(uniform grid)"
    : grid_type == RANDOM_PERTURBED ? "(random grid)"
    : grid_type == GEOMETRIC        ? "(geometric grid)"
                                    : "(Chebyshev grid)" );

  fmt::print(
    "{:^10} │ {:^12} │ {:^12} │ {:^12} │ {:^12} │ {:^12} │ {:^8}\n",
    "h",
    "2p error",
    "3p error",
    "4p error",
    "5p error",
    "exact",
    "type" );
  fmt::print( "{:-<90}\n", "" );

  std::vector<std::vector<Real>> all_errors( 4 );  // For 2p, 3p, 4p, 5p

  for ( Real h : steps )
  {
    std::vector<Real> grid = generate_grid( x0, 6, h, grid_type );  // 6 points for 5p
    std::vector<Real> y( grid.size() );

    for ( size_t i = 0; i < grid.size(); ++i ) { y[i] = test_function( grid[i] ); }

    Real exact = exact_first_derivative( x0 );
    Real errors[4];

    // 2 points
    errors[0] = std::abs( first_derivative_2p( grid[0], y[0], grid[1], y[1] ) - exact );
    all_errors[0].push_back( errors[0] );

    // 3 points
    if ( grid.size() >= 3 )
    {
      errors[1] = std::abs( first_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] ) - exact );
      all_errors[1].push_back( errors[1] );
    }

    // 4 points
    if ( grid.size() >= 4 )
    {
      errors[2] = std::abs( first_derivative_4p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3] ) - exact );
      all_errors[2].push_back( errors[2] );
    }

    // 5 points
    if ( grid.size() >= 5 )
    {
      errors[3] = std::abs(
        first_derivative_5p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3], grid[4], y[4] ) - exact );
      all_errors[3].push_back( errors[3] );
    }

    // Characteristic step (average of differences)
    Real h_char = 0.0;
    for ( size_t i = 1; i < grid.size(); ++i ) { h_char += grid[i] - grid[i - 1]; }
    h_char /= ( grid.size() - 1 );

    // Print results
    fmt::print( "{:10.3e} │ ", h_char );
    fmt::print(
      "{:12.3e} │ {:12.3e} │ {:12.3e} │ {:12.3e} │ {:12.3e} │ {}\n",
      errors[0],
      grid.size() >= 3 ? errors[1] : NAN,
      grid.size() >= 4 ? errors[2] : NAN,
      grid.size() >= 5 ? errors[3] : NAN,
      exact,
      grid_type == UNIFORM ? "uniform" : "non-uniform" );
  }

  // Compute convergence rates
  fmt::print( "\n{}\n", std::string( 90, '-' ) );
  fmt::print( "{:^90}\n", "ESTIMATED CONVERGENCE RATES" );
  fmt::print( "{:-<90}\n", "" );

  const char * methods[] = { "2 points", "3 points", "4 points", "5 points" };
  for ( int i = 0; i < 4; ++i )
  {
    if ( all_errors[i].size() >= 2 )
    {
      Real rate = compute_convergence_rate( all_errors[i], steps );
      fmt::print( "{:<10}: {:6.3f} (theoretical: {:6.3f}) ", methods[i], rate, i + 1.0 );

      // Color based on closeness to theoretical
      if ( std::abs( rate - ( i + 1.0 ) ) < 0.2 ) { fmt::print( fmt::fg( fmt::color::green ), "✓\n" ); }
      else if ( std::abs( rate - ( i + 1.0 ) ) < 0.5 ) { fmt::print( fmt::fg( fmt::color::yellow ), "⚠\n" ); }
      else
      {
        fmt::print( fmt::fg( fmt::color::red ), "✗\n" );
      }
    }
  }
}

// Test second derivatives
void test_second_derivatives( Real x0, const std::vector<Real> & steps, GridType grid_type )
{
  fmt::print(
    fmt::fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "┌─────────────────────────────────────────────────────────┐\n"
    "│           TEST SECOND DERIVATIVES {:<21} │\n"
    "└─────────────────────────────────────────────────────────┘\n\n",
    grid_type == UNIFORM            ? "(uniform grid)"
    : grid_type == RANDOM_PERTURBED ? "(random grid)"
    : grid_type == GEOMETRIC        ? "(geometric grid)"
                                    : "(Chebyshev grid)" );

  fmt::print(
    "{:^10} │ {:^12} │ {:^12} │ {:^12} │ {:^12} │ {:^8}\n",
    "h",
    "3p error",
    "4p error",
    "5p error",
    "exact",
    "type" );
  fmt::print( "{:-<80}\n", "" );

  std::vector<std::vector<Real>> all_errors( 3 );  // For 3p, 4p, 5p

  for ( Real h : steps )
  {
    std::vector<Real> grid = generate_grid( x0, 6, h, grid_type );  // 6 points for 5p
    std::vector<Real> y( grid.size() );

    for ( size_t i = 0; i < grid.size(); ++i ) { y[i] = test_function( grid[i] ); }

    Real exact = exact_second_derivative( x0 );
    Real errors[3];

    // 3 points
    if ( grid.size() >= 3 )
    {
      errors[0] = std::abs( second_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] ) - exact );
      all_errors[0].push_back( errors[0] );
    }

    // 4 points
    if ( grid.size() >= 4 )
    {
      errors[1] = std::abs(
        second_derivative_4p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3] ) - exact );
      all_errors[1].push_back( errors[1] );
    }

    // 5 points
    if ( grid.size() >= 5 )
    {
      errors[2] = std::abs(
        second_derivative_5p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3], grid[4], y[4] ) - exact );
      all_errors[2].push_back( errors[2] );
    }

    // Characteristic step
    Real h_char = 0.0;
    for ( size_t i = 1; i < grid.size(); ++i ) { h_char += grid[i] - grid[i - 1]; }
    h_char /= ( grid.size() - 1 );

    // Print results
    fmt::print( "{:10.3e} │ ", h_char );
    fmt::print(
      "{:12.3e} │ {:12.3e} │ {:12.3e} │ {:12.3e} │ {}\n",
      grid.size() >= 3 ? errors[0] : NAN,
      grid.size() >= 4 ? errors[1] : NAN,
      grid.size() >= 5 ? errors[2] : NAN,
      exact,
      grid_type == UNIFORM ? "uniform" : "non-uniform" );
  }

  // Compute convergence rates
  fmt::print( "\n{}\n", std::string( 80, '-' ) );
  fmt::print( "{:^80}\n", "ESTIMATED CONVERGENCE RATES" );
  fmt::print( "{:-<80}\n", "" );

  const char * methods[]           = { "3 points", "4 points", "5 points" };
  const Real   theoretical_rates[] = { 1.0, 2.0, 3.0 };  // Theoretical orders for second derivatives

  for ( int i = 0; i < 3; ++i )
  {
    if ( all_errors[i].size() >= 2 )
    {
      Real rate = compute_convergence_rate( all_errors[i], steps );
      fmt::print( "{:<10}: {:6.3f} (theoretical: {:6.3f}) ", methods[i], rate, theoretical_rates[i] );

      if ( std::abs( rate - theoretical_rates[i] ) < 0.2 ) { fmt::print( fmt::fg( fmt::color::green ), "✓\n" ); }
      else if ( std::abs( rate - theoretical_rates[i] ) < 0.5 ) { fmt::print( fmt::fg( fmt::color::yellow ), "⚠\n" ); }
      else
      {
        fmt::print( fmt::fg( fmt::color::red ), "✗\n" );
      }
    }
  }
}

// Consistency test
void test_consistency()
{
  fmt::print(
    fmt::fg( fmt::color::blue ) | fmt::emphasis::bold,
    "\n"
    "┌─────────────────────────────────────────────────────────┐\n"
    "│                   CONSISTENCY TEST                      │\n"
    "└─────────────────────────────────────────────────────────┘\n\n" );

  // Test with 2nd degree polynomial: formulas should be exact
  auto poly2              = []( Real x ) { return 3.0 * x * x + 2.0 * x + 1.0; };
  auto poly2_prime        = []( Real x ) { return 6.0 * x + 2.0; };
  auto poly2_double_prime = []( Real x ) { return 6.0; };

  Real x0 = 2.0;
  Real h  = 0.5;

  // Non-uniform grid
  std::vector<Real> grid = { x0, x0 + 0.3 * h, x0 + 1.7 * h, x0 + 2.5 * h, x0 + 4.0 * h };
  std::vector<Real> y( grid.size() );

  for ( size_t i = 0; i < grid.size(); ++i ) { y[i] = poly2( grid[i] ); }

  // Test first derivatives
  Real fd2         = first_derivative_2p( grid[0], y[0], grid[1], y[1] );
  Real fd3         = first_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] );
  Real fd4         = first_derivative_4p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3] );
  Real fd5         = first_derivative_5p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3], grid[4], y[4] );
  Real exact_prime = poly2_prime( x0 );

  fmt::print( "Test with 2nd degree polynomial (exact first derivative for 3p, 4p, 5p):\n" );
  fmt::print( "  Exact:     {:12.6f}\n", exact_prime );
  fmt::print(
    "  2 points:  {:12.6f}  error: {:8.2e} {}\n",
    fd2,
    std::abs( fd2 - exact_prime ),
    std::abs( fd2 - exact_prime ) < 1e-10 ? fmt::format( fmt::fg( fmt::color::green ), "✓" ) : "✗" );
  fmt::print(
    "  3 points:  {:12.6f}  error: {:8.2e} {}\n",
    fd3,
    std::abs( fd3 - exact_prime ),
    std::abs( fd3 - exact_prime ) < 1e-10 ? fmt::format( fmt::fg( fmt::color::green ), "✓" ) : "✗" );
  fmt::print(
    "  4 points:  {:12.6f}  error: {:8.2e} {}\n",
    fd4,
    std::abs( fd4 - exact_prime ),
    std::abs( fd4 - exact_prime ) < 1e-10 ? fmt::format( fmt::fg( fmt::color::green ), "✓" ) : "✗" );
  fmt::print(
    "  5 points:  {:12.6f}  error: {:8.2e} {}\n",
    fd5,
    std::abs( fd5 - exact_prime ),
    std::abs( fd5 - exact_prime ) < 1e-10 ? fmt::format( fmt::fg( fmt::color::green ), "✓" ) : "✗" );

  // Test second derivatives
  Real sd3          = second_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] );
  Real sd4          = second_derivative_4p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3] );
  Real sd5          = second_derivative_5p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3], grid[4], y[4] );
  Real exact_double = poly2_double_prime( x0 );

  fmt::print( "\nTest with 2nd degree polynomial (exact second derivative for 3p, 4p, 5p):\n" );
  fmt::print( "  Exact:     {:12.6f}\n", exact_double );
  fmt::print(
    "  3 points:  {:12.6f}  error: {:8.2e} {}\n",
    sd3,
    std::abs( sd3 - exact_double ),
    std::abs( sd3 - exact_double ) < 1e-10 ? fmt::format( fmt::fg( fmt::color::green ), "✓" ) : "✗" );
  fmt::print(
    "  4 points:  {:12.6f}  error: {:8.2e} {}\n",
    sd4,
    std::abs( sd4 - exact_double ),
    std::abs( sd4 - exact_double ) < 1e-10 ? fmt::format( fmt::fg( fmt::color::green ), "✓" ) : "✗" );
  fmt::print(
    "  5 points:  {:12.6f}  error: {:8.2e} {}\n",
    sd5,
    std::abs( sd5 - exact_double ),
    std::abs( sd5 - exact_double ) < 1e-10 ? fmt::format( fmt::fg( fmt::color::green ), "✓" ) : "✗" );
}

// Test with decreasing grid
void test_decreasing_grid()
{
  fmt::print(
    fmt::fg( fmt::color::orange ) | fmt::emphasis::bold,
    "\n"
    "┌─────────────────────────────────────────────────────────┐\n"
    "│                 TEST DECREASING GRID                    │\n"
    "└─────────────────────────────────────────────────────────┘\n\n" );

  Real x0 = 5.0;
  Real h  = 0.5;

  // Decreasing grid
  std::vector<Real> grid = { x0, x0 - 0.7 * h, x0 - 1.5 * h, x0 - 2.8 * h, x0 - 4.2 * h };
  std::vector<Real> y( grid.size() );

  for ( size_t i = 0; i < grid.size(); ++i ) { y[i] = test_function( grid[i] ); }

  Real exact_prime  = exact_first_derivative( x0 );
  Real exact_double = exact_second_derivative( x0 );

  fmt::print( "Decreasing grid: x = " );
  for ( auto xi : grid ) fmt::print( "{:6.3f} ", xi );
  fmt::print( "\n\n" );

  fmt::print( "First derivative:\n" );
  fmt::print( "  Exact:     {:12.6f}\n", exact_prime );
  fmt::print(
    "  2 points:  {:12.6f}  error: {:8.2e}\n",
    first_derivative_2p( grid[0], y[0], grid[1], y[1] ),
    std::abs( first_derivative_2p( grid[0], y[0], grid[1], y[1] ) - exact_prime ) );
  fmt::print(
    "  3 points:  {:12.6f}  error: {:8.2e}\n",
    first_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] ),
    std::abs( first_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] ) - exact_prime ) );

  fmt::print( "\nSecond derivative:\n" );
  fmt::print( "  Exact:     {:12.6f}\n", exact_double );
  fmt::print(
    "  3 points:  {:12.6f}  error: {:8.2e}\n",
    second_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] ),
    std::abs( second_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] ) - exact_double ) );
}

// Performance benchmark
void benchmark_performance()
{
  fmt::print(
    fmt::fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "┌─────────────────────────────────────────────────────────┐\n"
    "│                 PERFORMANCE BENCHMARK                   │\n"
    "└─────────────────────────────────────────────────────────┘\n\n" );

  const int         N = 1000000;
  std::vector<Real> x( N ), y( N );

  // Initialization
  for ( int i = 0; i < N; ++i )
  {
    x[i] = i * 0.01;
    y[i] = test_function( x[i] );
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Benchmark first derivative 3 points
  Real sum = 0.0;
  for ( int i = 0; i < N - 2; ++i )
  {
    sum += first_derivative_3p( x[i], y[i], x[i + 1], y[i + 1], x[i + 2], y[i + 2] );
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start );

  fmt::print( "First derivative 3 points:\n" );
  fmt::print( "  {} evaluations in {} ms\n", N - 2, duration.count() );
  fmt::print( "  {:.1f} evaluations/ms\n", ( N - 2 ) / double( duration.count() ) );
  fmt::print( "  Sum (check): {:.6f}\n\n", sum );
}

void test_polynomial_exactness()
{
  fmt::print(
    fmt::fg( fmt::color::magenta ) | fmt::emphasis::bold,
    "\n"
    "┌─────────────────────────────────────────────────────────┐\n"
    "│              POLYNOMIAL EXACTNESS TEST                  │\n"
    "└─────────────────────────────────────────────────────────┘\n\n" );

  // Test cubic polynomial: f(x) = x^3 - 2x^2 + 3x - 1
  // f'(x) = 3x^2 - 4x + 3
  // f''(x) = 6x - 4

  auto cubic              = []( Real x ) { return x * x * x - 2.0 * x * x + 3.0 * x - 1.0; };
  auto cubic_prime        = []( Real x ) { return 3.0 * x * x - 4.0 * x + 3.0; };
  auto cubic_double_prime = []( Real x ) { return 6.0 * x - 4.0; };

  Real x0 = 1.5;

  // Non-uniform grid
  Real              h    = 0.5;
  std::vector<Real> grid = { x0, x0 + 0.3 * h, x0 + 0.7 * h, x0 + 1.2 * h, x0 + 1.8 * h };
  std::vector<Real> y( grid.size() );

  for ( size_t i = 0; i < grid.size(); ++i ) { y[i] = cubic( grid[i] ); }

  Real exact_prime  = cubic_prime( x0 );
  Real exact_double = cubic_double_prime( x0 );

  fmt::print( "Cubic polynomial test at x = {:.3f}\n", x0 );
  fmt::print( "Exact f'(x) = {:.8f}, f''(x) = {:.8f}\n\n", exact_prime, exact_double );

  // First derivatives
  Real df2 = first_derivative_2p( grid[0], y[0], grid[1], y[1] );
  Real df3 = first_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] );
  Real df4 = first_derivative_4p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3] );
  Real df5 = first_derivative_5p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3], grid[4], y[4] );
  fmt::print( "First derivatives:\n" );
  fmt::print( "  2 points: {:.8f}  error: {:.2e}\n", df2, std::abs( df2 - exact_prime ) );
  fmt::print( "  3 points: {:.8f}  error: {:.2e}\n", df3, std::abs( df3 - exact_prime ) );
  fmt::print( "  4 points: {:.8f}  error: {:.2e}\n", df4, std::abs( df4 - exact_prime ) );
  fmt::print( "  5 points: {:.8f}  error: {:.2e}\n\n", df5, std::abs( df5 - exact_prime ) );

  Real ddf3 = second_derivative_3p( grid[0], y[0], grid[1], y[1], grid[2], y[2] );
  Real ddf4 = second_derivative_4p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3] );
  Real ddf5 = second_derivative_5p( grid[0], y[0], grid[1], y[1], grid[2], y[2], grid[3], y[3], grid[4], y[4] );

  // Second derivatives
  fmt::print( "Second derivatives:\n" );
  fmt::print( "  3 points: {:.8f}  error: {:.2e}\n", ddf3, std::abs( ddf3 - exact_double ) );
  fmt::print( "  4 points: {:.8f}  error: {:.2e}\n", ddf4, std::abs( ddf4 - exact_double ) );
  fmt::print( "  5 points: {:.8f}  error: {:.2e}\n", ddf5, std::abs( ddf5 - exact_double ) );
}

int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_green ) | fmt::emphasis::bold,
    "╔═══════════════════════════════════════════════════════════╗\n"
    "║       NON-UNIFORM FINITE DIFFERENCES TEST C++17           ║\n"
    "╚═══════════════════════════════════════════════════════════╝\n\n" );

  test_polynomial_exactness();

  Real              x0    = 2.5;
  std::vector<Real> steps = { 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0010101 };

  // Test 1: Uniform grids
  test_first_derivatives( x0, steps, UNIFORM );
  test_second_derivatives( x0, steps, UNIFORM );

  // Test 2: Random perturbed grids
  test_first_derivatives( x0, steps, RANDOM_PERTURBED );
  test_second_derivatives( x0, steps, RANDOM_PERTURBED );

  // Test 3: Geometric grids
  test_first_derivatives( x0, steps, GEOMETRIC );
  test_second_derivatives( x0, steps, GEOMETRIC );

  // Test 4: Consistency test
  test_consistency();

  // Test 5: Decreasing grid
  test_decreasing_grid();

  // Test 6: Performance benchmark
  benchmark_performance();


  fmt::print(
    fmt::fg( fmt::color::light_green ) | fmt::emphasis::bold,
    "\n"
    "╔═══════════════════════════════════════════════════════════╗\n"
    "║                TESTS COMPLETED SUCCESSFULLY               ║\n"
    "╚═══════════════════════════════════════════════════════════╝\n" );

  return 0;
}
