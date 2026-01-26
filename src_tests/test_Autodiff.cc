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

#include "Utils.hh"
#include "Utils_autodiff.hh"
#include <array>
#include <vector>
#include <functional>
#include <iomanip>
#include <cmath>
#include <limits>

using namespace std;
using namespace autodiff;
using namespace Utils;

// ============================================================================
// SECTION: Utility Functions for Testing
// ============================================================================

namespace TestUtils
{

  // Unicode symbols
  constexpr const char * CHECKMARK = "✓";
  constexpr const char * XMARK     = "✗";
  constexpr const char * WARNING   = "⚠";
  constexpr const char * INFO      = "ℹ";

  // Print a colored header
  void print_header( const string & title, char fill_char = '=', int width = 80 )
  {
    string fill_str( 5, fill_char );
    int    padding = ( width - static_cast<int>( title.length() ) - 10 ) / 2;
    if ( padding < 0 ) padding = 0;

    fmt::print(
      fg( fmt::color::cyan ) | fmt::emphasis::bold,
      "\n{}{: <{}} {} {: <{}}{}\n",
      fill_str,
      "",
      padding,
      title,
      "",
      padding,
      fill_str );
  }

  void print_subheader( const string & title, char fill_char = '-', int width = 60 )
  {
    string fill_str( 5, fill_char );
    int    padding = ( width - static_cast<int>( title.length() ) - 10 ) / 2;
    if ( padding < 0 ) padding = 0;

    fmt::print(
      fg( fmt::color::cyan ) | fmt::emphasis::bold,
      "\n{}{: <{}} {} {: <{}}{}\n",
      fill_str,
      "",
      padding,
      title,
      "",
      padding,
      fill_str );
  }

template <typename T>
bool check_result( const std::string & test_name, const T & computed, const T & expected, double tolerance = 1e-10 )
{
    double error;
    bool success;

    // 1. Gestione speciale per gli infiniti
    if ( std::isinf( computed ) && std::isinf( expected ) )
    {
        // Entrambi infiniti: successo se hanno lo stesso segno
        // (computed > 0) == (expected > 0) verifica la concordanza di segno
        success = ( ( computed > 0 ) == ( expected > 0 ) );
        error   = success ? 0.0 : std::numeric_limits<double>::infinity();
    }
    else
    {
        // 2. Calcolo standard (gestisce anche Finito vs Infinito correttamente)
        // Se uno è finito e l'altro infinito, error sarà 'inf', che è > tolerance -> FAIL
        error   = std::abs( computed - expected );
        
        // Gestione del caso NaN nel calcolo dell'errore (opzionale ma consigliato)
        if ( std::isnan( error ) ) success = false;
        else success = ( error <= tolerance );
    }

    // --- Output (invariato o leggermente adattato) ---

    // Test name
    fmt::print( "  {:<45}", test_name );

    // Status with color
    if ( success )
    {
        fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "{} PASS ", CHECKMARK );
    }
    else
    {
        fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "{} FAIL ", XMARK );
    }

    // Values format
    // Usiamo {:>12.6g} o {:>12.6f}. .6g passa automaticamente a notazione scientifica se serve.
    fmt::print( "computed: {:>12.6g}, expected: {:>12.6g}, error: {:>8.2e}\n", 
                computed, expected, error );

    return success;
}

  bool check_approx( double computed, double expected, double tolerance = 1e-10 )
  {
    return abs( computed - expected ) <= tolerance;
  }

}  // namespace TestUtils

// ============================================================================
// SECTION: Test Functions
// ============================================================================

// Basic function tests
dual f1( dual x )
{
  return 1 + x + x * x + 1 / x + log( x );
}

dual f2( dual x )
{
  return sin( x ) * cos( x ) + exp( x );
}

dual f3( dual x )
{
  return tan( x ) - asin( x ) + acos( x );
}

// Multi-variable functions
dual g1( dual x, dual y )
{
  return x * x + y * y + x * y;
}

dual g2( dual x, dual y )
{
  return sin( x * y ) + cos( x / y );
}

dual g3( dual x, dual y, dual z )
{
  return exp( x * y * z ) + log( x + y + z );
}

// Higher order derivatives
dual2nd f_second( dual2nd x )
{
  return x * x * x + sin( x ) * cos( x );
}

// ============================================================================
// SECTION: Analytical Derivatives (for verification)
// ============================================================================

double df1_analytical( double x )
{
  return 1 + 2 * x - 1 / ( x * x ) + 1 / x;
}

double df2_analytical( double x )
{
  return cos( x ) * cos( x ) - sin( x ) * sin( x ) + exp( x );
}

double d2f2_analytical( double x )
{
  return -4 * sin( x ) * cos( x ) + exp( x );
}

// Gradient for multi-variable functions
array<double, 2> dg1_analytical( double x, double y )
{
  return { 2 * x + y, 2 * y + x };
}

// ============================================================================
// SECTION: Comprehensive Math Function Tests
// ============================================================================
// Replace the test_all_math_functions() function with this corrected version:

void test_all_math_functions()
{
  using namespace TestUtils;

  print_header( "COMPREHENSIVE MATH FUNCTION TESTS" );

  int passed = 0, total = 0;

  // Test points for different ranges
  vector<double> test_points_positive = { 0.5, 1.0, 2.0, M_PI / 4 };
  vector<double> test_points_all      = { -2.0, -1.0, 0.5, 1.0, 2.0 };

  // ========================================================================
  // TRIGONOMETRIC FUNCTIONS
  // ========================================================================

  print_subheader( "TRIGONOMETRIC FUNCTIONS" );

  for ( double x_val : test_points_all )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // sin(x) -> derivative is cos(x)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return sin( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = cos( x_val );
      bool   ok       = check_result( "d/dx[sin(x)] = cos(x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // cos(x) -> derivative is -sin(x)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return cos( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = -sin( x_val );
      bool   ok       = check_result( "d/dx[cos(x)] = -sin(x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // tan(x) -> derivative is 1/cos²(x)
    if ( abs( cos( x_val ) ) > 0.01 )
    {  // Avoid singularities
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return tan( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / ( cos( x_val ) * cos( x_val ) );
      bool   ok       = check_result( "d/dx[tan(x)] = 1/cos²(x)", deriv, expected, 1e-9 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // INVERSE TRIGONOMETRIC FUNCTIONS
  // ========================================================================

  print_subheader( "INVERSE TRIGONOMETRIC FUNCTIONS" );

  vector<double> test_points_inverse = { -0.5, 0.0, 0.5, 0.8 };

  for ( double x_val : test_points_inverse )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // asin(x) -> derivative is 1/sqrt(1-x²)
    if ( abs( x_val ) < 0.99 )
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return asin( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / sqrt( 1.0 - x_val * x_val );
      bool   ok       = check_result( "d/dx[asin(x)] = 1/√(1-x²)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // acos(x) -> derivative is -1/sqrt(1-x²)
    if ( abs( x_val ) < 0.99 )
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return acos( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = -1.0 / sqrt( 1.0 - x_val * x_val );
      bool   ok       = check_result( "d/dx[acos(x)] = -1/√(1-x²)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // atan(x) -> derivative is 1/(1+x²)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return atan( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / ( 1.0 + x_val * x_val );
      bool   ok       = check_result( "d/dx[atan(x)] = 1/(1+x²)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // HYPERBOLIC FUNCTIONS
  // ========================================================================

  print_subheader( "HYPERBOLIC FUNCTIONS" );

  for ( double x_val : test_points_all )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // sinh(x) -> derivative is cosh(x)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return sinh( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = cosh( x_val );
      bool   ok       = check_result( "d/dx[sinh(x)] = cosh(x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // cosh(x) -> derivative is sinh(x)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return cosh( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = sinh( x_val );
      bool   ok       = check_result( "d/dx[cosh(x)] = sinh(x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // tanh(x) -> derivative is 1/cosh²(x)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return tanh( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double cosh_val = cosh( x_val );
      double expected = 1.0 / ( cosh_val * cosh_val );
      bool   ok       = check_result( "d/dx[tanh(x)] = 1/cosh²(x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // INVERSE HYPERBOLIC FUNCTIONS
  // ========================================================================

  print_subheader( "INVERSE HYPERBOLIC FUNCTIONS" );

  for ( double x_val : test_points_positive )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // asinh(x) -> derivative is 1/sqrt(1+x²)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return asinh( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / sqrt( 1.0 + x_val * x_val );
      bool   ok       = check_result( "d/dx[asinh(x)] = 1/√(1+x²)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // acosh(x) -> derivative is 1/sqrt(x²-1), valid for x > 1
    if ( x_val > 1.01 )
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return acosh( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / sqrt( x_val * x_val - 1 );
      bool   ok       = check_result( "d/dx[acosh(x)] = 1/√(x²-1)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // atanh(x) -> derivative is 1/(1-x²), valid for |x| < 1
    if ( abs( x_val ) < 0.99 )
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return atanh( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / ( 1.0 - x_val * x_val );
      bool   ok       = check_result( "d/dx[atanh(x)] = 1/(1-x²)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // EXPONENTIAL AND LOGARITHMIC FUNCTIONS
  // ========================================================================

  print_subheader( "EXPONENTIAL AND LOGARITHMIC FUNCTIONS" );

  for ( double x_val : test_points_positive )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // exp(x) -> derivative is exp(x)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return exp( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = exp( x_val );
      bool   ok       = check_result( "d/dx[exp(x)] = exp(x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // log(x) -> derivative is 1/x
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return log( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / x_val;
      bool   ok       = check_result( "d/dx[log(x)] = 1/x", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // POWER FUNCTIONS
  // ========================================================================

  print_subheader( "POWER FUNCTIONS" );

  for ( double x_val : test_points_positive )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // sqrt(x) -> derivative is 1/(2*sqrt(x))
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return sqrt( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / ( 2.0 * sqrt( x_val ) );
      bool   ok       = check_result( "d/dx[√x] = 1/(2√x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // cbrt(x) -> derivative is 1/(3 * cbrt(x)^2)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return cbrt( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 1.0 / ( 3.0 * pow( x_val, 2.0 / 3.0 ) );  // equivalente a 1/(3 * cbrt(x)^2)
      bool   ok       = check_result( "d/dx[cbrt(x)] = 1/(3*cbrt(x)^2)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // pow(x, 3) -> derivative is 3*x²
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return pow( x, 3.0 ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 3.0 * x_val * x_val;
      bool   ok       = check_result( "d/dx[x³] = 3x²", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // pow(x, 0.5) -> derivative is 0.5*x^(-0.5)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return pow( x, 0.5 ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 0.5 * pow( x_val, -0.5 );
      bool   ok       = check_result( "d/dx[x^0.5] = 0.5·x^(-0.5)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // ABSOLUTE VALUE AND SIGN FUNCTIONS
  // ========================================================================

  print_subheader( "ABSOLUTE VALUE AND RELATED FUNCTIONS" );

  vector<double> test_points_sign = { -2.0, -0.5, 0.5, 2.0 };

  for ( double x_val : test_points_sign )
  {
    if ( abs( x_val ) < 0.01 ) continue;  // Skip near-zero for abs

    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // abs(x) -> derivative is sign(x) for x != 0
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return abs( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = ( x_val > 0 ) ? 1.0 : -1.0;
      bool   ok       = check_result( "d/dx[|x|] = sign(x)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // SPECIAL FUNCTIONS
  // ========================================================================

  print_subheader( "SPECIAL FUNCTIONS" );

  for ( double x_val : { 0.3, 1.5, 2.7 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // erf(x) -> derivative is (2/√π)*exp(-x²)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return erf( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = ( 2.0 / sqrt( M_PI ) ) * exp( -x_val * x_val );
      bool   ok       = check_result( "d/dx[erf(x)] = (2/√π)·exp(-x²)", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // COMBINED OPERATIONS
  // ========================================================================

  print_subheader( "COMBINED OPERATIONS" );

  for ( double x_val : { 0.5, 1.0, 2.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // sin(x) + cos(x) -> derivative is cos(x) - sin(x)
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return sin( x ) + cos( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = cos( x_val ) - sin( x_val );
      bool   ok       = check_result( "d/dx[sin(x)+cos(x)]", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // exp(x) * sin(x) -> derivative is exp(x)*(sin(x) + cos(x))
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return exp( x ) * sin( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = exp( x_val ) * ( sin( x_val ) + cos( x_val ) );
      bool   ok       = check_result( "d/dx[exp(x)·sin(x)]", deriv, expected, 1e-10 );
      passed += ok;
      total += 1;
    }

    // log(sin(x)) for x > 0 -> derivative is cot(x)
    if ( sin( x_val ) > 0.01 )
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return log( sin( x ) ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = cos( x_val ) / sin( x_val );
      bool   ok       = check_result( "d/dx[log(sin(x))] = cot(x)", deriv, expected, 1e-9 );
      passed += ok;
      total += 1;
    }
  }

  // Print summary
  fmt::print( "\n" );
  fmt::print( "{}\n", string( 80, '-' ) );
  if ( passed == total )
  {
    fmt::print(
      fg( fmt::color::green ) | fmt::emphasis::bold,
      "Math Functions: {}/{} tests passed ✓\n",
      passed,
      total );
  }
  else
  {
    fmt::print( fg( fmt::color::orange ) | fmt::emphasis::bold, "Math Functions: {}/{} tests passed\n", passed, total );
  }
  fmt::print( "{}\n", string( 80, '-' ) );
}


// ============================================================================
// SECTION: Original Test Functions
// ============================================================================

void test_basic_functions()
{
  using namespace TestUtils;

  print_header( "BASIC FUNCTION TESTS" );

  vector<double> test_points = { 0.5, 1.0, 2.0, 3.0, 5.0 };
  int            passed = 0, total = 0;

  for ( double x_val : test_points )
  {
    print_subheader( "Test point: x = " + to_string( x_val ) );

    // Test f1
    dual   x    = x_val;
    dual   u    = f1( x );
    double dudx = derivative( f1, wrt( x ), at( x ) );
    bool   ok1  = check_result( "f(x)=1+x+x²+1/x+ln(x)", dudx, df1_analytical( x_val ), 1e-9 );

    // Test f2
    x        = x_val;
    u        = f2( x );
    dudx     = derivative( f2, wrt( x ), at( x ) );
    bool ok2 = check_result( "f(x)=sin(x)cos(x)+exp(x)", dudx, df2_analytical( x_val ), 1e-9 );

    passed += ok1 + ok2;
    total += 2;

    // Test composition of functions
    auto f_composite = []( dual x ) -> dual { return sin( exp( x ) ) * cos( log( 1 + x ) ); };

    // Finite difference for verification
    double h       = 1e-7;
    double f_plus  = sin( exp( x_val + h ) ) * cos( log( 1 + x_val + h ) );
    double f_minus = sin( exp( x_val - h ) ) * cos( log( 1 + x_val - h ) );
    double df_fd   = ( f_plus - f_minus ) / ( 2 * h );

    x            = x_val;
    double df_ad = derivative( f_composite, wrt( x ), at( x ) );
    bool   ok3   = check_result( "Composite: sin(exp(x))*cos(log(1+x))", df_ad, df_fd, 1e-6 );

    passed += ok3;
    total += 1;
  }

  fmt::print( "\n" );
  if ( passed == total )
  {
    fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "Basic Functions: {}/{} tests passed\n", passed, total );
  }
  else
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "Basic Functions: {}/{} tests passed\n", passed, total );
  }
}

void test_multi_variable()
{
  using namespace TestUtils;

  print_header( "MULTI-VARIABLE FUNCTION TESTS" );

  vector<array<double, 2>> test_points = { { 1.0, 2.0 }, { 0.5, 3.0 }, { 2.0, 1.0 }, { 3.0, 0.5 } };

  int passed = 0, total = 0;

  for ( const auto & point : test_points )
  {
    double x_val = point[0], y_val = point[1];

    print_subheader( fmt::format( "Test point: (x,y) = ({}, {})", x_val, y_val ) );

    // Test g1
    dual x = x_val, y = y_val;
    dual u = g1( x, y );

    double dudx = derivative( g1, wrt( x ), at( x, y ) );
    double dudy = derivative( g1, wrt( y ), at( x, y ) );

    auto grad_analytical = dg1_analytical( x_val, y_val );

    bool ok1 = check_result( "∂g/∂x for x²+y²+xy", dudx, grad_analytical[0], 1e-9 );
    bool ok2 = check_result( "∂g/∂y for x²+y²+xy", dudy, grad_analytical[1], 1e-9 );

    passed += ok1 + ok2;
    total += 2;

    // Test g3 (three variables)
    double z_val = 0.5;
    dual   z     = z_val;
    x            = x_val;
    y            = y_val;
    u            = g3( x, y, z );

    dudx = derivative( g3, wrt( x ), at( x, y, z ) );
    dudy = derivative( g3, wrt( y ), at( x, y, z ) );
    // double dudz = derivative( g3, wrt( z ), at( x, y, z ) );

    // Finite difference check for dudx
    double h       = 1e-7;
    double f_plus  = exp( ( x_val + h ) * y_val * z_val ) + log( ( x_val + h ) + y_val + z_val );
    double f_minus = exp( ( x_val - h ) * y_val * z_val ) + log( ( x_val - h ) + y_val + z_val );
    double dudx_fd = ( f_plus - f_minus ) / ( 2 * h );

    bool ok3 = check_result( "∂g/∂x for exp(xyz)+ln(x+y+z)", dudx, dudx_fd, 1e-6 );

    passed += ok3;
    total += 1;
  }

  fmt::print( "\n" );
  if ( passed == total )
  {
    fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "Multi-variable: {}/{} tests passed\n", passed, total );
  }
  else
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "Multi-variable: {}/{} tests passed\n", passed, total );
  }
}

void test_higher_order()
{
  using namespace TestUtils;

  print_header( "HIGHER ORDER DERIVATIVE TESTS" );

  vector<double> test_points = { 0.5, 1.0, 2.0, 3.14159 / 4 };
  int            passed = 0, total = 0;

  for ( double x_val : test_points )
  {
    print_subheader( "Test point: x = " + to_string( x_val ) );

    // First and second derivatives using dual2nd
    dual2nd x = x_val;

    // Test f_second
    auto   result_f = derivatives( f_second, wrt( x, x ), at( x ) );
    double fx       = result_f[1];
    double fxx      = result_f[2];

    // Analytical first derivative
    double fx_analytical = 3 * x_val * x_val + cos( x_val ) * cos( x_val ) - sin( x_val ) * sin( x_val );

    // Analytical second derivative
    double fxx_analytical = 6 * x_val - 4 * sin( x_val ) * cos( x_val );

    bool ok1 = check_result( "f'(x) for x³+sin(x)cos(x)", fx, fx_analytical, 1e-9 );
    bool ok2 = check_result( "f''(x) for x³+sin(x)cos(x)", fxx, fxx_analytical, 1e-9 );

    // Test second derivative for f2
    dual2nd x2 = x_val;
    auto    result_f2 =
      derivatives( []( dual2nd x ) -> dual2nd { return sin( x ) * cos( x ) + exp( x ); }, wrt( x2, x2 ), at( x2 ) );

    double f2_xx = result_f2[2];
    bool   ok3   = check_result( "f''(x) for sin(x)cos(x)+exp(x)", f2_xx, d2f2_analytical( x_val ), 1e-9 );

    passed += ok1 + ok2 + ok3;
    total += 3;
  }

  fmt::print( "\n" );
  if ( passed == total )
  {
    fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "Higher Order: {}/{} tests passed\n", passed, total );
  }
  else
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "Higher Order: {}/{} tests passed\n", passed, total );
  }
}

void test_edge_cases()
{
  using namespace TestUtils;

  print_header( "EDGE CASE TESTS" );

  int passed = 0, total = 0;

  vector<double> edge_points = { 1e-10, 1e-5, 1.0, 10.0 };

  for ( double x_val : edge_points )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTesting at x = {}\n", x_val );

    // Test near zero
    dual x = x_val;

    // Test function with division - derivata analitica: (x*cos(x) - sin(x) - 1)/x²
    if ( abs( x_val ) > 1e-15 )
    {  // Avoid division by zero
      auto   div_func = []( dual x ) -> dual { return 1 / x + sin( x ) / x; };
      double ddiv_ad  = derivative( div_func, wrt( x ), at( x ) );

      // Derivata analitica: d/dx[1/x + sin(x)/x] = -1/x² + (x*cos(x) - sin(x))/x²
      // = (x*cos(x) - sin(x) - 1)/x²
      double ddiv_analytical = ( x_val * cos( x_val ) - sin( x_val ) - 1 ) / ( x_val * x_val );

      bool ok1 = check_result( "d/dx[1/x + sin(x)/x]", ddiv_ad, ddiv_analytical, 1e-9 );
      passed += ok1;
      total += 1;
    }

    // Test exponential for large values
    auto   exp_func        = []( dual x ) -> dual { return exp( x ) + exp( -x ); };
    double dexp_ad         = derivative( exp_func, wrt( x ), at( x ) );
    double dexp_analytical = exp( x_val ) - exp( -x_val );

    bool ok2 = check_result( "d/dx[exp(x)+exp(-x)]", dexp_ad, dexp_analytical, 1e-9 );

    // Test log
    if ( x_val > 0 )
    {
      auto   log_func        = []( dual x ) -> dual { return log( 1 + x ) + log( x ); };
      double dlog_ad         = derivative( log_func, wrt( x ), at( x ) );
      double dlog_analytical = 1 / ( 1 + x_val ) + 1 / x_val;

      bool ok3 = check_result( "d/dx[ln(1+x)+ln(x)]", dlog_ad, dlog_analytical, 1e-9 );
      passed += ok3;
      total += 1;
    }

    passed += ok2;
    total += 1;

    // Test aggiuntivo: funzione complessa con composizione
    auto   complex_func = []( dual x ) -> dual { return exp( sin( x ) ) * cos( x ) / sqrt( 1 + x * x ); };
    double dcomplex_ad  = derivative( complex_func, wrt( x ), at( x ) );

    // Derivata analitica corretta:
    // f(x) = u * v * w
    // u = exp(sin(x)), v = cos(x), w = 1/sqrt(1+x^2)
    // f' = u'vw + uv'w + uvw'
    
    double sinx  = sin( x_val );
    double cosx  = cos( x_val );
    double sq_xp1 = 1.0 + x_val * x_val; // (1+x^2)
    double denom = sqrt( sq_xp1 );       // sqrt(1+x^2)

    double term1 = ( cosx * cosx ) / denom;             // da d(exp(sin))/dx
    double term2 = -sinx / denom;                       // da d(cos)/dx
    double term3 = -( x_val * cosx ) / ( denom * sq_xp1 ); // da d(1/sqrt)/dx -> o pow(..., 1.5)

    double dcomplex_analytical = exp( sinx ) * ( term1 + term2 + term3 );

    bool ok4 = check_result( "d/dx[exp(sin(x))*cos(x)/sqrt(1+x²)]", dcomplex_ad, dcomplex_analytical, 1e-8 );
    passed += ok4;
    total += 1;

  }

  // Test su punti critici particolari
  fmt::print( fg( fmt::color::yellow ), "\nTesting special points\n" );

  vector<pair<double, string>> special_points = { { 0.0, "zero" },
                                                  { M_PI, "pi" },
                                                  { M_PI / 2, "pi/2" },
                                                  { 2 * M_PI, "2pi" } };

  for ( auto & [x_val, name] : special_points )
  {
    if ( x_val == 0.0 ) continue;  // già gestito sopra

    fmt::print( fg( fmt::color::yellow ), "\n  Testing at x = {} ({})\n", x_val, name );
    dual x = x_val;

    // Test funzione tangente
    auto   tan_func        = []( dual x ) -> dual { return tan( x ); };
    double dtan_ad         = derivative( tan_func, wrt( x ), at( x ) );
    double dtan_analytical = 1.0 / ( cos( x_val ) * cos( x_val ) );

    bool ok = check_result( "d/dx[tan(x)]", dtan_ad, dtan_analytical, 1e-9 );
    passed += ok;
    total += 1;
  }

  // Test NaN e Inf handling
  fmt::print( fg( fmt::color::yellow ), "\nTesting special values\n" );

  dual x = numeric_limits<double>::quiet_NaN();
  dual u = sin( x );
  fmt::print( fg( fmt::color::yellow ), "  {} sin(NaN) = {}\n", WARNING, u.val );

  x = numeric_limits<double>::infinity();
  u = exp( x );
  fmt::print( fg( fmt::color::yellow ), "  {} exp(Inf) = {}\n", WARNING, u.val );

  // Test asintotici
  x = 1e-6;
  u = sin( 1 / x );
  fmt::print( fg( fmt::color::yellow ), "  {} sin(1/1e-6) = {}\n", INFO, u.val );

  fmt::print( "\n" );
  if ( passed == total )
  {
    fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "Edge Cases: {}/{} tests passed\n", passed, total );
  }
  else
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "Edge Cases: {}/{} tests passed\n", passed, total );
  }
}


void test_performance()
{
  using namespace TestUtils;

  print_header( "PERFORMANCE TESTS" );

  const int N = 100000;

  // Time autodiff
  auto start = chrono::high_resolution_clock::now();

  double sum_ad = 0.0;
  for ( int i = 0; i < N; ++i )
  {
    double x_val = 1.0 + i * 0.0001;
    dual   x     = x_val;

    auto func = []( dual x ) -> dual { return sin( x ) * cos( x ) + exp( x ) + log( 1 + x ); };

    sum_ad += derivative( func, wrt( x ), at( x ) );
  }

  auto end         = chrono::high_resolution_clock::now();
  auto duration_ad = chrono::duration<double>( end - start ).count();

  // Time finite differences (for comparison)
  start = chrono::high_resolution_clock::now();

  double sum_fd = 0.0;
  double h      = 1e-7;
  for ( int i = 0; i < N; ++i )
  {
    double x_val = 1.0 + i * 0.0001;

    auto func = []( double x ) -> double { return sin( x ) * cos( x ) + exp( x ) + log( 1 + x ); };

    double df_fd = ( func( x_val + h ) - func( x_val - h ) ) / ( 2 * h );
    sum_fd += df_fd;
  }

  end              = chrono::high_resolution_clock::now();
  auto duration_fd = chrono::duration<double>( end - start ).count();

  fmt::print( "\nPerformance comparison ({} evaluations):\n", N );
  fmt::print( "  Autodiff:    {:.6f} seconds\n", duration_ad );
  fmt::print( "  Finite diff: {:.6f} seconds\n", duration_fd );
  fmt::print( "  Speed ratio: {:.2f}x\n", duration_fd / duration_ad );
  fmt::print( "  Result diff: {:.2e}\n", abs( sum_ad - sum_fd ) );
}

// ============================================================================
// SECTION: Test Functions Defined in Utils_autodiff.hh
// ============================================================================

void test_utils_functions()
{
  using namespace TestUtils;

  print_header( "UTILS_AUTODIFF.HH FUNCTIONS TESTS" );

  vector<double> test_points_positive = { 0.5, 1.0, 2.0, 3.0, 5.0 };
  vector<double> test_points_all      = { -2.0, -0.5, 0.0, 0.5, 2.0 };

  int passed = 0, total = 0;

  // ========================================================================
  // TEST CBRT (cubic root)
  // ========================================================================
  print_subheader( "CBRT FUNCTION (cubic root)" );

  for ( double x_val : test_points_all )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    dual   x     = x_val;
    auto   func  = []( dual x ) -> dual { return cbrt( x ); };
    double deriv = derivative( func, wrt( x ), at( x ) );

    // Analytical derivative: d/dx[cbrt(x)] = 1/(3 * cbrt(x)^2)
    double expected = 1.0 / ( 3.0 * pow( abs( x_val ), 2.0 / 3.0 ) );

    bool ok = check_result( "d/dx[cbrt(x)]", deriv, expected, 1e-9 );
    passed += ok;
    total += 1;
  }

  // ========================================================================
  // TEST ERFC (complementary error function)
  // ========================================================================
  print_subheader( "ERFC FUNCTION (complementary error)" );

  for ( double x_val : { -2.0, -1.0, 0.0, 1.0, 2.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    dual   x     = x_val;
    auto   func  = []( dual x ) -> dual { return erfc( x ); };
    double deriv = derivative( func, wrt( x ), at( x ) );

    // Analytical derivative: d/dx[erfc(x)] = -2/√π * exp(-x²)
    double expected = -2.0 / sqrt( M_PI ) * exp( -x_val * x_val );

    bool ok = check_result( "d/dx[erfc(x)]", deriv, expected, 1e-9 );
    passed += ok;
    total += 1;
  }

  // ========================================================================
  // TEST ROUND, FLOOR, CEIL (should have zero derivative)
  // ========================================================================
  print_subheader( "ROUND, FLOOR, CEIL FUNCTIONS" );

  for ( double x_val : { 0.3, 0.7, 1.2, 2.8 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // Test round
    {
      dual   x     = x_val;
      auto   func  = []( dual x ) -> dual { return round( x ); };
      double deriv = derivative( func, wrt( x ), at( x ) );
      bool   ok    = check_result( "d/dx[round(x)] (should be 0)", deriv, 0.0, 1e-9 );
      passed += ok;
      total += 1;
    }

    // Test floor
    {
      dual   x     = x_val;
      auto   func  = []( dual x ) -> dual { return floor( x ); };
      double deriv = derivative( func, wrt( x ), at( x ) );
      bool   ok    = check_result( "d/dx[floor(x)] (should be 0)", deriv, 0.0, 1e-9 );
      passed += ok;
      total += 1;
    }

    // Test ceil
    {
      dual   x     = x_val;
      auto   func  = []( dual x ) -> dual { return ceil( x ); };
      double deriv = derivative( func, wrt( x ), at( x ) );
      bool   ok    = check_result( "d/dx[ceil(x)] (should be 0)", deriv, 0.0, 1e-9 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // TEST LOG1P (log(1 + x))
  // ========================================================================
  print_subheader( "LOG1P FUNCTION (log(1 + x))" );

  for ( double x_val : { -0.5, 0.0, 0.5, 1.0, 2.0 } )
  {
    if ( x_val <= -1.0 ) continue;  // Domain restriction

    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    dual   x     = x_val;
    auto   func  = []( dual x ) -> dual { return log1p( x ); };
    double deriv = derivative( func, wrt( x ), at( x ) );

    // Analytical derivative: d/dx[log(1 + x)] = 1/(1 + x)
    double expected = 1.0 / ( 1.0 + x_val );

    bool ok = check_result( "d/dx[log1p(x)]", deriv, expected, 1e-9 );
    passed += ok;
    total += 1;
  }

  // ========================================================================
  // TEST ATANH, ASINH, ACOSH
  // ========================================================================
  print_subheader( "INVERSE HYPERBOLIC FUNCTIONS" );

  // ATANH
  for ( double x_val : { -0.8, -0.5, 0.0, 0.5, 0.8 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point for atanh: x = {:.4f}\n", x_val );

    dual   x     = x_val;
    auto   func  = []( dual x ) -> dual { return atanh( x ); };
    double deriv = derivative( func, wrt( x ), at( x ) );

    // Analytical derivative: d/dx[atanh(x)] = 1/(1 - x²)
    double expected = 1.0 / ( 1.0 - x_val * x_val );

    bool ok = check_result( "d/dx[atanh(x)]", deriv, expected, 1e-9 );
    passed += ok;
    total += 1;
  }

  // ASINH
  for ( double x_val : { -2.0, -1.0, 0.0, 1.0, 2.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point for asinh: x = {:.4f}\n", x_val );

    dual   x     = x_val;
    auto   func  = []( dual x ) -> dual { return asinh( x ); };
    double deriv = derivative( func, wrt( x ), at( x ) );

    // Analytical derivative: d/dx[asinh(x)] = 1/√(1 + x²)
    double expected = 1.0 / sqrt( 1.0 + x_val * x_val );

    bool ok = check_result( "d/dx[asinh(x)]", deriv, expected, 1e-9 );
    passed += ok;
    total += 1;
  }

  // ACOSH (domain: x >= 1)
  for ( double x_val : { 1.1, 1.5, 2.0, 3.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point for acosh: x = {:.4f}\n", x_val );

    dual   x     = x_val;
    auto   func  = []( dual x ) -> dual { return acosh( x ); };
    double deriv = derivative( func, wrt( x ), at( x ) );

    // Analytical derivative: d/dx[acosh(x)] = 1/√(x² - 1)
    double expected = 1.0 / sqrt( x_val * x_val - 1.0 );

    bool ok = check_result( "d/dx[acosh(x)]", deriv, expected, 1e-9 );
    passed += ok;
    total += 1;
  }

  // ========================================================================
  // TEST POWER FUNCTIONS (power2 to power8, rpower2 to rpower8)
  // ========================================================================
  print_subheader( "POWER FUNCTIONS" );

  for ( double x_val : { 0.5, 1.0, 1.5, 2.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // Test power2
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return power2( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 2.0 * x_val;
      bool   ok       = check_result( "d/dx[x²] via power2", deriv, expected, 1e-9 );
      passed += ok;
      total += 1;
    }

    // Test power3
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return power3( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 3.0 * x_val * x_val;
      bool   ok       = check_result( "d/dx[x³] via power3", deriv, expected, 1e-9 );
      passed += ok;
      total += 1;
    }

    // Test power4
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return power4( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = 4.0 * pow( x_val, 3.0 );
      bool   ok       = check_result( "d/dx[x⁴] via power4", deriv, expected, 1e-9 );
      passed += ok;
      total += 1;
    }

    // Test rpower2
    if ( abs( x_val ) > 1e-6 )
    {
      dual   x        = x_val;
      auto   func     = []( dual x ) -> dual { return rpower2( x ); };
      double deriv    = derivative( func, wrt( x ), at( x ) );
      double expected = -2.0 / ( x_val * x_val * x_val );  // d/dx[1/x²] = -2/x³
      bool   ok       = check_result( "d/dx[1/x²] via rpower2", deriv, expected, 1e-9 );
      passed += ok;
      total += 1;
    }
  }

  // Print summary
  fmt::print( "\n" );
  fmt::print( "{}\n", string( 80, '-' ) );
  if ( passed == total )
  {
    fmt::print(
      fg( fmt::color::green ) | fmt::emphasis::bold,
      "Utils Functions: {}/{} tests passed ✓\n",
      passed,
      total );
  }
  else
  {
    fmt::print(
      fg( fmt::color::orange ) | fmt::emphasis::bold,
      "Utils Functions: {}/{} tests passed\n",
      passed,
      total );
  }
  fmt::print( "{}\n", string( 80, '-' ) );
}

// ============================================================================
// SECTION: Complex Combinations and Conditional Functions
// ============================================================================

void test_complex_combinations()
{
  using namespace TestUtils;

  print_header( "COMPLEX COMBINATIONS AND CONDITIONAL FUNCTIONS" );

  int passed = 0, total = 0;

  // ========================================================================
  // TEST 1: Complex combination of multiple functions
  // ========================================================================
  print_subheader( "Complex Combination 1: sin(exp(x)) * cos(log1p(x))" );

  for ( double x_val : { 0.1, 0.5, 1.0, 2.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    auto complex_func = []( dual x ) -> dual { return sin( exp( x ) ) * cos( log1p( x ) ); };

    dual   x        = x_val;
    double deriv_ad = derivative( complex_func, wrt( x ), at( x ) );

    // Finite difference for verification
    double h        = 1e-7;
    double f_plus   = sin( exp( x_val + h ) ) * cos( log1p( x_val + h ) );
    double f_minus  = sin( exp( x_val - h ) ) * cos( log1p( x_val - h ) );
    double deriv_fd = ( f_plus - f_minus ) / ( 2 * h );

    bool ok = check_result( "Complex combo 1", deriv_ad, deriv_fd, 1e-6 );
    passed += ok;
    total += 1;
  }

  // ========================================================================
  // TEST 2: Nested power functions with hyperbolic functions
  // ========================================================================
  print_subheader( "Complex Combination 2: tanh(x) * power3(sin(x)) + asinh(x)" );

  for ( double x_val : { 0.2, 0.5, 0.8 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    auto complex_func = []( dual x ) -> dual { return tanh( x ) * power3( sin( x ) ) + asinh( x ); };

    dual   x        = x_val;
    double deriv_ad = derivative( complex_func, wrt( x ), at( x ) );

    // Finite difference for verification
    double h        = 1e-7;
    double f_plus   = tanh( x_val + h ) * pow( sin( x_val + h ), 3 ) + asinh( x_val + h );
    double f_minus  = tanh( x_val - h ) * pow( sin( x_val - h ), 3 ) + asinh( x_val - h );
    double deriv_fd = ( f_plus - f_minus ) / ( 2 * h );

    bool ok = check_result( "Complex combo 2", deriv_ad, deriv_fd, 1e-6 );
    passed += ok;
    total += 1;
  }

  // ========================================================================
  // TEST 3: Function with conditional (if) - Piecewise function
  // ========================================================================
  print_subheader( "Conditional Function: Piecewise (ReLU-like)" );

  // Define a piecewise function: f(x) = { x² if x < 1, exp(x) if x >= 1 }
  auto piecewise_func = []( dual x ) -> dual
  {
    // Note: This is a test - in real autodiff, branching can cause issues
    // We'll use a smooth approximation instead
    dual threshold = 1.0;

    // Smooth approximation of step function
    dual k      = 50.0;  // Smoothing factor
    dual weight = 1.0 / ( 1.0 + exp( -k * ( x - threshold ) ) );

    // Piecewise components
    dual part1 = power2( x );  // x² for x < 1
    dual part2 = exp( x );     // exp(x) for x >= 1

    // Smooth blend
    return ( 1.0 - weight ) * part1 + weight * part2;
  };

  for ( double x_val : { 0.0, 0.5, 1.0, 1.5, 2.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    dual   x        = x_val;
    double deriv_ad = derivative( piecewise_func, wrt( x ), at( x ) );

    // Finite difference for verification
    double h = 1e-7;
    // Re-evaluate function at shifted points
    double x_plus      = x_val + h;
    double weight_plus = 1.0 / ( 1.0 + exp( -50.0 * ( x_plus - 1.0 ) ) );
    double f_plus      = ( 1.0 - weight_plus ) * ( x_plus * x_plus ) + weight_plus * exp( x_plus );

    double x_minus      = x_val - h;
    double weight_minus = 1.0 / ( 1.0 + exp( -50.0 * ( x_minus - 1.0 ) ) );
    double f_minus      = ( 1.0 - weight_minus ) * ( x_minus * x_minus ) + weight_minus * exp( x_minus );

    double deriv_fd = ( f_plus - f_minus ) / ( 2 * h );

    bool ok = check_result( "Piecewise function (smooth)", deriv_ad, deriv_fd, 1e-5 );
    passed += ok;
    total += 1;
  }

  // ========================================================================
  // TEST 4: Complex multi-variable conditional
  // ========================================================================
  print_subheader( "Multi-variable Conditional Function" );

  auto multi_conditional = []( dual x, dual y ) -> dual
  {
    // Smooth conditional: f(x,y) = if x > y then sin(x)*cos(y) else cos(x)*sin(y)
    dual k      = 50.0;
    dual weight = 1.0 / ( 1.0 + exp( -k * ( x - y ) ) );

    dual part1 = sin( x ) * cos( y );  // when x > y
    dual part2 = cos( x ) * sin( y );  // when x <= y

    return weight * part1 + ( 1.0 - weight ) * part2;
  };

  vector<array<double, 2>> test_points = { { 0.5, 1.0 }, { 1.0, 0.5 }, { 1.0, 1.0 }, { 1.5, 0.8 }, { 0.8, 1.5 } };

  for ( const auto & point : test_points )
  {
    double x_val = point[0], y_val = point[1];
    fmt::print( fg( fmt::color::yellow ), "\nTest point: (x,y) = ({:.4f}, {:.4f})\n", x_val, y_val );

    dual x = x_val, y = y_val;

    // Test derivative with respect to x
    double deriv_x_ad = derivative( multi_conditional, wrt( x ), at( x, y ) );

    // Finite difference for ∂f/∂x
    double h      = 1e-7;
    double f_plus = [&]( double x, double y )
    {
      dual xd = x, yd = y;
      return multi_conditional( xd, yd ).val;
    }( x_val + h, y_val );

    double f_minus = [&]( double x, double y )
    {
      dual xd = x, yd = y;
      return multi_conditional( xd, yd ).val;
    }( x_val - h, y_val );

    double deriv_x_fd = ( f_plus - f_minus ) / ( 2 * h );

    bool ok_x = check_result( "∂f/∂x for multi-conditional", deriv_x_ad, deriv_x_fd, 1e-5 );

    // Test derivative with respect to y
    double deriv_y_ad = derivative( multi_conditional, wrt( y ), at( x, y ) );

    f_plus = [&]( double x, double y )
    {
      dual xd = x, yd = y;
      return multi_conditional( xd, yd ).val;
    }( x_val, y_val + h );

    f_minus = [&]( double x, double y )
    {
      dual xd = x, yd = y;
      return multi_conditional( xd, yd ).val;
    }( x_val, y_val - h );

    double deriv_y_fd = ( f_plus - f_minus ) / ( 2 * h );

    bool ok_y = check_result( "∂f/∂y for multi-conditional", deriv_y_ad, deriv_y_fd, 1e-5 );

    passed += ok_x + ok_y;
    total += 2;
  }

  // ========================================================================
  // TEST 5: Higher-order derivatives of complex combinations
  // ========================================================================
  print_subheader( "Higher-order Derivatives of Complex Functions" );

  for ( double x_val : { 0.5, 1.0 } )
  {
    fmt::print( fg( fmt::color::yellow ), "\nTest point: x = {:.4f}\n", x_val );

    // Complex function: f(x) = exp(sin(x)) * cbrt(1 + x²)
    auto complex_func_higher = []( dual2nd x ) -> dual2nd { return exp( sin( x ) ) * cbrt( 1.0 + power2( x ) ); };

    dual2nd x      = x_val;
    auto    result = derivatives( complex_func_higher, wrt( x, x ), at( x ) );
    double  deriv1 = result[1];  // First derivative
    double  deriv2 = result[2];  // Second derivative

    // Finite difference for first derivative
    double h        = 1e-6;
    auto   func_val = []( double x ) -> double { return exp( sin( x ) ) * pow( 1.0 + x * x, 1.0 / 3.0 ); };

    double f_plus    = func_val( x_val + h );
    double f_minus   = func_val( x_val - h );
    double deriv1_fd = ( f_plus - f_minus ) / ( 2 * h );

    // Finite difference for second derivative
    double f0        = func_val( x_val );
    double deriv2_fd = ( f_plus - 2 * f0 + f_minus ) / ( h * h );

    bool ok1 = check_result( "f'(x) for complex higher-order", deriv1, deriv1_fd, 1e-4 );
    bool ok2 = check_result( "f''(x) for complex higher-order", deriv2, deriv2_fd, 1e-3 );

    passed += ok1 + ok2;
    total += 2;
  }

  // ========================================================================
  // TEST 6: Function with multiple conditionals (smooth max/min)
  // ========================================================================
  print_subheader( "Smooth Maximum and Minimum Functions" );

  // Smooth maximum: log(exp(k*x) + exp(k*y))/k
  auto smooth_max = []( dual x, dual y ) -> dual
  {
    dual k = 10.0;
    return log( exp( k * x ) + exp( k * y ) ) / k;
  };

  for ( const auto & point : vector<array<double, 2>>{ { 0.0, 1.0 }, { 1.0, 0.5 }, { 2.0, 2.0 } } )
  {
    double x_val = point[0], y_val = point[1];
    fmt::print( fg( fmt::color::yellow ), "\nTest point: (x,y) = ({:.4f}, {:.4f})\n", x_val, y_val );

    dual x = x_val, y = y_val;

    // Derivative with respect to x
    double deriv_x_ad = derivative( smooth_max, wrt( x ), at( x, y ) );

    // Analytical derivative: ∂/∂x smooth_max = exp(k*x) / (exp(k*x) + exp(k*y))
    double k                  = 10.0;
    double deriv_x_analytical = exp( k * x_val ) / ( exp( k * x_val ) + exp( k * y_val ) );

    bool ok_x = check_result( "∂/∂x smooth_max(x,y)", deriv_x_ad, deriv_x_analytical, 1e-6 );

    // Derivative with respect to y
    double deriv_y_ad         = derivative( smooth_max, wrt( y ), at( x, y ) );
    double deriv_y_analytical = exp( k * y_val ) / ( exp( k * x_val ) + exp( k * y_val ) );

    bool ok_y = check_result( "∂/∂y smooth_max(x,y)", deriv_y_ad, deriv_y_analytical, 1e-6 );

    passed += ok_x + ok_y;
    total += 2;
  }

  // Print summary
  fmt::print( "\n" );
  fmt::print( "{}\n", string( 80, '-' ) );
  if ( passed == total )
  {
    fmt::print(
      fg( fmt::color::green ) | fmt::emphasis::bold,
      "Complex Combinations: {}/{} tests passed ✓\n",
      passed,
      total );
  }
  else
  {
    fmt::print(
      fg( fmt::color::orange ) | fmt::emphasis::bold,
      "Complex Combinations: {}/{} tests passed\n",
      passed,
      total );
  }
  fmt::print( "{}\n", string( 80, '-' ) );
}

// ============================================================================
// SECTION: Test Macros for Derivatives (2 to 6 arguments)
// ============================================================================

// Instead of using macros inside structs, we'll create standalone functions
namespace MacroTestFunctions
{

  // ========================================================================
  // 2 ARGUMENTS
  // ========================================================================

  double test_func2( double x, double y )
  {
    return x * x * y + sin( x ) * cos( y );
  }

  // Analytical derivatives for test_func2
  inline double test_func2_D_1_analytic( double x, double y )
  {
    return 2 * x * y + cos( x ) * cos( y );
  }

  inline double test_func2_D_2_analytic( double x, double y )
  {
    return x * x - sin( x ) * sin( y );
  }

  inline double test_func2_D_1_1_analytic( double x, double y )
  {
    return 2 * y - sin( x ) * cos( y );
  }

  inline double test_func2_D_1_2_analytic( double x, double y )
  {
    return 2 * x - cos( x ) * sin( y );
  }

  inline double test_func2_D_2_2_analytic( double x, double y )
  {
    return -sin( x ) * cos( y );
  }

  // ========================================================================
  // 3 ARGUMENTS
  // ========================================================================

  double test_func3( double x, double y, double z )
  {
    return x * y * z + sin( x ) * cos( y ) * exp( z );
  }

  // Analytical derivatives for test_func3
  inline double test_func3_D_1_analytic( double x, double y, double z )
  {
    return y * z + cos( x ) * cos( y ) * exp( z );
  }

  inline double test_func3_D_2_analytic( double x, double y, double z )
  {
    return x * z - sin( x ) * sin( y ) * exp( z );
  }

  inline double test_func3_D_3_analytic( double x, double y, double z )
  {
    return x * y + sin( x ) * cos( y ) * exp( z );
  }

  inline double test_func3_D_1_1_analytic( double x, double y, double z )
  {
    return -sin( x ) * cos( y ) * exp( z );
  }

  inline double test_func3_D_1_2_analytic( double x, double y, double z )
  {
    return z - cos( x ) * sin( y ) * exp( z );
  }

  inline double test_func3_D_1_3_analytic( double x, double y, double z )
  {
    return y + cos( x ) * cos( y ) * exp( z );
  }

  inline double test_func3_D_2_2_analytic( double x, double y, double z )
  {
    return -sin( x ) * cos( y ) * exp( z );
  }

  inline double test_func3_D_2_3_analytic( double x, double y, double z )
  {
    return x - sin( x ) * sin( y ) * exp( z );
  }

  inline double test_func3_D_3_3_analytic( double x, double y, double z )
  {
    return sin( x ) * cos( y ) * exp( z );
  }

  // ========================================================================
  // 4 ARGUMENTS
  // ========================================================================

  double test_func4( double x, double y, double z, double w )
  {
    return x * y + z * w + sin( x * z ) * cos( y * w );
  }

  // ========================================================================
  // 5 ARGUMENTS
  // ========================================================================

  double test_func5( double x1, double x2, double x3, double x4, double x5 )
  {
    return x1 * x2 + x3 * x4 * x5 + sin( x1 * x3 ) * cos( x2 * x4 ) * exp( x5 );
  }

  // ========================================================================
  // 6 ARGUMENTS
  // ========================================================================

  double test_func6( double x1, double x2, double x3, double x4, double x5, double x6 )
  {
    return x1 * x2 * x3 + x4 * x5 * x6 + sin( x1 * x4 ) * cos( x2 * x5 ) * exp( x3 * x6 );
  }

}  // namespace MacroTestFunctions

// Create autodiff versions of the functions
namespace MacroTestAutodiff
{

  // ========================================================================
  // 2 ARGUMENTS
  // ========================================================================

  dual test_func2_ad( dual x, dual y )
  {
    return x * x * y + sin( x ) * cos( y );
  }

  dual2nd test_func2_ad2( dual2nd x, dual2nd y )
  {
    return x * x * y + sin( x ) * cos( y );
  }

  // ========================================================================
  // 3 ARGUMENTS
  // ========================================================================

  dual test_func3_ad( dual x, dual y, dual z )
  {
    return x * y * z + sin( x ) * cos( y ) * exp( z );
  }

  dual2nd test_func3_ad2( dual2nd x, dual2nd y, dual2nd z )
  {
    return x * y * z + sin( x ) * cos( y ) * exp( z );
  }

  // ========================================================================
  // 4 ARGUMENTS
  // ========================================================================

  dual test_func4_ad( dual x, dual y, dual z, dual w )
  {
    return x * y + z * w + sin( x * z ) * cos( y * w );
  }

  // ========================================================================
  // 5 ARGUMENTS
  // ========================================================================

  dual test_func5_ad( dual x1, dual x2, dual x3, dual x4, dual x5 )
  {
    return x1 * x2 + x3 * x4 * x5 + sin( x1 * x3 ) * cos( x2 * x4 ) * exp( x5 );
  }

  // ========================================================================
  // 6 ARGUMENTS
  // ========================================================================

  dual test_func6_ad( dual x1, dual x2, dual x3, dual x4, dual x5, dual x6 )
  {
    return x1 * x2 * x3 + x4 * x5 * x6 + sin( x1 * x4 ) * cos( x2 * x5 ) * exp( x3 * x6 );
  }

}  // namespace MacroTestAutodiff

void test_macro_derivatives()
{
  using namespace TestUtils;

  print_header( "DERIVATIVE TESTS (2-6 ARGUMENTS)" );

  int passed = 0, total = 0;
  double h = 1e-7; // for finite differences

  // ========================================================================
  // 2 ARGUMENTS - COMPLETE TESTS
  // ========================================================================
  print_subheader( "2 ARGUMENTS TESTS" );

  vector<array<double, 2>> points2 = { { 0.5, 0.3 }, { 1.0, 0.5 }, { 2.0, 1.0 }, { M_PI / 4, M_PI / 3 } };

  for ( const auto & point : points2 )
  {
    double x = point[0], y = point[1];
    fmt::print( fg( fmt::color::yellow ), "\nTest point: (x,y) = ({:.4f}, {:.4f})\n", x, y );

    // Test first derivatives using autodiff
    {
      dual x_ad = x, y_ad = y;
      double deriv_ad = derivative( MacroTestAutodiff::test_func2_ad, wrt( x_ad ), at( x_ad, y_ad ) );
      double deriv_anal  = MacroTestFunctions::test_func2_D_1_analytic( x, y );
      bool   ok          = check_result( "∂f/∂x (2 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual x_ad = x, y_ad = y;
      double deriv_ad = derivative( MacroTestAutodiff::test_func2_ad, wrt( y_ad ), at( x_ad, y_ad ) );
      double deriv_anal  = MacroTestFunctions::test_func2_D_2_analytic( x, y );
      bool   ok          = check_result( "∂f/∂y (2 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    // Test second derivatives using dual2nd
    {
      dual2nd x_ad = x, y_ad = y;
      auto result = derivatives( MacroTestAutodiff::test_func2_ad2, wrt( x_ad, x_ad ), at( x_ad, y_ad ) );
      double deriv_ad = result[2];  // Second derivative
      double deriv_anal  = MacroTestFunctions::test_func2_D_1_1_analytic( x, y );
      bool   ok          = check_result( "∂²f/∂x² (2 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual2nd x_ad = x, y_ad = y;
      auto result = derivatives( MacroTestAutodiff::test_func2_ad2, wrt( x_ad, y_ad ), at( x_ad, y_ad ) );
      double deriv_ad = result[2];  // Mixed second derivative
      double deriv_anal  = MacroTestFunctions::test_func2_D_1_2_analytic( x, y );
      bool   ok          = check_result( "∂²f/∂x∂y (2 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual2nd x_ad = x, y_ad = y;
      auto result = derivatives( MacroTestAutodiff::test_func2_ad2, wrt( y_ad, y_ad ), at( x_ad, y_ad ) );
      double deriv_ad = result[2];  // Second derivative
      double deriv_anal  = MacroTestFunctions::test_func2_D_2_2_analytic( x, y );
      bool   ok          = check_result( "∂²f/∂y² (2 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // 3 ARGUMENTS - COMPLETE TESTS
  // ========================================================================
  print_subheader( "3 ARGUMENTS TESTS" );

  vector<array<double, 3>> points3 = {
    { 0.5, 0.3, 0.2 }, { 1.0, 0.5, 0.3 }, { 2.0, 1.0, 0.5 }, { M_PI / 4, M_PI / 3, 0.5 }
  };

  for ( const auto & point : points3 )
  {
    double x = point[0], y = point[1], z = point[2];
    fmt::print( fg( fmt::color::yellow ), "\nTest point: (x,y,z) = ({:.4f}, {:.4f}, {:.4f})\n", x, y, z );

    // First derivatives
    {
      dual x_ad = x, y_ad = y, z_ad = z;
      double deriv_ad = derivative( MacroTestAutodiff::test_func3_ad, wrt( x_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_anal  = MacroTestFunctions::test_func3_D_1_analytic( x, y, z );
      bool   ok          = check_result( "∂f/∂x (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual x_ad = x, y_ad = y, z_ad = z;
      double deriv_ad = derivative( MacroTestAutodiff::test_func3_ad, wrt( y_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_anal  = MacroTestFunctions::test_func3_D_2_analytic( x, y, z );
      bool   ok          = check_result( "∂f/∂y (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual x_ad = x, y_ad = y, z_ad = z;
      double deriv_ad = derivative( MacroTestAutodiff::test_func3_ad, wrt( z_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_anal  = MacroTestFunctions::test_func3_D_3_analytic( x, y, z );
      bool   ok          = check_result( "∂f/∂z (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    // Second derivatives
    {
      dual2nd x_ad = x, y_ad = y, z_ad = z;
      auto result = derivatives( MacroTestAutodiff::test_func3_ad2, wrt( x_ad, x_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_ad = result[2];
      double deriv_anal  = MacroTestFunctions::test_func3_D_1_1_analytic( x, y, z );
      bool   ok          = check_result( "∂²f/∂x² (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual2nd x_ad = x, y_ad = y, z_ad = z;
      auto result = derivatives( MacroTestAutodiff::test_func3_ad2, wrt( x_ad, y_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_ad = result[2];
      double deriv_anal  = MacroTestFunctions::test_func3_D_1_2_analytic( x, y, z );
      bool   ok          = check_result( "∂²f/∂x∂y (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual2nd x_ad = x, y_ad = y, z_ad = z;
      auto result = derivatives( MacroTestAutodiff::test_func3_ad2, wrt( x_ad, z_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_ad = result[2];
      double deriv_anal  = MacroTestFunctions::test_func3_D_1_3_analytic( x, y, z );
      bool   ok          = check_result( "∂²f/∂x∂z (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual2nd x_ad = x, y_ad = y, z_ad = z;
      auto result = derivatives( MacroTestAutodiff::test_func3_ad2, wrt( y_ad, y_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_ad = result[2];
      double deriv_anal  = MacroTestFunctions::test_func3_D_2_2_analytic( x, y, z );
      bool   ok          = check_result( "∂²f/∂y² (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual2nd x_ad = x, y_ad = y, z_ad = z;
      auto result = derivatives( MacroTestAutodiff::test_func3_ad2, wrt( y_ad, z_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_ad = result[2];
      double deriv_anal  = MacroTestFunctions::test_func3_D_2_3_analytic( x, y, z );
      bool   ok          = check_result( "∂²f/∂y∂z (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }

    {
      dual2nd x_ad = x, y_ad = y, z_ad = z;
      auto result = derivatives( MacroTestAutodiff::test_func3_ad2, wrt( z_ad, z_ad ), at( x_ad, y_ad, z_ad ) );
      double deriv_ad = result[2];
      double deriv_anal  = MacroTestFunctions::test_func3_D_3_3_analytic( x, y, z );
      bool   ok          = check_result( "∂²f/∂z² (3 args)", deriv_ad, deriv_anal, 1e-9 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // 4 ARGUMENTS - COMPLETE TESTS WITH FINITE DIFFERENCES
  // ========================================================================
  print_subheader( "4 ARGUMENTS TESTS (finite difference verification)" );

  vector<array<double, 4>> points4 = {
    { 0.5, 0.3, 0.2, 0.1 },
    { 1.0, 0.5, 0.3, 0.2 },
  };

  for ( const auto & point : points4 )
  {
    double x = point[0], y = point[1], z = point[2], w = point[3];
    fmt::print( fg( fmt::color::yellow ), "\nTest point: (x,y,z,w) = ({:.4f}, {:.4f}, {:.4f}, {:.4f})\n", x, y, z, w );

    // Test first derivatives with autodiff
    {
      dual x_ad = x, y_ad = y, z_ad = z, w_ad = w;
      double deriv_ad = derivative( MacroTestAutodiff::test_func4_ad, wrt( x_ad ), at( x_ad, y_ad, z_ad, w_ad ) );
      // Finite difference
      double f_plus = MacroTestFunctions::test_func4( x + h, y, z, w );
      double f_minus = MacroTestFunctions::test_func4( x - h, y, z, w );
      double deriv_fd = ( f_plus - f_minus ) / ( 2 * h );
      bool ok = check_result( "∂f/∂x (4 args)", deriv_ad, deriv_fd, 1e-6 );
      passed += ok;
      total += 1;
    }

    {
      dual x_ad = x, y_ad = y, z_ad = z, w_ad = w;
      double deriv_ad = derivative( MacroTestAutodiff::test_func4_ad, wrt( y_ad ), at( x_ad, y_ad, z_ad, w_ad ) );
      double f_plus = MacroTestFunctions::test_func4( x, y + h, z, w );
      double f_minus = MacroTestFunctions::test_func4( x, y - h, z, w );
      double deriv_fd = ( f_plus - f_minus ) / ( 2 * h );
      bool ok = check_result( "∂f/∂y (4 args)", deriv_ad, deriv_fd, 1e-6 );
      passed += ok;
      total += 1;
    }

    {
      dual x_ad = x, y_ad = y, z_ad = z, w_ad = w;
      double deriv_ad = derivative( MacroTestAutodiff::test_func4_ad, wrt( z_ad ), at( x_ad, y_ad, z_ad, w_ad ) );
      double f_plus = MacroTestFunctions::test_func4( x, y, z + h, w );
      double f_minus = MacroTestFunctions::test_func4( x, y, z - h, w );
      double deriv_fd = ( f_plus - f_minus ) / ( 2 * h );
      bool ok = check_result( "∂f/∂z (4 args)", deriv_ad, deriv_fd, 1e-6 );
      passed += ok;
      total += 1;
    }

    {
      dual x_ad = x, y_ad = y, z_ad = z, w_ad = w;
      double deriv_ad = derivative( MacroTestAutodiff::test_func4_ad, wrt( w_ad ), at( x_ad, y_ad, z_ad, w_ad ) );
      double f_plus = MacroTestFunctions::test_func4( x, y, z, w + h );
      double f_minus = MacroTestFunctions::test_func4( x, y, z, w - h );
      double deriv_fd = ( f_plus - f_minus ) / ( 2 * h );
      bool ok = check_result( "∂f/∂w (4 args)", deriv_ad, deriv_fd, 1e-6 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // 5 ARGUMENTS - COMPLETE TESTS WITH FINITE DIFFERENCES
  // ========================================================================
  print_subheader( "5 ARGUMENTS TESTS (finite difference verification)" );

  vector<array<double, 5>> points5 = {
    { 0.5, 0.3, 0.2, 0.1, 0.4 },
  };

  for ( const auto & point : points5 )
  {
    double x1 = point[0], x2 = point[1], x3 = point[2], x4 = point[3], x5 = point[4];
    fmt::print(
      fg( fmt::color::yellow ),
      "\nTest point: (x1..x5) = ({:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f})\n",
      x1, x2, x3, x4, x5 );

    // Test all first derivatives (5)
    vector<tuple<string, int>> first_derivs = {
      { "∂f/∂x1 (5 args)", 1 },
      { "∂f/∂x2 (5 args)", 2 },
      { "∂f/∂x3 (5 args)", 3 },
      { "∂f/∂x4 (5 args)", 4 },
      { "∂f/∂x5 (5 args)", 5 },
    };

    for ( const auto& t : first_derivs )
    {
      string name = std::get<0>(t);
      int    idx  = std::get<1>(t);
      dual x1_ad = x1, x2_ad = x2, x3_ad = x3, x4_ad = x4, x5_ad = x5;
      double deriv_ad = 0.0;
      
      if (idx == 1) deriv_ad = derivative( MacroTestAutodiff::test_func5_ad, wrt( x1_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad ) );
      else if (idx == 2) deriv_ad = derivative( MacroTestAutodiff::test_func5_ad, wrt( x2_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad ) );
      else if (idx == 3) deriv_ad = derivative( MacroTestAutodiff::test_func5_ad, wrt( x3_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad ) );
      else if (idx == 4) deriv_ad = derivative( MacroTestAutodiff::test_func5_ad, wrt( x4_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad ) );
      else if (idx == 5) deriv_ad = derivative( MacroTestAutodiff::test_func5_ad, wrt( x5_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad ) );
      
      // Finite difference
      auto shift = [&](double h) -> double {
        double p1 = x1, p2 = x2, p3 = x3, p4 = x4, p5 = x5;
        if (idx == 1) p1 += h; else if (idx == 2) p2 += h; else if (idx == 3) p3 += h; 
        else if (idx == 4) p4 += h; else if (idx == 5) p5 += h;
        return MacroTestFunctions::test_func5(p1, p2, p3, p4, p5);
      };
      double deriv_fd = (shift(h) - shift(-h)) / (2 * h);
      bool ok = check_result( name, deriv_ad, deriv_fd, 1e-6 );
      passed += ok;
      total += 1;
    }
  }

  // ========================================================================
  // 6 ARGUMENTS - COMPLETE TESTS WITH FINITE DIFFERENCES
  // ========================================================================
  print_subheader( "6 ARGUMENTS TESTS (finite difference verification)" );

  vector<array<double, 6>> points6 = {
    { 0.5, 0.3, 0.2, 0.1, 0.4, 0.6 },
  };

  for ( const auto & point : points6 )
  {
    double x1 = point[0], x2 = point[1], x3 = point[2], x4 = point[3], x5 = point[4], x6 = point[5];
    fmt::print(
      fg( fmt::color::yellow ),
      "\nTest point: (x1..x6) = ({:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f})\n",
      x1, x2, x3, x4, x5, x6 );

    // Test all first derivatives (6)
    vector<tuple<string, int>> first_derivs = {
      { "∂f/∂x1 (6 args)", 1 },
      { "∂f/∂x2 (6 args)", 2 },
      { "∂f/∂x3 (6 args)", 3 },
      { "∂f/∂x4 (6 args)", 4 },
      { "∂f/∂x5 (6 args)", 5 },
      { "∂f/∂x6 (6 args)", 6 },
    };

    for ( const auto& t : first_derivs )
    {
      string name = std::get<0>(t);
      int    idx  = std::get<1>(t);

      dual x1_ad = x1, x2_ad = x2, x3_ad = x3, x4_ad = x4, x5_ad = x5, x6_ad = x6;
      double deriv_ad = 0.0;
      
      if (idx == 1) deriv_ad = derivative( MacroTestAutodiff::test_func6_ad, wrt( x1_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad, x6_ad ) );
      else if (idx == 2) deriv_ad = derivative( MacroTestAutodiff::test_func6_ad, wrt( x2_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad, x6_ad ) );
      else if (idx == 3) deriv_ad = derivative( MacroTestAutodiff::test_func6_ad, wrt( x3_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad, x6_ad ) );
      else if (idx == 4) deriv_ad = derivative( MacroTestAutodiff::test_func6_ad, wrt( x4_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad, x6_ad ) );
      else if (idx == 5) deriv_ad = derivative( MacroTestAutodiff::test_func6_ad, wrt( x5_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad, x6_ad ) );
      else if (idx == 6) deriv_ad = derivative( MacroTestAutodiff::test_func6_ad, wrt( x6_ad ), at( x1_ad, x2_ad, x3_ad, x4_ad, x5_ad, x6_ad ) );
      
      // Finite difference
      auto shift = [&](double h) -> double {
        double p1 = x1, p2 = x2, p3 = x3, p4 = x4, p5 = x5, p6 = x6;
        if (idx == 1) p1 += h; else if (idx == 2) p2 += h; else if (idx == 3) p3 += h; 
        else if (idx == 4) p4 += h; else if (idx == 5) p5 += h; else if (idx == 6) p6 += h;
        return MacroTestFunctions::test_func6(p1, p2, p3, p4, p5, p6);
      };
      double deriv_fd = (shift(h) - shift(-h)) / (2 * h);
      bool ok = check_result( name, deriv_ad, deriv_fd, 1e-6 );
      passed += ok;
      total += 1;
    }
  }

  // Print summary
  fmt::print( "\n" );
  fmt::print( "{}\n", string( 80, '-' ) );
  if ( passed == total )
  {
    fmt::print(
      fg( fmt::color::green ) | fmt::emphasis::bold,
      "Derivatives (2-6 args): {}/{} tests passed ✓\n",
      passed,
      total );
  }
  else
  {
    fmt::print(
      fg( fmt::color::orange ) | fmt::emphasis::bold,
      "Derivatives (2-6 args): {}/{} tests passed\n",
      passed,
      total );
  }
  fmt::print( "{}\n", string( 80, '-' ) );
}
// ============================================================================
// SECTION: Main Function
// ============================================================================

int main()
{
  using namespace TestUtils;

  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n================================ AUTODIFF LIBRARY TEST SUITE ================================\n" );
  fmt::print( "Testing autodiff implementation with comprehensive math function coverage...\n" );

  try
  {
    // Run all test suites
    test_all_math_functions();    // Comprehensive math tests
    test_utils_functions();       // Test functions from Utils_autodiff.hh
    test_complex_combinations();  // Test complex combinations and conditionals
    test_macro_derivatives();     // Test derivatives (2-6 args) - using direct autodiff calls
    test_basic_functions();
    test_multi_variable();
    test_higher_order();
    test_edge_cases();
    test_performance();
  }
  catch ( const exception & e )
  {
    fmt::print( stderr, fg( fmt::color::red ) | fmt::emphasis::bold, "\nException during testing: {}\n", e.what() );
    return 1;
  }

  fmt::print( "\n{}\n", string( 80, '=' ) );
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "✓ Test suite completed successfully!\n" );
  fmt::print( "{}\n", string( 80, '=' ) );

  return 0;
}
