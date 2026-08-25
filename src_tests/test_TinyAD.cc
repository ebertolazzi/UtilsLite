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

#include "Utils.hh"
#include "Utils_TinyAD.hh"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{

  constexpr double tolerance = 1e-12;

  bool close_to( double actual, double expected )
  { return std::abs( actual - expected ) <= tolerance; }

  bool report( std::string const & name, double actual, double expected )
  {
    const double error = std::abs( actual - expected );
    const bool   ok    = close_to( actual, expected );
    std::cout << "  " << std::left << std::setw( 24 ) << name << ( ok ? "[PASS]" : "[FAIL]" )
              << "  actual=" << std::setw( 15 ) << actual << " expected=" << std::setw( 15 ) << expected
              << " abs-error=" << error << '\n';
    return ok;
  }

  void banner( std::string const & title )
  {
    std::cout << "\n--------------------------------------------------------------------------------\n"
              << title << "\n--------------------------------------------------------------------------------\n";
  }

}  // namespace

int main()
{
  std::cout << std::setprecision( 16 );
  std::cout << "TinyAD integration test\nThis test checks value, gradient, and Hessian.\nTolerance: " << tolerance
            << '\n';
  int passed = 0, total = 0;

  banner( "1. Building the quadratic objective" );
  std::cout << "f(x, y) = x^2 + 3*x*y + y^2\n";
  auto quadratic = TinyAD::scalar_function<2>( TinyAD::range( 1 ) );
  quadratic.template add_elements<1>(
    TinyAD::range( 1 ),
    []( auto & element )
    {
      const auto x = element.variables( 0 );
      return x[0] * x[0] + 3.0 * x[0] * x[1] + x[1] * x[1];
    } );
  std::cout << "Objective assembled with 1 element and 2 variables.\n";

  banner( "2. Evaluating quadratic objective at x = (1, 2)" );
  Eigen::Vector2d point;
  point << 1.0, 2.0;
  std::cout << "Requesting value, gradient, and sparse Hessian from TinyAD...\n";
  const auto [value, gradient, hessian] = quadratic.eval_with_derivatives( point );
  std::cout << "TinyAD returned Hessian with " << hessian.nonZeros() << " non-zero coefficients.\n";
  passed += report( "value", value, 11.0 );
  ++total;
  passed += report( "gradient[0]", gradient[0], 8.0 );
  ++total;
  passed += report( "gradient[1]", gradient[1], 7.0 );
  ++total;
  passed += report( "hessian(0,0)", hessian.coeff( 0, 0 ), 2.0 );
  ++total;
  passed += report( "hessian(0,1)", hessian.coeff( 0, 1 ), 3.0 );
  ++total;
  passed += report( "hessian(1,0)", hessian.coeff( 1, 0 ), 3.0 );
  ++total;
  passed += report( "hessian(1,1)", hessian.coeff( 1, 1 ), 2.0 );
  ++total;

  banner( "3. Evaluating nonlinear objective at x = (0.7, -0.4)" );
  std::cout << "g(x, y) = x^2 + 3*x*y + y^2 + sin(x)*exp(y)\n";
  auto nonlinear = TinyAD::scalar_function<2>( TinyAD::range( 1 ) );
  nonlinear.template add_elements<1>(
    TinyAD::range( 1 ),
    []( auto & element )
    {
      const auto x = element.variables( 0 );
      using std::exp;
      using std::sin;
      return x[0] * x[0] + 3.0 * x[0] * x[1] + x[1] * x[1] + sin( x[0] ) * exp( x[1] );
    } );
  point << 0.7, -0.4;
  const auto [nonlinear_value, nonlinear_gradient, nonlinear_hessian] = nonlinear.eval_with_derivatives( point );
  const double sine_exp                                               = std::sin( point[0] ) * std::exp( point[1] );
  const double cosine_exp                                             = std::cos( point[0] ) * std::exp( point[1] );
  passed += report(
    "nonlinear value",
    nonlinear_value,
    point[0] * point[0] + 3.0 * point[0] * point[1] + point[1] * point[1] + sine_exp );
  ++total;
  passed += report( "nonlinear gradient[0]", nonlinear_gradient[0], 2.0 * point[0] + 3.0 * point[1] + cosine_exp );
  ++total;
  passed += report( "nonlinear gradient[1]", nonlinear_gradient[1], 3.0 * point[0] + 2.0 * point[1] + sine_exp );
  ++total;
  passed += report( "nonlinear hessian(0,0)", nonlinear_hessian.coeff( 0, 0 ), 2.0 - sine_exp );
  ++total;
  passed += report( "nonlinear hessian(0,1)", nonlinear_hessian.coeff( 0, 1 ), 3.0 + cosine_exp );
  ++total;
  passed += report( "nonlinear hessian(1,1)", nonlinear_hessian.coeff( 1, 1 ), 2.0 + sine_exp );
  ++total;

  banner( "4. Final result" );
  std::cout << "Checks passed: " << passed << '/' << total << '\n';
  if ( passed != total )
  {
    std::cerr << "TinyAD integration test FAILED.\n";
    return 1;
  }
  std::cout << "TinyAD integration test PASSED.\n";
  return 0;
}
