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

#include "Utils_autodiff.hh"
#include "Utils_TinyAD.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

constexpr int timing_rounds = 2000;

struct Derivatives
{
  double          value;
  Eigen::Vector2d gradient;
  Eigen::Matrix2d hessian;
};

double objective( double x, double y )
{
  return x * x + 3.0 * x * y + y * y + std::sin( x ) * std::exp( y ) +
         std::log( 2.0 + x * x + y * y ) + std::cos( x * y );
}

Derivatives analytical_derivatives( double x, double y )
{
  const double sine_exp = std::sin( x ) * std::exp( y );
  const double cosine_exp = std::cos( x ) * std::exp( y );
  const double q = 2.0 + x * x + y * y;
  const double q_squared = q * q;
  const double sine_xy = std::sin( x * y );
  const double cosine_xy = std::cos( x * y );
  Derivatives result;
  result.value    = objective( x, y );
  result.gradient << 2.0 * x + 3.0 * y + cosine_exp + 2.0 * x / q - y * sine_xy,
    3.0 * x + 2.0 * y + sine_exp + 2.0 * y / q - x * sine_xy;
  result.hessian << 2.0 - sine_exp + 2.0 / q - 4.0 * x * x / q_squared - y * y * cosine_xy,
    3.0 + cosine_exp - 4.0 * x * y / q_squared - sine_xy - x * y * cosine_xy,
    3.0 + cosine_exp - 4.0 * x * y / q_squared - sine_xy - x * y * cosine_xy,
    2.0 + sine_exp + 2.0 / q - 4.0 * y * y / q_squared - x * x * cosine_xy;
  return result;
}

Derivatives autodiff_derivatives( double x, double y )
{
  const auto function = []( autodiff::dual2nd x_ad, autodiff::dual2nd y_ad ) -> autodiff::dual2nd
  {
    return x_ad * x_ad + 3.0 * x_ad * y_ad + y_ad * y_ad + sin( x_ad ) * exp( y_ad ) +
           log( 2.0 + x_ad * x_ad + y_ad * y_ad ) + cos( x_ad * y_ad );
  };

  autodiff::dual2nd x_xx = x, y_xx = y;
  autodiff::dual2nd x_xy = x, y_xy = y;
  autodiff::dual2nd x_yy = x, y_yy = y;
  const auto dxx = derivatives( function, wrt( x_xx, x_xx ), at( x_xx, y_xx ) );
  const auto dxy = derivatives( function, wrt( x_xy, y_xy ), at( x_xy, y_xy ) );
  const auto dyy = derivatives( function, wrt( y_yy, y_yy ), at( x_yy, y_yy ) );

  Derivatives result;
  result.value    = dxx[0];
  result.gradient << dxx[1], dyy[1];
  result.hessian << dxx[2], dxy[2], dxy[2], dyy[2];
  return result;
}

Derivatives tinyad_derivatives( double x, double y )
{
  static auto function = []
  {
    auto result = TinyAD::scalar_function<2>( TinyAD::range( 1 ) );
    result.template add_elements<1>( TinyAD::range( 1 ), []( auto & element )
    {
      const auto variables = element.variables( 0 );
      using std::exp;
      using std::sin;
      using std::cos;
      using std::log;
      return variables[0] * variables[0] + 3.0 * variables[0] * variables[1] + variables[1] * variables[1] +
             sin( variables[0] ) * exp( variables[1] ) + log( 2.0 + variables[0] * variables[0] + variables[1] * variables[1] ) +
             cos( variables[0] * variables[1] );
    } );
    return result;
  }();

  Eigen::Vector2d point;
  point << x, y;
  const auto [value, gradient, hessian] = function.eval_with_derivatives( point );
  return { value, gradient, hessian.toDense() };
}

// TinyAD can also be used as a direct dual-number type, bypassing the
// ScalarFunction machinery (element handles, type erasure, and sparse-Hessian
// assembly). This is the comparable API to autodiff's dual2nd path.
Derivatives tinyad_scalar_derivatives( double x, double y )
{
  using Scalar = TinyAD::Scalar<2, double>;
  const Scalar x_ad( x, 0 );
  const Scalar y_ad( y, 1 );
  using std::cos;
  using std::exp;
  using std::log;
  using std::sin;
  const Scalar result = x_ad * x_ad + 3.0 * x_ad * y_ad + y_ad * y_ad + sin( x_ad ) * exp( y_ad ) +
                        log( 2.0 + x_ad * x_ad + y_ad * y_ad ) + cos( x_ad * y_ad );
  return { result.val, result.grad, result.Hess };
}

double maximum_error( Derivatives const & actual, Derivatives const & expected )
{
  return std::max(
    { std::abs( actual.value - expected.value ),
      ( actual.gradient - expected.gradient ).cwiseAbs().maxCoeff(),
      ( actual.hessian - expected.hessian ).cwiseAbs().maxCoeff() } );
}

void print_derivatives( char const * name, Derivatives const & derivatives )
{
  std::cout << "  " << std::left << std::setw( 10 ) << name << " value=" << std::setw( 20 ) << derivatives.value
            << " gradient=(" << derivatives.gradient[0] << ", " << derivatives.gradient[1] << ")\n"
            << "             Hessian=[" << derivatives.hessian( 0, 0 ) << ", " << derivatives.hessian( 0, 1 )
            << "; " << derivatives.hessian( 1, 0 ) << ", " << derivatives.hessian( 1, 1 ) << "]\n";
}

template <typename Function>
double elapsed_milliseconds( Function && function, std::vector<Eigen::Vector2d> const & points )
{
  volatile double sink = 0.0;
  volatile double input_x = 0.7;
  volatile double input_y = -0.4;
  const auto start = Clock::now();
  for ( int round = 0; round < timing_rounds; ++round )
  {
    for ( auto const & point : points )
    {
      const auto result = function( input_x + point[0], input_y + point[1] );
      sink += result.value + result.gradient.sum() + result.hessian.sum();
    }
  }
  const auto elapsed = std::chrono::duration<double, std::milli>( Clock::now() - start ).count();
  if ( sink == 0.0 ) std::abort();
  return elapsed;
}

} // namespace

int main()
{
  constexpr double tolerance = 1e-11;
  std::vector<Eigen::Vector2d> points;
  points.reserve( 120 );
  for ( int row = 0; row < 10; ++row )
    for ( int column = 0; column < 12; ++column )
      points.emplace_back( -0.95 + 1.9 * column / 11.0, -0.85 + 1.7 * row / 9.0 );

  std::cout << std::setprecision( 16 )
            << "Autodiff vs TinyAD: accuracy and timing comparison\n"
            << "Objective: f(x,y) = x^2 + 3*x*y + y^2 + sin(x)*exp(y) + log(2+x^2+y^2) + cos(x*y)\n"
            << "Accuracy tolerance: " << tolerance << "\n"
            << "Accuracy grid: " << points.size() << " points (10 x 12).\n"
            << "Build mode: "
#ifdef NDEBUG
            << "Release (NDEBUG, Eigen runtime checks disabled)\n";
#else
            << "Debug\n";
#endif
  std::cout << "TinyAD runtime checks: " << ( TINYAD_ENABLE_RUNTIME_CHECKS ? "enabled" : "disabled" ) << '\n';

  double worst_autodiff_error = 0.0;
  double worst_tinyad_error   = 0.0;
  double worst_tinyad_scalar_error = 0.0;
  double squared_autodiff_error = 0.0;
  double squared_tinyad_error = 0.0;
  double squared_tinyad_scalar_error = 0.0;
  for ( std::size_t i = 0; i < points.size(); ++i )
  {
    const double x = points[i][0];
    const double y = points[i][1];
    const bool show_details = i < 3 || i + 1 == points.size();
    if ( show_details ) std::cout << "\n[Accuracy case " << i + 1 << '/' << points.size() << "] evaluating at (" << x << ", " << y << ")\n";
    if ( show_details ) std::cout << "  Computing analytical reference...\n";
    const Derivatives analytical = analytical_derivatives( x, y );
    if ( show_details ) std::cout << "  Computing derivatives with autodiff (three second-order passes)...\n";
    const Derivatives autodiff = autodiff_derivatives( x, y );
    if ( show_details ) std::cout << "  Computing derivatives with TinyAD (one sparse-Hessian evaluation)...\n";
    const Derivatives tinyad = tinyad_derivatives( x, y );
    if ( show_details ) std::cout << "  Computing derivatives with TinyAD::Scalar (direct dual-number evaluation)...\n";
    const Derivatives tinyad_scalar = tinyad_scalar_derivatives( x, y );
    const double autodiff_error = maximum_error( autodiff, analytical );
    const double tinyad_error = maximum_error( tinyad, analytical );
    const double tinyad_scalar_error = maximum_error( tinyad_scalar, analytical );
    if ( show_details )
    {
      print_derivatives( "analytic", analytical );
      print_derivatives( "autodiff", autodiff );
      print_derivatives( "TinyAD", tinyad );
      print_derivatives( "TinyAD scalar", tinyad_scalar );
      std::cout << "  Maximum absolute error: autodiff=" << autodiff_error << ", TinyAD=" << tinyad_error
                << ", TinyAD::Scalar=" << tinyad_scalar_error << '\n';
    }
    worst_autodiff_error = std::max( worst_autodiff_error, autodiff_error );
    worst_tinyad_error = std::max( worst_tinyad_error, tinyad_error );
    worst_tinyad_scalar_error = std::max( worst_tinyad_scalar_error, tinyad_scalar_error );
    squared_autodiff_error += autodiff_error * autodiff_error;
    squared_tinyad_error += tinyad_error * tinyad_error;
    squared_tinyad_scalar_error += tinyad_scalar_error * tinyad_scalar_error;
  }

  std::cout << "\n[Accuracy summary] worst error: autodiff=" << worst_autodiff_error
            << ", TinyAD=" << worst_tinyad_error << ", TinyAD::Scalar=" << worst_tinyad_scalar_error << '\n';
  if ( worst_autodiff_error > tolerance || worst_tinyad_error > tolerance || worst_tinyad_scalar_error > tolerance )
  {
    std::cerr << "Derivative accuracy check failed.\n";
    return 1;
  }

  std::cout << "\n[Timing] Warming up both implementations before the benchmark...\n";
  (void)analytical_derivatives( points[0][0], points[0][1] );
  (void)autodiff_derivatives( points[0][0], points[0][1] );
  (void)tinyad_derivatives( points[0][0], points[0][1] );
  (void)tinyad_scalar_derivatives( points[0][0], points[0][1] );
  const std::size_t timing_evaluations = static_cast<std::size_t>( timing_rounds ) * points.size();
  std::cout << "[Timing] Running " << timing_rounds << " rounds x " << points.size() << " points = "
            << timing_evaluations << " value/gradient/Hessian evaluations per implementation...\n";
  const double analytical_ms = elapsed_milliseconds( []( double px, double py ) { return analytical_derivatives( px, py ); }, points );
  const double autodiff_ms = elapsed_milliseconds( []( double px, double py ) { return autodiff_derivatives( px, py ); }, points );
  const double tinyad_ms   = elapsed_milliseconds( []( double px, double py ) { return tinyad_derivatives( px, py ); }, points );
  const double tinyad_scalar_ms = elapsed_milliseconds( []( double px, double py ) { return tinyad_scalar_derivatives( px, py ); }, points );

  const double analytical_rms_error = 0.0;
  const double autodiff_rms_error = std::sqrt( squared_autodiff_error / points.size() );
  const double tinyad_rms_error = std::sqrt( squared_tinyad_error / points.size() );
  const double tinyad_scalar_rms_error = std::sqrt( squared_tinyad_scalar_error / points.size() );
  std::cout << std::fixed << std::setprecision( 6 )
            << "\nFinal comparison table\n"
            << "+-------------+-------+----------------------+----------------------+--------------+-----------------+\n"
            << "| Library     | Cases | Maximum abs. error   | RMS abs. error       | Total time   | us/evaluation   |\n"
            << "+-------------+-------+----------------------+----------------------+--------------+-----------------+\n"
            << "| analytic    | " << std::setw( 5 ) << points.size() << " | " << std::scientific << std::setprecision( 3 )
            << std::setw( 20 ) << 0.0 << " | " << std::setw( 20 ) << analytical_rms_error << " | "
            << std::fixed << std::setprecision( 6 ) << std::setw( 10 ) << analytical_ms << " ms | "
            << std::setw( 15 ) << 1e3 * analytical_ms / timing_evaluations << " |\n"
            << "| autodiff    | " << std::setw( 5 ) << points.size() << " | " << std::scientific << std::setprecision( 3 )
            << std::setw( 20 ) << worst_autodiff_error << " | " << std::setw( 20 ) << autodiff_rms_error << " | "
            << std::fixed << std::setprecision( 6 ) << std::setw( 10 ) << autodiff_ms << " ms | "
            << std::setw( 15 ) << 1e3 * autodiff_ms / timing_evaluations << " |\n"
            << "| TinyAD      | " << std::setw( 5 ) << points.size() << " | " << std::scientific << std::setprecision( 3 )
            << std::setw( 20 ) << worst_tinyad_error << " | " << std::setw( 20 ) << tinyad_rms_error << " | "
            << std::fixed << std::setprecision( 6 ) << std::setw( 10 ) << tinyad_ms << " ms | "
            << std::setw( 15 ) << 1e3 * tinyad_ms / timing_evaluations << " |\n"
            << "| TinyAD dual | " << std::setw( 5 ) << points.size() << " | " << std::scientific << std::setprecision( 3 )
            << std::setw( 20 ) << worst_tinyad_scalar_error << " | " << std::setw( 20 ) << tinyad_scalar_rms_error << " | "
            << std::fixed << std::setprecision( 6 ) << std::setw( 10 ) << tinyad_scalar_ms << " ms | "
            << std::setw( 15 ) << 1e3 * tinyad_scalar_ms / timing_evaluations << " |\n"
            << "+-------------+-------+----------------------+----------------------+--------------+-----------------+\n"
            << "Timing ratios: autodiff/analytic=" << autodiff_ms / analytical_ms
            << ", TinyAD/analytic=" << tinyad_ms / analytical_ms
            << ", TinyAD::Scalar/analytic=" << tinyad_scalar_ms / analytical_ms
            << ", TinyAD::Scalar/autodiff=" << tinyad_scalar_ms / autodiff_ms
            << ", TinyAD/Scalar=" << tinyad_ms / tinyad_scalar_ms << "\n\nAll accuracy checks PASSED.\n";
  return 0;
}
