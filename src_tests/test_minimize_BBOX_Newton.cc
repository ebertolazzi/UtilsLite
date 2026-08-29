// SPDX-License-Identifier: MPL-2.0

#include "Utils_minimize_BBOX_Newton.hh"
#include "Utils_fmt.hh"

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace
{

  using Real           = double;
  using Vector         = Utils::MinimizeNewton::Vector<Real>;
  using ConstVectorRef = Utils::MinimizeNewton::ConstVectorRef<Real>;
  using VectorRef      = Utils::MinimizeNewton::VectorRef<Real>;
  using MatrixRef      = Utils::MinimizeNewton::MatrixRef<Real>;
  using Status         = Utils::MinimizeNewton::Status;

  bool check( bool condition, char const * message )
  {
    if ( !condition ) fmt::print( fmt::fg( fmt::color::red ), "FAILED: {}\n", message );
    return condition;
  }

}  // namespace

int main()
{
  bool passed = true;

  Utils::MinimizeNewton::Options<Real> options;
  options.set_tolerances( 1e-12 );
  options.max_iterations = 400;

  Real                     x0_data[]{ 2, -3 };
  Real                     lower_data[]{ -10, -10 };
  Real                     upper_data[]{ 10, 10 };
  Eigen::Map<Vector const> x0( x0_data, 2 );
  Eigen::Map<Vector const> lower( lower_data, 2 );
  Eigen::Map<Vector const> upper( upper_data, 2 );

  auto quadratic = Utils::MinimizeNewton::make_problem<Real>(
    []( ConstVectorRef x ) { return Real( 0.5 ) * ( x[0] * x[0] + Real( 4 ) * x[1] * x[1] ); },
    []( ConstVectorRef x, VectorRef g )
    {
      g[0] = x[0];
      g[1] = Real( 4 ) * x[1];
    },
    []( ConstVectorRef, MatrixRef H )
    {
      H.setZero();
      H( 0, 0 ) = Real( 1 );
      H( 1, 1 ) = Real( 4 );
    } );

  Utils::MinimizeNewton::Solver<Real> solver( 2, options );
  auto                                result = solver.solve( quadratic, x0, lower, upper );
  passed &= check( result.status == Status::converged, "unconstrained quadratic status" );
  passed &= check( result.x.stableNorm() <= 1e-12, "unconstrained quadratic solution" );
  passed &= check(
    result.projected_gradient_norm <= result.optimality_tolerance,
    "unconstrained quadratic KKT residual" );
  passed &= check( result.step_norm <= result.effective_step_tolerance, "machine-scale final step" );

  // A large constant makes objective changes invisible in double precision;
  // derivative-driven regularized Newton refinement must still proceed.
  auto offset_quadratic = Utils::MinimizeNewton::make_problem<Real>(
    []( ConstVectorRef x )
    { return Real( 1e20 ) + Real( 0.5 ) * ( std::pow( x[0] - Real( 0.25 ), 2 ) + std::pow( x[1] + Real( 0.5 ), 2 ) ); },
    []( ConstVectorRef x, VectorRef g )
    {
      g[0] = x[0] - Real( 0.25 );
      g[1] = x[1] + Real( 0.5 );
    },
    []( ConstVectorRef, MatrixRef H ) { H.setIdentity(); } );
  result = solver.solve( offset_quadratic, x0, lower, upper );
  Vector exact( 2 );
  exact << Real( 0.25 ), Real( -0.5 );
  passed &= check( result.status == Status::converged, "large-offset quadratic status" );
  passed &= check( ( result.x - exact ).stableNorm() <= 1e-12, "large-offset derivative refinement" );

  Vector one_x0( 1 ), one_lower( 1 ), one_upper( 1 );
  one_x0[0]    = 0;
  one_lower[0] = 0;
  one_upper[0] = 1;

  // The constrained minimizer is at the upper bound and has nonzero ordinary
  // gradient, but zero projected KKT residual.
  auto boundary = Utils::MinimizeNewton::make_problem<Real>(
    []( ConstVectorRef x ) { return Real( 0.5 ) * std::pow( x[0] - Real( 2 ), 2 ); },
    []( ConstVectorRef x, VectorRef g ) { g[0] = x[0] - Real( 2 ); },
    []( ConstVectorRef, MatrixRef H ) { H.setConstant( Real( 1 ) ); } );
  Utils::MinimizeNewton::Solver<Real> one_solver( 1, options );
  auto                                one_result = one_solver.solve( boundary, one_x0, one_lower, one_upper );
  passed &= check( one_result.status == Status::converged, "BOX boundary status" );
  passed &= check( one_result.x[0] == Real( 1 ), "BOX projection to upper bound" );
  passed &= check( one_result.projected_gradient_norm == Real( 0 ), "BOX projected KKT residual" );

  // A weakly active point with zero gradient and negative feasible curvature is
  // not a minimum.  The eigensolver-based negative-curvature step must escape.
  auto boundary_maximum = Utils::MinimizeNewton::make_problem<Real>(
    []( ConstVectorRef x ) { return -Real( 0.5 ) * x[0] * x[0]; },
    []( ConstVectorRef x, VectorRef g ) { g[0] = -x[0]; },
    []( ConstVectorRef, MatrixRef H ) { H.setConstant( Real( -1 ) ); } );
  one_result = one_solver.solve( boundary_maximum, one_x0, one_lower, one_upper );
  passed &= check( one_result.status == Status::converged, "weakly active negative-curvature status" );
  passed &= check( one_result.x[0] == Real( 1 ), "weakly active negative-curvature escape" );
  passed &= check( one_result.objective <= Real( -0.5 ), "negative-curvature objective decrease" );

  fmt::print(
    passed ? fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold : fmt::fg( fmt::color::red ) | fmt::emphasis::bold,
    "Header-only Eigen 5 cubic Newton with BOX constraints: {}\n",
    passed ? "PASSED" : "FAILED" );
  return passed ? 0 : 1;
}
