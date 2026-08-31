// SPDX-License-Identifier: MPL-2.0

#include "Utils_minimize_BBOX_NewtonCubic.hh"

#include <cmath>
#include <iostream>
#include <limits>

namespace
{

  template <typename Real> bool test_quadratic()
  {
    using Vector         = Utils::Vector<Real>;
    using ConstVectorRef = Utils::ConstVectorRef<Real>;
    using VectorRef      = Utils::VectorRef<Real>;
    using MatrixRef      = Utils::MatrixRef<Real>;

    Vector x0( 2 ), lower( 2 ), upper( 2 );
    x0 << Real( 2 ), Real( -3 );
    lower.setConstant( Real( -10 ) );
    upper.setConstant( Real( 10 ) );

    auto problem = Utils::make_problem<Real>(
      []( ConstVectorRef x ) { return Real( 0.5 ) * ( x[0] * x[0] + Real( 4 ) * x[1] * x[1] ); },
      []( ConstVectorRef x, VectorRef gradient )
      {
        gradient[0] = x[0];
        gradient[1] = Real( 4 ) * x[1];
      },
      []( ConstVectorRef, MatrixRef hessian )
      {
        hessian.setZero();
        hessian( 0, 0 ) = Real( 1 );
        hessian( 1, 1 ) = Real( 4 );
      } );

    Utils::Options<Real> options;
    options.set_tolerances( Real( 256 ) * std::numeric_limits<Real>::epsilon() );
    Utils::Minimize_BBOX_NewtonCubic<Real> solver( x0.size(), options );
    auto const                             result = solver.solve( problem, x0, lower, upper );
    Real const                             error  = result.x.norm();
    return result.status == Utils::Status::converged && error <= Real( 4096 ) * std::numeric_limits<Real>::epsilon();
  }

  bool test_boundary_solution()
  {
    using Real           = double;
    using Vector         = Utils::Vector<Real>;
    using ConstVectorRef = Utils::ConstVectorRef<Real>;
    using VectorRef      = Utils::VectorRef<Real>;
    using MatrixRef      = Utils::MatrixRef<Real>;

    Vector x0( 1 ), lower( 1 ), upper( 1 );
    x0[0]    = 0;
    lower[0] = 0;
    upper[0] = 1;

    auto problem = Utils::make_problem<Real>(
      []( ConstVectorRef x ) { return Real( 0.5 ) * std::pow( x[0] - Real( 2 ), 2 ); },
      []( ConstVectorRef x, VectorRef gradient ) { gradient[0] = x[0] - Real( 2 ); },
      []( ConstVectorRef, MatrixRef hessian ) { hessian.setConstant( Real( 1 ) ); } );

    Utils::Minimize_BBOX_NewtonCubic<Real> solver( 1 );
    auto const                             upper_result = solver.solve( problem, x0, lower, upper );

    // Start outside the box and minimize at the lower bound.
    x0[0]              = 4;
    auto lower_problem = Utils::make_problem<Real>(
      []( ConstVectorRef x ) { return Real( 0.5 ) * std::pow( x[0] + Real( 2 ), 2 ); },
      []( ConstVectorRef x, VectorRef gradient ) { gradient[0] = x[0] + Real( 2 ); },
      []( ConstVectorRef, MatrixRef hessian ) { hessian.setConstant( Real( 1 ) ); } );
    auto const lower_result = solver.solve( lower_problem, x0, lower, upper );

    // A fixed variable (lower == upper) must remain feasible independently of
    // the initial point and of the ordinary gradient.
    x0[0]    = -100;
    lower[0] = upper[0]     = Real( 0.25 );
    auto const fixed_result = solver.solve( problem, x0, lower, upper );

    bool const passed = upper_result.status == Utils::Status::converged && upper_result.x[0] == Real( 1 ) &&
                        upper_result.projected_gradient_norm == Real( 0 ) &&
                        lower_result.status == Utils::Status::converged &&
                        std::abs( lower_result.x[0] ) <= Real( 1e-12 ) &&
                        lower_result.projected_gradient_norm <= Real( 1e-12 ) &&
                        fixed_result.status == Utils::Status::converged && fixed_result.x[0] == Real( 0.25 ) &&
                        fixed_result.primal_feasibility == Real( 0 );
    if ( !passed )
    {
      std::cerr << "upper: status=" << Utils::to_string( upper_result.status ) << " x=" << upper_result.x[0]
                << " p=" << upper_result.projected_gradient_norm << '\n';
      std::cerr << "lower: status=" << Utils::to_string( lower_result.status ) << " x=" << lower_result.x[0]
                << " p=" << lower_result.projected_gradient_norm << '\n';
      std::cerr << "fixed: status=" << Utils::to_string( fixed_result.status ) << " x=" << fixed_result.x[0]
                << " feasibility=" << fixed_result.primal_feasibility << '\n';
    }
    return passed;
  }

  bool test_escape_stationary_maximum()
  {
    using Real           = double;
    using Vector         = Utils::Vector<Real>;
    using ConstVectorRef = Utils::ConstVectorRef<Real>;
    using VectorRef      = Utils::VectorRef<Real>;
    using MatrixRef      = Utils::MatrixRef<Real>;

    Vector x0( 1 ), lower( 1 ), upper( 1 );
    x0[0]    = Real( 0 );  // Gradient zero, but this is a strict maximum.
    lower[0] = Real( -1 );
    upper[0] = Real( 1 );

    auto problem = Utils::make_problem<Real>(
      []( ConstVectorRef x ) { return -Real( 0.5 ) * x[0] * x[0]; },
      []( ConstVectorRef x, VectorRef gradient ) { gradient[0] = -x[0]; },
      []( ConstVectorRef, MatrixRef hessian ) { hessian.setConstant( Real( -1 ) ); } );

    Utils::Minimize_BBOX_NewtonCubic<Real> solver( 1 );
    auto const                             result = solver.solve( problem, x0, lower, upper );
    bool const passed = result.status == Utils::Status::converged &&
                        std::abs( std::abs( result.x[0] ) - Real( 1 ) ) <= Real( 1e-12 ) &&
                        result.objective <= -Real( 0.5 ) && result.projected_gradient_norm <= Real( 1e-12 );
    if ( !passed )
    {
      std::cerr << "stationary maximum: status=" << Utils::to_string( result.status ) << " x=" << result.x[0]
                << " f=" << result.objective << " p=" << result.projected_gradient_norm << '\n';
    }
    return passed;
  }

}  // namespace

int main()
{
  bool const passed = test_quadratic<float>() && test_quadratic<double>() && test_boundary_solution() &&
                      test_escape_stationary_maximum();
  std::cout << "Minimize_BBOX_NewtonCubic template/API test: " << ( passed ? "PASSED" : "FAILED" ) << '\n';
  return passed ? 0 : 1;
}
