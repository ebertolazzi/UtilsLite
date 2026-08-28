// SPDX-License-Identifier: MPL-2.0

#include "Utils_minimize_BBOX_Newton.hh"
#include "Utils_fmt.hh"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ND_func.cxx"

int main()
{
  using Real           = double;
  using Vector         = Utils::MinimizeNewton::Vector<Real>;
  using ConstVectorRef = Utils::MinimizeNewton::ConstVectorRef<Real>;
  using VectorRef      = Utils::MinimizeNewton::VectorRef<Real>;
  using MatrixRef      = Utils::MinimizeNewton::MatrixRef<Real>;
  using Status         = Utils::MinimizeNewton::Status;

  std::size_t converged            = 0;
  std::size_t stopped              = 0;
  std::size_t failed               = 0;
  std::size_t skipped              = 0;
  std::size_t iterations           = 0;
  std::size_t function_evaluations = 0;
  std::size_t gradient_evaluations = 0;
  std::size_t hessian_evaluations  = 0;

  for ( auto const & [function, name] : NL_list )
  {
    if ( name == "Katsuura10D" || name == "MichalewiczN10D" )
    {
      ++skipped;
      continue;
    }

    Vector const x0      = function->init();
    Vector const lower   = function->lower();
    Vector const upper   = function->upper();
    auto         problem = Utils::MinimizeNewton::make_problem<Real>(
      [&]( ConstVectorRef x ) { return ( *function )( x ); },
      [&]( ConstVectorRef x, VectorRef gradient ) { gradient = function->gradient( x ); },
      [&]( ConstVectorRef x, MatrixRef hessian ) { hessian = function->hessian( x ); } );

    Utils::MinimizeNewton::Options<Real> options;
    options.set_tolerances( 1e-12 );
    options.max_iterations           = 400;
    options.max_function_evaluations = 401;
    Utils::MinimizeNewton::Solver<Real> solver( x0.size(), options );
    auto const                          result = solver.solve( problem, x0, lower, upper );

    if ( result.status == Status::converged )
      ++converged;
    else if (
      result.status == Status::max_iterations || result.status == Status::max_function_evaluations ||
      result.status == Status::no_progress )
    {
      ++stopped;
      fmt::print(
        fmt::fg( fmt::color::gold ),
        "  stopped: {:<39} {:<36} iter={:<3} pi={:.3e}\n",
        name,
        Utils::MinimizeNewton::to_string( result.status ),
        result.iterations,
        result.projected_gradient_norm );
    }
    else
      ++failed;

    iterations += static_cast<std::size_t>( result.iterations );
    function_evaluations += static_cast<std::size_t>( result.function_evaluations );
    gradient_evaluations += static_cast<std::size_t>( result.gradient_evaluations );
    hessian_evaluations += static_cast<std::size_t>( result.hessian_evaluations );

    if ( !std::isfinite( result.objective ) ) ++failed;
  }

  fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "MinimizeNewton BOX — ND_func summary\n" );
  fmt::print( "  problems: {} converged, {} stopped, {} failed, {} skipped\n", converged, stopped, failed, skipped );
  fmt::print(
    "  work: {} iterations, {} f, {} g, {} H evaluations (limit 400)\n",
    iterations,
    function_evaluations,
    gradient_evaluations,
    hessian_evaluations );
  return failed == 0 ? 0 : 1;
}
