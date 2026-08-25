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

#include "Utils_Minimize1D.hh"
#include "Utils_fmt.hh"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace std;

using Utils::Minimize1D;
using Utils::Minimize1D_base_fun;

#include "1D_fun.cxx"

namespace
{
  struct TestStatistics
  {
    int tests                  = 0;
    int failures               = 0;
    int iterations             = 0;
    int function_evaluations   = 0;
    int derivative_evaluations = 0;
  };

  class MinFunction final : public Minimize1D_base_fun<real_type>
  {
    min1D const & m_data;

  public:
    explicit MinFunction( min1D const & data ) : m_data( data ) {}

    real_type eval( real_type x ) const override { return m_data.eval( x ); }
    real_type D( real_type x ) const override { return m_data.D( x ); }
  };

  bool close_to( real_type x, real_type expected )
  {
    real_type scale{ std::max( real_type( 1 ), std::abs( expected ) ) };
    real_type tol{ 128 * std::numeric_limits<real_type>::epsilon() * scale };
    return std::abs( x - expected ) <= tol;
  }

  string interval_kind( min1D const & data )
  {
    if ( std::isfinite( data.a() ) && std::isfinite( data.b() ) ) return "bounded";
    if ( std::isfinite( data.a() ) ) return "right-inf";
    if ( std::isfinite( data.b() ) ) return "left-inf";
    return "unbounded";
  }

  bool validate_result( min1D const & data, Minimize1D<real_type> const & solver, real_type x, string & reason )
  {
    if ( !solver.converged() )
    {
      reason = "not converged";
      return false;
    }
    if ( !( data.a() <= x && x <= data.b() ) )
    {
      reason = "infeasible result";
      return false;
    }
    if ( !close_to( x, data.x_min() ) )
    {
      reason = fmt::format( "x error={:.3e}", x - data.x_min() );
      return false;
    }
    if ( solver.num_fun_eval() != 1 )
    {
      reason = fmt::format( "unexpected #f={}", solver.num_fun_eval() );
      return false;
    }
    if ( solver.num_fun_D_eval() <= 0 )
    {
      reason = "derivative was not evaluated";
      return false;
    }
    if ( solver.used_iter() > solver.max_iterations() )
    {
      reason = "iteration limit exceeded";
      return false;
    }
    if ( solver.min_value() != data.eval( x ) )
    {
      reason = "stored objective mismatch";
      return false;
    }
    if ( solver.derivative() != data.D( x ) )
    {
      reason = "stored derivative mismatch";
      return false;
    }
    if ( solver.bracket_a() > solver.bracket_b() )
    {
      reason = "reversed final bracket";
      return false;
    }

    bool at_lower{ std::isfinite( data.a() ) && data.x_min() == data.a() };
    bool at_upper{ std::isfinite( data.b() ) && data.x_min() == data.b() };
    if ( at_lower && !( x == data.a() && solver.derivative() >= 0 ) )
    {
      reason = "lower-bound KKT failure";
      return false;
    }
    if ( at_upper && !( x == data.b() && solver.derivative() <= 0 ) )
    {
      reason = "upper-bound KKT failure";
      return false;
    }

    if ( std::isfinite( data.a() ) && data.eval( x ) > data.eval( data.a() ) )
    {
      reason = "objective exceeds f(a)";
      return false;
    }
    if ( std::isfinite( data.b() ) && data.eval( x ) > data.eval( data.b() ) )
    {
      reason = "objective exceeds f(b)";
      return false;
    }
    return true;
  }

  void run_dataset_test( min1D const & data, bool use_base_class, TestStatistics & stats )
  {
    Minimize1D<real_type> solver;
    real_type             x;

    if ( use_base_class )
    {
      MinFunction fun( data );
      x = solver.eval( data.a(), data.b(), &fun );
    }
    else
    {
      x = solver.eval2( data.a(), data.b(), data.function(), data.derivative() );
    }

    string reason;
    bool   ok{ validate_result( data, solver, x, reason ) };

    ++stats.tests;
    stats.iterations += solver.used_iter();
    stats.function_evaluations += solver.num_fun_eval();
    stats.derivative_evaluations += solver.num_fun_D_eval();
    if ( !ok ) ++stats.failures;

    auto status_style = ok ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );
    fmt::print(
      "  {:>3} {:<10} {:<7} {:>5} {:>5} {:>5} {:>20.12g} {:>20.12g} ",
      stats.tests,
      interval_kind( data ),
      use_base_class ? "virtual" : "lambda",
      solver.used_iter(),
      solver.num_fun_eval(),
      solver.num_fun_D_eval(),
      x,
      data.x_min() );
    fmt::print( status_style | fmt::emphasis::bold, "{}", ok ? "PASS" : "FAIL" );
    if ( !ok ) fmt::print( " ({}; {})", reason, data.info() );
    fmt::print( "\n" );
  }

  void run_iteration_limit_tests( TestStatistics & stats )
  {
    real_type const       inf{ std::numeric_limits<real_type>::infinity() };
    Minimize1D<real_type> solver;
    solver.set_max_iterations( 3 );

    real_type x = solver.eval2(
      -inf,
      inf,
      []( real_type t ) { return power2( t - 1024 ); },
      []( real_type t ) { return 2 * ( t - 1024 ); } );
    bool ok = !solver.converged() && solver.used_iter() == 3 && solver.num_fun_eval() == 1 && std::isfinite( x );
    ++stats.tests;
    if ( !ok ) ++stats.failures;
    fmt::print( "\n  iteration budget: {} (iter={}, x={})\n", ok ? "PASS" : "FAIL", solver.used_iter(), x );

    solver.set_max_iterations( 200 );
    x = solver.eval2(
      -inf,
      inf,
      []( real_type t ) { return power2( t - 4 ); },
      []( real_type t ) { return 2 * ( t - 4 ); } );
    ok = solver.converged() && close_to( x, 4 ) && solver.num_fun_eval() == 1;
    ++stats.tests;
    if ( !ok ) ++stats.failures;
    fmt::print( "  state reset/reuse: {} (iter={}, x={})\n", ok ? "PASS" : "FAIL", solver.used_iter(), x );

    solver.set_max_iterations( 12 );
    x  = solver.eval2( -inf, inf, []( real_type t ) { return t; }, []( real_type ) { return real_type( 1 ); } );
    ok = !solver.converged() && solver.used_iter() == 12 && solver.num_fun_eval() == 1;
    ++stats.tests;
    if ( !ok ) ++stats.failures;
    fmt::print( "  no attained minimum: {} (iter={}, x={})\n", ok ? "PASS" : "FAIL", solver.used_iter(), x );
  }
}  // namespace

int main()
{
  fmt::print(
    fmt::fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "  ║                               MINIMIZE 1D TEST SUITE                                         ║\n"
    "  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n\n" );

  fmt::print(
    "  {:>3} {:<10} {:<7} {:>5} {:>5} {:>5} {:>20} {:>20} {}\n",
    "#",
    "interval",
    "API",
    "iter",
    "#f",
    "#f'",
    "computed x",
    "expected x",
    "status" );
  fmt::print( "  {0:─^103}\n", "" );

  vector<unique_ptr<min1D>> problems;
  build_1dmin_list( problems );

  TestStatistics stats;
  for ( size_t i = 0; i < problems.size(); ++i ) run_dataset_test( *problems[i], i % 2 == 0, stats );

  run_iteration_limit_tests( stats );

  fmt::print(
    "\n  {:─^103}\n"
    "  Tests: {}  Failures: {}  Iterations: {}  f evaluations: {}  f' evaluations: {}\n",
    "",
    stats.tests,
    stats.failures,
    stats.iterations,
    stats.function_evaluations,
    stats.derivative_evaluations );

  if ( stats.failures == 0 )
  {
    fmt::print( fmt::fg( fmt::color::green ) | fmt::emphasis::bold, "  ALL TESTS PASSED\n\n" );
    return EXIT_SUCCESS;
  }

  fmt::print( fmt::fg( fmt::color::red ) | fmt::emphasis::bold, "  TEST SUITE FAILED\n\n" );
  return EXIT_FAILURE;
}
