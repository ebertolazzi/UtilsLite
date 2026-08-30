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

#include "Utils_minimize_BBOX_1D.hh"
#include "Utils_fmt.hh"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;

using Utils::Minimize_BBOX_1D;
using Utils::Minimize_BBOX_1D_base_fun;

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

  class MinFunction final : public Minimize_BBOX_1D_base_fun<real_type>
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

  bool validate_result( min1D const & data, Minimize_BBOX_1D<real_type> const & solver, real_type x, string & reason )
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
    if ( solver.num_fun_eval() <= 0 )
    {
      reason = "objective was not evaluated";
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
    Minimize_BBOX_1D<real_type> solver;
    real_type                   x;

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
    real_type const             inf{ std::numeric_limits<real_type>::infinity() };
    Minimize_BBOX_1D<real_type> solver;
    solver.set_max_iterations( 3 );

    real_type x = solver.eval2(
      -inf,
      inf,
      []( real_type t ) { return power2( t - 1024 ); },
      []( real_type t ) { return 2 * ( t - 1024 ); } );
    bool ok = !solver.converged() && solver.used_iter() == 3 && solver.num_fun_eval() > 0 && std::isfinite( x );
    ++stats.tests;
    if ( !ok ) ++stats.failures;
    fmt::print( "\n  iteration budget: {} (iter={}, x={})\n", ok ? "PASS" : "FAIL", solver.used_iter(), x );

    solver.set_max_iterations( 200 );
    x = solver.eval2(
      -inf,
      inf,
      []( real_type t ) { return power2( t - 4 ); },
      []( real_type t ) { return 2 * ( t - 4 ); } );
    ok = solver.converged() && close_to( x, 4 ) && solver.num_fun_eval() > 0;
    ++stats.tests;
    if ( !ok ) ++stats.failures;
    fmt::print( "  state reset/reuse: {} (iter={}, x={})\n", ok ? "PASS" : "FAIL", solver.used_iter(), x );

    solver.set_max_iterations( 12 );
    x  = solver.eval2( -inf, inf, []( real_type t ) { return t; }, []( real_type ) { return real_type( 1 ); } );
    ok = !solver.converged() && solver.used_iter() == 12 && solver.num_fun_eval() > 0;
    ++stats.tests;
    if ( !ok ) ++stats.failures;
    fmt::print( "  no attained minimum: {} (iter={}, x={})\n", ok ? "PASS" : "FAIL", solver.used_iter(), x );
  }

  void run_guess_tests( TestStatistics & stats )
  {
    real_type const inf{ std::numeric_limits<real_type>::infinity() };

    struct Quadratic final : Minimize_BBOX_1D_base_fun<real_type>
    {
      real_type center;
      explicit Quadratic( real_type c ) : center( c ) {}
      real_type eval( real_type x ) const override { return power2( x - center ); }
      real_type D( real_type x ) const override { return 2 * ( x - center ); }
    };

    auto check = [&]( string const & label, real_type xguess, real_type a, real_type b, real_type expected )
    {
      Minimize_BBOX_1D<real_type> solver;
      real_type                   x = solver.eval2(
        xguess,
        a,
        b,
        [expected]( real_type t ) { return power2( t - expected ); },
        [expected]( real_type t ) { return 2 * ( t - expected ); } );
      bool ok = solver.converged() && close_to( x, expected );
      ++stats.tests;
      stats.iterations += solver.used_iter();
      stats.function_evaluations += solver.num_fun_eval();
      stats.derivative_evaluations += solver.num_fun_D_eval();
      if ( !ok ) ++stats.failures;
      fmt::print(
        "  guess {:<18}: {} (xg={:.3g}, x={:.17g}, iter={}, #f'={})\n",
        label,
        ok ? "PASS" : "FAIL",
        xguess,
        x,
        solver.used_iter(),
        solver.num_fun_D_eval() );
    };

    fmt::print( "\n" );
    check( "whole/right", -100, -inf, inf, 7 );
    check( "whole/left", 100, -inf, inf, 7 );
    check( "right half-line", 1, 0, inf, 9 );
    check( "left half-line", -1, -inf, 0, -9 );
    check( "lower-bound KKT", 4, 0, 10, 0 );
    check( "exact stationary", 3, -inf, inf, 3 );

    Quadratic                   fun{ 5 };
    Minimize_BBOX_1D<real_type> solver;
    real_type                   x  = solver.eval( -20, -inf, inf, &fun );
    bool                        ok = solver.converged() && close_to( x, 5 );
    ++stats.tests;
    stats.iterations += solver.used_iter();
    stats.function_evaluations += solver.num_fun_eval();
    stats.derivative_evaluations += solver.num_fun_D_eval();
    if ( !ok ) ++stats.failures;
    fmt::print( "  guess {:<18}: {} (x={:.17g})\n", "virtual overload", ok ? "PASS" : "FAIL", x );
  }

  void run_out_of_domain_tests( TestStatistics & stats )
  {
    real_type const inf{ std::numeric_limits<real_type>::infinity() };
    real_type const nan{ std::numeric_limits<real_type>::quiet_NaN() };

    auto check = [&]<typename FUN, typename FUN_D>(
                   string const & label,
                   real_type      xguess,
                   real_type      a,
                   real_type      b,
                   real_type      expected,
                   FUN &&         fun,
                   FUN_D &&       fun_D )
    {
      Minimize_BBOX_1D<real_type> solver;
      real_type x  = solver.eval2( xguess, a, b, std::forward<FUN>( fun ), std::forward<FUN_D>( fun_D ) );
      bool      ok = solver.converged() && close_to( x, expected ) && std::isfinite( solver.min_value() ) &&
                     std::isfinite( solver.derivative() );
      ++stats.tests;
      stats.iterations += solver.used_iter();
      stats.function_evaluations += solver.num_fun_eval();
      stats.derivative_evaluations += solver.num_fun_D_eval();
      if ( !ok ) ++stats.failures;
      fmt::print(
        "  domain {:<22}: {} (x={:.17g}, iter={}, #f={}, #f'={})\n",
        label,
        ok ? "PASS" : "FAIL",
        x,
        solver.used_iter(),
        solver.num_fun_eval(),
        solver.num_fun_D_eval() );
    };

    fmt::print( "\n" );

    // f detects the invalid endpoint even though the derivative formula is
    // finite there.  Bisection from x=4 to the rejected x=0 lands on x*=2.
    check(
      "f=Inf, finite f'",
      4,
      0,
      10,
      2,
      [inf]( real_type x ) { return x > 0 ? power2( x - 2 ) : inf; },
      []( real_type x ) { return 2 * ( x - 2 ); } );

    // Geometric expansion jumps from the valid domain x<5 to x=7.  The
    // valid/invalid bisection must return to x=4 before root refinement.
    check(
      "expansion to NaN f'",
      0,
      -inf,
      inf,
      4,
      [inf]( real_type x ) { return x < 5 ? power2( x - 4 ) : inf; },
      [nan]( real_type x ) { return x < 5 ? 2 * ( x - 4 ) : nan; } );

    // A signed infinity is invalid independently of its apparent sign; it
    // must never be mistaken for a derivative bracket endpoint.
    check(
      "infinite derivative",
      4,
      -10,
      10,
      1,
      [inf]( real_type x ) { return x > 0 ? power2( x - 1 ) : inf; },
      [inf]( real_type x ) { return x > 0 ? 2 * ( x - 1 ) : inf; } );

    // The default half-line guesses are the finite endpoints.  These two
    // tests exercise automatic relocation when that endpoint is excluded.
    check(
      "invalid lower guess",
      0,
      0,
      inf,
      2,
      [inf]( real_type x ) { return x > 0 ? power2( x - 2 ) : inf; },
      [nan]( real_type x ) { return x > 0 ? 2 * ( x - 2 ) : nan; } );
    check(
      "invalid upper guess",
      0,
      -inf,
      0,
      -2,
      [inf]( real_type x ) { return x < 0 ? power2( x + 2 ) : inf; },
      [nan]( real_type x ) { return x < 0 ? 2 * ( x + 2 ) : nan; } );

    // Guess, box endpoints and the first coarse samples are all invalid.
    // Dyadic refinement must discover the connected interior domain.
    check(
      "interior narrow domain",
      0.5,
      0,
      1,
      0.65,
      [inf]( real_type x ) { return x > 0.6 && x < 0.7 ? power2( x - 0.65 ) : inf; },
      [nan]( real_type x ) { return x > 0.6 && x < 0.7 ? 2 * ( x - 0.65 ) : nan; } );

    // The declared boxes are deliberately wider than the effective domain
    // [-1,1].  Cover an interior stationary point and minima attained at both
    // effective-domain boundaries.  The boundary cases have nonzero
    // derivatives and therefore exercise the one-sided numerical KKT test.
    auto finite_on_unit_box = [inf]( real_type x, real_type value ) { return std::abs( x ) <= 1 ? value : inf; };
    check(
      "[-1,1], interior min",
      0,
      -2,
      2,
      0.25,
      [finite_on_unit_box]( real_type x ) { return finite_on_unit_box( x, power2( x - 0.25 ) ); },
      [nan]( real_type x ) { return std::abs( x ) <= 1 ? 2 * ( x - 0.25 ) : nan; } );
    check(
      "[-1,1], lower min",
      0,
      -2,
      2,
      -1,
      [finite_on_unit_box]( real_type x ) { return finite_on_unit_box( x, x ); },
      [nan]( real_type x ) { return std::abs( x ) <= 1 ? real_type( 1 ) : nan; } );
    check(
      "[-1,1], upper min",
      0,
      -100,
      100,
      1,
      [finite_on_unit_box]( real_type x ) { return finite_on_unit_box( x, -x ); },
      [nan]( real_type x ) { return std::abs( x ) <= 1 ? real_type( -1 ) : nan; } );

    // On the real open domain (-1,1), log(1-x^2)+1e-6*x has no attained
    // mathematical minimum and tends to -Inf at both sides.  In floating
    // point, however, the linear perturbation selects the last representable
    // point to the right of -1 as the numerical minimum.  The solver must
    // reach that point without feeding log() an invalid argument to the root
    // solver and without returning NaN/Inf as the stored objective.
    {
      real_type expected = std::nextafter( real_type( -1 ), real_type( 0 ) );
      check(
        "log open unit domain",
        0,
        -2,
        2,
        expected,
        [inf]( real_type x ) { return std::abs( x ) < 1 ? std::log( 1 - x * x ) + real_type( 1e-6 ) * x : -inf; },
        [nan]( real_type x ) { return std::abs( x ) < 1 ? -2 * x / ( 1 - x * x ) + real_type( 1e-6 ) : nan; } );
    }

    // Convex logarithmic barrier on the same declared box.  Unlike the
    // previous concave logarithm, this objective tends to +Inf at the open
    // boundary and has a true stationary minimum just inside x=-1:
    //   1 + 2*eps*x/(1-x^2) = 0,
    //   x = eps - sqrt(1+eps^2), eps=1e-6.
    {
      real_type constexpr eps  = 1e-6;
      real_type const expected = eps - std::sqrt( 1 + eps * eps );
      check(
        "-eps*log barrier + x",
        0,
        -2,
        2,
        expected,
        [inf]( real_type x ) { return std::abs( x ) < 1 ? -real_type( 1e-6 ) * std::log( 1 - x * x ) + x : inf; },
        [nan]( real_type x )
        { return std::abs( x ) < 1 ? real_type( 1 ) + real_type( 2e-6 ) * x / ( 1 - x * x ) : nan; } );
    }

    // f(x)=x on the open domain x>0 has only an unattained infimum at zero.
    // Re-entering the domain must not turn that excluded boundary into a
    // falsely converged KKT point.
    {
      Minimize_BBOX_1D<real_type> solver;
      solver.set_max_iterations( 80 );
      real_type x = solver.eval2(
        0.5,
        0,
        1,
        [inf]( real_type t ) { return t > 0 ? t : inf; },
        [nan]( real_type t ) { return t > 0 ? real_type( 1 ) : nan; } );
      bool ok = !solver.converged() && solver.hit_max_iterations() && x > 0 && std::isfinite( solver.min_value() );
      ++stats.tests;
      stats.iterations += solver.used_iter();
      stats.function_evaluations += solver.num_fun_eval();
      stats.derivative_evaluations += solver.num_fun_D_eval();
      if ( !ok ) ++stats.failures;
      fmt::print(
        "  domain {:<22}: {} (x={:.3e}, iter={})\n",
        "unattained boundary",
        ok ? "PASS" : "FAIL",
        x,
        solver.used_iter() );
    }
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
  run_guess_tests( stats );
  run_out_of_domain_tests( stats );

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
