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

/**
 * @file Utils_minimize_BBOX_1D.hh
 * @brief Header-only minimization of a differentiable scalar function on an
 *        interval whose endpoints may be infinite.
 */

/**
 * @page minimize_bbox_1d Algorithm for one-dimensional box minimization
 *
 * This header solves the differentiable, convex problem
 * \f[
 *   \min_{x\in[a,b]} f(x), \qquad
 *   -\infty\leq a < b\leq +\infty .
 * \f]
 * Convexity makes \f$f'\f$ nondecreasing.  Hence the KKT conditions are
 * \f[
 * \begin{array}{ll}
 *   f'(x_\star)=0,   & a<x_\star<b,\\
 *   f'(a)\geq0,      & x_\star=a,\\
 *   f'(b)\leq0,      & x_\star=b.
 * \end{array}
 * \f]
 *
 * The effective domain may be smaller than the supplied box.  A point is
 * considered usable only when both \f$f(x)\f$ and \f$f'(x)\f$ are finite;
 * `NaN` and either signed infinity mean that the trial point lies outside the
 * numerical domain.  Given a finite initial guess (projected onto
 * \f$[a,b]\f$ if necessary), the algorithm first obtains such a usable
 * sample.  A zero derivative finishes immediately.  Otherwise
 * its sign identifies the only direction that can contain the minimum:
 * \f[
 *   f'(x_g)<0 \Longrightarrow x_\star>x_g,
 *   \qquad
 *   f'(x_g)>0 \Longrightarrow x_\star<x_g.
 * \f]
 * At a finite endpoint the appropriate KKT condition is tested.  At an
 * infinite endpoint, finite trial points are generated with geometrically
 * increasing steps until a negative-to-positive derivative bracket is found.
 * If a trial leaves the numerical domain, the last usable point and the
 * rejected point delimit a valid/invalid interval.  Safeguarded bisection of
 * that interval moves back into the domain and continues looking for the
 * derivative sign change without ever passing a non-finite value to the root
 * solver.  Infinity itself is never passed to the objective or derivative.
 *
 * Inside a bracket \f$[\ell,r]\f$ satisfying
 * \f$f'(\ell)<0<f'(r)\f$, the stationary point is computed by
 * Chandrupatla's safeguarded interpolation/bisection method.  Termination is
 * scale-aware: it occurs when the interval is below the requested tolerance,
 * below a multiple of \f$\epsilon\max(1,|\ell|,|r|)\f$, or its endpoints are
 * adjacent floating-point numbers.  This avoids impossible convergence
 * requests near very large or subnormal abscissae.
 *
 * The overloads without a guess choose a deterministic finite one: the safe
 * midpoint on a bounded interval, its finite endpoint on a half-line, and
 * zero on the whole real line.  Correct global minimization assumes convexity
 * (or, more generally, a single negative-to-positive transition of \f$f'\f$).
 * The finite part of the domain is assumed to be an interval.  If an open
 * boundary is localized down to adjacent floating-point numbers, the last
 * finite representable point is treated as the numerical domain boundary and
 * accepted only when its one-sided derivative satisfies the corresponding
 * KKT sign.  If no finite sample can be located, or the iteration budget ends
 * before that localization is complete, the result is marked not converged.
 */

#pragma once

#ifndef UTILS_MINIMIZE_BBOX_1D_dot_HH
#define UTILS_MINIMIZE_BBOX_1D_dot_HH

#include "Utils_AlgoBracket.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

namespace Utils
{

  /**
   * @brief Interface for the objective and its first derivative.
   * @tparam Real Floating-point type.
   */
  template <typename Real> class Minimize_BBOX_1D_base_fun
  {
  public:
    virtual ~Minimize_BBOX_1D_base_fun() = default;

    virtual Real eval( Real x ) const = 0;
    virtual Real D( Real x ) const    = 0;
  };

  /** @brief Adapter for callable objective and derivative objects. */
  template <typename Real, typename PFUN, typename PFUN_D> class Minimize_BBOX_1D_fun
    : public Minimize_BBOX_1D_base_fun<Real>
  {
    PFUN   m_fun;
    PFUN_D m_fun_D;

  public:
    template <typename F, typename FD> Minimize_BBOX_1D_fun( F && fun, FD && fun_D )
      : m_fun( std::forward<F>( fun ) ), m_fun_D( std::forward<FD>( fun_D ) )
    {
    }

    Real eval( Real x ) const override { return m_fun( x ); }
    Real D( Real x ) const override { return m_fun_D( x ); }
  };

  /**
   * @brief One-dimensional minimizer for a differentiable convex function.
   *
   * The method starts from a finite guess and uses the sign of its derivative
   * to select the only direction that can contain the minimizer.  It then
   * checks the relevant boundary KKT condition or enlarges a finite bracket
   * geometrically on an unbounded side.  The zero of f' is computed by
   * Utils::AlgoBracket using Chandrupatla's method.
   *
   * No user tolerance is used: the root solver receives the smallest positive
   * representable tolerances and stops at its intrinsic floating-point limit
   * (or at the maximum iteration count).  Consequently a small but nonzero
   * derivative is never by itself considered convergence.
   *
   * @note The global-minimum guarantee requires f to be convex on [a,b]
   *       (equivalently, f' is nondecreasing).  For a merely unimodal smooth
   *       function the same algorithm applies provided f' has a single
   *       negative-to-positive transition.
   *
   * @tparam Real Floating-point type supported by AlgoBracket (float or double).
   */
  template <typename Real> class Minimize_BBOX_1D
  {
  public:
    using Integer = int;

    static_assert(
      std::is_same_v<Real, float> || std::is_same_v<Real, double>,
      "Minimize_BBOX_1D<Real> requires Real to be float or double" );

  private:
    using Limits = std::numeric_limits<Real>;

    struct Sample
    {
      Real x{ 0 };
      Real f{ Limits::quiet_NaN() };
      Real Df{ Limits::quiet_NaN() };
      bool valid{ false };
    };

    Minimize_BBOX_1D_base_fun<Real> const * m_function = nullptr;
    AlgoBracket<Real>                       m_bracket;

    Integer m_max_iteration          = 200;
    Integer m_iteration_count        = 0;
    Integer m_fun_evaluation_count   = 0;
    Integer m_fun_D_evaluation_count = 0;

    bool m_converged          = false;
    bool m_hit_max_iterations = false;
    Real m_x_min              = 0;
    Real m_f_min              = 0;
    Real m_Df_min             = 0;
    Real m_bracket_a          = 0;
    Real m_bracket_b          = 0;

    static Real smallest_positive()
    {
      Real tol = Limits::denorm_min();
      if ( !( tol > 0 ) ) tol = Limits::min();
      return tol;
    }

    Real evaluate( Real x )
    {
      ++m_fun_evaluation_count;
      return m_function->eval( x );
    }

    Real evaluate_D( Real x )
    {
      ++m_fun_D_evaluation_count;
      return m_function->D( x );
    }

    /**
     * Evaluate a trial without throwing merely because it is outside the
     * numerical domain.  The derivative is evaluated first: when it is
     * already non-finite there is no reason to call the usually more
     * expensive objective.  A finite derivative alone is not sufficient,
     * because some user formulas remain finite outside the objective domain.
     */
    Sample probe( Real x )
    {
      Sample p;
      p.x  = x;
      p.Df = evaluate_D( x );
      if ( std::isfinite( p.Df ) )
      {
        p.f     = evaluate( x );
        p.valid = std::isfinite( p.f );
      }
      return p;
    }

    Real finish( Real x, Real dfx, bool ok )
    {
      m_x_min     = x;
      m_Df_min    = dfx;
      m_f_min     = evaluate( x );
      m_converged = ok && std::isfinite( dfx ) && std::isfinite( m_f_min );
      return x;
    }

    /** Store a sample that has already been evaluated, avoiding duplicate f. */
    Real finish( Sample const & p, bool ok )
    {
      m_x_min     = p.x;
      m_Df_min    = p.Df;
      m_f_min     = p.f;
      m_converged = ok && p.valid;
      return p.x;
    }

    static Real initial_step( Real x, Real direction )
    {
      Real limit   = direction > 0 ? Limits::max() : Limits::lowest();
      Real near_x  = std::nextafter( x, limit );
      Real spacing = std::abs( near_x - x );
      return std::max( Real( 1 ), spacing );
    }

    static Real expanded_point( Real x, Real direction, Real step )
    {
      Real limit = direction > 0 ? Limits::max() : Limits::lowest();
      if ( x == limit ) return x;

      Real next = direction > 0 ? x + step : x - step;
      if ( !std::isfinite( next ) ) next = limit;

      if ( direction > 0 )
      {
        if ( next <= x ) next = std::nextafter( x, limit );
      }
      else
      {
        if ( next >= x ) next = std::nextafter( x, limit );
      }
      return next;
    }

    static void double_step( Real & step )
    {
      Real max_value = Limits::max();
      step           = step > max_value / 2 ? max_value : 2 * step;
    }

    /** Select the finite guess used by the legacy overloads. */
    static Real default_guess( Real a, Real b )
    {
      if ( std::isfinite( a ) && std::isfinite( b ) ) return std::midpoint( a, b );
      if ( std::isfinite( a ) ) return a;
      if ( std::isfinite( b ) ) return b;
      return Real( 0 );
    }

    /**
     * Locate the first point at which both f and f' are finite.
     *
     * A supplied guess can itself be on an open domain boundary (a common
     * example is x=0 for a logarithmic barrier on [0,+inf)).  On a bounded
     * box, successively finer dyadic grids avoid assuming on which side of
     * the guess the domain lies.  On an unbounded box, finite trials expand
     * geometrically in every feasible direction.  The search is necessarily
     * bounded by the common iteration budget: an arbitrarily narrow unknown
     * domain cannot be guaranteed discoverable by a black-box algorithm.
     */
    bool find_valid_start( Real xguess, Real a, Real b, Sample & result )
    {
      result = probe( xguess );
      if ( result.valid ) return true;

      auto try_point = [&]( Real x ) -> bool
      {
        if ( m_iteration_count >= m_max_iteration ) return false;
        Sample candidate = probe( x );
        ++m_iteration_count;
        if ( candidate.valid )
        {
          result = candidate;
          return true;
        }
        return false;
      };

      if ( std::isfinite( a ) && std::isfinite( b ) )
      {
        if ( xguess != a && try_point( a ) ) return true;
        if ( xguess != b && try_point( b ) ) return true;

        // Odd nodes are new at every dyadic refinement level, so no sample is
        // repeated. std::lerp remains well behaved even for a and b of large
        // opposite magnitude, where the naive expression a+t*(b-a) overflows.
        for ( Integer denominator = 2; m_iteration_count < m_max_iteration; )
        {
          for ( Integer numerator = 1; numerator < denominator && m_iteration_count < m_max_iteration;
                numerator += 2 )
          {
            Real t = Real( numerator ) / Real( denominator );
            Real x = std::lerp( a, b, t );
            if ( x != xguess && try_point( x ) ) return true;
          }
          if ( denominator > std::numeric_limits<Integer>::max() / 2 ) break;
          denominator *= 2;
        }
      }
      else
      {
        // Check any finite endpoint explicitly before exploring a half-line.
        if ( std::isfinite( a ) && xguess != a && try_point( a ) ) return true;
        if ( std::isfinite( b ) && xguess != b && try_point( b ) ) return true;

        Real scale = std::max( Real( 1 ), std::abs( xguess ) );
        Real step  = std::sqrt( Limits::epsilon() ) * scale;
        if ( !( step > 0 ) || !std::isfinite( step ) ) step = Real( 1 );

        Real last_right = xguess;
        Real last_left  = xguess;
        bool can_right  = xguess < b;
        bool can_left   = xguess > a;

        while ( m_iteration_count < m_max_iteration && ( can_right || can_left ) )
        {
          if ( can_right )
          {
            Real x = expanded_point( xguess, Real( 1 ), step );
            if ( std::isfinite( b ) ) x = std::min( x, b );
            can_right = x != last_right;
            if ( can_right )
            {
              last_right = x;
              if ( try_point( x ) ) return true;
              if ( x == b ) can_right = false;
            }
          }
          if ( can_left && m_iteration_count < m_max_iteration )
          {
            Real x = expanded_point( xguess, Real( -1 ), step );
            if ( std::isfinite( a ) ) x = std::max( x, a );
            can_left = x != last_left;
            if ( can_left )
            {
              last_left = x;
              if ( try_point( x ) ) return true;
              if ( x == a ) can_left = false;
            }
          }
          double_step( step );
        }
      }

      if ( m_iteration_count >= m_max_iteration ) m_hit_max_iterations = true;
      return false;
    }

    Real solve_bracket( Real a, Real Da, Real b, Real Db )
    {
      m_bracket_a = a;
      m_bracket_b = b;

      // AlgoBracket expects finite function values once its selected root
      // algorithm starts.  A perfectly valid derivative can overflow far
      // from the minimizer, so first move every infinite endpoint inward
      // while retaining the negative-to-positive bracket.
      while ( ( !std::isfinite( Da ) || !std::isfinite( Db ) || !std::isfinite( b - a ) ) &&
              m_iteration_count < m_max_iteration )
      {
        Real c = std::midpoint( a, b );
        if ( c == a || c == b ) break;

        Sample pc = probe( c );
        ++m_iteration_count;
        if ( !pc.valid )
        {
          // Two valid endpoints with opposite derivative signs must have a
          // valid midpoint when the finite domain is an interval.  Failing
          // this test means the documented domain assumption is violated.
          return std::abs( Da ) <= std::abs( Db ) ? finish( a, Da, false ) : finish( b, Db, false );
        }
        Real Dc = pc.Df;
        if ( Dc == 0 )
        {
          m_bracket_a = m_bracket_b = c;
          return finish( pc, true );
        }

        if ( Dc < 0 )
        {
          a  = c;
          Da = Dc;
        }
        else
        {
          b  = c;
          Db = Dc;
        }
        m_bracket_a = a;
        m_bracket_b = b;
      }

      Integer remaining        = m_max_iteration - m_iteration_count;
      bool    budget_exhausted = remaining <= 0;
      if ( budget_exhausted || !std::isfinite( Da ) || !std::isfinite( Db ) || !std::isfinite( b - a ) )
      {
        if ( budget_exhausted ) m_hit_max_iterations = true;
        return std::abs( Da ) <= std::abs( Db ) ? finish( a, Da, false ) : finish( b, Db, false );
      }

      m_bracket.reset();
      m_bracket.select( AlgoBracket<Real>::Method::CHANDRUPATLA );
      m_bracket.set_tolerance_x( smallest_positive() );
      m_bracket.set_tolerance_f( smallest_positive() );
      m_bracket.set_max_iterations( remaining );

      Real x{ m_bracket.eval3( a, b, Da, Db, [this]( Real t ) { return this->evaluate_D( t ); } ) };
      m_iteration_count += m_bracket.used_iter();
      m_bracket_a = m_bracket.a();
      m_bracket_b = m_bracket.b();

      // AlgoBracket's endpoint residuals are part of its internal state and
      // are not guaranteed to equal f'(x) on return.  In particular, Brent
      // may collapse the final bracket and store a conventional zero residual
      // without evaluating the derivative again at the returned x.  Evaluate
      // it explicitly so derivative() always reports the true user function
      // value and the derivative counter remains exact.
      Real Dx = evaluate_D( x );
      if ( !m_bracket.converged() && m_bracket.used_iter() >= remaining ) m_hit_max_iterations = true;
      return finish( x, Dx, m_bracket.converged() );
    }

    /**
     * Recover after an outward trial left the numerical domain.
     *
     * `inside` is the last finite sample and `outside_x` is a rejected trial
     * in `direction`.  Bisection preserves that valid/invalid ordering.  Each
     * finite midpoint either supplies the missing derivative sign and starts
     * the Chandrupatla solve, or becomes the new inside point.  A non-finite
     * midpoint simply tightens the outside side.
     */
    Real recover_from_invalid( Sample inside, Real outside_x, Real direction )
    {
      while ( m_iteration_count < m_max_iteration )
      {
        m_bracket_a = std::min( inside.x, outside_x );
        m_bracket_b = std::max( inside.x, outside_x );

        Real c = std::midpoint( inside.x, outside_x );
        if ( c == inside.x || c == outside_x )
        {
          // No representable abscissa exists between the last finite point
          // and the rejected point.  Therefore `inside` is the effective
          // floating-point domain boundary.  Accept it only if the derivative
          // still points out of the domain: this is exactly the one-sided KKT
          // condition at that numerical boundary.
          bool boundary_kkt = direction > 0 ? inside.Df <= 0 : inside.Df >= 0;
          return finish( inside, boundary_kkt );
        }

        Sample trial = probe( c );
        ++m_iteration_count;
        if ( !trial.valid )
        {
          outside_x = c;
          continue;
        }
        if ( trial.Df == 0 )
        {
          m_bracket_a = m_bracket_b = c;
          return finish( trial, true );
        }

        if ( direction > 0 )
        {
          // The search started with f'(inside)<0 and moves to the right.
          if ( trial.Df > 0 ) return solve_bracket( inside.x, inside.Df, trial.x, trial.Df );
        }
        else
        {
          // The search started with f'(inside)>0 and moves to the left.
          if ( trial.Df < 0 ) return solve_bracket( trial.x, trial.Df, inside.x, inside.Df );
        }
        inside = trial;
      }

      m_hit_max_iterations = true;
      return finish( inside, false );
    }

    Real expand_and_solve( Sample current, Real direction )
    {
      Real step             = initial_step( current.x, direction );
      bool budget_exhausted = true;

      while ( m_iteration_count < m_max_iteration )
      {
        Real next_x = expanded_point( current.x, direction, step );
        if ( next_x == current.x )
        {
          budget_exhausted = false;  // expansion saturated at the domain limit, not the iteration cap
          break;
        }

        Sample next = probe( next_x );
        ++m_iteration_count;

        if ( direction > 0 )
        {
          m_bracket_a = current.x;
          m_bracket_b = next_x;
        }
        else
        {
          m_bracket_a = next_x;
          m_bracket_b = current.x;
        }

        if ( !next.valid ) return recover_from_invalid( current, next_x, direction );

        if ( next.Df == 0 )
        {
          m_bracket_a = m_bracket_b = next.x;
          return finish( next, true );
        }

        if ( direction > 0 )
        {
          if ( next.Df > 0 ) return solve_bracket( current.x, current.Df, next.x, next.Df );
        }
        else
        {
          if ( next.Df < 0 ) return solve_bracket( next.x, next.Df, current.x, current.Df );
        }

        current = next;
        double_step( step );
      }

      if ( budget_exhausted ) m_hit_max_iterations = true;
      return finish( current, false );
    }

    /**
     * Core implementation shared by all public interfaces.
     *
     * Both the objective and derivative validate the guess initially.  Once
     * a finite sample has been found, the derivative sign discards one half
     * of the domain before any endpoint evaluation.  This is particularly
     * useful on half-lines and when the finite box is wide.
     */
    Real eval_impl( Real xguess, Real a, Real b )
    {
      m_iteration_count        = 0;
      m_fun_evaluation_count   = 0;
      m_fun_D_evaluation_count = 0;
      m_converged              = false;
      m_hit_max_iterations     = false;

      Utils::Check( m_function != nullptr, "Minimize_BBOX_1D::eval(), function pointer is null\n" );
      Utils::Check(
        !std::isnan( a ) && !std::isnan( b ) && a < b,
        "Minimize_BBOX_1D::eval(a={}, b={}), expected a < b and non-NaN bounds\n",
        a,
        b );
      Utils::Check(
        std::isfinite( xguess ),
        "Minimize_BBOX_1D::eval(xguess={}, a={}, b={}), expected a finite guess\n",
        xguess,
        a,
        b );
      // A box solver naturally interprets an infeasible guess through its
      // Euclidean projection.  This also makes the overload robust to tiny
      // feasibility drift produced by an outer iteration.
      xguess = std::clamp( xguess, a, b );

      m_bracket_a = a;
      m_bracket_b = b;

      Sample guess;
      if ( !find_valid_start( xguess, a, b, guess ) ) return finish( guess, false );

      xguess     = guess.x;
      Real Dguess = guess.Df;
      if ( Dguess == 0 )
      {
        m_bracket_a = m_bracket_b = xguess;
        return finish( guess, true );
      }

      if ( Dguess < 0 )
      {
        // At the upper bound f'(b)<=0 is the upper-bound KKT condition.
        if ( xguess == b )
        {
          m_bracket_a = m_bracket_b = b;
          return finish( guess, true );
        }
        if ( std::isfinite( b ) )
        {
          Sample endpoint = probe( b );
          if ( !endpoint.valid ) return recover_from_invalid( guess, b, Real( 1 ) );
          if ( endpoint.Df <= 0 )
          {
            m_bracket_a = m_bracket_b = b;
            return finish( endpoint, true );
          }
          return solve_bracket( xguess, Dguess, b, endpoint.Df );
        }
        return expand_and_solve( guess, Real( 1 ) );
      }

      // Dguess>0: by monotonicity the minimum can only lie to the left.
      // At the lower bound f'(a)>=0 is precisely the lower-bound KKT test.
      if ( xguess == a )
      {
        m_bracket_a = m_bracket_b = a;
        return finish( guess, true );
      }
      if ( std::isfinite( a ) )
      {
        Sample endpoint = probe( a );
        if ( !endpoint.valid ) return recover_from_invalid( guess, a, Real( -1 ) );
        if ( endpoint.Df >= 0 )
        {
          m_bracket_a = m_bracket_b = a;
          return finish( endpoint, true );
        }
        return solve_bracket( a, endpoint.Df, xguess, Dguess );
      }
      return expand_and_solve( guess, Real( -1 ) );
    }

  public:
    Minimize_BBOX_1D()  = default;
    ~Minimize_BBOX_1D() = default;

    Minimize_BBOX_1D( Minimize_BBOX_1D const & )             = delete;
    Minimize_BBOX_1D & operator=( Minimize_BBOX_1D const & ) = delete;
    Minimize_BBOX_1D( Minimize_BBOX_1D && )                  = delete;
    Minimize_BBOX_1D & operator=( Minimize_BBOX_1D && )      = delete;

    /**
     * @brief Minimize using an object derived from Minimize_BBOX_1D_base_fun.
     * @details A deterministic finite guess is selected from the interval.
     */
    Real eval( Real a, Real b, Minimize_BBOX_1D_base_fun<Real> const * fun )
    {
      m_function = fun;
      return eval_impl( default_guess( a, b ), a, b );
    }

    /**
     * @brief Minimize from a user-supplied initial guess.
     * @param xguess Finite starting point; projected onto `[a,b]` if needed.
     * @param a Lower bound, possibly negative infinity.
     * @param b Upper bound, possibly positive infinity.
     * @param fun Objective/derivative object; it must remain valid for the call.
     * @return Computed minimizer.
     */
    Real eval( Real xguess, Real a, Real b, Minimize_BBOX_1D_base_fun<Real> const * fun )
    {
      m_function = fun;
      return eval_impl( xguess, a, b );
    }

    /** @brief Minimize using callable objects for f and f'. */
    template <typename PFUN, typename PFUN_D> Real eval2( Real a, Real b, PFUN && fun, PFUN_D && fun_D )
    {
      Minimize_BBOX_1D_fun<Real, std::decay_t<PFUN>, std::decay_t<PFUN_D>> wrapped(
        std::forward<PFUN>( fun ),
        std::forward<PFUN_D>( fun_D ) );
      m_function = &wrapped;
      return eval_impl( default_guess( a, b ), a, b );
    }

    /**
     * @brief Callable interface with a finite initial guess.
     * @param xguess Finite starting point; projected onto `[a,b]` if needed.
     * @param a Lower bound, possibly negative infinity.
     * @param b Upper bound, possibly positive infinity.
     * @param fun Callable implementing \f$f(x)\f$.
     * @param fun_D Callable implementing \f$f'(x)\f$.
     */
    template <typename PFUN, typename PFUN_D> Real eval2( Real xguess, Real a, Real b, PFUN && fun, PFUN_D && fun_D )
    {
      Minimize_BBOX_1D_fun<Real, std::decay_t<PFUN>, std::decay_t<PFUN_D>> wrapped(
        std::forward<PFUN>( fun ),
        std::forward<PFUN_D>( fun_D ) );
      m_function = &wrapped;
      return eval_impl( xguess, a, b );
    }

    /** @brief Set the only algorithmic parameter. */
    void set_max_iterations( Integer mit )
    {
      Utils::Check( mit > 0, "Minimize_BBOX_1D::set_max_iterations({}) argument must be > 0\n", mit );
      m_max_iteration = mit;
    }

    Integer max_iterations() const { return m_max_iteration; }
    Integer used_iter() const { return m_iteration_count; }
    Integer num_fun_eval() const { return m_fun_evaluation_count; }
    Integer num_fun_D_eval() const { return m_fun_D_evaluation_count; }

    //! True if the last call returned without converging specifically
    //! because m_max_iteration was exhausted (as opposed to, e.g., the
    //! outward expansion legitimately reaching the domain limit with no
    //! sign change found). Since tolerances are pinned near zero, a wide
    //! or poorly-conditioned bracket may need more than the default 200
    //! iterations to collapse to full floating-point precision: check this
    //! before assuming !converged() means no minimum exists.
    bool hit_max_iterations() const { return m_hit_max_iterations; }

    bool converged() const { return m_converged; }
    Real x_minimum() const { return m_x_min; }
    Real min_value() const { return m_f_min; }
    Real derivative() const { return m_Df_min; }
    Real bracket_a() const { return m_bracket_a; }
    Real bracket_b() const { return m_bracket_b; }
  };

}  // namespace Utils

#endif

// EOF: Utils_minimize_BBOX_1D.hh
