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
 * @file Utils_Minimize1D.hh
 * @brief Header-only minimization of a differentiable scalar function on an
 *        interval whose endpoints may be infinite.
 */

#pragma once

#ifndef UTILS_MINIMIZE_1D_dot_HH
#define UTILS_MINIMIZE_1D_dot_HH

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
  template <typename Real> class Minimize1D_base_fun
  {
  public:
    virtual ~Minimize1D_base_fun() = default;

    virtual Real eval( Real x ) const = 0;
    virtual Real D( Real x ) const    = 0;
  };

  /** @brief Adapter for callable objective and derivative objects. */
  template <typename Real, typename PFUN, typename PFUN_D> class Minimize1D_fun : public Minimize1D_base_fun<Real>
  {
    PFUN   m_fun;
    PFUN_D m_fun_D;

  public:
    template <typename F, typename FD> Minimize1D_fun( F && fun, FD && fun_D )
      : m_fun( std::forward<F>( fun ) ), m_fun_D( std::forward<FD>( fun_D ) )
    {
    }

    Real eval( Real x ) const override { return m_fun( x ); }
    Real D( Real x ) const override { return m_fun_D( x ); }
  };

  /**
   * @brief One-dimensional minimizer for a differentiable convex function.
   *
   * The method first checks the KKT conditions at every finite boundary.  If
   * neither boundary is optimal, it brackets a negative-to-positive sign
   * change of f'(x).  On an unbounded side the bracket is enlarged
   * geometrically, without evaluating at infinity.  The zero of f' is then
   * computed by Utils::AlgoBracket using Brent's method.
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
  template <typename Real> class Minimize1D
  {
  public:
    using Integer = int;

    static_assert(
      std::is_same_v<Real, float> || std::is_same_v<Real, double>,
      "Minimize1D<Real> requires Real to be float or double" );

  private:
    using Limits = std::numeric_limits<Real>;

    Minimize1D_base_fun<Real> const * m_function = nullptr;
    AlgoBracket<Real>                 m_bracket;

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
      Real fx = m_function->eval( x );
      Utils::Check( !std::isnan( fx ), "Minimize1D::eval(), f({}) is NaN\n", x );
      return fx;
    }

    Real evaluate_D( Real x )
    {
      ++m_fun_D_evaluation_count;
      Real dfx = m_function->D( x );
      Utils::Check( !std::isnan( dfx ), "Minimize1D::eval(), f'({}) is NaN\n", x );
      return dfx;
    }

    Real finish( Real x, Real dfx, bool ok )
    {
      m_x_min     = x;
      m_Df_min    = dfx;
      m_f_min     = evaluate( x );
      m_converged = ok;
      return x;
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

        Real Dc = evaluate_D( c );
        ++m_iteration_count;
        if ( Dc == 0 )
        {
          m_bracket_a = m_bracket_b = c;
          return finish( c, Dc, true );
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
      m_bracket.select( 3 );  // Brent: includes a scale-aware machine-precision stopping test.
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

    Real expand_and_solve( Real x, Real Dx, Real direction )
    {
      Real step             = initial_step( x, direction );
      bool budget_exhausted = true;

      while ( m_iteration_count < m_max_iteration )
      {
        Real next = expanded_point( x, direction, step );
        if ( next == x )
        {
          budget_exhausted = false;  // expansion saturated at the domain limit, not the iteration cap
          break;
        }

        Real Dnext = evaluate_D( next );
        ++m_iteration_count;

        if ( direction > 0 )
        {
          m_bracket_a = x;
          m_bracket_b = next;
        }
        else
        {
          m_bracket_a = next;
          m_bracket_b = x;
        }

        if ( Dnext == 0 )
        {
          m_bracket_a = m_bracket_b = next;
          return finish( next, Dnext, true );
        }

        if ( direction > 0 )
        {
          if ( Dnext > 0 ) return solve_bracket( x, Dx, next, Dnext );
        }
        else
        {
          if ( Dnext < 0 ) return solve_bracket( next, Dnext, x, Dx );
        }

        x  = next;
        Dx = Dnext;
        double_step( step );
      }

      if ( budget_exhausted ) m_hit_max_iterations = true;
      return finish( x, Dx, false );
    }

    Real eval_impl( Real a, Real b )
    {
      m_iteration_count        = 0;
      m_fun_evaluation_count   = 0;
      m_fun_D_evaluation_count = 0;
      m_converged              = false;
      m_hit_max_iterations     = false;

      Utils::Check( m_function != nullptr, "Minimize1D::eval(), function pointer is null\n" );
      Utils::Check(
        !std::isnan( a ) && !std::isnan( b ) && a < b,
        "Minimize1D::eval(a={}, b={}), expected a < b and non-NaN bounds\n",
        a,
        b );

      m_bracket_a = a;
      m_bracket_b = b;

      bool finite_a = std::isfinite( a );
      bool finite_b = std::isfinite( b );

      Real Da = 0;
      Real Db = 0;

      // KKT at the lower bound: f'(a) >= 0.
      if ( finite_a )
      {
        Da = evaluate_D( a );
        if ( Da >= 0 )
        {
          m_bracket_a = m_bracket_b = a;
          return finish( a, Da, true );
        }
      }

      // KKT at the upper bound: f'(b) <= 0.
      if ( finite_b )
      {
        Db = evaluate_D( b );
        if ( Db <= 0 )
        {
          m_bracket_a = m_bracket_b = b;
          return finish( b, Db, true );
        }
      }

      // With two finite endpoints the failed boundary KKT checks imply
      // f'(a) < 0 < f'(b), hence the stationary minimum is bracketed.
      if ( finite_a && finite_b ) return solve_bracket( a, Da, b, Db );

      // On a half-line start at its finite boundary and expand inward.
      if ( finite_a ) return expand_and_solve( a, Da, Real( 1 ) );
      if ( finite_b ) return expand_and_solve( b, Db, Real( -1 ) );

      // On the whole real line zero is a neutral, deterministic initial point.
      Real x  = 0;
      Real Dx = evaluate_D( x );
      if ( Dx == 0 )
      {
        m_bracket_a = m_bracket_b = x;
        return finish( x, Dx, true );
      }
      return expand_and_solve( x, Dx, Dx < 0 ? Real( 1 ) : Real( -1 ) );
    }

  public:
    Minimize1D()  = default;
    ~Minimize1D() = default;

    Minimize1D( Minimize1D const & )             = delete;
    Minimize1D & operator=( Minimize1D const & ) = delete;
    Minimize1D( Minimize1D && )                  = delete;
    Minimize1D & operator=( Minimize1D && )      = delete;

    /** @brief Minimize using an object derived from Minimize1D_base_fun. */
    Real eval( Real a, Real b, Minimize1D_base_fun<Real> const * fun )
    {
      m_function = fun;
      return eval_impl( a, b );
    }

    /** @brief Minimize using callable objects for f and f'. */
    template <typename PFUN, typename PFUN_D> Real eval2( Real a, Real b, PFUN && fun, PFUN_D && fun_D )
    {
      Minimize1D_fun<Real, std::decay_t<PFUN>, std::decay_t<PFUN_D>> wrapped(
        std::forward<PFUN>( fun ),
        std::forward<PFUN_D>( fun_D ) );
      m_function = &wrapped;
      return eval_impl( a, b );
    }

    /** @brief Set the only algorithmic parameter. */
    void set_max_iterations( Integer mit )
    {
      Utils::Check( mit > 0, "Minimize1D::set_max_iterations({}) argument must be > 0\n", mit );
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

// EOF: Utils_Minimize1D.hh
