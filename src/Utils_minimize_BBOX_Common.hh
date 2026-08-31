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

#pragma once
#ifndef UTILS_MINIMIZE_BBOX_COMMON_DOT_HH
#define UTILS_MINIMIZE_BBOX_COMMON_DOT_HH

#include "Utils_eigen.hh"
#include "Utils_fmt.hh"
#include "Utils_ssolver.hh"
#include <type_traits>

#if EIGEN_MAJOR_VERSION < 5
#error "Utils BBOX Common requires Eigen 5 or newer"
#endif

namespace Utils
{

  // ===========================================================================
  // Common dense types — zero-copy via Eigen::Ref / Map
  // ===========================================================================
  template <typename Real> using Vector         = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
  template <typename Real> using Matrix         = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
  template <typename Real> using ConstVectorRef = Eigen::Ref<Vector<Real> const>;
  template <typename Real> using VectorRef      = Eigen::Ref<Vector<Real>>;
  template <typename Real> using ConstMatrixRef = Eigen::Ref<Matrix<Real> const>;
  template <typename Real> using MatrixRef      = Eigen::Ref<Matrix<Real>>;

  // ===========================================================================
  // Problem interface — f(x), grad(x,g), hess(x,H) dense with boolean return
  // ===========================================================================
  template <typename Problem, typename Real>
  concept ProblemFor = requires( Problem & p, ConstVectorRef<Real> x, VectorRef<Real> g, MatrixRef<Real> H, Real & f ) {
    { p.objective( x, f ) } -> std::convertible_to<bool>;
    { p.gradient( x, g ) } -> std::convertible_to<bool>;
    { p.hessian( x, H ) } -> std::convertible_to<bool>;
  };

  template <typename Real, typename Obj, typename Grad, typename Hess> class CallableProblem
  {
  public:
    CallableProblem( Obj obj, Grad grad, Hess hess )
      : m_obj( std::move( obj ) ), m_grad( std::move( grad ) ), m_hess( std::move( hess ) )
    {
    }
    bool objective( ConstVectorRef<Real> x, Real & f ) { return m_obj( x, f ); }
    bool gradient( ConstVectorRef<Real> x, VectorRef<Real> g ) { return m_grad( x, g ); }
    bool hessian( ConstVectorRef<Real> x, MatrixRef<Real> H ) { return m_hess( x, H ); }

  private:
    Obj  m_obj;
    Grad m_grad;
    Hess m_hess;
  };

  template <typename Real = double, typename Obj, typename Grad, typename Hess>
  auto make_problem( Obj && obj, Grad && grad, Hess && hess )
  {
    auto w_obj = [o = std::forward<Obj>( obj )]( ConstVectorRef<Real> x, Real & f ) -> bool {
      using DObj = std::decay_t<Obj>;
      if constexpr ( std::is_invocable_r_v<bool, DObj, ConstVectorRef<Real>, Real &> )
        return o( x, f );
      else if constexpr ( std::is_invocable_r_v<Real, DObj, ConstVectorRef<Real>> )
      {
        f = static_cast<Real>( o( x ) );
        return std::isfinite( f );
      }
      else
      {
        static_assert( sizeof( DObj ) == 0, "Objective must be bool(ConstVectorRef,Real&) or Real(ConstVectorRef)" );
        return false;
      }
    };
    auto w_grad = [g = std::forward<Grad>( grad )]( ConstVectorRef<Real> x, VectorRef<Real> gg ) -> bool {
      using DGrad = std::decay_t<Grad>;
      if constexpr ( std::is_invocable_r_v<bool, DGrad, ConstVectorRef<Real>, VectorRef<Real>> )
        return g( x, gg );
      else if constexpr ( std::is_invocable_r_v<void, DGrad, ConstVectorRef<Real>, VectorRef<Real>> )
      {
        g( x, gg );
        return gg.allFinite();
      }
      else
      {
        static_assert( sizeof( DGrad ) == 0, "Gradient must be bool(ConstVectorRef,VectorRef) or void(ConstVectorRef,VectorRef)" );
        return false;
      }
    };
    auto w_hess = [h = std::forward<Hess>( hess )]( ConstVectorRef<Real> x, MatrixRef<Real> H ) -> bool {
      using DHess = std::decay_t<Hess>;
      if constexpr ( std::is_invocable_r_v<bool, DHess, ConstVectorRef<Real>, MatrixRef<Real>> )
        return h( x, H );
      else if constexpr ( std::is_invocable_r_v<void, DHess, ConstVectorRef<Real>, MatrixRef<Real>> )
      {
        h( x, H );
        return H.allFinite();
      }
      else
      {
        static_assert( sizeof( DHess ) == 0, "Hessian must be bool(ConstVectorRef,MatrixRef) or void(ConstVectorRef,MatrixRef)" );
        return false;
      }
    };
    using WObj  = decltype( w_obj );
    using WGrad = decltype( w_grad );
    using WHess = decltype( w_hess );
    return CallableProblem<Real, WObj, WGrad, WHess>( std::move( w_obj ), std::move( w_grad ), std::move( w_hess ) );
  }

  // ===========================================================================
  // Common BBOX utilities — box projection, KKT, trust-region helpers
  // ===========================================================================
  namespace BBOXCommon
  {

    // P_[l,u](x) = cwiseMax(l).cwiseMin(u) — unico punto dove la scatola entra
    template <typename Z, typename X, typename L, typename U> inline void project(
      Eigen::MatrixBase<Z> &       z,
      Eigen::MatrixBase<X> const & x,
      Eigen::MatrixBase<L> const & l,
      Eigen::MatrixBase<U> const & u )
    { z = x.cwiseMax( l ).cwiseMin( u ); }

    template <typename S, typename X, typename D, typename L, typename U, typename Real> inline void project_step(
      Eigen::MatrixBase<S> &       s,
      Eigen::MatrixBase<X> const & x,
      Eigen::MatrixBase<D> const & d,
      Eigen::MatrixBase<L> const & l,
      Eigen::MatrixBase<U> const & u,
      Real                         alpha )
    { s = ( x + alpha * d ).cwiseMax( l ).cwiseMin( u ) - x; }

    template <typename Real> struct Breakpoints
    {
      Eigen::Index count = 0;
      Real         min   = Real( 0 );
      Real         max   = Real( 0 );
    };

    template <typename Real> [[nodiscard]] inline Breakpoints<Real> breakpoints(
      Vector<Real> const & x,
      Vector<Real> const & d,
      Vector<Real> const & l,
      Vector<Real> const & u )
    {
      constexpr Real    inf = std::numeric_limits<Real>::infinity();
      Breakpoints<Real> b{ 0, inf, Real( 0 ) };
      for ( Eigen::Index i = 0; i < x.size(); ++i )
      {
        Real t;
        if ( d[i] > Real( 0 ) )
          t = ( u[i] - x[i] ) / d[i];
        else if ( d[i] < Real( 0 ) )
          t = ( l[i] - x[i] ) / d[i];
        else
          continue;
        ++b.count;
        b.min = std::min( b.min, t );
        b.max = std::max( b.max, t );
      }
      if ( b.count == 0 ) return { 0, Real( 0 ), Real( 0 ) };
      return b;
    }

    template <typename Real> [[nodiscard]] inline bool is_strongly_active( Real xi, Real gg, Real li, Real ui )
    {
      constexpr Real eps   = std::numeric_limits<Real>::epsilon();
      Real const     scale = std::max(
        { Real( 1 ),
          std::isfinite( li ) ? std::abs( li ) : Real( 0 ),
          std::isfinite( ui ) ? std::abs( ui ) : Real( 0 ) } );
      Real const x_tol    = Real( 32 ) * eps * scale;
      Real const g_tol    = Real( 32 ) * eps;
      bool const fixed    = std::isfinite( li ) && std::isfinite( ui ) && ui - li <= x_tol;
      bool const at_lower = std::isfinite( li ) && xi <= li + x_tol;
      bool const at_upper = std::isfinite( ui ) && xi >= ui - x_tol;
      return fixed || ( at_lower && gg > g_tol ) || ( at_upper && gg < -g_tol );
    }

    enum class CurvatureStatus
    {
      certified,
      negative,
      failure
    };

    template <typename Real> [[nodiscard]] inline std::pair<Real, Real> hs_slope_qs(
      Matrix<Real> const & H,
      Vector<Real> const & s,
      Vector<Real> const & g,
      Vector<Real> &       Hs )
    {
      Hs.noalias()     = H * s;
      Real const slope = g.dot( s );
      Real const qs    = Real( 0.5 ) * s.dot( Hs ) + slope;
      return { slope, qs };
    }

    // p(x) = x - P(x-g) — misura KKT usata ovunque, P=I => p=g
    template <typename Real> inline void projected_gradient(
      Vector<Real> const & x,
      Vector<Real> const & g,
      Vector<Real> const & l,
      Vector<Real> const & u,
      Vector<Real> &       p )
    { p = x - ( x - g ).cwiseMax( l ).cwiseMin( u ); }

    [[nodiscard]] inline double bound_violation_auto( auto const & x, auto const & l, auto const & u )
    {
      using Real = typename std::decay_t<decltype( x )>::Scalar;
      return ( l - x ).cwiseMax( x - u ).cwiseMax( Real( 0 ) ).norm();
    }

    // Trust-region dense direct solver (SmallTRON style) — H symmetric, eigendecomp
    template <typename Real> [[nodiscard]] inline int direct_trust_region(
      Matrix<Real> const & H,
      Vector<Real> const & rhs,
      Real                 radius,
      Vector<Real> &       sol );

    template <typename Real> [[nodiscard]] inline int direct_trust_region(
      Matrix<Real> const & H,
      Vector<Real> const & rhs,
      Real                 radius,
      Vector<Real> &       sol )
    {
      using ESolver        = Eigen::SelfAdjointEigenSolver<Matrix<Real>>;
      Eigen::Index const n = rhs.size();
      sol.setZero( n );
      if ( n == 0 ) return 0;  // stationary
      if ( !H.allFinite() || !rhs.allFinite() || !std::isfinite( radius ) ) return -1;
      if ( radius <= Real( 0 ) ) return 1;  // boundary

      constexpr Real eps       = std::numeric_limits<Real>::epsilon();
      Real const     alg_tol   = Real( 256 ) * eps;
      Real const     mat_scale = std::max( Real( 1 ), H.cwiseAbs().maxCoeff() );

      Eigen::LDLT<Matrix<Real>> ldlt( H );
      if ( ldlt.info() == Eigen::Success && ldlt.isPositive() )
      {
        Vector<Real> ns = ldlt.solve( rhs );
        if ( ldlt.info() == Eigen::Success && ns.allFinite() )
        {
          Real const res = ( H * ns - rhs ).norm();
          Real const scl = Real( 1 ) + rhs.norm() + mat_scale * ns.norm();
          if ( res <= alg_tol * scl && ns.norm() <= radius * ( Real( 1 ) + alg_tol ) )
          {
            sol = std::move( ns );
            return 0;
          }
        }
      }
      ESolver es( H );
      if ( es.info() != Eigen::Success ) return -1;
      auto const & d = es.eigenvalues();
      auto const & Q = es.eigenvectors();
      if ( !d.allFinite() || !Q.allFinite() ) return -1;
      Vector<Real> c       = Q.transpose() * rhs;
      Real const   s_scale = std::max( Real( 1 ), d.cwiseAbs().maxCoeff() );
      Real const   s_tol   = Real( 64 ) * eps * s_scale * Real( std::max<Eigen::Index>( 1, n ) );
      Real const   lmin    = d[0];
      Vector<Real> s_step( n );
      if ( lmin > s_tol )
      {
        s_step = c.cwiseQuotient( d );
        if ( s_step.allFinite() && s_step.norm() <= radius * ( Real( 1 ) + alg_tol ) )
        {
          sol.noalias() = Q * s_step;
          return 0;
        }
      }
      bool const psd    = lmin >= -s_tol;
      Real const lshift = psd ? Real( 0 ) : -lmin;
      Real const ftol   = s_tol * ( Real( 1 ) + rhs.norm() );
      s_step.setZero();
      bool         sing  = false;
      Eigen::Index hard  = 0;
      bool         found = false;
      for ( Eigen::Index i = 0; i < n; ++i )
      {
        Real den = d[i] + lshift;
        if ( den > s_tol )
          s_step[i] = c[i] / den;
        else
        {
          if ( !found )
          {
            hard  = i;
            found = true;
          }
          sing = sing || std::abs( c[i] ) > ftol;
        }
      }
      Real const lnorm = s_step.norm();
      if ( !sing && lnorm <= radius * ( Real( 1 ) + alg_tol ) )
      {
        if ( psd )
        {
          sol.noalias() = Q * s_step;
          return 0;
        }
        Real rem2     = std::max( Real( 0 ), radius * radius - lnorm * lnorm );
        s_step[hard]  = std::copysign( std::sqrt( rem2 ), c[hard] == Real( 0 ) ? Real( 1 ) : c[hard] );
        sol.noalias() = Q * s_step;
        return 1;
      }
      auto snorm = [&]( Real sft )
      {
        Real s2 = Real( 0 );
        for ( Eigen::Index i = 0; i < n; ++i )
        {
          Real den = d[i] + sft;
          if ( den <= Real( 0 ) ) return std::numeric_limits<Real>::infinity();
          Real v = c[i] / den;
          s2 += v * v;
          if ( !std::isfinite( s2 ) ) return std::numeric_limits<Real>::infinity();
        }
        return std::sqrt( s2 );
      };
      Real lo = lshift, hi = std::max( Real( 1 ), lshift + Real( 1 ) );
      for ( int k = 0; snorm( hi ) > radius && k < 128; ++k )
      {
        Real nxt = Real( 2 ) * hi + Real( 1 );
        if ( !std::isfinite( nxt ) ) return -1;
        hi = nxt;
      }
      if ( snorm( hi ) > radius ) return -1;
      for ( int k = 0; k < 2 * std::numeric_limits<Real>::digits; ++k )
      {
        Real mid = lo + Real( 0.5 ) * ( hi - lo );
        if ( snorm( mid ) > radius )
          lo = mid;
        else
          hi = mid;
        if ( hi - lo <= alg_tol * ( Real( 1 ) + hi ) ) break;
      }
      for ( Eigen::Index i = 0; i < n; ++i ) s_step[i] = c[i] / ( d[i] + hi );
      if ( !s_step.allFinite() ) return -1;
      if ( s_step.norm() > radius ) s_step *= radius / s_step.norm();
      sol.noalias() = Q * s_step;
      return sol.allFinite() ? 1 : -1;
    }


  }  // namespace BBOXCommon

}  // namespace Utils

#endif  // UTILS_MINIMIZE_BBOX_COMMON_DOT_HH
