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

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Utils::m_pi;
using real_type = double;

static inline real_type power2( real_type x )
{ return x * x; }
static inline real_type power3( real_type x )
{ return x * x * x; }
static inline real_type power4( real_type x )
{ return power2( power2( x ) ); }
static inline real_type power5( real_type x )
{ return power4( x ) * x; }
static inline real_type power6( real_type x )
{ return power4( x ) * power2( x ); }
static inline real_type power7( real_type x )
{ return power4( x ) * power3( x ); }
static inline real_type power8( real_type x )
{ return power2( power4( x ) ); }

using FUN1D = std::function<real_type( real_type )>;

class fun1D
{
  real_type m_a0;
  real_type m_b0;
  FUN1D     m_fun;
  string    m_info;

public:
  fun1D() = delete;

  explicit fun1D( real_type a0, real_type b0, string_view info, FUN1D && f )
    : m_a0( a0 ), m_b0( b0 ), m_fun( f ), m_info( fmt::format( "{} ini:[{},{}]", info, a0, b0 ) )
  {
  }

  real_type operator()( real_type x ) const { return m_fun( x ); }
  real_type eval( real_type x ) const { return m_fun( x ); }
  real_type a0() const { return m_a0; }
  real_type b0() const { return m_b0; }
  string    info() const { return m_info; }
  FUN1D
  function() const { return m_fun; }
};

// Data used by test_Minimize1D.cc.  Every objective is differentiable and
// convex on the whole real line, so the unconstrained minimizer can be safely
// projected onto each test interval.
class min1D
{
  real_type m_a;
  real_type m_b;
  real_type m_x_min;
  FUN1D     m_fun;
  FUN1D     m_fun_D;
  string    m_info;

public:
  min1D() = delete;

  explicit min1D(
    real_type   a,
    real_type   b,
    real_type   unconstrained_minimum,
    string_view info,
    FUN1D &&    fun,
    FUN1D &&    fun_D )
    : m_a( a )
    , m_b( b )
    , m_x_min( std::clamp( unconstrained_minimum, a, b ) )
    , m_fun( std::move( fun ) )
    , m_fun_D( std::move( fun_D ) )
    , m_info( fmt::format( "{} on [{},{}]", info, a, b ) )
  {
  }

  real_type      a() const { return m_a; }
  real_type      b() const { return m_b; }
  real_type      x_min() const { return m_x_min; }
  real_type      eval( real_type x ) const { return m_fun( x ); }
  real_type      D( real_type x ) const { return m_fun_D( x ); }
  string const & info() const { return m_info; }
  FUN1D          function() const { return m_fun; }
  FUN1D          derivative() const { return m_fun_D; }
};

class fun1 : public fun1D
{
public:
  fun1( int i )
    : fun1D(
        power2( i ) + 1e-9,
        power2( i + 1 ) - 1e-9,
        "f(x) = -2*sum_{i=1}^20 (2*i-5)^2/(x-i^2)^3",
        []( real_type x ) -> real_type
        {
          real_type res = 0;
          for ( int i = 1; i <= 20; ++i ) res += power2( 2 * i - 5 ) / power3( x - i * i );
          return -2 * res;
        } )
  {
  }
};

[[maybe_unused]] static void build_1dfun_list( std::vector<std::unique_ptr<fun1D>> & f_list )
{
  f_list.clear();

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0, 6, "f(x) = |x-5|(x-5)", []( real_type x ) -> real_type { return abs( x - 5 ) * ( x - 5 ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -1,
      2,
      "f(x) = x^9",
      []( real_type x ) -> real_type
      {
        real_type x2{ x * x };
        real_type x4{ x2 * x2 };
        return x4 * x4 * x;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -1,
      1,
      "f(x) = 1/3+sign(x)|x|^(1/3)+x^3",
      []( real_type x ) -> real_type
      {
        real_type s{ real_type( x > 0 ? 1 : -1 ) };
        return 1.0 / 3.0 + power3( x ) + s * pow( abs( x ), 1.0 / 3.0 );
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0, 1.5, "f(x) = sin(x) - 1/2", []( real_type x ) -> real_type { return sin( x ) - 0.5; } ) ) );

  for ( int n : { 1, 5, 15, 20, 200 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        0,
        1,
        fmt::format( "f(x) = 2 * x * exp( -n ) - 2 * exp( -n * x )+1, n={}", n ),
        [n]( real_type x ) -> real_type { return 2 * x * exp( -n ) - 2 * exp( -n * x ) + 1; } ) ) );

  for ( int n : { 2, 5, 15, 20, 200 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        0,
        1,
        fmt::format( "f(x) = ( 1 + (1-n)^2 ) * x - ( 1 - n * x )^2, n={}", n ),
        [n]( real_type x ) -> real_type { return ( 1 + power2( 1 - n ) ) * x - power2( 1 - n * x ); } ) ) );

  for ( int i : { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } ) f_list.emplace_back( std::unique_ptr<fun1>( new fun1( i ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0.5, 5, "f(x) = log(x)", []( real_type x ) -> real_type { return log( x ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0.5,
      8,
      "f(x) = (10-x)*exp(-10*x)-pow(x,10)+1",
      []( real_type x ) -> real_type { return ( 10 - x ) * exp( -10 * x ) - pow( x, 10 ) + 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      1,
      4,
      "f(x) = exp(sin(x))-x-1",
      []( real_type x ) -> real_type { return exp( sin( x ) ) - x - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0.5, 1, "f(x) = 11*x^11-1", []( real_type x ) -> real_type { return 11 * pow( x, 11 ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0.1, m_pi / 3, "f(x) = 2*sin(x)-1", []( real_type x ) -> real_type { return 2 * sin( x ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      1,
      "f(x) = x^2+sin(x/10)-1/4",
      []( real_type x ) -> real_type { return power2( x ) + sin( x / 10 ) - 0.25; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0, 1.5, "f(x) = (x-1)*exp(x)", []( real_type x ) -> real_type { return ( x - 1 ) * exp( x ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0, 1.7, "f(x) = cos(x)-x", []( real_type x ) -> real_type { return cos( x ) - x; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 1.5, 3, "f(x) = (x-1)^3-1", []( real_type x ) -> real_type { return power3( x - 1 ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      2.6,
      3.5,
      "f(x) = exp(x^2+7*x-30)-1",
      []( real_type x ) -> real_type { return exp( x * x + 7 * x - 30 ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( -1.0, 1.0, "f(x) = tan( x-1/10 )", []( real_type x ) -> real_type { return tan( x - 0.1 ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0.0,
      1.0,
      "f(x) = tan( pi * ( x^8 - 1/2 ) )",
      []( real_type x ) -> real_type { return tan( m_pi * ( power8( x ) - 0.5 ) ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 1, 8, "f(x) = atan(x)-1", []( real_type x ) -> real_type { return atan( x ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0.2, 3, "f(x) = exp(x)-2*x-1", []( real_type x ) -> real_type { return exp( x ) - 2 * x - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      0.5,
      "f(x) = exp(-x)-x-sin(x)",
      []( real_type x ) -> real_type { return exp( -x ) - x - sin( x ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0.1, 1.5, "f(x) = x^3-1", []( real_type x ) -> real_type { return power3( x ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -1,
      2,
      "f(x) = x^2-sin(x)^2-1",
      []( real_type x ) -> real_type { return power2( x ) - power2( sin( x ) ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( -0.5, 1 / 3.0, "f(x) = x^3", []( real_type x ) -> real_type { return power3( x ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( -0.5, 1 / 3.0, "f(x) = x^5", []( real_type x ) -> real_type { return power5( x ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( -0.5, 1 / 3.0, "f(x) = x^9", []( real_type x ) -> real_type { return x * power8( x ); } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -1.0,
      1.0,
      "f(x) = x > 0 ? 1/(1-x) : x-1",
      []( real_type x ) -> real_type { return x > 0 ? 1 / ( 1 - x ) : x - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      m_pi / 2,
      m_pi,
      "f(x) = sin(x) - x/2",
      []( real_type x ) -> real_type { return sin( x ) - x / 2; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>(
      new fun1D( 0.0, 1.0, "f(x) = x * exp(x) - 1", []( real_type x ) -> real_type { return x * exp( x ) - 1; } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -9,
      31,
      "f(x) = a * x * exp( b * x ), a=-40, b=-1",
      []( real_type x ) -> real_type
      {
        real_type a{ -40 }, b{ -1 };
        return a * x * exp( b * x );
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -9,
      31,
      "f(x) = a * x * exp( b * x ), a=-100, b=-2",
      []( real_type x ) -> real_type
      {
        real_type a{ -100 }, b{ -2 };
        return a * x * exp( b * x );
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -9,
      31,
      "f(x) = a * x * exp( b * x ), a=-200, b=-3",
      []( real_type x ) -> real_type
      {
        real_type a{ -200 }, b{ -3 };
        return a * x * exp( b * x );
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=4, a=0.2",
      []( real_type x ) -> real_type
      {
        real_type n{ 4 }, a{ 0.2 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=6, a=0.2",
      []( real_type x ) -> real_type
      {
        real_type n{ 6 }, a{ 0.2 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=8, a=0.2",
      []( real_type x ) -> real_type
      {
        real_type n{ 8 }, a{ 0.2 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=10, a=0.2",
      []( real_type x ) -> real_type
      {
        real_type n{ 10 }, a{ 0.2 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=12, a=0.2",
      []( real_type x ) -> real_type
      {
        real_type n{ 12 }, a{ 0.2 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=8, a=1",
      []( real_type x ) -> real_type
      {
        real_type n{ 8 }, a = 1;
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=10, a=1",
      []( real_type x ) -> real_type
      {
        real_type n{ 10 }, a = 1;
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=12, a=1",
      []( real_type x ) -> real_type
      {
        real_type n{ 12 }, a = 1;
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=14, a=1",
      []( real_type x ) -> real_type
      {
        real_type n{ 14 }, a = 1;
        return pow( x, n ) - a;
      } ) ) );

  for ( int n : { 2, 5, 10, 15, 20 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        0,
        1,
        fmt::format( "f(x) = x^2-(1-x)^n, n={}", n ),
        [n]( real_type x ) -> real_type { return power2( x ) - pow( 1 - x, n ); } ) ) );

  for ( int n : { 1, 2, 3, 5, 8, 15, 20 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        0,
        1,
        fmt::format( "f(x) = (1+(1-n)^4)*x-(1-n*x)^4, n={}", n ),
        [n]( real_type x ) -> real_type { return ( 1 + power4( 1 - n ) ) * x - power4( 1 - n * x ); } ) ) );

  for ( int n : { 1, 5, 10, 15, 20 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        0,
        1,
        fmt::format( "f(x) = exp(-n*x)*(x-1)+x^n, n={}", n ),
        [n]( real_type x ) -> real_type { return exp( -n * x ) * ( x - 1 ) + pow( x, n ); } ) ) );

  for ( int n : { 2, 5, 10, 15, 20 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        0.01,
        1,
        fmt::format( "f(x) = (n*x-1)/((n-1)*x), n={}", n ),
        [n]( real_type x ) -> real_type { return ( n * x - 1 ) / ( ( n - 1 ) * x ); } ) ) );

  for ( int n : { 2, 3, 6, 9, 11, 15, 20, 25, 33 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        0,
        100,
        fmt::format( "f(x) = x^(1/n)-n^(1/n), n={}", n ),
        [n]( real_type x ) -> real_type
        {
          real_type p{ real_type( 1.0 / n ) };
          return pow( x, p ) - pow( n, p );
        } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      -1,
      4,
      "f(x) = x == 0 ? 0 : x*exp(-1/x^2)",
      []( real_type x ) -> real_type { return x == 0 ? 0 : x * exp( -1 / ( x * x ) ); } ) ) );

  for ( int n : { 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        -1e4,
        m_pi / 2,
        fmt::format( "f(x) = x < 0 ? -n/20 : (n/20)*(x/1.5+sin(x)-1), n={}", n ),
        [n]( real_type x ) -> real_type
        {
          if ( x < 0 ) return -n / 20.0;
          return ( n / 20.0 ) * ( x / 1.5 + sin( x ) - 1 );
        } ) ) );

  for ( int n : { 20, 30, 40, 100, 200, 400, 600, 800, 1000 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        -1e4,
        1e-4,
        fmt::format( "f(x) = [exp(1)-1.859,-0.859,exp( (n+1)*0.5e3*x )-1.859], n={}", n ),
        [n]( real_type x ) -> real_type
        {
          if ( x > 2e-3 / ( 1 + n ) ) return exp( 1 ) - 1.859;
          if ( x < 0 ) return -0.859;
          return exp( ( n + 1 ) * 500 * x ) - 1.859;
        } ) ) );


  // -----------------------------------------------------------------------
  // Additional classical/pathological scalar root-finding benchmarks.
  // The intervals are chosen to contain at least one zero.  The collection
  // deliberately mixes simple, multiple, flat, endpoint, oscillatory and
  // badly scaled roots.
  // -----------------------------------------------------------------------

  auto add_zero = [&f_list]( real_type a, real_type b, string_view info, FUN1D fun )
  { f_list.emplace_back( std::unique_ptr<fun1D>( new fun1D( a, b, info, std::move( fun ) ) ) ); };

  add_zero( 0, 2, "Dekker-Brent: f(x)=x^3+x-1", []( real_type x ) { return power3( x ) + x - 1; } );
  add_zero( 0, 1, "f(x)=exp(-x)-x", []( real_type x ) { return std::exp( -x ) - x; } );
  add_zero( 1, 2, "f(x)=x^3-2*x-5", []( real_type x ) { return power3( x ) - 2 * x - 5; } );
  add_zero( 2, 3, "f(x)=log(x)+x^2-3", []( real_type x ) { return std::log( x ) + power2( x ) - 3; } );
  add_zero( 0, 1, "f(x)=cos(x)-x^3", []( real_type x ) { return std::cos( x ) - power3( x ); } );
  add_zero( 1, 2, "f(x)=x^3-2", []( real_type x ) { return power3( x ) - 2; } );
  add_zero( 0, 2, "f(x)=x^5-x-1", []( real_type x ) { return power5( x ) - x - 1; } );
  add_zero( 0, 2, "f(x)=exp(x)-3", []( real_type x ) { return std::exp( x ) - 3; } );
  add_zero( 0, 2, "f(x)=log(1+x)-1/2", []( real_type x ) { return std::log1p( x ) - 0.5; } );
  add_zero( 0, 2, "f(x)=sinh(x)-1", []( real_type x ) { return std::sinh( x ) - 1; } );
  add_zero( 0, 2, "f(x)=tanh(x)-1/2", []( real_type x ) { return std::tanh( x ) - 0.5; } );

  // Multiple and very flat roots: difficult for methods relying on local slope.
  for ( int n : { 2, 3, 4, 5, 8, 12 } )
    add_zero(
      0,
      2,
      fmt::format( "multiple root: f(x)=(x-1)^{}, root=1", n ),
      [n]( real_type x ) { return std::pow( x - 1, n ); } );

  for ( int n : { 3, 5, 7, 9, 15 } )
    add_zero(
      -1,
      1,
      fmt::format( "flat sign-changing root: f(x)=x^{}, root=0", n ),
      [n]( real_type x ) { return std::pow( x, n ); } );

  add_zero(
    -1,
    1,
    "C-infinity flat root: f(x)=sign(x)*exp(-1/x^2)",
    []( real_type x ) { return x == 0 ? 0.0 : std::copysign( std::exp( -1 / power2( x ) ), x ); } );

  // Roots very close to an endpoint.
  for ( real_type eps : { 1e-2, 1e-6, 1e-12 } )
    add_zero(
      0,
      1,
      fmt::format( "near-left-endpoint root: x-eps, eps={:.1e}", eps ),
      [eps]( real_type x ) { return x - eps; } );

  add_zero( 0, 1, "endpoint root: f(x)=x", []( real_type x ) { return x; } );
  add_zero( 0, 1, "endpoint root: f(x)=x-1", []( real_type x ) { return x - 1; } );

  // Strongly unbalanced function values across the bracket.
  for ( int k : { 2, 6, 12, 20 } )
  {
    real_type scale{ std::pow( 10.0, k ) };
    add_zero(
      0,
      2,
      fmt::format( "bad scaling: exp({}*(x-1))-1", k ),
      [scale, k]( real_type x ) { return std::exp( k * ( x - 1 ) ) - 1; } );
    add_zero( 0, 2, fmt::format( "bad scaling: 1e{}*(x-1)", k ), [scale]( real_type x ) { return scale * ( x - 1 ); } );
  }

  // Oscillatory problems.  Brackets contain several roots on purpose.
  for ( int n : { 5, 20, 100 } )
    add_zero(
      0.01,
      1,
      fmt::format( "oscillatory: sin({}*x), multiple zeros", n ),
      [n]( real_type x ) { return std::sin( n * x ); } );

  add_zero( 0.1, 1, "oscillatory: sin(1/x), multiple zeros", []( real_type x ) { return std::sin( 1 / x ); } );

  // Transcendental equations frequently used as textbook root tests.
  add_zero( 0, 1, "Kepler-like: x-0.9*sin(x)-0.1", []( real_type x ) { return x - 0.9 * std::sin( x ) - 0.1; } );
  add_zero( 0.1, 2, "Colebrook-like core: x+2*log10(x)-1", []( real_type x ) { return x + 2 * std::log10( x ) - 1; } );
  add_zero( 0.1, 2, "f(x)=x*log(x)-1", []( real_type x ) { return x * std::log( x ) - 1; } );
  add_zero( 0, 2, "f(x)=x*exp(x)-2", []( real_type x ) { return x * std::exp( x ) - 2; } );
  add_zero( 0, 2, "f(x)=erf(x)-1/2", []( real_type x ) { return std::erf( x ) - 0.5; } );
  add_zero( 0, 4, "f(x)=lgamma(x+1)-1", []( real_type x ) { return std::lgamma( x + 1 ) - 1; } );

  // Cancellation-sensitive forms around the root.
  add_zero( -1, 1, "cancellation: expm1(x), root=0", []( real_type x ) { return std::expm1( x ); } );
  add_zero( -0.5, 1, "cancellation: log1p(x), root=0", []( real_type x ) { return std::log1p( x ); } );
  add_zero( -1, 1, "cancellation: sin(x), root=0", []( real_type x ) { return std::sin( x ); } );

  for ( real_type RHS : { -229.970950036057, 0.0, 10.0 } )
    f_list.emplace_back(
      std::unique_ptr<fun1D>( new fun1D(
        -100,
        100,
        fmt::format( "f(x) = penalty(x) RHS={}", RHS ),
        [RHS]( real_type x_in ) -> real_type
        {
          real_type m_h       = 0.01;
          real_type m_epsilon = 0.01;
          real_type m_A       = 1 / m_h;
          real_type m_A1      = ( 1 - m_epsilon ) * power2( m_h / ( 1 - m_h ) );
          real_type x         = abs( x_in );
          real_type Xh        = x / m_h;
          real_type res       = 2 * m_epsilon * Xh;
          if ( Xh > 1 ) res += 2 * m_A1 * ( Xh - 1 );
          res /= m_h;
          if ( x > 1 ) res += 2 * m_A * ( x - 1 );
          return ( x_in < 0 ? -res : res ) - RHS;
        } ) ) );
}

[[maybe_unused]] static void build_1dmin_list( std::vector<std::unique_ptr<min1D>> & f_list )
{
  f_list.clear();

  real_type const inf{ std::numeric_limits<real_type>::infinity() };

  auto add_problem = [&f_list, inf]( string_view info, real_type x_min, FUN1D fun, FUN1D fun_D )
  {
    for ( real_type a : { real_type( -2 ), -inf } )
      for ( real_type b : { real_type( 1 ), inf } )
        f_list.emplace_back( std::unique_ptr<min1D>( new min1D( a, b, x_min, info, FUN1D( fun ), FUN1D( fun_D ) ) ) );
  };

  add_problem(
    "quadratic, x*=1/4",
    0.25,
    []( real_type x ) { return power2( x - 0.25 ) + 1; },
    []( real_type x ) { return 2 * ( x - 0.25 ); } );

  add_problem(
    "quadratic, x*=-4 (left-bound KKT cases)",
    -4,
    []( real_type x ) { return power2( x + 4 ) - 3; },
    []( real_type x ) { return 2 * ( x + 4 ); } );

  add_problem(
    "quadratic, x*=3 (right-bound KKT cases)",
    3,
    []( real_type x ) { return 0.5 * power2( x - 3 ) + 2; },
    []( real_type x ) { return x - 3; } );

  real_type const flat_min{ std::sqrt( 2.0 ) - 1 };
  add_problem(
    "flat quartic, x*=sqrt(2)-1",
    flat_min,
    [flat_min]( real_type x ) { return power4( x - flat_min ) + 0.125; },
    [flat_min]( real_type x ) { return 4 * power3( x - flat_min ); } );

  real_type const very_flat_min{ -m_pi };
  add_problem(
    "very flat degree-8 polynomial, x*=-pi",
    very_flat_min,
    [very_flat_min]( real_type x ) { return power8( x - very_flat_min ) + 1; },
    [very_flat_min]( real_type x ) { return 8 * power7( x - very_flat_min ); } );

  real_type const rounded_min{ 2.5 };
  add_problem(
    "rounded absolute value, x*=5/2",
    rounded_min,
    [rounded_min]( real_type x ) { return std::hypot( x - rounded_min, 1e-3 ); },
    [rounded_min]( real_type x ) { return ( x - rounded_min ) / std::hypot( x - rounded_min, 1e-3 ); } );

  add_problem(
    "exponential quadratic-free, f=exp(x)-2*x",
    std::log( 2.0 ),
    []( real_type x ) { return std::exp( x ) - 2 * x; },
    []( real_type x ) { return std::exp( x ) - 2; } );

  for ( real_type x_min : { real_type( -4.25 ), real_type( 2.75 ) } )
  {
    real_type slope{ std::exp( x_min ) };
    add_problem(
      fmt::format( "shifted exponential, x*={}", x_min ),
      x_min,
      [slope]( real_type x ) { return std::exp( x ) - slope * x; },
      [slope]( real_type x ) { return std::exp( x ) - slope; } );
  }

  real_type const close_to_upper{ std::nextafter( real_type( 1 ), real_type( 0 ) ) };
  add_problem(
    "minimum one ulp below upper bound",
    close_to_upper,
    [close_to_upper]( real_type x ) { return power2( x - close_to_upper ); },
    [close_to_upper]( real_type x ) { return 2 * ( x - close_to_upper ); } );

  real_type const close_to_lower{ std::nextafter( real_type( -2 ), real_type( 0 ) ) };
  add_problem(
    "minimum one ulp above lower bound",
    close_to_lower,
    [close_to_lower]( real_type x ) { return power2( x - close_to_lower ); },
    [close_to_lower]( real_type x ) { return 2 * ( x - close_to_lower ); } );


  // -----------------------------------------------------------------------
  // Additional smooth convex minimization benchmarks.  All problems satisfy
  // the assumptions of min1D: differentiable and convex on R, with a known
  // unconstrained minimizer that can be projected onto [a,b].
  // -----------------------------------------------------------------------

  add_problem(
    "cosh bowl, x*=0.3",
    0.3,
    []( real_type x ) { return std::cosh( x - 0.3 ); },
    []( real_type x ) { return std::sinh( x - 0.3 ); } );

  add_problem(
    "log-cosh, x*=-0.7",
    -0.7,
    []( real_type x ) { return std::log( std::cosh( x + 0.7 ) ); },
    []( real_type x ) { return std::tanh( x + 0.7 ); } );

  add_problem(
    "sqrt bowl, x*=0.4, eps=1e-6",
    0.4,
    []( real_type x ) { return std::hypot( x - 0.4, 1e-6 ); },
    []( real_type x ) { return ( x - 0.4 ) / std::hypot( x - 0.4, 1e-6 ); } );

  // f(x)=exp(x)+exp(-x) has its minimum at zero and rapidly growing tails.
  add_problem(
    "symmetric exponential, f=exp(x)+exp(-x), x*=0",
    0,
    []( real_type x ) { return std::exp( x ) + std::exp( -x ); },
    []( real_type x ) { return std::exp( x ) - std::exp( -x ); } );

  // exp(x)+exp(-2x): exp(x)=2 exp(-2x), hence x*=log(2)/3.
  real_type const exp_mix_min = std::log( 2.0 ) / 3.0;
  add_problem(
    "asymmetric exponential, exp(x)+exp(-2*x)",
    exp_mix_min,
    []( real_type x ) { return std::exp( x ) + std::exp( -2 * x ); },
    []( real_type x ) { return std::exp( x ) - 2 * std::exp( -2 * x ); } );

  // x^2 + exp(-x): 2*x=exp(-x), solution x=W(1/2), numerical constant.
  real_type const quad_exp_min{ 0.35173371124919584 };
  add_problem(
    "quadratic plus exponential, x^2+exp(-x)",
    quad_exp_min,
    []( real_type x ) { return power2( x ) + std::exp( -x ); },
    []( real_type x ) { return 2 * x - std::exp( -x ); } );

  // Softplus minus p*x is strictly convex; minimizer is log(p/(1-p)).
  for ( real_type p : { 0.01, 0.1, 0.5, 0.9, 0.99 } )
  {
    real_type const xmin{ std::log( p / ( 1 - p ) ) };
    add_problem(
      fmt::format( "softplus-p*x, p={}", p ),
      xmin,
      [p]( real_type x )
      {
        // Stable softplus.
        real_type sp{ x > 0 ? x + std::log1p( std::exp( -x ) ) : std::log1p( std::exp( x ) ) };
        return sp - p * x;
      },
      [p]( real_type x )
      {
        real_type sigmoid{ x >= 0 ? 1 / ( 1 + std::exp( -x ) ) : std::exp( x ) / ( 1 + std::exp( x ) ) };
        return sigmoid - p;
      } );
  }

  // Increasing even powers give progressively flatter minima.
  for ( int n : { 2, 4, 6, 8 } )
  {
    real_type const xmin{ 0.125 };
    add_problem(
      fmt::format( "even-power bowl degree {}, x*=1/8", n ),
      xmin,
      [xmin, n]( real_type x ) { return std::pow( x - xmin, n ); },
      [xmin, n]( real_type x ) { return n * std::pow( x - xmin, n - 1 ); } );
  }

  // Ill-conditioned quadratic bowls: same minimizer, widely different scale.
  for ( real_type scale : { 1e-12, 1e-6, 1.0, 1e6, 1e12 } )
  {
    real_type const xmin{ -0.375 };
    add_problem(
      fmt::format( "scaled quadratic, scale={:.1e}", scale ),
      xmin,
      [xmin, scale]( real_type x ) { return scale * power2( x - xmin ); },
      [xmin, scale]( real_type x ) { return 2 * scale * ( x - xmin ); } );
  }

  // A quadratic with a tiny curvature plus a quartic term tests transition
  // between nearly flat and strongly curved regions.
  for ( real_type eps : { 1e-12, 1e-8, 1e-4, 1e-2 } )
  {
    real_type const xmin{ 0.6 };
    add_problem(
      fmt::format( "quartic + eps quadratic, eps={:.1e}", eps ),
      xmin,
      [xmin, eps]( real_type x )
      {
        real_type d{ x - xmin };
        return power4( d ) + eps * power2( d );
      },
      [xmin, eps]( real_type x )
      {
        real_type d{ x - xmin };
        return 4 * power3( d ) + 2 * eps * d;
      } );
  }

  // Pseudo-Huber loss: smooth approximation of |x-x*|.
  for ( real_type delta : { 1e-6, 1e-3, 1.0 } )
  {
    real_type const xmin{ -0.2 };
    add_problem(
      fmt::format( "pseudo-Huber, delta={:.1e}", delta ),
      xmin,
      [xmin, delta]( real_type x )
      {
        real_type d{ ( x - xmin ) / delta };
        return delta * delta * ( std::hypot( 1.0, d ) - 1 );
      },
      [xmin, delta]( real_type x )
      {
        real_type d{ ( x - xmin ) / delta };
        return ( x - xmin ) / std::hypot( 1.0, d );
      } );
  }
}

// static
// real_type
// fun_penalty( real_type x_in, real_type RHS ) {
//   real_type m_h       = 0.01;
//   real_type m_epsilon = 0.01;
//   real_type m_A       = 1/m_h;
//   real_type m_A1      = (1-m_epsilon)*power2(m_h/(1-m_h));
//   real_type x         = abs(x_in);
//   real_type Xh        = x/m_h;
//   real_type res       = 2*m_epsilon*Xh;
//   if ( Xh > 1 ) res += 2*m_A1 * (Xh-1);
//   res /= m_h;
//   if ( x > 1 ) res += 2*m_A * (x-1);
//   return (x_in < 0 ? -res : res) - RHS;
// }
