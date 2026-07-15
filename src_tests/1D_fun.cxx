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

#include <functional>
#include <memory>
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
          real_type res{ 0 };
          for ( int i{ 1 }; i <= 20; ++i ) res += power2( 2 * i - 5 ) / power3( x - i * i );
          return -2 * res;
        } )
  {
  }
};

static void build_1dfun_list( std::vector<std::unique_ptr<fun1D>> & f_list )
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
        real_type n{ 8 }, a{ 1 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=10, a=1",
      []( real_type x ) -> real_type
      {
        real_type n{ 10 }, a{ 1 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=12, a=1",
      []( real_type x ) -> real_type
      {
        real_type n{ 12 }, a{ 1 };
        return pow( x, n ) - a;
      } ) ) );

  f_list.emplace_back(
    std::unique_ptr<fun1D>( new fun1D(
      0,
      5,
      "f(x) = x^n-a, n=14, a=1",
      []( real_type x ) -> real_type
      {
        real_type n{ 14 }, a{ 1 };
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
