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

#include "Utils_TinyAD.hh"

#include <cmath>
#include <iostream>

namespace
{

  struct MacroExample
  {
    using real_type = double;

    template <typename T> static T f1( T const & x ) { return x * x; }

    template <typename T> static T f( T const & x, T const & y )
    {
      using std::sin;
      return x * x + 3.0 * x * y + sin( y );
    }

    template <typename T> static T f3( T const & x, T const & y, T const & z ) { return x * y + z * z; }
    template <typename T> static T f4( T const & a, T const & b, T const & c, T const & d ) { return a * b + c * d; }
    template <typename T> static T f5( T const & a, T const & b, T const & c, T const & d, T const & e )
    { return a * b + c * d + e * e; }
    template <typename T> static T f6( T const & a, T const & b, T const & c, T const & d, T const & e, T const & f )
    { return a * b + c * d + e * f; }

    UTILS_TINYAD_DERIV_1ARG( inline, static, f1_, f1, )
    UTILS_TINYAD_DERIV_2ARG( inline, static, f_, f, )
    UTILS_TINYAD_DERIV_3ARG( inline, static, f3_, f3, )
    UTILS_TINYAD_DERIV_4ARG( inline, static, f4_, f4, )
    UTILS_TINYAD_DERIV_5ARG( inline, static, f5_, f5, )
    UTILS_TINYAD_DERIV_6ARG( inline, static, f6_, f6, )
  };

  bool close_to( double actual, double expected )
  { return std::abs( actual - expected ) < 1e-12; }

}  // namespace

int main()
{
  constexpr double x = 0.7;
  constexpr double y = -0.4;
  const bool ok = close_to( MacroExample::f1_D( x ), 2.0 * x ) && close_to( MacroExample::f1_DD( x ), 2.0 ) &&
                  close_to( MacroExample::f_D_1( x, y ), 2.0 * x + 3.0 * y ) &&
                  close_to( MacroExample::f_D_2( x, y ), 3.0 * x + std::cos( y ) ) &&
                  close_to( MacroExample::f_D_1_1( x, y ), 2.0 ) && close_to( MacroExample::f_D_1_2( x, y ), 3.0 ) &&
                  close_to( MacroExample::f_D_2_2( x, y ), -std::sin( y ) ) &&
                  close_to( MacroExample::f3_D_3( x, y, 0.2 ), 0.4 ) &&
                  close_to( MacroExample::f4_D_1_2( x, y, 0.2, 0.3 ), 1.0 ) &&
                  close_to( MacroExample::f5_D_5_5( x, y, 0.2, 0.3, 0.4 ), 2.0 ) &&
                  close_to( MacroExample::f6_D_5_6( x, y, 0.2, 0.3, 0.4, 0.5 ), 1.0 );

  std::cout << "Testing UTILS_TINYAD_DERIV_2ARG: " << ( ok ? "PASS" : "FAIL" ) << '\n';
  return ok ? 0 : 1;
}
