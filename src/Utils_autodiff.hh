/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2017                                                      |
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

//
// file: Utils_autodiff.hh
//
#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef UTILS_AUTODIFF_dot_HH
#define UTILS_AUTODIFF_dot_HH

#include "Utils.hh"
#include "Utils_fmt.hh"

#ifdef _MSC_VER
  #pragma warning( disable : 4127 )
#endif

#include "Utils/3rd/autodiff/forward/dual.hpp"
#include "Utils/3rd/autodiff/forward/real.hpp"

namespace fmt {
  template <> struct formatter<autodiff::dual> : ostream_formatter {};
  template <> struct formatter<autodiff::dual2nd> : ostream_formatter {};
}

#include <type_traits>

#define UTILS_AUTODIFF_ADD_UNARY_FUNCTION(FUN) \
struct FUN##Op {};                             \
template<typename R, Requires<isExpr<R>> = true> AUTODIFF_DEVICE_FUNC constexpr auto FUN(R&& r) -> UnaryExpr<FUN##Op, R> { return { r }; } \
template<typename T, typename G> AUTODIFF_DEVICE_FUNC constexpr void apply(Dual<T, G>& self, FUN##Op)

namespace autodiff::detail {

  using std::erfc;

  template <size_t A, size_t... Rest>
  struct MaxN {
    static constexpr size_t value = [](){
      if constexpr (sizeof...(Rest) == 0) { return A; } 
      else { size_t max_rest = MaxN<Rest...>::value; return A > max_rest ? A : max_rest; }
    }();
  };

  template<typename T> using GetDual_t = HigherOrderDual<NumberTraits<T>::Order, typename NumberTraits<T>::NumericType>;

  template <typename T> constexpr auto to_dual( T const & x ) { return GetDual_t<T>(x); }

  // Caso base: nessun tipo (valore 0 per default)
  template <typename... Ts> struct DualOrder { static constexpr size_t value = 0; };
  template <typename T, typename... Ts>
  struct DualOrder<T, Ts...> {
    static constexpr size_t value = []() {
      if constexpr (sizeof...(Ts) == 0) {
        return NumberTraits<T>::Order;  // Caso singolo tipo
      } else {
        constexpr size_t current_order = NumberTraits<T>::Order;
        constexpr size_t rest_order    = DualOrder<Ts...>::value;
        return (current_order > rest_order) ? current_order : rest_order;
      }
    }();
  };

  /*
  //       _          _
  //   ___| |__  _ __| |_
  //  / __| '_ \| '__| __|
  // | (__| |_) | |  | |_
  //  \___|_.__/|_|   \__|
  */

  // missing code for cbrt
  struct CbrtOp{};  // CUBIC ROOT OPERATOR
  template<typename R> using CbrtExpr = UnaryExpr<CbrtOp, R>;
  template<typename R, Requires<isExpr<R>> = true> AUTODIFF_DEVICE_FUNC constexpr auto cbrt(R&& r) -> CbrtExpr<R> { return { r }; }

  template<typename T, typename G>
  AUTODIFF_DEVICE_FUNC constexpr void apply(Dual<T, G>& self, CbrtOp)
  {
    self.val = cbrt(self.val);
    self.grad *= 1 / (3*self.val*self.val);
  }

  /*
  //              __
  //    ___ _ __ / _| ___
  //   / _ \ '__| |_ / __|
  //  |  __/ |  |  _| (__
  //   \___|_|  |_|  \___|
  */

  // missing code for erfc
  struct ErfcOp{};  // ERROR FUNCTION OPERATOR
  template<typename R> using ErfcExpr = UnaryExpr<ErfcOp, R>;
  template<typename R, Requires<isExpr<R>> = true> AUTODIFF_DEVICE_FUNC constexpr auto erfc(R&& r) -> ErfcExpr<R> { return { r }; }
  
  template<typename T, typename G>
  AUTODIFF_DEVICE_FUNC constexpr void apply(Dual<T, G>& self, ErfcOp)
  {
    constexpr NumericType<T> sqrt_pi = 1.7724538509055160272981674833411451872554456638435;
    const T aux = self.val;
    self.val = erfc(aux);
    self.grad *= -2.0 * exp(-aux*aux)/sqrt_pi;
  }

  /*
  //                              _
  //   _ __ ___  _   _ _ __   __| |
  //  | '__/ _ \| | | | '_ \ / _` |
  //  | | | (_) | |_| | | | | (_| |
  //  |_|  \___/ \__,_|_| |_|\__,_| 
  //    
  */

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto round( Real<N, T> const & x ) {
    Real<N, T> res;
    res[0] = std::round(x[0]);
    if constexpr (N > 0) {
      For<1, N + 1>( [&](auto i) constexpr { res[i] = 0; } );
    }
    return res;
  }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(round) {
    using std::round;
    self.val  = round(val(self.val));
    self.grad = 0;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  round(T const& x) { return round(Real<0, T>{x})[0]; }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto floor( Real<N, T> const & x ) {
    Real<N, T> res;
    res[0] = std::floor(x[0]);
    if constexpr (N > 0) {
      For<1, N + 1>( [&](auto i) constexpr { res[i] = 0; } );
    }
    return res;
  }

  /*
  //    __ _
  //   / _| | ___   ___  _ __
  //  | |_| |/ _ \ / _ \| '__|
  //  |  _| | (_) | (_) | |
  //  |_| |_|\___/ \___/|_| 
  //    
  */

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(floor) {
    self.val  = std::floor(val(self.val));
    self.grad = 0;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  floor(T const& x) { return floor(Real<0, T>{x})[0]; }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto ceil( Real<N, T> const & x ) {
    Real<N, T> res;
    res[0] = std::ceil(x[0]);
    if constexpr (N > 0) {
      For<1, N + 1>( [&](auto i) constexpr { res[i] = 0; } );
    }
    return res;
  }

  /*
  //            _ _
  //    ___ ___(_) |
  //   / __/ _ \ | |
  //  | (_|  __/ | |
  //   \___\___|_|_|
  //   
  */

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(ceil) {
    using std::ceil;
    self.val  = ceil(val(self.val));
    self.grad = 0;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  ceil(T const& x) { return ceil(Real<0, T>{x})[0]; }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto log1p( Real<N, T> const & x ) {
    assert(x[0] != 1 && "autodiff::log(1+x) has undefined value and derivatives when x = -1");
    Real<N, T> log1px;
    log1px[0] = std::log1p(x[0]);
    T one_plus_x0 = T(1) + x[0];
    For<1, N + 1>([&](auto i) constexpr {
      log1px[i] = x[i] - Sum<1, i>( [&](auto j) constexpr {
        constexpr auto c = BinomialCoefficient<i.index - 1, j.index - 1>;
        return c * x[i - j] * log1px[j];
      } );
      log1px[i] /= one_plus_x0;
    });
    return log1px;
  }

  /*
  //   _             _
  //  | | ___   __ _/ |_ __
  //  | |/ _ \ / _` | | '_ \
  //  | | (_) | (_| | | |_) |
  //  |_|\___/ \__, |_| .__/
  //           |___/  |_|
  */

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(log1p) {
    using std::log1p;
    const T aux = One<T>() / (One<T>() + self.val); // 1 / (1 + x)
    self.val = log1p(self.val);                     // log(1 + x)
    self.grad *= aux;                               // grad * 1/(1 + x)
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  log1p(T const& x) { return log1p(Real<0, T>{x})[0]; }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(power2) {
    const T aux = 2*self.val;
    self.val  *= self.val;
    self.grad *= aux;
  }

  /*
  //  
  //   _ __   _____      _____ _ __
  //  | '_ \ / _ \ \ /\ / / _ \ '__|
  //  | |_) | (_) \ V  V /  __/ |
  //  | .__/ \___/ \_/\_/ \___|_|
  //  |_|
  */

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto power2( Real<N, T> const & x ) {
    Real<N, T> res;
    res[0] = x[0]*x[0];
    if constexpr (N > 1) { res[1] = 2*x[0]; }
    if constexpr (N > 2) { res[2] = 2; }
    if constexpr (N > 3) { For<3, N + 1>( [&](auto i) constexpr { res[i] = 0; } ); }
    return res;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  power2(T const& x) { return power2(Real<0, T>{x})[0]; }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(power3) {
    const T aux = self.val*self.val;
    self.val  *= aux;
    self.grad *= 3*aux;
  }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto power3( Real<N, T> const & x ) {
    Real<N, T> res;
    T x2{x[0]*x[0]};
    res[0] = x[0]*x2;
    if constexpr (N > 1) { res[1] = 3*x2; }
    if constexpr (N > 2) { res[2] = 6*x[0]; }
    if constexpr (N > 3) { res[3] = 6; }
    if constexpr (N > 4) { For<4, N + 1>( [&](auto i) constexpr { res[i] = 0; } ); }
    return res;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  power3(T const& x) { return power3(Real<0, T>{x})[0]; }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(power4) {
    const T aux3 = self.val*self.val*self.val;
    self.val  *= aux3;
    self.grad *= 4*aux3;
  }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto power4( Real<N, T> const & x ) {
    Real<N, T> res;
    T x2{x[0]*x[0]};
    res[0] = x2*x2;
    if constexpr (N > 1) { res[1] = 4*x2*x[0]; }
    if constexpr (N > 2) { res[2] = 12*x2; }
    if constexpr (N > 3) { res[3] = 24*x[0]; }
    if constexpr (N > 4) { res[4] = 24; }
    if constexpr (N > 5) { For<5, N + 1>( [&](auto i) constexpr { res[i] = 0; } ); }
    return res;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  power4(T const& x) { return power4(Real<0, T>{x})[0]; }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(power5) {
    const T aux2 = self.val*self.val;
    const T aux4 = aux2*aux2;
    self.val  *= aux4;
    self.grad *= 5*aux4;
  }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto power5( Real<N, T> const & x ) {
    Real<N, T> res;
    T x2{x[0]*x[0]};
    T x4{x2*x2};
    res[0] = x4*x2;
    if constexpr (N > 1) { res[1] = 5*x4; }
    if constexpr (N > 2) { res[2] = 20*x2*x[0]; }
    if constexpr (N > 3) { res[3] = 60*x2; }
    if constexpr (N > 4) { res[4] = 120*x[0]; }
    if constexpr (N > 5) { res[5] = 120; }
    if constexpr (N > 6) { For<6, N + 1>( [&](auto i) constexpr { res[i] = 0; } ); }
    return res;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  power5(T const& x) { return power5(Real<0, T>{x})[0]; }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(power6) {
    const T aux2 = self.val*self.val;
    const T aux5 = self.val*aux2*aux2;
    self.val  *= aux5;
    self.grad *= 6*aux5;
  }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto power6( Real<N, T> const & x ) {
    Real<N, T> res;
    T x2{x[0]*x[0]};
    T x4{x2*x2};
    res[0] = x4*x2;
    if constexpr (N > 1) { res[1] = 6*x4*x[0]; }
    if constexpr (N > 2) { res[2] = 30*x4; }
    if constexpr (N > 3) { res[3] = 120*x2*x[0]; }
    if constexpr (N > 4) { res[4] = 360*x2; }
    if constexpr (N > 5) { res[5] = 720*x[0]; }
    if constexpr (N > 6) { res[6] = 720; }
    if constexpr (N > 7) { For<7, N + 1>( [&](auto i) constexpr { res[i] = 0; } ); }
    return res;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  power6(T const& x) { return power6(Real<0, T>{x})[0]; }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(power7) {
    const T aux2 = self.val*self.val;
    const T aux6 = aux2*aux2*aux2;
    self.val  *= aux6;
    self.grad *= 7*aux6;
  }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto power7( Real<N, T> const & x ) {
    Real<N, T> res;
    T x2{x[0]*x[0]};
    T x4{x2*x2};
    res[0] = x4*x2*x[0];
    if constexpr (N > 1) { res[1] = 7*x4*x2; }
    if constexpr (N > 2) { res[2] = 42*x4*x[0]; }
    if constexpr (N > 3) { res[3] = 210*x4; }
    if constexpr (N > 4) { res[4] = 840*x2*x[0]; }
    if constexpr (N > 5) { res[5] = 2520*x2; }
    if constexpr (N > 6) { res[6] = 5040*x[0]; }
    if constexpr (N > 7) { res[7] = 5040; }
    if constexpr (N > 8) { For<8, N + 1>( [&](auto i) constexpr { res[i] = 0; } ); }
    return res;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  power7(T const& x) { return power7(Real<0, T>{x})[0]; }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(power8) {
    const T aux2 = self.val*self.val;
    const T aux4 = aux2*aux2;
    const T aux7 = aux4*aux2*self.val;
    self.val  *= aux7;
    self.grad *= 8*aux7;
  }

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto power8( Real<N, T> const & x ) {
    Real<N, T> res;
    T x2{x[0]*x[0]};
    T x4{x2*x2};
    res[0] = x4*x4;
    if constexpr (N > 1) { res[1] = 8*x4*x2*x[0]; }
    if constexpr (N > 2) { res[2] = 56*x4*x2; }
    if constexpr (N > 3) { res[3] = 336*x4*x[0]; }
    if constexpr (N > 4) { res[4] = 1680*x4; }
    if constexpr (N > 5) { res[5] = 6720*x2*x[0]; }
    if constexpr (N > 6) { res[6] = 20160*x2; }
    if constexpr (N > 7) { res[7] = 40320*x[0]; }
    if constexpr (N > 8) { res[8] = 40320; }
    if constexpr (N > 9) { For<9, N + 1>( [&](auto i) constexpr { res[i] = 0; } ); }
    return res;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  power8(T const& x) { return power8(Real<0, T>{x})[0]; }
}

namespace Utils {
  using autodiff::detail::round;
  using autodiff::detail::floor;
  using autodiff::detail::ceil;
  using autodiff::detail::log1p;
  using autodiff::detail::erfc;
  using autodiff::detail::power2;
  using autodiff::detail::power3;
  using autodiff::detail::power4;
  using autodiff::detail::power5;
  using autodiff::detail::power6;
  using autodiff::detail::power7;
}

#endif

#endif

//
// eof: Utils_autodiff.hh
//
