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

  /*
  //    __ _
  //   / _| | ___   ___  _ __
  //  | |_| |/ _ \ / _ \| '__|
  //  |  _| | (_) | (_) | |
  //  |_| |_|\___/ \___/|_| 
  //    
  */

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto floor( Real<N, T> const & x ) {
    Real<N, T> res;
    res[0] = std::floor(x[0]);
    if constexpr (N > 0) {
      For<1, N + 1>( [&](auto i) constexpr { res[i] = 0; } );
    }
    return res;
  }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(floor) {
    self.val  = std::floor(val(self.val));
    self.grad = 0;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  floor(T const& x) { return floor(Real<0, T>{x})[0]; }

  /*
  //            _ _
  //    ___ ___(_) |
  //   / __/ _ \ | |
  //  | (_|  __/ | |
  //   \___\___|_|_|
  //   
  */

  template<size_t N, typename T>
  AUTODIFF_DEVICE_FUNC constexpr auto ceil( Real<N, T> const & x ) {
    Real<N, T> res;
    res[0] = std::ceil(x[0]);
    if constexpr (N > 0) {
      For<1, N + 1>( [&](auto i) constexpr { res[i] = 0; } );
    }
    return res;
  }

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(ceil) {
    using std::ceil;
    self.val  = ceil(val(self.val));
    self.grad = 0;
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  ceil(T const& x) { return ceil(Real<0, T>{x})[0]; }

  /*
  //   _             _
  //  | | ___   __ _/ |_ __
  //  | |/ _ \ / _` | | '_ \
  //  | | (_) | (_| | | |_) |
  //  |_|\___/ \__, |_| .__/
  //           |___/  |_|
  */

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

  UTILS_AUTODIFF_ADD_UNARY_FUNCTION(log1p) {
    using std::log1p;
    const T aux = One<T>() / (One<T>() + self.val); // 1 / (1 + x)
    self.val = log1p(self.val);                     // log(1 + x)
    self.grad *= aux;                               // grad * 1/(1 + x)
  }

  // overload per tipi floating-point (double, float, ...)
  template <typename T> constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
  log1p(T const& x) { return log1p(Real<0, T>{x})[0]; }

  /*
  //  
  //   _ __   _____      _____ _ __
  //  | '_ \ / _ \ \ /\ / / _ \ '__|
  //  | |_) | (_) \ V  V /  __/ |
  //  | .__/ \___/ \_/\_/ \___|_|
  //  |_|
  */

  template <typename T> inline auto power2( T const & a ) { return a*a; }
  template <typename T> inline auto power3( T const & a ) { return a*a*a; }
  template <typename T> inline auto power4( T const & a ) { auto a2{a*a}; return a2*a2; }
  template <typename T> inline auto power5( T const & a ) { auto a2{a*a}; return a2*a2*a; }
  template <typename T> inline auto power6( T const & a ) { auto a2{a*a}; return a2*a2*a2; }
  template <typename T> inline auto power7( T const & a ) { auto a2{a*a}; return a2*a2*a2*a; }
  template <typename T> inline auto power8( T const & a ) { auto a2{a*a}; auto a4{a2*a2}; return a4*a4; }

  template <typename T> inline auto rpower2( T const & a ) { return 1/(a*a); }
  template <typename T> inline auto rpower3( T const & a ) { return 1/(a*a*a); }
  template <typename T> inline auto rpower4( T const & a ) { auto a2{a*a}; return 1/(a2*a2); }
  template <typename T> inline auto rpower5( T const & a ) { auto a2{a*a}; return 1/(a2*a2*a); }
  template <typename T> inline auto rpower6( T const & a ) { auto a2{a*a}; return 1/(a2*a2*a2); }
  template <typename T> inline auto rpower7( T const & a ) { auto a2{a*a}; return 1/(a2*a2*a2*a); }
  template <typename T> inline auto rpower8( T const & a ) { auto a2{a*a}; auto a4{a2*a2}; return 1/(a4*a4); }
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
  using autodiff::detail::power8;
  using autodiff::detail::rpower2;
  using autodiff::detail::rpower3;
  using autodiff::detail::rpower4;
  using autodiff::detail::rpower5;
  using autodiff::detail::rpower6;
  using autodiff::detail::rpower7;
  using autodiff::detail::rpower8;
}

#endif

#endif

//
// eof: Utils_autodiff.hh
//
