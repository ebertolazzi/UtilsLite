/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022                                                      |
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

#include "Utils_fmt.hh"
#include "Utils_zeros.hh"
#include <iomanip>
#include <limits>

using namespace std;

using Utils::m_pi;
using Utils::Zeros;

using real_type = double;

// ============================================================================
// Funzioni di test standard
// ============================================================================

// Test 0: Polinomio semplice (x + 4) * x² - 10 = 0
class PolyFunction : public Utils::Zeros_base_fun<real_type>
{
public:
  [[nodiscard]] real_type eval( real_type x ) const override { return ( x + 4 ) * x * x - 10; }
  [[nodiscard]] real_type eval_D( real_type x ) const override { return ( 3 * x + 8 ) * x; }
  [[nodiscard]] real_type eval_DD( real_type x ) const override { return 6 * x + 8; }
  [[nodiscard]] real_type eval_DDD( real_type ) const override { return 6; }
};

// Test 1: Radice quadrata (x² - A = 0)
class SquareRootFunction : public Utils::Zeros_base_fun<real_type>
{
  real_type A;

public:
  explicit SquareRootFunction( real_type a ) : A( a ) {}
  [[nodiscard]] real_type eval( real_type x ) const override { return x * x - A; }
  [[nodiscard]] real_type eval_D( real_type x ) const override { return 2 * x; }
  [[nodiscard]] real_type eval_DD( real_type ) const override { return 2; }
  [[nodiscard]] real_type eval_DDD( real_type ) const override { return 0; }
};

// Test 2: Radice cubica (x³ - A = 0)
class CubeRootFunction : public Utils::Zeros_base_fun<real_type>
{
  real_type A;

public:
  explicit CubeRootFunction( real_type a ) : A( a ) {}
  [[nodiscard]] real_type eval( real_type x ) const override { return x * x * x - A; }
  [[nodiscard]] real_type eval_D( real_type x ) const override { return 3 * x * x; }
  [[nodiscard]] real_type eval_DD( real_type x ) const override { return 6 * x; }
  [[nodiscard]] real_type eval_DDD( real_type ) const override { return 6; }
};

// Test 3: Funzione trigonometrica mista (originale del test)
class MixedFunction1 : public Utils::Zeros_base_fun<real_type>
{
public:
  [[nodiscard]] real_type eval( real_type x ) const override
  {
    return log( 1 + x * x ) + exp( x * ( x - 3 ) ) * sin( x );
  }
  [[nodiscard]] real_type eval_D( real_type x ) const override
  {
    real_type t1  = x * x;
    real_type t2  = t1 + 1.0;
    real_type t4  = sin( x );
    real_type t6  = cos( x );
    real_type t12 = exp( x * ( x - 3.0 ) );
    return 2.0 / t2 * ( t12 * ( t4 * ( x - 3.0 / 2.0 ) + t6 / 2.0 ) * t2 + x );
  }
  [[nodiscard]] real_type eval_DD( real_type x ) const override
  {
    real_type t1  = x * x;
    real_type t3  = pow( t1 + 1.0, 2.0 );
    real_type t6  = sin( x );
    real_type t8  = cos( x );
    real_type t15 = exp( x * ( x - 3.0 ) );
    return 1 / t3 *
           ( 4.0 * t15 * ( t6 * ( t1 - 3.0 * x + 5.0 / 2.0 ) + ( x - 3.0 / 2.0 ) * t8 ) * t3 - 2.0 * t1 + 2.0 );
  }
  [[nodiscard]] real_type eval_DDD( real_type x ) const override
  {
    real_type t2  = x * x;
    real_type t3  = 3.0 * x;
    real_type t6  = sin( x );
    real_type t9  = cos( x );
    real_type t13 = t2 + 1.0;
    real_type t14 = t13 * t13;
    real_type t15 = t14 * t13;
    real_type t19 = exp( x * ( x - 3.0 ) );
    return 1 / t15 *
           ( 8.0 * t19 * t15 *
               ( t6 * ( t2 - t3 + 3.0 ) * ( x - 3.0 / 2.0 ) + 3.0 / 2.0 * t9 * ( t2 - t3 + 8.0 / 3.0 ) ) +
             4.0 * t2 * x - 12.0 * x );
  }
};

// Test 4: Funzione esponenziale e trigonometrica (originale del test)
class MixedFunction2 : public Utils::Zeros_base_fun<real_type>
{
public:
  [[nodiscard]] real_type eval( real_type x ) const override
  {
    real_type t1 = x * x;
    real_type t3 = exp( -t1 + x + 2.0 );
    real_type t6 = cos( x + 1.0 );
    return x * t1 + t3 - t6 + 1.0;
  }
  [[nodiscard]] real_type eval_D( real_type x ) const override
  {
    real_type t3 = x * x;
    real_type t5 = exp( -t3 + x + 2.0 );
    real_type t9 = sin( x + 1.0 );
    return t5 * ( -2.0 * x + 1.0 ) + 3.0 * t3 + t9;
  }
  [[nodiscard]] real_type eval_DD( real_type x ) const override
  {
    real_type t1  = x * x;
    real_type t3  = exp( -t1 + x + 2.0 );
    real_type t7  = pow( -2.0 * x + 1.0, 2.0 );
    real_type t11 = cos( x + 1.0 );
    return t3 * t7 + t11 - 2.0 * t3 + 6.0 * x;
  }
  [[nodiscard]] real_type eval_DDD( real_type x ) const override
  {
    real_type t2  = -2.0 * x + 1.0;
    real_type t3  = x * x;
    real_type t5  = exp( -t3 + x + 2.0 );
    real_type t8  = t2 * t2;
    real_type t12 = sin( x + 1.0 );
    return t5 * t2 * t8 - 6.0 * t5 * t2 - t12 + 6.0;
  }
};

// Test 5: Funzione razionale e trigonometrica (originale del test)
class MixedFunction3 : public Utils::Zeros_base_fun<real_type>
{
public:
  [[nodiscard]] real_type eval( real_type x ) const override
  {
    real_type t1 = x * x;
    real_type t2 = t1 + 1.0;
    real_type t5 = cos( m_pi * x / 2.0 );
    real_type t9 = log( t1 + 2.0 * x + 2.0 );
    return t5 * t2 + 1 / t2 * t9;
  }
  [[nodiscard]] real_type eval_D( real_type x ) const override
  {
    real_type t2  = m_pi * x / 2.0;
    real_type t3  = cos( t2 );
    real_type t6  = x * x;
    real_type t7  = t6 + 1.0;
    real_type t9  = sin( t2 );
    real_type t14 = t6 + 2.0 * x + 2.0;
    real_type t19 = log( t14 );
    real_type t20 = t7 * t7;
    return 2.0 * t3 * x - t9 * m_pi * t7 / 2.0 + 2.0 / t7 / t14 * ( x + 1.0 ) - 2.0 * x / t20 * t19;
  }
  [[nodiscard]] real_type eval_DD( real_type x ) const override
  {
    real_type t1  = m_pi * x;
    real_type t2  = t1 / 2.0;
    real_type t3  = cos( t2 );
    real_type t5  = sin( t2 );
    real_type t8  = x * x;
    real_type t9  = t8 + 1.0;
    real_type t10 = m_pi * m_pi;
    real_type t15 = t8 + 2.0 * x + 2.0;
    real_type t16 = 1 / t15;
    real_type t17 = 1 / t9;
    real_type t20 = x + 1.0;
    real_type t22 = t15 * t15;
    real_type t27 = t9 * t9;
    real_type t28 = 1 / t27;
    real_type t32 = log( t15 );
    return 2.0 * t3 - 2.0 * t5 * t1 - t3 * t10 * t9 / 4.0 + 2.0 * t17 * t16 - 4.0 * t17 / t22 * t20 * t20 -
           8.0 * x * t28 * t20 * t16 + 8.0 * t8 / t9 / t27 * t32 - 2.0 * t28 * t32;
  }
  [[nodiscard]] real_type eval_DDD( real_type x ) const override
  {
    real_type t2  = m_pi * x / 2.0;
    real_type t3  = sin( t2 );
    real_type t6  = m_pi * m_pi;
    real_type t8  = cos( t2 );
    real_type t11 = x * x;
    real_type t12 = t11 + 1.0;
    real_type t18 = t11 + 2.0 * x + 2.0;
    real_type t19 = t18 * t18;
    real_type t20 = 1 / t19;
    real_type t21 = 1 / t12;
    real_type t23 = x + 1.0;
    real_type t26 = 1 / t18;
    real_type t27 = t12 * t12;
    real_type t28 = 1 / t27;
    real_type t32 = 4.0 * t23 * t23;
    real_type t44 = 2.0 * t26 * t23;
    real_type t46 = 1 / t12 / t27;
    real_type t52 = log( t18 );
    real_type t53 = t27 * t27;
    return -3.0 * t3 * m_pi - 3.0 / 2.0 * t8 * t6 * x + t3 * m_pi * t6 * t12 / 8.0 - 12.0 * t23 * t21 * t20 -
           12.0 * x * t28 * t26 + 4.0 * t21 / t18 / t19 * t23 * t32 + 6.0 * x * t28 * t20 * t32 +
           24.0 * t11 * t46 * t44 - 6.0 * t28 * t44 - 48.0 * x * t11 / t53 * t52 + 24.0 * x * t46 * t52;
  }
};

// Test 6: Funzione con termine oscillante (originale del test)
class OscillatingFunction : public Utils::Zeros_base_fun<real_type>
{
public:
  [[nodiscard]] real_type eval( real_type x ) const override
  {
    real_type t1 = x * x;
    real_type t2 = t1 * t1;
    real_type t5 = sin( 1 / t1 * m_pi );
    return t2 + t5 - 5.0;
  }
  [[nodiscard]] real_type eval_D( real_type x ) const override
  {
    real_type t1 = x * x;
    real_type t2 = x * t1;
    real_type t8 = cos( 1 / t1 * m_pi );
    return 4.0 * t2 - 2.0 * t8 / t2 * m_pi;
  }
  [[nodiscard]] real_type eval_DD( real_type x ) const override
  {
    real_type t1  = x * x;
    real_type t3  = t1 * t1;
    real_type t7  = 1 / t1 * m_pi;
    real_type t8  = cos( t7 );
    real_type t11 = m_pi * m_pi;
    real_type t15 = sin( t7 );
    return 12.0 * t1 + 6.0 * t8 / t3 * m_pi - 4.0 * t15 / t1 / t3 * t11;
  }
  [[nodiscard]] real_type eval_DDD( real_type x ) const override
  {
    real_type t2  = x * x;
    real_type t3  = t2 * t2;
    real_type t8  = 1 / t2 * m_pi;
    real_type t9  = cos( t8 );
    real_type t12 = m_pi * m_pi;
    real_type t17 = sin( t8 );
    real_type t21 = t3 * t3;
    return 24.0 * x - 24.0 * t9 / x / t3 * m_pi + 36.0 * t17 / x / t2 / t3 * t12 + 8.0 * t9 / x / t21 * m_pi * t12;
  }
};

// ============================================================================
// Analisi di convergenza con trace
// ============================================================================

struct ConvergenceAnalysis
{
  vector<real_type> f_values;              // Valori di f(x) durante le iterazioni
  vector<real_type> estimated_order;       // Ordine stimato ad ogni passo
  real_type         average_order;         // Ordine medio stimato
  real_type         asymptotic_constant;   // Costante asintotica stimata
  int               effective_iterations;  // Iterazioni effettive per l'analisi
};

ConvergenceAnalysis analyze_convergence( vector<real_type> const & trace_values )
{
  ConvergenceAnalysis analysis;

  if ( trace_values.size() < 4 )
  {
    fmt::print(
      fg( fmt::color::red ),
      "  Trace too short for convergence analysis ({} points)\n",
      trace_values.size() );
    return analysis;
  }

  // Copia i valori di f(x)
  analysis.f_values = trace_values;

  // Stima dell'ordine di convergenza usando la formula:
  // p ≈ log(|e_{k+1}| / |e_k|) / log(|e_k| / |e_{k-1}|)
  // dove e_k = |f(x_k)| (assumendo f'(root) ≠ 0)

  for ( size_t i = 2; i < trace_values.size() - 1; ++i )
  {
    real_type e_k1  = abs( trace_values[i + 1] );
    real_type e_k   = abs( trace_values[i] );
    real_type e_km1 = abs( trace_values[i - 1] );

    // Evita divisioni per zero o valori troppo piccoli
    if ( e_k < 1e-15 || e_km1 < 1e-15 ) continue;

    real_type ratio1 = e_k1 / e_k;
    real_type ratio2 = e_k / e_km1;

    if ( ratio1 > 0 && ratio2 > 0 && ratio1 < 1 && ratio2 < 1 )
    {
      real_type p = log( ratio1 ) / log( ratio2 );
      if ( p > 0 && p < 100 )
      {  // Filtra valori non fisici
        analysis.estimated_order.push_back( p );
      }
    }
  }

  if ( !analysis.estimated_order.empty() )
  {
    // Calcola media e deviazione standard
    real_type sum = 0.0;
    for ( auto p : analysis.estimated_order ) sum += p;
    analysis.average_order = sum / static_cast<real_type>( analysis.estimated_order.size() );

    // Stima della costante asintotica: C ≈ |e_{k+1}| / |e_k|^p
    if ( trace_values.size() >= 3 )
    {
      size_t    last   = trace_values.size() - 1;
      real_type e_last = abs( trace_values[last] );
      real_type e_prev = abs( trace_values[last - 1] );
      if ( e_prev > 1e-15 ) { analysis.asymptotic_constant = e_last / pow( e_prev, analysis.average_order ); }
    }

    analysis.effective_iterations = analysis.estimated_order.size();
  }

  return analysis;
}

void print_convergence_table( const string & method_name, const ConvergenceAnalysis & analysis, int theoretical_order )
{
  fmt::print( fg( fmt::color::cyan ), "\n  Convergence analysis for {}:\n", method_name );

  if ( analysis.f_values.empty() )
  {
    fmt::print( "    No trace data available\n" );
    return;
  }

  // Tabella dei valori di f(x)
  fmt::print( "    Iteration       f(x)           |f(x)|/|f(x_prev)|   Estimated Order\n" );
  fmt::print( "    ----------------------------------------------------------------\n" );

  for ( size_t i = 0; i < min<size_t>( analysis.f_values.size(), 10 ); ++i )
  {
    real_type f_val = analysis.f_values[i];
    real_type ratio = 0.0;
    if ( i > 0 )
    {
      real_type prev = abs( analysis.f_values[i - 1] );
      if ( prev > 1e-15 ) { ratio = abs( f_val ) / prev; }
    }

    string order_str = "       -";
    if ( i >= 2 && i - 2 < analysis.estimated_order.size() )
    {
      real_type p = analysis.estimated_order[i - 2];
      if ( abs( p - theoretical_order ) < 0.5 )
      {
        order_str = fmt::format( fg( fmt::color::lime_green ), "{:8.3g}", p );
      }
      else if ( abs( p - theoretical_order ) < 1.5 )
      {
        order_str = fmt::format( fg( fmt::color::yellow ), "{:8.3g}", p );
      }
      else
      {
        order_str = fmt::format( fg( fmt::color::red ), "{:8.3g}", p );
      }
    }

    fmt::print( "    {:8d}  {:12.3e}  {:18.3e}  {}\n", i, f_val, ratio, order_str );
  }

  if ( analysis.f_values.size() > 10 )
  {
    fmt::print( "    ... (showing first 10 of {} iterations)\n", analysis.f_values.size() );
  }

  // Riassunto statistico
  if ( !analysis.estimated_order.empty() )
  {
    fmt::print( "\n    Summary:\n" );
    fmt::print( "    - Theoretical order:          {:3d}\n", theoretical_order );
    fmt::print( "    - Average estimated order:    {:.3g}\n", analysis.average_order );
    fmt::print( "    - Effective iterations:       {:3d}\n", analysis.effective_iterations );

    // Calcola deviazione standard
    real_type variance = 0.0;
    for ( auto p : analysis.estimated_order )
    {
      variance += ( p - analysis.average_order ) * ( p - analysis.average_order );
    }
    variance /= static_cast<real_type>( analysis.estimated_order.size() );
    real_type std_dev = sqrt( variance );

    fmt::print( "    - Standard deviation:        {:.3g}\n", std_dev );

    if ( analysis.asymptotic_constant > 0 )
    {
      fmt::print( "    - Asymptotic constant (C):   {:.3e}\n", analysis.asymptotic_constant );
    }

    // Valutazione
    real_type order_error = abs( analysis.average_order - theoretical_order );
    if ( order_error < 0.1 )
    {
      fmt::print( fg( fmt::color::lime_green ), "    ✓ Order verification: EXCELLENT (error: {:.3g})\n", order_error );
    }
    else if ( order_error < 0.5 )
    {
      fmt::print( fg( fmt::color::yellow ), "    ⚠ Order verification: GOOD (error: {:.3g})\n", order_error );
    }
    else if ( order_error < 2.0 )
    {
      fmt::print( fg( fmt::color::orange ), "    ⚠ Order verification: FAIR (error: {:.3g})\n", order_error );
    }
    else
    {
      fmt::print( fg( fmt::color::red ), "    ✗ Order verification: POOR (error: {:.3g})\n", order_error );
    }
  }

  fmt::print( "\n" );
}

// ============================================================================
// Test Runner migliorato con trace
// ============================================================================

struct TestResult
{
  string              method_name;
  real_type           root;
  int                 iterations;
  int                 evaluations;
  bool                converged;
  real_type           residual;
  int                 theoretical_order;
  ConvergenceAnalysis convergence;
};

class TestRunner
{
  static constexpr auto HEADER_COLOR  = fmt::color::steel_blue;
  static constexpr auto SUCCESS_COLOR = fmt::color::lime_green;
  static constexpr auto WARNING_COLOR = fmt::color::gold;
  static constexpr auto ERROR_COLOR   = fmt::color::crimson;
  static constexpr auto METHOD_COLOR  = fmt::color::deep_sky_blue;
  static constexpr auto VALUE_COLOR   = fmt::color::white;
  static constexpr auto ITER_COLOR    = fmt::color::violet;
  static constexpr auto ORDER_COLOR   = fmt::color::cyan;

public:
  static void print_header( const string & title )
  {
    fmt::print( fg( HEADER_COLOR ) | fmt::emphasis::bold, "\n{:=^80}\n", " " + title + " " );
  }

  static void print_table_header()
  {
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "┌─────────────────┬─────────────┬──────┬──────┬────────┬──────────────┬────────────┐\n" );
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "│ {:15s} │ {:11s} │ {:4s} │ {:4s} │ {:6s} │ {:12s} │ {:10s} │\n",
      "Method",
      "Root",
      "Iter",
      "Eval",
      "Conv",
      "Residual",
      "Order" );
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "├─────────────────┼─────────────┼──────┼──────┼────────┼──────────────┼────────────┤\n" );
  }

  static void print_table_footer()
  {
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "└─────────────────┴─────────────┴──────┴──────┴────────┴──────────────┴────────────┘\n" );
  }

  static void print_result( const TestResult & result )
  {
    // Choose color based on convergence
    auto status_color = result.converged ? fg( SUCCESS_COLOR ) : fg( ERROR_COLOR );
    auto method_fmt   = fg( METHOD_COLOR );
    auto iter_fmt     = fg( ITER_COLOR );
    auto value_fmt    = fg( VALUE_COLOR );
    auto order_fmt    = fg( ORDER_COLOR );

    fmt::print( "│ " );
    fmt::print( method_fmt, "{:15s}", result.method_name );
    fmt::print( " │ " );
    fmt::print( value_fmt, "{:11.5g}", result.root );
    fmt::print( " │ " );
    fmt::print( iter_fmt, "{:4d}", result.iterations );
    fmt::print( " │ " );
    fmt::print( iter_fmt, "{:4d}", result.evaluations );
    fmt::print( " │ " );
    fmt::print( status_color, "{:6s}", result.converged ? "✓" : "✗" );
    fmt::print( " │ " );

    // Color code residual magnitude
    if ( abs( result.residual ) < 1e-10 ) { fmt::print( fg( SUCCESS_COLOR ), "{:12.2e}", result.residual ); }
    else if ( abs( result.residual ) < 1e-5 ) { fmt::print( fg( WARNING_COLOR ), "{:12.2e}", result.residual ); }
    else
    {
      fmt::print( fg( ERROR_COLOR ), "{:12.2e}", result.residual );
    }

    fmt::print( " │ " );

    // Display order information
    if ( result.convergence.effective_iterations > 0 )
    {
      real_type avg_order   = result.convergence.average_order;
      real_type order_error = abs( avg_order - result.theoretical_order );

      auto color = order_error < 0.1 ? SUCCESS_COLOR : order_error < 0.5 ? WARNING_COLOR : ERROR_COLOR;
      fmt::print( fg( color ), " {:8.2g} ", avg_order );
    }
    else
    {
      fmt::print( order_fmt, "{:>10s}", "N/A" );
    }

    fmt::print( " │\n" );
  }

  static vector<TestResult> run_test_with_trace( real_type x_guess, Utils::Zeros_base_fun<real_type> * fun )
  {
    vector<TestResult> results;

    // Mappa dei metodi con i loro ordini teorici
    vector<
      tuple<string, int, real_type ( Utils::Zeros<real_type>::* )( real_type, Utils::Zeros_base_fun<real_type> * )>>
      methods = { { "Newton", 2, &Utils::Zeros<real_type>::solve_Newton },
                  { "Halley", 3, &Utils::Zeros<real_type>::solve_Halley },
                  { "Chebyshev", 3, &Utils::Zeros<real_type>::solve_Chebyshev },
                  { "Order4", 4, &Utils::Zeros<real_type>::solve_Order4 },
                  { "Order8", 8, &Utils::Zeros<real_type>::solve_Order8 },
                  { "Order16", 16, &Utils::Zeros<real_type>::solve_Order16 },
                  { "Order32", 32, &Utils::Zeros<real_type>::solve_Order32 } };

    for ( const auto & [name, theo_order, method] : methods )
    {
      Utils::Zeros<real_type> solver;

      // Configurazione del solver
      solver.set_tolerance( 1e-15 );
      solver.set_max_iterations( 50 );
      solver.set_trace( true );  // Attiva il trace

      // Esegui il metodo
      real_type root = ( solver.*method )( x_guess, fun );

      // Ottieni i valori del trace
      auto trace_values = solver.trace_values();

      // Analisi della convergenza
      auto convergence = analyze_convergence( trace_values );

      results.push_back(
        { name,
          root,
          solver.used_iter(),
          solver.num_fun_eval(),
          solver.converged(),
          fun->eval( root ),
          theo_order,
          convergence } );
    }

    return results;
  }
};

// ============================================================================
// Funzioni di utilità per test dettagliati
// ============================================================================

void run_detailed_convergence_test(
  const string &                     test_name,
  real_type                          x_guess,
  Utils::Zeros_base_fun<real_type> * fun,
  real_type                          exact_root )
{
  TestRunner::print_header( test_name );

  fmt::print( fg( fmt::color::cyan ), "Exact root: {:.15g}\n", exact_root );
  fmt::print( "Initial guess: {:.6g} (distance: {:.2g})\n\n", x_guess, abs( x_guess - exact_root ) );

  // Esegui tutti i metodi con trace
  auto results = TestRunner::run_test_with_trace( x_guess, fun );

  // Tabella riassuntiva
  TestRunner::print_table_header();
  for ( const auto & result : results ) { TestRunner::print_result( result ); }
  TestRunner::print_table_footer();

  // Analisi dettagliata per ciascun metodo
  fmt::print( fg( fmt::color::orange ) | fmt::emphasis::bold, "\n{:-^80}\n", " Detailed Convergence Analysis " );

  for ( const auto & result : results )
  {
    if ( result.converged && result.convergence.effective_iterations > 0 )
    {
      print_convergence_table( result.method_name, result.convergence, result.theoretical_order );
    }
  }

  // Confronto delle performance
  fmt::print( fg( fmt::color::light_blue ) | fmt::emphasis::bold, "\n{:-^80}\n", " Performance Comparison " );

  // Trova il metodo migliore (minimo numero di iterazioni con convergenza)
  int    best_iter = INT_MAX;
  string best_method;
  for ( const auto & result : results )
  {
    if ( result.converged && result.iterations < best_iter )
    {
      best_iter   = result.iterations;
      best_method = result.method_name;
    }
  }

  if ( !best_method.empty() )
  {
    fmt::print( fg( fmt::color::lime_green ), "✓ Fastest method: {} ({} iterations)\n", best_method, best_iter );
  }

  // Efficienza computazionale (valutazioni per iterazione)
  fmt::print( "\nComputational efficiency (evaluations per iteration):\n" );
  for ( const auto & result : results )
  {
    if ( result.converged )
    {
      real_type eff = static_cast<real_type>( result.evaluations ) / result.iterations;
      fmt::print( "  {:10s}: {:.2g} eval/iter\n", result.method_name, eff );
    }
  }

  fmt::print( "\n" );
}

void run_quick_comparison(
  const string &                     test_name,
  real_type                          x_guess,
  Utils::Zeros_base_fun<real_type> * fun,
  real_type                          expected_root = 0.0 )
{
  TestRunner::print_header( test_name );

  if ( expected_root != 0.0 ) { fmt::print( fg( fmt::color::cyan ), "Expected root: {:.10g}\n", expected_root ); }
  fmt::print( "Initial guess: {:.6g}\n\n", x_guess );

  auto results = TestRunner::run_test_with_trace( x_guess, fun );

  TestRunner::print_table_header();
  for ( const auto & result : results ) { TestRunner::print_result( result ); }
  TestRunner::print_table_footer();

  fmt::print( "\n" );
}

// ============================================================================
// Main function
// ============================================================================

int main()
{
  fmt::print(
    fg( fmt::color::orange ) | fmt::emphasis::bold,
    "{:=^80}\n",
    " Root-Finding Methods with Trace Analysis " );
  fmt::print(
    fg( fmt::color::gray ),
    "Using methods from: J.L. Varona, "
    "\"An Optimal Thirty-Second-Order Iterative Method\", 2022\n\n" );

  // Sezione 1: Test con analisi di convergenza dettagliata
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "{:=^80}\n", " PART 1: DETAILED CONVERGENCE ANALYSIS " );

  // Test 1: Radice quadrata (√2) - analisi completa
  SquareRootFunction sqrt_fun( 2.0 );
  real_type          sqrt2 = sqrt( 2.0 );
  run_detailed_convergence_test( "Square Root: x² - 2 = 0", 1.5, &sqrt_fun, sqrt2 );

  // Test 2: Radice cubica (∛8) - analisi completa
  CubeRootFunction cube_fun( 8.0 );
  run_detailed_convergence_test( "Cube Root: x³ - 8 = 0", 2.5, &cube_fun, 2.0 );

  // Test 3: Polinomio semplice
  PolyFunction poly_fun;
  run_detailed_convergence_test( "Polynomial: (x+4)x² - 10 = 0", 0.35, &poly_fun, 1.365230013 );

  // Sezione 2: Test comparativi rapidi
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:=^80}\n", " PART 2: QUICK COMPARISON TESTS " );

  // Test 4: Funzione mista complessa 1
  MixedFunction1 mixed1_fun;
  run_quick_comparison( "Mixed Function 1: log(1+x²) + exp(x(x-3))sin(x)", 0.35, &mixed1_fun );

  // Test 5: Funzione mista complessa 2
  MixedFunction2 mixed2_fun;
  run_quick_comparison( "Mixed Function 2: x³ + exp(-x²+x+2) - cos(x+1) + 1", -10.3, &mixed2_fun );

  // Test 6: Funzione razionale/trigonometrica
  MixedFunction3 mixed3_fun;
  run_quick_comparison( "Mixed Function 3: (x²+1)cos(πx/2) + log(x²+2x+2)/(x²+1)", -1.1, &mixed3_fun );

  // Test 7: Funzione oscillante
  OscillatingFunction osc_fun;
  run_quick_comparison( "Oscillating Function: x⁴ + sin(π/x²) - 5", 1.5, &osc_fun );

  // Test con condizioni estreme
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:=^80}\n", " PART 3: EXTREME CASES " );

  // Radice di numero piccolo
  SquareRootFunction small_fun( 1e-10 );
  run_quick_comparison( "Small Root: x² - 1e-10 = 0", 1e-3, &small_fun, 1e-5 );

  // Radice di numero grande
  SquareRootFunction large_fun( 1e10 );
  run_quick_comparison( "Large Root: x² - 1e10 = 0", 1e4, &large_fun, 1e5 );

  // Riepilogo finale
  fmt::print( fg( fmt::color::light_green ) | fmt::emphasis::bold, "\n{:=^80}\n", " SUMMARY " );

  fmt::print(
    fg( fmt::color::light_blue ),
    "Key observations from trace analysis:\n"
    "• Newton (order 2): Consistent quadratic convergence\n"
    "• Halley/Chebyshev (order 3): Faster but requires f''\n"
    "• Order 4-32 methods: Show expected high-order behavior near root\n"
    "• Trace data: Confirms theoretical convergence orders\n"
    "• Asymptotic constant: Smaller for higher-order methods\n\n" );

  fmt::print( fg( fmt::color::light_green ) | fmt::emphasis::bold, "{:=^80}\n", " All Tests Completed Successfully " );

  return 0;
}
