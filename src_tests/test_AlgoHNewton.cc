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

#include "Utils_AlgoHNewton.hh"
#include "Utils_fmt.hh"

using namespace std;

using Utils::AlgoHNewton;
using Utils::m_pi;

using real_type = double;

static inline real_type power2( real_type const x )
{
  return x * x;
}
static inline real_type power3( real_type const x )
{
  return x * x * x;
}
static inline real_type power4( real_type const x )
{
  return power2( power2( x ) );
}
static inline real_type power5( real_type const x )
{
  return power4( x ) * x;
}

static int ntest{ 0 };
static int nfuneval{ 0 };

// Struct to hold test results
struct TestResult
{
  int       test_id;
  string    category;
  string    test_name;
  int       iterations;
  int       nfun;
  int       nfun_D;
  bool      converged;
  real_type root;
  real_type f_root;
  real_type a;
  real_type b;
};

static vector<TestResult> test_results;

/**
 * @brief Print results in a clean table format
 */
/**
 * @brief Helper to print a table line
 */
static void print_line(
  string const &      left,
  string const &      mid,
  string const &      right,
  string const &      cross,
  vector<int> const & widths )
{
  fmt::print( "{}", left );
  for ( size_t i = 0; i < widths.size(); ++i )
  {
    string fmt = fmt::format( "{{:{}^{{}}}}", mid );
    fmt::print( fmt, "", widths[i] );
    if ( i < widths.size() - 1 ) fmt::print( "{}", cross );
  }
  fmt::print( "{}\n", right );
}

/**
 * @brief Print results in a professional boxed table format
 */
static void print_results_table()
{
  vector<int>    w       = { 6, 32, 6, 6, 6, 8, 14, 14 };
  vector<string> headers = { "ID", "Test Name", "Iter", "#f", "#f'", "Status", "Root", "f(Root)" };

  auto c_head = Utils::PrintColors::HEADER;

  fmt::print( "\n" );

  // 1. Top Border
  print_line( "╔", "═", "╗", "╤", w );

  // 2. Header Row
  // CORREZIONE: Allineamento intestazioni
  // Numeri -> Destra (>), Testo -> Sinistra (<), Status -> Centro (^)
  fmt::print( "║" );
  fmt::print( c_head, " {:>{}} ", headers[0], w[0] - 2 );
  fmt::print( "│" );  // ID >
  fmt::print( c_head, " {:<{}} ", headers[1], w[1] - 2 );
  fmt::print( "│" );  // Name <
  fmt::print( c_head, " {:>{}} ", headers[2], w[2] - 2 );
  fmt::print( "│" );  // Iter >
  fmt::print( c_head, " {:>{}} ", headers[3], w[3] - 2 );
  fmt::print( "│" );  // #f >
  fmt::print( c_head, " {:>{}} ", headers[4], w[4] - 2 );
  fmt::print( "│" );  // #f' >
  fmt::print( c_head, " {:^{}} ", headers[5], w[5] - 2 );
  fmt::print( "│" );  // Status ^
  fmt::print( c_head, " {:>{}} ", headers[6], w[6] - 2 );
  fmt::print( "│" );  // Root >
  fmt::print( c_head, " {:>{}} ", headers[7], w[7] - 2 );
  fmt::print( "║\n" );  // f(Root) >

  // 3. Header Separator
  // print_line( "╠", "═", "╣", "╪", w );

  string current_cat = "";

  for ( const auto & tr : test_results )
  {
    if ( tr.category != current_cat )
    {
      print_line( "╠", "═", "╣", "╧", w );
      current_cat     = tr.category;
      int total_width = 0;
      for ( int cw : w ) total_width += cw;
      total_width += (int) ( w.size() - 1 );

      fmt::print( "║" );
      fmt::print( Utils::PrintColors::INFO, " {:^{}}", current_cat, total_width - 1 );
      fmt::print( "║\n" );

      print_line( "╠", "═", "╣", "╤", w );
    }

    string name = tr.test_name;
    if ( (int) name.length() > w[1] - 2 ) name = name.substr( 0, w[1] - 5 ) + "...";

    string status   = tr.converged ? "OK" : "FAIL";
    auto   c_status = tr.converged ? Utils::PrintColors::SUCCESS : Utils::PrintColors::ERROR;

    // DATA ROWS
    fmt::print( "║" );
    fmt::print( " {:>{}} ", tr.test_id, w[0] - 2 );
    fmt::print( "│" );
    fmt::print( " {:<{}} ", name, w[1] - 2 );
    fmt::print( "│" );
    fmt::print( " {:>{}} ", tr.iterations, w[2] - 2 );
    fmt::print( "│" );
    fmt::print( " {:>{}} ", tr.nfun, w[3] - 2 );
    fmt::print( "│" );
    fmt::print( " {:>{}} ", tr.nfun_D, w[4] - 2 );
    fmt::print( "│" );
    fmt::print( c_status, " {:^{}} ", status, w[5] - 2 );
    fmt::print( "│" );
    fmt::print( " {:>{}.6g} ", tr.root, w[6] - 2 );
    fmt::print( "│" );
    fmt::print( " {:>{}.4g} ", tr.f_root, w[7] - 2 );
    fmt::print( "║\n" );
  }

  // 4. Bottom Border
  print_line( "╚", "═", "╝", "╧", w );

  // --- Statistics Section ---
  int converged_count = 0;
  int tot_iter        = 0;
  for ( auto & t : test_results )
  {
    if ( t.converged ) converged_count++;
    tot_iter += t.iterations;
  }

  int table_width = 0;
  for ( int cw : w ) table_width += cw;
  table_width += (int) ( w.size() - 1 );

  int w_lbl = table_width / 2;
  int w_val = table_width - w_lbl - 1;

  fmt::print( "\n" );
  fmt::print( "╔{:═^{}}╤{:═^{}}╗\n", "", w_lbl, "", w_val );

  fmt::print( "║" );
  fmt::print( c_head, " {:<{}} ", "Metric", w_lbl - 2 );
  fmt::print( "│" );
  fmt::print( c_head, " {:<{}} ", "Value", w_val - 2 );
  fmt::print( "│\n" );

  fmt::print( "╠{:═^{}}╪{:═^{}}╣\n", "", w_lbl, "", w_val );

  fmt::print( "║" );
  fmt::print( " {:<{}} ", "Total Tests", w_lbl - 2 );
  fmt::print( "│" );
  fmt::print( Utils::PrintColors::INFO, " {:<{}} ", test_results.size(), w_val - 2 );
  fmt::print( "║\n" );

  fmt::print( "║" );
  fmt::print( " {:<{}} ", "Passed", w_lbl - 2 );
  fmt::print( "│" );
  string passed_str = fmt::format(
    "{} ({:.1f}%)",
    converged_count,
    100.0 * converged_count / static_cast<double>( test_results.size() ) );
  fmt::print( Utils::PrintColors::SUCCESS, " {:<{}} ", passed_str, w_val - 2 );
  fmt::print( "║\n" );

  if ( test_results.size() - converged_count > 0 )
  {
    fmt::print( "║" );
    fmt::print( " {:<{}} ", "Failed", w_lbl - 2 );
    fmt::print( "│" );
    fmt::print( Utils::PrintColors::ERROR, " {:<{}} ", test_results.size() - converged_count, w_val - 2 );
    fmt::print( "║\n" );
  }

  fmt::print( "║" );
  fmt::print( " {:<{}} ", "Avg Iterations", w_lbl - 2 );
  fmt::print( "│" );
  fmt::print( " {:<{}.2f} ", static_cast<double>( tot_iter ) / static_cast<double>( test_results.size() ), w_val - 2 );
  fmt::print( "║\n" );

  fmt::print( "╚{:═^{}}╧{:═^{}}╝\n", "", w_lbl, "", w_val );
}


/**
 * @brief Perform a test and collect detailed results
 */
static void do_solve(
  real_type const                                a,
  real_type const                                b,
  Utils::AlgoHNewton_base_fun<real_type> const * f,
  string const &                                 test_name = "",
  string const &                                 category  = "General" )  // NUOVO PARAMETRO
{
  AlgoHNewton<real_type> solver;
  real_type              res  = solver.eval( a, b, f );
  real_type              fres = f->eval( res );

  ++ntest;
  nfuneval += solver.num_fun_eval();

  TestResult tr;
  tr.test_id    = ntest;
  tr.category   = category;  // ASSEGNAZIONE
  tr.test_name  = test_name.empty() ? fmt::format( "Test {}", ntest ) : test_name;
  tr.iterations = solver.used_iter();
  tr.nfun       = solver.num_fun_eval();
  tr.nfun_D     = solver.num_fun_D_eval();
  tr.converged  = solver.converged();
  tr.root       = res;
  tr.f_root     = fres;
  tr.a          = a;
  tr.b          = b;

  test_results.push_back( tr );
}

// ============================================================================
// TEST FUNCTION DEFINITIONS
// ============================================================================

class FUN1 : public Utils::AlgoHNewton_base_fun<real_type>
{
public:
  real_type eval( real_type const x ) const override
  {
    real_type res{ 0 };
    for ( int i{ 1 }; i <= 20; ++i ) { res += power2( 2 * i - 5 ) / power3( x - i * i ); }
    return -2 * res;
  }

  real_type D( real_type const x ) const override
  {
    real_type res{ 0 };
    for ( int i{ 1 }; i <= 20; ++i ) { res += power2( 2 * i - 5 ) / power4( x - i * i ); }
    return 6 * res;
  }
};
FUN1 fun1;

class FUN2 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type a;
  real_type b;

public:
  FUN2() = delete;
  explicit FUN2( real_type const _a, real_type const _b ) : a( _a ), b( _b ) {}
  real_type eval( real_type const x ) const override { return a * x * exp( b * x ); }
  real_type D( real_type const x ) const override { return a * exp( b * x ) * ( b * x + 1 ); }
};

class FUN3 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;
  real_type a;

public:
  FUN3() = delete;
  explicit FUN3( real_type const _n, real_type const _a ) : n( _n ), a( _a ) {}
  real_type eval( real_type const x ) const override { return pow( x, n ) - a; }
  real_type D( real_type const x ) const override { return n * pow( x, n - 1 ); }
};

class FUN4 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN4() = delete;
  explicit FUN4( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override { return x * exp( -n ) - exp( -n * x ) + 0.5; }
  real_type D( real_type const x ) const override { return exp( -n ) + n * exp( -n * x ); }
};

class FUN5 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN5() = delete;
  explicit FUN5( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override { return ( 1 + power2( 1 - n ) ) * x - power2( 1 - n * x ); }
  real_type D( real_type const x ) const override { return 2 + ( 1 - 2 * x ) * power2( n ); }
};

class FUN6 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN6() = delete;
  explicit FUN6( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override { return power2( x ) - pow( 1 - x, n ); }
  real_type D( real_type const x ) const override { return 2 * x + n * pow( 1 - x, n - 1 ); }
};

class FUN7 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN7() = delete;
  explicit FUN7( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override { return ( 1 + power4( 1 - n ) ) * x - power4( 1 - n * x ); }
  real_type D( real_type const x ) const override { return 1 + power4( 1 - n ) + 4 * power3( 1 - n * x ) * n; }
};

class FUN8 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN8() = delete;
  explicit FUN8( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override { return ( 1 + power4( 1 - n ) ) * x - power4( 1 - n * x ); }
  real_type D( real_type const x ) const override
  {
    return ( 1 + ( 1 - x ) * n ) * exp( -n * x ) + n * pow( x, n - 1 );
  }
};

class FUN9 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN9() = delete;
  explicit FUN9( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override { return ( n * x - 1 ) / ( ( n - 1 ) * x ); }
  real_type D( real_type const x ) const override { return 1 / ( ( n - 1 ) * x * x ); }
};

class FUN10 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN10() = delete;
  explicit FUN10( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override { return pow( x, 1 / n ) - pow( n, 1 / n ); }
  real_type D( real_type const x ) const override { return pow( x, ( 1 - n ) / n ) / n; }
};

class FUN11 : public Utils::AlgoHNewton_base_fun<real_type>
{
public:
  real_type eval( real_type const x ) const override
  {
    if ( x == 0 ) return 0;
    return x * exp( -1 / ( x * x ) );
  }

  real_type D( real_type const x ) const override
  {
    if ( x == 0 ) return 0;
    real_type const x2{ x * x };
    return exp( -1 / x2 ) * ( 1 + 2 / x2 );
  }
};
FUN11 fun11;

class FUN12 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN12() = delete;
  explicit FUN12( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override
  {
    if ( x < 0 ) return -n / 20.0;
    return ( n / 20.0 ) * ( x / 1.5 + sin( x ) - 1 );
  }

  real_type D( real_type const x ) const override
  {
    if ( x < 0 ) return 0;
    return n * cos( x ) / 20.0 + n / 30.0;
  }
};

class FUN13 : public Utils::AlgoHNewton_base_fun<real_type>
{
  real_type n;

public:
  FUN13() = delete;
  explicit FUN13( real_type const _n ) : n( _n ) {}
  real_type eval( real_type const x ) const override
  {
    if ( x > 2e-3 / ( 1 + n ) ) return exp( 1 ) - 1.859;
    if ( x < 0 ) return -0.859;
    return exp( ( n + 1 ) * 0.5e3 * x ) - 1.859;
  }

  real_type D( real_type const x ) const override
  {
    if ( x > 2e-3 / ( 1 + n ) ) return 0;
    if ( x < 0 ) return 0;
    return 500 * ( n + 1 ) * exp( 500 * ( n + 1 ) * x );
  }
};

// ============================================================================
// MAIN TEST FUNCTION
// ============================================================================


int main()
{
  fmt::print( Utils::PrintColors::HEADER, "\n" );
  // Banner largo 98 caratteri
  fmt::print(
    Utils::PrintColors::HEADER,
    "╔════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                  HNEWTON ALGORITHM TEST SUITE                                  ║\n"
    "╚════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  fmt::print( Utils::PrintColors::INFO, "Starting comprehensive test suite...\n\n" );

  // Test 1: Piecewise function
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return x > 0 ? 1 / ( 1 - x ) : x - 1; }
      real_type D( real_type const x ) const override { return x > 0 ? 1 / power2( 1 - x ) : 1; }
    };
    FUN f;
    do_solve( -1.0, 1.0, &f, "Piecewise function", "Basics" );
  }

  // Test 2: sin(x) - x/2
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return sin( x ) - x / 2; }
      real_type D( real_type const x ) const override { return cos( x ) - 0.5; }
    };
    FUN f;
    do_solve( m_pi / 2, m_pi, &f, "sin(x) - x/2", "Trigonometric" );
  }

  // Tests 3-12: FUN1 (sum of rational functions)
  for ( int i = 1; i <= 10; ++i )
  {
    do_solve(
      power2( i ) + 1e-9,
      power2( i + 1 ) - 1e-9,
      &fun1,
      fmt::format( "FUN1 (i={})", i ),
      "Rational Functions" );
  }

  // Tests 13-15: FUN2 (exponential functions)
  {
    FUN2 f1( -40, -1 );
    FUN2 f2( -100, -2 );
    FUN2 f3( -200, -3 );
    do_solve( -9, 31, &f1, "FUN2: a=-40, b=-1", "Exponential Functions" );
    do_solve( -9, 31, &f2, "FUN2: a=-100, b=-2", "Exponential Functions" );
    do_solve( -9, 31, &f3, "FUN2: a=-200, b=-3", "Exponential Functions" );
  }

  // Tests 16-24: FUN3 (power functions)
  {
    FUN3 f1( 4, 0.2 );
    FUN3 f2( 6, 0.2 );
    FUN3 f3( 8, 0.2 );
    FUN3 f4( 10, 0.2 );
    FUN3 f5( 12, 0.2 );
    FUN3 f6( 8, 1 );
    FUN3 f7( 10, 1 );
    FUN3 f8( 12, 1 );
    FUN3 f9( 14, 1 );
    do_solve( 0, 5, &f1, "FUN3: n=4, a=0.2", "Power Functions" );
    do_solve( 0, 5, &f2, "FUN3: n=6, a=0.2", "Power Functions" );
    do_solve( 0, 5, &f3, "FUN3: n=8, a=0.2", "Power Functions" );
    do_solve( 0, 5, &f4, "FUN3: n=10, a=0.2", "Power Functions" );
    do_solve( 0, 5, &f5, "FUN3: n=12, a=0.2", "Power Functions" );
    do_solve( 0, 5, &f6, "FUN3: n=8, a=1", "Power Functions" );
    do_solve( 0, 5, &f7, "FUN3: n=10, a=1", "Power Functions" );
    do_solve( 0, 5, &f8, "FUN3: n=12, a=1", "Power Functions" );
    do_solve( 0, 5, &f9, "FUN3: n=14, a=1", "Power Functions" );
  }

  // Test 25: sin(x) - 0.5
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return sin( x ) - 0.5; }
      real_type D( real_type const x ) const override { return cos( x ); }
    };
    FUN f;
    do_solve( 0, 1.5, &f, "sin(x) - 0.5", "Trigonometric" );
  }

  // Tests 26-35: FUN4 (exponential equations)
  for ( int i = 1; i <= 5; ++i )
  {
    FUN4 f( i );
    do_solve( 0, 1, &f, fmt::format( "FUN4: n={}", i ), "Exponential Equations" );
  }
  for ( int i = 20; i <= 100; i += 20 )
  {
    FUN4 f( i );
    do_solve( 0, 1, &f, fmt::format( "FUN4: n={}", i ), "Exponential Equations" );
  }

  // Tests 36-38: FUN5
  {
    FUN5 f1( 5 );
    FUN5 f2( 10 );
    FUN5 f3( 20 );
    do_solve( 0, 1, &f1, "FUN5: n=5", "Polynomials" );
    do_solve( 0, 1, &f2, "FUN5: n=10", "Polynomials" );
    do_solve( 0, 1, &f3, "FUN5: n=20", "Polynomials" );
  }

  // Tests 39-43: FUN6
  {
    FUN6 f1( 2 );
    FUN6 f2( 5 );
    FUN6 f3( 10 );
    FUN6 f4( 15 );
    FUN6 f5( 20 );
    do_solve( 0, 1, &f1, "FUN6: n=2", "Polynomials" );
    do_solve( 0, 1, &f2, "FUN6: n=5", "Polynomials" );
    do_solve( 0, 1, &f3, "FUN6: n=10", "Polynomials" );
    do_solve( 0, 1, &f4, "FUN6: n=15", "Polynomials" );
    do_solve( 0, 1, &f5, "FUN6: n=20", "Polynomials" );
  }

  // Tests 44-50: FUN7
  {
    FUN7 f1( 1 );
    FUN7 f2( 2 );
    FUN7 f3( 3 );
    FUN7 f4( 5 );
    FUN7 f5( 8 );
    FUN7 f6( 15 );
    FUN7 f7( 20 );
    do_solve( 0, 1, &f1, "FUN7: n=1", "Polynomials" );
    do_solve( 0, 1, &f2, "FUN7: n=2", "Polynomials" );
    do_solve( 0, 1, &f3, "FUN7: n=3", "Polynomials" );
    do_solve( 0, 1, &f4, "FUN7: n=5", "Polynomials" );
    do_solve( 0, 1, &f5, "FUN7: n=8", "Polynomials" );
    do_solve( 0, 1, &f6, "FUN7: n=15", "Polynomials" );
    do_solve( 0, 1, &f7, "FUN7: n=20", "Polynomials" );
  }

  // Tests 51-55: FUN8
  {
    FUN8 f1( 1 );
    FUN8 f2( 5 );
    FUN8 f3( 10 );
    FUN8 f4( 15 );
    FUN8 f5( 20 );
    do_solve( 0, 1, &f1, "FUN8: n=1", "Polynomials" );
    do_solve( 0, 1, &f2, "FUN8: n=5", "Polynomials" );
    do_solve( 0, 1, &f3, "FUN8: n=10", "Polynomials" );
    do_solve( 0, 1, &f4, "FUN8: n=15", "Polynomials" );
    do_solve( 0, 1, &f5, "FUN8: n=20", "Polynomials" );
  }

  // Tests 56-59: FUN9
  {
    FUN9 f1( 2 );
    FUN9 f2( 5 );
    FUN9 f3( 15 );
    FUN9 f4( 20 );
    do_solve( 0.01, 1, &f1, "FUN9: n=2", "Polynomials" );
    do_solve( 0.01, 1, &f2, "FUN9: n=5", "Polynomials" );
    do_solve( 0.01, 1, &f3, "FUN9: n=15", "Polynomials" );
    do_solve( 0.01, 1, &f4, "FUN9: n=20", "Polynomials" );
  }

  // Tests 60-...: FUN10 (various n values)
  for ( int i = 2; i <= 9; ++i )
  {
    FUN10 f( i );
    do_solve( 1, 100, &f, fmt::format( "FUN10: n={}", i ), "Polynomials" );
  }
  for ( int i = 11; i <= 33; i += 2 )
  {
    FUN10 f( i );
    do_solve( 1, 100, &f, fmt::format( "FUN10: n={}", i ), "Polynomials" );
  }

  // Test: FUN11
  do_solve( -1, 4, &fun11, "FUN11: x*exp(-1/x²)", "Special Functions" );

  // Tests: FUN12
  for ( int i{ 1 }; i <= 8; ++i )
  {
    FUN12 f( i );
    do_solve( -1e4, m_pi / 2, &f, fmt::format( "FUN12: n={}", i ), "Trigonometric / Power" );
  }
  for ( int i{ 0 }; i <= 40; i += 10 )
  {
    FUN12 f( i );
    do_solve( -1e4, m_pi / 2, &f, fmt::format( "FUN12: n={}", i ), "Trigonometric / Power" );
  }

  // Tests: FUN13
  for ( int i{ 20 }; i <= 40; ++i )
  {
    FUN13 f( i );
    do_solve( -1e4, 1e-4, &f, fmt::format( "FUN13: n={}", i ), "Complex Roots / Special" );
  }
  for ( int i{ 100 }; i <= 1000; i += 100 )
  {
    FUN13 f( i );
    do_solve( -1e4, 1e-4, &f, fmt::format( "FUN13: n={}", i ), "Complex Roots / Special" );
  }

  // Additional miscellaneous tests
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return log( x ); }
      real_type D( real_type const x ) const override { return 1 / x; }
    };
    FUN f;
    do_solve( 0.5, 5, &f, "log(x)", "Logarithmic" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return ( 10 - x ) * exp( -10 * x ) - pow( x, 10 ) + 1; }
      real_type D( real_type const x ) const override { return ( 10 * x - 101 ) * exp( -10 * x ) - 10 * pow( x, 9 ); }
    };
    FUN f;
    do_solve( 0.5, 8, &f, "(10-x)*exp(-10x) - x¹⁰ + 1", "Mixed Transcendentals" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return exp( sin( x ) ) - x - 1; }
      real_type D( real_type const x ) const override { return cos( x ) * exp( sin( x ) ) - 1; }
    };
    FUN f;
    do_solve( 1, 4, &f, "exp(sin(x)) - x - 1", "Mixed Transcendentals" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return 11 * pow( x, 11 ) - 1; }
      real_type D( real_type const x ) const override { return 121 * pow( x, 10 ); }
    };
    FUN f;
    do_solve( 0.5, 1, &f, "11x¹¹ - 1", "Polynomials" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return 2 * sin( x ) - 1; }
      real_type D( real_type const x ) const override { return 2 * cos( x ); }
    };
    FUN f;
    do_solve( 0.1, m_pi / 3, &f, "2sin(x) - 1", "Trigonometric" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return power2( x ) + sin( x / 10 ) - 0.25; }
      real_type D( real_type const x ) const override { return 2 * x + cos( x / 10 ) / 10; }
    };
    FUN f;
    do_solve( 0, 1, &f, "x² + sin(x/10) - 0.25", "Mixed Transcendentals" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return ( x - 1 ) * exp( x ); }
      real_type D( real_type const x ) const override { return exp( x ) * x; }
    };
    FUN f;
    do_solve( 0, 1.5, &f, "(x-1)*exp(x)", "Mixed Transcendentals" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return cos( x ) - x; }
      real_type D( real_type const x ) const override { return -sin( x ) - 1; }
    };
    FUN f;
    do_solve( 0, 1.7, &f, "cos(x) - x", "Trigonometric" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return power3( x - 1 ) - 1; }
      real_type D( real_type const x ) const override { return 3 * power2( x - 1 ); }
    };
    FUN f;
    do_solve( 1.5, 3, &f, "(x-1)³ - 1", "Polynomials" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return exp( x * x + 7 * x - 30 ) - 1; }
      real_type D( real_type const x ) const override { return ( 2 * x + 7 ) * exp( ( x + 10 ) * ( x - 3 ) ); }
    };
    FUN f;
    do_solve( 2.6, 3.5, &f, "exp(x²+7x-30) - 1", "Exponential Composition" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return atan( x ) - 1; }
      real_type D( real_type const x ) const override { return 1 / ( x * x + 1 ); }
    };
    FUN f;
    do_solve( 1, 8, &f, "atan(x) - 1", "Inverse Trigonometric" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return exp( x ) - 2 * x - 1; }
      real_type D( real_type const x ) const override { return exp( x ) - 2; }
    };
    FUN f;
    do_solve( 0.2, 3, &f, "exp(x) - 2x - 1", "Exponential" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return exp( -x ) - x - sin( x ); }
      real_type D( real_type const x ) const override { return -exp( -x ) - 1 - cos( x ); }
    };
    FUN f;
    do_solve( 0, 0.5, &f, "exp(-x) - x - sin(x)", "Mixed Transcendentals" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return power3( x ) - 1; }
      real_type D( real_type const x ) const override { return 3 * power2( x ); }
    };
    FUN f;
    do_solve( 0.1, 1.5, &f, "x³ - 1", "Polynomials" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return power2( x ) - power2( sin( x ) ) - 1; }
      real_type D( real_type const x ) const override { return 2 * x - sin( 2 * x ); }
    };
    FUN f;
    do_solve( -1, 2, &f, "x² - sin²(x) - 1", "Mixed Transcendentals" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return power3( x ); }
      real_type D( real_type const x ) const override { return 3 * power2( x ); }
    };
    FUN f;
    do_solve( -0.5, 1 / 3.0, &f, "x³ (root at 0)", "Roots with Multiplicity" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override { return power5( x ); }
      real_type D( real_type const x ) const override { return 5 * power4( x ); }
    };
    FUN f;
    do_solve( -0.5, 1 / 3.0, &f, "x⁵ (root at 0)", "Roots with Multiplicity" );
  }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type>
    {
    public:
      real_type eval( real_type const x ) const override
      {
        real_type const x2{ x * x };
        real_type const x4{ x2 * x2 };
        real_type const x8{ x4 * x4 };
        return tan( m_pi * ( x8 - 0.5 ) );
      }
      real_type D( real_type const x ) const override
      {
        real_type const x2{ x * x };
        real_type const x4{ x2 * x2 };
        real_type const x7{ x4 * x2 * x };
        real_type const x8{ x4 * x4 };
        return 8 * m_pi * x7 * power2( 1 / cos( m_pi * ( x8 - 0.5 ) ) );
      }
    };
    FUN f;
    do_solve( 0.0, 1.0, &f, "tan(π(x⁸-0.5))", "Hard Trigonometric" );
  }

  // Print final results table
  print_results_table();

  // Print total function evaluations
  fmt::print( "\n" );
  fmt::print( Utils::PrintColors::ITERATION, "Total function evaluations: {}\n", nfuneval );

  // Final message largo 98 caratteri
  fmt::print( Utils::PrintColors::SUCCESS, "\n" );
  fmt::print(
    Utils::PrintColors::SUCCESS,
    "╔════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                      ALL TESTS COMPLETED!                                      ║\n"
    "╚════════════════════════════════════════════════════════════════════════════════════════════════╝\n" );

  return 0;
}
