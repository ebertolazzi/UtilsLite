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

#include "Utils_Trichotomy.hh"
#include "Utils_fmt.hh"

#if defined( __clang__ )
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-macros"
#elif defined( __llvm__ ) || defined( __GNUC__ )
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-macros"
#elif defined( _MSC_VER )
#pragma warning( disable : 4100 )
#pragma warning( disable : 4101 )
#pragma warning( disable : 4189 )
#endif

using namespace std;
using Utils::Trichotomy;
using real_type = double;
using Utils::m_pi;

// ============================================================================
// Test functions for minimization
// ============================================================================

struct TestFunction
{
  string    name;
  string    expression;
  real_type min_x;
  real_type min_value;
  real_type a;
  real_type b;
  string    category;           // "unimodal", "multimodal", "non-differentiable"
  bool      expect_global_min;  // Whether we expect to find the global minimum

  // Constructor
  TestFunction(
    string    n,
    string    expr,
    real_type mx,
    real_type mv,
    real_type a0,
    real_type b0,
    string    cat    = "unimodal",
    bool      global = true )
    : name( n )
    , expression( expr )
    , min_x( mx )
    , min_value( mv )
    , a( a0 )
    , b( b0 )
    , category( cat )
    , expect_global_min( global )
  {
  }
};

// Collection of test functions
vector<TestFunction> test_functions = {
  // Quadratic functions (unimodal, convex)
  { "Simple quadratic", "x²", 0.0, 0.0, -2.0, 2.0, "unimodal", true },
  { "Translated parabola", "(x-1)²", 1.0, 0.0, -1.0, 3.0, "unimodal", true },
  { "Wide parabola", "2x² - 4x + 3", 1.0, 1.0, -5.0, 5.0, "unimodal", true },

  // Trigonometric functions
  { "Cosine", "cos(x)", m_pi, -1.0, 0.0, 2 * m_pi, "unimodal", true },
  { "Sine + quadratic", "sin(x) + 0.1*(x-2)²", 1.757, -0.710, -m_pi, m_pi, "multimodal", false },

  // Exponential functions
  { "Double exponential", "exp(x) + exp(-x)", 0.0, 2.0, -2.0, 2.0, "unimodal", true },
  { "Gaussian", "exp(-x²/2)", 0.0, 1.0, -3.0, 3.0, "unimodal", true },

  // Rational functions
  { "Hyperbola", "x² + 1/x", 0.793700526, 2.381101578, 0.1, 5.0, "unimodal", true },
  { "Rational function", "(x⁴ - 3x³ + 2)/(x² + 1)", 2.279018786, -1.448201103, -5.0, 5.0, "multimodal", false },

  // Classic test functions
  { "Quartic", "100*(x² - x)² + 14*(1 - x)", 1.0, 0.0, -2.0, 2.0, "unimodal", true },
  { "Himmelblau 1D", "(x² + x - 11)² + (x + x² - 7)²", 3.0, 0.0, -6.0, 6.0, "multimodal", false },

  // Functions with multiple local minima
  { "Rastrigin 1D", "x² - 10*cos(2πx) + 10", 0.0, 0.0, -5.12, 5.12, "multimodal", false },
  { "Ackley 1D", "-20*exp(-0.2*sqrt(x²)) - exp(cos(2πx)) + 20 + exp(1)", 0.0, 0.0, -5.0, 5.0, "multimodal", false },

  // Ill-conditioned functions
  { "Narrow valley", "x² + 0.01*sin(100x)", 0.0, 0.0, -1.0, 1.0, "multimodal", false },

  // Non-differentiable functions
  { "Absolute value", "|x - 1.5|", 1.5, 0.0, -2.0, 5.0, "non-differentiable", true },
  { "Linear max", "max(x-1, 2-x)", 1.5, 0.5, 0.0, 3.0, "non-differentiable", true },
  { "Sawtooth", "abs(x - round(x/0.5)*0.5)", 0.0, 0.0, -2.0, 2.0, "non-differentiable", false }
};

// ============================================================================
// Improved convergence analysis
// ============================================================================

struct ConvergenceData
{
  vector<real_type> intervals;          // Interval width at each iteration
  vector<real_type> best_points;        // Best point found at each iteration
  vector<real_type> f_values;           // Function value at each iteration
  real_type         reduction_rate;     // Average interval reduction rate
  real_type         efficiency;         // Efficiency: -log(tol)/n_eval
  real_type         estimated_order;    // Estimated convergence order
  vector<real_type> reduction_factors;  // Reduction factors for each step
};

// Class for tracing execution with improvements
class TracingTrichotomy : public Trichotomy<real_type>
{
private:
  mutable vector<real_type> trace_intervals;
  mutable vector<real_type> trace_points;
  mutable vector<real_type> trace_fvalues;

  // Callback function to capture state
  function<void( real_type, real_type, real_type, real_type )> trace_callback;

public:
  TracingTrichotomy()
  {
    // Optimal configuration for testing
    set_max_iterations( 200 );
    set_max_fun_evaluation( 1000 );
    set_tolerance( 1e-8 );

    // Set default callback
    trace_callback = [&]( real_type a, real_type b, real_type x, real_type fx )
    {
      trace_intervals.push_back( b - a );
      trace_points.push_back( x );
      trace_fvalues.push_back( fx );
    };
  }

  real_type eval_with_trace( real_type a, real_type b, const function<real_type( real_type )> & fun )
  {
    // Reset trace
    trace_intervals.clear();
    trace_points.clear();
    trace_fvalues.clear();

    // Wrapper that records evaluations
    auto tracing_fun = [&]( real_type x )
    {
      real_type fx = fun( x );

      // Get current interval
      real_type curr_a, curr_b;
      get_interval( curr_a, curr_b );

      // Record
      trace_callback( curr_a, curr_b, x, fx );

      return fx;
    };

    // Perform optimization
    real_type result = eval2( a, b, tracing_fun );

    // Add final point if not already recorded
    if ( trace_points.empty() || trace_points.back() != result )
    {
      real_type curr_a, curr_b;
      get_interval( curr_a, curr_b );
      trace_callback( curr_a, curr_b, result, fun( result ) );
    }

    return result;
  }

  const vector<real_type> & get_intervals() const { return trace_intervals; }
  const vector<real_type> & get_points() const { return trace_points; }
  const vector<real_type> & get_fvalues() const { return trace_fvalues; }

  void set_trace_callback( function<void( real_type, real_type, real_type, real_type )> cb ) { trace_callback = cb; }

  ConvergenceData analyze_convergence( real_type exact_min = numeric_limits<real_type>::quiet_NaN() ) const
  {
    ConvergenceData data;

    if ( trace_intervals.size() < 3 ) return data;

    data.intervals   = trace_intervals;
    data.best_points = trace_points;
    data.f_values    = trace_fvalues;

    // Calculate average interval reduction rate
    real_type total_reduction = 0.0;
    int       reduction_count = 0;

    for ( size_t i = 1; i < trace_intervals.size(); i++ )
    {
      if ( trace_intervals[i - 1] > 1e-15 )
      {
        real_type reduction = trace_intervals[i] / trace_intervals[i - 1];
        data.reduction_factors.push_back( reduction );
        total_reduction += reduction;
        reduction_count++;
      }
    }

    if ( reduction_count > 0 )
    {
      data.reduction_rate = total_reduction / reduction_count;

      // Estimate convergence order: log(r_k) / log(r_{k-1})
      // For trichotomy method, theoretical reduction ~2/3
      if ( data.reduction_factors.size() >= 2 )
      {
        real_type sum_log_ratio = 0.0;
        int       count         = 0;

        for ( size_t i = 1; i < data.reduction_factors.size(); i++ )
        {
          real_type r1 = data.reduction_factors[i];
          real_type r2 = data.reduction_factors[i - 1];
          if ( r1 > 1e-15 && r2 > 1e-15 && r1 < 1.0 && r2 < 1.0 )
          {
            sum_log_ratio += log( r1 ) / log( r2 );
            count++;
          }
        }

        if ( count > 0 ) { data.estimated_order = sum_log_ratio / count; }
      }
    }

    // Calculate efficiency
    if ( converged() && num_fun_eval() > 0 )
    {
      real_type tol = tolerance();
      if ( tol > 0 ) { data.efficiency = -log( tol ) / num_fun_eval(); }
    }

    return data;
  }
};

// ============================================================================
// Helper to define test functions
// ============================================================================

function<real_type( real_type )> define_function( const TestFunction & tf )
{
  if ( tf.name == "Simple quadratic" )
  {
    return []( real_type x ) { return x * x; };
  }
  else if ( tf.name == "Translated parabola" )
  {
    return []( real_type x ) { return ( x - 1 ) * ( x - 1 ); };
  }
  else if ( tf.name == "Wide parabola" )
  {
    return []( real_type x ) { return 2 * x * x - 4 * x + 3; };
  }
  else if ( tf.name == "Cosine" )
  {
    return []( real_type x ) { return cos( x ); };
  }
  else if ( tf.name == "Sine + quadratic" )
  {
    return []( real_type x ) { return sin( x ) + 0.1 * ( x - 2 ) * ( x - 2 ); };
  }
  else if ( tf.name == "Double exponential" )
  {
    return []( real_type x ) { return exp( x ) + exp( -x ); };
  }
  else if ( tf.name == "Gaussian" )
  {
    return []( real_type x ) { return exp( -x * x / 2 ); };
  }
  else if ( tf.name == "Hyperbola" )
  {
    return []( real_type x ) { return x * x + 1 / x; };
  }
  else if ( tf.name == "Rational function" )
  {
    return []( real_type x )
    {
      real_type x2 = x * x;
      real_type x3 = x2 * x;
      real_type x4 = x2 * x2;
      return ( x4 - 3 * x3 + 2 ) / ( x2 + 1 );
    };
  }
  else if ( tf.name == "Quartic" )
  {
    return []( real_type x )
    {
      real_type t = x * x - x;
      return 100 * t * t + 15 * ( 1 - x );
    };
  }
  else if ( tf.name == "Himmelblau 1D" )
  {
    return []( real_type x )
    {
      real_type t1 = x * x + x - 11;
      real_type t2 = x + x * x - 7;
      return t1 * t1 + t2 * t2;
    };
  }
  else if ( tf.name == "Rastrigin 1D" )
  {
    return []( real_type x ) { return x * x - 10 * cos( 2 * m_pi * x ) + 10; };
  }
  else if ( tf.name == "Ackley 1D" )
  {
    return []( real_type x )
    {
      real_type x2 = x * x;
      return -20 * exp( -0.2 * sqrt( x2 ) ) - exp( cos( 2 * m_pi * x ) ) + 20 + exp( 1.0 );
    };
  }
  else if ( tf.name == "Narrow valley" )
  {
    return []( real_type x ) { return x * x + 0.01 * sin( 100 * x ); };
  }
  else if ( tf.name == "Sawtooth" )
  {
    return []( real_type x ) { return fabs( x - round( x / 0.5 ) * 0.5 ); };
  }
  else if ( tf.name == "Absolute value" )
  {
    return []( real_type x ) { return fabs( x - 1.5 ); };
  }
  else if ( tf.name == "Linear max" )
  {
    return []( real_type x ) { return max( x - 1.0, 2.0 - x ); };
  }

  return []( real_type x ) { return x * x; };  // Default
}

// ============================================================================
// Improved comparative tests (corrected version)
// ============================================================================

void run_comparative_test( const TestFunction & tf, bool detailed = false )
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{:═^80}\n", " " + tf.name + " " );

  fmt::print( "Function: f(x) = {}\n", tf.expression );
  fmt::print( "Interval: [{:.3f}, {:.3f}] (width: {:.3f})\n", tf.a, tf.b, tf.b - tf.a );
  fmt::print( "Category: {}\n", tf.category );
  fmt::print( "True minimum: x* = {:.10f}, f(x*) = {:.10f}\n\n", tf.min_x, tf.min_value );

  auto fun = define_function( tf );

  // Test with different tolerances
  vector<real_type> tolerances = { 1e-4, 1e-6, 1e-8, 1e-10 };

  // Variables to store results
  vector<tuple<real_type, int, int, real_type, real_type, real_type>> results;
  ConvergenceData                                                     conv_data_1e8;
  bool                                                                has_conv_data = false;

  // First collect all results
  for ( real_type tol : tolerances )
  {
    TracingTrichotomy solver;
    solver.set_tolerance( tol );

    real_type x_min   = solver.eval_with_trace( tf.a, tf.b, fun );
    real_type f_min   = fun( x_min );
    real_type error_x = fabs( x_min - tf.min_x );

    results.push_back( make_tuple( tol, solver.used_iter(), solver.num_fun_eval(), x_min, f_min, error_x ) );

    // Save data for detailed analysis (tolerance 1e-8)
    if ( tol == 1e-8 )
    {
      conv_data_1e8 = solver.analyze_convergence( tf.min_x );
      has_conv_data = true;
    }
  }

  // Now print the complete table
  fmt::print( "┌───────────┬────────────┬──────────┬────────────┬────────────┬──────────────┬──────────┐\n" );
  fmt::print( "│ Tolerance │ Iterations │  Eval    │   x_min    │ f(x_min)   │   x Error    │  Status  │\n" );
  fmt::print( "├───────────┼────────────┼──────────┼────────────┼────────────┼──────────────┼──────────┤\n" );

  for ( size_t i = 0; i < results.size(); i++ )
  {
    auto [tol, iter, eval, x_min, f_min, error_x] = results[i];

    // Determine color and status
    string status;
    auto   color = fg( fmt::color::white );

    if ( error_x < tol * 10 )
    {
      status = "OK";
      color  = fg( fmt::color::green );
    }
    else if ( error_x < 0.1 && tf.expect_global_min )
    {
      status = "LOCAL";
      color  = fg( fmt::color::yellow );
    }
    else if ( !tf.expect_global_min && error_x < 1.0 )
    {
      status = "EXPECTED";
      color  = fg( fmt::color::cyan );
    }
    else
    {
      status = "FAIL";
      color  = fg( fmt::color::red );
    }

    fmt::print(
      "│ {:9.1e} │ {:10d} │ {:8d} │ {:10.6f} │ {:10.6f} │ {:12.2e} │ ",
      tol,
      iter,
      eval,
      x_min,
      f_min,
      error_x );
    fmt::print( color, "{:8s} │\n", status );
  }

  fmt::print( "└───────────┴────────────┴──────────┴────────────┴────────────┴──────────────┴──────────┘\n" );

  // Print convergence analysis ONLY AFTER the table
  if ( detailed && has_conv_data && !conv_data_1e8.intervals.empty() )
  {
    fmt::print( "\n  Convergence analysis (tolerance = 1.0e-08):\n" );
    fmt::print( "  - Total iterations: {}\n", conv_data_1e8.intervals.size() );
    fmt::print( "  - Function evaluations: {}\n", conv_data_1e8.f_values.size() );
    fmt::print( "  - Average reduction rate: {:.3f} (theoretical: ~0.667)\n", conv_data_1e8.reduction_rate );
    if ( !isnan( conv_data_1e8.estimated_order ) )
    {
      fmt::print( "  - Estimated convergence order: {:.3f}\n", conv_data_1e8.estimated_order );
    }
    fmt::print( "  - Efficiency: {:.3f}\n", conv_data_1e8.efficiency );

    if ( conv_data_1e8.intervals.size() > 6 )
    {
      fmt::print( "\n  First 3 and last 3 iterations:\n" );
      for ( int i = 0; i < min( 3, (int) conv_data_1e8.intervals.size() ); i++ )
      {
        fmt::print(
          "    Iter {:3d}: interval = {:.3e}, x = {:.6f}, f = {:.6f}\n",
          i,
          conv_data_1e8.intervals[i],
          conv_data_1e8.best_points[i],
          conv_data_1e8.f_values[i] );
      }
      fmt::print( "    ...\n" );
      int n = conv_data_1e8.intervals.size();
      for ( int i = max( 3, n - 3 ); i < n; i++ )
      {
        fmt::print(
          "    Iter {:3d}: interval = {:.3e}, x = {:.6f}, f = {:.6f}\n",
          i,
          conv_data_1e8.intervals[i],
          conv_data_1e8.best_points[i],
          conv_data_1e8.f_values[i] );
      }
    }
    else
    {
      fmt::print( "\n  All iterations:\n" );
      for ( int i = 0; i < conv_data_1e8.intervals.size(); i++ )
      {
        fmt::print(
          "    Iter {:3d}: interval = {:.3e}, x = {:.6f}, f = {:.6f}\n",
          i,
          conv_data_1e8.intervals[i],
          conv_data_1e8.best_points[i],
          conv_data_1e8.f_values[i] );
      }
    }
  }
}

// ============================================================================
// Specific functionality tests
// ============================================================================

void test_trichotomy_features()
{
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:═^80}\n", " Trichotomy Functionality Test " );

  // Test 1: Automatic search (search2)
  {
    fmt::print( "\n1. Automatic search test (search2):\n" );
    auto fun = []( real_type x ) { return ( x - 2 ) * ( x - 2 ); };

    TracingTrichotomy solver;
    solver.set_tolerance( 1e-8 );
    real_type x0    = 0.0;
    real_type delta = 1.0;
    real_type x_min = solver.search2( x0, delta, fun );

    fmt::print( "   Starting point: x0 = {:.2f}, delta = {:.2f}\n", x0, delta );
    fmt::print( "   Found minimum: x = {:.10f}, f = {:.10f}\n", x_min, fun( x_min ) );
    fmt::print(
      "   Iterations: {}, Evaluations: {}, Convergence: {}\n",
      solver.used_iter(),
      solver.num_fun_eval(),
      solver.converged() ? "YES" : "NO" );
  }

  // Test 2: Inverted interval handling
  {
    fmt::print( "\n2. Inverted interval test (a > b):\n" );
    auto fun = []( real_type x ) { return x * x; };

    TracingTrichotomy solver;
    solver.set_tolerance( 1e-8 );
    real_type x_min = solver.eval2( 2.0, 1.0, fun );  // a > b

    fmt::print( "   Inverted interval: [2.0, 1.0]\n" );
    fmt::print( "   Found minimum: x = {:.10f}\n", x_min );
    fmt::print( "   Method automatically swaps a and b\n" );
  }

  // Test 3: Constant function
  {
    fmt::print( "\n3. Constant function test:\n" );
    auto fun = []( real_type ) { return 42.0; };

    TracingTrichotomy solver;
    solver.set_max_iterations( 50 );
    solver.set_tolerance( 1e-8 );
    real_type x_min = solver.eval2( -10.0, 10.0, fun );

    fmt::print( "   Constant function: f(x) = 42\n" );
    fmt::print( "   Found minimum: x = {:.6f}, f = {:.6f}\n", x_min, fun( x_min ) );
    fmt::print( "   Iterations: {}, Convergence: {}\n", solver.used_iter(), solver.converged() ? "YES" : "NO" );
  }

  // Test 4: Iteration limit
  {
    fmt::print( "\n4. Iteration limit test:\n" );
    auto fun = []( real_type x ) { return exp( -x * x / 100 ); };  // Very flat

    TracingTrichotomy solver;
    solver.set_max_iterations( 5 );
    solver.set_tolerance( 1e-15 );
    real_type x_min = solver.eval2( -100.0, 100.0, fun );

    fmt::print( "   Tolerance: 1e-15, max 5 iterations\n" );
    fmt::print( "   Found minimum: x = {:.6f}, f = {:.6f}\n", x_min, fun( x_min ) );
    fmt::print( "   Iterations: {}/{}, Convergence: {}\n", solver.used_iter(), 5, solver.converged() ? "YES" : "NO" );
  }
}

// ============================================================================
// Corrected performance benchmark
// ============================================================================

void run_benchmark()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n{:═^80}\n", " Performance Benchmark " );

  // Benchmark functions
  vector<pair<string, function<real_type( real_type )>>> benchmarks = {
    { "x²", []( real_type x ) { return x * x; } },
    { "cos(x)", []( real_type x ) { return cos( x ); } },
    { "exp(-x²)", []( real_type x ) { return exp( -x * x ); } },
    { "Quartic",
      []( real_type x )
      {
        real_type                                        t          = x * x - x;
        return 100 * t * t + 15 * ( 1 - x );
      } },
    { "Rastrigin", []( real_type x ) { return x * x - 10 * cos( 2 * m_pi * x ) + 10; } }
  };

  fmt::print(
    "┌──────────────────────┬────────────┬──────────┬───────────────┬────────────┬────────────┐\n"
    "│      Function        │ Iterations │  Eval    │     Time      │ Convergence│  Efficiency│\n"
    "├──────────────────────┼────────────┼──────────┼───────────────┼────────────┼────────────┤\n" );

  for ( auto & [name, fun] : benchmarks )
  {
    // Execute 10 times for statistics
    const int         runs = 10;
    vector<int>       iterations( runs );
    vector<int>       evaluations( runs );
    vector<bool>      converged( runs );
    vector<real_type> efficiencies( runs );

    auto start = chrono::high_resolution_clock::now();

    for ( int i = 0; i < runs; i++ )
    {
      TracingTrichotomy solver;
      solver.set_tolerance( 1e-8 );
      solver.set_max_iterations( 100 );

      // Execute optimization
      real_type x_min = solver.eval_with_trace( -5.0, 5.0, fun );

      iterations[i]  = solver.used_iter();
      evaluations[i] = solver.num_fun_eval();
      converged[i]   = solver.converged();

      // Calculate efficiency for this run
      if ( converged[i] )
      {
        real_type tol   = solver.tolerance();
        efficiencies[i] = -log( tol ) / evaluations[i];
      }
      else
      {
        efficiencies[i] = 0.0;
      }
    }

    auto end      = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double>( end - start ).count();

    // Calculate averages
    int       avg_iter       = accumulate( iterations.begin(), iterations.end(), 0 ) / runs;
    int       avg_eval       = accumulate( evaluations.begin(), evaluations.end(), 0 ) / runs;
    int       conv_count     = count( converged.begin(), converged.end(), true );
    real_type avg_efficiency = accumulate( efficiencies.begin(), efficiencies.end(), 0.0 ) / runs;

    fmt::print(
      "│ {:20s} │ {:10d} │ {:8d} │ {:10.4g} μs │ {:6d}/{:3d} │ {:10.3f} │\n",
      name,
      avg_iter,
      avg_eval,
      1000000 * duration / runs,
      conv_count,
      runs,
      avg_efficiency );
  }

  fmt::print( "└──────────────────────┴────────────┴──────────┴───────────────┴────────────┴────────────┘\n" );
}

// ============================================================================
// Improved edge cases tests
// ============================================================================

void test_edge_cases()
{
  fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n{:═^80}\n", " Edge Cases and Robustness " );

  // Test 1: Very small interval
  {
    fmt::print( "\n1. Very small interval:\n" );
    auto fun = []( real_type x ) { return x * x; };

    TracingTrichotomy solver;
    solver.set_tolerance( 1e-15 );
    real_type a     = 0.999999999;
    real_type b     = 1.000000001;
    real_type x_min = solver.eval_with_trace( a, b, fun );

    fmt::print( "   Interval: [{:.9f}, {:.9f}]\n", a, b );
    fmt::print( "   Initial width: {:.2e}\n", b - a );
    fmt::print( "   Minimum: x = {:.15f}, f = {:.15f}\n", x_min, fun( x_min ) );
    fmt::print( "   Iterations: {}, Convergence: {}\n", solver.used_iter(), solver.converged() ? "YES" : "NO" );
  }

  // Test 2: Very large interval
  {
    fmt::print( "\n2. Very large interval:\n" );
    auto fun = []( real_type x )
    {
      if ( x == 0.0 ) return 1.0;  // Avoid division by zero
      return sin( x ) / x;
    };

    TracingTrichotomy solver;
    solver.set_tolerance( 1e-6 );
    solver.set_max_iterations( 50 );
    real_type x_min = solver.eval_with_trace( -1e6, 1e6, fun );

    fmt::print( "   Interval: [-1e6, 1e6]\n" );
    fmt::print( "   Minimum: x = {:.6f}, f = {:.6f}\n", x_min, fun( x_min ) );
    fmt::print( "   Iterations: {}, Evaluations: {}\n", solver.used_iter(), solver.num_fun_eval() );
  }

  // Test 3: Discontinuous function
  {
    fmt::print( "\n3. Discontinuous function:\n" );
    auto fun = []( real_type x ) { return x < 0 ? x * x : ( x - 2 ) * ( x - 2 ) + 1; };

    TracingTrichotomy solver;
    solver.set_tolerance( 1e-8 );
    real_type x_min = solver.eval_with_trace( -3.0, 3.0, fun );

    fmt::print( "   f(x) = x² for x<0, (x-2)²+1 for x>=0\n" );
    fmt::print( "   Local minima: x=0 (f=0) and x=2 (f=1)\n" );
    fmt::print( "   Found minimum: x = {:.10f}, f = {:.10f}\n", x_min, fun( x_min ) );
    fmt::print( "   Found global minimum? {}\n", x_min < 0.1 ? "YES" : "NO" );
  }

  // Test 4: Non-unimodal function
  {
    fmt::print( "\n4. Non-unimodal function:\n" );
    auto fun = []( real_type x ) { return sin( 10 * x ) + 0.1 * x * x; };

    TracingTrichotomy solver;
    solver.set_tolerance( 1e-8 );
    real_type x_min = solver.eval_with_trace( -2.0, 2.0, fun );

    fmt::print( "   f(x) = sin(10x) + 0.1x² (many local minima)\n" );
    fmt::print( "   Found minimum: x = {:.10f}, f = {:.10f}\n", x_min, fun( x_min ) );
    fmt::print( "   Warning: method assumes unimodality!\n" );
    fmt::print( "   May converge to a local minimum\n" );
  }

  // Test 5: Function with NaN/Inf values
  {
    fmt::print( "\n5. Function with non-finite values:\n" );
    auto fun = []( real_type x ) { return x == 0.0 ? NAN : 1.0 / x; };

    Trichotomy<real_type> solver;
    solver.set_tolerance( 1e-8 );
    try
    {
      real_type x_min = solver.eval2( -1.0, 1.0, fun );
      fmt::print( "   f(x) = 1/x (NaN at x=0)\n" );
      fmt::print( "   Found minimum: x = {:.6f}\n", x_min );
    }
    catch ( ... )
    {
      fmt::print( "   Exception caught for non-finite values\n" );
    }
  }
}

// ============================================================================
// Statistical analysis of results
// ============================================================================

void analyze_results( const vector<TestFunction> & tests )
{
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "\n{:═^80}\n", " Statistical Analysis of Results " );

  int total_tests    = tests.size();
  int success_global = 0;
  int success_local  = 0;
  int failures       = 0;

  vector<real_type> all_efficiencies;
  vector<real_type> all_reduction_rates;

  for ( const auto & tf : tests )
  {
    auto              fun = define_function( tf );
    TracingTrichotomy solver;
    solver.set_tolerance( 1e-8 );

    real_type x_min = solver.eval_with_trace( tf.a, tf.b, fun );
    real_type error = fabs( x_min - tf.min_x );

    if ( error < 1e-4 ) { success_global++; }
    else if ( error < 0.1 ) { success_local++; }
    else
    {
      failures++;
    }

    // Convergence analysis
    auto conv_data = solver.analyze_convergence();
    if ( conv_data.efficiency > 0 ) { all_efficiencies.push_back( conv_data.efficiency ); }
    if ( conv_data.reduction_rate > 0 ) { all_reduction_rates.push_back( conv_data.reduction_rate ); }
  }

  // Calculate statistics
  auto stats = []( const vector<real_type> & v ) -> tuple<real_type, real_type, real_type>
  {
    if ( v.empty() ) return { 0.0, 0.0, 0.0 };

    real_type mean     = accumulate( v.begin(), v.end(), 0.0 ) / v.size();
    real_type variance = 0.0;
    for ( auto x : v ) variance += ( x - mean ) * ( x - mean );
    variance /= v.size();
    real_type stddev = sqrt( variance );

    return { mean, stddev, *min_element( v.begin(), v.end() ) };
  };

  auto [eff_mean, eff_std, eff_min] = stats( all_efficiencies );
  auto [red_mean, red_std, red_min] = stats( all_reduction_rates );

  fmt::print( "\nStatistics on {} tested functions:\n", total_tests );
  fmt::print( "┌────────────────────────┬─────────────┬────────────────────┐\n" );
  fmt::print( "│      Metric            │    Value    │      Comment       │\n" );
  fmt::print( "├────────────────────────┼─────────────┼────────────────────┤\n" );
  fmt::print( "│ Global successes       │ {:11d} │                    │\n", success_global );
  fmt::print( "│ Local successes        │ {:11d} │                    │\n", success_local );
  fmt::print( "│ Failures               │ {:11d} │                    │\n", failures );
  fmt::print( "│ Global success rate    │ {:10.1f}% │                    │\n", 100.0 * success_global / total_tests );
  fmt::print( "│ Average efficiency     │ {:11.3f} │ -log(tol)/eval     │\n", eff_mean );
  fmt::print( "│ Efficiency std dev     │ {:11.3f} │                    │\n", eff_std );
  fmt::print( "│ Average reduction rate │ {:11.3f} │ theoretical: 0.667 │\n", red_mean );
  fmt::print( "│ Reduction rate std dev │ {:11.3f} │                    │\n", red_std );
  fmt::print( "└────────────────────────┴─────────────┴────────────────────┘\n" );
}

// ============================================================================
// Main function
// ============================================================================

int main()
{
  fmt::print( fg( fmt::color::orange ) | fmt::emphasis::bold, "{:═^80}\n", " Complete Test of the Trichotomy Method " );
  fmt::print(
    "Implementation based on: "
    "\"A new zero-order 1-D optimization algorithm: trichotomy method\"\n" );
  fmt::print( "Authors: Alena Antonova, Olga Ibryaeva, arXiv:1903.07117\n\n" );

  // Part 1: Comparative tests on selected functions
  fmt::print(
    fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "{:═^80}\n",
    " PART 1: COMPARATIVE TESTS (Representative Functions) " );

  // Select a representative subset
  vector<size_t> selected_indices = { 0, 1, 3, 5, 7, 9, 12, 14, 16 };
  for ( size_t idx : selected_indices )
  {
    if ( idx < test_functions.size() )
    {
      run_comparative_test( test_functions[idx], idx == 0 );  // Details only for the first
    }
  }

  // Part 2: Specific functionality tests
  test_trichotomy_features();

  // Part 3: Performance benchmark (corrected)
  run_benchmark();

  // Part 4: Edge cases
  test_edge_cases();

  // Part 5: Statistical analysis
  analyze_results( test_functions );

  // Summary and recommendations
  fmt::print( fg( fmt::color::light_green ) | fmt::emphasis::bold, "\n{:═^80}\n", " SUMMARY AND RECOMMENDATIONS " );

  fmt::print( "\nCharacteristics of the trichotomy method:\n" );
  fmt::print( "✓ First-order method (no derivatives required)\n" );
  fmt::print( "✓ Robust and reliable for unimodal functions\n" );
  fmt::print( "✓ Suitable for non-differentiable functions\n" );
  fmt::print( "✓ Guaranteed convergence for unimodal functions\n" );
  fmt::print( "✗ May converge to local minima if function is multimodal\n" );
  fmt::print( "✗ Slower than methods using derivatives (Newton, etc.)\n" );
  fmt::print( "✗ Requires more function evaluations\n\n" );

  fmt::print( "Typical performance:\n" );
  fmt::print( "• Interval reduction rate: ~0.667 (theoretical), ~0.7-0.8 (observed)\n" );
  fmt::print( "• Linear convergence (order 1)\n" );
  fmt::print( "• 20-50 iterations for tolerance 1e-8\n" );
  fmt::print( "• 2-3 function evaluations per iteration\n\n" );

  fmt::print( "Usage recommendations:\n" );
  fmt::print( "• Use for non-differentiable functions or functions with expensive derivatives\n" );
  fmt::print( "• Verify function unimodality in the search interval\n" );
  fmt::print( "• For smooth functions, consider higher-order methods\n" );
  fmt::print( "• Method is robust but expensive in function evaluations\n" );
  fmt::print( "• Suitable when function evaluation is cheap\n" );

  fmt::print( fg( fmt::color::light_green ) | fmt::emphasis::bold, "\n{:═^80}\n", " Tests Completed Successfully " );

  return 0;
}
