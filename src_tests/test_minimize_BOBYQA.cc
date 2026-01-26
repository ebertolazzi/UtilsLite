/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  BOBYQA test suite per problemi di ottimizzazione senza gradienti        |
 |                                                                          |
 |  Adattamento per la classe BOBYQA_minimizer di Utils_minimize_BOBYQA.hh  |
\*--------------------------------------------------------------------------*/

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "Utils_minimize_BOBYQA.hh"

using Scalar  = double;
using Vector  = Utils::BOBYQA_minimizer<Scalar>::Vector;
using Status  = Utils::BOBYQA_minimizer<Scalar>::Status;
using integer = Utils::BOBYQA_minimizer<Scalar>::integer;

// ===========================================================================
// MIGLIORAMENTO: Struttura TestResult con norma gradiente stimata
// ===========================================================================
struct TestResult
{
  std::string problem_name;
  std::string linesearch_name;
  bool        converged;
  size_t      iterations;
  size_t      function_evals;
  Scalar      final_f;
  Vector      final_x;
  size_t      dimension;
  Scalar      estimated_gradient_norm{ 0.0 };  // NUOVO: stima norma gradiente
};

// Struttura per statistiche
struct LineSearchStats
{
  std::string name;
  size_t      total_tests{ 0 };
  size_t      successful_tests{ 0 };
  size_t      total_iterations{ 0 };
  size_t      total_function_evals{ 0 };
  Scalar      avg_gradient_norm{ 0.0 };  // NUOVO: media norma gradiente stimata
};

// Collettore globale
std::vector<TestResult>                global_test_results;
std::map<std::string, LineSearchStats> line_search_statistics;

#include "ND_func.cxx"

// ===========================================================================
// MIGLIORAMENTO: Funzione per stimare la norma del gradiente via differenza
// finita
// ===========================================================================
Scalar
estimate_gradient_norm( std::function<Scalar( Vector const & )> const & f, Vector const & x, Scalar epsilon = 1e-6 )
{
  integer n = x.size();
  Vector  grad( n );
  Vector  x_pert = x;

  for ( integer i = 0; i < n; ++i )
  {
    // Forward difference
    x_pert( i )    = x( i ) + epsilon;
    Scalar f_plus  = f( x_pert );
    x_pert( i )    = x( i ) - epsilon;
    Scalar f_minus = f( x_pert );
    x_pert( i )    = x( i );  // reset

    grad( i ) = ( f_plus - f_minus ) / ( 2.0 * epsilon );
  }

  return grad.norm();
}

// ===========================================================================
// Aggiorna statistiche con norma gradiente stimata
// ===========================================================================
void
update_line_search_statistics( const TestResult & result )
{
  auto & stats = line_search_statistics[result.linesearch_name];
  stats.name   = result.linesearch_name;
  stats.total_tests++;

  if ( result.converged )
  {
    stats.successful_tests++;
    stats.total_iterations += result.iterations;
    stats.total_function_evals += result.function_evals;
    stats.avg_gradient_norm += result.estimated_gradient_norm;
  }
}

// ===========================================================================
// Funzione per formattare il vettore
// ===========================================================================
inline std::string
format_reduced_vector( Vector const & v, size_t max_size = 10 )
{
  std::string tmp{ "[" };
  integer     v_size = v.size();
  if ( v_size <= max_size )
  {
    for ( integer i = 0; i < v_size; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
  }
  else
  {
    for ( integer i{ 0 }; i < max_size - 3; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
    tmp += "..., ";
    for ( integer i{ v_size - 3 }; i < v_size; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
  }
  tmp.pop_back();
  tmp.pop_back();
  tmp += "]";
  return tmp;
}

// ===========================================================================
// Test runner BOBYQA con stima norma gradiente
// ===========================================================================
template <typename Problem>
void
test( Problem & prob, std::string const & problem_name )
{
  fmt::print(
    fmt::fg( fmt::color::cyan ),
    "\n"
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║ TEST FUNCTION: {:<47} ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n",
    problem_name );

  // Dimensione del problema e punto iniziale
  Vector                                         x0 = prob.init();
  const Utils::BOBYQA_minimizer<Scalar>::integer n  = x0.size();

  // Bound del problema
  Vector lower = prob.lower();
  Vector upper = prob.upper();

  // Calcolo NPT
  Utils::BOBYQA_minimizer<Scalar>::integer npt     = 2 * n + 1;
  Utils::BOBYQA_minimizer<Scalar>::integer npt_min = n + 2;
  Utils::BOBYQA_minimizer<Scalar>::integer npt_max = ( n + 1 ) * ( n + 2 ) / 2;

  if ( npt < npt_min ) npt = npt_min;
  if ( npt > npt_max ) npt = npt_max;

  // Wrapper per la funzione obiettivo che conta le valutazioni
  size_t nfev   = 0;
  auto   objfun = [&prob, &nfev]( Vector const & x ) -> Scalar
  {
    ++nfev;
    return prob( x );
  };

  // Creazione dell'ottimizzatore BOBYQA
  Utils::BOBYQA_minimizer<Scalar> optimizer;

  // Impostazione dei parametri
  optimizer.set_verbosity( 1 );
  optimizer.set_maxfun( 5000 );
  optimizer.set_rho( 0.1, 0.001 );

  // Chiamata all'algoritmo di ottimizzazione
  Status status = optimizer.minimize( n, npt, objfun, x0, lower, upper );

  // Valutazione finale
  Scalar final_f = prob( x0 );

  // ===========================================================================
  // MIGLIORAMENTO: Stima della norma del gradiente finale
  // ===========================================================================
  Scalar estimated_grad_norm = estimate_gradient_norm( [&prob]( Vector const & x ) { return prob( x ); }, x0, 1e-6 );

  // Costruzione del risultato
  TestResult result;
  result.problem_name            = problem_name;
  result.linesearch_name         = "BOBYQA";
  result.converged               = ( status == Status::BOBYQA_SUCCESS );
  result.iterations              = nfev;
  result.function_evals          = nfev;
  result.final_f                 = final_f;
  result.final_x                 = x0;
  result.dimension               = static_cast<size_t>( n );
  result.estimated_gradient_norm = estimated_grad_norm;

  global_test_results.push_back( result );
  update_line_search_statistics( result );

  // Output dei risultati
  std::string status_str = result.converged ? "CONVERGED" : "FAILED";

  // Colori per la norma del gradiente stimata
  auto grad_color = ( estimated_grad_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                    : ( estimated_grad_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                     : fmt::fg( fmt::color::red );

  fmt::print( "{} - BOBYQA: {} after {} evaluations\n", problem_name, status_str, nfev );
  fmt::print( "   f = {:.6e}, ‖g_estimated‖ = ", final_f );
  fmt::print( grad_color, "{:.6e}\n", estimated_grad_norm );
  fmt::print( "   Solution: {}\n\n", format_reduced_vector( x0, 10 ) );
}

// ===========================================================================
// Tabella riassuntiva colorata per BOBYQA
// ===========================================================================
void
print_summary_table()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔════════════════════════════════════════════════════════════════"
    "════════════════════════╗\n"
    "║                                 BOBYQA GLOBAL SUMMARY          "
    "                        ║\n"
    "╠════════════════════════╤════════╤══════════╤════════════════╤══"
    "════════════╤═══════════╣\n"
    "║ Function               │ Dim    │ Eval     │ Final Value    │ "
    "‖g_estimated‖│ Status    ║\n"
    "╠════════════════════════╪════════╪══════════╪════════════════╪══"
    "════════════╪═══════════╣\n" );

  for ( auto const & result : global_test_results )
  {
    std::string status_str = result.converged ? "CONVERGED" : "FAILED";

    // Colori per lo status
    auto status_color = result.converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );

    // Colori per la norma del gradiente stimata
    auto grad_color = ( result.estimated_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                      : ( result.estimated_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                                  : fmt::fg( fmt::color::red );

    std::string problem_name = result.problem_name;
    if ( problem_name.length() > 22 ) { problem_name = problem_name.substr( 0, 19 ) + "..."; }

    fmt::print(
      "║ {:<22} │ {:>6} │ {:>8} │ {:<14.6g} │ ",
      problem_name,
      result.dimension,
      result.function_evals,
      result.final_f );

    // Norma gradiente stimata colorata
    fmt::print( grad_color, "{:<12.6g}", result.estimated_gradient_norm );
    fmt::print( " │ " );

    // Status colorato
    fmt::print( status_color, "{:<9}", status_str );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚════════════════════════╧════════╧══════════╧════════════════╧══"
    "════════════╧═══════════╝\n" );
}

// ===========================================================================
// Statistiche delle line search (per BOBYQA c'è solo BOBYQA)
// ===========================================================================
void
print_line_search_statistics()
{
  // Calcola medie finali
  for ( auto & [name, stats] : line_search_statistics )
  {
    if ( stats.successful_tests > 0 ) { stats.avg_gradient_norm /= stats.successful_tests; }
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔════════════════════════════════════════════════════════════════════════╗\n"
    "║                            BOBYQA SUMMARY STATISTICS                   ║\n"
    "╠═══════════════════╤══════════╤═════════════╤════════════╤══════════════╣\n"
    "║ Optimizer         │ Tests    │ Success %   │ Avg Eval   │ Avg ‖g_est‖  ║\n"
    "╠═══════════════════╪══════════╪═════════════╪════════════╪══════════════╣\n" );

  for ( auto const & [_, s] : line_search_statistics )
  {
    Scalar success_rate = ( s.total_tests > 0 ) ? 100.0 * s.successful_tests / s.total_tests : 0.0;

    Scalar avg_eval = ( s.successful_tests > 0 ) ? static_cast<Scalar>( s.total_function_evals ) / s.successful_tests
                                                 : 0.0;

    auto color = ( success_rate >= 80.0 )   ? fmt::fg( fmt::color::green )
                 : ( success_rate >= 60.0 ) ? fmt::fg( fmt::color::yellow )
                                            : fmt::fg( fmt::color::red );

    auto grad_color = ( s.avg_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                      : ( s.avg_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                       : fmt::fg( fmt::color::red );

    fmt::print( "║ {:<17} │ {:>8} │ ", s.name, s.total_tests );
    fmt::print( color, "{:>10.1f}% ", success_rate );
    fmt::print( "│ {:>10.1f} │ ", avg_eval );
    fmt::print( grad_color, "{:>12.4g}", s.avg_gradient_norm );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚═══════════════════╧══════════╧═════════════╧════════════╧══════════════╝\n" );

  // Statistiche globali
  size_t total_tests     = global_test_results.size();
  size_t converged_tests = std::count_if(
    global_test_results.begin(),
    global_test_results.end(),
    []( const TestResult & r ) { return r.converged; } );

  size_t accumulated_evals{ 0 };
  Scalar total_grad_norm{ 0.0 };
  size_t grad_count{ 0 };

  for ( auto const & r : global_test_results )
  {
    if ( r.converged )
    {
      accumulated_evals += r.function_evals;
      total_grad_norm += r.estimated_gradient_norm;
      grad_count++;
    }
  }

  fmt::print( fmt::fg( fmt::color::light_blue ), "\n📊 Global Statistics:\n" );
  fmt::print( "   • Total problems: {}\n", total_tests );
  fmt::print( "   • Converged: {} ({:.1f}%)\n", converged_tests, ( 100.0 * converged_tests / total_tests ) );
  fmt::print( "   • Total function evaluations: {}\n", accumulated_evals );
  if ( grad_count > 0 ) { fmt::print( "   • Average estimated ‖g‖: {:.2g}\n", total_grad_norm / grad_count ); }
}

// ===========================================================================
// MAIN
// ===========================================================================
int
main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║              BOBYQA Optimization Test Suite                    ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  // Test in ordine alfabetico
  AckleyN<Scalar, 15> ackley;
  test( ackley, "AckleyN15D" );

  Beale2D<Scalar> beale;
  test( beale, "Beale2D" );

  Booth2D<Scalar> booth;
  test( booth, "Booth2D" );

  BroydenTridiagonalN<Scalar, 12> broy;
  test( broy, "BroydenTridiagonal12D" );

  BrownAlmostLinearN<Scalar, 10> brown;
  test( brown, "BrownAlmostLinear10D" );

  ExtendedWoodN<Scalar, 16> woodN;
  test( woodN, "ExtendedWood16D" );

  FreudensteinRoth2D<Scalar> fr;
  test( fr, "FreudensteinRoth2D" );

  GriewankN<Scalar, 10> griewank;
  test( griewank, "GriewankN10D" );

  HelicalValley3D<Scalar> heli;
  test( heli, "HelicalValley3D" );

  Himmelblau2D<Scalar> himm;
  test( himm, "Himmelblau2D" );

  IllConditionedQuadraticN<Scalar, 20> illq;
  test( illq, "IllConditionedQuadratic20D" );

  LevyN<Scalar, 10> levy;
  test( levy, "LevyN10D" );

  Matyas2D<Scalar> matyas;
  test( matyas, "Matyas2D" );

  McCormick2D<Scalar> mccormick;
  test( mccormick, "McCormick2D" );

  MichalewiczN<Scalar, 10> michalewicz;
  test( michalewicz, "MichalewiczN10D" );

  NesterovChebyshevRosenbrock<Scalar, 128> nesterov;
  test( nesterov, "NesterovChebyshevRosenbrock128D" );

  PowellBadlyScaled2D<Scalar> pbs;
  test( pbs, "PowellBadlyScaled2D" );

  PowellSingularN<Scalar, 16> powellN;
  test( powellN, "PowellSingular16D" );

  RastriginN<Scalar, 15> rastrigin;
  test( rastrigin, "RastriginN15D" );

  Rosenbrock2D<Scalar> rosen;
  test( rosen, "Rosenbrock2D" );

  RosenbrockN<Scalar, 10> rosenN;
  test( rosenN, "Rosenbrock10D" );

  Schaffer2D<Scalar> schaffer;
  test( schaffer, "Schaffer2D" );

  SchwefelN<Scalar, 15> schwefel;
  test( schwefel, "SchwefelN15D" );

  ThreeHumpCamel2D<Scalar> camel;
  test( camel, "ThreeHumpCamel2D" );

  TrigonometricSumN<Scalar, 15> trig;
  test( trig, "TrigonometricSum15D" );

  // Stampa dei risultati
  print_summary_table();
  print_line_search_statistics();

  return 0;
}
