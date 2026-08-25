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

#include "Utils_AlgoBracket.hh"
#include "Utils_fmt.hh"

using namespace std;

using Utils::AlgoBracket;

#include "1D_fun.cxx"

constexpr int NA = 8;

static int ntest        = 0;
static int nfuneval[NA] = { 0, 0, 0, 0, 0, 0, 0, 0 };
static int niter[NA]    = { 0, 0, 0, 0, 0, 0, 0, 0 };
static int nconv[NA]    = { 0, 0, 0, 0, 0, 0, 0, 0 };

template <typename FUN> void do_solve( string_view name, real_type a, real_type b, FUN f )
{
  fmt::print( "\n" );
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "  ┌──────────────────────────────────────────────────────────────────────────┐\n" );
  fmt::print( fmt::fg( fmt::color::light_blue ), "  │{:^74}│\n", fmt::format( "Test #{:03d}", ++ntest ) );
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "  └──────────────────────────────────────────────────────────────────────────┘\n" );
  fmt::print( "  Function: {}\n", name );
  fmt::print( "  Interval: [{:.6f}, {:.6f}]\n\n", a, b );

  // Intestazione tabella
  fmt::print( "  {0:─^76}\n", "" );
  fmt::print(
    "  {:<15} {:>6} {:>5} {:>4} {:>12} {:>16} {:>10}\n",
    "Algorithm",
    "Evals",
    "Iter",
    "Conv",
    "x_min",
    "f(x_min)",
    "b-a" );
  fmt::print( "  {0:─^76}\n", "" );

  for ( const unsigned ialgo : { 0, 1, 2, 3, 4, 5, 6 } )
  {
    AlgoBracket<real_type> solver;
    solver.select( ialgo );
    real_type res  = solver.eval2( a, b, f );
    real_type fres = f( res );
    nfuneval[ialgo] += solver.num_fun_eval();
    niter[ialgo] += solver.used_iter();

    if ( solver.converged() ) { nconv[ialgo]++; }

    // Formattazione condizionale per valori vicini a zero
    string x_str, f_str, interval_str;

    // Formatta x_min
    if ( abs( res ) < 1e-10 ) { x_str = fmt::format( fmt::fg( fmt::color::dark_gray ), "{:12.2e}", res ); }
    else
    {
      x_str = fmt::format( "{:12.6f}", res );
    }

    // Formatta f(x_min)
    if ( abs( fres ) < 1e-10 ) { f_str = fmt::format( fmt::fg( fmt::color::dark_gray ), "{:16.2e}", fres ); }
    else
    {
      f_str = fmt::format( "{:16.6e}", fres );
    }

    // Formatta b-a
    real_type interval = solver.b() - solver.a();
    if ( interval < 1e-10 ) { interval_str = fmt::format( fmt::fg( fmt::color::green ), "{:10.2e}", interval ); }
    else if ( interval > 1.0 ) { interval_str = fmt::format( fmt::fg( fmt::color::yellow ), "{:10.6f}", interval ); }
    else
    {
      interval_str = fmt::format( "{:10.6f}", interval );
    }

    // Status di convergenza
    string conv_status;
    if ( solver.converged() )
    {
      conv_status = fmt::format( fmt::fg( fmt::color::green ) | fmt::emphasis::bold, "{:^4}", "✓" );
    }
    else
    {
      conv_status = fmt::format( fmt::fg( fmt::color::red ), "{:^4}", "✗" );
    }

    fmt::print(
      "  {:<15} {:>6} {:>5} {} {} {} {}\n",
      solver.algo(),
      solver.num_fun_eval(),
      solver.used_iter(),
      conv_status,
      x_str,
      f_str,
      interval_str );
  }

  fmt::print( "  {0:─^76}\n", "" );
}

int main()
{
  // Titolo principale
  fmt::print(
    fmt::fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "  ╔══════════════════════════════════════════════════════════════════════════╗\n"
    "  ║                  ALGORITHM BRACKETING TEST SUITE                         ║\n"
    "  ╚══════════════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  std::vector<std::unique_ptr<fun1D>> f_list;
  build_1dfun_list( f_list );

  for ( const auto & f : f_list ) do_solve( f->info(), f->a0(), f->b0(), f->function() );

  // Riepilogo finale
  fmt::print(
    fmt::fg( fmt::color::yellow ) | fmt::emphasis::bold,
    "\n\n"
    "  ┌──────────────────────────────────────────────────────────────────────────┐\n"
    "  │                            FINAL SUMMARY                                 │\n"
    "  └──────────────────────────────────────────────────────────────────────────┘\n"
    "\n" );

  // Intestazione tabella riepilogo
  fmt::print( "  {0:─^76}\n", "" );
  fmt::print( "  {:<15} {:>10} {:>10} {:>10} {:>12}\n", "Algorithm", "Evals", "Iter", "Conv", "Evals/Iter" );
  fmt::print( "  {0:─^76}\n", "" );

  // Calcola valori min/max per evidenziazione
  int min_eval = INT_MAX;
  int max_eval = 0;
  int min_iter = INT_MAX;

  for ( unsigned ialgo = 0; ialgo < 7; ialgo++ )
  {
    if ( nfuneval[ialgo] < min_eval ) min_eval = nfuneval[ialgo];
    if ( nfuneval[ialgo] > max_eval ) max_eval = nfuneval[ialgo];
    if ( niter[ialgo] < min_iter ) min_iter = niter[ialgo];
  }

  // Stampa dati riepilogo
  for ( unsigned const ialgo : { 0, 1, 2, 3, 4, 5, 6 } )
  {
    AlgoBracket<real_type> solver;
    solver.select( ialgo );

    double evals_per_iter = ( niter[ialgo] > 0 ) ? static_cast<double>( nfuneval[ialgo] ) / niter[ialgo] : 0.0;

    // Evidenzia i migliori risultati
    fmt::text_style eval_style = fmt::text_style();
    fmt::text_style iter_style = fmt::text_style();

    if ( nfuneval[ialgo] == min_eval ) { eval_style = fmt::fg( fmt::color::green ) | fmt::emphasis::bold; }
    else if ( nfuneval[ialgo] == max_eval ) { eval_style = fmt::fg( fmt::color::red ); }

    if ( niter[ialgo] == min_iter ) { iter_style = fmt::fg( fmt::color::green ) | fmt::emphasis::bold; }

    fmt::print( "  {:<15} ", solver.algo() );
    fmt::print( eval_style, "{:>10}", nfuneval[ialgo] );
    fmt::print( iter_style, "{:>10}", niter[ialgo] );
    fmt::print( " {:>10}/{} ", nconv[ialgo], ntest );
    fmt::print( "{:>12.2f}\n", evals_per_iter );
  }

  fmt::print( "  {0:─^76}\n", "" );

  // Statistiche aggiuntive
  fmt::print( "\n" );
  fmt::print( fmt::fg( fmt::color::light_gray ), "  Statistics:\n" );
  fmt::print( fmt::fg( fmt::color::light_gray ), "  • Total tests executed: {}\n", ntest );
  fmt::print( fmt::fg( fmt::color::light_gray ), "  • Most efficient algorithm: " );

  // Trova l'algoritmo più efficiente
  vector<pair<int, string>> efficiency;
  for ( unsigned ialgo = 0; ialgo < 7; ialgo++ )
  {
    AlgoBracket<real_type> solver;
    solver.select( ialgo );
    efficiency.emplace_back( nfuneval[ialgo], solver.algo() );
  }

  sort( efficiency.begin(), efficiency.end() );

  size_t sz = efficiency.size();
  if ( sz > 3 ) sz = 3;

  for ( size_t i = 0; i < sz; i++ )
  {
    if ( i == 0 ) { fmt::print( fmt::fg( fmt::color::green ) | fmt::emphasis::bold, "{}", efficiency[i].second ); }
    else
    {
      fmt::print( fmt::fg( fmt::color::light_gray ), ", {}", efficiency[i].second );
    }
    if ( i < 2 && i < efficiency.size() - 1 ) { fmt::print( fmt::fg( fmt::color::light_gray ), " < " ); }
  }

  fmt::print( "\n\n" );
  fmt::print(
    fmt::fg( fmt::color::green ) | fmt::emphasis::bold,
    "  ╔══════════════════════════════════════════════════════════════════════════╗\n"
    "  ║                  ALL TESTS COMPLETED SUCCESSFULLY                        ║\n"
    "  ╚══════════════════════════════════════════════════════════════════════════╝\n" );

  return 0;
}
