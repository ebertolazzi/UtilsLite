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

#include <iomanip>
#include <chrono>

#include "Utils_fmt.hh"
#include "Utils_nonlinear_system.hh"

namespace Utils
{

  using integer   = int;
  using real_type = double;

  // =========================================================================
  // TEST 1: Verifica delle soluzioni esatte
  // =========================================================================
  void test_exact_solutions()
  {
    using NS         = NonlinearSystem;
    const double tol = 1e-8;

    // ------------------------------------------------------------
    // Determina larghezza colonne dinamicamente
    // ------------------------------------------------------------
    size_t Wname = 10;
    for ( auto * sys : nonlinear_system_tests )
    {
      if ( !sys ) continue;
      Wname = std::max( Wname, sys->title().size() );
    }
    Wname += 2;

    const size_t Wsol  = 6;
    const size_t Wres  = 16;
    const size_t Wstat = 12;

    auto rep = []( size_t n, std::string const & s )
    {
      std::string res;
      for ( size_t k{ 0 }; k < n; ++k ) res += s;
      return res;
    };

    fmt::print( "\n" );

    // ------------------------------------------------------------
    // Header con bordo esterno doppio e interno singolo
    // ------------------------------------------------------------
    fmt::print(
      "╔{}╦{}╦{}╦{}╗\n",
      rep( Wname + 2, "═" ),
      rep( Wsol + 2, "═" ),
      rep( Wres + 2, "═" ),
      rep( Wstat + 2, "═" ) );

    // Header
    fmt::print(
      "║ {:^{}} ║ {:^{}} ║ {:^{}} ║ {:^{}} ║\n",
      "Test",
      Wname,
      "Sol#",
      Wsol,
      "Residuo",
      Wres,
      "Status",
      Wstat );

    fmt::print(
      "╠{}╬{}╬{}╬{}╣\n",
      rep( Wname + 2, "═" ),
      rep( Wsol + 2, "═" ),
      rep( Wres + 2, "═" ),
      rep( Wstat + 2, "═" ) );

    // ------------------------------------------------------------
    // Corpo tabella
    // ------------------------------------------------------------
    int total_tests  = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    int error_tests  = 0;

    for ( auto * sys : nonlinear_system_tests )
    {
      if ( !sys ) continue;

      std::vector<NS::Vector> sols;
      sys->exact_solution( sols );

      // Nessuna soluzione
      if ( sols.empty() )
      {
        fmt::print(
          "║ {:<{}} ║ {:>{}} ║ {:>{}} ║ {:^{}} ║\n",
          sys->title(),
          Wname,
          "-",
          Wsol,
          "-",
          Wres,
          "n/a",
          Wstat );
        continue;
      }

      // Soluzioni esatte
      for ( size_t i = 0; i < sols.size(); ++i )
      {
        total_tests++;
        const auto & x = sols[i];
        NS::Vector   f( sys->num_equations() );
        double       res = 0.0;
        bool         ok  = false;

        try
        {
          sys->evaluate( x, f );
          res = f.norm();
          ok  = ( res <= tol );

          if ( ok )
            passed_tests++;
          else
            failed_tests++;
        }
        catch ( ... )
        {
          error_tests++;
          fmt::print( "║ {:<{}} ║ {:>{}} ║ {:>{}} ║ ", sys->title(), Wname, int( i + 1 ), Wsol, "eval-err", Wres );

          fmt::print( fg( fmt::color::orange_red ), "{:^{}}", "ERR", Wstat );
          fmt::print( " ║\n" );
          continue;
        }

        // Prepara le stringhe con colori
        std::string res_str    = fmt::format( "{:.8e}", res );
        std::string status_str = ok ? "✓  OK" : "✗  NO";

        fmt::print( "║ {:<{}} ║ {:>{}} ║ ", sys->title(), Wname, int( i + 1 ), Wsol );

        // Colore per il residuo
        if ( ok ) { fmt::print( fg( fmt::color::green ), "{:>{}}", res_str, Wres ); }
        else
        {
          fmt::print( fg( fmt::color::red ), "{:>{}}", res_str, Wres );
        }

        fmt::print( " ║ " );

        // Colore per lo status
        if ( ok ) { fmt::print( fg( fmt::color::green ), "{:^{}}", status_str, Wstat ); }
        else
        {
          fmt::print( fg( fmt::color::red ), "{:^{}}", status_str, Wstat );
        }

        fmt::print( " ║\n" );
      }
    }

    // Footer
    fmt::print(
      "╚{}╩{}╩{}╩{}╝\n",
      rep( Wname + 2, "═" ),
      rep( Wsol + 2, "═" ),
      rep( Wres + 2, "═" ),
      rep( Wstat + 2, "═" ) );

    // Summary
    fmt::print( "\nExact Solutions Test Summary:\n" );
    fmt::print( "  Total solutions tested: {}\n", total_tests );
    fmt::print( "  Passed: {} ({:.1f}%)\n", passed_tests, ( 100.0 * passed_tests ) / total_tests );
    fmt::print( "  Failed: {} ({:.1f}%)\n", failed_tests, ( 100.0 * failed_tests ) / total_tests );
    fmt::print( "  Errors: {} ({:.1f}%)\n", error_tests, ( 100.0 * error_tests ) / total_tests );
    fmt::print( "  Tolerance: {:.1e}\n", tol );
  }

  // =========================================================================
  // TEST 2: Verifica dello Jacobiano
  // =========================================================================
  void test_jacobian_verification()
  {
    using NS                      = NonlinearSystem;
    const real_type fd_eps        = 1e-6;  // epsilon for finite differences
    const real_type tolerance     = 1e-5;
    const integer   max_dimension = 50;  // Skip tests with dimension > 10

    // ------------------------------------------------------------
    // Determine column widths dynamically (only for small systems)
    // ------------------------------------------------------------
    size_t Wname       = 10;
    size_t Wpoint      = 6;
    size_t Werror      = 16;
    size_t Wpos        = 8;
    size_t Wanalytical = 12;
    size_t Wfd         = 12;

    // Calculate Wname only for systems with dimension <= max_dimension
    for ( auto * sys : nonlinear_system_tests )
    {
      if ( !sys ) continue;
      if ( sys->num_equations() > max_dimension ) continue;
      Wname = std::max( Wname, sys->title().size() );
    }

    // Reduce the first column width and cap it at reasonable size
    Wname = std::min( Wname, size_t( 60 ) );  // Cap at 25 characters
    Wname += 2;

    auto rep = []( size_t n, const std::string & s )
    {
      std::string res;
      for ( size_t k = 0; k < n; ++k ) res += s;
      return res;
    };

    fmt::print(
      "\n╔{}╦{}╦{}╦{}╦{}╦{}╗\n",
      rep( Wname + 2, "═" ),
      rep( Wpoint + 2, "═" ),
      rep( Wpos + 2, "═" ),
      rep( Werror + 2, "═" ),
      rep( Wanalytical + 2, "═" ),
      rep( Wfd + 2, "═" ) );

    fmt::print(
      "║ {:^{}} ║ {:^{}} ║ {:^{}} ║ {:^{}} ║ {:^{}} ║ {:^{}} ║\n",
      "Test",
      Wname,
      "Point",
      Wpoint,
      "Position",
      Wpos,
      "Error",
      Werror,
      "Analytical",
      Wanalytical,
      "FiniteDiff",
      Wfd );

    fmt::print(
      "╠{}╬{}╬{}╬{}╬{}╬{}╣\n",
      rep( Wname + 2, "═" ),
      rep( Wpoint + 2, "═" ),
      rep( Wpos + 2, "═" ),
      rep( Werror + 2, "═" ),
      rep( Wanalytical + 2, "═" ),
      rep( Wfd + 2, "═" ) );

    int total_tests   = 0;
    int passed_tests  = 0;
    int failed_tests  = 0;
    int error_tests   = 0;
    int skipped_tests = 0;

    // Variabili per le statistiche di tempo
    double total_jacobian_time = 0.0;
    double total_fd_time       = 0.0;

    for ( auto * sys : nonlinear_system_tests )
    {
      if ( !sys ) continue;

      // Skip systems with dimension > max_dimension
      integer n_eq = sys->num_equations();
      if ( n_eq > max_dimension )
      {
        skipped_tests++;
        continue;
      }

      std::vector<NS::Vector> initial_points;
      sys->initial_points( initial_points );

      for ( size_t ip = 0; ip < initial_points.size(); ++ip )
      {
        const auto & x = initial_points[ip];

        try
        {
          // Timing per Jacobiano analitico
          auto start_jac = std::chrono::high_resolution_clock::now();

          // Compute analytical Jacobian
          NS::SparseMatrix jac_analytical( n_eq, n_eq );
          sys->jacobian( x, jac_analytical );

          auto   end_jac  = std::chrono::high_resolution_clock::now();
          double jac_time = std::chrono::duration<double>( end_jac - start_jac ).count();
          total_jacobian_time += jac_time;

          // Timing per differenze finite
          auto start_fd = std::chrono::high_resolution_clock::now();

          // Compute finite difference Jacobian
          NS::SparseMatrix jac_fd( n_eq, n_eq );
          NS::Vector       f_plus( n_eq ), f_minus( n_eq ), f_base( n_eq );

          // Evaluate at base point
          sys->evaluate( x, f_base );

          // Finite differences for each variable
          for ( integer j = 0; j < n_eq; ++j )
          {
            NS::Vector x_plus  = x;
            NS::Vector x_minus = x;

            real_type h = fd_eps * ( 1.0 + std::abs( x( j ) ) );
            x_plus( j ) += h;
            x_minus( j ) -= h;

            sys->evaluate( x_plus, f_plus );
            sys->evaluate( x_minus, f_minus );

            for ( integer i = 0; i < n_eq; ++i )
            {
              real_type fd_deriv      = ( f_plus( i ) - f_minus( i ) ) / ( 2.0 * h );
              jac_fd.coeffRef( i, j ) = fd_deriv;
            }
          }

          auto   end_fd  = std::chrono::high_resolution_clock::now();
          double fd_time = std::chrono::duration<double>( end_fd - start_fd ).count();
          total_fd_time += fd_time;

          // Compare Jacobians and find maximum error
          real_type max_error = 0.0;
          integer   max_i = -1, max_j = -1;
          real_type analytical_val = 0.0, fd_val = 0.0;

          for ( integer j = 0; j < n_eq; ++j )
          {
            for ( integer i = 0; i < n_eq; ++i )
            {
              real_type a = jac_analytical.coeff( i, j );
              real_type b = jac_fd.coeff( i, j );

              // Handle near-zero values
              real_type denom = std::max( 1.0, std::abs( a ) );
              real_type error = std::abs( a - b ) / denom;

              if ( error > max_error )
              {
                max_error      = error;
                max_i          = i;
                max_j          = j;
                analytical_val = a;
                fd_val         = b;
              }
            }
          }

          total_tests++;

          if ( max_error > tolerance )
          {
            failed_tests++;

            std::string pos_str = fmt::format( "({},{})", max_i, max_j );

            fmt::print( "║ {:<{}} ║ {:>{}} ║ {:>{}} ║ ", sys->title(), Wname, int( ip + 1 ), Wpoint, pos_str, Wpos );

            // Color coding for error
            if ( max_error > 1e-2 ) { fmt::print( fg( fmt::color::red ), "{:>{}.3e}", max_error, Werror ); }
            else if ( max_error > 1e-4 ) { fmt::print( fg( fmt::color::orange ), "{:>{}.3e}", max_error, Werror ); }
            else
            {
              fmt::print( fg( fmt::color::yellow ), "{:>{}.3e}", max_error, Werror );
            }

            fmt::print( " ║ {:>{}.3e} ║ {:>{}.3e} ║\n", analytical_val, Wanalytical, fd_val, Wfd );
          }
          else
          {
            passed_tests++;
          }
        }
        catch ( const std::exception & e )
        {
          total_tests++;
          error_tests++;

          fmt::print( "║ {:<{}} ║ {:>{}} ║ {:>{}} ║ ", sys->title(), Wname, int( ip + 1 ), Wpoint, "ERR", Wpos );

          fmt::print( fg( fmt::color::red ), "{:>{}}", "EXCEPTION", Werror );
          fmt::print( " ║ {:>{}} ║ {:>{}} ║\n", "-", Wanalytical, "-", Wfd );
        }
        catch ( ... )
        {
          total_tests++;
          error_tests++;

          fmt::print( "║ {:<{}} ║ {:>{}} ║ {:>{}} ║ ", sys->title(), Wname, int( ip + 1 ), Wpoint, "ERR", Wpos );

          fmt::print( fg( fmt::color::red ), "{:>{}}", "UNKNOWN", Werror );
          fmt::print( " ║ {:>{}} ║ {:>{}} ║\n", "-", Wanalytical, "-", Wfd );
        }
      }
    }

    fmt::print(
      "╚{}╩{}╩{}╩{}╩{}╩{}╝\n",
      rep( Wname + 2, "═" ),
      rep( Wpoint + 2, "═" ),
      rep( Wpos + 2, "═" ),
      rep( Werror + 2, "═" ),
      rep( Wanalytical + 2, "═" ),
      rep( Wfd + 2, "═" ) );

    // Summary
    fmt::print( "\nJacobian Verification Summary:\n" );
    fmt::print( "  Maximum dimension tested: {}\n", max_dimension );
    fmt::print( "  Total tests: {}\n", total_tests );
    fmt::print( "  Passed: {} ({:.1f}%)\n", passed_tests, ( 100.0 * passed_tests ) / total_tests );
    fmt::print( "  Failed: {} ({:.1f}%)\n", failed_tests, ( 100.0 * failed_tests ) / total_tests );
    fmt::print( "  Errors: {} ({:.1f}%)\n", error_tests, ( 100.0 * error_tests ) / total_tests );
    fmt::print( "  Skipped tests (dimension > {}): {}\n", max_dimension, skipped_tests );
    fmt::print( "  Tolerance: {:.1e}\n", tolerance );
    fmt::print( "  Finite difference epsilon: {:.1e}\n", fd_eps );

    // Timing statistics
    if ( total_tests > 0 )
    {
      fmt::print( "\nTiming Statistics:\n" );
      fmt::print( "  Total analytical Jacobian time: {:.3f} s\n", total_jacobian_time );
      fmt::print( "  Total finite differences time: {:.3f} s\n", total_fd_time );
      fmt::print( "  Average time per test:\n" );
      fmt::print( "    - Analytical: {:.3f} ms\n", ( total_jacobian_time / total_tests ) * 1000 );
      fmt::print( "    - Finite diff: {:.3f} ms\n", ( total_fd_time / total_tests ) * 1000 );
      fmt::print( "    - Total: {:.3f} ms\n", ( ( total_jacobian_time + total_fd_time ) / total_tests ) * 1000 );
      fmt::print( "  Speed ratio (FD/Analytical): {:.1f}x\n", total_fd_time / total_jacobian_time );
    }
  }

}  // namespace Utils

// =========================================================================
// MAIN FUNCTION
// =========================================================================
int main()
{
  using namespace std::chrono;

  auto total_start = high_resolution_clock::now();

  fmt::print( "=========================================\n" );
  fmt::print( "   NONLINEAR SYSTEM TEST SUITE\n" );
  fmt::print( "=========================================\n\n" );

  // Initialize test systems
  auto init_start = high_resolution_clock::now();
  Utils::init_nonlinear_system_tests();
  auto   init_end  = high_resolution_clock::now();
  double init_time = duration<double>( init_end - init_start ).count();

  fmt::print( "Initialization time: {:.3f} ms\n\n", init_time * 1000 );

  // Run exact solutions test
  fmt::print( "\n" );
  fmt::print( "╔════════════════════════════════════════════════════════════════╗\n" );
  fmt::print( "║                    EXACT SOLUTIONS TEST                        ║\n" );
  fmt::print( "╚════════════════════════════════════════════════════════════════╝\n" );

  auto exact_start = high_resolution_clock::now();
  Utils::test_exact_solutions();
  auto   exact_end  = high_resolution_clock::now();
  double exact_time = duration<double>( exact_end - exact_start ).count();

  // Run Jacobian verification test
  fmt::print( "\n" );
  fmt::print( "╔════════════════════════════════════════════════════════════════╗\n" );
  fmt::print( "║                  JACOBIAN VERIFICATION TEST                    ║\n" );
  fmt::print( "╚════════════════════════════════════════════════════════════════╝\n" );

  auto jac_start = high_resolution_clock::now();
  Utils::test_jacobian_verification();
  auto   jac_end  = high_resolution_clock::now();
  double jac_time = duration<double>( jac_end - jac_start ).count();

  auto   total_end  = high_resolution_clock::now();
  double total_time = duration<double>( total_end - total_start ).count();

  // Final summary
  fmt::print( "\n" );
  fmt::print( "╔════════════════════════════════════════════════════════════════╗\n" );
  fmt::print( "║                         FINAL SUMMARY                          ║\n" );
  fmt::print( "╠════════════════════════════════════════════════════════════════╣\n" );
  fmt::print( "║  Test                         Time (ms)   Percentage          ║\n" );
  fmt::print( "╠════════════════════════════════════════════════════════════════╣\n" );
  fmt::print(
    "║  Initialization               {:>8.1f}      {:>6.1f}%%          ║\n",
    init_time * 1000,
    ( init_time / total_time ) * 100 );
  fmt::print(
    "║  Exact Solutions Test         {:>8.1f}      {:>6.1f}%%          ║\n",
    exact_time * 1000,
    ( exact_time / total_time ) * 100 );
  fmt::print(
    "║  Jacobian Verification Test   {:>8.1f}      {:>6.1f}%%          ║\n",
    jac_time * 1000,
    ( jac_time / total_time ) * 100 );
  fmt::print( "╠════════════════════════════════════════════════════════════════╣\n" );
  fmt::print( "║  TOTAL                        {:>8.1f}      {:>6.1f}%%          ║\n", total_time * 1000, 100.0 );
  fmt::print( "╚════════════════════════════════════════════════════════════════╝\n" );

  fmt::print( "\nAll tests completed in {:.3f} seconds!\n", total_time );

  return 0;
}
