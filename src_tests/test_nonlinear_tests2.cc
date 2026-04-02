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

#include "Utils_fmt.hh"
#include "Utils_nonlinear_system.hh"

namespace Utils
{

  using integer   = int;
  using real_type = double;

  void test_jacobian_verification()
  {
    using NS = NonlinearSystem;
    // const real_type eps = 1e-8;
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
    int failed_tests  = 0;
    int skipped_tests = 0;

    for ( auto * sys : nonlinear_system_tests )
    {
      if ( !sys ) continue;

      // Skip systems with dimension > max_dimension
      integer n_eq = static_cast<integer>( sys->num_equations() );
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
          // Compute analytical Jacobian
          NS::SparseMatrix jac_analytical( n_eq, n_eq );
          sys->jacobian( x, jac_analytical );

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
        }
        catch ( const std::exception & )
        {
          total_tests++;
          failed_tests++;

          fmt::print( "║ {:<{}} ║ {:>{}} ║ {:>{}} ║ ", sys->title(), Wname, int( ip + 1 ), Wpoint, "ERR", Wpos );

          fmt::print( fg( fmt::color::red ), "{:>{}}", "EXCEPTION", Werror );
          fmt::print( " ║ {:>{}} ║ {:>{}} ║\n", "-", Wanalytical, "-", Wfd );
        }
        catch ( ... )
        {
          total_tests++;
          failed_tests++;

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
    fmt::print( "  Failed tests: {} ({:.1f}%)\n", failed_tests, ( 100.0 * failed_tests ) / total_tests );
    fmt::print( "  Skipped tests (dimension > {}): {}\n", max_dimension, skipped_tests );
    fmt::print( "  Tolerance: {:.1e}\n", tolerance );
    fmt::print( "  Finite difference epsilon: {:.1e}\n", fd_eps );
  }

}  // namespace Utils

int main()
{
  Utils::init_nonlinear_system_tests();
  Utils::test_jacobian_verification();
  fmt::print( "\nJacobian verification completed!\n" );
  return 0;
}
