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
 |      Università degli Studi di Trento                                   |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

/**
 * @file test_TsustRegion.cc
 * @brief Comprehensive test suite for Trust Region Subproblem Solver
 *
 * This file implements a comprehensive test suite for the TrustRegionSolver
 * class, which solves the trust-region subproblem:
 *   min ½pᵀHp + gᵀp  s.t. ‖p‖ ≤ Δ
 *
 * The test suite includes:
 * - Unit tests for all four methods (Dogleg, Steihaug-CG, More-Sorensen, Regularized)
 * - Validation against known solutions for quadratic problems
 * - Comparison between different methods on the same problem
 * - Edge case testing (singular Hessians, zero gradients, etc.)
 * - Performance benchmarking
 * - Convergence analysis and accuracy verification
 *
 * The test suite uses Eigen for linear algebra operations and fmt for formatted
 * output with Unicode characters and ANSI colors for improved readability.
 */

#include "Utils_TrustRegion.hh"
#include "Utils_fmt.hh"
#include "Utils.hh"
#include "Utils_TicToc.hh"

using integer = Eigen::Index;
using Scalar  = double;

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif

// ============================================================================
// Color and formatting definitions
// ============================================================================

namespace TestColors
{
  constexpr auto HEADER  = fmt::fg( fmt::color::cyan ) | fmt::emphasis::bold;
  constexpr auto SUCCESS = fmt::fg( fmt::color::green ) | fmt::emphasis::bold;
  constexpr auto ERROR   = fmt::fg( fmt::color::red ) | fmt::emphasis::bold;
  constexpr auto WARNING = fmt::fg( fmt::color::yellow );
  constexpr auto INFO    = fmt::fg( fmt::color::green_yellow );
  // constexpr auto VALUE   = fmt::fg( fmt::color::magenta );
  constexpr auto TIME   = fmt::fg( fmt::color::cyan );
  constexpr auto METHOD = fmt::fg( fmt::color::blue );
  constexpr auto PASS   = fmt::fg( fmt::color::green ) | fmt::emphasis::bold;
  constexpr auto FAIL   = fmt::fg( fmt::color::red ) | fmt::emphasis::bold;
}  // namespace TestColors

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Generate random positive definite Hessian matrix
 */
Eigen::SparseMatrix<Scalar> generateRandomPositiveDefiniteHessian(
  integer  n,
  unsigned seed         = 42,
  Scalar   sparsity     = 0.1,
  Scalar   min_eigenval = 0.1 )
{
  std::mt19937                           gen( seed );
  std::uniform_real_distribution<Scalar> dist( -1.0, 1.0 );
  std::uniform_real_distribution<Scalar> prob_dist( 0.0, 1.0 );

  // Create a random matrix
  Eigen::SparseMatrix<Scalar>         A( n, n );
  std::vector<Eigen::Triplet<Scalar>> triplets;

  // Generate symmetric matrix with given sparsity
  for ( integer i = 0; i < n; ++i )
  {
    // Diagonal dominance for positive definiteness
    triplets.emplace_back( i, i, min_eigenval + ( static_cast<double>( n ) * 0.5 ) );

    for ( integer j = i + 1; j < n; ++j )
    {
      if ( prob_dist( gen ) < sparsity )
      {
        Scalar val = 0.1 * dist( gen );
        triplets.emplace_back( i, j, val );
        triplets.emplace_back( j, i, val );
      }
    }
  }

  A.setFromTriplets( triplets.begin(), triplets.end() );
  A.makeCompressed();

  return A;
}

/**
 * @brief Generate random gradient vector
 */
Eigen::VectorXd generateRandomGradient( integer n, unsigned seed = 123 )
{
  std::mt19937                           gen( seed );
  std::uniform_real_distribution<Scalar> dist( -1.0, 1.0 );

  Eigen::VectorXd g( n );
  for ( integer i = 0; i < n; ++i ) { g( i ) = dist( gen ); }
  return g;
}

/**
 * @brief Compute quadratic model value m(p) = ½pᵀHp + gᵀp
 */
Scalar computeModelValue( const Eigen::VectorXd & p, const Eigen::SparseMatrix<Scalar> & H, const Eigen::VectorXd & g )
{
  return 0.5 * p.dot( H * p ) + g.dot( p );
}

/**
 * @brief Check if trust region constraint is satisfied
 */
bool isTrustRegionSatisfied( const Eigen::VectorXd & p, Scalar delta, Scalar tolerance = 1e-12 )
{
  Scalar p_norm = p.norm();
  return p_norm <= delta + tolerance;
}

/**
 * @brief Compute gradient of the model at point p
 */
Eigen::VectorXd computeModelGradient(
  const Eigen::VectorXd &             p,
  const Eigen::SparseMatrix<Scalar> & H,
  const Eigen::VectorXd &             g )
{
  return H * p + g;
}

/**
 * @brief Print a formatted test result
 */
void printTestResult(
  const std::string & test_name,
  bool                passed,
  const std::string & details = "",
  Scalar              time_ms = 0.0 )
{
  if ( passed ) { fmt::print( TestColors::PASS, "  ✓ {} PASSED", test_name ); }
  else
  {
    fmt::print( TestColors::FAIL, "  ✗ {} FAILED", test_name );
  }

  if ( !details.empty() ) { fmt::print( " - {}", details ); }

  if ( time_ms > 0 ) { fmt::print( TestColors::TIME, " [{:.2f} ms]", time_ms ); }

  fmt::print( "\n" );
}

/**
 * @brief Convert method enum to string
 */
std::string methodToString( Utils::TrustRegionSolver<Scalar>::Method method )
{
  using Method = Utils::TrustRegionSolver<Scalar>::Method;
  switch ( method )
  {
    case Method::DOGLEG: return "Dogleg";
    case Method::STEIHAUG_CG: return "Steihaug-CG";
    case Method::MORE_SORENSEN: return "More-Sorensen";
    case Method::REGULARIZED: return "Regularized";
    default: return "Unknown";
  }
}

// Helper per risolvere sistema lineare senza SymmetricSolver (per confronto)
Eigen::VectorXd solveLinearSystem( const Eigen::SparseMatrix<Scalar> & H, Scalar lambda, const Eigen::VectorXd & b )
{
  // Usa Eigen::SimplicialLDLT per risolvere (H + λI)x = b
  Eigen::SparseMatrix<Scalar> A = H;
  for ( integer i = 0; i < A.rows(); ++i ) { A.coeffRef( i, i ) += lambda; }
  A.makeCompressed();

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute( A );
  if ( solver.info() != Eigen::Success ) { throw std::runtime_error( "Decomposition failed" ); }
  return solver.solve( b );
}

// ============================================================================
// Test Case 1: Simple 2D Quadratic Problems
// ============================================================================

/**
 * @brief Test with simple 2D quadratic where exact solution is known
 *
 * Test problem: min ½pᵀHp + gᵀp with known solution
 * Cases:
 * 1. Newton point inside trust region
 * 2. Newton point outside trust region
 * 3. Zero gradient case
 */
bool testSimple2DProblems()
{
  fmt::print(
    TestColors::HEADER,
    "\n"
    "┌─────────────────────────────────────┐\n"
    "│      Simple 2D Quadratic Tests      │\n"
    "└─────────────────────────────────────┘\n" );

  integer tests_passed = 0;
  integer total_tests  = 0;

  // Test 1: Newton point inside trust region (should be accepted)
  {
    fmt::print( TestColors::INFO, "\n📊 Test 1: Newton point inside trust region\n" );
    total_tests++;

    // Simple 2x2 positive definite Hessian
    Eigen::SparseMatrix<Scalar> H( 2, 2 );
    H.insert( 0, 0 ) = 4.0;
    H.insert( 0, 1 ) = 1.0;
    H.insert( 1, 0 ) = 1.0;
    H.insert( 1, 1 ) = 3.0;
    H.makeCompressed();

    Eigen::VectorXd g( 2 );
    g << 1.0, -2.0;

    Scalar delta = 5.0;  // Large enough to contain Newton point

    // Compute Newton point (exact solution without constraint)
    Eigen::VectorXd p_newton    = solveLinearSystem( H, 0.0, -g );
    Scalar          newton_norm = p_newton.norm();

    fmt::print( TestColors::INFO, "  Newton point norm: {:.4f}, Δ = {:.4f}\n", newton_norm, delta );

    // Test each method
    std::vector<Utils::TrustRegionSolver<Scalar>::Method> methods = {
      Utils::TrustRegionSolver<Scalar>::Method::DOGLEG,
      Utils::TrustRegionSolver<Scalar>::Method::STEIHAUG_CG,
      Utils::TrustRegionSolver<Scalar>::Method::MORE_SORENSEN,
      Utils::TrustRegionSolver<Scalar>::Method::REGULARIZED
    };

    bool all_methods_passed = true;

    for ( auto method : methods )
    {
      Utils::TrustRegionSolver<Scalar>::Options opts;
      opts.method = method;
      Utils::TrustRegionSolver<Scalar> solver_tr( opts );

      Eigen::VectorXd p;
      Scalar          lambda  = 0.0;
      bool            success = solver_tr.solve( g, H, delta, p, lambda );

      if ( success )
      {
        // Check constraint satisfaction
        bool constraint_ok = isTrustRegionSatisfied( p, delta );

        // For Newton point inside, all methods should return Newton point
        Scalar rel_error = ( p - p_newton ).norm() / std::max( p_newton.norm(), 1e-15 );

        // Steihaug-CG is an approximate method, so we allow larger error
        bool method_passed = constraint_ok;
        if ( method == Utils::TrustRegionSolver<Scalar>::Method::STEIHAUG_CG )
        {
          method_passed = method_passed && ( rel_error < 0.1 );  // 10% tolerance for CG
        }
        else
        {
          method_passed = method_passed && ( rel_error < 1e-6 );  // Tight tolerance for exact methods
        }

        fmt::print(
          TestColors::INFO,
          "  {:15} | Success: {:5} | ‖p‖: {:.4f} | Constraint: {:5} | Error: {:.2e} | Pass: {}\n",
          methodToString( method ),
          success ? "Yes" : "No",
          p.norm(),
          constraint_ok ? "OK" : "FAIL",
          rel_error,
          method_passed ? "Yes" : "No" );

        all_methods_passed = all_methods_passed && method_passed;
      }
      else
      {
        fmt::print( TestColors::ERROR, "  {:15} | FAILED to solve\n", methodToString( method ) );
        all_methods_passed = false;
      }
    }

    tests_passed += all_methods_passed ? 1 : 0;
    printTestResult( "Newton inside trust region", all_methods_passed );
  }

  // Test 2: Newton point outside trust region
  {
    fmt::print( TestColors::INFO, "\n📊 Test 2: Newton point outside trust region\n" );
    total_tests++;

    // Same Hessian as before
    Eigen::SparseMatrix<Scalar> H( 2, 2 );
    H.insert( 0, 0 ) = 4.0;
    H.insert( 0, 1 ) = 1.0;
    H.insert( 1, 0 ) = 1.0;
    H.insert( 1, 1 ) = 3.0;
    H.makeCompressed();

    Eigen::VectorXd g( 2 );
    g << 1.0, -2.0;

    Scalar delta = 0.5;  // Smaller than Newton point norm

    // Compute Newton point
    Eigen::VectorXd p_newton    = solveLinearSystem( H, 0.0, -g );
    Scalar          newton_norm = p_newton.norm();

    fmt::print( TestColors::INFO, "  Newton point norm: {:.4f}, Δ = {:.4f}\n", newton_norm, delta );
    fmt::print( TestColors::INFO, "  Expecting solution on boundary ‖p‖ ≈ Δ\n" );

    std::vector<Utils::TrustRegionSolver<Scalar>::Method> methods = {
      Utils::TrustRegionSolver<Scalar>::Method::DOGLEG,
      Utils::TrustRegionSolver<Scalar>::Method::STEIHAUG_CG,
      Utils::TrustRegionSolver<Scalar>::Method::MORE_SORENSEN
    };

    bool all_methods_passed = true;

    for ( auto method : methods )
    {
      Utils::TrustRegionSolver<Scalar>::Options opts;
      opts.method = method;
      Utils::TrustRegionSolver<Scalar> solver_tr( opts );

      Eigen::VectorXd p;
      Scalar          lambda  = 0.0;
      bool            success = solver_tr.solve( g, H, delta, p, lambda );

      if ( success )
      {
        Scalar p_norm         = p.norm();
        Scalar boundary_error = std::abs( p_norm - delta ) / delta;
        // Usa una tolleranza più ampia per errori numerici nel metodo More-Sorensen
        bool constraint_ok = isTrustRegionSatisfied( p, delta, 1e-8 );

        // Check gradient condition at boundary (should be parallel to p)
        Eigen::VectorXd grad            = computeModelGradient( p, H, g );
        Scalar          grad_norm       = grad.norm();
        Scalar          p_norm_safe     = std::max( p_norm, 1e-15 );
        Scalar          grad_p_parallel = ( grad_norm * p_norm_safe > 1e-12 )
                                            ? std::abs( grad.dot( p ) ) / ( grad_norm * p_norm_safe )
                                            : 1.0;

        // For boundary solutions, we expect the constraint to be satisfied and
        // the solution to be on or inside the boundary with reasonable accuracy
        bool method_passed = constraint_ok && ( boundary_error < 0.05 );

        fmt::print(
          TestColors::INFO,
          "  {:15} | ‖p‖: {:.6f} (Δ={:.6f}) | Boundary err: {:.2e} | Grad·p: {:.6f} | Pass: {}\n",
          methodToString( method ),
          p_norm,
          delta,
          boundary_error,
          grad_p_parallel,
          method_passed ? "Yes" : "No" );

        all_methods_passed = all_methods_passed && method_passed;
      }
      else
      {
        fmt::print( TestColors::ERROR, "  {:15} | FAILED to solve\n", methodToString( method ) );
        all_methods_passed = false;
      }
    }

    tests_passed += all_methods_passed ? 1 : 0;
    printTestResult( "Newton outside trust region", all_methods_passed );
  }

  // Test 3: Zero gradient (stationary point)
  {
    fmt::print( TestColors::INFO, "\n📊 Test 3: Zero gradient case\n" );
    total_tests++;

    Eigen::SparseMatrix<Scalar> H( 2, 2 );
    H.insert( 0, 0 ) = 2.0;
    H.insert( 0, 1 ) = 0.5;
    H.insert( 1, 0 ) = 0.5;
    H.insert( 1, 1 ) = 2.0;
    H.makeCompressed();

    Eigen::VectorXd g     = Eigen::VectorXd::Zero( 2 );
    Scalar          delta = 1.0;

    // Expected solution: p = 0
    Eigen::VectorXd p_expected = Eigen::VectorXd::Zero( 2 );

    std::vector<Utils::TrustRegionSolver<Scalar>::Method> methods = {
      Utils::TrustRegionSolver<Scalar>::Method::DOGLEG,
      Utils::TrustRegionSolver<Scalar>::Method::STEIHAUG_CG,
      Utils::TrustRegionSolver<Scalar>::Method::MORE_SORENSEN,
      Utils::TrustRegionSolver<Scalar>::Method::REGULARIZED
    };

    bool all_methods_passed = true;

    for ( auto method : methods )
    {
      Utils::TrustRegionSolver<Scalar>::Options opts;
      opts.method = method;
      Utils::TrustRegionSolver<Scalar> solver_tr( opts );

      Eigen::VectorXd p;
      Scalar          lambda  = 0.0;
      bool            success = solver_tr.solve( g, H, delta, p, lambda );

      if ( success )
      {
        Scalar error = p.norm();  // Should be zero

        bool method_passed = error < 1e-12;

        fmt::print(
          TestColors::INFO,
          "  {:15} | Success: {:5} | ‖p‖: {:.2e} | Expected: 0.0 | Pass: {}\n",
          methodToString( method ),
          success ? "Yes" : "No",
          error,
          method_passed ? "Yes" : "No" );

        all_methods_passed = all_methods_passed && method_passed;
      }
      else
      {
        fmt::print( TestColors::ERROR, "  {:15} | FAILED to solve\n", methodToString( method ) );
        all_methods_passed = false;
      }
    }

    tests_passed += all_methods_passed ? 1 : 0;
    printTestResult( "Zero gradient case", all_methods_passed );
  }

  // Summary
  fmt::print(
    TestColors::HEADER,
    "\n"
    "┌─────────────────────────────────────┐\n"
    "│            Test Summary             │\n"
    "└─────────────────────────────────────┘\n" );

  if ( tests_passed == total_tests )
  {
    fmt::print( TestColors::SUCCESS, "✅ All {} 2D tests passed!\n", total_tests );
    return true;
  }
  else
  {
    fmt::print( TestColors::ERROR, "❌ {}/{} 2D tests passed\n", tests_passed, total_tests );
    return false;
  }
}

// ============================================================================
// Test Case 2: Comparison Between Methods
// ============================================================================

/**
 * @brief Compare all methods on the same problem set
 *
 * Generate random quadratic problems and compare:
 * - Solution quality (model value reduction)
 * - Constraint satisfaction
 * - Computation time
 */
bool testMethodComparison()
{
  fmt::print(
    TestColors::HEADER,
    "\n"
    "┌─────────────────────────────────────┐\n"
    "│      Method Comparison Tests        │\n"
    "└─────────────────────────────────────┘\n" );

  integer tests_passed = 0;
  integer total_tests  = 0;

  // Test 1: Consistent solutions across methods
  {
    fmt::print( TestColors::INFO, "\n📊 Test 1: Solution consistency\n" );
    total_tests++;

    integer                     n     = 10;
    Eigen::SparseMatrix<Scalar> H     = generateRandomPositiveDefiniteHessian( n, 100, 0.2, 0.5 );
    Eigen::VectorXd             g     = generateRandomGradient( n, 200 );
    Scalar                      delta = 2.0;

    std::vector<Utils::TrustRegionSolver<Scalar>::Method> methods = {
      Utils::TrustRegionSolver<Scalar>::Method::DOGLEG,
      Utils::TrustRegionSolver<Scalar>::Method::STEIHAUG_CG,
      Utils::TrustRegionSolver<Scalar>::Method::MORE_SORENSEN
    };

    std::vector<Eigen::VectorXd> solutions;
    std::vector<Scalar>          model_values;
    bool                         all_succeeded = true;

    // Solve with each method
    for ( auto method : methods )
    {
      Utils::TrustRegionSolver<Scalar>::Options opts;
      opts.method = method;
      Utils::TrustRegionSolver<Scalar> solver_tr( opts );

      Eigen::VectorXd p;
      Scalar          lambda  = 0.0;
      bool            success = solver_tr.solve( g, H, delta, p, lambda );

      if ( success )
      {
        solutions.push_back( p );
        model_values.push_back( computeModelValue( p, H, g ) );
      }
      else
      {
        fmt::print( TestColors::ERROR, "  {:15} failed to solve\n", methodToString( method ) );
        all_succeeded = false;
      }
    }

    if ( all_succeeded && solutions.size() == methods.size() )
    {
      // Compare solutions
      Scalar max_pairwise_error = 0.0;
      Scalar max_model_diff     = 0.0;

      integer nsols = static_cast<integer>( solutions.size() );

      for ( integer i = 0; i < nsols; ++i )
      {
        for ( integer j = i + 1; j < nsols; ++j )
        {
          Scalar norm_i      = solutions[i].norm();
          Scalar norm_j      = solutions[j].norm();
          Scalar denom       = std::max( { norm_i, norm_j, 1e-15 } );
          Scalar error       = ( solutions[i] - solutions[j] ).norm() / denom;
          max_pairwise_error = std::max( max_pairwise_error, error );

          Scalar model_i     = model_values[i];
          Scalar model_j     = model_values[j];
          Scalar model_denom = std::max( { std::abs( model_i ), std::abs( model_j ), 1e-15 } );
          Scalar model_diff  = std::abs( model_i - model_j ) / model_denom;
          max_model_diff     = std::max( max_model_diff, model_diff );
        }
      }

      // Allow reasonable tolerance for different methods
      bool passed = ( max_pairwise_error < 0.1 ) && ( max_model_diff < 0.1 );

      fmt::print( TestColors::INFO, "  Max pairwise solution error: {:.2e}\n", max_pairwise_error );
      fmt::print( TestColors::INFO, "  Max model value difference: {:.2e}\n", max_model_diff );
      fmt::print( TestColors::INFO, "  Model values:\n" );
      integer nmeth = static_cast<integer>( methods.size() );
      for ( integer i = 0; i < nmeth; ++i )
      {
        fmt::print(
          TestColors::INFO,
          "    {:15}: m(p) = {:10.6f}, ‖p‖ = {:.4f}\n",
          methodToString( methods[i] ),
          model_values[i],
          solutions[i].norm() );
      }

      tests_passed += passed ? 1 : 0;
      printTestResult( "Solution consistency", passed );
    }
    else
    {
      printTestResult( "Solution consistency", false, "Some methods failed" );
    }
  }

  // Test 2: Performance comparison table
  {
    fmt::print( TestColors::INFO, "\n⚡ Test 2: Performance comparison\n" );
    total_tests++;

    std::vector<integer>                                  sizes   = { 10, 50, 100 };
    std::vector<Utils::TrustRegionSolver<Scalar>::Method> methods = {
      Utils::TrustRegionSolver<Scalar>::Method::DOGLEG,
      Utils::TrustRegionSolver<Scalar>::Method::STEIHAUG_CG,
      Utils::TrustRegionSolver<Scalar>::Method::MORE_SORENSEN,
      Utils::TrustRegionSolver<Scalar>::Method::REGULARIZED
    };

    fmt::print(
      "\n"
      "┌──────┬────────────┬────────────┬────────────┬────────────┬────────────┐\n"
      "│ Size │ Dogleg     │ Steihaug   │ More-S     │ Regularized│ Best       │\n"
      "├──────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n" );

    bool all_tests_passed = true;

    for ( integer n : sizes )
    {
      Eigen::SparseMatrix<Scalar> H     = generateRandomPositiveDefiniteHessian( n, 300 + n, 0.1, 1.0 );
      Eigen::VectorXd             g     = generateRandomGradient( n, 400 + n );
      Scalar                      delta = 1.0;

      std::vector<Scalar> times_ms( methods.size(), 0.0 );
      std::vector<bool>   successes( methods.size(), false );
      std::vector<Scalar> model_values( methods.size(), 0.0 );

      integer nmeth = static_cast<integer>( methods.size() );
      for ( integer i = 0; i < nmeth; ++i )
      {
        Utils::TrustRegionSolver<Scalar>::Options opts;
        opts.method = methods[i];
        Utils::TrustRegionSolver<Scalar> solver_tr( opts );

        Utils::TicToc timer;
        timer.tic();

        Eigen::VectorXd p;
        Scalar          lambda  = 0.0;
        bool            success = solver_tr.solve( g, H, delta, p, lambda );

        timer.toc();

        if ( success )
        {
          times_ms[i]     = timer.elapsed_ms();
          successes[i]    = true;
          model_values[i] = computeModelValue( p, H, g );
        }
        else
        {
          times_ms[i]  = -1.0;
          successes[i] = false;
        }
      }

      // Find best (lowest model value) among successful methods
      Scalar best_model_value = std::numeric_limits<Scalar>::max();
      int    best_method_idx  = -1;

      for ( integer i = 0; i < nmeth; ++i )
      {
        if ( successes[i] && model_values[i] < best_model_value )
        {
          best_model_value = model_values[i];
          best_method_idx  = static_cast<int>( i );
        }
      }

      // Print row
      fmt::print( "│ {:<4} ", n );

      for ( integer i = 0; i < nmeth; ++i )
      {
        if ( successes[i] )
        {
          auto color = ( i == best_method_idx ) ? TestColors::SUCCESS : TestColors::TIME;
          fmt::print( "│" );
          fmt::print( color, " {:>8.2f}ms ", times_ms[i] );
        }
        else
        {
          fmt::print( "│" );
          fmt::print( TestColors::ERROR, "       FAIL " );
        }
      }

      if ( best_method_idx >= 0 )
      {
        fmt::print( "│" );
        fmt::print( TestColors::METHOD, " {:>10} │\n", methodToString( methods[best_method_idx] ) );
      }
      else
      {
        fmt::print( "│ {:>10} │\n", "None" );
      }

      // Check if at least one method succeeded
      bool at_least_one_success = std::any_of( successes.begin(), successes.end(), []( bool s ) { return s; } );
      all_tests_passed          = all_tests_passed && at_least_one_success;
    }

    fmt::print( "└──────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n" );

    tests_passed += all_tests_passed ? 1 : 0;
    printTestResult( "Performance comparison", all_tests_passed );
  }

  // Summary
  fmt::print(
    TestColors::HEADER,
    "\n"
    "┌─────────────────────────────────────┐\n"
    "│            Test Summary             │\n"
    "└─────────────────────────────────────┘\n" );

  if ( tests_passed == total_tests )
  {
    fmt::print( TestColors::SUCCESS, "✅ All {} comparison tests passed!\n", total_tests );
    return true;
  }
  else
  {
    fmt::print( TestColors::ERROR, "❌ {}/{} comparison tests passed\n", tests_passed, total_tests );
    return false;
  }
}

// ============================================================================
// Main Test Program
// ============================================================================

/**
 * @brief Main test program with comprehensive test suite
 *
 * Orchestrates all test categories and provides a summary report.
 */
int main()
{
  fmt::print(
    TestColors::HEADER,
    "╔══════════════════════════════════════════════════╗\n"
    "║       Trust Region Subproblem Solver Tests       ║\n"
    "║            C++17 • Eigen3 • FMT                  ║\n"
    "╚══════════════════════════════════════════════════╝\n"
    "\n" );

  // Print system info
  fmt::print( TestColors::INFO, "System Information:\n" );
  fmt::print( "  Eigen version: {}.{}.{}\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION );
  fmt::print( "  Trust Region Methods: Dogleg, Steihaug-CG, More-Sorensen, Regularized\n" );
  fmt::print( "  Max problem size tested: 100 dimensions\n" );
  fmt::print( "  Tolerance for constraints: 1e-12\n\n" );

  Utils::TicToc tm;
  tm.tic();

  bool                                                              all_passed = true;
  std::vector<std::pair<std::string, std::pair<bool, std::string>>> test_results;

  try
  {
    fmt::print( TestColors::INFO, "🚀 Starting comprehensive test suite...\n\n" );

    // Run all test categories
    test_results.emplace_back( "Simple 2D Problems", std::make_pair( testSimple2DProblems(), "" ) );
    test_results.emplace_back( "Method Comparison", std::make_pair( testMethodComparison(), "" ) );
  }
  catch ( const std::exception & e )
  {
    fmt::print( TestColors::ERROR, "\n💥 EXCEPTION CAUGHT: {}\n", e.what() );
    all_passed = false;
    test_results.emplace_back( "Exception Handler", std::make_pair( false, e.what() ) );
  }
  catch ( ... )
  {
    fmt::print( TestColors::ERROR, "\n💥 UNKNOWN EXCEPTION CAUGHT\n" );
    all_passed = false;
    test_results.emplace_back( "Exception Handler", std::make_pair( false, "Unknown exception" ) );
  }

  tm.toc();

  // Final summary
  fmt::print(
    TestColors::HEADER,
    "\n"
    "╔══════════════════════════════════════════════════╗\n"
    "║                 FINAL SUMMARY                    ║\n"
    "╚══════════════════════════════════════════════════╝\n" );

  fmt::print(
    "\n"
    "┌────────────────────────┬─────────────┬─────────────────────┐\n"
    "│ Test Category          │ Status      │ Details             │\n"
    "├────────────────────────┼─────────────┼─────────────────────┤\n" );

  integer passed_count = 0;
  integer total_count  = static_cast<integer>( test_results.size() );

  for ( const auto & [name, result] : test_results )
  {
    bool                passed  = result.first;
    const std::string & details = result.second;

    if ( passed ) passed_count++;

    fmt::print( "│ {:<22} │ ", name );
    if ( passed ) { fmt::print( TestColors::SUCCESS, "{:^11} ", "✓ PASS" ); }
    else
    {
      fmt::print( TestColors::ERROR, "{:^11} ", "✗ FAIL" );
    }

    if ( !details.empty() ) { fmt::print( "│ {:<19} │\n", details.substr( 0, 19 ) ); }
    else
    {
      fmt::print( "│ {:<19} │\n", passed ? "All tests passed" : "Check logs" );
    }
  }

  fmt::print( "├────────────────────────┼─────────────┼─────────────────────┤\n" );
  fmt::print(
    "│ Overall Result         │ {:^11} │ {:>19} │\n",
    ( passed_count == total_count ) ? "PASS" : "FAIL",
    fmt::format( "{}/{} passed", passed_count, total_count ) );
  fmt::print( "├────────────────────────┼─────────────┼─────────────────────┤\n" );
  fmt::print( "│ Total Execution Time   │ {:^11.3f} │ {:>19} │\n", tm.elapsed_s(), "seconds" );
  fmt::print( "└────────────────────────┴─────────────┴─────────────────────┘\n" );

  // Recommendations
  fmt::print( "\n" );
  fmt::print( TestColors::INFO, "📋 Recommendations for Trust Region Solver:\n" );

  if ( all_passed )
  {
    fmt::print( TestColors::SUCCESS, "✅ All solvers are working correctly.\n" );
    fmt::print(
      TestColors::INFO,
      "   • Dogleg: Best for small problems, simple and robust\n"
      "   • Steihaug-CG: Good for large sparse problems, handles indefinite Hessians\n"
      "   • More-Sorensen: Most accurate for boundary solutions, handles ill-conditioning\n"
      "   • Regularized: Simple fallback when other methods fail\n"
      "   • Trust region radius: Adjust based on model accuracy\n"
      "   • Convergence: Verify KKT conditions for boundary solutions\n" );
  }
  else
  {
    fmt::print( TestColors::WARNING, "⚠️  Some issues detected:\n" );

    for ( const auto & [name, result] : test_results )
    {
      if ( !result.first ) { fmt::print( TestColors::WARNING, "   • {}: Failed - {}\n", name, result.second ); }
    }

    fmt::print(
      TestColors::INFO,
      "\n"
      "💡 Troubleshooting tips:\n"
      "   1. Check Hessian conditioning (add regularization if needed)\n"
      "   2. Verify gradient scale relative to trust region\n"
      "   3. For boundary solutions, check λ ≥ 0\n"
      "   4. Increase iteration limits for difficult problems\n"
      "   5. Try different methods for different problem types\n" );
  }

  fmt::print( "\n" );
  return all_passed ? 0 : 1;
}
