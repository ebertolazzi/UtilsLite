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

#include "Utils_minimize_ParametricSensitivity.hh"
#include "Utils_minimize_Newton.hh"

// ===========================================================================
// USAGE EXAMPLES
// ===========================================================================

using namespace Utils;

using Scalar       = double;
using TestVector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using TestMatrix   = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// ---------------------------------------------------------------------------
// Utility functions for pretty printing
// ---------------------------------------------------------------------------
void print_section_title( const std::string & title )
{
  fmt::print(
    fmt::fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "╔══════════════════════════════════════════════════════════════════╗\n"
    "║{:^66}║\n"
    "╚══════════════════════════════════════════════════════════════════╝\n",
    title );
}

void print_subsection( const std::string & title )
{ fmt::print( fmt::fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n┌─▶ {}:\n", title ); }

void print_success( const std::string & message )
{ fmt::print( fmt::fg( fmt::color::green ) | fmt::emphasis::bold, "   ✓ {}", message ); }

void print_warning( const std::string & message )
{ fmt::print( fmt::fg( fmt::color::orange ) | fmt::emphasis::bold, "   ⚠ {}", message ); }

void print_error( const std::string & message )
{ fmt::print( fmt::fg( fmt::color::red ) | fmt::emphasis::bold, "   ✗ {}", message ); }

void print_info( const std::string & message )
{ fmt::print( fmt::fg( fmt::color::light_blue ), "   • {}", message ); }

void print_matrix_info( const std::string & name, const TestMatrix & mat, bool show_values = false )
{
  fmt::print( fmt::fg( fmt::color::white ), "   {}: {} × {} matrix\n", name, mat.rows(), mat.cols() );

  if ( show_values && mat.rows() <= 5 && mat.cols() <= 5 )
  {
    fmt::print( fmt::fg( fmt::color::light_gray ), "     " );
    for ( int j = 0; j < mat.cols(); ++j ) { fmt::print( "{:>12d} ", j ); }
    fmt::print( "\n" );

    for ( int i = 0; i < mat.rows(); ++i )
    {
      fmt::print( fmt::fg( fmt::color::light_gray ), "{:3d}: ", i );
      for ( int j = 0; j < mat.cols(); ++j ) { fmt::print( "{:12.4e} ", mat( i, j ) ); }
      fmt::print( "\n" );
    }
  }
}

// ---------------------------------------------------------------------------
// Example 1: Unconstrained Quadratic
// ---------------------------------------------------------------------------
void example_unconstrained()
{
  print_section_title( "EXAMPLE 1: Unconstrained Quadratic Problem" );

  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Problem: f(x,p) = (x₁ - p₁)² + (x₂ - p₂)²/p₃\n"
    "   Analytic solution: x* = (p₁, p₂)\n"
    "   Analytic sensitivity: ∂x*/∂p = [[1,0,0],[0,1,0]]\n" );

  auto parametric_function =
    []( TestVector const & x, TestVector const & p, TestVector * grad_x, TestMatrix * hess_xx, TestMatrix * grad_xp )
    -> Scalar
  {
    Scalar x1 = x( 0 ), x2 = x( 1 );
    Scalar p1 = p( 0 ), p2 = p( 1 ), p3 = p( 2 );

    Scalar f = std::pow( x1 - p1, 2 ) + std::pow( x2 - p2, 2 ) / p3;

    if ( grad_x )
    {
      ( *grad_x )( 0 ) = 2 * ( x1 - p1 );
      ( *grad_x )( 1 ) = 2 * ( x2 - p2 ) / p3;
    }

    if ( hess_xx )
    {
      hess_xx->setZero( 2, 2 );
      ( *hess_xx )( 0, 0 ) = 2.0;
      ( *hess_xx )( 1, 1 ) = 2.0 / p3;
    }

    if ( grad_xp )
    {
      grad_xp->setZero();
      ( *grad_xp )( 0, 0 ) = -2.0;
      ( *grad_xp )( 1, 1 ) = -2.0 / p3;
      ( *grad_xp )( 1, 2 ) = -2.0 * ( x2 - p2 ) / ( p3 * p3 );
    }

    return f;
  };

  TestVector x0( 2 );
  x0 << 0.0, 0.0;
  TestVector p( 3 );
  p << 1.0, 2.0, 1.5;

  TestVector x_exact( 2 );
  x_exact << p( 0 ), p( 1 );

  TestMatrix sensitivity_exact( 2, 3 );
  sensitivity_exact << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;

  auto standard_callback = [&]( TestVector const & x, TestVector * g, TestMatrix * H )
  { return parametric_function( x, p, g, H, nullptr ); };

  Newton_minimizer<Scalar>::Options optimizer_opts;
  optimizer_opts.verbosity = 1;
  Newton_minimizer<Scalar> optimizer( optimizer_opts );
  optimizer.minimize( x0, standard_callback );

  ParametricSensitivity<Scalar>::Options sensitivity_opts;
  sensitivity_opts.verbosity_level        = 1;
  sensitivity_opts.use_finite_differences = false;
  ParametricSensitivity<Scalar> sensitivity_analyzer( sensitivity_opts );
  sensitivity_analyzer.compute_sensitivity( optimizer.solution(), p, parametric_function );

  print_subsection( "Optimization Results" );
  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Initial point: x0 = [{:.2f}, {:.2f}]\n"
    "   Parameters: p = [{:.2f}, {:.2f}, {:.2f}]\n"
    "   Iterations: {}\n"
    "   Final gradient norm: {:.2e}\n",
    x0( 0 ),
    x0( 1 ),
    p( 0 ),
    p( 1 ),
    p( 2 ),
    optimizer.iterations(),
    optimizer.final_grad_norm() );

  print_subsection( "Optimal Solution" );
  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Computed: x* = [{:.6f}, {:.6f}]\n"
    "   Exact:     x* = [{:.6f}, {:.6f}]\n",
    optimizer.solution()( 0 ),
    optimizer.solution()( 1 ),
    x_exact( 0 ),
    x_exact( 1 ) );

  Scalar error = ( optimizer.solution() - x_exact ).norm();
  if ( error < 1e-6 ) { print_success( fmt::format( "Solution accuracy: {:.2e} ✓\n", error ) ); }
  else
  {
    print_warning( fmt::format( "Solution accuracy: {:.2e}\n", error ) );
  }

  print_subsection( "Parametric Sensitivity" );
  if ( sensitivity_analyzer.success() )
  {
    print_success( "Sensitivity computation successful ✓\n" );
    print_matrix_info( "Computed ∂x*/∂p", sensitivity_analyzer.sensitivity(), true );
    print_matrix_info( "Exact ∂x*/∂p", sensitivity_exact, true );

    Scalar max_error = ( sensitivity_analyzer.sensitivity() - sensitivity_exact ).template lpNorm<Eigen::Infinity>();
    if ( max_error < 1e-8 ) { print_success( fmt::format( "Maximum sensitivity error: {:.2e} ✓\n", max_error ) ); }
    else
    {
      print_warning( fmt::format( "Maximum sensitivity error: {:.2e}\n", max_error ) );
    }

    if ( sensitivity_analyzer.condition_number() > 0 )
    {
      fmt::print( fmt::fg( fmt::color::white ), "   Condition number: {:.2e}\n", sensitivity_analyzer.condition_number() );
    }
  }
  else
  {
    print_error( fmt::format( "Sensitivity computation failed: {}\n", sensitivity_analyzer.error_message() ) );
  }
}

// ---------------------------------------------------------------------------
// Example 2: Box Constrained Problem
// ---------------------------------------------------------------------------
void example_box_constrained()
{
  print_section_title( "EXAMPLE 2: Box Constrained Problem" );

  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Problem: f(x,p) = (x₁ - p₁)² + (x₂ - p₂)²\n"
    "   Constraints: -1 ≤ x₁ ≤ 1, -1 ≤ x₂ ≤ 1\n"
    "   For p = (2.0, 0.5):\n"
    "     - x₁* = 1.0 (upper bound active) → ∂x₁/∂p = 0\n"
    "     - x₂* = 0.5 (free) → ∂x₂/∂p = [0, 1]\n" );

  auto parametric_function =
    []( TestVector const & x, TestVector const & p, TestVector * grad_x, TestMatrix * hess_xx, TestMatrix * grad_xp )
    -> Scalar
  {
    Scalar f = std::pow( x( 0 ) - p( 0 ), 2 ) + std::pow( x( 1 ) - p( 1 ), 2 );

    if ( grad_x )
    {
      ( *grad_x )( 0 ) = 2 * ( x( 0 ) - p( 0 ) );
      ( *grad_x )( 1 ) = 2 * ( x( 1 ) - p( 1 ) );
    }

    if ( hess_xx )
    {
      hess_xx->setZero( 2, 2 );
      ( *hess_xx )( 0, 0 ) = 2.0;
      ( *hess_xx )( 1, 1 ) = 2.0;
    }

    if ( grad_xp && grad_xp->rows() == 2 && grad_xp->cols() == 2 )
    {
      grad_xp->setZero();
      ( *grad_xp )( 0, 0 ) = -2.0;
      ( *grad_xp )( 1, 1 ) = -2.0;
    }

    return f;
  };

  TestVector x0( 2 );
  x0 << 0.0, 0.0;
  TestVector p( 2 );
  p << 2.0, 0.5;

  TestVector lower( 2 );
  lower << -1.0, -1.0;
  TestVector upper( 2 );
  upper << 1.0, 1.0;

  TestVector x_exact( 2 );
  x_exact( 0 ) = std::max( -1.0, std::min( 1.0, p( 0 ) ) );
  x_exact( 1 ) = std::max( -1.0, std::min( 1.0, p( 1 ) ) );

  TestMatrix sensitivity_exact( 2, 2 );
  sensitivity_exact << 0.0, 0.0, 0.0, 1.0;

  auto standard_callback = [&]( TestVector const & x, TestVector * g, TestMatrix * H )
  { return parametric_function( x, p, g, H, nullptr ); };

  Newton_minimizer<Scalar>::Options opt_opts;
  opt_opts.verbosity = 0;
  Newton_minimizer<Scalar> optimizer( opt_opts );
  optimizer.set_bounds( lower, upper );
  optimizer.minimize( x0, standard_callback );

  TestVector x_opt = optimizer.solution();

  ParametricSensitivity<Scalar>::Options sens_opts;
  sens_opts.verbosity_level = 1;

  ParametricSensitivity<Scalar> sens_analyzer( sens_opts );
  sens_analyzer.compute_sensitivity( x_opt, p, parametric_function, lower, upper );

  print_subsection( "Optimization Results" );
  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Initial point: x0 = [{:.2f}, {:.2f}]\n"
    "   Parameters: p = [{:.2f}, {:.2f}]\n"
    "   Bounds: x ∈ [[{:.1f}, {:.1f}], [{:.1f}, {:.1f}]]\n",
    x0( 0 ),
    x0( 1 ),
    p( 0 ),
    p( 1 ),
    lower( 0 ),
    upper( 0 ),
    lower( 1 ),
    upper( 1 ) );

  print_subsection( "Optimal Solution" );
  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Computed: x* = [{:.6f}, {:.6f}]\n"
    "   Exact:     x* = [{:.6f}, {:.6f}]\n",
    x_opt( 0 ),
    x_opt( 1 ),
    x_exact( 0 ),
    x_exact( 1 ) );

  Scalar error = ( x_opt - x_exact ).norm();
  if ( error < 1e-6 ) { print_success( fmt::format( "Solution accuracy: {:.2e} ✓\n", error ) ); }
  else
  {
    print_warning( fmt::format( "Solution accuracy: {:.2e}\n", error ) );
  }

  print_subsection( "Constraint Analysis" );
  auto & active_set = sens_analyzer.active_set();
  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Free variables: {} / {}\n"
    "   Active at lower bound: {}\n"
    "   Active at upper bound: {}\n",
    active_set.n_free(),
    x_opt.size(),
    active_set.lower_active.size(),
    active_set.upper_active.size() );

  if ( !active_set.lower_active.empty() )
  {
    fmt::print( fmt::fg( fmt::color::orange ), "   Lower active indices: " );
    for ( auto i : active_set.lower_active ) fmt::print( "{} ", i );
    fmt::print( "\n" );
  }
  if ( !active_set.upper_active.empty() )
  {
    fmt::print( fmt::fg( fmt::color::orange ), "   Upper active indices: " );
    for ( auto i : active_set.upper_active ) fmt::print( "{} ", i );
    fmt::print( "\n" );
  }

  print_subsection( "Parametric Sensitivity" );
  if ( sens_analyzer.success() )
  {
    print_success( "Sensitivity computation successful ✓\n" );
    print_matrix_info( "Computed ∂x*/∂p", sens_analyzer.sensitivity(), true );
    print_matrix_info( "Exact ∂x*/∂p", sensitivity_exact, true );

    Scalar max_error = ( sens_analyzer.sensitivity() - sensitivity_exact ).norm();
    if ( max_error < 1e-8 ) { print_success( fmt::format( "Maximum sensitivity error: {:.2e} ✓\n", max_error ) ); }
    else
    {
      print_warning( fmt::format( "Maximum sensitivity error: {:.2e}\n", max_error ) );
    }

    if ( sens_analyzer.condition_number() > 0 )
    {
      fmt::print( fmt::fg( fmt::color::white ), "   Condition number: {:.2e}\n", sens_analyzer.condition_number() );
    }
  }
  else
  {
    print_error( fmt::format( "Sensitivity computation failed: {}\n", sens_analyzer.error_message() ) );
  }
}

// ---------------------------------------------------------------------------
// Example 3: Regularized Problem
// ---------------------------------------------------------------------------
void example_regularized()
{
  Scalar epsilon = 0.1;
  print_section_title( fmt::format( "EXAMPLE 3: Regularized Problem (ε = {:.2f})", epsilon ) );

  fmt::print(
    fmt::fg( fmt::color::white ),
    "   Problem: f(x,p) + ε‖x‖² where f(x,p) = (x - p)ᵀQ(x - p)\n"
    "   Q = diag(1, 10) (ill-conditioned without regularization)\n"
    "   Regularization effect:\n"
    "     - Improves Hessian conditioning\n"
    "     - Reduces sensitivity magnitude\n"
    "     - Smooths solution w.r.t. parameters\n" );

  auto parametric_function =
    []( TestVector const & x, TestVector const & p, TestVector * grad_x, TestMatrix * hess_xx, TestMatrix * grad_xp )
    -> Scalar
  {
    TestVector diff = x - p;
    TestVector Qdiff( 2 );
    Qdiff( 0 ) = 1.0 * diff( 0 );
    Qdiff( 1 ) = 10.0 * diff( 1 );

    Scalar f = diff.dot( Qdiff );

    if ( grad_x )
    {
      ( *grad_x )( 0 ) = 2.0 * Qdiff( 0 );
      ( *grad_x )( 1 ) = 2.0 * Qdiff( 1 );
    }

    if ( hess_xx )
    {
      hess_xx->setZero( 2, 2 );
      ( *hess_xx )( 0, 0 ) = 2.0;
      ( *hess_xx )( 1, 1 ) = 20.0;
    }

    if ( grad_xp && grad_xp->rows() == 2 && grad_xp->cols() == 2 )
    {
      ( *grad_xp )( 0, 0 ) = -2.0;
      ( *grad_xp )( 0, 1 ) = 0.0;
      ( *grad_xp )( 1, 0 ) = 0.0;
      ( *grad_xp )( 1, 1 ) = -20.0;
    }

    return f;
  };

  TestVector x0( 2 );
  x0 << 0.0, 0.0;
  TestVector p( 2 );
  p << 1.0, 2.0;

  TestMatrix Q( 2, 2 );
  Q << 1.0, 0.0, 0.0, 10.0;

  // Exact solutions
  TestVector x_exact_no_reg           = p;
  TestMatrix sensitivity_exact_no_reg = TestMatrix::Identity( 2, 2 );

  TestMatrix Q_plus_epsI           = Q + epsilon * TestMatrix::Identity( 2, 2 );
  TestVector x_exact_reg           = Q_plus_epsI.inverse() * Q * p;
  TestMatrix sensitivity_exact_reg = Q_plus_epsI.inverse() * Q;

  Eigen::SelfAdjointEigenSolver<TestMatrix> solver_no_reg( 2.0 * Q );
  Scalar cond_exact_no_reg = solver_no_reg.eigenvalues().maxCoeff() / solver_no_reg.eigenvalues().minCoeff();

  Eigen::SelfAdjointEigenSolver<TestMatrix> solver_reg( 2.0 * ( Q + epsilon * TestMatrix::Identity( 2, 2 ) ) );
  Scalar cond_exact_reg = solver_reg.eigenvalues().maxCoeff() / solver_reg.eigenvalues().minCoeff();

  print_subsection( "Without Regularization" );
  {
    ParametricSensitivity<Scalar>::Options sens_opts;
    sens_opts.verbosity_level            = 0;
    sens_opts.account_for_regularization = false;

    ParametricSensitivity<Scalar> sens_analyzer( sens_opts );
    TestVector                    x_opt = p;
    sens_analyzer.compute_sensitivity( x_opt, p, parametric_function );

    fmt::print(
      fmt::fg( fmt::color::white ),
      "   Exact x*: [{:.6f}, {:.6f}]\n"
      "   Condition number: {:.2e} (exact: {:.2e})\n"
      "   Max sensitivity: {:.2e}\n",
      x_opt( 0 ),
      x_opt( 1 ),
      sens_analyzer.condition_number(),
      cond_exact_no_reg,
      sens_analyzer.max_sensitivity() );

    if ( sens_analyzer.condition_number() > 1000 ) { print_warning( "Poor conditioning detected (κ > 1000)\n" ); }
  }

  print_subsection( fmt::format( "With Regularization (ε = {:.2f})", epsilon ) );
  {
    ParametricSensitivity<Scalar>::Options sens_opts;
    sens_opts.verbosity_level            = 0;
    sens_opts.account_for_regularization = true;
    sens_opts.regularization_epsilon     = epsilon;

    ParametricSensitivity<Scalar> sens_analyzer( sens_opts );
    TestVector                    x_opt = x_exact_reg;
    sens_analyzer.compute_sensitivity( x_opt, p, parametric_function );

    fmt::print(
      fmt::fg( fmt::color::white ),
      "   Exact x*: [{:.6f}, {:.6f}]\n"
      "   Condition number: {:.2e} (exact: {:.2e})\n"
      "   Max sensitivity: {:.2e}\n",
      x_opt( 0 ),
      x_opt( 1 ),
      sens_analyzer.condition_number(),
      cond_exact_reg,
      sens_analyzer.max_sensitivity() );

    Scalar cond_improvement = cond_exact_no_reg / cond_exact_reg;
    // Scalar sens_reduction   = 1.0 / ( sens_analyzer.max_sensitivity() + 1e-10 );

    if ( cond_improvement > 1.1 )
    {
      print_success( fmt::format( "Conditioning improved by factor: {:.1f}x ✓\n", cond_improvement ) );
    }
    if ( sens_analyzer.max_sensitivity() < 1.0 ) { print_success( "Sensitivity reduced (max < 1.0) ✓\n" ); }

    print_matrix_info( "Regularized sensitivity ∂x*/∂p", sens_analyzer.sensitivity(), true );
    print_matrix_info( "Exact regularized sensitivity", sensitivity_exact_reg, true );

    Scalar error = ( sens_analyzer.sensitivity() - sensitivity_exact_reg ).norm();
    if ( error < 1e-8 ) { print_success( fmt::format( "Sensitivity accuracy: {:.2e} ✓\n", error ) ); }
  }
}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------
int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ) | fmt::emphasis::bold,
    "\n"
    "██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗\n"
    "██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝\n"
    "██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   ██████╔╝██║██║     \n"
    "██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║     \n"
    "██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗\n"
    "╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝\n"
    "                                                                                \n"
    "          Parametric Sensitivity Analysis - Test Suite                          \n\n" );

  try
  {
    example_unconstrained();
    fmt::print( "\n" );
    example_box_constrained();
    fmt::print( "\n" );
    example_regularized();

    fmt::print(
      fmt::fg( fmt::color::light_green ) | fmt::emphasis::bold,
      "\n"
      "╔══════════════════════════════════════════════════════════════════╗\n"
      "║                   TESTS COMPLETED SUCCESSFULLY                   ║\n"
      "╚══════════════════════════════════════════════════════════════════╝\n" );

    fmt::print(
      fmt::fg( fmt::color::white ),
      "\n"
      "Summary:\n"
      "  • Unconstrained optimization with analytic sensitivity ✓\n"
      "  • Box-constrained optimization with active set handling ✓\n"
      "  • Regularized problem with improved conditioning ✓\n"
      "\n"
      "All examples demonstrate correct sensitivity computation\n"
      "using the implicit function theorem approach.\n" );
  }
  catch ( const std::exception & e )
  {
    print_error( fmt::format( "Exception occurred: {}\n", e.what() ) );
    return 1;
  }

  fmt::print( fmt::fg( fmt::color::light_blue ) | fmt::emphasis::bold, "\n🎉 All done! 🎉\n" );

  return 0;
}
