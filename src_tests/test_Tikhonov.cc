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

#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>
#include <tuple>

#include "Utils_fmt.hh"
#include "Utils_eigen.hh"
#include "Utils_Tikhonov.hh"
#include "Utils_TicToc.hh"

using namespace Eigen;
using std::pair;
using std::string;
using std::vector;

//==============================================================================
static vector<double>         lambda_values = { 10, 1e-6, 1e-10, 0 };
static vector<pair<int, int>> sizes         = {
  { 50, 50 }, { 100, 50 }, { 500, 50 }, { 100, 100 }, { 500, 100 }, { 3000, 200 }
};

//==============================================================================
// Funzione per generare una matrice sparsa casuale
template <typename Scalar>
Eigen::SparseMatrix<Scalar> generate_random_sparse_matrix( int m, int n, double density = 0.3 )
{
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  SparseMatrix mat( m, n );

  size_t nnz = static_cast<size_t>( density * m * n );

  static std::random_device              rd;
  static std::mt19937                    gen( rd() );
  std::uniform_int_distribution<>        row_dist( 0, m - 1 );
  std::uniform_int_distribution<>        col_dist( 0, n - 1 );
  std::uniform_real_distribution<Scalar> val_dist( -1.0, 1.0 );

  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve( nnz );

  std::set<std::pair<int, int>> positions;

  // Nota: std::set è lento per matrici grandi, ma ok per test.
  // Per performance migliori usare direttamente triplette con duplicati e sumup
  // di Eigen.
  while ( triplets.size() < nnz )
  {
    int i = row_dist( gen );
    int j = col_dist( gen );
    if ( positions.find( { i, j } ) == positions.end() )
    {
      positions.insert( { i, j } );
      triplets.emplace_back( i, j, val_dist( gen ) );
    }
  }

  mat.setFromTriplets( triplets.begin(), triplets.end() );
  mat.makeCompressed();

  return mat;
}

//==============================================================================
// Genera un vettore casuale con elementi in [min_val, max_val]
template <typename Scalar>
Matrix<Scalar, Dynamic, 1> generate_random_vector( int n, Scalar min_val = 0.1, Scalar max_val = 10.0 )
{
  static std::random_device              rd;
  static std::mt19937                    gen( rd() );
  std::uniform_real_distribution<Scalar> dist( min_val, max_val );

  Matrix<Scalar, Dynamic, 1> vec( n );
  for ( int i = 0; i < n; ++i ) { vec( i ) = dist( gen ); }

  return vec;
}

//==============================================================================
// FORMATTER UTILS
//==============================================================================
inline void header( std::string const & msg )
{
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n"
    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    "┃ {:^76} ┃\n"
    "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n",
    msg );
}

inline std::string sci( double v )
{ return fmt::format( "{:>12.3e}", v ); }
inline std::string ms_format( double v )
{ return fmt::format( "    {}", fmt::format( "{:.2f} ms", v ) ); }
inline std::string ratio_format( double v )
{ return fmt::format( "    {}", fmt::format( "{:.2f}x", v ) ); }
inline std::string small_ms_format( double v )
{ return fmt::format( "    {}", fmt::format( "{:.4f} ms", v ) ); }

//==============================================================================
// RUN TEST STANDARD (D = I)
//==============================================================================
static void run_test_standard( int m, int n, double lambda )
{
  Utils::TicToc tm;

  header( fmt::format( "STANDARD TEST (D=I): m={}, n={}, λ={}", m, n, lambda ) );

  // Genera matrice sparsa casuale
  SparseMatrix<double> AS = generate_random_sparse_matrix<double>( m, n, 0.3 );
  // Converti in densa per i solver dense
  MatrixXd AD = MatrixXd( AS );

  VectorXd b = VectorXd::Random( m );
  VectorXd c = VectorXd::Random( n );

  //--------------------------------------------------------------------------
  // Build TikhonovSolver (QR) - DENSE
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver<double> qr_solver( AD, lambda );
  tm.toc();
  double t_build_qr = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build TikhonovSolver2 (KKT dense)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver2<double> kkt_dense_solver( AD, lambda );
  tm.toc();
  double t_build_kkt_dense = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver (sparse QR)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::SP_TikhonovSolver<double> sp_solver( AS, lambda );
  tm.toc();
  double t_build_sp = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver2 (sparse KKT)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::SP_TikhonovSolver2<double> sp2_solver( AS, lambda );
  tm.toc();
  double t_build_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Accuracy - Solve min ‖A x - b‖^2 + λ‖x‖²
  //--------------------------------------------------------------------------
  VectorXd x_qr = qr_solver.solve( b );
  double   r_qr = ( AD * x_qr - b ).norm();

  VectorXd x_kkt_dense = kkt_dense_solver.solve( b );
  double   r_kkt_dense = ( AD * x_kkt_dense - b ).norm();

  VectorXd x_sp = sp_solver.solve( b );
  double   r_sp = ( AS * x_sp - b ).norm();

  VectorXd x_sp2 = sp2_solver.solve( b );
  double   r_sp2 = ( AS * x_sp2 - b ).norm();

  double dx_qr_kkt  = ( x_qr - x_kkt_dense ).norm();
  double dx_qr_sp   = ( x_qr - x_sp ).norm();
  double dx_qr_sp2  = ( x_qr - x_sp2 ).norm();
  double dx_kkt_sp  = ( x_kkt_dense - x_sp ).norm();
  double dx_kkt_sp2 = ( x_kkt_dense - x_sp2 ).norm();
  double dx_sp_sp2  = ( x_sp - x_sp2 ).norm();

  //--------------------------------------------------------------------------
  // Objective function values
  //--------------------------------------------------------------------------
  auto compute_objective = [&]( const MatrixXd & A_mat, const VectorXd & x, double lam ) -> double
  { return ( A_mat * x - b ).squaredNorm() + lam * x.squaredNorm(); };

  double obj_qr  = compute_objective( AD, x_qr, lambda );
  double obj_kkt = compute_objective( AD, x_kkt_dense, lambda );
  double obj_sp  = compute_objective( AD, x_sp, lambda );
  double obj_sp2 = compute_objective( AD, x_sp2, lambda );

  //--------------------------------------------------------------------------
  // Speed test - Multiple solves
  //--------------------------------------------------------------------------
  const int NTEST   = 100;
  double    t100_qr = 0.0, t100_kkt_dense = 0.0, t100_sp = 0.0, t100_sp2 = 0.0;

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = qr_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_qr = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = kkt_dense_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_kkt_dense = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = sp_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_sp = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = sp2_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // OUTPUT TABLE
  //--------------------------------------------------------------------------

  // Accuracy table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " ACCURACY METRICS " );
  fmt::print( "{:<54} {:<25}\n", "    Metric", "    Value" );
  fmt::print( "{:─<54} {:─<25}\n", "─", "─" );

  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_qr − b‖ (QR Solver)", sci( r_qr ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_kkt − b‖ (KKT dense)", sci( r_kkt_dense ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_sp − b‖ (SP QR Solver)", sci( r_sp ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_sp2 − b‖ (SP KKT Solver)", sci( r_sp2 ) );

  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_kkt‖ (QR vs KKT dense)", sci( dx_qr_kkt ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_sp‖ (QR vs SP QR)", sci( dx_qr_sp ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_sp2‖ (QR vs SP KKT)", sci( dx_qr_sp2 ) );
  fmt::print(
    fg( fmt::color::violet ),
    "{:<54} {:<25}\n",
    "    ‖x_kkt − x_sp‖ (KKT dense vs SP QR)",
    sci( dx_kkt_sp ) );
  fmt::print(
    fg( fmt::color::violet ),
    "{:<54} {:<25}\n",
    "    ‖x_kkt − x_sp2‖ (KKT dense vs SP KKT)",
    sci( dx_kkt_sp2 ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_sp − x_sp2‖ (SP QR vs SP KKT)", sci( dx_sp_sp2 ) );

  // Objective function values
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " OBJECTIVE FUNCTION VALUES " );
  fmt::print( "{:<30} {:<25}\n", "    Solver", "    f(x) = ‖Ax-b‖² + λ‖²x‖²" );
  fmt::print( "{:─<30} {:─<25}\n", "─", "─" );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    QR Solver", sci( obj_qr ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    KKT Solver (dense)", sci( obj_kkt ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    SP QR Solver", sci( obj_sp ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    SP KKT Solver", sci( obj_sp2 ) );

  // Build times table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " BUILD TIMES " );
  fmt::print( "{:<26} {:<26} {:<26}\n", "    Method", "    Time", "    Relative" );
  fmt::print( "{:─<26} {:─<26} {:─<26}\n", "─", "─", "─" );

  std::vector<std::pair<double, std::string>> build_times = { { t_build_qr, "QR Solver (dense)" },
                                                              { t_build_kkt_dense, "KKT Solver (dense)" },
                                                              { t_build_sp, "SP QR Solver (sparse)" },
                                                              { t_build_sp2, "SP KKT Solver (sparse)" } };

  double min_build = std::numeric_limits<double>::infinity();
  for ( const auto & bt : build_times )
  {
    if ( bt.first > 0 && bt.first < min_build ) min_build = bt.first;
  }

  for ( const auto & bt : build_times )
  {
    fmt::print(
      fg( fmt::color::light_green ),
      "{:<26} {:<26} {:<26}\n",
      "    " + bt.second,
      bt.first > 0 ? ms_format( bt.first ) : "    FAILED",
      bt.first > 0 ? ratio_format( bt.first / min_build ) : "    N/A" );
  }

  // Solve times table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " SOLVE TIMES (100 iterations) " );
  fmt::print( "{:<29} {:<16} {:<16} {:<16}\n", "    Method", "    Total Time", "    Time/Solve", "    Speed Ratio" );
  fmt::print( "{:─<29} {:─<16} {:─<16} {:─<16}\n", "─", "─", "─", "─" );

  std::vector<std::tuple<double, std::string>> solve_times = { { t100_qr, "QR Solver (dense)" },
                                                               { t100_kkt_dense, "KKT Solver (dense)" },
                                                               { t100_sp, "SP QR Solver (sparse)" },
                                                               { t100_sp2, "SP KKT Solver (sparse)" } };

  double min_solve = std::numeric_limits<double>::infinity();
  for ( const auto & st : solve_times )
  {
    if ( std::get<0>( st ) < min_solve ) min_solve = std::get<0>( st );
  }

  for ( const auto & st : solve_times )
  {
    double              time = std::get<0>( st );
    const std::string & name = std::get<1>( st );

    fmt::print(
      fg( fmt::color::light_green ),
      "{:<29} {:<16} {:<16} {:<16}\n",
      "    " + name,
      ms_format( time ),
      small_ms_format( time / NTEST ),
      ratio_format( time / min_solve ) );
  }
}
//==============================================================================
// RUN TEST WITH DIAGONAL PENALTY MATRIX D ≠ I
//==============================================================================
static void run_test_with_diagonal( int m, int n, double lambda )
{
  Utils::TicToc tm;

  header( fmt::format( "TEST WITH DIAGONAL D ≠ I: m={}, n={}, λ={}", m, n, lambda ) );

  // ===========================================================================
  // 1. SETUP PROBLEM DATA
  // ===========================================================================

  // Generate random sparse matrix and convert to dense
  SparseMatrix<double> AS = generate_random_sparse_matrix<double>( m, n, 0.3 );
  MatrixXd             AD = MatrixXd( AS );

  // Generate random diagonal matrix D (not identity)
  VectorXd D     = generate_random_vector<double>( n, 0.1, 10.0 );
  VectorXd D_inv = D.cwiseInverse();

  // Right-hand side vector
  VectorXd b = VectorXd::Random( m );

  // Center vector c = 0 (simplified case)
  VectorXd c = VectorXd::Zero( n );

  // ===========================================================================
  // 2. APPROACH A: QR SOLVERS (TRANSFORMATION METHOD)
  // ===========================================================================

  // PROBLEM: minₓ ‖A x - b‖² + λ ‖D x‖²

  // Transformation for QR solvers: y = D x → x = D⁻¹ y
  // Problem becomes: min_y ‖A D⁻¹ y - b‖² + λ ‖y‖²
  MatrixXd             AD_transformed = AD * D_inv.asDiagonal();
  SparseMatrix<double> AS_transformed = AS * D_inv.asDiagonal();

  // ===========================================================================
  // 3. CREATE AND RUN ALL 4 SOLVERS CORRECTLY
  // ===========================================================================

  struct SolverResult
  {
    std::string name;
    std::string type;
    std::string method;
    VectorXd    solution;
    double      build_time_ms;
    double      solve_time_ms;
    double      residual_norm;
    double      objective_value;
  };

  std::vector<SolverResult> results;

  // ---------------------------------------------------------------------------
  // 3.1 DENSE QR SOLVER (TRANSFORMED APPROACH)
  // ---------------------------------------------------------------------------
  {
    tm.tic();
    // Note: Using transformed matrix A' = A D⁻¹
    Utils::TikhonovSolver<double> solver( AD_transformed, lambda );
    VectorXd                      y_solution = solver.solve( b );
    VectorXd                      x_solution = D_inv.asDiagonal() * y_solution;  // x = D⁻¹ y
    tm.toc();
    double build_time = tm.elapsed_ms();

    double residual  = ( AD * x_solution - b ).norm();
    double objective = ( AD * x_solution - b ).squaredNorm() + lambda * ( D.asDiagonal() * x_solution ).squaredNorm();

    results.push_back( { "Dense QR", "QR", "Transformed", x_solution, build_time, 0.0, residual, objective } );
  }

  // ---------------------------------------------------------------------------
  // 3.2 DENSE KKT SOLVER (NATIVE APPROACH)
  // ---------------------------------------------------------------------------
  {
    tm.tic();
    // Pass λ and D separately, not λ*D²!
    // Il solver KKT gestisce internamente il termine λ ‖Dx‖²
    Utils::TikhonovSolver2<double> solver( AD, lambda, D, c );
    VectorXd                       x_solution = solver.solve( b );
    tm.toc();
    double build_time = tm.elapsed_ms();

    double residual  = ( AD * x_solution - b ).norm();
    double objective = ( AD * x_solution - b ).squaredNorm() + lambda * ( D.asDiagonal() * x_solution ).squaredNorm();

    results.push_back( { "Dense KKT", "KKT", "Native", x_solution, build_time, 0.0, residual, objective } );
  }

  // ---------------------------------------------------------------------------
  // 3.3 SPARSE QR SOLVER (TRANSFORMED APPROACH)
  // ---------------------------------------------------------------------------
  {
    tm.tic();
    // Using transformed sparse matrix
    Utils::SP_TikhonovSolver<double> solver( AS_transformed, lambda );
    VectorXd                         y_solution = solver.solve( b );
    VectorXd                         x_solution = D_inv.asDiagonal() * y_solution;
    tm.toc();
    double build_time = tm.elapsed_ms();

    double residual  = ( AD * x_solution - b ).norm();
    double objective = ( AD * x_solution - b ).squaredNorm() + lambda * ( D.asDiagonal() * x_solution ).squaredNorm();

    results.push_back( { "Sparse QR", "QR", "Transformed", x_solution, build_time, 0.0, residual, objective } );
  }

  // ---------------------------------------------------------------------------
  // 3.4 SPARSE KKT SOLVER (NATIVE APPROACH)
  // ---------------------------------------------------------------------------
  {
    tm.tic();
    // Pass λ and D separately
    Utils::SP_TikhonovSolver2<double> solver( AS, lambda, D, c );
    VectorXd                          x_solution = solver.solve( b );
    tm.toc();
    double build_time = tm.elapsed_ms();

    double residual  = ( AD * x_solution - b ).norm();
    double objective = ( AD * x_solution - b ).squaredNorm() + lambda * ( D.asDiagonal() * x_solution ).squaredNorm();

    results.push_back( { "Sparse KKT", "KKT", "Native", x_solution, build_time, 0.0, residual, objective } );
  }

  // ===========================================================================
  // 4. COMPUTE SOLUTION DIFFERENCES
  // ===========================================================================
  const int                N_SOLVERS         = 4;
  MatrixXd                 difference_matrix = MatrixXd::Zero( N_SOLVERS, N_SOLVERS );
  std::vector<std::string> solver_names;

  for ( const auto & r : results ) { solver_names.push_back( r.name ); }

  for ( int i = 0; i < N_SOLVERS; ++i )
  {
    for ( int j = i + 1; j < N_SOLVERS; ++j )
    {
      double diff_norm          = ( results[i].solution - results[j].solution ).norm();
      difference_matrix( i, j ) = diff_norm;
      difference_matrix( j, i ) = diff_norm;
    }
  }

  // ===========================================================================
  // 5. DISPLAY RESULTS
  // ===========================================================================

  // ... (stessa visualizzazione dell'altro test, ma con differenze corrette)

  // ---------------------------------------------------------------------------
  // 5.1 SOLUTION DIFFERENCES
  // ---------------------------------------------------------------------------
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:━^80}\n", " SOLUTION DIFFERENCES ‖xᵢ - xⱼ‖ " );

  // Header
  fmt::print( "{:>20}", "" );
  for ( const auto & name : solver_names ) { fmt::print( "{:>15}", name ); }
  fmt::print( "\n" );

  // Separator
  fmt::print( "{:━>20}", "" );
  for ( size_t i = 0; i < solver_names.size(); ++i ) { fmt::print( "{:━<15}", "" ); }
  fmt::print( "\n" );

  // Matrix - now differences should be near machine epsilon
  for ( int i = 0; i < N_SOLVERS; ++i )
  {
    fmt::print( "     {:<15}", solver_names[i] );
    for ( int j = 0; j < N_SOLVERS; ++j )
    {
      if ( i == j )
        fmt::print( "{:>15}", "———" );
      else
      {
        double diff = difference_matrix( i, j );
        // Color code based on magnitude
        auto color = diff < 1e-10 ? fmt::color::green : diff < 1e-5 ? fmt::color::yellow : fmt::color::red;
        fmt::print( fg( color ), "{:>15}", sci( diff ) );
      }
    }
    fmt::print( "\n" );
  }

  // ===========================================================================
  // 6. VERIFICATION - Check if all methods agree
  // ===========================================================================
  double max_difference = 0.0;
  for ( int i = 0; i < N_SOLVERS; ++i )
  {
    for ( int j = i + 1; j < N_SOLVERS; ++j )
    {
      max_difference = std::max( max_difference, difference_matrix( i, j ) );
    }
  }

  const double tolerance = 1e-6;
  if ( max_difference < tolerance )
  {
    fmt::print(
      fg( fmt::color::green ) | fmt::emphasis::bold,
      "\n✓ VERIFIED: All 4 solvers produce consistent results.!\n" );
  }
  else
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n✗ The methods are inconsistent!\n" );
  }
  fmt::print( "  Maximum difference: {} < {}\n", sci( max_difference ), sci( tolerance ) );
}


// Funzione per stampare la legenda dei metodi
static void print_solver_legend()
{
  fmt::print(
    fg( fmt::color::white ) | fmt::emphasis::bold,
    "\n{:━^80}\n",
    " Methods legend and mathematical formulation " );

  // Descrizione del problema generale
  fmt::print( " Problem: " );
  fmt::print( fg( fmt::color::cyan ), "argmin_x ‖Ax - b‖² + λ ‖Dx‖²\n\n" );

  // -----------------------------------------------------------
  // 1. x_qr (Dense QR)
  // -----------------------------------------------------------
  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, " 1. x_qr (Dense QR)\n" );
  fmt::print( "    • Method: QR factorization on a 'stacked' matrix (full density).\n" );
  fmt::print( "    • Transforms the problem into a standard least-squares problem:\n" );
  fmt::print(
    fg( fmt::color::yellow ),
    "      min ‖ ⎡   A   ⎤     ⎡ b ⎤ ‖ 2\n"
    "          ‖ ⎢       ⎥ x - ⎢   ⎥ ‖\n"
    "          ‖ ⎣ √λ D  ⎦     ⎣ 0 ⎦ ‖\n" );
  fmt::print( "    • Note: Very robust (conditioning √K), but slow for large dimensions.\n\n" );

  // -----------------------------------------------------------
  // 2. x_kkt (Dense KKT)
  // -----------------------------------------------------------
  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, " 2. x_kkt (Dense KKT / Normal Eq)\n" );
  fmt::print( "    • Method: Direct solution of the Normal Equations (full density).\n" );
  fmt::print( "    • Solves the symmetric positive-definite linear system:\n" );
  fmt::print( fg( fmt::color::yellow ), "      ( AᵀA + λ DᵀD ) x = Aᵀb\n" );
  fmt::print(
    "    • Note: Faster than QR (builds the AᵀA matrix), but the conditioning\n"
    "      is squared (K) compared to QR. Less numerically stable.\n\n" );

  // -----------------------------------------------------------
  // 3. x_sp (Sparse QR)
  // -----------------------------------------------------------
  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, " 3. x_sp (Sparse QR)\n" );
  fmt::print( "    • Method: Same as x_qr, but exploits the sparsity of A and D.\n" );
  fmt::print( "    • Uses Eigen::SPQR or Eigen::SparseQR on the stacked matrix:\n" );
  fmt::print(
    fg( fmt::color::yellow ),
    "      min ‖ ⎡ A ⎤ x - ⎡ b ⎤ ‖\n"
    "          ‖ ⎣ Γ ⎦     ⎣ 0 ⎦ ‖   (where Γ = √λ D)\n" );
  fmt::print( "    • Note: Excellent for ill-conditioned sparse matrices.\n\n" );

  // -----------------------------------------------------------
  // 4. x_sp2 (Sparse KKT)
  // -----------------------------------------------------------
  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, " 4. x_sp2 (Sparse KKT / Cholesky)\n" );
  fmt::print( "    • Method: Sparse normal equations with Cholesky decomposition (SimplicialLDLT).\n" );
  fmt::print( "    • Solves:\n" );
  fmt::print( fg( fmt::color::yellow ), "      ( AᵀA + diag(λ dᵢ²) ) x = Aᵀb\n" );
  fmt::print( "    • Note: Generally the fastest method for well-conditioned sparse problems.\n" );

  fmt::print( fg( fmt::color::white ) | fmt::emphasis::bold, "{:━^80}\n", "" );
}

//==============================================================================
// MAIN
//==============================================================================
int main()
{
  print_solver_legend();

  fmt::print(
    fg( fmt::color::lime_green ) | fmt::emphasis::bold,
    "\n{:━^80}\n",
    " BENCHMARK TIKHONOV SOLVERS: QR vs KKT (dense) vs SP QR vs SP KKT " );

  // Test standard (D = I)
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{:━^80}\n", " STANDARD TESTS (D = I) " );

  size_t total_tests  = sizes.size() * lambda_values.size();
  size_t current_test = 0;

  for ( double lambda : lambda_values )
  {
    fmt::print(
      fg( fmt::color::red ) | fmt::emphasis::bold,
      "\n\n"
      "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
      "┃ {:^76} ┃\n"
      "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
      "\n\n",
      fmt::format( "lambda = {}", lambda ) );
    for ( auto sz : sizes )
    {
      ++current_test;
      fmt::print(
        fg( fmt::color::gray ) | fmt::emphasis::faint,
        "\n[{:2d}/{:2d}] ",
        static_cast<int>( current_test ),
        static_cast<int>( total_tests ) );
      try
      {
        run_test_standard( sz.first, sz.second, lambda );
      }
      catch ( const std::exception & e )
      {
        fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\nTest failed with exception: {}\n", e.what() );
      }
    }
  }

  // Test con matrice diagonale D (solo per lambda=0.1 per ridurre il numero di
  // test)
  fmt::print(
    fg( fmt::color::red ) | fmt::emphasis::bold,
    "\n\n"
    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    "┃ {:^76} ┃\n"
    "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
    "\n\n",
    " TESTS WITH DIAGONAL PENALTY MATRIX D (λ=0.1) " );

  double lambda_diag      = 0.1;
  current_test            = 0;
  size_t total_diag_tests = sizes.size();

  for ( auto sz : sizes )
  {
    ++current_test;
    fmt::print(
      fg( fmt::color::gray ) | fmt::emphasis::faint,
      "\n[DIAG {:2d}/{:2d}] ",
      static_cast<int>( current_test ),
      static_cast<int>( total_diag_tests ) );
    try
    {
      run_test_with_diagonal( sz.first, sz.second, lambda_diag );
    }
    catch ( const std::exception & e )
    {
      fmt::print(
        fg( fmt::color::red ) | fmt::emphasis::bold,
        "\nTest with diagonal failed with exception: {}\n",
        e.what() );
    }
  }

  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, "\n{:━^80}\n", " BENCHMARK COMPLETED " );

  return 0;
}
