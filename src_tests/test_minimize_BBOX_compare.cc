// SPDX-License-Identifier: MPL-2.0

#include "Utils_fmt.hh"
#include "Utils_minimize_BBOX_IPNewton.hh"
#include "Utils_minimize_BBOX_Newton.hh"
#include "Utils_minimize_BBOX_NewtonCubic.hh"
#include "Utils_minimize_BBOX_TRON.hh"
#include "Utils_minimize_BBOX_small_TRON.hh"
#include "Utils_minimize_Newton.hh"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ND_func.cxx"

namespace
{
  using Scalar       = double;
  using Vector       = Eigen::VectorXd;
  using Matrix       = Eigen::MatrixXd;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  constexpr std::size_t MAX_ITERATIONS = 400;
  constexpr int SOLVER_WIDTH = 27, STATUS_WIDTH = 15, TABLE_WIDTH = 116;

  enum class Outcome { converged, stopped, failed };

  struct Metrics
  {
    std::string solver, status{ "FAILED" }, message;
    Outcome outcome{ Outcome::failed };
    std::size_t iterations{ 0 }, function_evaluations{ 0 }, gradient_evaluations{ 0 }, second_order_evaluations{ 0 };
    Scalar objective{ std::numeric_limits<Scalar>::quiet_NaN() };
    Scalar projected_gradient_norm{ std::numeric_limits<Scalar>::quiet_NaN() };
    Vector x;
  };

  struct Comparison { std::string name; int dimension{ 0 }; std::array<Metrics, 6> results; };
  std::vector<Comparison> comparisons;

  class NDProblemAdapter
  {
  public:
    explicit NDProblemAdapter( std::shared_ptr<NDbase<Scalar>> problem ) : m_problem( std::move( problem ) ) {}
    Scalar objective( Vector const & x ) { return ( *m_problem )( x ); }
    void gradient( Vector const & x, Vector & g ) { g = m_problem->gradient( x ); }
    void gradient( Utils::ConstVectorRef<Scalar> x, Utils::VectorRef<Scalar> g ) { g = m_problem->gradient( x ); }
    void hessian( Vector const & x, Matrix & H ) { update( x ); H = Matrix( m_H ); }
    void hessian( Vector const & x, Utils::MatrixRef<Scalar> H ) { update( x ); H = m_H; }
    void hprod( Vector const & x, Vector const & v, Vector & Hv ) { update( x ); Hv.noalias() = m_H * v; }
    SparseMatrix const & sparse_hessian( Vector const & x ) { update( x ); return m_H; }
  private:
    void update( Vector const & x )
    {
      if ( !m_valid || m_x.size() != x.size() || ( m_x.array() != x.array() ).any() )
      { m_x = x; m_H = m_problem->hessian( x ); m_valid = true; }
    }
    std::shared_ptr<NDbase<Scalar>> m_problem;
    Vector m_x; SparseMatrix m_H; bool m_valid{ false };
  };

  fmt::text_style outcome_style( Outcome outcome )
  {
    if ( outcome == Outcome::converged ) return fmt::fg( fmt::color::lime_green ) | fmt::emphasis::bold;
    if ( outcome == Outcome::stopped ) return fmt::fg( fmt::color::gold );
    return fmt::fg( fmt::color::red ) | fmt::emphasis::bold;
  }

  template <typename Function> Metrics guarded_run( std::string name, Function && function )
  {
    try { Metrics m = function(); m.solver = std::move( name ); return m; }
    catch ( std::exception const & e ) { Metrics m; m.solver = std::move( name ); m.status = "EXCEPTION"; m.message = e.what(); return m; }
    catch ( ... ) { Metrics m; m.solver = std::move( name ); m.status = "EXCEPTION"; m.message = "unknown exception"; return m; }
  }

  Metrics run_ipnewton( std::shared_ptr<NDbase<Scalar>> const & p, Vector const & x0, Vector const & l, Vector const & u )
  {
    NDProblemAdapter problem( p );
    using Solver = Utils::minimize_BBOX_IPNewton<Scalar>;
    Solver::Options options; options.max_outer_iterations = MAX_ITERATIONS; options.max_inner_iterations = MAX_ITERATIONS;
    options.tol = 1e-12; options.verbosity = 0;
    Solver solver( x0.size(), options ); auto const r = solver.solve( problem, x0, l, u );
    Metrics m; m.status = Solver::to_string( r.status ); m.iterations = r.iterations;
    m.function_evaluations = r.function_evaluations; m.second_order_evaluations = r.hessian_evaluations;
    m.objective = r.objective; m.projected_gradient_norm = r.projected_gradient_norm; m.x = r.x;
    if ( r.status == Solver::Status::CONVERGED ) m.outcome = Outcome::converged;
    else if ( r.status == Solver::Status::MAX_ITERATIONS || r.status == Solver::Status::BARRIER_FAILED ) m.outcome = Outcome::stopped;
    return m;
  }

  template <typename Solver> Metrics run_newton_family(
    std::shared_ptr<NDbase<Scalar>> const & p, Vector const & x0, Vector const & l, Vector const & u )
  {
    NDProblemAdapter problem( p ); Utils::Options<Scalar> options;
    options.max_iterations = MAX_ITERATIONS; options.max_function_evaluations = MAX_ITERATIONS + 1; options.set_tolerances( 1e-12 );
    Solver solver( x0.size(), options ); auto const r = solver.solve( problem, x0, l, u );
    Metrics m; m.status = std::string( Utils::to_string( r.status ) ); m.iterations = r.iterations;
    m.function_evaluations = r.function_evaluations; m.gradient_evaluations = r.gradient_evaluations;
    m.second_order_evaluations = r.hessian_evaluations; m.objective = r.objective;
    m.projected_gradient_norm = r.projected_gradient_norm; m.x = r.x;
    if ( r.status == Utils::Status::converged ) m.outcome = Outcome::converged;
    else if ( r.status == Utils::Status::max_iterations || r.status == Utils::Status::max_function_evaluations ||
              r.status == Utils::Status::no_progress ) m.outcome = Outcome::stopped;
    return m;
  }

  Metrics run_smalltron( std::shared_ptr<NDbase<Scalar>> const & p, Vector const & x0, Vector const & l, Vector const & u )
  {
    NDProblemAdapter problem( p ); Utils::SmallTRON_details::Options<Scalar> options;
    options.max_iter = MAX_ITERATIONS; options.max_eval = MAX_ITERATIONS + 1; options.set_tolerances( 1e-12 );
    Utils::Minimize_BBOX_SmallTRON<Scalar> solver( x0.size(), options ); auto const r = solver.solve( problem, x0, l, u );
    Metrics m; m.status = std::string( Utils::SmallTRON_details::to_string( r.status ) ); m.iterations = r.iter;
    m.function_evaluations = r.obj_evals; m.gradient_evaluations = r.grad_evals; m.second_order_evaluations = r.hess_evals;
    m.objective = r.objective; m.projected_gradient_norm = r.dual_feas; m.x = r.x;
    using Status = Utils::SmallTRON_details::Status;
    if ( r.status == Status::first_order ) m.outcome = Outcome::converged;
    else if ( r.status == Status::max_iter || r.status == Status::max_eval || r.status == Status::small_step ) m.outcome = Outcome::stopped;
    return m;
  }

  Metrics run_tron( std::shared_ptr<NDbase<Scalar>> const & p, Vector const & x0, Vector const & l, Vector const & u )
  {
    NDProblemAdapter problem( p ); Utils::TRON2_details::Options<Scalar> options;
    options.max_iterations = MAX_ITERATIONS; options.max_function_evaluations = MAX_ITERATIONS + 1;
    options.absolute_tolerance = options.relative_tolerance = 1e-12; options.cg_tolerance = 1e-6;
    Utils::Minimize_BBOX_TRON<Scalar> solver( x0.size(), options );
    auto const r = solver.solve( x0, l, u,
      [&]( Vector const & x ) { return problem.objective( x ); },
      [&]( Vector const & x, Vector & g ) { problem.gradient( x, g ); },
      [&]( Vector const & x, Vector const & v, Vector & Hv ) { problem.hprod( x, v, Hv ); } );
    Metrics m; m.status = std::string( Utils::TRON2_details::to_string( r.status ) ); m.iterations = r.iterations;
    m.function_evaluations = r.function_evaluations; m.gradient_evaluations = r.gradient_evaluations;
    m.second_order_evaluations = r.hessian_vector_evaluations; m.objective = r.objective;
    m.projected_gradient_norm = r.projected_gradient_norm; m.x = r.x;
    using Status = Utils::TRON2_details::Status;
    if ( r.status == Status::converged ) m.outcome = Outcome::converged;
    else if ( r.status == Status::max_iterations || r.status == Status::max_function_evaluations || r.status == Status::small_step ) m.outcome = Outcome::stopped;
    return m;
  }

  Metrics run_legacy_newton( std::shared_ptr<NDbase<Scalar>> const & p, Vector const & x0, Vector const & l, Vector const & u )
  {
    NDProblemAdapter problem( p ); using Solver = Utils::Newton_minimizer<Scalar>;
    Solver::Options options; options.max_iter = MAX_ITERATIONS; options.g_tol = 1e-12; options.verbosity = 0;
    Solver solver( options ); solver.set_bounds( l, u );
    Solver::Callback callback = [&]( Vector const & x, Vector * g, Matrix * H ) -> Scalar
    { if ( g ) problem.gradient( x, *g ); if ( H ) problem.hessian( x, *H ); return problem.objective( x ); };
    solver.minimize( x0, callback ); Metrics m; m.status = Solver::to_string( solver.status() );
    m.iterations = solver.iterations(); m.function_evaluations = solver.function_evals();
    m.second_order_evaluations = solver.hessian_evals(); m.objective = solver.final_f();
    m.projected_gradient_norm = solver.final_grad_norm(); m.x = solver.solution();
    if ( solver.status() == Solver::Status::CONVERGED ) m.outcome = Outcome::converged;
    else if ( solver.status() == Solver::Status::MAX_ITERATIONS || solver.status() == Solver::Status::STALLED ||
              solver.status() == Solver::Status::LINE_SEARCH_FAILED ) m.outcome = Outcome::stopped;
    return m;
  }

  void print_rule( std::string_view fill = "─" ) { for ( int i = 0; i < TABLE_WIDTH; ++i ) fmt::print( "{}", fill ); fmt::print( "\n" ); }

  void print_metrics( Metrics const & m )
  {
    std::string_view status = m.status; if ( status.size() > STATUS_WIDTH ) status = status.substr( 0, STATUS_WIDTH );
    fmt::print( "  {:<{}} │ ", m.solver, SOLVER_WIDTH ); fmt::print( outcome_style( m.outcome ), "{:<{}}", status, STATUS_WIDTH );
    fmt::print( " {:>6} {:>7} {:>7} {:>7} {:>14.6e} {:>12.3e}\n", m.iterations, m.function_evaluations,
                m.gradient_evaluations, m.second_order_evaluations, m.objective, m.projected_gradient_norm );
    if ( !m.message.empty() ) fmt::print( fmt::fg( fmt::color::red ), "      {}\n", m.message );
  }

  void print_comparison( Comparison const & c )
  {
    print_rule( "═" ); fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "TEST: {}  [dimensione {}]\n", c.name, c.dimension );
    print_rule(); fmt::print( fmt::emphasis::bold, "  {:<{}} │ {:<{}} {:>6} {:>7} {:>7} {:>7} {:>14} {:>12}\n",
      "Solver", SOLVER_WIDTH, "Status", STATUS_WIDTH, "Iter", "F", "G", "H/Hv", "f(x)", "ǁPgradǁ" ); print_rule();
    for ( auto const & m : c.results ) print_metrics( m );
  }

  struct Totals { std::string solver; std::size_t converged{}, stopped{}, failed{}, iterations{}, f{}, g{}, h{}, best{}; };
  void print_summary( std::size_t skipped )
  {
    std::array<Totals, 6> totals; if ( !comparisons.empty() ) for ( int i = 0; i < 6; ++i ) totals[i].solver = comparisons.front().results[i].solver;
    for ( auto const & c : comparisons )
    {
      Scalar best = std::numeric_limits<Scalar>::infinity(); for ( auto const & m : c.results ) if ( std::isfinite( m.objective ) ) best = std::min( best, m.objective );
      Scalar tol = 1e-10 * std::max( Scalar( 1 ), std::abs( best ) );
      for ( int i = 0; i < 6; ++i ) { auto const & m = c.results[i]; auto & t = totals[i];
        if ( m.outcome == Outcome::converged ) ++t.converged; else if ( m.outcome == Outcome::stopped ) ++t.stopped; else ++t.failed;
        t.iterations += m.iterations; t.f += m.function_evaluations; t.g += m.gradient_evaluations; t.h += m.second_order_evaluations;
        if ( std::isfinite( m.objective ) && m.objective <= best + tol ) ++t.best; }
    }
    print_rule( "═" ); fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "RIEPILOGO DEI SEI SOLVER\n" );
    fmt::print( "Problemi confrontati: {}; saltati: {}; limite iterazioni: {}\n", comparisons.size(), skipped, MAX_ITERATIONS ); print_rule();
    fmt::print( fmt::emphasis::bold, "  {:<{}} {:>6} {:>6} {:>6} {:>8} {:>9} {:>9} {:>10} {:>6}\n",
      "Solver", SOLVER_WIDTH, "Conv", "Stop", "Fail", "Iter", "F", "G", "H/Hv", "Best" );
    for ( auto const & t : totals ) fmt::print( "  {:<{}} {:>6} {:>6} {:>6} {:>8} {:>9} {:>9} {:>10} {:>6}\n",
      t.solver, SOLVER_WIDTH, t.converged, t.stopped, t.failed, t.iterations, t.f, t.g, t.h, t.best ); print_rule( "═" );
  }
}

int main()
{
  fmt::print( "\n" ); print_rule( "═" );
  fmt::print( fmt::emphasis::bold | fmt::fg( fmt::color::cyan ), "CONFRONTO VERTICALE DEI SOLVER BBOX — {} PROBLEMI ND_func\n", NL_list.size() );
  fmt::print( "Ogni test è seguito da sei righe, una per solver.\n" );
  std::size_t skipped{};
  for ( auto const & [problem, name] : NL_list )
  {
    if ( name == "Katsuura10D" || name == "MichalewiczN10D" ) { ++skipped; continue; }
    Eigen::VectorXd const lower = problem->lower(), upper = problem->upper(), x0 = problem->init();
    Comparison c; c.name = name; c.dimension = x0.size();
    c.results = {
      guarded_run( "minimize_BBOX_IPNewton", [&] { return run_ipnewton( problem, x0, lower, upper ); } ),
      guarded_run( "Minimize_BBOX_Newton", [&] { return run_newton_family<Utils::Minimize_BBOX_Newton<Scalar>>( problem, x0, lower, upper ); } ),
      guarded_run( "Minimize_BBOX_NewtonCubic", [&] { return run_newton_family<Utils::Minimize_BBOX_NewtonCubic<Scalar>>( problem, x0, lower, upper ); } ),
      guarded_run( "Minimize_BBOX_SmallTRON", [&] { return run_smalltron( problem, x0, lower, upper ); } ),
      guarded_run( "Minimize_BBOX_TRON", [&] { return run_tron( problem, x0, lower, upper ); } ),
      guarded_run( "Newton_minimizer", [&] { return run_legacy_newton( problem, x0, lower, upper ); } ) };
    comparisons.emplace_back( std::move( c ) ); print_comparison( comparisons.back() );
  }
  print_summary( skipped ); return comparisons.empty() ? 1 : 0;
}
