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
 |      Universita degli Studi di Trento                                    |
 |                                                                          |
\*--------------------------------------------------------------------------*/

// file: Utils_minimize_ParametricSensitivity.hh

/**
 * \file Utils_minimize_ParametricSensitivity.hh
 * \brief Solver-independent parametric sensitivity analysis for smooth
 *        unconstrained and box-constrained minimization problems.
 *
 * Consider the parameter-dependent optimization problem
 * \f[
 *   x^\star(p) = \operatorname*{argmin}_{x} f(x,p).
 * \f]
 *
 * For an unconstrained local minimizer satisfying
 * \f[
 *   \nabla_x f(x^\star(p),p) = 0,
 * \f]
 * differentiation with respect to the parameter vector gives
 * \f[
 *   H\,S + G = 0,
 * \f]
 * where
 * \f[
 *   H = \nabla^2_{xx}f(x^\star,p), \qquad
 *   G = \nabla^2_{xp}f(x^\star,p), \qquad
 *   S = \frac{\partial x^\star}{\partial p}.
 * \f]
 * Therefore
 * \f[
 *   S = -H^{-1}G.
 * \f]
 *
 * For box constraints
 * \f[
 *   \ell \le x \le u,
 * \f]
 * and a locally constant active set, active variables remain fixed while the
 * free variables satisfy the reduced system
 * \f[
 *   H_{FF} S_F = -G_F, \qquad S_A = 0.
 * \f]
 * This formula is valid only while the active set remains unchanged.
 * Sensitivities may be discontinuous at active-set transitions.
 *
 * If the minimized objective contains an additional quadratic regularization
 * \f[
 *   \varepsilon \|x\|^2,
 * \f]
 * the effective Hessian is
 * \f[
 *   H_{\mathrm{eff}} = H + 2\varepsilon I.
 * \f]
 *
 * The class in this file does not own or invoke any minimizer.  The optimum
 * x^star may therefore be produced by Newton, TRON, BBOX, IPOPT, or any other
 * solver.  Sensitivity data can be supplied either directly as derivative
 * matrices or through a parametric derivative callback.
 */

#pragma once

#ifndef UTILS_MINIMIZE_PARAMETRIC_SENSITIVITY_dot_HH
#define UTILS_MINIMIZE_PARAMETRIC_SENSITIVITY_dot_HH

#include "Utils_fmt.hh"
#include "Utils_eigen.hh"

namespace Utils
{

  /**
   * \brief Solver-independent sensitivity analyzer for parametric minima.
   *
   * \tparam Scalar Floating-point scalar type.
   */
  template <typename Scalar = double> class ParametricSensitivity
  {
  public:
using Vector       = ::Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix       = ::Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Index  = Eigen::Index;

    /**
     * \brief Derivative callback evaluated at fixed \f$(x,p)\f$.
     *
     * The callback is intentionally unrelated to any minimizer interface.
     * A null output pointer means that the corresponding derivative is not
     * requested.
     *
     * \param x       Decision variables.
     * \param p       Parameters.
     * \param grad_x  Optional output for \f$\nabla_x f\f$.
     * \param hess_xx Optional output for \f$\nabla^2_{xx}f\f$.
     * \param grad_xp Optional output for \f$\nabla^2_{xp}f\f$.
     * \return Objective value. The sensitivity analyzer does not use it.
     */
    using ParametricCallback = std::function<Scalar(
      Vector const & x,
      Vector const & p,
      Vector *       grad_x,
      Matrix *       hess_xx,
      Matrix *       grad_xp )>;

    /** \brief Active-set classification for box constraints. */
    struct ActiveSet
    {
      std::vector<std::size_t> lower_active;
      std::vector<std::size_t> upper_active;
      std::vector<std::size_t> free;

      [[nodiscard]] bool is_active( std::size_t i ) const
      {
        return std::find( lower_active.begin(), lower_active.end(), i ) != lower_active.end() ||
               std::find( upper_active.begin(), upper_active.end(), i ) != upper_active.end();
      }

      [[nodiscard]] std::size_t n_free() const { return free.size(); }

      [[nodiscard]] std::size_t n_active() const
      {
        return lower_active.size() + upper_active.size();
      }
    };

    /** \brief Numerical and diagnostic options. */
    struct Options
    {
      /** Use finite differences for the mixed derivative matrix G. */
      bool use_finite_differences{ false };

      /** Relative finite-difference step used for parameter perturbations. */
      Scalar fd_epsilon{ Scalar( 1e-7 ) };

      /** Tolerance used to identify active bounds and check complementarity. */
      Scalar active_set_tolerance{ Scalar( 1e-8 ) };

      /** Warn when the free projected gradient exceeds this threshold. */
      Scalar optimality_tolerance{ Scalar( 1e-4 ) };

      /** Add 2*epsilon*I to H before solving the sensitivity equations. */
      bool account_for_regularization{ false };

      /** epsilon in f(x,p) + epsilon*||x||^2. */
      Scalar regularization_epsilon{ Scalar( 0 ) };

      /** 0 = silent, 1 = summary, 2 = diagnostics, 3 = detailed diagnostics. */
      std::size_t verbosity_level{ 1 };
    };

  private:
    Options m_opts;

    Matrix    m_sensitivity;
    ActiveSet m_active_set;

    Scalar      m_condition_number{ Scalar( 0 ) };
    bool        m_success{ false };
    std::string m_error_message;

    [[nodiscard]] static bool finite( Scalar x )
    {
      using std::isfinite;
      return isfinite( x );
    }

    [[nodiscard]] static bool all_finite( Vector const & v )
    {
      return v.array().isFinite().all();
    }

    [[nodiscard]] static bool all_finite( Matrix const & M )
    {
      return M.array().isFinite().all();
    }

    void reset_results()
    {
      m_sensitivity.resize( 0, 0 );
      m_active_set = {};
      m_condition_number = Scalar( 0 );
      m_success = false;
      m_error_message.clear();
    }

    bool fail( std::string msg )
    {
      m_success       = false;
      m_error_message = std::move( msg );
      if ( m_opts.verbosity_level >= 1 )
        fmt::print( fmt::fg( fmt::color::red ), "[Sensitivity] ERROR: {}\n", m_error_message );
      return false;
    }

    [[nodiscard]] ActiveSet identify_active_set(
      Vector const & x,
      Vector const & grad,
      Vector const & lower,
      Vector const & upper ) const
    {
      ActiveSet aset;
      Index const n = x.size();

      aset.lower_active.reserve( static_cast<std::size_t>( n ) );
      aset.upper_active.reserve( static_cast<std::size_t>( n ) );
      aset.free.reserve( static_cast<std::size_t>( n ) );

      Scalar const tol = m_opts.active_set_tolerance;

      for ( Index i = 0; i < n; ++i )
      {
        bool const at_lower = std::abs( x( i ) - lower( i ) ) <= tol;
        bool const at_upper = std::abs( x( i ) - upper( i ) ) <= tol;

        // KKT sign convention for min f(x) with lower <= x <= upper:
        // at lower bound: grad_i >= 0
        // at upper bound: grad_i <= 0
        bool const lower_active = at_lower && grad( i ) >= -tol;
        bool const upper_active = at_upper && grad( i ) <=  tol;

        if ( lower_active && !upper_active )
          aset.lower_active.push_back( static_cast<std::size_t>( i ) );
        else if ( upper_active && !lower_active )
          aset.upper_active.push_back( static_cast<std::size_t>( i ) );
        else if ( lower_active && upper_active )
        {
          // Fixed variable: lower == upper within tolerance.
          aset.lower_active.push_back( static_cast<std::size_t>( i ) );
        }
        else
          aset.free.push_back( static_cast<std::size_t>( i ) );
      }

      return aset;
    }

    [[nodiscard]] Matrix extract_free_submatrix(
      Matrix const & A,
      std::vector<std::size_t> const & free ) const
    {
      Index const nf = static_cast<Index>( free.size() );
      Matrix R( nf, nf );

      for ( Index i = 0; i < nf; ++i )
        for ( Index j = 0; j < nf; ++j )
          R( i, j ) = A(
            static_cast<Index>( free[static_cast<std::size_t>( i )] ),
            static_cast<Index>( free[static_cast<std::size_t>( j )] ) );

      return R;
    }

    [[nodiscard]] Matrix extract_free_rows(
      Matrix const & A,
      std::vector<std::size_t> const & free ) const
    {
      Index const nf = static_cast<Index>( free.size() );
      Matrix R( nf, A.cols() );

      for ( Index i = 0; i < nf; ++i )
        R.row( i ) = A.row( static_cast<Index>( free[static_cast<std::size_t>( i )] ) );

      return R;
    }

    [[nodiscard]] Vector extract_free_entries(
      Vector const & v,
      std::vector<std::size_t> const & free ) const
    {
      Index const nf = static_cast<Index>( free.size() );
      Vector R( nf );

      for ( Index i = 0; i < nf; ++i )
        R( i ) = v( static_cast<Index>( free[static_cast<std::size_t>( i )] ) );

      return R;
    }

    [[nodiscard]] Matrix compute_mixed_derivatives_fd(
      Vector const & x,
      Vector const & p,
      ParametricCallback const & callback ) const
    {
      Index const nx = x.size();
      Index const np = p.size();

      Matrix G( nx, np );
      Vector grad_base( nx );
      callback( x, p, &grad_base, nullptr, nullptr );

      Vector p_pert = p;
      Vector grad_pert( nx );

      for ( Index j = 0; j < np; ++j )
      {
        Scalar const h = m_opts.fd_epsilon * std::max( Scalar( 1 ), std::abs( p( j ) ) );
        p_pert( j ) = p( j ) + h;
        callback( x, p_pert, &grad_pert, nullptr, nullptr );
        G.col( j ) = ( grad_pert - grad_base ) / h;
        p_pert( j ) = p( j );
      }

      return G;
    }

    bool solve_reduced_system(
      Matrix const & hess_xx,
      Matrix const & grad_xp,
      ActiveSet const & active_set )
    {
      Index const nx = hess_xx.rows();
      Index const np = grad_xp.cols();

      m_active_set = active_set;
      m_sensitivity.setZero( nx, np );

      if ( active_set.n_free() == 0 )
      {
        m_success = true;
        m_error_message.clear();
        m_condition_number = Scalar( 0 );

        if ( m_opts.verbosity_level >= 1 )
          fmt::print( fmt::fg( fmt::color::yellow ),
                      "[Sensitivity] all variables are active; dx/dp = 0\n" );
        return true;
      }

      Matrix H = extract_free_submatrix( hess_xx, active_set.free );
      Matrix G = extract_free_rows( grad_xp, active_set.free );

      // Numerical symmetry is useful because LDLT and the condition estimate
      // both assume a self-adjoint reduced Hessian.
      H = Scalar( 0.5 ) * ( H + H.transpose() );

      Eigen::LDLT<Matrix> ldlt;
      ldlt.compute( H );
      if ( ldlt.info() != Eigen::Success )
        return fail( "reduced Hessian LDLT factorization failed" );

      Matrix const S = ldlt.solve( -G );
      if ( ldlt.info() != Eigen::Success || !all_finite( S ) )
        return fail( "reduced sensitivity system could not be solved" );

      Index const nf = static_cast<Index>( active_set.free.size() );
      for ( Index i = 0; i < nf; ++i )
        m_sensitivity.row(
          static_cast<Index>( active_set.free[static_cast<std::size_t>( i )] ) ) = S.row( i );

      Eigen::SelfAdjointEigenSolver<Matrix> eig( H, Eigen::EigenvaluesOnly );
      if ( eig.info() == Eigen::Success && eig.eigenvalues().size() > 0 )
      {
        auto const ev_abs = eig.eigenvalues().cwiseAbs();
        Scalar const emax = ev_abs.maxCoeff();
        Scalar const emin = ev_abs.minCoeff();
        Scalar const eps  = std::numeric_limits<Scalar>::epsilon();
        m_condition_number = emin > eps * std::max( Scalar( 1 ), emax )
                           ? emax / emin
                           : std::numeric_limits<Scalar>::infinity();
      }
      else
        m_condition_number = std::numeric_limits<Scalar>::quiet_NaN();

      m_success = true;
      m_error_message.clear();
      return true;
    }

    void print_summary( bool has_bounds ) const
    {
      if ( m_opts.verbosity_level == 0 ) return;

      if ( m_success )
      {
        fmt::print( fmt::fg( fmt::color::green ),
                    "[Sensitivity] sensitivity successfully computed\n" );
        if ( has_bounds )
          fmt::print( "  free: {}, active: {}\n",
                      m_active_set.n_free(), m_active_set.n_active() );
        fmt::print( "  condition number: {:.3e}\n", m_condition_number );
        if ( m_sensitivity.size() > 0 )
          fmt::print( "  max |dx_i/dp_j|: {:.3e}\n",
                      m_sensitivity.cwiseAbs().maxCoeff() );
      }
    }

  public:
    explicit ParametricSensitivity( Options opts = {} )
      : m_opts( std::move( opts ) )
    {}

    [[nodiscard]] Options const & options() const { return m_opts; }
    Options & options() { return m_opts; }

    [[nodiscard]] Matrix const & sensitivity() const { return m_sensitivity; }
    [[nodiscard]] ActiveSet const & active_set() const { return m_active_set; }
    [[nodiscard]] Scalar condition_number() const { return m_condition_number; }
    [[nodiscard]] bool success() const { return m_success; }
    [[nodiscard]] std::string const & error_message() const { return m_error_message; }

    [[nodiscard]] Scalar sensitivity_norm() const
    {
      return m_sensitivity.size() > 0 ? m_sensitivity.norm() : Scalar( 0 );
    }

    [[nodiscard]] Scalar max_sensitivity() const
    {
      return m_sensitivity.size() > 0
           ? m_sensitivity.cwiseAbs().maxCoeff()
           : Scalar( 0 );
    }

    /**
     * \brief Compute sensitivity directly from derivative data.
     *
     * This is the lowest-level and most solver-independent interface.
     *
     * \param x_opt   Optimal point.
     * \param grad_x  Gradient \f$\nabla_x f(x^\star,p)\f$.
     * \param hess_xx Hessian \f$\nabla^2_{xx}f(x^\star,p)\f$.
     * \param grad_xp Mixed derivative matrix \f$\nabla^2_{xp}f(x^\star,p)\f$.
     * \param lower   Optional lower bounds. Must be paired with upper.
     * \param upper   Optional upper bounds. Must be paired with lower.
     * \return true on success.
     */
    bool compute_sensitivity(
      Vector const & x_opt,
      Vector const & grad_x,
      Matrix const & hess_xx,
      Matrix const & grad_xp,
      Vector const & lower = Vector(),
      Vector const & upper = Vector() )
    {
      reset_results();

      Index const nx = x_opt.size();
      bool const no_bounds  = lower.size() == 0 && upper.size() == 0;
      bool const has_bounds = lower.size() == nx && upper.size() == nx;

      if ( nx == 0 )
        return fail( "x_opt is empty" );
      if ( grad_x.size() != nx )
        return fail( "grad_x has incompatible size" );
      if ( hess_xx.rows() != nx || hess_xx.cols() != nx )
        return fail( "hess_xx must be nx-by-nx" );
      if ( grad_xp.rows() != nx )
        return fail( "grad_xp must have nx rows" );
      if ( !no_bounds && !has_bounds )
        return fail( "lower and upper must both be empty or both have size nx" );
      if ( !all_finite( x_opt ) || !all_finite( grad_x ) ||
           !all_finite( hess_xx ) || !all_finite( grad_xp ) )
        return fail( "non-finite derivative data" );
      if ( has_bounds )
      {
        if ( !all_finite( lower ) || !all_finite( upper ) )
          return fail( "non-finite bounds" );
        if ( ( lower.array() > upper.array() ).any() )
          return fail( "inconsistent bounds: lower > upper" );
      }

      Matrix H = hess_xx;
      if ( m_opts.account_for_regularization )
      {
        if ( m_opts.regularization_epsilon < Scalar( 0 ) )
          return fail( "regularization_epsilon must be non-negative" );
        H.diagonal().array() += Scalar( 2 ) * m_opts.regularization_epsilon;
      }

      ActiveSet aset;
      if ( has_bounds )
        aset = identify_active_set( x_opt, grad_x, lower, upper );
      else
      {
        aset.free.resize( static_cast<std::size_t>( nx ) );
        std::iota( aset.free.begin(), aset.free.end(), std::size_t( 0 ) );
      }

      if ( !aset.free.empty() )
      {
        Vector const g_free = extract_free_entries( grad_x, aset.free );
        Scalar const gn = g_free.norm();
        if ( gn > m_opts.optimality_tolerance && m_opts.verbosity_level >= 1 )
          fmt::print( fmt::fg( fmt::color::yellow ),
                      "[Sensitivity] WARNING: ||grad_free|| = {:.3e} > {:.3e}\n",
                      gn, m_opts.optimality_tolerance );
      }

      bool const ok = solve_reduced_system( H, grad_xp, aset );
      print_summary( has_bounds );
      return ok;
    }

    /**
     * \brief Compute sensitivity by requesting derivatives from a callback.
     *
     * The callback is evaluated once at \f$(x^\star,p)\f$ to obtain gradient
     * and Hessian. The mixed derivative matrix is obtained analytically unless
     * Options::use_finite_differences is true.
     *
     * \param x_opt    Optimal point supplied by an arbitrary external solver.
     * \param p        Current parameter vector.
     * \param callback Derivative callback.
     * \param lower    Optional lower bounds.
     * \param upper    Optional upper bounds.
     * \return true on success.
     */
    bool compute_sensitivity(
      Vector const &             x_opt,
      Vector const &             p,
      ParametricCallback const & callback,
      Vector const &             lower = Vector(),
      Vector const &             upper = Vector() )
    {
      reset_results();

      Index const nx = x_opt.size();
      Index const np = p.size();
      if ( nx == 0 ) return fail( "x_opt is empty" );

      Vector grad_x( nx );
      Matrix hess_xx( nx, nx );
      // The callback API historically expects a pre-sized output matrix and
      // fills it with setZero()/coefficient assignments.  Keep that contract
      // while using NaN sentinels so an incomplete callback is still caught.
      Matrix grad_xp( nx, np );
      grad_xp.setConstant( std::numeric_limits<Scalar>::quiet_NaN() );

      if ( m_opts.use_finite_differences )
      {
        callback( x_opt, p, &grad_x, &hess_xx, nullptr );
        grad_xp = compute_mixed_derivatives_fd( x_opt, p, callback );
      }
      else
      {
        callback( x_opt, p, &grad_x, &hess_xx, &grad_xp );

        if ( grad_xp.rows() != nx || grad_xp.cols() != np || !all_finite( grad_xp ) )
          return fail(
            "callback did not provide a complete finite nx-by-np mixed derivative matrix; "
            "enable use_finite_differences or provide grad_xp analytically" );
      }

      return compute_sensitivity( x_opt, grad_x, hess_xx, grad_xp, lower, upper );
    }

    /**
     * \brief Predict the optimum after a parameter perturbation.
     *
     * Uses the first-order approximation
     * \f[
     *   x^\star(p+\delta p) \approx x^\star(p) + S\,\delta p.
     * \f]
     */
    [[nodiscard]] Vector predict_new_optimum(
      Vector const & x_opt,
      Vector const & delta_p ) const
    {
      return x_opt + m_sensitivity * delta_p;
    }
  };

} // namespace Utils

#endif
