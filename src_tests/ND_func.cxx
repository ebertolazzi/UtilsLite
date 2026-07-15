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

/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  MADS test suite for gradient-free optimization problems                 |
 |                                                                          |
 |  Updated adaptation with verbose output and final x saving               |
\*--------------------------------------------------------------------------*/

using std::string;

template <typename T> class NDbase
{
public:
  NDbase()          = default;
  virtual ~NDbase() = default;

  virtual std::string bibtex() const = 0;

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  virtual Vector lower() const = 0;
  virtual Vector upper() const = 0;
  virtual Vector init() const  = 0;
  virtual Vector exact() const = 0;

  /// Objective function
  virtual T operator()( Vector const & x ) const = 0;

  /// Gradient ∇f(x)
  virtual Vector gradient( Vector const & x ) const = 0;

  virtual SparseMatrix hessian( Vector const & x ) const = 0;
};

// -------------------------------------------------------------------
// Ackley Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class AckleyN
 * @brief Ackley function, a multimodal test function
 *
 * The Ackley function is widely used for testing optimization algorithms.
 * It has many local minima and a global minimum at the origin with value 0.
 * The function features an exponential term wrapped by a cosine wave.
 */
template <typename T, int N> class AckleyN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{ackley1987,\n"
           "  title={A connectionist machine for genetic hillclimbing},\n"
           "  author={Ackley, David H.},\n"
           "  journal={Kluwer Academic Publishers},\n"
           "  year={1987}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "AckleyN requires N >= 1" );

  Vector lower() const override { return Vector::Constant( N, -32.768 ); }
  Vector upper() const override { return Vector::Constant( N, 32.768 ); }
  Vector init() const override { return Vector::Constant( N, 1.0 ); }
  Vector exact() const override { return Vector::Zero( N ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    const T a = 20.0;
    const T b = 0.2;
    const T c = 2.0 * M_PI;

    T sum1 = 0.0;
    T sum2 = 0.0;

    for ( int i = 0; i < N; ++i )
    {
      sum1 += x[i] * x[i];
      sum2 += std::cos( c * x[i] );
    }

    sum1 /= T( N );
    sum2 /= T( N );

    return -a * std::exp( -b * std::sqrt( sum1 ) ) - std::exp( sum2 ) + a + std::exp( 1.0 );
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    const T a = 20.0;
    const T b = 0.2;
    const T c = 2.0 * M_PI;

    T sum1          = x.squaredNorm();
    T avg_sum1      = sum1 / T( N );
    T sqrt_avg_sum1 = std::sqrt( avg_sum1 );

    T sum2 = 0.0;
    for ( int i = 0; i < N; ++i ) sum2 += std::cos( c * x[i] );
    T avg_sum2 = sum2 / T( N );

    T exp1 = std::exp( -b * sqrt_avg_sum1 );
    T exp2 = std::exp( avg_sum2 );

    Vector g( N );
    // Evitiamo divisione per zero nell'origine
    T common_prefix = ( sqrt_avg_sum1 > 1e-14 ) ? ( a * b * exp1 ) / ( T( N ) * sqrt_avg_sum1 ) : 0.0;

    for ( int i = 0; i < N; ++i )
    {
      T term1 = common_prefix * x[i];
      T term2 = ( exp2 * c / T( N ) ) * std::sin( c * x[i] );
      g[i]    = term1 + term2;
    }
    return g;
  }

  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N * N );

    const T a = 20.0;
    const T b = 0.2;
    const T c = 2.0 * M_PI;

    const T sum1          = x.squaredNorm();
    const T avg_sum1      = sum1 / T( N );
    const T sqrt_avg_sum1 = std::sqrt( avg_sum1 );

    T sum2 = 0.0;
    for ( int i = 0; i < N; ++i ) sum2 += std::cos( c * x[i] );
    const T avg_sum2 = sum2 / T( N );

    const T exp1 = std::exp( -b * sqrt_avg_sum1 );
    const T exp2 = std::exp( avg_sum2 );

    if ( sqrt_avg_sum1 < 1e-12 )
    {
      SparseMatrix H( N, N );
      return H;
    }

    const T common_T1 = ( a * b * exp1 ) / ( T( N ) * sqrt_avg_sum1 );
    const T common_T2 = ( c * c * exp2 ) / T( N );
    const T inv_N     = 1.0 / T( N );

    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        T Hij_T1 = common_T1 * ( ( ( i == j ) ? 1.0 : 0.0 ) - ( x[i] * x[j] ) / ( T( N ) * avg_sum1 ) -
                                 ( b * x[i] * x[j] ) / ( T( N ) * sqrt_avg_sum1 ) );

        // CORREZIONE: sostituito '+' con '-' prima del termine dei seni
        T Hij_T2 = common_T2 * ( ( ( i == j ) ? std::cos( c * x[i] ) : 0.0 ) -
                                 ( std::sin( c * x[i] ) * std::sin( c * x[j] ) ) * inv_N );

        T Hij = Hij_T1 + Hij_T2;

        if ( std::abs( Hij ) > 1e-18 ) { triplets.emplace_back( i, j, Hij ); }
      }
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Beale Function (2D)
// -------------------------------------------------------------------

/**
 * @class Beale2D
 * @brief Beale function, a 2D optimization test function
 *
 * The Beale function has a global minimum at (3,0.5) with value 0.
 * It features steep valleys and is multimodal.
 */
template <typename T> class Beale2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{beale1958,\n"
           "  title={On an iterative method for finding a local minimum of a function of more than one variable},\n"
           "  author={Beale, E. M. L.},\n"
           "  journal={The Computer Journal},\n"
           "  volume={1},\n"
           "  number={1},\n"
           "  pages={5--8},\n"
           "  year={1958}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -4.5 ); }
  Vector upper() const override { return Vector::Constant( 2, 4.5 ); }

  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 1.0, 1.0;
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 3.0, 0.5;
    return x;
  }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T t1 = 1.5 - x1 * ( 1.0 - x2 );
    T t2 = 2.25 - x1 * ( 1.0 - x2 * x2 );
    T t3 = 2.625 - x1 * ( 1.0 - x2 * x2 * x2 );
    return t1 * t1 + t2 * t2 + t3 * t3;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];

    T t1 = 1.5 - x1 * ( 1.0 - x2 );
    T t2 = 2.25 - x1 * ( 1.0 - x2 * x2 );
    T t3 = 2.625 - x1 * ( 1.0 - x2 * x2 * x2 );

    Vector g( 2 );
    g[0] = -2.0 * t1 * ( 1.0 - x2 ) - 2.0 * t2 * ( 1.0 - x2 * x2 ) - 2.0 * t3 * ( 1.0 - x2 * x2 * x2 );
    g[1] = 2.0 * t1 * x1 + 4.0 * t2 * x1 * x2 + 6.0 * t3 * x1 * x2 * x2;

    return g;
  }

  /// Hessian ∇²f(x) sparse 2x2
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];

    T t1 = 1.5 - x1 * ( 1.0 - x2 );
    T t2 = 2.25 - x1 * ( 1.0 - x2 * x2 );
    T t3 = 2.625 - x1 * ( 1.0 - x2 * x2 * x2 );

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    // ∂²f/∂x1²
    H.insert( 0, 0 ) = 2.0 * ( 1.0 - x2 ) * ( 1.0 - x2 ) + 2.0 * ( 1.0 - x2 * x2 ) * ( 1.0 - x2 * x2 ) +
                       2.0 * ( 1.0 - x2 * x2 * x2 ) * ( 1.0 - x2 * x2 * x2 );

    // ∂²f/∂x1∂x2 = ∂²f/∂x2∂x1 (cross derivative)
    // Derivando g[1] rispetto a x1, oppure g[0] rispetto a x2
    T cross_deriv = 2.0 * ( t1 + 2.0 * x2 * t2 + 3.0 * x2 * x2 * t3 ) - 2.0 * x1 * ( 1.0 - x2 ) -
                    4.0 * x1 * x2 * ( 1.0 - x2 * x2 ) - 6.0 * x1 * x2 * x2 * ( 1.0 - x2 * x2 * x2 );

    H.insert( 0, 1 ) = cross_deriv;
    H.insert( 1, 0 ) = cross_deriv;

    // ∂²f/∂x2²
    H.insert( 1, 1 ) = 2.0 * x1 * x1 + 8.0 * x1 * x1 * x2 * x2 + 18.0 * x1 * x1 * x2 * x2 * x2 * x2 + 4.0 * x1 * t2 +
                       12.0 * x1 * x2 * t3;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Booth Function (2D)
// -------------------------------------------------------------------

/**
 * @class Booth2D
 * @brief Booth function, a 2D optimization test function
 *
 * The Booth function has a global minimum at (1,3) with value 0.
 * It is a simple quadratic function used for testing optimization algorithms.
 */
template <typename T> class Booth2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@book{booth1975,\n"
           "  title={Numerical methods},\n"
           "  author={Booth, A. D.},\n"
           "  publisher={Butterworths},\n"
           "  year={1975}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }

  // CORRETTO: Punto iniziale lontano dall'ottimo (1,3) per testare la convergenza
  Vector init() const override
  {
    Vector x( 2 );
    x << -10.0, -10.0;
    return x;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 1.0, 3.0;
    return x;
  }

  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    return std::pow( x1 + 2 * x2 - 7, 2 ) + std::pow( 2 * x1 + x2 - 5, 2 );
  }

  Vector gradient( Vector const & x ) const override
  {
    T      x1 = x[0], x2 = x[1];
    Vector g( 2 );
    // Semplificazione algebrica opzionale ma più pulita:
    // df/dx = 10x + 8y - 34
    // df/dy = 8x + 10y - 38
    g[0] = 2 * ( x1 + 2 * x2 - 7 ) + 4 * ( 2 * x1 + x2 - 5 );
    g[1] = 4 * ( x1 + 2 * x2 - 7 ) + 2 * ( 2 * x1 + x2 - 5 );
    return g;
  }

  SparseMatrix hessian( Vector const & ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    // CORRETTO: Gli elementi diagonali sono entrambi 10.0
    triplets.emplace_back( 0, 0, 10.0 );
    triplets.emplace_back( 0, 1, 8.0 );
    triplets.emplace_back( 1, 0, 8.0 );
    triplets.emplace_back( 1, 1, 10.0 );  // Era 8.0 nel tuo codice

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Broyden Tridiagonal Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class BroydenTridiagonalN
 * @brief Broyden tridiagonal function, a scalable test function
 *
 * This function represents a tridiagonal system and is useful for testing
 * algorithms on problems with banded structure. The global minimum is at
 * (1,1,...,1).
 */
template <typename T, int N> class BroydenTridiagonalN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{broyden1965,\n"
           "  title={A class of methods for solving nonlinear simultaneous equations},\n"
           "  author={Broyden, C. G.},\n"
           "  journal={Mathematics of Computation},\n"
           "  volume={19},\n"
           "  pages={577--593},\n"
           "  year={1965}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 2, "BroydenTridiagonalN requires N >= 2" );

  Vector lower() const override { return Vector::Constant( N, -10.0 ); }
  Vector upper() const override { return Vector::Constant( N, 10.0 ); }
  Vector init() const override { return Vector::Constant( N, 0.5 ); }
  Vector exact() const override { return Vector::Constant( N, 1.0 ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = 0.0;
    for ( int i = 0; i < N; ++i )
    {
      T xim1 = ( i == 0 ) ? 0.0 : x[i - 1];
      T t    = ( 3.0 - 2.0 * x[i] ) * x[i] - 2.0 * xim1 + 1.0;
      f += t * t;
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );
    for ( int i = 0; i < N; ++i )
    {
      T xim1 = ( i == 0 ) ? 0.0 : x[i - 1];
      T t    = ( 3.0 - 2.0 * x[i] ) * x[i] - 2.0 * xim1 + 1.0;

      T dt_dx = 3.0 - 4.0 * x[i];
      g[i] += 2.0 * t * dt_dx;

      if ( i > 0 ) g[i - 1] += 2.0 * t * ( -2.0 );
    }
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );
    H.setZero();

    for ( int i = 0; i < N; ++i )
    {
      T xim1 = ( i == 0 ) ? T( 0 ) : x[i - 1];
      T t    = ( 3.0 - 2.0 * x[i] ) * x[i] - 2.0 * xim1 + 1.0;

      T dt_dx   = 3.0 - 4.0 * x[i];
      T d2t_dx2 = -4.0;

      // diagonale i,i
      H.coeffRef( i, i ) += 2.0 * ( dt_dx * dt_dx + t * d2t_dx2 );

      if ( i > 0 )
      {
        // termini fuori diagonale
        T v = 2.0 * dt_dx * ( -2.0 );
        H.coeffRef( i, i - 1 ) += v;
        H.coeffRef( i - 1, i ) += v;

        // diagonale (i-1,i-1)
        H.coeffRef( i - 1, i - 1 ) += 2.0 * 4.0;
      }
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Brown Almost Linear Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class BrownAlmostLinearN
 * @brief Brown's almost linear function, a scalable test function
 *
 * This function has a global minimum when the product of all variables
 * equals 1. The function is designed to test algorithms on problems with nearly
 * linear constraints.
 */
template <typename T, int N> class BrownAlmostLinearN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{brown1969,\n"
           "  title={A note on the convergence of Powell's method},\n"
           "  author={Brown, K. M.},\n"
           "  journal={IMA Journal of Applied Mathematics},\n"
           "  year={1969}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 2, "BrownAlmostLinearN requires N >= 2" );

  Vector lower() const override { return Vector::Constant( N, -5.0 ); }
  Vector upper() const override { return Vector::Constant( N, 5.0 ); }
  Vector init() const override { return Vector::Constant( N, 0.5 ); }
  Vector exact() const override { return Vector::Constant( N, 1.0 ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = 0.0;
    for ( int i = 0; i < N - 1; ++i )
    {
      T t = x[i] + x[i + 1] * x[i + 1] * x[i + 1] - 3.0;
      f += t * t;
    }
    for ( int i = 0; i < N; ++i ) f += 1e-3 * x[i] * x[i];
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );
    for ( int i = 0; i < N - 1; ++i )
    {
      T t = x[i] + x[i + 1] * x[i + 1] * x[i + 1] - 3.0;
      g[i] += 2.0 * t;
      g[i + 1] += 2.0 * t * 3.0 * x[i + 1] * x[i + 1];
    }
    for ( int i = 0; i < N; ++i ) g[i] += 2e-3 * x[i];
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );
    H.setZero();

    for ( int i = 0; i < N - 1; ++i )
    {
      T xi1 = x[i + 1];
      T t   = x[i] + xi1 * xi1 * xi1 - 3.0;

      // d²/dx_i² (t^2)
      H.coeffRef( i, i ) += 2.0;

      // d²/dx_{i+1}² (t^2)
      // = 2 * [ (3 x^2)^2 + t * (6 x) ]
      H.coeffRef( i + 1, i + 1 ) += 2.0 * ( 9.0 * xi1 * xi1 * xi1 * xi1 + 6.0 * t * xi1 );

      // termini misti
      T v = 2.0 * 3.0 * xi1 * xi1;
      H.coeffRef( i, i + 1 ) += v;
      H.coeffRef( i + 1, i ) += v;
    }

    // regolarizzazione
    for ( int i = 0; i < N; ++i ) H.coeffRef( i, i ) += 2e-3;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Bukin Function N.6 (2D)
// -------------------------------------------------------------------
/**
 * @class Bukin6
 * @brief Bukin Function N.6, a difficult minimization problem with a ridge.
 *
 * f(x, y) = 100 * sqrt(abs(y - 0.01*x^2)) + 0.01 * abs(x + 10)
 *
 * The minimum lies in a valley shaped like a parabola y = 0.01x^2.
 * The function is non-differentiable along this valley and at x = -10.
 *
 * Global Minimum: f(-10, 1) = 0
 * Domain: Usually x in [-15, -5], y in [-3, 3]
 */
template <typename T> class Bukin6 : public NDbase<T>
{
  T sgn( T val ) const { return ( T( 0 ) < val ) - ( val < T( 0 ) ); }

public:
  std::string bibtex() const override
  {
    return "@article{bukin1997,\n"
           "  title={New test functions for global optimization},\n"
           "  author={Bukin, A. D.},\n"
           "  journal={Journal of Optimization},\n"
           "  year={1997}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  // Dominio tipico asimmetrico
  Vector lower() const override
  {
    Vector l( 2 );
    l << -15.0, -3.0;
    return l;
  }
  Vector upper() const override
  {
    Vector u( 2 );
    u << -5.0, 3.0;
    return u;
  }

  // Punto iniziale (lontano dal minimo e dalla valle singolare)
  Vector init() const override
  {
    Vector x0( 2 );
    x0 << -12.0, 2.0;
    return x0;
  }

  // Minimo globale
  Vector exact() const override
  {
    Vector x( 2 );
    x << -10.0, 1.0;
    return x;
  }

  // -----------------------------------------------------------
  // Objective Function
  // u = y - 0.01 * x^2
  // v = x + 10
  // f = 100 * sqrt(|u|) + 0.01 * |v|
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0];
    T x2 = x[1];

    T u = x2 - 0.01 * x1 * x1;
    T v = x1 + 10.0;

    return 100.0 * std::sqrt( std::abs( u ) ) + 0.01 * std::abs( v );
  }

  // -----------------------------------------------------------
  // Gradient Analitico CORRETTO
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0];
    T x2 = x[1];

    T u = x2 - 0.01 * x1 * x1;
    T v = x1 + 10.0;

    T abs_u  = std::abs( u );
    T sqrt_u = std::sqrt( abs_u );

    Vector g( 2 );

    // Derivata rispetto a x1
    T d_term1_dx1 = 0.0;
    if ( sqrt_u > std::numeric_limits<T>::epsilon() )
    {
      // d(100*sqrt|u|)/dx1 = 100 * 0.5 * sgn(u)/sqrt|u| * (-0.02*x1)
      //                    = -x1 * sgn(u)/sqrt|u|
      d_term1_dx1 = -x1 * sgn( u ) / sqrt_u;
    }

    T d_term2_dx1 = 0.01 * sgn( v );

    g[0] = d_term1_dx1 + d_term2_dx1;

    // Derivata rispetto a x2
    if ( sqrt_u > std::numeric_limits<T>::epsilon() )
    {
      // d(100*sqrt|u|)/dx2 = 100 * 0.5 * sgn(u)/sqrt|u| * 1
      //                    = 50 * sgn(u)/sqrt|u|
      g[1] = 50.0 * sgn( u ) / sqrt_u;
    }
    else
    {
      g[1] = 0.0;
    }

    return g;
  }

  // -----------------------------------------------------------
  // Hessiana Analitica CORRETTA
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0];
    T x2 = x[1];

    T u      = x2 - 0.01 * x1 * x1;
    T abs_u  = std::abs( u );
    T sqrt_u = std::sqrt( abs_u );

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    // Per proteggere dalla singolarità
    if ( abs_u < std::numeric_limits<T>::epsilon() )
    {
      // Vicino a u=0, usiamo una approssimazione stabile
      // In pratica, l'hessiana è indefinita qui, ma per metodi numerici
      // possiamo usare una matrice di protezione
      H.insert( 0, 0 ) = 1.0;
      H.insert( 1, 1 ) = 1.0;
      H.insert( 0, 1 ) = 0.0;
      H.insert( 1, 0 ) = 0.0;
    }
    else
    {
      // Calcoli esatti per u ≠ 0
      T sgn_u = sgn( u );

      // Termini per la parte sqrt(|u|)
      T term1 = 50.0 * sgn_u / sqrt_u;       // phi'(u)
      T term2 = -25.0 / ( abs_u * sqrt_u );  // phi''(u)

      // Derivate di u
      T du_dx1   = -0.02 * x1;
      T du_dx2   = 1.0;
      T d2u_dx12 = -0.02;

      // Elementi dell'hessiana
      // H[0,0] = phi''(u) * (du/dx1)^2 + phi'(u) * d²u/dx1²
      T H_xx = term2 * du_dx1 * du_dx1 + term1 * d2u_dx12;

      // H[1,1] = phi''(u) * (du/dx2)^2
      T H_yy = term2 * du_dx2 * du_dx2;

      // H[0,1] = H[1,0] = phi''(u) * (du/dx1) * (du/dx2)
      T H_xy = term2 * du_dx1 * du_dx2;

      // Il termine 0.01*|x+10| ha hessiana zero quasi ovunque
      // (la distribuzione di Dirac a x=-10 non è catturata qui)

      H.insert( 0, 0 ) = H_xx;
      H.insert( 1, 1 ) = H_yy;
      H.insert( 0, 1 ) = H_xy;
      H.insert( 1, 0 ) = H_xy;
    }

    H.makeCompressed();
    return H;
  }

  // -----------------------------------------------------------
  // METODO AGGIUNTIVO: Gradiente con differenze finite per verifica
  // -----------------------------------------------------------
  Vector gradient_fd( Vector const & x, T eps = 1e-7 ) const
  {
    Vector g( 2 );
    Vector x_plus = x, x_minus = x;

    for ( int i = 0; i < 2; ++i )
    {
      x_plus[i]  = x[i] + eps;
      x_minus[i] = x[i] - eps;
      g[i]       = ( operator()( x_plus ) - operator()( x_minus ) ) / ( 2 * eps );
      x_plus[i]  = x[i];
      x_minus[i] = x[i];
    }

    return g;
  }
};

// -------------------------------------------------------------------
// Cross-in-Tray Function (2D)
// -------------------------------------------------------------------

/**
 * @class CrossInTray2D
 * @brief Cross-in-Tray function, a 2D optimization test function
 *
 * The Cross-in-Tray function has multiple global minima.
 * It features a cross-like shape with steep ridges and valleys.
 * Global Minima: f(x*) = -2.06261 at (+/- 1.34941, +/- 1.34941)
 * Domain: Usually x_i in [-10, 10]
 */
template <typename T> class CrossInTray2D : public NDbase<T>
{
  T sgn( T val ) const { return ( T( 0 ) < val ) - ( val < T( 0 ) ); }

public:
  std::string bibtex() const override
  {
    return "@article{crossintray,\n"
           "  title={Test functions for optimization needs},\n"
           "  author={Molga, M. and Smutnicki, C.},\n"
           "  journal={Test Functions for Optimization},\n"
           "  year={2005}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  // Dominio tipico [-10, 10]
  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }

  // Punto iniziale (lontano dall'ottimo per testare la convergenza)
  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 2.0, 2.0;
    return x0;
  }

  // Uno dei 4 minimi globali esatti
  Vector exact() const override
  {
    Vector x( 2 );
    x << 1.34941, 1.34941;
    return x;
  }

  /// Objective function
  /// f(x) = -0.0001 * ( |sin(x)sin(y)exp(|100 - sqrt(x^2+y^2)/pi|)| + 1 )^0.1
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T pi = Utils::m_pi;

    T R        = std::sqrt( x1 * x1 + x2 * x2 );
    T arg_exp  = std::abs( 100.0 - R / pi );
    T term_exp = std::exp( arg_exp );
    T term_sin = std::sin( x1 ) * std::sin( x2 );

    T inner = std::abs( term_sin * term_exp ) + 1.0;

    return -0.0001 * std::pow( inner, 0.1 );
  }

  /// Gradient ∇f(x)
  /// Calcolato analiticamente usando la regola della catena e la funzione sgn()
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T pi  = Utils::m_pi;
    T eps = 1e-8;  // Per evitare divisione per zero a (0,0)

    // Termini intermedi
    T R = std::sqrt( x1 * x1 + x2 * x2 );
    if ( R < eps ) R = eps;  // Protezione singolarità

    T arg_inside_abs = 100.0 - R / pi;
    T sign_arg       = sgn( arg_inside_abs );  // sgn(100 - R/pi)

    T E_val = std::exp( std::abs( arg_inside_abs ) );
    T S_val = std::sin( x1 ) * std::sin( x2 );
    T prod  = S_val * E_val;

    T sign_prod         = sgn( prod );  // sgn(sin(x)sin(y)exp(...))
    T abs_prod_plus_one = std::abs( prod ) + 1.0;

    // Prefattore comune derivato da: d/dx [ -0.0001 * (u)^0.1 ] -> -0.00001 * u^-0.9 * du/dx
    T common_factor = -0.00001 * std::pow( abs_prod_plus_one, -0.9 ) * sign_prod * E_val;

    // Derivate parziali dell'argomento dell'esponenziale: d/dx(|100 - R/pi|)
    // = sgn(100 - R/pi) * (-1/pi) * (x / R)
    T d_abs_arg_dx1 = sign_arg * ( -1.0 / pi ) * ( x1 / R );
    T d_abs_arg_dx2 = sign_arg * ( -1.0 / pi ) * ( x2 / R );

    // Costruzione Gradiente
    Vector g( 2 );

    // df/dx1
    // Chain rule: sin(x2)*[ cos(x1) + sin(x1)*d_abs_arg_dx1 ]
    T term_x1 = std::cos( x1 ) * std::sin( x2 ) + S_val * d_abs_arg_dx1;
    g[0]      = common_factor * term_x1;

    // df/dx2
    // Chain rule: sin(x1)*[ cos(x2) + sin(x2)*d_abs_arg_dx2 ]
    T term_x2 = std::sin( x1 ) * std::cos( x2 ) + S_val * d_abs_arg_dx2;
    g[1]      = common_factor * term_x2;

    return g;
  }

  /// Hessian ∇²f(x) sparse 2x2
  /// IMPLEMENTAZIONE NUMERICA (Finite Differences)
  /// La derivata analitica seconda è troppo complessa e discontinua per un'implementazione robusta diretta.
  SparseMatrix hessian( Vector const & x ) const override
  {
    T            h = 1e-5;  // Passo per differenze finite
    auto         n = x.size();
    SparseMatrix H( n, n );

    Vector g_plus, g_minus;
    Vector x_temp = x;

    // Calcolo colonne dell'Hessiana usando differenze centrali sul gradiente
    // H_col_j = ( grad(x + h*ej) - grad(x - h*ej) ) / (2*h)
    for ( auto j = 0; j < n; ++j )
    {
      T original_xj = x[j];

      // x + h
      x_temp[j] = original_xj + h;
      g_plus    = this->gradient( x_temp );

      // x - h
      x_temp[j] = original_xj - h;
      g_minus   = this->gradient( x_temp );

      // Ripristino
      x_temp[j] = original_xj;

      // Approssimazione colonna j
      Vector col_j = ( g_plus - g_minus ) / ( 2.0 * h );

      for ( int i = 0; i < n; ++i )
      {
        // Inserisce solo se significativamente diverso da zero (o sempre per matrici dense piccole)
        H.insert( i, j ) = col_j[i];
      }
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Deceptive Function (CEC / Hedar)
// -------------------------------------------------------------------

/**
 * @class Deceptive
 * @brief Deceptive Function, designed to mislead optimization algorithms.
 *
 * This function is constructed with a large basin of attraction leading to a
 * local minimum (the deceptive trap) and a small basin leading to the global minimum.
 *
 * It is typically separable and defined as:
 * f(x) = - [ (1/D) * Sum( g(x_i) ) ]^beta
 *
 * Where g(x) is a piecewise linear function on [0, 1]:
 * - Increases from 0 to 0.8 at x = alpha (local max)
 * - Decreases to 0 at x = beta
 * - Increases to 1.0 at x = 1 (global max)
 *
 * Parameters used: alpha = 0.3, beta_pt = 0.8 (boundary between basins), beta_exp = 1.
 * Global Minimum: f(1, 1) = -1.0
 * Domain: x_i in [0, 1]
 */
template <typename T> class Deceptive : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{cecdeceptive,\n"
           "  title={Benchmark Functions for CEC 2005},\n"
           "  author={Suganthan, P. N. et al.},\n"
           "  journal={IEEE Congress on Evolutionary Computation},\n"
           "  year={2005}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, 0.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 1.0 ); }

  // Punto iniziale: posizionato nel bacino ingannevole (vicino a 0)
  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 0.1, 0.1;
    return x0;
  }

  // Minimo globale
  Vector exact() const override
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }

  // Parametri della funzione
  const T alpha    = 0.3;  // Posizione del massimo locale (ingannevole)
  const T mid_pt   = 0.8;  // Punto di minimo locale tra i picchi (confine bacino)
  const T val_loc  = 0.8;  // Valore del picco locale
  const T val_glo  = 1.0;  // Valore del picco globale
  const T exponent = 1.0;  // Esponente beta (1 = separabile lineare)

  // -----------------------------------------------------------
  // Helper: Calcolo g(x) e g'(x) per una dimensione
  // -----------------------------------------------------------
  void get_g_vals( T xi, T & val, T & grad ) const
  {
    // 3 Regioni:
    // 1. [0, alpha] -> sale linearmente da 0 a 0.8
    // 2. [alpha, mid_pt] -> scende linearmente da 0.8 a 0
    // 3. [mid_pt, 1] -> sale linearmente da 0 a 1 (ripido)

    if ( xi < alpha )
    {
      T slope = val_loc / alpha;  // 0.8 / 0.3 = 2.66...
      val     = slope * xi;
      grad    = slope;
    }
    else if ( xi < mid_pt )
    {
      // Linea da (0.3, 0.8) a (0.8, 0.0)
      T slope = ( 0.0 - val_loc ) / ( mid_pt - alpha );  // -0.8 / 0.5 = -1.6
      // y - 0 = m * (x - 0.8)
      val  = slope * ( xi - mid_pt );
      grad = slope;
    }
    else
    {
      // Linea da (0.8, 0.0) a (1.0, 1.0)
      T slope = ( val_glo - 0.0 ) / ( 1.0 - mid_pt );  // 1.0 / 0.2 = 5.0
      val     = slope * ( xi - mid_pt );
      grad    = slope;
    }
  }

  // -----------------------------------------------------------
  // Objective Function
  // f(x) = - ( avg( g(xi) ) )^beta
  // Qui beta=1, quindi f(x) = - sum(g(xi)) / D
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T sum_g = 0.0;
    T dummy_g;

    for ( int i = 0; i < x.size(); ++i )
    {
      T g_val;
      get_g_vals( x[i], g_val, dummy_g );
      sum_g += g_val;
    }

    // Se beta != 1: return -std::pow(sum_g / x.size(), exponent);
    return -( sum_g / T( x.size() ) );
  }

  // -----------------------------------------------------------
  // Gradient Analitico
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    Vector g( x.size() );
    T      D = T( x.size() );

    // df/dxi = - (1/D) * g'(xi) (se beta=1)

    for ( int i = 0; i < x.size(); ++i )
    {
      T dummy_v, g_prime;
      get_g_vals( x[i], dummy_v, g_prime );
      g[i] = -g_prime / D;
    }

    return g;
  }

  // -----------------------------------------------------------
  // Hessiana Analitica
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    // Poiché g(x) è lineare a tratti, g''(x) è 0 ovunque (tranne nei kink).
    // Restituiamo una matrice vuota (tutti zeri).
    // NOTA: Algoritmi di Newton puro falliranno (matrice singolare).
    // Si consiglia di aggiungere una regolarizzazione (es. mu * I) nel solver.

    SparseMatrix H( x.size(), x.size() );
    H.setZero();  // Curvatura nulla sui piani

    Eigen::Index n = x.size();
    for ( Eigen::Index i = 0; i < n; ++i ) H.insert( i, i ) = 0;

    // Per stabilità numerica nei solver che richiedono diagonale non nulla:
    // Potremmo inserire un epsilon piccolissimo, ma matematicamente è 0.
    H.makeCompressed();
    return H;
  }
};
// -------------------------------------------------------------------
// -------------------------------------------------------------------

/**
 * @class DixonPriceN
 * @brief Dixon-Price function, a scalable test function for optimization.
 *
 * The Dixon-Price function is defined as:
 *
 * f(x) = (x₁ - 1)² + Σ_{i=2}^{n} i (2x_i² - x_{i-1})²
 *
 * The function is unimodal and has a global minimum at:
 * x_i = 2^{-(2^i - 2)/2^i} for i = 1,...,n
 * In particular, when n=2, the minimum is at x = [1, 1/√2] ≈ [1, 0.7071]
 * For the standard formulation, the minimum is at x_i = 1 for all i.
 *
 * Reference: Dixon, L. C. W., & Szegö, G. P. (1978). Nonlinear Optimization.
 *
 * @tparam T Floating point type (float, double, etc.)
 * @tparam N Dimension of the problem (must be > 0)
 */
template <typename T, int N> class DixonPriceN : public NDbase<T>
{
  static_assert( N > 0, "Dimension N must be positive" );

public:
  /// @brief Returns the BibTeX reference for the function
  std::string bibtex() const override
  {
    return "@book{dixon1978,\n"
           "  title={Nonlinear Optimization},\n"
           "  author={Dixon, L. C. W. and Szegö, G. P.},\n"
           "  publisher={Academic Press},\n"
           "  year={1978}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  /// @brief Returns the lower bounds of the search domain
  /// @return Vector of lower bounds, typically -10 for all dimensions
  Vector lower() const override { return Vector::Constant( N, -10.0 ); }

  /// @brief Returns the upper bounds of the search domain
  /// @return Vector of upper bounds, typically 10 for all dimensions
  Vector upper() const override { return Vector::Constant( N, 10.0 ); }

  /// @brief Returns a suggested starting point for optimization
  /// @return Initial point, typically [1, 1, ..., 1]
  Vector init() const override { return Vector::Constant( N, 1.0 ); }

  /// @brief Returns the exact global minimum point
  /// @note For the standard formulation used here, the minimum is at x_i = 1
  Vector exact() const override
  {
    // For the formulation: (x₁ - 1)² + Σ i(2x_i² - x_{i-1})²
    // The global minimum is at x_i = 1 for all i
    return Vector::Ones( N );
  }

  /// @brief Evaluates the Dixon-Price function at point x
  /// @param x Input vector of dimension N
  /// @return Function value f(x)
  T operator()( Vector const & x ) const override
  {
    T f = ( x[0] - 1 ) * ( x[0] - 1 );
    for ( int i = 1; i < N; ++i )
    {
      T term = 2 * x[i] * x[i] - x[i - 1];
      f += ( i + 1 ) * term * term;
    }

    T f_min = static_cast<T>( N * ( N + 1 ) / 2 - 1 );
    return f - f_min;
  }

  /// @brief Computes the analytical gradient of the Dixon-Price function
  ///
  /// The gradient components are:
  /// ∂f/∂x₁ = 2(x₁ - 1) - 2(2x₂² - x₁)
  /// ∂f/∂x_i = 4(i+1)x_i(2x_i² - x_{i-1}) - 2(i+2)(2x_{i+1}² - x_i) for i=2..N-1
  /// ∂f/∂x_N = 4(N+1)x_N(2x_N² - x_{N-1})
  ///
  /// @param x Input vector of dimension N
  /// @return Gradient vector ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );

    // First component (i = 0)
    // f contains (x₀ - 1)² and the term from i=1: 2(2x₁² - x₀)²
    // ∂/∂x₀ of (x₀ - 1)² = 2(x₀ - 1)
    // ∂/∂x₀ of 2(2x₁² - x₀)² = 2 * 2(2x₁² - x₀) * (-1) = -4(2x₁² - x₀)
    g[0] = 2 * ( x[0] - 1 ) - 4 * ( 2 * x[1] * x[1] - x[0] );

    // Middle components (i = 1 to N-2)
    for ( int i = 1; i < N - 1; ++i )
    {
      // Term from i: (i+1)(2x_i² - x_{i-1})²
      // ∂/∂x_i = 2(i+1)(2x_i² - x_{i-1}) * 4x_i = 8(i+1)x_i(2x_i² - x_{i-1})

      // Term from i+1: (i+2)(2x_{i+1}² - x_i)²
      // ∂/∂x_i = 2(i+2)(2x_{i+1}² - x_i) * (-1) = -2(i+2)(2x_{i+1}² - x_i)

      g[i] += 8 * ( i + 1 ) * x[i] * ( 2 * x[i] * x[i] - x[i - 1] );
      g[i] -= 2 * ( i + 2 ) * ( 2 * x[i + 1] * x[i + 1] - x[i] );
    }

    // Last component (i = N-1)
    if ( N > 1 )
    {
      // Only has term from i = N-1: N(2x_{N-1}² - x_{N-2})²
      // ∂/∂x_{N-1} = 2N(2x_{N-1}² - x_{N-2}) * 4x_{N-1} = 8N x_{N-1}(2x_{N-1}² - x_{N-2})
      int i = N - 1;
      g[i]  = 8 * ( i + 1 ) * x[i] * ( 2 * x[i] * x[i] - x[i - 1] );
    }

    return g;
  }

  /// @brief Computes the analytical Hessian of the Dixon-Price function
  ///
  /// The Hessian is tridiagonal with elements:
  /// H[0,0] = 2 + 4
  /// H[i,i] = 8(i+1)(6x_i² - x_{i-1}) + 2(i+2) for i=1..N-2
  /// H[N-1,N-1] = 8N(6x_{N-1}² - x_{N-2})
  /// H[i,i-1] = H[i-1,i] = -8(i+1)x_i for i=1..N-1
  /// H[i,i+1] = H[i+1,i] = -4(i+2)x_{i+1} for i=0..N-2
  ///
  /// @param x Input vector of dimension N
  /// @return Hessian matrix ∇²f(x) in sparse format
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );

    // Reserve space for tridiagonal structure (3 diagonals)
    H.reserve( Eigen::VectorXi::Constant( N, 3 ) );

    // Main diagonal
    H.insert( 0, 0 ) = 2 + 4;  // = 6

    // Off-diagonal elements (only insert once per unique pair)
    if ( N > 1 )
    {
      H.insert( 0, 1 ) = -16 * x[1];
      // H.insert( 1, 0 ) = -16 * x[1];

      // Middle diagonals
      for ( int i = 1; i < N - 1; ++i )
      {
        // Diagonal
        T diag           = 8 * ( i + 1 ) * ( 6 * x[i] * x[i] - x[i - 1] ) + 2 * ( i + 2 );
        H.insert( i, i ) = diag;

        // Sub-diagonal (already inserted when processing i-1)
        // So we only need to insert for current i to i-1
        H.insert( i, i - 1 ) = -8 * ( i + 1 ) * x[i];

        // Super-diagonal
        H.insert( i, i + 1 ) = -8 * ( i + 2 ) * x[i + 1];
      }

      // Last row
      int i                = N - 1;
      T   diag             = 8 * ( i + 1 ) * ( 6 * x[i] * x[i] - x[i - 1] );
      H.insert( i, i )     = diag;
      H.insert( i, i - 1 ) = -8 * ( i + 1 ) * x[i];
    }

    // Note: Eigen will automatically fill the symmetric entries if we make the matrix symmetric
    // But we need to explicitly set them or use a symmetric storage format
    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Drop-Wave Function (2D)
// -------------------------------------------------------------------

/**
 * @class DropWave2D
 * @brief Drop-Wave function, a highly multimodal optimization test function.
 *
 * The function looks like ripples created by a drop falling into water.
 * It is radially symmetric.
 *
 * f(x) = - (1 + cos(12 * ||x||)) / (0.5 * ||x||^2 + 2)
 *
 * Global Minimum: f(0,0) = -1.0
 * Domain: Usually x_i in [-5.12, 5.12]
 */
template <typename T> class DropWave2D : public NDbase<T>
{
  static constexpr T m_eps    = 1e-4;
  static constexpr T m_eps_sq = m_eps * m_eps;

private:
  // -----------------------------------------------------------
  // Helper functions for Taylor expansions near origin
  // -----------------------------------------------------------

  // Taylor expansion for sin(12R)/R for small R
  // sin(12R)/R = 12 - (1728/6)R^2 + (248832/120)R^4 - (35831808/5040)R^6 + ...
  //            = 12 - 288R^2 + (20736/5)R^4 - (248832/35)R^6 + O(R^8)
  static T sin12R_over_R_expansion( T r2 )
  {
    const T R2 = r2;
    const T R4 = R2 * R2;
    const T R6 = R4 * R2;
    return 12.0 - 288.0 * R2 + ( 20736.0 / 5.0 ) * R4 - ( 248832.0 / 35.0 ) * R6;
  }

  // Taylor expansion for cos(12R) for small R
  // cos(12R) = 1 - (144/2)R^2 + (20736/24)R^4 - (2985984/720)R^6 + ...
  //          = 1 - 72R^2 + 864R^4 - (124416/35)R^6 + O(R^8)
  static T cos12R_expansion( T r2 )
  {
    const T R2 = r2;
    const T R4 = R2 * R2;
    const T R6 = R4 * R2;
    return 1.0 - 72.0 * R2 + 864.0 * R4 - ( 124416.0 / 35.0 ) * R6;
  }

  // Taylor expansion for d/dR[sin(12R)/R] for small R
  // d/dR[sin(12R)/R] = -576R + (82944/5)R^3 - (1492992/35)R^5 + O(R^7)
  static T d_sin12R_over_R_expansion( T r2 )
  {
    const T R2 = r2;
    const T R4 = R2 * R2;
    const T R  = std::sqrt( r2 );
    return R * ( -576.0 + ( 82944.0 / 5.0 ) * R2 - ( 1492992.0 / 35.0 ) * R4 );
  }

  // Taylor expansion for d²/dR²[sin(12R)/R] for small R
  // d²/dR²[sin(12R)/R] = -576 + (248832/5)R^2 - (7464960/35)R^4 + O(R^6)
  static T d2_sin12R_over_R_expansion( T r2 )
  {
    const T R2 = r2;
    const T R4 = R2 * R2;
    return -576.0 + ( 248832.0 / 5.0 ) * R2 - ( 7464960.0 / 35.0 ) * R4;
  }

  // Taylor expansion for sin(12R) for small R
  // sin(12R) = 12R - 288R^3 + (20736/5)R^5 - (248832/35)R^7 + O(R^9)
  static T sin12R_expansion( T r2 )
  {
    const T R  = std::sqrt( r2 );
    const T R2 = r2;
    const T R3 = R * R2;
    const T R5 = R3 * R2;
    const T R7 = R5 * R2;
    return 12.0 * R - 288.0 * R3 + ( 20736.0 / 5.0 ) * R5 - ( 248832.0 / 35.0 ) * R7;
  }

public:
  std::string bibtex() const override
  {
    return "@article{dropwave,\n"
           "  title={Global optimization test problems},\n"
           "  author={Hedar, A. R.},\n"
           "  journal={IEEE CEC},\n"
           "  year={2005}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -5.12 ); }
  Vector upper() const override { return Vector::Constant( 2, 5.12 ); }

  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 1.0, 1.0;
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }

  // -----------------------------------------------------------
  // Objective Function
  // f(x) = -(1 + cos(12R)) / (0.5R² + 2)
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T r2 = x.squaredNorm();

    // Use Taylor expansion near origin for numerical stability
    if ( r2 < m_eps_sq )
    {
      T cos12R = cos12R_expansion( r2 );
      T num    = 1.0 + cos12R;
      T den    = 0.5 * r2 + 2.0;
      return -num / den;
    }

    T R   = std::sqrt( r2 );
    T arg = 12.0 * R;
    T num = 1.0 + std::cos( arg );
    T den = 0.5 * r2 + 2.0;

    return -num / den;
  }

  // -----------------------------------------------------------
  // Gradient
  // ∇f = h'(R) * (x/R) = [(N + 12·D·sin(12R)/R] / D² * x
  // where N = 1 + cos(12R), D = 0.5R² + 2
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    T r2 = x.squaredNorm();

    // Near origin: use Taylor expansions for stability
    if ( r2 < m_eps_sq )
    {
      T cos12R        = cos12R_expansion( r2 );
      T sin12R_over_R = sin12R_over_R_expansion( r2 );

      T N  = 1.0 + cos12R;
      T D  = 0.5 * r2 + 2.0;
      T D2 = D * D;

      T factor = ( N + 12.0 * D * sin12R_over_R ) / D2;
      return x * factor;
    }

    // Regular computation away from origin
    T R   = std::sqrt( r2 );
    T arg = 12.0 * R;

    T cos_arg        = std::cos( arg );
    T sin_arg_over_R = std::sin( arg ) / R;

    T N  = 1.0 + cos_arg;
    T D  = 0.5 * r2 + 2.0;
    T D2 = D * D;

    T factor = ( N + 12.0 * D * sin_arg_over_R ) / D2;
    return x * factor;
  }

  // -----------------------------------------------------------
  // Hessian
  // ∇²f = [(h''(R) - h'(R)/R)/R²] · xx^T + [h'(R)/R] · I
  // where h'(R) and h''(R) are computed using expansions near origin
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    T r2 = x.squaredNorm();

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    // Near origin: use Taylor expansions for all quantities
    if ( r2 < m_eps_sq )
    {
      // Compute all necessary Taylor expansions
      T cos12R        = cos12R_expansion( r2 );
      T sin12R_over_R = sin12R_over_R_expansion( r2 );
      // T d_sin12R_over_R = d_sin12R_over_R_expansion(r2);
      // T d2_sin12R_over_R = d2_sin12R_over_R_expansion(r2);

      // Compute N and its derivatives
      T N       = 1.0 + cos12R;
      T N_prime = -12.0 * sin12R_expansion( r2 );  // dN/dR = -12 sin(12R)

      // For small R, we need N'' = d²N/dR² = -144 cos(12R)
      // Use cos12R expansion already computed
      T N_double = -144.0 * cos12R;

      // Compute D and its derivatives
      T D        = 0.5 * r2 + 2.0;
      T D_prime  = std::sqrt( r2 );  // R = sqrt(r2)
      T D_double = 1.0;

      T D2 = D * D;
      T D3 = D2 * D;

      // Compute h'(R) = [N·R + 12·D·sin(12R)] / D²
      T R           = D_prime;
      T sin12R      = sin12R_over_R * R;
      T h_prime_num = N * R + 12.0 * D * sin12R;
      T h_prime     = h_prime_num / D2;

      // Alternative computation of h''(R) using product rule
      // h(R) = -N/D
      // h'(R) = -(N'·D - N·D')/D²
      // h''(R) = -[(N''·D - N·D'')·D - 2·D'·(N'·D - N·D')]/D³
      T term1        = N_double * D - N * D_double;
      T term2        = N_prime * D - N * D_prime;
      T h_double_num = -( term1 * D - 2.0 * D_prime * term2 );
      T h_double     = h_double_num / D3;

      // Hessian components
      T h_prime_over_R = h_prime / ( R + std::numeric_limits<T>::epsilon() );
      T term_a         = ( h_double - h_prime_over_R ) / ( r2 + std::numeric_limits<T>::epsilon() );
      T term_b         = h_prime_over_R;

      // Populate Hessian
      T x0 = x[0];
      T x1 = x[1];

      H.insert( 0, 0 ) = term_a * x0 * x0 + term_b;
      H.insert( 1, 1 ) = term_a * x1 * x1 + term_b;
      H.insert( 0, 1 ) = term_a * x0 * x1;
      H.insert( 1, 0 ) = term_a * x0 * x1;

      H.makeCompressed();
      return H;
    }

    // Regular computation away from origin
    T R   = std::sqrt( r2 );
    T arg = 12.0 * R;

    T cos_arg = std::cos( arg );
    T sin_arg = std::sin( arg );

    T N  = 1.0 + cos_arg;
    T D  = 0.5 * r2 + 2.0;
    T D2 = D * D;
    T D3 = D2 * D;

    T N_prime = -12.0 * sin_arg;
    T D_prime = R;

    T h_prime_num = N * R + 12.0 * D * sin_arg;
    T h_prime     = h_prime_num / D2;

    T N_double = -144.0 * cos_arg;
    T D_double = 1.0;

    T term1        = N_double * D - N * D_double;
    T term2        = N_prime * D - N * D_prime;
    T h_double_num = -( term1 * D - 2.0 * D_prime * term2 );
    T h_double     = h_double_num / D3;

    T h_prime_over_R = h_prime / R;
    T term_a         = ( h_double - h_prime_over_R ) / r2;
    T term_b         = h_prime_over_R;

    T x0    = x[0];
    T x1    = x[1];
    T x0_x1 = x0 * x1;

    H.insert( 0, 0 ) = term_a * x0 * x0 + term_b;
    H.insert( 1, 1 ) = term_a * x1 * x1 + term_b;
    H.insert( 0, 1 ) = term_a * x0_x1;
    H.insert( 1, 0 ) = term_a * x0_x1;

    H.makeCompressed();
    return H;
  }
};


// -------------------------------------------------------------------
// Eggholder Function (2D)
// -------------------------------------------------------------------

/**
 * @class Eggholder2D
 * @brief Eggholder function, a difficult 2D optimization test problem.
 *
 * The Eggholder function is a difficult function to optimize because of the
 * large number of local minima.
 * Global Minimum: f(x*) = -959.6407 at (512, 404.2319)
 * Domain: Usually x_i in [-512, 512]
 */
template <typename T> class Eggholder2D : public NDbase<T>
{
  T sgn( T val ) const { return ( T( 0 ) < val ) - ( val < T( 0 ) ); }

public:
  std::string bibtex() const override
  {
    return "@article{eggholder,\n"
           "  title={Global optimization test problems},\n"
           "  author={Mishra, S. K.},\n"
           "  journal={Some Aspects of Global Optimization},\n"
           "  year={2006}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -512.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 512.0 ); }

  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 0.0, 0.0;
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 512.0, 404.2319;
    return x;
  }

  // -----------------------------------------------------------
  // Objective Function
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0];
    T x2 = x[1];
    T c  = x2 + 47.0;  // Termine comune (y+47)

    T argA = 0.5 * x1 + c;
    T argB = x1 - c;

    return -c * std::sin( std::sqrt( std::abs( argA ) ) ) - x1 * std::sin( std::sqrt( std::abs( argB ) ) );
  }

  // -----------------------------------------------------------
  // Gradient Analitico
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    T x1  = x[0];
    T x2  = x[1];
    T c   = x2 + 47.0;
    T eps = 1e-10;

    // Argomenti interni ai valori assoluti
    T u = 0.5 * x1 + c;
    T v = x1 - c;

    T abs_u = std::abs( u );
    T abs_v = std::abs( v );

    T sq_u = std::sqrt( abs_u );
    T sq_v = std::sqrt( abs_v );

    // Derivate prime delle radici rispetto agli argomenti interni u, v
    // d(sqrt(|k|))/dk = sgn(k) / (2 * sqrt(|k|))
    T d_sq_u = ( sq_u > eps ) ? ( 0.5 * sgn( u ) / sq_u ) : 0.0;
    T d_sq_v = ( sq_v > eps ) ? ( 0.5 * sgn( v ) / sq_v ) : 0.0;

    // Termini trigonometrici
    T S_u = std::sin( sq_u );
    T C_u = std::cos( sq_u );
    T S_v = std::sin( sq_v );
    T C_v = std::cos( sq_v );

    Vector g( 2 );

    // 1. Derivate parziali di sqrt(|u|) e sqrt(|v|) rispetto a x1 e x2
    // u = 0.5*x + y + 47  => du/dx = 0.5, du/dy = 1
    // v = x - y - 47      => dv/dx = 1,   dv/dy = -1
    T du_dx = 0.5;
    T du_dy = 1.0;
    T dv_dx = 1.0;
    T dv_dy = -1.0;

    T d_sq_u_dx = d_sq_u * du_dx;
    T d_sq_u_dy = d_sq_u * du_dy;
    T d_sq_v_dx = d_sq_v * dv_dx;
    T d_sq_v_dy = d_sq_v * dv_dy;

    // 2. Costruzione del gradiente
    // f = -c * S_u - x1 * S_v

    // df/dx = -c * C_u * (d_sq_u_dx) - [ 1 * S_v + x1 * C_v * (d_sq_v_dx) ]
    g[0] = -c * C_u * d_sq_u_dx - S_v - x1 * C_v * d_sq_v_dx;

    // df/dy = - [ 1 * S_u + c * C_u * (d_sq_u_dy) ] - x1 * C_v * (d_sq_v_dy)
    // Nota: d(c)/dy = 1
    g[1] = -S_u - c * C_u * d_sq_u_dy - x1 * C_v * d_sq_v_dy;

    return g;
  }

  // -----------------------------------------------------------
  // Hessiana Analitica
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1  = x[0];
    T x2  = x[1];
    T c   = x2 + 47.0;
    T eps = 1e-10;

    // --- 1. Calcolo preliminare derivate di sqrt(|u|) e sqrt(|v|) ---
    T u = 0.5 * x1 + c;
    T v = x1 - c;

    T abs_u = std::abs( u );
    T abs_v = std::abs( v );
    T sq_u  = std::sqrt( abs_u );
    T sq_v  = std::sqrt( abs_v );

    T S_u = std::sin( sq_u );
    T C_u = std::cos( sq_u );
    T S_v = std::sin( sq_v );
    T C_v = std::cos( sq_v );

    // Derivate prime rispetto all'argomento interno (chain rule base)
    T phi_u_1 = ( sq_u > eps ) ? ( 0.5 * sgn( u ) / sq_u ) : 0.0;
    T phi_v_1 = ( sq_v > eps ) ? ( 0.5 * sgn( v ) / sq_v ) : 0.0;

    // Derivate seconde rispetto all'argomento interno
    // d^2/dk^2 (sqrt(|k|)) = -1 / (4 * |k|^1.5)
    T phi_u_2 = ( sq_u > eps ) ? ( -0.25 / ( abs_u * sq_u ) ) : 0.0;
    T phi_v_2 = ( sq_v > eps ) ? ( -0.25 / ( abs_v * sq_v ) ) : 0.0;

    // Gradienti di u e v (costanti)
    T ux = 0.5, uy = 1.0;
    T vx = 1.0, vy = -1.0;

    // Derivate prime composte (d_sq_u / dx, ecc.)
    T Au_x = phi_u_1 * ux;
    T Au_y = phi_u_1 * uy;
    T Bv_x = phi_v_1 * vx;
    T Bv_y = phi_v_1 * vy;

    // Derivate seconde composte (d^2_sq_u / dx^2, ecc.)
    // Poiché u e v sono lineari, non ci sono termini con derivate seconde di u o v.
    // Chain rule: d/dx (phi_u_1 * ux) = (phi_u_2 * ux) * ux
    T Au_xx = phi_u_2 * ux * ux;
    T Au_xy = phi_u_2 * ux * uy;
    T Au_yy = phi_u_2 * uy * uy;

    T Bv_xx = phi_v_2 * vx * vx;
    T Bv_xy = phi_v_2 * vx * vy;
    T Bv_yy = phi_v_2 * vy * vy;

    // --- 2. Assemblaggio Hessiana ---
    // f(x,y) = T1 + T2
    // T1 = -c * sin(sq_u)
    // T2 = -x1 * sin(sq_v)

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    // H[0,0] -> d^2f / dx^2
    // Derivata di: -c * C_u * Au_x - S_v - x1 * C_v * Bv_x
    // Parte T1: -c * [ -S_u * (Au_x)^2 + C_u * Au_xx ]
    // Parte T2: - [ C_v * Bv_x ] - [ 1 * C_v * Bv_x + x1 * ( -S_v * (Bv_x)^2 + C_v * Bv_xx ) ]
    T T1_xx          = -c * ( -S_u * Au_x * Au_x + C_u * Au_xx );
    T T2_xx          = -C_v * Bv_x - ( C_v * Bv_x + x1 * ( -S_v * Bv_x * Bv_x + C_v * Bv_xx ) );
    H.insert( 0, 0 ) = T1_xx + T2_xx;

    // H[1,1] -> d^2f / dy^2
    // Derivata di: -S_u - c * C_u * Au_y - x1 * C_v * Bv_y
    // Parte T1: - [ C_u * Au_y ] - [ 1 * C_u * Au_y + c * ( -S_u * (Au_y)^2 + C_u * Au_yy ) ]
    // Parte T2: - x1 * [ -S_v * (Bv_y)^2 + C_v * Bv_yy ]
    T T1_yy          = -C_u * Au_y - ( C_u * Au_y + c * ( -S_u * Au_y * Au_y + C_u * Au_yy ) );
    T T2_yy          = -x1 * ( -S_v * Bv_y * Bv_y + C_v * Bv_yy );
    H.insert( 1, 1 ) = T1_yy + T2_yy;

    // H[0,1] = H[1,0] -> d^2f / dx dy
    // Deriviamo df/dy rispetto a x
    // df/dy = -S_u - c * C_u * Au_y - x1 * C_v * Bv_y
    //
    // Termine 1 (-S_u): -C_u * Au_x
    // Termine 2 (-c * C_u * Au_y):
    //    d(-c)/dx * ... = 0 (perché c=y+47)
    //    -c * d(C_u * Au_y)/dx = -c * [ -S_u * Au_x * Au_y + C_u * Au_xy ]
    // Termine 3 (-x1 * C_v * Bv_y):
    //    d(-x1)/dx * ... = -1 * C_v * Bv_y
    //    -x1 * d(C_v * Bv_y)/dx = -x1 * [ -S_v * Bv_x * Bv_y + C_v * Bv_xy ]

    T term1_xy = -C_u * Au_x;
    T term2_xy = -c * ( -S_u * Au_x * Au_y + C_u * Au_xy );
    T term3_xy = -C_v * Bv_y - x1 * ( -S_v * Bv_x * Bv_y + C_v * Bv_xy );

    T H_xy = term1_xy + term2_xy + term3_xy;

    H.insert( 0, 1 ) = H_xy;
    H.insert( 1, 0 ) = H_xy;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Extended Wood Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class ExtendedWoodN
 * @brief Extended Wood function
 *
 * Scalable version of the Wood function, extended to N dimensions.
 * Composed of quartets of variables with the Wood function structure.
 * The global minimum is at (1,1,...,1) with value 0.
 */
template <typename T, int N> class ExtendedWoodN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{wood1968,\n"
           "  title={An algorithm for solving nonlinear programming problems},\n"
           "  author={Wood, R. L.},\n"
           "  journal={Computer Journal},\n"
           "  year={1968}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  ExtendedWoodN() { static_assert( N % 4 == 0, "ExtendedWoodN requires N to be a multiple of 4" ); }

  Vector lower() const override { return Vector::Constant( N, -3.0 ); }
  Vector upper() const override { return Vector::Constant( N, 3.0 ); }

  Vector init() const override
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) x0[i] = ( i % 2 == 0 ) ? -3.0 : -1.0;
    return x0;
  }

  Vector exact() const override { return Vector::Constant( N, 1.0 ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = 0;
    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;
      T   t1 = x[k1] * x[k1] - x[k2];
      T   t2 = x[k1] - 1.0;
      T   t3 = x[k3] * x[k3] - x[k4];
      T   t4 = x[k3] - 1.0;
      T   t5 = x[k2] + x[k4] - 2.0;
      T   t6 = x[k2] - x[k4];

      f += 100.0 * t1 * t1 + t2 * t2 + 90.0 * t3 * t3 + t4 * t4 + 10.0 * t5 * t5 + 0.1 * t6 * t6;
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] * x[k1] - x[k2];
      T t2 = x[k1] - 1.0;
      T t3 = x[k3] * x[k3] - x[k4];
      T t4 = x[k3] - 1.0;
      T t5 = x[k2] + x[k4] - 2.0;
      T t6 = x[k2] - x[k4];

      g[k1] += 400.0 * t1 * x[k1] + 2.0 * t2;
      g[k2] += -200.0 * t1 + 20.0 * t5 + 0.2 * t6;
      g[k3] += 180.0 * t3 * x[k3] + 2.0 * t4;
      g[k4] += -90.0 * t3 + 20.0 * t5 - 0.2 * t6;
    }

    return g;
  }

  /// Hessian ∇²f(x) sparse 4x4 blocks
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );
    // Each row has at most 4 non-zero elements (within its 4x4 block)
    H.reserve( Eigen::VectorXi::Constant( N, 4 ) );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T a = x[k1], b = x[k2], c = x[k3], d = x[k4];

      // Second derivatives for variables in the current quartet
      // Using the correct formulas derived from the gradient:

      // ∂²f/∂a² = 1200*a² - 400*b + 2
      H.coeffRef( k1, k1 ) += 1200.0 * a * a - 400.0 * b + 2.0;

      // ∂²f/∂a∂b = -400*a (symmetric)
      H.coeffRef( k1, k2 ) += -400.0 * a;
      H.coeffRef( k2, k1 ) += -400.0 * a;

      // ∂²f/∂b² = 220.2 (constant)
      H.coeffRef( k2, k2 ) += 220.2;

      // ∂²f/∂b∂d = 19.8 (constant, symmetric)
      H.coeffRef( k2, k4 ) += 19.8;
      H.coeffRef( k4, k2 ) += 19.8;  // Missing in original!

      // ∂²f/∂c² = 1080*c² - 360*d + 2
      H.coeffRef( k3, k3 ) += 1080.0 * c * c - 360.0 * d + 2.0;

      // ∂²f/∂c∂d = -360*c (symmetric)
      H.coeffRef( k3, k4 ) += -360.0 * c;
      H.coeffRef( k4, k3 ) += -360.0 * c;

      // ∂²f/∂d² = 200.2 (constant)
      H.coeffRef( k4, k4 ) += 200.2;
    }

    H.makeCompressed();
    return H;
  }
};


// -------------------------------------------------------------------
// Freudenstein-Roth Function (2D) - Versione migliorata
// -------------------------------------------------------------------

/**
 * @class FreudensteinRoth2D
 * @brief Freudenstein and Roth function, a 2D nonlinear test function
 *
 * This function has a global minimum at (5,4) with value 0 and
 * a local minimum at (11.41..., -0.8968...) with value 48.9842...
 */
template <typename T> class FreudensteinRoth2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{freudenstein1963,\n"
           "  title={Numerical solutions of systems of nonlinear equations},\n"
           "  author={Freudenstein, F. and Roth, B.},\n"
           "  journal={Journal of the ACM},\n"
           "  year={1963}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }

  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 0.5, -2.0;
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 5.0, 4.0;
    return x;
  }

  // Funzione helper per calcolare f1 e f2 in modo stabile
  struct FunctionValues
  {
    T f1, f2;
    T df1_dx2, df2_dx2;
    T d2f1_dx2x2, d2f2_dx2x2;
  };

  FunctionValues compute_values( const Vector & x ) const
  {
    const T x1 = x[0];
    const T x2 = x[1];

    // Calcoli intermedi con minimizzazione degli errori floating-point
    // const T x2_sq = x2 * x2;
    // const T x2_cu = x2 * x2_sq;

    // Calcolo f1 e f2 in forma annidata (Horner-like) per stabilità
    // f1 = -13 + x1 + (5 - x2)*x2² - 2*x2
    //    = -13 + x1 + 5*x2² - x2³ - 2*x2
    const T poly_f1 = x2 * ( ( 5.0 - x2 ) * x2 - 2.0 );  // (5-x2)*x2² - 2*x2
    const T f1      = ( -13.0 + x1 ) + poly_f1;

    // f2 = -29 + x1 + (x2 + 1)*x2² - 14*x2
    //    = -29 + x1 + x2³ + x2² - 14*x2
    const T poly_f2 = x2 * ( ( x2 + 1.0 ) * x2 - 14.0 );  // (x2+1)*x2² - 14*x2
    const T f2      = ( -29.0 + x1 ) + poly_f2;

    // Derivate prime rispetto a x2
    // df1/dx2 = 10*x2 - 3*x2² - 2
    const T df1_dx2 = x2 * ( 10.0 - 3.0 * x2 ) - 2.0;

    // df2/dx2 = 3*x2² + 2*x2 - 14
    const T df2_dx2 = x2 * ( 3.0 * x2 + 2.0 ) - 14.0;

    // Seconde derivate rispetto a x2
    const T d2f1_dx2x2 = 10.0 - 6.0 * x2;
    const T d2f2_dx2x2 = 6.0 * x2 + 2.0;

    return { f1, f2, df1_dx2, df2_dx2, d2f1_dx2x2, d2f2_dx2x2 };
  }

  /// Objective function - Versione stabile
  T operator()( Vector const & x ) const override
  {
    const auto vals = compute_values( x );
    return vals.f1 * vals.f1 + vals.f2 * vals.f2;
  }

  /// Gradient ∇f(x) - CORRETTO
  Vector gradient( Vector const & x ) const override
  {
    const auto vals = compute_values( x );

    Vector g( 2 );
    // ∂f/∂x1 = 2*f1*∂f1/∂x1 + 2*f2*∂f2/∂x1 = 2*f1*1 + 2*f2*1 = 2*(f1 + f2)
    g[0] = 2.0 * ( vals.f1 + vals.f2 );

    // ∂f/∂x2 = 2*f1*df1/dx2 + 2*f2*df2/dx2
    g[1] = 2.0 * ( vals.f1 * vals.df1_dx2 + vals.f2 * vals.df2_dx2 );

    return g;
  }

  /// Hessian ∇²f(x) sparse 2x2 - OTTIMIZZATO e STABILE
  SparseMatrix hessian( Vector const & x ) const override
  {
    const auto vals = compute_values( x );

    // Elementi dell'Hessiano
    // ∂²f/∂x1² = 2*(∂f1/∂x1*∂f1/∂x1 + ∂f2/∂x1*∂f2/∂x1) = 2*(1*1 + 1*1) = 4
    const T H11 = 4.0;

    // ∂²f/∂x1∂x2 = 2*(∂f1/∂x1*∂f1/∂x2 + ∂f2/∂x1*∂f2/∂x2) = 2*(1*df1/dx2 + 1*df2/dx2)
    const T H12 = 2.0 * ( vals.df1_dx2 + vals.df2_dx2 );

    // ∂²f/∂x2² = 2*[(df1/dx2)² + f1*d²f1/dx2² + (df2/dx2)² + f2*d²f2/dx2²]
    const T df1_sq = vals.df1_dx2 * vals.df1_dx2;
    const T df2_sq = vals.df2_dx2 * vals.df2_dx2;
    const T H22    = 2.0 * ( df1_sq + vals.f1 * vals.d2f1_dx2x2 + df2_sq + vals.f2 * vals.d2f2_dx2x2 );

    // Costruzione matrice sparsa
    SparseMatrix H( 2, 2 );
    H.reserve( 3 );  // Solo 3 elementi unici (simmetria)

    // Inserimento efficiente degli elementi
    typedef Eigen::Triplet<T> Triplet;
    std::vector<Triplet>      triplets;
    triplets.reserve( 4 );

    triplets.emplace_back( 0, 0, H11 );
    triplets.emplace_back( 0, 1, H12 );
    triplets.emplace_back( 1, 0, H12 );
    triplets.emplace_back( 1, 1, H22 );

    H.setFromTriplets( triplets.begin(), triplets.end() );

    return H;
  }
};


// -------------------------------------------------------------------
// Griewank Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class GriewankN
 * @brief Griewank function with Full Hessian calculation.
 *
 * f(x) = 1 + Sum(x_i^2 / 4000) - Prod( cos(x_i / sqrt(i+1)) )
 *
 * Global Minimum: f(0) = 0
 * Search Domain: [-600, 600]
 */
template <typename T, int N> class GriewankN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{griewank1981,\n"
           "  title={Generalized descent for global optimization},\n"
           "  author={Griewank, A. O.},\n"
           "  journal={Journal of Optimization Theory and Applications},\n"
           "  year={1981}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( N, -600.0 ); }
  Vector upper() const override { return Vector::Constant( N, 600.0 ); }

  // CORRETTO: Punto di partenza al bordo del dominio (worst-case scenario)
  Vector init() const override { return Vector::Constant( N, 600.0 ); }

  Vector exact() const override { return Vector::Zero( N ); }

  // -----------------------------------------------------------
  // Valutazione Funzione
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T sum  = 0.0;
    T prod = 1.0;
    for ( int i = 0; i < N; ++i )
    {
      sum += ( x[i] * x[i] );
      prod *= std::cos( x[i] / std::sqrt( T( i + 1 ) ) );
    }
    return 1.0 + ( sum / 4000.0 ) - prod;
  }

  // -----------------------------------------------------------
  // Gradiente
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    T      prod_all = 1.0;

    // Calcolo preliminare della produttoria completa
    for ( int i = 0; i < N; ++i ) prod_all *= std::cos( x[i] / std::sqrt( T( i + 1 ) ) );

    for ( int i = 0; i < N; ++i )
    {
      T sqrt_k = std::sqrt( T( i + 1 ) );
      T arg    = x[i] / sqrt_k;

      // g_i = x_i/2000 + Prod * tan(arg) / sqrt(k)
      T term_linear = x[i] / 2000.0;

      // Nota: tan(arg) = sin(arg)/cos(arg).
      // Il termine 'prod_all' contiene cos(arg), che si cancella col denominatore della tan.
      // g_i = x_i/2000 + (Prod_{j!=i} cos_j) * sin_i / sqrt_k
      // Usiamo tan per compattezza codice, assumendo cos(arg) != 0 nei float
      T term_trig = prod_all * std::tan( arg ) / sqrt_k;

      g[i] = term_linear + term_trig;
    }
    return g;
  }

  // -----------------------------------------------------------
  // Hessiana Completa (Densa)
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    // Griewank ha termini misti non nulli.
    // H_ij = - Prod * tan(x_i/k_i)/k_i * tan(x_j/k_j)/k_j  (per i != j)

    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    // Riserviamo spazio per N*N elementi (matrice piena)
    triplets.reserve( N * N );

    // 1. Precalcolo Produttoria e Fattori Tangente
    T prod_all = 1.0;
    T tan_factors[N];  // Memorizza: tan(x_k/sqrt_k) / sqrt_k

    for ( int k = 0; k < N; ++k )
    {
      T sqrt_k = std::sqrt( T( k + 1 ) );
      T arg    = x[k] / sqrt_k;

      prod_all *= std::cos( arg );
      tan_factors[k] = std::tan( arg ) / sqrt_k;
    }

    // 2. Costruzione Matrice
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        T val = 0.0;

        if ( i == j )
        {
          // Elemento Diagonale
          // H_ii = 1/2000 + Prod / (i+1)
          T denom = T( i + 1 );
          val     = ( 1.0 / 2000.0 ) + ( prod_all / denom );
        }
        else
        {
          // Elemento Off-Diagonal (Termine Misto)
          // H_ij = - Prod * (tan_i/sqrt_i) * (tan_j/sqrt_j)
          val = -prod_all * tan_factors[i] * tan_factors[j];
        }

        // Inseriamo anche se molto piccolo, per correttezza strutturale
        triplets.emplace_back( i, j, val );
      }
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// HappyCat Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class HappyCatN
 * @brief HappyCat function, a difficult benchmark from CEC 2014.
 *
 * f(x) = | ||x||^2 - N |^0.25 + (0.5 * ||x||^2 + sum(x)) / N + 0.5
 *
 * Global Minimum: x = (-1, -1, ..., -1) with f(x) = 0
 * Domain: Usually evaluated in [-20, 20]
 */
template <typename T, int N> class HappyCatN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{cec2014,\n"
           "  title={Benchmark Functions for the CEC 2014 Special Session},\n"
           "  author={Li, X. et al.},\n"
           "  journal={IEEE CEC},\n"
           "  year={2014}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  // Standard literature bounds are typically wider than [-2, 2]
  Vector lower() const override { return Vector::Constant( N, -20.0 ); }
  Vector upper() const override { return Vector::Constant( N, 20.0 ); }

  // Init point: lontano dalla soluzione (-1, -1...)
  Vector init() const override { return Vector::Constant( N, 5.0 ); }

  Vector exact() const override { return Vector::Constant( N, -1.0 ); }

  // -----------------------------------------------------------
  // Objective Function
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T r2 = x.squaredNorm();
    T s  = x.sum();
    // f = |r^2 - N|^0.25 + (0.5*r^2 + s)/N + 0.5
    return std::pow( std::abs( r2 - T( N ) ), 0.25 ) + ( 0.5 * r2 + s ) / T( N ) + 0.5;
  }

  // -----------------------------------------------------------
  // Gradient
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    T      r2   = x.squaredNorm();
    T      diff = r2 - T( N );

    // Handle singularity at r2 = N more carefully
    if ( std::abs( diff ) < 1e-12 )
    {
      // At the singularity, the gradient from the first term is not well-defined
      // We only return the gradient from the quadratic part
      for ( int i = 0; i < N; ++i ) { g[i] = ( x[i] + 1.0 ) / T( N ); }
      return g;
    }

    // Compute derivative of first term: d/dx [|u|^0.25] where u = r^2 - N
    // d/dx_i [|u|^0.25] = 0.25 * |u|^(-0.75) * sign(u) * du/dx_i
    // du/dx_i = 2*x_i
    // So: 0.25 * |u|^(-0.75) * sign(u) * 2*x_i = 0.5 * x_i * sign(u) * |u|^(-0.75)

    T abs_diff  = std::abs( diff );
    T sign_diff = ( diff > 0 ) ? T( 1 ) : ( diff < 0 ) ? T( -1 ) : T( 0 );

    // Avoid potential division by very small number
    if ( abs_diff < 1e-12 ) { abs_diff = 1e-12; }

    T term_pow = std::pow( abs_diff, -0.75 );  // |diff|^(-0.75)
    T factor   = 0.5 * sign_diff * term_pow;

    for ( int i = 0; i < N; ++i )
    {
      // First term: derivative of |r^2 - N|^0.25
      T first_term = factor * x[i];

      // Second term: derivative of (0.5*r^2 + sum(x))/N
      T second_term = ( x[i] + 1.0 ) / T( N );

      g[i] = first_term + second_term;
    }
    return g;
  }

  // -----------------------------------------------------------
  // Hessian
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N * N );

    T r2   = x.squaredNorm();
    T diff = r2 - T( N );

    // Handle singularity at r2 = N
    if ( std::abs( diff ) < 1e-9 )
    {
      // Return only the Hessian from the quadratic part
      SparseMatrix H( N, N );
      for ( int i = 0; i < N; ++i ) { triplets.emplace_back( i, i, 1.0 / T( N ) ); }
      H.setFromTriplets( triplets.begin(), triplets.end() );
      return H;
    }

    T abs_diff  = std::abs( diff );
    T sign_diff = ( diff > 0 ) ? T( 1 ) : T( -1 );

    // Avoid division by very small numbers
    if ( abs_diff < 1e-12 ) { abs_diff = 1e-12; }

    // Compute powers
    T pow_m075 = std::pow( abs_diff, -0.75 );  // |diff|^(-0.75)
    T pow_m175 = std::pow( abs_diff, -1.75 );  // |diff|^(-1.75)

    // Coefficients from the Hessian calculation
    // First term: 0.5 * sign(diff) * |diff|^(-0.75) * I
    // Second term: -0.75 * |diff|^(-1.75) * x * x^T
    // Note: The sign(diff)^2 = 1, so it doesn't appear in the second term

    T coeff_diag  = 0.5 * sign_diff * pow_m075;
    T coeff_rank1 = 0.75 * pow_m175;

    // Add the quadratic part: (1/N) * I
    coeff_diag += 1.0 / T( N );

    // Fill the Hessian matrix
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        T val = 0.0;

        if ( i == j )
        {
          // Diagonal elements
          val += coeff_diag;
        }

        // Rank-1 update from the first term (always present, not just off-diagonal)
        val -= coeff_rank1 * x[i] * x[j];

        triplets.emplace_back( i, j, val );
      }
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// HGBat Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class HGBatN
 * @brief HGBat function (High Conditioning Bat), CEC 2014 Benchmark.
 *
 * f(x) = | (r^2)^2 - s^2 |^0.5 + (0.5 * r^2 + s) / N + 0.5
 *
 * Global Minimum: x = (-1, ..., -1) -> f(x) = 0
 * Domain: Usually [-100, 100], strictly defined in CEC 2014.
 */
template <typename T, int N> class HGBatN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{cec2014,\n"
           "  title={Benchmark Functions for the CEC 2014 Special Session},\n"
           "  author={Li, X. et al.},\n"
           "  journal={IEEE CEC},\n"
           "  year={2014}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  // Estensione bounds allo standard per testare la ricerca globale
  Vector lower() const override { return Vector::Constant( N, -100.0 ); }
  Vector upper() const override { return Vector::Constant( N, 100.0 ); }

  // Init: Lontano da 0 (singolarità) e da -1 (soluzione)
  Vector init() const override { return Vector::Constant( N, 5.0 ); }

  Vector exact() const override { return Vector::Constant( N, -1.0 ); }

  // -----------------------------------------------------------
  // Objective Function
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T r2 = x.squaredNorm();
    T s  = x.sum();
    T u  = ( r2 * r2 ) - ( s * s );
    return std::pow( std::abs( u ), 0.5 ) + ( 0.5 * r2 + s ) / T( N ) + 0.5;
  }

  // -----------------------------------------------------------
  // Gradient - CORRECTED VERSION
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    T      r2 = x.squaredNorm();
    T      s  = x.sum();
    T      u  = ( r2 * r2 ) - ( s * s );

    // Handle singularity at u = 0
    if ( std::abs( u ) < 1e-12 )
    {
      for ( int i = 0; i < N; ++i ) g[i] = ( x[i] + 1.0 ) / T( N );
      return g;
    }

    // Compute derivative of first term: d/dx [|u|^{1/2}]
    // Let v = sqrt(|u|) = |u|^{1/2}
    // dv/dx_i = 0.5 * |u|^{-1/2} * sign(u) * du/dx_i

    T abs_u  = std::abs( u );
    T sign_u = ( u > 0 ) ? T( 1 ) : T( -1 );

    // Compute du/dx_i = 4*r^2*x_i - 2*s
    // This is correct

    // Factor for the first term: 0.5 * sign(u) / sqrt(|u|)
    T factor = 0.5 * sign_u / std::sqrt( abs_u );

    for ( int i = 0; i < N; ++i )
    {
      T du_dxi    = 4.0 * r2 * x[i] - 2.0 * s;
      T grad_root = factor * du_dxi;  // This is correct

      // Linear part: (x_i + 1)/N
      T grad_lin = ( x[i] + 1.0 ) / T( N );

      g[i] = grad_root + grad_lin;
    }
    return g;
  }

  // -----------------------------------------------------------
  // Hessian - CORRECTED VERSION
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N * N );

    T r2    = x.squaredNorm();
    T s     = x.sum();
    T u     = ( r2 * r2 ) - ( s * s );
    T abs_u = std::abs( u );

    // Handle singularity at u = 0
    if ( abs_u < 1e-9 )
    {
      // Return only the Hessian from the quadratic part
      SparseMatrix H( N, N );
      for ( int i = 0; i < N; ++i ) { triplets.emplace_back( i, i, 1.0 / T( N ) ); }
      H.setFromTriplets( triplets.begin(), triplets.end() );
      return H;
    }

    T sign_u = ( u > 0 ) ? T( 1 ) : T( -1 );

    // Compute first and second derivatives of sqrt(|u|)
    // g(u) = sqrt(|u|) = |u|^{1/2}
    // g'(u) = 0.5 * sign(u) / sqrt(|u|)
    // g''(u) = -0.25 / (|u|^{3/2})  [Note: the sign disappears in the second derivative]

    T sqrt_abs_u   = std::sqrt( abs_u );
    T abs_u_pow_15 = abs_u * sqrt_abs_u;  // |u|^{3/2}

    T g_prime        = 0.5 * sign_u / sqrt_abs_u;
    T g_double_prime = -0.25 / abs_u_pow_15;

    // Precompute du/dx for all components
    Vector du_dx( N );
    for ( int k = 0; k < N; ++k ) { du_dx[k] = 4.0 * r2 * x[k] - 2.0 * s; }

    // Linear part Hessian: (1/N) * I
    T H_lin = 1.0 / T( N );

    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        // Compute second derivative of u: d²u/(dx_i dx_j)
        // From du/dx_i = 4*r^2*x_i - 2*s
        // d/dx_j (4*r^2*x_i) = 4*(d(r^2)/dx_j * x_i + r^2 * δ_ij)
        // d(r^2)/dx_j = 2*x_j
        // So: 4*(2*x_j*x_i + r^2*δ_ij) = 8*x_i*x_j + 4*r^2*δ_ij
        // d/dx_j (-2*s) = -2
        // Therefore: d²u/(dx_i dx_j) = 8*x_i*x_j + 4*r^2*δ_ij - 2

        T delta_ij   = ( i == j ) ? T( 1 ) : T( 0 );
        T d2u_dxidxj = 8.0 * x[i] * x[j] + 4.0 * r2 * delta_ij - 2.0;

        // Hessian from chain rule:
        // H = g'(u) * H_u + g''(u) * (∇u * ∇u^T)
        T val = g_prime * d2u_dxidxj + g_double_prime * du_dx[i] * du_dx[j];

        // Add linear part
        if ( i == j ) { val += H_lin; }

        triplets.emplace_back( i, j, val );
      }
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Helical Valley Function (3D)
// -------------------------------------------------------------------

/**
 * @class HelicalValley3D
 * @brief Helical Valley function, a 3D optimization test function
 *
 * This function has a global minimum at (1,0,0) with value 0.
 * The function features a helical valley that winds around the z-axis.
 */
template <typename T> class HelicalValley3D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{powell1970,\n"
           "  title={A hybrid method for nonlinear equations},\n"
           "  author={Powell, M. J. D.},\n"
           "  journal={Numerical Methods for Nonlinear Algebraic Equations},\n"
           "  year={1970}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  // Pi greco costante
  const T PI = std::acos( T( -1.0 ) );

  Vector lower() const override { return Vector::Constant( 3, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 3, 10.0 ); }

  Vector init() const override
  {
    Vector x0( 3 );
    x0 << T( -1.0 ), T( 0.0 ), T( 0.0 );
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 3 );
    x << T( 1.0 ), T( 0.0 ), T( 0.0 );
    return x;
  }

  /// Objective function f(x)
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1], x3 = x[2];
    T theta;

    if ( x1 > 0 )
      theta = std::atan( x2 / x1 ) / ( 2.0 * PI );
    else if ( x1 < 0 )
      theta = ( std::atan( x2 / x1 ) / ( 2.0 * PI ) ) + 0.5;
    else
      theta = ( x2 >= 0 ) ? 0.25 : -0.25;

    T f1 = 10.0 * ( x3 - 10.0 * theta );
    T f2 = 10.0 * ( std::sqrt( x1 * x1 + x2 * x2 ) - 1.0 );
    T f3 = x3;

    return f1 * f1 + f2 * f2 + f3 * f3;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1], x3 = x[2];
    T r2 = x1 * x1 + x2 * x2;
    T r  = std::sqrt( r2 );

    T theta, dtheta_dx1, dtheta_dx2;
    if ( r2 > 1e-12 )
    {  // Evitiamo divisione per zero all'origine
      if ( x1 > 0 )
        theta = std::atan( x2 / x1 ) / ( 2.0 * PI );
      else if ( x1 < 0 )
        theta = std::atan( x2 / x1 ) / ( 2.0 * PI ) + 0.5;
      else
        theta = ( x2 >= 0 ) ? 0.25 : -0.25;

      dtheta_dx1 = -x2 / ( 2.0 * PI * r2 );
      dtheta_dx2 = x1 / ( 2.0 * PI * r2 );
    }
    else
    {
      theta      = 0;
      dtheta_dx1 = 0;
      dtheta_dx2 = 0;
    }

    T f1 = 10.0 * ( x3 - 10.0 * theta );
    T f2 = 10.0 * ( r - 1.0 );

    // Derivate componenti:
    // df1/dx1 = 10 * (-10 * dtheta/dx1) = -100 * dtheta/dx1
    // df2/dx1 = 10 * (x1/r)

    Vector g( 3 );
    g[0] = 2.0 * f1 * ( -100.0 * dtheta_dx1 ) + 2.0 * f2 * ( 10.0 * x1 / r );
    g[1] = 2.0 * f1 * ( -100.0 * dtheta_dx2 ) + 2.0 * f2 * ( 10.0 * x2 / r );
    g[2] = 2.0 * f1 * 10.0 + 2.0 * x3;

    return g;
  }

  /// Hessian ∇²f(x)
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1], x3 = x[2];
    T r2 = x1 * x1 + x2 * x2;
    T r  = std::sqrt( r2 );

    T dtheta_dx1 = -x2 / ( 2.0 * PI * r2 );
    T dtheta_dx2 = x1 / ( 2.0 * PI * r2 );
    T theta;
    if ( x1 > 0 )
      theta = std::atan( x2 / x1 ) / ( 2.0 * PI );
    else if ( x1 < 0 )
      theta = std::atan( x2 / x1 ) / ( 2.0 * PI ) + 0.5;
    else
      theta = ( x2 >= 0 ) ? 0.25 : -0.25;

    T f1 = 10.0 * ( x3 - 10.0 * theta );
    T f2 = 10.0 * ( r - 1.0 );

    // Derivate seconde di theta
    T d2theta_dx1dx1 = ( 2.0 * x1 * x2 ) / ( 2.0 * PI * r2 * r2 );
    T d2theta_dx2dx2 = ( -2.0 * x1 * x2 ) / ( 2.0 * PI * r2 * r2 );
    T d2theta_dx1dx2 = ( x1 * x1 - x2 * x2 ) / ( 2.0 * PI * r2 * r2 );

    SparseMatrix H( 3, 3 );
    H.reserve( Eigen::VectorXi::Constant( 3, 3 ) );

    // f1 = 10(x3 - 10*theta) -> df1/dx1 = -100 * dtheta_dx1
    T df1_dx1 = -100.0 * dtheta_dx1;
    T df1_dx2 = -100.0 * dtheta_dx2;
    T df1_dx3 = 10.0;

    // f2 = 10(r - 1) -> df2/dx1 = 10 * x1/r
    T df2_dx1 = 10.0 * x1 / r;
    T df2_dx2 = 10.0 * x2 / r;

    // H(0,0)
    H.insert( 0, 0 ) = 2.0 * ( df1_dx1 * df1_dx1 + f1 * ( -100.0 * d2theta_dx1dx1 ) + df2_dx1 * df2_dx1 +
                               f2 * ( 10.0 * ( r2 - x1 * x1 ) / ( r * r2 ) ) );

    // H(1,1)
    H.insert( 1, 1 ) = 2.0 * ( df1_dx2 * df1_dx2 + f1 * ( -100.0 * d2theta_dx2dx2 ) + df2_dx2 * df2_dx2 +
                               f2 * ( 10.0 * ( r2 - x2 * x2 ) / ( r * r2 ) ) );

    // H(2,2)
    H.insert( 2, 2 ) = 2.0 * ( df1_dx3 * df1_dx3 ) + 2.0;  // (2 * 10^2) + 2 = 202

    // H(0,1) Simmetrica
    T h01            = 2.0 * ( df1_dx1 * df1_dx2 + f1 * ( -100.0 * d2theta_dx1dx2 ) + df2_dx1 * df2_dx2 +
                               f2 * ( -10.0 * x1 * x2 / ( r * r2 ) ) );
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = h01;

    // H(0,2) e H(1,2)
    H.insert( 0, 2 ) = H.insert( 2, 0 ) = 2.0 * df1_dx1 * 10.0;
    H.insert( 1, 2 ) = H.insert( 2, 1 ) = 2.0 * df1_dx2 * 10.0;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Himmelblau Function (2D)
// -------------------------------------------------------------------

/**
 * @class Himmelblau2D
 * @brief Himmelblau's function, a classic 2D optimization test function
 *
 * This function has four identical local minima with value 0, making it
 * useful for testing global optimization algorithms.
 * Minima at: (3.0,2.0), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848)
 */
template <typename T> class Himmelblau2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{himmelblau1972,\n"
           "  title={Applied Nonlinear Programming},\n"
           "  author={Himmelblau, D. M.},\n"
           "  publisher={McGraw-Hill},\n"
           "  year={1972}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -6.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 6.0 ); }

  Vector init() const override
  {
    Vector x0( 2 );
    x0 << -3.0, -3.0;
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 3.0, 2.0;
    return x;
  }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T f1 = x1 * x1 + x2 - 11.0;
    T f2 = x1 + x2 * x2 - 7.0;
    return f1 * f1 + f2 * f2;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T f1 = x1 * x1 + x2 - 11.0;
    T f2 = x1 + x2 * x2 - 7.0;

    Vector g( 2 );
    g[0] = 4.0 * x1 * f1 + 2.0 * f2;
    g[1] = 2.0 * f1 + 4.0 * x2 * f2;

    return g;
  }

  /// Hessian ∇²f(x) sparse 2x2
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T f1 = x1 * x1 + x2 - 11.0;
    T f2 = x1 + x2 * x2 - 7.0;

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    H.insert( 0, 0 ) = 4.0 * f1 + 8.0 * x1 * x1 + 2.0;
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = 4.0 * x1 + 4.0 * x2;
    H.insert( 1, 1 )                    = 2.0 + 4.0 * f2 + 8.0 * x2 * x2;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Holder Table Function (2D)
// -------------------------------------------------------------------

/**
 * @class HolderTable2D
 * @brief Holder Table function, a multi-modal optimization test function.
 *
 * The Holder Table function has many local minima, with four global minima.
 * Formula: f(x) = - | sin(x) * cos(y) * exp( |1 - sqrt(x^2+y^2)/pi| ) |
 * Global Minima: f(x*) = -19.2085 at (+/- 8.05502, +/- 9.66459)
 * Domain: Usually x_i in [-10, 10]
 */
template <typename T> class HolderTable2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{holdertable,\n"
           "  title={Benchmark functions for global optimization},\n"
           "  author={Hedar, A. R.},\n"
           "  journal={Global Optimization Benchmarks},\n"
           "  year={2005}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }

  // Punto iniziale
  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 0.5, 0.5;
    return x0;
  }

  // Uno dei 4 minimi globali
  Vector exact() const override
  {
    Vector x( 2 );
    x << 8.05502, 9.66459;
    return x;
  }

  // -----------------------------------------------------------
  // Objective Function
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T pi = Utils::m_pi;

    T R       = std::sqrt( x1 * x1 + x2 * x2 );
    T exp_arg = std::abs( 1.0 - R / pi );

    T term = std::sin( x1 ) * std::cos( x2 ) * std::exp( exp_arg );
    return -std::abs( term );
  }

  // -----------------------------------------------------------
  // Gradient
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T pi = Utils::m_pi;

    // Compute intermediate values
    T R = std::sqrt( x1 * x1 + x2 * x2 );

    // Handle R = 0 case to avoid division by zero
    if ( R < 1e-12 ) { R = 1e-12; }

    T inner     = 1.0 - R / pi;
    T abs_inner = std::abs( inner );
    T exp_val   = std::exp( abs_inner );

    // The function inside the outer absolute value
    T g = std::sin( x1 ) * std::cos( x2 ) * exp_val;

    // If g is exactly 0, the function is not differentiable
    // Return a zero gradient as a subgradient
    if ( std::abs( g ) < 1e-12 ) { return Vector::Zero( 2 ); }

    // Compute gradient of g (without the outer absolute value)
    T sign_inner = ( inner > 0 ) ? T( 1 ) : T( -1 );

    // Partial derivatives
    // dg/dx1 = cos(x1)*cos(x2)*exp_val + sin(x1)*cos(x2)*exp_val*d(exp_arg)/dx1
    // d(exp_arg)/dx1 = d|1-R/π|/dx1 = sign(1-R/π) * (-1/π) * (x1/R)

    T d_exp_arg_dx1 = sign_inner * ( -x1 ) / ( pi * R );
    T dg_dx1 = std::cos( x1 ) * std::cos( x2 ) * exp_val + std::sin( x1 ) * std::cos( x2 ) * exp_val * d_exp_arg_dx1;

    T d_exp_arg_dx2 = sign_inner * ( -x2 ) / ( pi * R );
    T dg_dx2        = std::sin( x1 ) * ( -std::sin( x2 ) ) * exp_val +
                      std::sin( x1 ) * std::cos( x2 ) * exp_val * d_exp_arg_dx2;

    // Now apply the outer absolute value: f = -|g|
    // df/dx = -sign(g) * dg/dx
    T sign_g = ( g > 0 ) ? T( 1 ) : T( -1 );

    Vector grad( 2 );
    grad[0] = -sign_g * dg_dx1;
    grad[1] = -sign_g * dg_dx2;

    return grad;
  }

  // -----------------------------------------------------------
  // Hessian
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T pi = Utils::m_pi;

    // Compute intermediate values
    T R = std::sqrt( x1 * x1 + x2 * x2 );

    // Handle R = 0 case
    if ( R < 1e-12 ) { R = 1e-12; }

    T inner     = 1.0 - R / pi;
    T abs_inner = std::abs( inner );
    T exp_val   = std::exp( abs_inner );

    // The function inside the outer absolute value
    T g = std::sin( x1 ) * std::cos( x2 ) * exp_val;

    // If g is near 0, the Hessian is not well-defined
    // Return a zero matrix as a subgradient
    if ( std::abs( g ) < 1e-12 )
    {
      SparseMatrix H( 2, 2 );
      H.setZero();
      return H;
    }

    T sign_inner = ( inner > 0 ) ? T( 1 ) : T( -1 );
    T sign_g     = ( g > 0 ) ? T( 1 ) : T( -1 );

    // Precompute common terms
    T s1 = std::sin( x1 ), c1 = std::cos( x1 );
    T s2 = std::sin( x2 ), c2 = std::cos( x2 );

    T factor = -sign_inner / ( pi * R );
    // T x1_over_R = x1 / R;
    // T x2_over_R = x2 / R;

    // First derivatives of g
    // T dg_dx1 = c1 * c2 * exp_val + s1 * c2 * exp_val * factor * x1;
    // T dg_dx2 = s1 * (-s2) * exp_val + s1 * c2 * exp_val * factor * x2;

    // Second derivatives of the exponential argument
    T d2_exp_arg_dx1dx1 = sign_inner * ( -1.0 / pi ) * ( 1.0 / R - x1 * x1 / ( R * R * R ) );
    T d2_exp_arg_dx1dx2 = sign_inner * ( -1.0 / pi ) * ( -x1 * x2 / ( R * R * R ) );
    T d2_exp_arg_dx2dx2 = sign_inner * ( -1.0 / pi ) * ( 1.0 / R - x2 * x2 / ( R * R * R ) );

    // Second derivatives of g
    T d2g_dx1dx1 = -s1 * c2 * exp_val + 2.0 * c1 * c2 * exp_val * factor * x1 +
                   s1 * c2 * exp_val * ( factor * factor * x1 * x1 + d2_exp_arg_dx1dx1 );

    T d2g_dx1dx2 = -c1 * s2 * exp_val + c1 * c2 * exp_val * factor * x2 + s1 * ( -s2 ) * exp_val * factor * x1 +
                   s1 * c2 * exp_val * ( factor * factor * x1 * x2 + d2_exp_arg_dx1dx2 );

    T d2g_dx2dx2 = -s1 * c2 * exp_val + 2.0 * s1 * ( -s2 ) * exp_val * factor * x2 +
                   s1 * c2 * exp_val * ( factor * factor * x2 * x2 + d2_exp_arg_dx2dx2 );

    // Apply the outer absolute value: f = -|g|
    // Hessian: H_f = -sign(g) * H_g  (ignoring the Dirac delta at g=0)
    T h11 = -sign_g * d2g_dx1dx1;
    T h12 = -sign_g * d2g_dx1dx2;
    T h22 = -sign_g * d2g_dx2dx2;

    // Create sparse matrix
    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    H.insert( 0, 0 ) = h11;
    H.insert( 0, 1 ) = h12;
    H.insert( 1, 0 ) = h12;  // Symmetric
    H.insert( 1, 1 ) = h22;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Ill-conditioned Quadratic Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class IllConditionedQuadraticN
 * @brief Ill-conditioned quadratic function
 *
 * This function features exponentially increasing eigenvalues, creating
 * a very ill-conditioned Hessian matrix. Useful for testing algorithm
 * robustness to poor conditioning. The global minimum is at the origin.
 */
template <typename T, int N> class IllConditionedQuadraticN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@book{nocedal2006,\n"
           "  title={Numerical Optimization},\n"
           "  author={Nocedal, J. and Wright, S.},\n"
           "  publisher={Springer},\n"
           "  year={2006}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 2, "IllConditionedQuadraticN requires N >= 2" );

  Vector lower() const override { return Vector::Constant( N, -10.0 ); }
  Vector upper() const override { return Vector::Constant( N, 10.0 ); }

  Vector init() const override
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) x0[i] = ( i % 2 == 0 ) ? 1.0 : -1.0;
    return x0;
  }

  Vector exact() const override { return Vector::Zero( N ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = 0;
    for ( int i = 0; i < N; ++i )
    {
      T lambda = std::pow( 1e6, T( i ) / T( N - 1 ) );
      f += lambda * x[i] * x[i];
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i )
    {
      T lambda = std::pow( 1e6, T( i ) / T( N - 1 ) );
      g[i]     = 2.0 * lambda * x[i];
    }
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN (diagonal)
  SparseMatrix hessian( Vector const & ) const override
  {
    SparseMatrix H( N, N );
    H.reserve( N );

    for ( int i = 0; i < N; ++i )
    {
      T lambda         = std::pow( 1e6, T( i ) / T( N - 1 ) );
      H.insert( i, i ) = 2.0 * lambda;
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Katsuura Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class KatsuuraN
 * @brief Katsuura Function, continuous but nowhere differentiable.
 *
 * f(x) = Product_{i=0}^{D-1} [ 1 + (i+1) * Sum_{k=1}^{32} ( |2^k x_i - round(2^k x_i)| / 2^k ) ]
 *
 * It is highly multimodal and has a fractal structure.
 *
 * Global Minimum: f(0, ..., 0) = 1
 * Domain: x_i in [-100, 100] (usually)
 */
template <typename T, int N> class KatsuuraN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{katsuura1967,\n"
           "  title={Continuous nowhere differentiable functions},\n"
           "  author={Katsuura, H.},\n"
           "  journal={Journal of Mathematical Analysis},\n"
           "  year={1967}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( N, -100.0 ); }
  Vector upper() const override { return Vector::Constant( N, 100.0 ); }

  // Punto iniziale suggerito (lontano dall'origine)
  Vector init() const override { return Vector::Constant( N, 10.0 ); }

  // Minimo globale
  Vector exact() const override { return Vector::Zero( N ); }

  // -----------------------------------------------------------
  // Objective Function
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T product = 1.0;
    // Potenza di 2 pre-calcolata per efficienza o calcolata al volo
    // Il loop interno va tipicamente fino a d=32 (precisione bit standard)
    int d = 32;

    for ( int i = 0; i < x.size(); ++i )
    {
      T sum = 0.0;
      T xi  = x[i];

      // Calcolo della sommatoria interna
      // sum = Sum_{k=1}^{32} | 2^k*xi - round(2^k*xi) | / 2^k

      T two_pow_k = 2.0;  // Inizia da 2^1
      for ( int k = 1; k <= d; ++k )
      {
        T term = two_pow_k * xi;
        // | term - round(term) | / 2^k
        // std::round arrotonda all'intero più vicino
        T numerator = std::abs( term - std::round( term ) );

        sum += numerator / two_pow_k;

        two_pow_k *= 2.0;  // Incrementa potenza
      }

      // Moltiplicatore per la dimensione i (1-based index usually)
      // Formula: (1 + (i+1)*sum)
      product *= ( 1.0 + ( T( i ) + 1.0 ) * sum );
    }

    return product;
  }

  // -----------------------------------------------------------
  // Gradient
  // La funzione è "nowhere differentiable", il gradiente analitico non esiste.
  // Restituiamo 0.
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override { return Vector::Zero( x.size() ); }

  // -----------------------------------------------------------
  // Hessian
  // Non esiste. Restituiamo matrice nulla.
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( x.size(), x.size() );
    H.setZero();
    Eigen::Index n = x.size();
    for ( Eigen::Index i = 0; i < n; ++i ) H.insert( i, i ) = 0;
    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Langermann Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class Langermann
 * @brief Langermann Function, a multimodal test function with unevenly distributed minima.
 *
 * f(x) = - sum_{i=1}^{m} c_i * exp( -1/pi * sum((x_j - A_ij)^2) ) * cos( pi * sum((x_j - A_ij)^2) )
 *
 * The landscape is characterized by several local minima determined by the matrix A.
 * The minima are extremely steep and surrounded by oscillatory rings.
 *
 * Standard Parameters (m=5, dim=2):
 * Global Minimum: f(2.00299, 1.00609) approx -5.1621259
 * Domain: x_i in [0, 10]
 */

template <typename T, int N> class Langermann : public NDbase<T>
{
private:
  int                                              m;  // Numero di massimi/minimi (default 5)
  Eigen::Matrix<T, Eigen::Dynamic, 1>              c;  // Vettore coefficienti (ampiezze)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;  // Matrice posizioni (m x dim)

public:
  // Dichiarare i tipi ALIAS PRIMA dei metodi che li usano
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  int const dim = N;

  Langermann() : m( 5 )
  {
    // Inizializzazione standard per il caso 2D (m=5)
    // Se d > 2, la matrice A userà 0 per le dimensioni extra o deve essere riconfigurata
    c.resize( m );
    c << 1.0, 2.0, 5.0, 2.0, 3.0;

    A.resize( m, dim );
    A.setZero();

    // Standard values for first 2 dimensions
    // Row 0
    A( 0, 0 ) = 3.0;
    if ( dim >= 2 ) A( 0, 1 ) = 5.0;
    // Row 1
    A( 1, 0 ) = 5.0;
    if ( dim >= 2 ) A( 1, 1 ) = 2.0;
    // Row 2 (Vicino al minimo globale)
    A( 2, 0 ) = 2.0;
    if ( dim >= 2 ) A( 2, 1 ) = 1.0;
    // Row 3
    A( 3, 0 ) = 1.0;
    if ( dim >= 2 ) A( 3, 1 ) = 4.0;
    // Row 4
    A( 4, 0 ) = 7.0;
    if ( dim >= 2 ) A( 4, 1 ) = 9.0;

    // Per dimensioni > 2, impostiamo valori di default
    for ( int i = 0; i < m; ++i )
    {
      for ( int j = 2; j < dim; ++j )
      {
        A( i, j ) = T( 0.0 );  // Valore di default
      }
    }
  }

  std::string bibtex() const override
  {
    return "@article{langermann1999,\n"
           "  title={Numerical optimization benchmarks},\n"
           "  author={Langermann, K.},\n"
           "  journal={Numerical Methods},\n"
           "  year={1999}\n"
           "}\n";
  }

  Vector lower() const override { return Vector::Constant( dim, 0.0 ); }
  Vector upper() const override { return Vector::Constant( dim, 10.0 ); }

  // Punto iniziale
  Vector init() const override
  {
    return Vector::Constant( dim, 5.0 );  // Centro del dominio
  }

  // Minimo globale approssimato (per i parametri standard m=5)
  Vector exact() const override
  {
    // Nota: Il minimo esatto non è analitico semplice come x=0,
    // ma per A(2) = [2, 1] e c(2)=5, il minimo è molto vicino a questo punto.
    Vector x( dim );
    x.setZero();
    x( 0 ) = 2.002992;
    if ( dim >= 2 ) x( 1 ) = 1.006096;
    // Per dim > 2, gli altri componenti rimangono 0
    return x;
  }

  // -----------------------------------------------------------
  // Objective Function
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T sum_outer = 0.0;
    T pi        = Utils::m_pi;

    for ( int i = 0; i < m; ++i )
    {
      T sum_sq_diff = 0.0;
      for ( int j = 0; j < dim; ++j )
      {
        T diff = x[j] - A( i, j );
        sum_sq_diff += diff * diff;
      }

      T term1 = std::exp( -sum_sq_diff / pi );
      T term2 = std::cos( pi * sum_sq_diff );

      sum_outer += c[i] * term1 * term2;
    }

    return -sum_outer;  // Minimizzazione
  }

  // -----------------------------------------------------------
  // Gradient Analitico
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    Vector g  = Vector::Zero( dim );
    T      pi = Utils::m_pi;

    for ( int i = 0; i < m; ++i )
    {
      // Calcolo u_i = sum (x_j - A_ij)^2
      T u_i = 0.0;
      for ( int j = 0; j < dim; ++j )
      {
        T diff = x[j] - A( i, j );
        u_i += diff * diff;
      }

      // Precalcolo termini comuni
      T exp_term = std::exp( -u_i / pi );
      T cos_term = std::cos( pi * u_i );
      T sin_term = std::sin( pi * u_i );

      // Derivata parziale rispetto a u_i della parte interna (c * e * cos)
      // d/du [ c * e^(-u/pi) * cos(pi*u) ]
      // = c * [ (-1/pi)e * cos - pi * e * sin ]
      // = -c * e * [ (1/pi)*cos + pi*sin ]

      T factor = -c[i] * exp_term * ( ( 1.0 / pi ) * cos_term + pi * sin_term );

      // Chain rule: d(u_i)/dx_k = 2 * (x_k - A_ik)
      // Poiché f è la somma negativa: df/dx_k = - sum_i [ factor * 2 * (x_k - A_ik) ]
      for ( int k = 0; k < dim; ++k ) { g[k] -= factor * 2.0 * ( x[k] - A( i, k ) ); }
    }
    return g;
  }

  // -----------------------------------------------------------
  // Hessiana Analitica
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    // L'Hessiana è densa a blocchi a causa della somma di esponenziali
    // Calcoliamo la matrice densa e la convertiamo in sparse.
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> H_dense( dim, dim );
    H_dense.setZero();

    T pi        = Utils::m_pi;
    T pi_sq     = pi * pi;
    T inv_pi_sq = 1.0 / pi_sq;

    for ( int i = 0; i < m; ++i )
    {
      T u_i = 0.0;
      for ( int j = 0; j < dim; ++j )
      {
        T diff = x[j] - A( i, j );
        u_i += diff * diff;
      }

      T exp_t = std::exp( -u_i / pi );
      T cos_t = std::cos( pi * u_i );
      T sin_t = std::sin( pi * u_i );

      // Termini derivati dall'espressione (vedere note matematiche)
      // Phi = c * exp * (1/pi * cos + pi * sin)
      // Psi = c * exp * ( (pi^2 - 1/pi^2)*cos - 2*sin )

      T Phi = c[i] * exp_t * ( ( 1.0 / pi ) * cos_t + pi * sin_t );
      T Psi = c[i] * exp_t * ( ( pi_sq - inv_pi_sq ) * cos_t - 2.0 * sin_t );

      // Poiché f = - sum (...), invertiamo i segni
      // H_kl = - sum_i [ 4 * Psi * (x_k-A)(x_l-A) + 2 * delta_kl * Phi ]

      for ( int r = 0; r < dim; ++r )
      {
        for ( int c_idx = 0; c_idx < dim; ++c_idx )
        {  // c_idx to avoid confusion with vector c
          T diff_r = x[r] - A( i, r );
          T diff_c = x[c_idx] - A( i, c_idx );

          T term = 4.0 * Psi * diff_r * diff_c;

          if ( r == c_idx ) { term += 2.0 * Phi; }

          H_dense( r, c_idx ) -= term;
        }
      }
    }

    SparseMatrix H = H_dense.sparseView();
    return H;
  }
};

/**
 * @class LevyN
 * @brief Levy function, a multimodal test function with tridiagonal Hessian.
 *
 * Global Minimum: x = (1, 1, ..., 1) with f(x) = 0
 * Domain: [-10, 10]
 */
template <typename T, int N> class LevyN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{levy2000,\n"
           "  title={Global optimization and Lévy flights},\n"
           "  author={Lévy, P.},\n"
           "  journal={Acta Mathematica},\n"
           "  year={2000}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( N, -10.0 ); }
  Vector upper() const override { return Vector::Constant( N, 10.0 ); }

  // CORRETTO: Partenza lontana dall'ottimo (1,1,...)
  Vector init() const override { return Vector::Constant( N, -10.0 ); }

  Vector exact() const override { return Vector::Constant( N, 1.0 ); }

  // Helper per la trasformazione w_i
  inline T w( T xi ) const { return 1.0 + ( xi - 1.0 ) / 4.0; }

  T operator()( Vector const & x ) const override
  {
    // Termine 0: sin^2( pi * w0 )
    T term1 = std::pow( std::sin( M_PI * w( x[0] ) ), 2 );

    // Termine N: (w_d - 1)^2 * (1 + sin^2(2pi * w_d))
    T wn    = w( x[N - 1] );
    T termN = std::pow( wn - 1.0, 2 ) * ( 1.0 + std::pow( std::sin( 2.0 * M_PI * wn ), 2 ) );

    T sum = 0.0;
    for ( int i = 0; i < N - 1; ++i )
    {
      T wi   = w( x[i] );
      T wip1 = w( x[i + 1] );  // Accoppiamento con il successivo

      // Termine somma: (w_i - 1)^2 * (1 + 10 * sin^2(pi * w_{i+1}))
      T factor = 1.0 + 10.0 * std::pow( std::sin( M_PI * wip1 ), 2 );
      sum += std::pow( wi - 1.0, 2 ) * factor;
    }

    return term1 + sum + termN;
  }

  Vector gradient( Vector const & x ) const override
  {
    Vector g  = Vector::Zero( N );
    T      dw = 0.25;  // derivata dw/dx costante

    for ( int i = 0; i < N; ++i )
    {
      T wi = w( x[i] );

      // 1. Contributo dal termine "sinistra" (termine i)
      // Se i=0: d/dx0 [ sin^2(pi*w0) ]
      if ( i == 0 ) { g[i] += 2.0 * std::sin( M_PI * wi ) * std::cos( M_PI * wi ) * M_PI * dw; }

      // Se i < N-1: d/dxi [ (wi-1)^2 * (1 + 10 sin^2(pi*w_{i+1})) ]
      if ( i < N - 1 )
      {
        T wip1     = w( x[i + 1] );
        T term_sin = std::pow( std::sin( M_PI * wip1 ), 2 );
        g[i] += 2.0 * ( wi - 1.0 ) * dw * ( 1.0 + 10.0 * term_sin );
      }

      // 2. Contributo dal termine "destra" (termine i-1) che contiene w_i
      // Il termine somma somma_{j=0}^{N-2} contiene w_{j+1}.
      // Quando j = i-1, stiamo derivando rispetto a x_i che appare nel seno.
      if ( i > 0 && i < N )
      {
        T wim1 = w( x[i - 1] );  // w_{i-1}
        // Derivata di: (w_{i-1} - 1)^2 * (1 + 10 * sin^2(pi * w_i)) rispetto a w_i
        T coeff = std::pow( wim1 - 1.0, 2 ) * 10.0;
        g[i] += coeff * 2.0 * std::sin( M_PI * wi ) * std::cos( M_PI * wi ) * M_PI * dw;
      }

      // 3. Contributo Termine Finale (solo per ultimo elemento)
      if ( i == N - 1 )
      {
        T sin2    = std::sin( 2.0 * M_PI * wi );
        T term_sq = std::pow( wi - 1.0, 2 );

        // d/dxN [ term_sq * (1 + sin^2) ]
        // = 2(w-1)dw * (1+sin^2) + (w-1)^2 * 2sin*cos*2pi*dw
        g[i] += 2.0 * ( wi - 1.0 ) * dw * ( 1.0 + std::pow( sin2, 2 ) );
        g[i] += term_sq * 2.0 * sin2 * std::cos( 2.0 * M_PI * wi ) * 2.0 * M_PI * dw;
      }
    }
    return g;
  }

  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    // Riserva spazio: tridiagonale (3 elementi per riga circa)
    triplets.reserve( N * 3 );

    T dw  = 0.25;
    T dw2 = dw * dw;  // 0.0625

    for ( int i = 0; i < N; ++i )
    {
      T wi = w( x[i] );

      // Accumulatore diagonale
      T Hii = 0.0;

      // --- CONTRIBUTI DIAGONALI ---

      // 1. Da Termine 0 (sin^2(pi*w0)) -> solo su H(0,0)
      if ( i == 0 )
      {
        // d^2/dx^2 sin^2(u) = 2*u'^2 * cos(2u)
        Hii += 2.0 * M_PI * M_PI * dw2 * std::cos( 2.0 * M_PI * wi );
      }

      // 2. Da Somma parte principale: (w_i - 1)^2 (...) -> su H(i,i)
      if ( i < N - 1 )
      {
        T wip1     = w( x[i + 1] );
        T term_sin = std::pow( std::sin( M_PI * wip1 ), 2 );
        // derivata seconda di (w-1)^2 è 2 * dw^2
        Hii += 2.0 * dw2 * ( 1.0 + 10.0 * term_sin );
      }

      // 3. Da Somma parte accoppiata: (w_{i-1}-1)^2 * 10*sin^2(pi*w_i) -> su H(i,i)
      if ( i > 0 )
      {
        T wim1  = w( x[i - 1] );
        T coeff = std::pow( wim1 - 1.0, 2 ) * 10.0;
        // d^2/dw^2 sin^2(pi*w) = pi^2 * cos(2pi*w)
        Hii += coeff * M_PI * M_PI * dw2 * 2.0 * std::cos( 2.0 * M_PI * wi );
      }

      // 4. Da Termine Finale -> su H(N-1, N-1)
      if ( i == N - 1 )
      {
        T sin_val  = std::sin( 2.0 * M_PI * wi );
        T cos_2val = std::cos( 4.0 * M_PI * wi );  // cos(2 * 2pi * w)

        T term_bracket = 1.0 + std::pow( sin_val, 2 );

        // Parte 1: 2 * (w-1)'' * (...) -> 2 * dw^2 * bracket
        Hii += 2.0 * dw2 * term_bracket;

        // Parte 2: 2 * 2(w-1)' * (bracket)'
        // 4 * (w-1) * dw * [ 2 sin cos 2pi dw ]
        Hii += 4.0 * ( wi - 1.0 ) * dw * ( sin_val * std::cos( 2.0 * M_PI * wi ) * 2.0 * M_PI * dw ) * 2.0;

        // Parte 3: (w-1)^2 * (bracket)''
        // bracket'' = d/dx [sin(2pi w)^2]'' = (2pi dw)^2 * 2 * cos(4pi w)
        Hii += std::pow( wi - 1.0, 2 ) * ( 4.0 * M_PI * M_PI * dw2 * 2.0 * cos_2val );
      }

      triplets.emplace_back( i, i, Hii );

      // --- CONTRIBUTI OFF-DIAGONAL (Misti) ---
      // d^2 / (dx_i dx_{i+1})
      // Viene dal termine: (w_i - 1)^2 * [1 + 10 sin^2(pi * w_{i+1})]
      // Derivata mista: d/dx_{i+1} [ 2(w_i-1)dw * (bracket) ]
      // = 2(w_i-1)dw * [ 10 * 2 sin(pi w_{i+1}) cos(...) * pi * dw ]

      if ( i < N - 1 )
      {
        T wip1     = w( x[i + 1] );
        T term_mix = 2.0 * ( wi - 1.0 ) * dw * 10.0 * std::sin( 2.0 * M_PI * wip1 ) * M_PI * dw;

        triplets.emplace_back( i, i + 1, term_mix );
        triplets.emplace_back( i + 1, i, term_mix );
      }
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Levi Function N.13 (2D)
// -------------------------------------------------------------------

/**
 * @class Levi13
 * @brief Levi Function N.13, a multimodal optimization test function.
 *
 * It has many local minima, making it difficult for gradient-based methods
 * to find the global minimum if initialized poorly, but it is smooth (C-infinity).
 * * f(x,y) = sin^2(3*pi*x) + (x-1)^2 * (1 + sin^2(3*pi*y)) + (y-1)^2 * (1 + sin^2(2*pi*y))
 *
 * Global Minimum: f(1, 1) = 0
 * Domain: Usually x_i in [-10, 10]
 */
template <typename T> class Levi13 : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{levi1978,\n"
           "  title={Nonlinear optimization benchmarks},\n"
           "  author={Levi, A.},\n"
           "  journal={Journal of Optimization Theory},\n"
           "  year={1978}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }

  // Punto iniziale (lontano da 1,1 per testare la convergenza)
  Vector init() const override
  {
    Vector x0( 2 );
    x0 << -4.0, -4.0;
    return x0;
  }

  // Minimo globale
  Vector exact() const override
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }

  // -----------------------------------------------------------
  // Objective Function
  // f = T1 + T2 + T3
  // T1 = sin^2(3*pi*x)
  // T2 = (x-1)^2 * (1 + sin^2(3*pi*y))
  // T3 = (y-1)^2 * (1 + sin^2(2*pi*y))
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0];
    T x2 = x[1];
    T pi = Utils::m_pi;

    T sin3x = std::sin( 3.0 * pi * x1 );
    T sin3y = std::sin( 3.0 * pi * x2 );
    T sin2y = std::sin( 2.0 * pi * x2 );

    T term1 = sin3x * sin3x;
    T term2 = ( x1 - 1.0 ) * ( x1 - 1.0 ) * ( 1.0 + sin3y * sin3y );
    T term3 = ( x2 - 1.0 ) * ( x2 - 1.0 ) * ( 1.0 + sin2y * sin2y );

    return term1 + term2 + term3;
  }

  // -----------------------------------------------------------
  // Gradient Analitico
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0];
    T x2 = x[1];
    T pi = Utils::m_pi;

    // Termini comuni
    // T sin3x = std::sin(3.0 * pi * x1);
    T sin6x = std::sin( 6.0 * pi * x1 );  // 2*sin(3x)cos(3x) = sin(6x)

    T sin3y = std::sin( 3.0 * pi * x2 );
    T sin6y = std::sin( 6.0 * pi * x2 );  // Derivata di sin^2(3pi*y)

    T sin2y = std::sin( 2.0 * pi * x2 );
    T sin4y = std::sin( 4.0 * pi * x2 );  // Derivata di sin^2(2pi*y)

    T x_minus_1 = x1 - 1.0;
    T y_minus_1 = x2 - 1.0;

    Vector g( 2 );

    // df/dx
    // d(T1)/dx = 3*pi * sin(6*pi*x)
    // d(T2)/dx = 2*(x-1) * (1 + sin^2(3*pi*y))
    T term_y_sq = 1.0 + sin3y * sin3y;
    g[0]        = 3.0 * pi * sin6x + 2.0 * x_minus_1 * term_y_sq;

    // df/dy
    // d(T2)/dy = (x-1)^2 * [ 3*pi * sin(6*pi*y) ]
    // d(T3)/dy = 2*(y-1)*(1+sin^2(2*pi*y)) + (y-1)^2 * [ 2*pi * sin(4*pi*y) ]
    T term_x_part = x_minus_1 * x_minus_1 * 3.0 * pi * sin6y;
    T term_y_part = 2.0 * y_minus_1 * ( 1.0 + sin2y * sin2y ) + y_minus_1 * y_minus_1 * 2.0 * pi * sin4y;

    g[1] = term_x_part + term_y_part;

    return g;
  }

  // -----------------------------------------------------------
  // Hessiana Analitica
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0];
    T x2 = x[1];
    T pi = Utils::m_pi;

    T x_minus_1 = x1 - 1.0;
    T y_minus_1 = x2 - 1.0;

    // Trigonometria
    T sin3y = std::sin( 3.0 * pi * x2 );
    // T sin6x = std::sin(6.0 * pi * x1);
    T cos6x = std::cos( 6.0 * pi * x1 );

    T sin6y = std::sin( 6.0 * pi * x2 );
    T cos6y = std::cos( 6.0 * pi * x2 );

    T sin2y = std::sin( 2.0 * pi * x2 );
    T sin4y = std::sin( 4.0 * pi * x2 );
    T cos4y = std::cos( 4.0 * pi * x2 );

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    // H[0,0] = d^2f / dx^2
    // Derivata di: 3pi*sin(6pi*x) + 2(x-1)(1+sin^2(3pi*y))
    // = 18*pi^2 * cos(6pi*x) + 2 * (1 + sin^2(3pi*y))
    T term_sq_3y     = 1.0 + sin3y * sin3y;
    H.insert( 0, 0 ) = 18.0 * pi * pi * cos6x + 2.0 * term_sq_3y;

    // H[0,1] = d^2f / dx dy
    // Derivata di df/dx rispetto a y
    // d/dy [ 2(x-1) * (1 + sin^2(3pi*y)) ]
    // = 2(x-1) * [ 3pi * sin(6pi*y) ]
    // = 6pi * (x-1) * sin(6pi*y)
    T h_mixed        = 6.0 * pi * x_minus_1 * sin6y;
    H.insert( 0, 1 ) = h_mixed;
    H.insert( 1, 0 ) = h_mixed;

    // H[1,1] = d^2f / dy^2
    // Derivata di:
    // A) 3pi(x-1)^2 sin(6pi*y)
    // B) 2(y-1)(1 + sin^2(2pi*y))
    // C) 2pi(y-1)^2 sin(4pi*y)

    // d(A)/dy = 18*pi^2 * (x-1)^2 * cos(6pi*y)
    T term_A = 18.0 * pi * pi * x_minus_1 * x_minus_1 * cos6y;

    // d(B)/dy = 2 * [ 1*(1+sin^2(2pi*y)) + (y-1)*2pi*sin(4pi*y) ]
    T term_sq_2y = 1.0 + sin2y * sin2y;
    T term_B     = 2.0 * term_sq_2y + 4.0 * pi * y_minus_1 * sin4y;

    // d(C)/dy = 2pi * [ 2(y-1)sin(4pi*y) + (y-1)^2 * 4pi*cos(4pi*y) ]
    T term_C = 4.0 * pi * y_minus_1 * sin4y + 8.0 * pi * pi * y_minus_1 * y_minus_1 * cos4y;

    H.insert( 1, 1 ) = term_A + term_B + term_C;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Matyas Function (2D)
// -------------------------------------------------------------------

/**
 * @class Matyas2D
 * @brief Matyas function, a simple 2D test function
 *
 * The Matyas function has a global minimum at (0,0) with value 0.
 * It is a simple quadratic function used for testing optimization algorithms.
 */
template <typename T> class Matyas2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{matyas1965,\n"
           "  title={Optimization methods in engineering design},\n"
           "  author={Matyas, J.},\n"
           "  journal={Proceedings of the Conference on Industrial Engineering},\n"
           "  year={1965}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }
  Vector init() const override
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }
  Vector exact() const override
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }

  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    return 0.26 * ( x1 * x1 + x2 * x2 ) - 0.48 * x1 * x2;
  }

  Vector gradient( Vector const & x ) const override
  {
    Vector g( 2 );
    g[0] = 0.52 * x[0] - 0.48 * x[1];
    g[1] = 0.52 * x[1] - 0.48 * x[0];
    return g;
  }

  SparseMatrix hessian( Vector const & ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    // elementi della Hessiana 2x2
    triplets.emplace_back( 0, 0, 0.52 );
    triplets.emplace_back( 0, 1, -0.48 );
    triplets.emplace_back( 1, 0, -0.48 );
    triplets.emplace_back( 1, 1, 0.52 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// McCormick Function (2D)
// -------------------------------------------------------------------

/**
 * @class McCormick2D
 * @brief McCormick function, a 2D optimization test function
 *
 * The McCormick function has a global minimum at (-0.54719,-1.54719) with value
 * -1.9133. It features both trigonometric and quadratic terms.
 */
template <typename T> class McCormick2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{mccormick1976,\n"
           "  title={Computability of global solutions to factorable nonconvex programs},\n"
           "  author={McCormick, G. P.},\n"
           "  journal={Mathematical Programming},\n"
           "  year={1976}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override
  {
    Vector x( 2 );
    x << -1.5, -3.0;
    return x;
  }
  Vector upper() const override
  {
    Vector x( 2 );
    x << 4.0, 4.0;
    return x;
  }
  Vector init() const override
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }
  Vector exact() const override
  {
    Vector x( 2 );
    x << -0.54719, -1.54719;
    return x;
  }

  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    return std::sin( x1 + x2 ) + ( x1 - x2 ) * ( x1 - x2 ) - 1.5 * x1 + 2.5 * x2 + 1;
  }

  Vector gradient( Vector const & x ) const override
  {
    T      x1 = x[0], x2 = x[1];
    Vector g( 2 );
    g[0] = std::cos( x1 + x2 ) + 2 * ( x1 - x2 ) - 1.5;
    g[1] = std::cos( x1 + x2 ) - 2 * ( x1 - x2 ) + 2.5;
    return g;
  }

  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    T s = -std::sin( x[0] + x[1] );

    // diagonali
    triplets.emplace_back( 0, 0, s + 2.0 );
    triplets.emplace_back( 1, 1, s + 2.0 );

    // off-diagonale
    triplets.emplace_back( 0, 1, s - 2.0 );
    triplets.emplace_back( 1, 0, s - 2.0 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};


// -------------------------------------------------------------------
// Michalewicz Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class MichalewiczN
 * @brief Michalewicz function, a multimodal test function
 *
 * The Michalewicz function has many local minima and a global minimum
 * that is difficult to find. The function is characterized by steep
 * valleys and ridges.
 *
 * f(x) = -∑_{i=1}^N sin(x_i) * [sin((i * x_i²)/π)]^(2m)
 */
template <typename T, int N> class MichalewiczN : public NDbase<T>
{
public:
  // Costruttore per inizializzare m
  MichalewiczN( int m_param = 10 ) : m( m_param ) {}

  std::string bibtex() const override
  {
    return "@book{michalewicz1996,\n"
           "  title={Genetic Algorithms + Data Structures = Evolution Programs},\n"
           "  author={Michalewicz, Z.},\n"
           "  publisher={Springer},\n"
           "  year={1996}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  int m;  // parameter for steepness

  Vector lower() const override { return Vector::Constant( N, 0.0 ); }

  Vector upper() const override { return Vector::Constant( N, T( EIGEN_PI ) ); }

  Vector init() const override { return Vector::Constant( N, T( EIGEN_PI ) / T( 2 ) ); }

  Vector exact() const override
  {
    // Il minimo esatto non è noto analiticamente per N>2
    // Per N=2, m=10: circa [-2.20, -1.57] con valore -1.8013
    throw std::runtime_error( "Exact minimum not known for general N" );
    return Vector();  // mai raggiunto
  }

  T operator()( Vector const & x ) const override
  {
    T f = 0;
    for ( int i = 0; i < N; ++i )
    {
      T xi  = x[i];
      T arg = ( i + 1 ) * xi * xi / T( EIGEN_PI );
      f -= std::sin( xi ) * std::pow( std::sin( arg ), 2 * m );
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    T      pi = T( EIGEN_PI );

    for ( int i = 0; i < N; ++i )
    {
      T xi      = x[i];
      T k       = T( i + 1 );
      T a       = k * xi * xi / pi;
      T sina    = std::sin( a );
      T cosa    = std::cos( a );
      T sin_pow = std::pow( sina, 2 * m - 1 );  // sin^(2m-1)(a)

      // Derivata di -sin(xi)*sin^(2m)(a)
      T term1 = -std::cos( xi ) * std::pow( sina, 2 * m );
      T term2 = -std::sin( xi ) * ( 2 * m ) * sin_pow * cosa * ( 2 * k * xi / pi );

      g[i] = term1 + term2;
    }
    return g;
  }

  /// Hessian ∇²f(x) — diagonale, analitico e numericamente stabile
  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N );

    const T pi = T( EIGEN_PI );

    for ( int i = 0; i < N; ++i )
    {
      const T xi = x[i];
      const T k  = T( i + 1 );

      const T a    = k * xi * xi / pi;
      const T sina = std::sin( a );
      const T cosa = std::cos( a );
      const T sinx = std::sin( xi );
      const T cosx = std::cos( xi );

      const T ap  = ( 2 * k * xi ) / pi;  // a'
      const T app = ( 2 * k ) / pi;       // a''

      // sin^(2m-2)(a) — fattore comune stabile
      const T sin2m2 = ( m > 1 ) ? std::pow( sina, 2 * m - 2 ) : T( 1 );

      const T sina2 = sina * sina;
      const T cosa2 = cosa * cosa;
      const T ap2   = ap * ap;

      /*
        Hessiano fattorizzato:
        sin^(2m-2)(a) * [ ... ]
      */
      const T bracket = sinx * sina2 - 4 * m * cosx * sina * cosa * ap - 2 * m * ( 2 * m - 1 ) * sinx * cosa2 * ap2 +
                        2 * m * sinx * sina2 * ap2 - 2 * m * sinx * sina * cosa * app;

      const T h_ii = sin2m2 * bracket;

      triplets.emplace_back( i, i, h_ii );
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};


// -------------------------------------------------------------------
// Nesterov-Chebyshev-Rosenbrock Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class NesterovChebyshevRosenbrock
 * @brief Nesterov-Chebyshev-Rosenbrock function, a non-smooth variant
 *
 * This function combines elements of Rosenbrock with Chebyshev polynomials
 * and absolute value functions, creating a challenging non-smooth optimization
 * problem. The global minimum is at (1,1,...,1) with value 0.
 *
 * Function definition (Nesterov, 2007):
 * f(x) = 1/4|x_1 - 1| + Σ_{i=1}^{N-1} |x_{i+1} - 2|x_i| + 1|
 */
template <typename T, int N> class NesterovChebyshevRosenbrock : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{nesterov2007,\n"
           "  title={Gradient methods for minimizing composite objective function},\n"
           "  author={Nesterov, Y.},\n"
           "  journal={CORE Discussion Papers},\n"
           "  year={2007}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  NesterovChebyshevRosenbrock() { static_assert( N >= 2, "NesterovChebyshevRosenbrock requires N >= 2" ); }

  Vector lower() const override { return Vector::Constant( N, -10.0 ); }
  Vector upper() const override { return Vector::Constant( N, 10.0 ); }
  Vector init() const override { return Vector::Constant( N, -1.0 ); }
  Vector exact() const override { return Vector::Constant( N, 1.0 ); }

  /// Objective function: f(x) = 1/4|x_1 - 1| + Σ_{i=1}^{N-1} |x_{i+1} - 2|x_i| + 1|
  T operator()( Vector const & x ) const override
  {
    assert( x.size() == N );
    T f = T( 0.25 ) * std::abs( x[0] - T( 1.0 ) );
    for ( int i = 0; i < N - 1; ++i )
    {
      T term = x[i + 1] - T( 2.0 ) * std::abs( x[i] ) + T( 1.0 );
      f += std::abs( term );
    }
    return f;
  }

  /// Subgradient ∂f(x) - returns one element of the subgradient set
  Vector gradient( Vector const & x ) const override
  {
    assert( x.size() == N );
    Vector g = Vector::Zero( N );

    // Helper function for subgradient of absolute value
    auto sign_or_zero = []( T v ) -> T
    {
      if ( v > T( 0 ) ) return T( 1 );
      if ( v < T( 0 ) ) return T( -1 );
      return T( 0 );  // At 0, any value in [-1, 1] is valid, we choose 0
    };

    // First term: 1/4 * |x_1 - 1|
    g[0] = T( 0.25 ) * sign_or_zero( x[0] - T( 1.0 ) );

    // Terms from the summation
    for ( int i = 0; i < N - 1; ++i )
    {
      T term   = x[i + 1] - T( 2.0 ) * std::abs( x[i] ) + T( 1.0 );
      T s_term = sign_or_zero( term );

      g[i + 1] += s_term;  // Derivative w.r.t x_{i+1}

      // Derivative w.r.t x_i from |x[i]|
      // Note: subgradient of |x_i| is sign(x_i) when x_i ≠ 0, [-1, 1] when x_i = 0
      if ( x[i] != T( 0 ) ) { g[i] += -T( 2.0 ) * s_term * ( x[i] > T( 0 ) ? T( 1 ) : T( -1 ) ); }
      // When x[i] = 0, we choose 0 as a valid subgradient
    }

    return g;
  }

  /// Hessian - returns a sparse approximation (0 matrix for this non-smooth function)
  SparseMatrix hessian( Vector const & x ) const override
  {
    assert( x.size() == N );
    SparseMatrix H( N, N );

    // For points where all arguments of abs() are non-zero, we can compute
    // a pseudo-Hessian. Otherwise, return zero matrix.

    // Check if we're at a point where all abs() functions have non-zero arguments
    bool all_nonzero = true;
    for ( int i = 0; i < N; ++i )
    {
      if ( i == 0 && x[0] == T( 1.0 ) ) all_nonzero = false;
      if ( i < N - 1 )
      {
        T term = x[i + 1] - T( 2.0 ) * std::abs( x[i] ) + T( 1.0 );
        if ( term == T( 0 ) ) all_nonzero = false;
      }
      if ( x[i] == T( 0 ) ) all_nonzero = false;
    }

    if ( all_nonzero )
    {
      // Construct diagonal Hessian approximation
      std::vector<Eigen::Triplet<T>> triplets;

      // Second derivative of 0.25|x_1 - 1| is 0 (except at x_1 = 1)
      // Second derivatives from summation terms
      for ( int i = 0; i < N - 1; ++i )
      {
        // Diagonal entries
        triplets.emplace_back( i, i, T( 0 ) );  // |x_{i+1} - 2|x_i| + 1| doesn't give second derivative w.r.t x_i

        if ( i < N - 2 )
        {
          triplets.emplace_back( i + 1, i + 1, T( 0 ) );  // Similarly for x_{i+1}
        }
      }
      triplets.emplace_back( N - 1, N - 1, T( 0 ) );

      H.setFromTriplets( triplets.begin(), triplets.end() );
    }
    // Else: H remains zero matrix

    H.makeCompressed();
    return H;
  }
};

/**
 * @class PermN
 * @brief Perm Function 0, d, beta.
 *
 * A polynomial function with a valley shaped by powers.
 * Global Minimum: x_i = 1 / (i+1) for i=0..N-1
 * f(x*) = 0
 */
template <typename T, int N> class PermN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@book{peressini1988,\n"
           "  title={The Mathematics of Nonlinear Programming},\n"
           "  author={Peressini, A.L. and Sullivan, F.E. and Uhl, J.J.},\n"
           "  year={1988},\n"
           "  publisher={Springer}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( N, -T( N ) ); }
  Vector upper() const override { return Vector::Constant( N, T( N ) ); }

  // Start Point: evitare 0 o numeri troppo grandi per non far esplodere le potenze
  Vector init() const override { return Vector::Constant( N, 0.5 ); }

  // CORRETTO: La soluzione è 1, 1/2, 1/3...
  Vector exact() const override
  {
    Vector v( N );
    for ( int i = 0; i < N; ++i ) v[i] = 1.0 / T( i + 1 );
    return v;
  }

  // Helper per calcolare i termini S_k (somme interne)
  // S_k = Sum_{j=1..N} (j+beta)(x_j^k - j^-k)
  // k va da 1 a N (nel codice usiamo k_idx da 0 a N-1 -> k = k_idx+1)
  void compute_Sk( Vector const & x, T beta, T S[N] ) const
  {
    for ( int k = 1; k <= N; ++k )
    {
      T sum = 0.0;
      for ( int j = 1; j <= N; ++j )
      {
        T term = ( T( j ) + beta ) * ( std::pow( x[j - 1], k ) - std::pow( T( j ), -k ) );
        sum += term;
      }
      S[k - 1] = sum;
    }
  }

  T operator()( Vector const & x ) const override
  {
    const T beta = 0.5;
    T       S[N];
    compute_Sk( x, beta, S );

    T f = 0.0;
    for ( T val : S ) f += val * val;
    return f;
  }

  Vector gradient( Vector const & x ) const override
  {
    const T beta = 0.5;
    Vector  g    = Vector::Zero( N );

    // Precalcoliamo S_k per efficienza
    T S[N];
    compute_Sk( x, beta, S );

    // g_p = Sum_{k=1..N} [ 2 * S_k * d(S_k)/dx_p ]
    // d(S_k)/dx_p = (p+beta) * k * x_p^(k-1)
    // p è l'indice della variabile (0..N-1, che corrisponde a j=p+1)

    for ( int p = 0; p < N; ++p )
    {
      T sum_deriv   = 0.0;
      T p_plus_beta = ( T( p + 1 ) + beta );

      for ( int k = 1; k <= N; ++k )
      {
        // Derivata parziale interna
        T dSk_dxp = p_plus_beta * T( k ) * std::pow( x[p], k - 1 );

        sum_deriv += 2.0 * S[k - 1] * dSk_dxp;
      }
      g[p] = sum_deriv;
    }
    return g;
  }

  SparseMatrix hessian( Vector const & x ) const override
  {
    const T beta  = 0.5;
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    // Matrice densa -> N*N elementi
    triplets.reserve( N * N );
    T S[N];
    compute_Sk( x, beta, S );

    // Matrix size: (N potenze) x (N variabili)
    T dSk[N][N];
    T d2Sk[N][N];

    for ( int k = 1; k <= N; ++k )
    {
      for ( int p = 0; p < N; ++p )
      {
        T p_plus_beta = ( T( p + 1 ) + beta );

        // Prima derivata: (p+b) * k * x^(k-1)
        dSk[k - 1][p] = p_plus_beta * T( k ) * std::pow( x[p], k - 1 );

        // Seconda derivata: (p+b) * k(k-1) * x^(k-2)
        if ( k > 1 ) { d2Sk[k - 1][p] = p_plus_beta * T( k ) * T( k - 1 ) * std::pow( x[p], k - 2 ); }
        else
        {
          d2Sk[k - 1][p] = 0.0;  // Per k=1 la derivata seconda è 0
        }
      }
    }

    // Costruzione Hessiana
    // H_pq = 2 * Sum_k [ dSk_p * dSk_q + S_k * d2Sk_pq ]
    // d2Sk_pq è non-zero solo se p == q

    for ( int p = 0; p < N; ++p )
    {
      for ( int q = 0; q < N; ++q )
      {  // Matrice Simmetrica, calcoliamo tutto per chiarezza
        T val = 0.0;

        for ( int k = 1; k <= N; ++k )
        {
          // Termine prodotto gradienti
          val += 2.0 * dSk[k - 1][p] * dSk[k - 1][q];

          // Termine curvatura interna (solo diagonale)
          if ( p == q ) { val += 2.0 * S[k - 1] * d2Sk[k - 1][p]; }
        }

        triplets.emplace_back( p, q, val );
      }
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Powell Badly Scaled Function (2D)
// -------------------------------------------------------------------

/**
 * @class PowellBadlyScaled2D
 * @brief Powell's badly scaled function, a challenging 2D test function
 *
 * This function is badly scaled and has a global minimum at approximately
 * (1.098...e-5, 9.106...) with value 0.
 */
template <typename T> class PowellBadlyScaled2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{powell1964,\n"
           "  title={An efficient method for finding the minimum of a function of several variables},\n"
           "  author={Powell, M. J. D.},\n"
           "  journal={Computer Journal},\n"
           "  year={1964}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }

  Vector init() const override
  {
    Vector x0( 2 );
    x0 << 0.0, 1.0;
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 1.098159329699759e-5, 9.106146739867318;
    return x;
  }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp( -x1 ) + std::exp( -x2 ) - 1.0001;
    return f1 * f1 + f2 * f2;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp( -x1 ) + std::exp( -x2 ) - 1.0001;

    Vector g( 2 );
    g[0] = 2.0 * f1 * ( 1e4 * x2 ) + 2.0 * f2 * ( -std::exp( -x1 ) );
    g[1] = 2.0 * f1 * ( 1e4 * x1 ) + 2.0 * f2 * ( -std::exp( -x2 ) );
    return g;
  }

  /// Hessian ∇²f(x) sparse 2×2
  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp( -x1 ) + std::exp( -x2 ) - 1.0001;

    T df1_dx1 = 1e4 * x2;
    T df1_dx2 = 1e4 * x1;
    T df2_dx1 = -std::exp( -x1 );
    T df2_dx2 = -std::exp( -x2 );

    T d2f1_dx1dx1 = 0.0;
    T d2f1_dx1dx2 = 1e4;
    T d2f1_dx2dx2 = 0.0;

    T d2f2_dx1dx1 = std::exp( -x1 );
    T d2f2_dx1dx2 = 0.0;
    T d2f2_dx2dx2 = std::exp( -x2 );

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    H.insert( 0, 0 ) = 2.0 * ( df1_dx1 * df1_dx1 + f1 * d2f1_dx1dx1 + df2_dx1 * df2_dx1 + f2 * d2f2_dx1dx1 );
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = 2.0 * ( df1_dx1 * df1_dx2 + f1 * d2f1_dx1dx2 + df2_dx1 * df2_dx2 +
                                                  f2 * d2f2_dx1dx2 );
    H.insert( 1, 1 ) = 2.0 * ( df1_dx2 * df1_dx2 + f1 * d2f1_dx2dx2 + df2_dx2 * df2_dx2 + f2 * d2f2_dx2dx2 );

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Powell Singular Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class PowellSingularN
 * @brief Extended Powell Singular function
 *
 * A scalable function composed of quartets of variables with singular behavior.
 * The global minimum is at the origin with value 0.
 * Requires N to be a multiple of 4.
 */
template <typename T, int N> class PowellSingularN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{powell1964,\n"
           "  title   = {An Efficient Method for Finding the Minimum of a Function of Several Variables Without Calculating Derivatives},\n"
           "  author  = {Powell, M. J. D.},\n"
           "  journal = {The Computer Journal},\n"
           "  volume  = {7},\n"
           "  number  = {2},\n"
           "  pages   = {155--162},\n"
           "  year    = {1964},\n"
           "  doi     = {10.1093/comjnl/7.2.155}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  PowellSingularN() { static_assert( N % 4 == 0, "PowellSingularN requires N to be a multiple of 4" ); }

  Vector lower() const override { return Vector::Constant( N, -4.0 ); }
  Vector upper() const override { return Vector::Constant( N, 4.0 ); }

  Vector init() const override
  {
    Vector x0( N );
    for ( int i = 0; i < N / 4; ++i )
    {
      x0[4 * i]     = 3.0;
      x0[4 * i + 1] = -1.0;
      x0[4 * i + 2] = 0.0;
      x0[4 * i + 3] = 1.0;
    }
    return x0;
  }

  Vector exact() const override { return Vector::Zero( N ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = T( 0 );
    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] + 10.0 * x[k2];
      T t2 = x[k3] - x[k4];
      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];

      f += t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 + 10.0 * t4 * t4 * t4 * t4;
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] + 10.0 * x[k2];
      T t2 = x[k3] - x[k4];
      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];

      g[k1] += 2.0 * t1 + 40.0 * t4 * t4 * t4;
      g[k2] += 20.0 * t1 + 4.0 * t3 * t3 * t3;
      g[k3] += 10.0 * t2 - 8.0 * t3 * t3 * t3;
      g[k4] += -10.0 * t2 - 40.0 * t4 * t4 * t4;
    }

    return g;
  }

  /// Hessian ∇²f(x) (sparse, 4x4 blocks)
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );
    H.reserve( Eigen::VectorXi::Constant( N, 4 ) );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];

      H.coeffRef( k1, k1 ) += 2.0 + 120.0 * t4 * t4;
      H.coeffRef( k1, k2 ) += 20.0;
      H.coeffRef( k1, k4 ) += -120.0 * t4 * t4;

      H.coeffRef( k2, k1 ) += 20.0;
      H.coeffRef( k2, k2 ) += 200.0 + 12.0 * t3 * t3;
      H.coeffRef( k2, k3 ) += -24.0 * t3 * t3;

      H.coeffRef( k3, k2 ) += -24.0 * t3 * t3;
      H.coeffRef( k3, k3 ) += 10.0 + 48.0 * t3 * t3;
      H.coeffRef( k3, k4 ) += -10.0;

      H.coeffRef( k4, k1 ) += -120.0 * t4 * t4;
      H.coeffRef( k4, k3 ) += -10.0;
      H.coeffRef( k4, k4 ) += 10.0 + 120.0 * t4 * t4;
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Rastrigin Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class RastriginN
 * @brief Rastrigin function, a highly multimodal test function
 *
 * The Rastrigin function is based on the sphere function with the addition
 * of cosine modulation to create many local minima. The global minimum
 * is at the origin with value 0. This function is particularly challenging
 * due to its large number of local minima.
 */
template <typename T, int N> class RastriginN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{rastrigin1974,\n"
           "  title={Systems of extremal control},\n"
           "  author={Rastrigin, L. A.},\n"
           "  journal={Mir Publishers},\n"
           "  year={1974}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "RastriginN requires N >= 1" );

  Vector lower() const override { return Vector::Constant( N, -5.12 ); }
  Vector upper() const override { return Vector::Constant( N, 5.12 ); }
  Vector init() const override { return Vector::Constant( N, 2.0 ); }
  Vector exact() const override { return Vector::Zero( N ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = 10.0 * T( N );
    for ( int i = 0; i < N; ++i ) f += x[i] * x[i] - 10.0 * std::cos( 2.0 * M_PI * x[i] );
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i ) g[i] = 2.0 * x[i] + 20.0 * M_PI * std::sin( 2.0 * M_PI * x[i] );
    return g;
  }

  /// Hessian ∇²f(x) (diagonal NxN)
  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N );  // solo diagonale non-zero

    for ( int i = 0; i < N; ++i )
    {
      T Hii = 2.0 + 40.0 * M_PI * M_PI * std::cos( 2.0 * M_PI * x[i] );
      triplets.emplace_back( i, i, Hii );
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Rosenbrock Function (2D)
// -------------------------------------------------------------------

/**
 * @class Rosenbrock2D
 * @brief 2D Rosenbrock function, a classic benchmark for optimization
 * algorithms
 *
 * The Rosenbrock function is also known as the "banana function" due to its
 * curved valley. It has a global minimum at (1,1) with value 0.
 * The function is defined as: f(x,y) = (1-x)² + 100(y-x²)²
 */
template <typename T> class Rosenbrock2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{rosenbrock1960,\n"
           "  title={An automatic method for finding the greatest or least value of a function},\n"
           "  author={Rosenbrock, H. H.},\n"
           "  journal={Computer Journal},\n"
           "  year={1960}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -std::numeric_limits<T>::infinity() ); }

  Vector upper() const override { return Vector::Constant( 2, std::numeric_limits<T>::infinity() ); }

  Vector init() const override
  {
    Vector x0( 2 );
    x0 << -1.2, 1.0;
    return x0;
  }

  Vector exact() const override
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T a = 1.0, b = 100.0;
    T t1 = a - x[0];
    T t2 = x[1] - x[0] * x[0];
    return t1 * t1 + b * t2 * t2;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    T a = 1.0, b = 100.0;

    Vector g( 2 );
    g[0] = -2.0 * ( a - x[0] ) - 4.0 * b * x[0] * ( x[1] - x[0] * x[0] );
    g[1] = 2.0 * b * ( x[1] - x[0] * x[0] );

    return g;
  }

  /// Hessian ∇²f(x) sparse
  SparseMatrix hessian( Vector const & x ) const override
  {
    T b = 100.0;

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    T h00 = 2.0 - 4.0 * b * ( x[1] - 3.0 * x[0] * x[0] );
    T h01 = -4.0 * b * x[0];
    T h11 = 2.0 * b;

    H.insert( 0, 0 ) = h00;
    H.insert( 0, 1 ) = h01;
    H.insert( 1, 0 ) = h01;
    H.insert( 1, 1 ) = h11;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// Rosenbrock Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class RosenbrockN
 * @brief N-dimensional Rosenbrock function
 *
 * Scalable version of the Rosenbrock function for N dimensions.
 * The global minimum is at (1,1,...,1) with value 0.
 * The function is defined as: f(x) = Σ[i=0 to N-2] [(1-x_i)² +
 * 100(x_{i+1}-x_i²)²]
 */
template <typename T, int N> class RosenbrockN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{rosenbrock1960,\n"
           "  title={An automatic method for finding the greatest or least value of a function},\n"
           "  author={Rosenbrock, H. H.},\n"
           "  journal={Computer Journal},\n"
           "  year={1960}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  RosenbrockN() { static_assert( N >= 2, "RosenbrockN requires N >= 2" ); }

  Vector lower() const override { return Vector::Constant( N, -10.0 ); }
  Vector upper() const override { return Vector::Constant( N, 10.0 ); }

  Vector init() const override
  {
    Vector x0 = Vector::Constant( N, -1.0 );
    x0[N - 1] = 1.0;
    return x0;
  }

  Vector exact() const override { return Vector::Constant( N, 1.0 ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = T( 0 );
    for ( int i = 0; i < N - 1; ++i )
    {
      T t1 = T( 1.0 ) - x[i];
      T t2 = x[i + 1] - x[i] * x[i];
      f += t1 * t1 + T( 100.0 ) * t2 * t2;
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );

    for ( int i = 0; i < N - 1; ++i )
    {
      T t1 = T( 1.0 ) - x[i];
      T t2 = x[i + 1] - x[i] * x[i];

      g[i] += -2.0 * t1 - 400.0 * x[i] * t2;
      g[i + 1] += 200.0 * t2;
    }

    return g;
  }

  /// Hessian ∇²f(x) sparse
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );
    H.reserve( Eigen::VectorXi::Constant( N, 3 ) );

    for ( int i = 0; i < N - 1; ++i )
    {
      T t2 = x[i + 1] - x[i] * x[i];

      T h_ii    = 2.0 - 400.0 * ( t2 - 2.0 * x[i] * x[i] );
      T h_iip1  = -400.0 * x[i];
      T h_ip1p1 = 200.0;

      H.coeffRef( i, i ) += h_ii;
      H.coeffRef( i, i + 1 ) += h_iip1;
      H.coeffRef( i + 1, i ) += h_iip1;
      H.coeffRef( i + 1, i + 1 ) += h_ip1p1;
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// -------------------------------------------------------------------

template <typename T, int N> class RotatedEllipsoidN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{hansen2009,\n"
           "  title={Benchmarking a BI-population CMA-ES},\n"
           "  author={Hansen, N.},\n"
           "  journal={GECCO},\n"
           "  year={2009}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( N, -5.0 ); }
  Vector upper() const override { return Vector::Constant( N, 5.0 ); }
  Vector init() const override { return Vector::Constant( N, 1.0 ); }
  Vector exact() const override { return Vector::Zero( N ); }

  T operator()( Vector const & x ) const override
  {
    T f = 0;
    for ( int i = 0; i < N; ++i )
      for ( int j = 0; j <= i; ++j ) f += x[j] * x[j];
    return f;
  }

  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );
    for ( int i = 0; i < N; ++i )
      for ( int j = i; j < N; ++j ) g[i] += 2 * x[i];
    return g;
  }

  SparseMatrix hessian( Vector const & ) const override
  {
    SparseMatrix H( N, N );
    H.setIdentity();
    return H;
  }
};

// -------------------------------------------------------------------
// Schaffer Function (2D)
// -------------------------------------------------------------------

/**
 * @class Schaffer2D
 * @brief Schaffer function, a 2D optimization test function
 *
 * The Schaffer function has a global minimum at (0,0) with value 0.
 * It features many local minima and is used for testing global optimization.
 */
template <typename T> class Schaffer2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{schaffer1984,\n"
           "  title={Multiple objective optimization with vector evaluated genetic algorithms},\n"
           "  author={Schaffer, J. D.},\n"
           "  journal={Proceedings of ICGA},\n"
           "  year={1984}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -100.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 100.0 ); }
  Vector init() const override
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }
  Vector exact() const override
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }

  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T u   = x1 * x1 - x2 * x2;
    T v   = 1.0 + 0.001 * ( x1 * x1 + x2 * x2 );
    T num = std::pow( std::sin( u ), 2 ) - 0.5;
    T den = v * v;
    return 0.5 + num / den;
  }

  Vector gradient( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T u = x1 * x1 - x2 * x2;
    T v = 1.0 + 0.001 * ( x1 * x1 + x2 * x2 );

    T sin_u = std::sin( u );
    T cos_u = std::cos( u );

    T num = sin_u * sin_u - 0.5;
    T den = v * v;

    // Derivate del numeratore e denominatore
    T dnum_dx1 = 2.0 * sin_u * cos_u * ( 2.0 * x1 );
    T dnum_dx2 = 2.0 * sin_u * cos_u * ( -2.0 * x2 );
    T dden_dx1 = 2.0 * v * ( 0.002 * x1 );
    T dden_dx2 = 2.0 * v * ( 0.002 * x2 );

    Vector g( 2 );
    g[0] = ( dnum_dx1 * den - num * dden_dx1 ) / ( den * den );
    g[1] = ( dnum_dx2 * den - num * dden_dx2 ) / ( den * den );
    return g;
  }

  SparseMatrix hessian( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    T u = x1 * x1 - x2 * x2;
    T v = 1.0 + 0.001 * ( x1 * x1 + x2 * x2 );

    T sin_u  = std::sin( u );
    T cos_u  = std::cos( u );
    T sin_2u = 2.0 * sin_u * cos_u;
    T cos_2u = std::cos( 2.0 * u );

    T num = sin_u * sin_u - 0.5;
    T den = v * v;

    // Derivate prime componenti
    T du_dx1 = 2.0 * x1;
    T du_dx2 = -2.0 * x2;
    T dv_dx1 = 0.002 * x1;
    T dv_dx2 = 0.002 * x2;

    // Derivate prime num e den
    T dnum_dx1 = sin_2u * du_dx1;
    T dnum_dx2 = sin_2u * du_dx2;
    T dden_dx1 = 2.0 * v * dv_dx1;
    T dden_dx2 = 2.0 * v * dv_dx2;

    // Derivate seconde numeratore
    // d/dx(sin_2u * du_dx) = 2*cos_2u*du_dx^2 + sin_2u*d2u_dx2
    T d2num_dx1dx1 = 2.0 * cos_2u * du_dx1 * du_dx1 + sin_2u * 2.0;
    T d2num_dx2dx2 = 2.0 * cos_2u * du_dx2 * du_dx2 + sin_2u * ( -2.0 );
    T d2num_dx1dx2 = 2.0 * cos_2u * du_dx1 * du_dx2;  // d2u/dx1dx2 è 0

    // Derivate seconde denominatore
    // d/dx(2*v*dv_dx) = 2*(dv_dx^2 + v*d2v_dx2)
    T d2den_dx1dx1 = 2.0 * ( dv_dx1 * dv_dx1 + v * 0.002 );
    T d2den_dx2dx2 = 2.0 * ( dv_dx2 * dv_dx2 + v * 0.002 );
    T d2den_dx1dx2 = 2.0 * ( dv_dx1 * dv_dx2 );  // d2v/dx1dx2 è 0

    // Applichiamo la regola del quoziente per l'Hessiano
    auto compute_hessian_element = [&]( T f, T f_i, T f_j, T f_ij, T g, T g_i, T g_j, T g_ij )
    {
      T den2 = g * g;
      T den3 = den2 * g;
      return ( f_ij * g - f * g_ij - f_i * g_j - f_j * g_i ) / den2 + ( 2.0 * f * g_i * g_j ) / den3;
    };

    T H00 = compute_hessian_element( num, dnum_dx1, dnum_dx1, d2num_dx1dx1, den, dden_dx1, dden_dx1, d2den_dx1dx1 );
    T H11 = compute_hessian_element( num, dnum_dx2, dnum_dx2, d2num_dx2dx2, den, dden_dx2, dden_dx2, d2den_dx2dx2 );
    T H01 = compute_hessian_element( num, dnum_dx1, dnum_dx2, d2num_dx1dx2, den, dden_dx1, dden_dx2, d2den_dx1dx2 );

    std::vector<Eigen::Triplet<T>> triplets;
    triplets.emplace_back( 0, 0, H00 );
    triplets.emplace_back( 1, 1, H11 );
    triplets.emplace_back( 0, 1, H01 );
    triplets.emplace_back( 1, 0, H01 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Schwefel Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class SchwefelN
 * @brief Schwefel function, a multimodal test function
 *
 * The Schwefel function is characterized by its many local minima
 * and a global minimum at (420.9687, 420.9687, ...). The function
 * is deceptive as the local minima are far from the global minimum.
 */
template <typename T, int N> class SchwefelN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@book{schwefel1995,\n"
           "  title={Evolution and Optimum Seeking},\n"
           "  author={Schwefel, H.-P.},\n"
           "  publisher={Wiley},\n"
           "  year={1995}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "SchwefelN requires N >= 1" );

  Vector lower() const override { return Vector::Constant( N, -500.0 ); }
  Vector upper() const override { return Vector::Constant( N, 500.0 ); }

  Vector init() const override
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) x0[i] = ( i % 2 == 0 ) ? 100.0 : -100.0;
    return x0;
  }

  Vector exact() const override { return Vector::Constant( N, 420.9687 ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = 418.9829 * static_cast<T>( N );
    for ( int i = 0; i < N; ++i ) f -= x[i] * std::sin( std::sqrt( std::abs( x[i] ) ) );
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i )
    {
      T xi       = x[i];
      T sgn      = ( xi >= 0 ) ? 1.0 : -1.0;
      T sqrt_abs = std::sqrt( std::abs( xi ) );
      if ( xi != 0 )
        g[i] = -std::sin( sqrt_abs ) - 0.5 * sgn * std::cos( sqrt_abs ) / sqrt_abs * xi;
      else
        g[i] = 0.0;
    }
    return g;
  }

  /// Hessian ∇²f(x) (diagonal, NxN)
  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N );  // solo diagonale

    for ( int i = 0; i < N; ++i )
    {
      T xi  = x[i];
      T Hii = 0.0;

      if ( xi != 0 )
      {
        T abs_x  = std::abs( xi );
        T sqrt_x = std::sqrt( abs_x );
        T sign_x = ( xi > 0 ) ? 1.0 : -1.0;
        T term1  = -0.5 * sign_x * ( std::cos( sqrt_x ) / sqrt_x );
        T term2  = 0.25 * xi / abs_x * std::sin( sqrt_x ) / sqrt_x;
        Hii      = term1 + term2;
      }

      triplets.emplace_back( i, i, Hii );
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// -------------------------------------------------------------------

template <typename T, int N> class StyblinskiTangN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{styblinski1990,\n"
           "  title={Experiments in nonconvex optimization},\n"
           "  author={Styblinski, M. and Tang, T. S.},\n"
           "  journal={Neural Networks},\n"
           "  year={1990}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "StyblinskiTangN requires N >= 1" );

  Vector lower() const override { return Vector::Constant( N, -5.0 ); }
  Vector upper() const override { return Vector::Constant( N, 5.0 ); }
  Vector init() const override { return Vector::Constant( N, 2.0 ); }
  Vector exact() const override { return Vector::Constant( N, -2.903534 ); }

  T operator()( Vector const & x ) const override
  {
    T f = 0;
    for ( int i = 0; i < N; ++i )
    {
      auto x2 = x[i] * x[i];
      f += ( x2 - 16 ) * x2 + 5 * x[i];
    }
    return 0.5 * f;
  }

  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i ) g[i] = ( 2 * ( x[i] * x[i] ) - 16 ) * x[i] + 2.5;
    return g;
  }

  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );
    H.reserve( N );
    for ( int i = 0; i < N; ++i ) H.insert( i, i ) = 6 * ( x[i] * x[i] ) - 16;
    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// -------------------------------------------------------------------

/**
 * @class Schubert2D
 * @brief Shubert Function (2D).
 *
 * Highly multimodal function with 18 global minima in [-10, 10]^2.
 * Value at global minimum: approx -186.7309
 */
template <typename T> class Schubert2D : public NDbase<T>
{
public:
  // Corretta citazione bibliografica
  std::string bibtex() const override
  {
    return "@article{shubert1972,\n"
           "  title={A sequential method seeking the global maximum of a function},\n"
           "  author={Shubert, B. O.},\n"
           "  journal={SIAM Journal on Numerical Analysis},\n"
           "  volume={9},\n"
           "  number={3},\n"
           "  pages={379--388},\n"
           "  year={1972}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -10.0 ); }
  Vector upper() const override { return Vector::Constant( 2, 10.0 ); }

  // Init: Punto non banale (0,0 è un massimo locale o sella)
  Vector init() const override { return Vector::Constant( 2, 0.0 ); }

  // Uno dei 18 minimi globali
  Vector exact() const override
  {
    Vector x( 2 );
    x << -7.08350643, 4.85805788;
    return x;
  }

  // --- Helper Structure per valori e derivate ---
  struct SumDerivs
  {
    T val;  // A(z)
    T d1;   // A'(z)
    T d2;   // A''(z)
  };

  // Calcola sommatoria e derivate per una singola variabile
  SumDerivs compute_sums( T z ) const
  {
    T val = 0.0;
    T d1  = 0.0;
    T d2  = 0.0;

    for ( int i = 1; i <= 5; ++i )
    {
      T ii  = T( i );
      T arg = ( ii + 1.0 ) * z + ii;

      T c = std::cos( arg );
      T s = std::sin( arg );

      T coeff_inner = ( ii + 1.0 );  // derivata dell'argomento

      val += ii * c;
      d1 -= ii * coeff_inner * s;                // d/dz cos = -sin * inner'
      d2 -= ii * coeff_inner * coeff_inner * c;  // d/dz -sin = -cos * inner'
    }
    return { val, d1, d2 };
  }

  // -----------------------------------------------------------
  // Evaluation
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    // Calcoliamo solo il valore
    T s1 = 0, s2 = 0;
    for ( int i = 1; i <= 5; ++i )
    {
      s1 += i * std::cos( ( i + 1 ) * x[0] + i );
      s2 += i * std::cos( ( i + 1 ) * x[1] + i );
    }
    return s1 * s2;
  }

  // -----------------------------------------------------------
  // Gradient
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    SumDerivs X = compute_sums( x[0] );
    SumDerivs Y = compute_sums( x[1] );

    Vector g( 2 );
    // Regola del prodotto: d/dx (X*Y) = X'*Y
    g[0] = X.d1 * Y.val;
    // d/dy (X*Y) = X*Y'
    g[1] = X.val * Y.d1;

    return g;
  }

  // -----------------------------------------------------------
  // Hessian
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    SumDerivs X = compute_sums( x[0] );
    SumDerivs Y = compute_sums( x[1] );

    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( 4 );

    // H_xx = X'' * Y
    triplets.emplace_back( 0, 0, X.d2 * Y.val );

    // H_yy = X * Y''
    triplets.emplace_back( 1, 1, X.val * Y.d2 );

    // H_xy = H_yx = X' * Y'
    T mixed = X.d1 * Y.d1;
    triplets.emplace_back( 0, 1, mixed );
    triplets.emplace_back( 1, 0, mixed );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Three-Hump Camel Function (2D)
// -------------------------------------------------------------------

/**
 * @class ThreeHumpCamel2D
 * @brief Three-hump camel function, a 2D test function
 *
 * The three-hump camel function has three local minima and a global minimum
 * at (0,0) with value 0.
 */
template <typename T> class ThreeHumpCamel2D : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@book{dixon1978,\n"
           "  title={Nonlinear Optimization},\n"
           "  author={Dixon, L. C. W. and Szegö, G. P.},\n"
           "  publisher={Academic Press},\n"
           "  year={1978}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( 2, -3 ); }
  Vector upper() const override { return Vector::Constant( 2, 3 ); }
  Vector init() const override
  {
    Vector x( 2 );
    x << -1, 2;
    return x;
  }
  Vector exact() const override
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }

  T operator()( Vector const & x ) const override
  {
    T x1 = x[0], x2 = x[1];
    return 2 * x1 * x1 - 1.05 * std::pow( x1, 4 ) + std::pow( x1, 6 ) / 6 + x1 * x2 + x2 * x2;
  }

  Vector gradient( Vector const & x ) const override
  {
    T      x1 = x[0], x2 = x[1];
    Vector g( 2 );
    g[0] = 4 * x1 - 4.2 * std::pow( x1, 3 ) + std::pow( x1, 5 ) + x2;
    g[1] = x1 + 2 * x2;
    return g;
  }

  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    T x1 = x[0];

    // diagonale e off-diagonale
    triplets.emplace_back( 0, 0, 4 - 12.6 * x1 * x1 + 5 * x1 * x1 * x1 * x1 );
    triplets.emplace_back( 0, 1, 1 );
    triplets.emplace_back( 1, 0, 1 );
    triplets.emplace_back( 1, 1, 2 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// Trigonometric Sum Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class TrigonometricSumN
 * @brief Trigonometric sum function
 *
 * This function combines trigonometric terms and is useful for testing
 * algorithms on oscillatory functions. The global minimum is at the origin.
 */
template <typename T, int N> class TrigonometricSumN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{more1981,\n"
           "  title={Testing unconstrained optimization software},\n"
           "  author={Moré, J. J. and Garbow, B. S. and Hillstrom, K. E.},\n"
           "  journal={ACM TOMS},\n"
           "  year={1981}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "TrigonometricSumN requires N >= 1" );

  Vector lower() const override { return Vector::Constant( N, -M_PI ); }
  Vector upper() const override { return Vector::Constant( N, M_PI ); }
  Vector init() const override { return Vector::Constant( N, 0.5 ); }
  Vector exact() const override { return Vector::Zero( N ); }

  /// Objective function
  T operator()( Vector const & x ) const override
  {
    T f = 0;
    for ( int i = 0; i < N; ++i )
    {
      T idx = T( i + 1 );
      T t   = std::sin( x[i] ) + idx * ( 1.0 - std::cos( x[i] ) );
      f += t * t;
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i )
    {
      T idx   = T( i + 1 );
      T t     = std::sin( x[i] ) + idx * ( 1.0 - std::cos( x[i] ) );
      T dt_dx = std::cos( x[i] ) + idx * std::sin( x[i] );
      g[i]    = 2.0 * t * dt_dx;
    }
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN (diagonal)
  SparseMatrix hessian( Vector const & x ) const override
  {
    SparseMatrix H( N, N );
    H.reserve( N );

    for ( int i = 0; i < N; ++i )
    {
      T idx            = T( i + 1 );
      T t              = std::sin( x[i] ) + idx * ( 1.0 - std::cos( x[i] ) );
      T dt_dx          = std::cos( x[i] ) + idx * std::sin( x[i] );
      T d2t_dx2        = -std::sin( x[i] ) + idx * std::cos( x[i] );
      H.insert( i, i ) = 2.0 * ( dt_dx * dt_dx + t * d2t_dx2 );
    }

    H.makeCompressed();
    return H;
  }
};


// -------------------------------------------------------------------
// WeierstrassN
// -------------------------------------------------------------------

/**
 * @class WeierstrassN
 * @brief Weierstrass Function.
 *
 * Mathematically continuous but nowhere differentiable.
 * Computationally, it creates a highly rugged landscape (fractal-like).
 * Global Minimum: x = (0, ..., 0) -> f(x) = 0
 * Domain: [-0.5, 0.5]
 */
template <typename T, int N> class WeierstrassN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@article{weierstrass1872,\n"
           "  title={Über continuirliche Funktionen eines reellen Arguments},\n"
           "  author={Weierstrass, K.},\n"
           "  journal={Mathematische Werke},\n"
           "  year={1872}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  // Parametri standard della funzione
  const T   a    = 0.5;
  const T   b    = 3.0;
  const int kmax = 20;

  Vector lower() const override { return Vector::Constant( N, -0.5 ); }
  Vector upper() const override { return Vector::Constant( N, 0.5 ); }

  // Init point vicino ma non sovrapposto al minimo globale
  Vector init() const override { return Vector::Constant( N, 0.1 ); }

  Vector exact() const override { return Vector::Zero( N ); }

  // Precalcolo delle potenze per efficienza
  std::vector<T> a_pow, b_pow, two_pi_b_pow;

  // Costruttore per precalcolare le potenze
  WeierstrassN()
  {
    // Usiamo static_cast per evitare warning di conversione
    a_pow.reserve( static_cast<size_t>( kmax + 1 ) );
    b_pow.reserve( static_cast<size_t>( kmax + 1 ) );
    two_pi_b_pow.reserve( static_cast<size_t>( kmax + 1 ) );

    for ( int k = 0; k <= kmax; ++k )
    {
      a_pow.push_back( std::pow( a, static_cast<T>( k ) ) );
      b_pow.push_back( std::pow( b, static_cast<T>( k ) ) );
      two_pi_b_pow.push_back( 2.0 * M_PI * b_pow.back() );
    }
  }

  // Helper per calcolare la costante di sottrazione (dipende solo da N)
  T compute_constant() const
  {
    T c = 0;
    for ( int k = 0; k <= kmax; ++k )
    {
      // ATTENZIONE: Corretto l'argomento del coseno
      // La costante deve usare lo stesso argomento che si ha quando x=0
      // Quando x=0: 2πb^k(x+0.5) = 2πb^k(0.5) = πb^k
      const T arg = M_PI * b_pow[static_cast<size_t>( k )];
      c += a_pow[static_cast<size_t>( k )] * std::cos( arg );
    }
    return T( N ) * c;
  }

  // -----------------------------------------------------------
  // Evaluation
  // -----------------------------------------------------------
  T operator()( Vector const & x ) const override
  {
    T f = 0;
    // Calcoliamo la somma principale
    for ( int i = 0; i < N; ++i )
    {
      for ( int k = 0; k <= kmax; ++k )
      {
        // ATTENZIONE: Corretto - mancava lo shift di 0.5 nell'argomento
        const size_t idx = static_cast<size_t>( k );
        const T      arg = two_pi_b_pow[idx] * ( x[i] + 0.5 );
        f += a_pow[idx] * std::cos( arg );
      }
    }

    // Sottraiamo la costante per avere min(f) = 0
    return f - compute_constant();
  }

  // -----------------------------------------------------------
  // Gradient
  // -----------------------------------------------------------
  Vector gradient( Vector const & x ) const override
  {
    Vector g = Vector::Zero( N );

    for ( int i = 0; i < N; ++i )
    {
      T val = 0.0;
      for ( int k = 0; k <= kmax; ++k )
      {
        const size_t idx = static_cast<size_t>( k );
        const T      arg = two_pi_b_pow[idx] * ( x[i] + 0.5 );

        // Derivata corretta:
        // d/dx [ a^k * cos( 2πb^k(x+0.5) ) ] = -a^k * sin(2πb^k(x+0.5)) * 2πb^k
        val -= a_pow[idx] * std::sin( arg ) * two_pi_b_pow[idx];
      }
      g[i] = val;
    }
    return g;
  }

  // -----------------------------------------------------------
  // Hessian (Diagonale)
  // -----------------------------------------------------------
  SparseMatrix hessian( Vector const & x ) const override
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( static_cast<size_t>( N ) );

    for ( int i = 0; i < N; ++i )
    {
      T val = 0.0;
      for ( int k = 0; k <= kmax; ++k )
      {
        const size_t idx = static_cast<size_t>( k );
        const T      arg = two_pi_b_pow[idx] * ( x[i] + 0.5 );

        // Derivata seconda corretta:
        // d²/dx² = -a^k * cos(2πb^k(x+0.5)) * (2πb^k)²
        val -= a_pow[idx] * std::cos( arg ) * two_pi_b_pow[idx] * two_pi_b_pow[idx];
      }
      triplets.emplace_back( i, i, val );
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};


// -------------------------------------------------------------------
// -------------------------------------------------------------------

template <typename T, int N> class ZakharovN : public NDbase<T>
{
public:
  std::string bibtex() const override
  {
    return "@book{zakharov1997,\n"
           "  title={Numerical Methods in Optimization},\n"
           "  author={Zakharov, A. A.},\n"
           "  publisher={CRC Press},\n"
           "  year={1997}\n"
           "}\n";
  }

  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector lower() const override { return Vector::Constant( N, -5.0 ); }
  Vector upper() const override { return Vector::Constant( N, 5.0 ); }
  Vector init() const override { return Vector::Constant( N, 1.0 ); }
  Vector exact() const override { return Vector::Zero( N ); }

  T operator()( Vector const & x ) const override
  {
    T s1 = 0, s2 = 0;
    for ( int i = 0; i < N; ++i )
    {
      s1 += x[i] * x[i];
      s2 += 0.5 * ( i + 1 ) * x[i];
    }
    return s1 + s2 * s2 + s2 * s2 * s2 * s2;
  }

  Vector gradient( Vector const & x ) const override
  {
    Vector g( N );
    T      s2 = 0;
    for ( int i = 0; i < N; ++i ) s2 += 0.5 * ( i + 1 ) * x[i];

    for ( int i = 0; i < N; ++i ) g[i] = 2 * x[i] + ( i + 1 ) * s2 + 2 * ( i + 1 ) * s2 * s2 * s2;

    return g;
  }

  SparseMatrix hessian( Vector const & ) const override
  {
    SparseMatrix H( N, N );
    H.reserve( N * N );
    for ( int i = 0; i < N; ++i )
      for ( int j = 0; j < N; ++j ) H.insert( i, j ) = ( i == j ? 2.0 : 0.0 ) + 0.5 * ( i + 1 ) * ( j + 1 );
    H.makeCompressed();
    return H;
  }
};

static std::vector<std::pair<std::shared_ptr<NDbase<double>>, std::string>> NL_list{
  { std::make_shared<AckleyN<double, 15>>(), "Ackley15D" },
  { std::make_shared<Beale2D<double>>(), "Beale2D" },
  { std::make_shared<Booth2D<double>>(), "Booth2D" },
  { std::make_shared<BroydenTridiagonalN<double, 12>>(), "BroydenTridiagonal12D" },
  { std::make_shared<BrownAlmostLinearN<double, 10>>(), "BrownAlmostLinear10D" },
  { std::make_shared<Bukin6<double>>(), "Bukin6" },
  { std::make_shared<CrossInTray2D<double>>(), "CrossInTray2D" },
  { std::make_shared<Deceptive<double>>(), "Deceptive" },
  { std::make_shared<DixonPriceN<double, 10>>(), "DixonPrice10" },
  { std::make_shared<DropWave2D<double>>(), "DropWave2D" },
  { std::make_shared<Eggholder2D<double>>(), "Eggholder2D" },
  { std::make_shared<ExtendedWoodN<double, 16>>(), "ExtendedWood16D" },
  { std::make_shared<FreudensteinRoth2D<double>>(), "FreudensteinRoth2D" },
  { std::make_shared<GriewankN<double, 10>>(), "Griewank10D" },
  { std::make_shared<HappyCatN<double, 10>>(), "HappyCat10D" },
  { std::make_shared<HGBatN<double, 10>>(), "HGBat10D" },
  { std::make_shared<HelicalValley3D<double>>(), "HelicalValley3D" },
  { std::make_shared<Himmelblau2D<double>>(), "Himmelblau2D" },
  { std::make_shared<HolderTable2D<double>>(), "HolderTable2D" },
  { std::make_shared<IllConditionedQuadraticN<double, 20>>(), "IllConditionedQuadratic20D" },
  //{ std::make_shared<KatsuuraN<double, 10>>(), "Katsuura10D" },
  { std::make_shared<Langermann<double, 2>>(), "Langermann2D" },
  { std::make_shared<LevyN<double, 10>>(), "Levy10D" },
  { std::make_shared<Levi13<double>>(), "Levi13" },
  { std::make_shared<Matyas2D<double>>(), "Matyas2D" },
  { std::make_shared<McCormick2D<double>>(), "McCormick2D" },
  { std::make_shared<MichalewiczN<double, 10>>(), "MichalewiczN10D" },
  { std::make_shared<NesterovChebyshevRosenbrock<double, 128>>(), "NesterovChebyshevRosenbrock128D" },
  { std::make_shared<PermN<double, 10>>(), "Perm10D" },
  { std::make_shared<PowellBadlyScaled2D<double>>(), "PowellBadlyScaled2D" },
  { std::make_shared<PowellSingularN<double, 16>>(), "PowellSingular16D" },
  { std::make_shared<RastriginN<double, 15>>(), "Rastrigin15D" },
  { std::make_shared<Rosenbrock2D<double>>(), "Rosenbrock2D" },
  { std::make_shared<RosenbrockN<double, 10>>(), "Rosenbrock10D" },
  { std::make_shared<RotatedEllipsoidN<double, 10>>(), "RotatedEllipsoid10D" },
  { std::make_shared<Schubert2D<double>>(), "Schubert2D" },
  { std::make_shared<Schaffer2D<double>>(), "Schaffer2D" },
  { std::make_shared<SchwefelN<double, 15>>(), "Schwefel15D" },
  { std::make_shared<StyblinskiTangN<double, 15>>(), "StyblinskiTang15D" },
  { std::make_shared<ThreeHumpCamel2D<double>>(), "ThreeHumpCamel2D" },
  { std::make_shared<TrigonometricSumN<double, 15>>(), "TrigonometricSum15D" },
  { std::make_shared<WeierstrassN<double, 10>>(), "Weierstrass10D" },
  { std::make_shared<ZakharovN<double, 10>>(), "Zakharov10D" }
};
