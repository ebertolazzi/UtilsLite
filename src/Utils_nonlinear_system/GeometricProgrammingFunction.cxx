/*\
 |
 |  Author:
 |    Enrico Bertolazzi
 |    University of Trento
 |    Department of Industrial Engineering
 |    Via Sommarive 9, I-38123, Povo, Trento, Italy
 |    email: enrico.bertolazzi@unitn.it
\*/

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class GeometricProgrammingFunction : public NonlinearSystem {

  using Matrix = Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic>;

public:
  
  GeometricProgrammingFunction( integer  neq )
  : NonlinearSystem(
      "Geometric Programming Function",
      "@techreport{Raydan:2004,\n"
      "  author = {William La Cruz and Jose Mario Martinez and Marcos Raydan},\n"
      "  title  = {Spectral residual method without gradient\n"
      "             information for solving large-scale nonlinear\n"
      "             systems of equations: Theory and experiments},\n"
      "  number = {Technical Report RT-04-08},\n"
      "  year   = {2004}\n"
      "}\n",
      neq
    )
  {
    check_min_equations(n,2);
  }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    f.fill(-1);
    for ( integer t = 1; t < 5; ++t ) {
      real_type t1 = 0.2*t;
      real_type t2 = t1-1;
      for ( integer i = 0; i < n; ++i ) {
        real_type tmp = t1;
        for ( integer k = 0; k < n; ++k ) {
          if ( i != k ) tmp *= pow(x(k),t1);
          else          tmp *= t2 > 0 ? pow(x(k),t2) : 1;
        }
        f(i) += tmp;
      }
    }
    for ( integer i = 0; i < n; ++i ) {
      real_type tmp = 1;
      for ( integer k = 0; k < n; ++k ) {
        if ( i != k ) tmp *= x(k);
      }
      f(i) += tmp;
    }
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    Matrix J_full(n,n);
    J_full.setZero();
    for ( integer t = 1; t < 5; ++t ) {
      real_type t1 = 0.2*t;
      real_type t2 = t1-1;
      real_type t3 = t2-1;
      for ( integer i = 0; i < n; ++i ) {
        for ( integer j = 0; j < n; ++j ) {
          real_type tmp = 1;
          if ( i == j ) {
            tmp *= t1*t2*pow(x(i),t3);
            for ( integer k = 0; k < n; ++k )
              if ( i != k )
                tmp *= pow(x(k),t1);
          } else {
            for ( integer k = 0; k < n; ++k ) {
              if ( k == i || k == j ) {
                tmp *= t1*pow(x(k),t2);
              } else {
                tmp *= pow(x(k),t1);
              }
            }
          }
          J_full(i,j) += tmp;
        }
      }
    }
    for ( integer i = 0; i < n; ++i ) {
      for ( integer j = 0; j < n; ++j ) {
        if ( i != j ) {
          real_type tmp = 1;
          for ( integer k = 0; k < n; ++k ) {
            if ( k == i || k == j ) continue;
            tmp *= x(k);
          }
          J_full(i,j) += tmp;
        }
      }
    }
    J.resize( n, n );
    J = J_full.sparseView();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0.setZero();
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0.fill(1);
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    for ( integer i = 0; i < n-1; ++i )
      UTILS_ASSERT( x(i)>0, "x[{}] = {} must be > 0", i, x(i));
  }

};
