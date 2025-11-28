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

class MieleAndCantrellFunction : public NonlinearSystem {

  using Matrix = Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic>;

public:

  MieleAndCantrellFunction()
  : NonlinearSystem(
      "Miele and Cantrell function",
      "@article{Grippo:1991,\n"
      "  author  = {Grippo, L. and Lampariello, F. and Lucidi, S.},\n"
      "  title   = {A Class of Nonmonotone Stabilization Methods\n"
      "             in Unconstrained Optimization},\n"
      "  journal = {Numer. Math.},\n"
      "  year    = {1991},\n"
      "  volume  = {59},\n"
      "  number  = {1},\n"
      "  pages   = {779--805},\n"
      "  doi     = {10.1007/BF01385810},\n"
      "}\n",
      4
    )
  {}

  void
  add_grad1( Vector const & x, Vector & g ) const {
    real_type tmp = 4*power3( exp(x(0)) - x(1) );
    g(0) += tmp*exp(x(0));
    g(1) -= tmp;
  }

  void
  add_hess1( Vector const & x, Matrix & h ) const {
    real_type e   = exp(x(0));
    real_type t   = e - x(1);
    real_type tte = t*t*e;
    h(0,0) += 4*tte*(4*e-1);
    h(0,1) += -12*tte;
    h(1,0) += -12*tte;
    h(1,1) += 12*t*t;
  }

  void
  add_grad2( Vector const & x, Vector & g ) const {
    real_type tmp = 600*power5(x(1)-x(2));
    g(1) += tmp;
    g(2) -= tmp;
  }

  void
  add_hess2( Vector const & x, Matrix & h ) const {
    real_type tmp = 3000*power4(x(1)-x(2));
    h(1,1) += tmp;
    h(1,2) -= tmp;
    h(2,1) -= tmp;
    h(2,2) += tmp;
  }

  void
  add_grad3( Vector const & x, Vector & g ) const {
    real_type t   = tan(x(2)-x(3));
    real_type tmp = 4*power3(t)*(1+power2(t));
    g(2) += tmp;
    g(3) -= tmp;
  }

  void
  add_hess3( Vector const & x, Matrix & h ) const {
    real_type t   = tan(x(2)-x(3));
    real_type t2  = t*t;
    real_type tmp = ((20*t2+32)*t2+12)*t2;
    h(2,2) += tmp;
    h(2,3) -= tmp;
    h(3,2) -= tmp;
    h(3,3) += tmp;
  }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    f(0) = 8*x(0)*power6(x(0));
    f(1) = f(2) = f(3) = 0;
    add_grad2( x, f );
    add_grad3( x, f );
    add_grad1( x, f );
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    Matrix h(4,4);
    h.setZero();
    h(0,0) = 8*7*power6(x(0));
    add_hess2( x, h );
    add_hess3( x, h );
    add_hess1( x, h );
    J.resize(n,n);
    J.setZero();
    for ( integer i = 0; i < 4; ++i )
      for ( integer j = 0; j < 4; ++j )
        J.insert(i,j) = h(i,j);
    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0 << 0, 1, 1, 1;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(2);
    auto & x0 { x_vec[0] };
    auto & x1 { x_vec[1] };
    x0.resize(n);
    x1.resize(n);
    x0 << 10, -10, -10, -10;
    x1 << 1, 2, 2, 2;
  }

};
