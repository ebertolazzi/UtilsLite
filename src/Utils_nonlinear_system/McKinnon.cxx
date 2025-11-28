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

class McKinnon : public NonlinearSystem {
  real_type const tau, theta, phi;
public:

  McKinnon()
  : NonlinearSystem(
      "McKinnon function",
      "@article{McKinnon:1998,\n"
      "  author  = {McKinnon, K.},\n"
      "  title   = {Convergence of the Nelder--Mead Simplex\n"
      "             Method to a Nonstationary Point},\n"
      "  journal = {SIAM Journal on Optimization},\n"
      "  volume  = {9},\n"
      "  number  = {1},\n"
      "  pages   = {148-158},\n"
      "  year    = {1998},\n"
      "  doi     = {10.1137/S1052623496303482},\n"
      "}\n",
      2
    )
  , tau(2.0)
  , theta(6.0)
  , phi(60.0)
  { }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    if ( x(0) <= 0.0 ) {
      f(0) = theta * tau * phi * pow(-x(0),tau) / x(0);
    } else {
      f(0) = theta * tau * pow(x(0),tau) / x(0);
    }
    f(1) = 1 + (2 * x(1));
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    if ( x(0) <= 0.0 ) {
      real_type t1 = theta * phi;
      real_type t2 = pow(-x(0),tau);
      real_type t3 = tau*tau;
      real_type t5 = x(0)*x(0);
      real_type t6 = 1 / t5;
      J.insert(0,0) = t1 * t2 * t3 * t6 - t1 * t2 * tau * t6;
    } else {
      real_type t1 = pow(x(0),tau);
      real_type t2 = theta * t1;
      real_type t3 = tau*tau;
      real_type t4 = x(0)*x(0);
      real_type t5 = 1 / t4;
      J.insert(0,0) = t2 * t3 * t5 - t2 * tau * t5;
    }
    J.insert(1,1) = 2;
    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0 << 0, -1;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0.fill(1);
  }

};
