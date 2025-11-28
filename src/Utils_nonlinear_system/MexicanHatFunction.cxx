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

static
inline
string
ini_msg_MexicanHatFunction( real_type tau ) {
  return fmt::format( "Mexican Hat Function, tau = {}", tau );
}

class MexicanHatFunction : public NonlinearSystem {
  real_type tau;
public:

  MexicanHatFunction( real_type tau_in )
  : NonlinearSystem(
      ini_msg_MexicanHatFunction(tau_in),
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
      2
    )
  , tau(tau_in)
  {}

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    real_type t1 = x(0)*x(0);
    real_type t2 = x(1)-t1;
    real_type t3 = t2*t2;
    real_type t6 = power2(1.0-x(0));
    real_type t7 = 10000.0*t3+t6-0.2E-1;
    f(0) = 2*(1-x(0)) + tau*4.0*t7*((1-20000.0*t2)*x(0)-1);
    f(1) = 2*(1-x(1)) + tau*40000.0*t7*t2;
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    real_type t1  = x(0)*x(0);
    real_type t2  = x(1)-t1;
    real_type t6  = (2-40000*t2)*x(0)-2;
    real_type t7  = t6*t6;
    real_type t8  = t2*t2;
    real_type t11 = power2(1-x(0));
    real_type t12 = 10000*t8+t11-0.02;
    real_type t20 = 20000*(x(1)-t1);
    real_type t25 = 2*t20*t6-80000*t12*x(0);
    real_type t26 = t20*t20;
    J.resize(n,n);
    J.setZero();
    J.insert(0,0) = -2 + 2*tau*(t7+t12*(120000*t1-40000*x(1)+2));
    J.insert(0,1) = tau*t25;
    J.insert(1,0) = tau*t25;
    J.insert(1,1) = -2 + tau*(2.0*t26+400000000*t8+40000*t11-800);
    J.makeCompressed();
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(3);
    auto & x0 { x_vec[0] };
    auto & x1 { x_vec[1] };
    auto & x2 { x_vec[2] };
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    x0 << 0.86, 0.72;
    x1 << 0.85858, 0.7371534;
    x2 << 1.1414204, 1.3028457;
  }

};
