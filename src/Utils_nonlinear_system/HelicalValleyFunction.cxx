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

class HelicalValleyFunction : public NonlinearSystem {

public:

  HelicalValleyFunction()
  : NonlinearSystem(
      "Helical valley function",
      "@article{Fletcher:1963,\n"
      "  author  = {Fletcher, R. and Powell, M. J. D.},\n"
      "  title   = {A Rapidly Convergent Descent Method for Minimization},\n"
      "  journal = {The Computer Journal},\n"
      "  year    = {1963},\n"
      "  volume  = {6},\n"
      "  number  = {2},\n"
      "  pages   = {163--168},\n"
      "  doi     = {10.1093/comjnl/6.2.163},\n"
      "}\n\n"
      "@article{More:1981,\n"
      "  author  = {Mor{\'e}, Jorge J. and Garbow, Burton S. and Hillstrom, Kenneth E.},\n"
      "  title   = {Testing Unconstrained Optimization Software},\n"
      "  journal = {ACM Trans. Math. Softw.},\n"
      "  year    = {1981},\n"
      "  volume  = {7},\n"
      "  number  = {1},\n"
      "  pages   = {17--41},\n"
      "  doi     = {10.1145/355934.355936},\n"
      "}\n",
      3
    )
  {}

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    //real_type theta = atan ( x(1) / x(0) ) / ( 2 * pi );
    //if ( x(0) < 0 ) theta += 0.5;
    real_type theta = atan2 ( x(1), x(0) ) / ( 2 * m_pi );
    f(0) = 10 * ( x(2) - 10 * theta );
    f(1) = 10 * ( sqrt ( power2(x(0)) + power2(x(1)) ) - 1 );
    f(2) = x(2);
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    
    real_type q2 = power2(x(0)) + power2(x(1));
    real_type q  = sqrt ( q2 );
    real_type c  = 50 / m_pi;
    J.insert(0,0) =   c * x(1) / q2;
    J.insert(0,1) = - c * x(0) / q2;
    J.insert(0,2) = 10;

    J.insert(1,0) = 10 * x(0) / q;
    J.insert(1,1) = 10 * x(1) / q;
    J.insert(1,2) = 0;

    J.insert(2,0) = 0;
    J.insert(2,1) = 0;
    J.insert(2,2) = 1;
    
    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0 << 1, 0, 0;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0 << -1, 0, 0;
  }

};
