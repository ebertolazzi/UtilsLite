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

class Function15 : public NonlinearSystem {
  typedef pair<integer,integer> INDEX;
  mutable map<INDEX,real_type> jac_idx_vals;
public:

  Function15( integer neq )
  : NonlinearSystem(
      "Function 15",
      "@article{LaCruz:2003,\n"
      "  author    = { William {La Cruz}  and  Marcos Raydan},\n"
      "  title     = {Nonmonotone Spectral Methods for Large-Scale Nonlinear Systems},\n"
      "  journal   = {Optimization Methods and Software},\n"
      "  year      = {2003},\n"
      "  volume    = {18},\n"
      "  number    = {5},\n"
      "  pages     = {583--599},\n"
      "  publisher = {Taylor & Francis},\n"
      "  doi       = {10.1080/10556780310001610493},\n"
      "}\n",
      neq
    )
  {
    check_min_equations(n,2);
    jac_idx_vals.clear();
    jac_idx_vals[INDEX(0,0)]     = 1;
    jac_idx_vals[INDEX(n-1,n-1)] = 1;
    jac_idx_vals[INDEX(n-1,n-2)] = 1;
    for ( integer i = 1; i < n-1; ++i ) {
      jac_idx_vals[INDEX(i,i-1)] = 1;
      jac_idx_vals[INDEX(i,i)]   = 1;
      jac_idx_vals[INDEX(i,i+1)] = 1;
    }
    for ( integer i = 0; i < n; ++i ) {
      jac_idx_vals[INDEX(i,n-5)] = 1;
      jac_idx_vals[INDEX(i,n-4)] = 1;
      jac_idx_vals[INDEX(i,n-3)] = 1;
      jac_idx_vals[INDEX(i,n-2)] = 1;
      jac_idx_vals[INDEX(i,n-1)] = 1;
    }
  }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    real_type bf = 3*x(n-5) - x(n-4) - x(n-3) + 0.5 * x(n-2) - x(n-1) +1;
    f(0)   = -2*x(0)*x(0)     + 3*x(0)            + bf;
    f(n-1) = -2*x(n-1)*x(n-1) + 3*x(n-1) - x(n-2) + bf;
    for ( integer i = 1; i < n-1; ++i )
      f(i) = -2*x(i)*x(i) + 3*x(i) - x(i-1) - 2*x(i+1) + bf;
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    J.insert(0,0)     = -4*x(0)   + 3;
    J.insert(n-1,n-1) = -4*x(n-1) + 3;
    J.insert(n-1,n-2) = -1;
    for ( integer i = 1; i < n-1; ++i ) {
      J.insert(i,i-1) = -1;
      J.insert(i,i)   =  -4*x(i) + 3;
      J.insert(i,i+1) = -2;
    }
    for ( integer i = 1; i < n-1; ++i ) {
      J.insert(i,n-5) += 3;
      J.insert(i,n-4) -= 1;
      J.insert(i,n-3) -= 1;
      J.insert(i,n-2) += 0.5;
      J.insert(i,n-1) -= 1;
    }
    J.makeCompressed();
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0.fill(-1);
  }

};
