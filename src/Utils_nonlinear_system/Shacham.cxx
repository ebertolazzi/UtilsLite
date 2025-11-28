/*\
 |
 |  Author:
 |    Enrico Bertolazzi
 |    University of Trento
 |    Department of Industrial Engineering
 |    Via Sommarive 9, I-38123, Povo, Trento, Italy
 |    email: enrico.bertolazzi@unitn.it
\*/

#define SHACHAM_BIBTEX \
"@inbook{eden2014proceedings,\n" \
"  author    = {M. Shacham},\n" \
"  title     = {Recent developments in solution techniques for\n" \
"               systems of nonlinear equations},\n" \
"  booktitle = {Proceedings of the 2nd International Conference\n" \
"               on Foundations of Computer-Aided Process Design},\n" \
"  editor    = {A.W. Westerberg, H.H. Chien},\n" \
"  series    = {Computer Aided Chemical Engineering},\n" \
"  year      = {1983},\n" \
"}\n\n" \
"@article{Meintjes:1990,\n" \
"  author  = {Meintjes, Keith and Morgan, Alexander P.},\n" \
"  title   = {Chemical Equilibrium Systems As Numerical Test Problems},\n" \
"  journal = {ACM Trans. Math. Softw.},\n" \
"  year    = {1990},\n" \
"  volume  = {16},\n" \
"  number  = {2},\n" \
"  pages   = {143--151},\n" \
"  doi     = {10.1145/78928.78930},\n" \
"}\n\n" \
"@article{Shacham:1985,\n" \
"  author  = {Mordechai Shacham},\n" \
"  title   = {Comparing software for the solution of systems\n" \
"              of nonlinear algebraic equations arising in\n" \
"              chemical engineering},\n" \
"  journal = {Computers \\& Chemical Engineering},\n" \
"  year    = {1985},\n" \
"  volume  = {9},\n" \
"  number  = {2},\n" \
"  pages   = {103--112},\n" \
"  doi     = {10.1016/0098-1354(85)85001-8}\n" \
"}\n"

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class ChemicalReactorEquilibriumConversion : public NonlinearSystem {
  mutable real_type y, T, k, kkp, k_1, kkp_1;

  void
  eval( Vector const & x ) const {
    T = x(0);
    y = x(1);
    UTILS_ASSERT( y<=1, "ChemicalReactorEquilibriumConversion::eval found y > 1, y = {}", y );
    UTILS_ASSERT( T>0,  "ChemicalReactorEquilibriumConversion::eval found T <= 0, T = {}", T );
    real_type bf1 = 149750.0/T;
    real_type bf2 = 192050.0/T;

    real_type k1 = 92.5 - bf1;
    real_type k2 = 116.7 - bf2 - 0.17*log(T);
    UTILS_ASSERT( k1 <= 350, "ChemicalReactorEquilibriumConversion::eval found k1 > 350, k1 = {}", k1 );
    UTILS_ASSERT( k2 <= 350, "ChemicalReactorEquilibriumConversion::eval found k2 > 350, k2 = {}", k2 );
    k     = exp(k1);
    kkp   = exp(k2);
    k_1   = k*bf1/T;
    kkp_1 = kkp*((bf2-0.17)/T);
  }

public:

  ChemicalReactorEquilibriumConversion()
  : NonlinearSystem("Chemical Reactor Equilibrium Conversion",SHACHAM_BIBTEX,2)
  { }

  real_type
  f1( real_type _y ) const
  { return sqrt(1-_y)*(1.82-_y)/(18.2-_y); }
  
  real_type
  f1_1( real_type _y ) const
  { return (_y*(26.39-0.5*_y)-32.942)/(sqrt(1-_y)*power2(_y-18.2)); }

  real_type
  f2( real_type _y ) const
  { return _y*_y*pow(1-_y,-1.5); }

  real_type
  f2_1( real_type _y ) const
  { return 0.5*_y*(4-_y)*pow(1-_y,-2.5); }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    eval(x);
    f(0) = k*f1(y) - kkp*f2(y);
    f(1) = T * (1.84*y+77.3) - 43260 * y - 105128; // nell'articolo originale trovo 150128!
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    eval(x); // T, y
    J.insert(0,0) = k_1*f1(y) - kkp_1 * f2(y);
    J.insert(0,1) = k*f1_1(y) - kkp * f2_1(y);
    J.insert(1,0) = 1.84*y+77.3;
    J.insert(1,1) = 1.84*T-43260;
    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0 << 1637.7032294649301990, 0.53337289955233521695;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(8);
    auto & x0{ x_vec[0] };
    auto & x1{ x_vec[1] };
    auto & x2{ x_vec[2] };
    auto & x3{ x_vec[3] };
    auto & x4{ x_vec[4] };
    auto & x5{ x_vec[5] };
    auto & x6{ x_vec[6] };
    auto & x7{ x_vec[7] };
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    x3.resize(n);
    x4.resize(n);
    x5.resize(n);
    x6.resize(n);
    x7.resize(n);
    // close to solution, but large initial residual
    x0 << 0.5, 1700.0;
    // hard to converge due to singular points
    x1 << 0.0, 1600.0;
    x2 << 0.9, 1600.0;
    x3 << 0.0, 1650.0;
    x4 << 0.9, 1700.0;
    // Luus I.V.
    x5 << 0.0, 1360.0;
    x6 << 0.0, 200;    // da controllare
    x7 << 0.0, 1650.0;
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    real_type _T = x(0);
    real_type _y = x(1);
    UTILS_ASSERT(
      _y<=1,
      "ChemicalReactorEquilibriumConversion::check_if_admissible found y > 1, y = {}", y
    );
    UTILS_ASSERT(
      _T>0,
      "ChemicalReactorEquilibriumConversion::check_if_admissible found T <= 0, T = {}", T
    );
    real_type bf1 = 149750.0/_T;
    real_type bf2 = 192050.0/_T;

    real_type k1 = 92.5 - bf1;
    real_type k2 = 116.7 - bf2 - 0.17*log(_T);
    UTILS_ASSERT(
      k1 <= 350,
      "ChemicalReactorEquilibriumConversion::check_if_admissible found k1 > 350, k1 = {}", k1
    );
    UTILS_ASSERT(
      k2 <= 350,
      "ChemicalReactorEquilibriumConversion::check_if_admissible found k2 > 350, k2 = {}", k2
    );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    L[0] = 100;       U[0] = 20000;
    L[1] = -real_max; U[1] = 1;
  }

};

/*
 *   MORDECHAI SHACHAM
 *   COMPARING SOFTWARE FOR THE SOLUTION OF SYSTEMS OF NONLINEAR ALGEBRAIC EQUATIONS ARISING IN CHEMICAL ENGINEERING
 *   Computers & Chemical Engineering Vol. 9. No. 2. pp. 103-I 12. 1985
 *   Problem 5.
 */
/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class ChemicalReactorSteadyState : public NonlinearSystem {
  mutable real_type y, T, arg, k;

  bool
  eval( Vector const & x ) const {
    y   = x(0);
    T   = x(1);
    arg = 12581.0*(T-298.0)/(298.0*T);
    if ( arg > 395 ) return false;
    //if ( T   < 0   ) return false;
    k   = 0.12 * exp( arg );
    return true;
  }

public:

  ChemicalReactorSteadyState()
  : NonlinearSystem("Chemical Reactor Steady State",SHACHAM_BIBTEX,2) {}

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    if ( eval(x) ) {
      f(0) = 120.0*y - 75.0*k*(1.0-y);
      f(1) = -y*(873.0-T) + 11.0*(T-300.0);
    } else {
      f(0) = f(1) = nan("ChemicalReactorSteadyState");
    }
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    UTILS_ASSERT( eval(x), "bad eval" );
    real_type dkdT = k * (12581.0 *298.0*T - 12581.0*(T-298.0)*298.0) / power2(298.0*T);
    J.insert(0,0) = 120.0 +75.0*k;
    J.insert(0,1) = -75.0*dkdT*(1.0-y);
    J.insert(1,0) = -(873.0-T);
    J.insert(1,1) = y + 11.0;
    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0 << 0.96386805127953300008, 346.16369814644557483739;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(9);
    auto & x0{ x_vec[0] };
    auto & x1{ x_vec[1] };
    auto & x2{ x_vec[2] };
    auto & x3{ x_vec[3] };
    auto & x4{ x_vec[4] };
    auto & x5{ x_vec[5] };
    auto & x6{ x_vec[6] };
    auto & x7{ x_vec[7] };
    auto & x8{ x_vec[8] };
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    x3.resize(n);
    x4.resize(n);
    x5.resize(n);
    x6.resize(n);
    x7.resize(n);
    x8.resize(n);

    x0 << 0.5, 320.0;
    x1 << 0.0, 300.0;
    x2 << 0.0, 350.0;
    x3 << 1.0, 400.0;
    // very close to the solution
    x4 << 0.964, 338.0;
    x5 << 0.9,   350.0;
    x6 << 0.9,   310.0;
    x7 << 0.9,   390.0;
    x8 << 0.906948356846E+00, 0.308103350833E+03;
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    real_type _y = x(0);
    real_type _T = x(1);
    //real_type T = x(1);
    UTILS_ASSERT( _y >= 0,     "bad point1");
    UTILS_ASSERT( _T >= 298.0, "bad point2");
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U[0] = real_max; L[0] = 0;
    U[1] = real_max; L[1] = -real_max;
  }

};
  
/*
 *  M. Shacham:
 *  Comparing Software for the Solution of Systems of Nonlinear Algebraic Equations Arising in Chemical Engineering.
 *  Computers & Chemical Engineering 9, 103-112, 1985 
 */
/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class ChemicalEquilibriumPartialMethaneOxidation : public NonlinearSystem {
public:

  ChemicalEquilibriumPartialMethaneOxidation()
  : NonlinearSystem(
      "Chemical equilibrium resulting from a Partial Methane Oxidation",
      SHACHAM_BIBTEX,
      7
    )
  {}

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    f(0) = 0.5*x(0) + x(1) + 0.5*x(2) - x(5)/x(6);
    f(1) = x(2) + x(3) + 2*x(4) - 2/x(6);
    f(2) = x(0) + x(1) + x(4) - 1/x(6);
    f(3) = -28837*x(0) - 139009*x(1) - 78213*x(2) + 18927*x(3) + 8427*x(4) + 13492/x(6) - 10690*x(5)/x(6);
    f(4) = x(0) + x(1) + x(2) + x(3) + x(4) - 1;
    f(5) = 400*x(0)*power3(x(3)) - 1.7837e5*x(2)*x(4);
    f(6) = x(0)*x(2) - 2.6058*x(1)*x(3);
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();

    J.insert(1-1,1-1) = 0.5;
    J.insert(1-1,2-1) = 1.0;
    J.insert(1-1,3-1) = 0.5;
    J.insert(1-1,6-1) = -1/x(6);
    J.insert(1-1,7-1) = x(5)/power2(x(6));

    J.insert(2-1,3-1) = 1.0;
    J.insert(2-1,4-1) = 1.0;
    J.insert(2-1,5-1) = 2.0;
    J.insert(2-1,7-1) = 2.0/power2(x(6));

    J.insert(3-1,1-1) = 1.0;
    J.insert(3-1,2-1) = 1.0;
    J.insert(3-1,5-1) = 1.0;
    J.insert(3-1,7-1) = 1.0/power2(x(6));

    J.insert(4-1,1-1) = -28837;
    J.insert(4-1,2-1) = -139009;
    J.insert(4-1,3-1) = -78213;
    J.insert(4-1,4-1) =  18927;
    J.insert(4-1,5-1) =  8427;
    J.insert(4-1,6-1) = -10690/x(6);
    J.insert(4-1,7-1) = -(13492 - 10690*x(5))/power2(x(6));

    J.insert(5-1,1-1) = 1.0;
    J.insert(5-1,2-1) = 1.0;
    J.insert(5-1,3-1) = 1.0;
    J.insert(5-1,4-1) = 1.0;
    J.insert(5-1,5-1) = 1.0;

    J.insert(6-1,1-1) = 400*power3(x(3));
    J.insert(6-1,3-1) = -1.7837e5*x(4);
    J.insert(6-1,4-1) = 1200*x(0)*power2(x(3));
    J.insert(6-1,5-1) = -1.7837e5*x(2);

    J.insert(7-1,1-1) = x(2);
    J.insert(7-1,2-1) = -2.6058*x(3);
    J.insert(7-1,3-1) = x(0);
    J.insert(7-1,4-1) = -2.6058*x(1);

    J.makeCompressed();
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(5);
    auto & x0{ x_vec[0] };
    auto & x1{ x_vec[1] };
    auto & x2{ x_vec[2] };
    auto & x3{ x_vec[3] };
    auto & x4{ x_vec[4] };
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    x3.resize(n);
    x4.resize(n);

    // the first hard initial guess
    x0 << 0.5,0.0,0.0,0.5,0.0,0.5,2.0;
    // the second hard initial guess
    x1 << 0.22,0.075,0.001,0.58,0.125,0.436,2.35;
    // an initial guess near the feasible solution
    x2 << 0.3,0.01,0.05,0.6,0.004,0.6,3.0;
    // an initial guess near Luus infeasible solution
    x3 << 1.5,-1.13,1.33,-0.66,-0.0007,0.8,3.0;
    // an initial guess hard for global residual oriented (2xx iterates)
    x4 << 1.5,0.001,1.33,1.e-3,1.e-4,0.8,3.0;
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    UTILS_ASSERT( x(0) > 0, "check_if_admissible: x(0) = {}", x(0) );
    for ( integer i = 0; i < n; ++i )
      UTILS_ASSERT(
        std::abs(x(i)) < 20,
        "check_if_admissible: x[{}] = {} out of [-20,20]", i, x(i)
      );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U.fill(20);
    L.fill(-20);
    L[0] = 0;
  }

};

static
inline
string
ini_msg_CutlipsSteadyStateForReactionRateEquations( int k_set ) {
  return fmt::format( "Cutlips steady state for reaction rate equations, k set N.{}", k_set );
}

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class CutlipsSteadyStateForReactionRateEquations : public NonlinearSystem {
  integer const k_set;
  real_type p_k1, p_k2, p_k3, p_kr1, p_kr2;

public:

  CutlipsSteadyStateForReactionRateEquations( integer k_set_in )
  : NonlinearSystem(
      ini_msg_CutlipsSteadyStateForReactionRateEquations(k_set_in),
      "@inbook{eden2014proceedings,\n"
      "  author    = {M. Shacham},\n"
      "  title     = {Recent developments in solution techniques for\n"
      "               systems of nonlinear equations},\n"
      "  booktitle = {Proceedings of the 2nd International Conference\n"
      "               on Foundations of Computer-Aided Process Design},\n"
      "  editor    = {A.W. Westerberg, H.H. Chien},\n"
      "  series    = {Computer Aided Chemical Engineering},\n"
      "  year      = {1983},\n"
      "}\n\n"
      "@article{Mordechai:10.1002/nme.1620230805,\n"
      "  author  = {Shacham Mordechai},\n"
      "  title   = {Numerical solution of constrained non-linear algebraic equations},\n"
      "  journal = {International Journal for Numerical Methods in Engineering},\n"
      "  year    = {1986},\n"
      "  volume  = {23},\n"
      "  number  = {8},\n"
      "  pages   = {1455-1481},\n"
      "  doi     = {10.1002/nme.1620230805},\n"
      "}\n",
      6
    )
  , k_set(k_set_in)
  {
    switch ( k_set ) {
    case 0:
      p_k1  = 31.24;
      p_kr1 = 2.062;
      p_k2  = 0.272;
      p_kr2 = 0.02;
      p_k3  = 303.03;
      break;
    case 1:
      p_k1  = 17.721;
      p_kr1 = 3.483;
      p_k2  = 0.118;
      p_kr2 = 0.033;
      p_k3  = 505.051;
      break;
    case 2:
      p_k1  = 17.721;
      p_kr1 = 6.966;
      p_k2  = 0.118;
      p_kr2 = 333.333;
      p_k3  = 505.051;
      break;
    }
  }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    f(0) = 1 - x(0) - p_k1*x(0)*x(5) + p_kr1*x(3);
    f(1) = 1 - x(1) - p_k2*x(1)*x(5) + p_kr2*x(4);
    f(2) = - x(2) + 2*p_k3*x(3)*x(4);
    f(3) = p_k1*x(0)*x(5) - p_kr1*x(3) - p_k3*x(3)*x(4);
    f(4) = 1.5*(p_k2*x(1)*x(5) - p_kr2*x(4)) - p_k3*x(3)*x(4);
    f(5) = 1 - x(3) - x(4) - x(5);
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();

    J.insert( 0, 0) = -1.0 - p_k1*x(5);
    J.insert( 0, 3) = p_kr1;
    J.insert( 0, 5) = -p_k1*x(0);

    J.insert( 1, 1) = -1.0 - p_k2*x(5);
    J.insert( 1, 4) = p_kr2;
    J.insert( 1, 5) = -p_k2*x(1);

    J.insert( 2, 2) = -1.0;
    J.insert( 2, 3) = 2*p_k3*x(4);
    J.insert( 2, 4) = 2*p_k3*x(3);

    J.insert( 3, 0) = p_k1*x(5);
    J.insert( 3, 3) = -p_kr1 - p_k3*x(4);
    J.insert( 3, 4) = -p_k3*x(3);
    J.insert( 3, 5) = p_k1*x(0);

    J.insert( 4, 1) = 1.5*p_k2*x(5);
    J.insert( 4, 3) = -p_k3*x(4);
    J.insert( 4, 4) = -1.5*p_kr2 - p_k3*x(3);
    J.insert( 4, 5) = 1.5*p_k2*x(1);

    J.insert( 5, 3) = -1.0;
    J.insert( 5, 4) = -1.0;
    J.insert( 5, 5) = -1.0;

    J.makeCompressed();
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(7);
    auto & x0 { x_vec[0] };
    auto & x1 { x_vec[1] };
    auto & x2 { x_vec[2] };
    auto & x3 { x_vec[3] };
    auto & x4 { x_vec[4] };
    auto & x5 { x_vec[5] };
    auto & x6 { x_vec[6] };
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    x3.resize(n);
    x4.resize(n);
    x5.resize(n);
    x6.resize(n);
    x0 << 0.99,    0.05,     0.05,     0.99,     0.05,    0.0;
    x1 << 0.05,    0.99,     0.05,     0.05,     0.99,    0.0;
    x2 << 0.97,    0.98,     0.06,     0.99,     0.0,     0.0;
    x3 << 3.56e-2, 3.57e-1,  1.92,     3.60e-2,  8.84e-2, 8.76e-1;
    x4 << 3.62e-2, 3.57e-1,  1.92,     9.36e-2,  3.40e-2, 8.72e-1;
    x5 << 1.03,    1.02,     -6.65e-2, 1.10e-4,  1.00,    -1.03e-3;
    x6 << 150.994, 1066.746, 1466.816, 0.438855, 0.54453, 0.016612;
  }

};

static
inline
string
ini_msg_Hiebert3ChemicalEquilibriumProblem( real_type R ) {
  return fmt::format( "Hiebert's 3rd Chemical Equilibrium Problem, R={}", R );
}

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class Hiebert3ChemicalEquilibriumProblem : public NonlinearSystem {
  real_type R;
public:

  Hiebert3ChemicalEquilibriumProblem( real_type r_in )
  : NonlinearSystem(
      ini_msg_Hiebert3ChemicalEquilibriumProblem(r_in),
      "@article{Mordechai:10.1002/nme.1620230805,\n"
      "  author  = {Shacham Mordechai},\n"
      "  title   = {Numerical solution of constrained non-linear algebraic equations},\n"
      "  journal = {International Journal for Numerical Methods in Engineering},\n"
      "  year    = {1986},\n"
      "  volume  = {23},\n"
      "  number  = {8},\n"
      "  pages   = {1455-1481},\n"
      "  doi     = {10.1002/nme.1620230805},\n"
      "}\n",
      10
    )
  , R(r_in)
  {}

  bool
  check_x( Vector const & x ) const {
    if ( x(0) >  0 &&
         x(1) >  0 &&
         x(2) >  0 &&
         x(3) >= 0 ) return true;
    return false;
  }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    if ( check_x( x ) ) {
      real_type s = x(0)+x(1)+x(2)+x(3)+x(4)+x(5)+x(6)+x(7)+x(8)+x(9);
      f(0) = x(0)+x(3)-3;
      f(1) = 2*x(0)+x(1)+x(3)+x(6)+x(7)+x(8)+2*x(9)-R;
      f(2) = 2*x(1)+2*x(4)+x(5)+x(6)-8;
      f(3) = 2*x(2)+x(4)-4*R;
      f(4) = x(0)*x(4)-0.193*x(1)*x(3);
      f(5) = x(5)*sqrt(x(1))-0.002597*sqrt(x(1)*x(3)*s);
      f(6) = x(6)*sqrt(x(3))-0.003448*sqrt(x(0)*x(3)*s);
      f(7) = x(7)*x(3)-1.799e-5*x(1)*s;
      f(8) = x(8)*x(3)-2.155e-4*x(0)*sqrt(x(2)*s);
      f(9) = x(3)*x(3)*(x(9)-3.846e-5*s);
    } else {
      f(0) = f(1) = f(2) = f(3) = f(4) = f(5) =
      f(6) = f(7) = f(8) = f(9) = nan("Hiebert3ChemicalEquilibriumProblem");
    }
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    
    real_type s = x(0)+x(1)+x(2)+x(3)+x(4)+x(5)+x(6)+x(7)+x(8)+x(9);
    
    J.insert(0,0) = 1.0; J.insert(0,1) = 0.0;
    J.insert(0,2) = 0.0; J.insert(0,3) = 1.0;
    J.insert(0,4) = 0.0; J.insert(0,5) = 0.0;
    J.insert(0,6) = 0.0; J.insert(0,7) = 0.0;
    J.insert(0,8) = 0.0; J.insert(0,9) = 0.0;

    J.insert(1,0) = 2.0; J.insert(1,1) = 1.0;
    J.insert(1,2) = 0.0; J.insert(1,3) = 1.0;
    J.insert(1,4) = 0.0; J.insert(1,5) = 0.0;
    J.insert(1,6) = 1.0; J.insert(1,7) = 1.0;
    J.insert(1,8) = 1.0; J.insert(1,9) = 2.0;

    J.insert(2,0) = 0.0; J.insert(2,1) = 2.0;
    J.insert(2,2) = 0.0; J.insert(2,3) = 0.0;
    J.insert(2,4) = 2.0; J.insert(2,5) = 1.0;
    J.insert(2,6) = 1.0; J.insert(2,7) = 0.0;
    J.insert(2,8) = 0.0; J.insert(2,9) = 0.0;

    J.insert(3,0) = 0.0; J.insert(3,1) = 0.0;
    J.insert(3,2) = 2.0; J.insert(3,3) = 0.0;
    J.insert(3,4) = 1.0; J.insert(3,5) = 0.0;
    J.insert(3,6) = 0.0; J.insert(3,7) = 0.0;
    J.insert(3,8) = 0.0; J.insert(3,9) = 0.0;

    J.insert(4,0) = x(4);
    J.insert(4,1) = -0.193*x(3);
    J.insert(4,2) = 0.0;
    J.insert(4,3) = -0.193*x(1);
    J.insert(4,4) = x(0);
    J.insert(4,5) = 0.0;
    J.insert(4,6) = 0.0;
    J.insert(4,7) = 0.0;
    J.insert(4,8) = 0.0;
    J.insert(4,9) = 0.0;

    real_type tmp1 = 0.0012985*sqrt(x(1)*x(3)/s);

    J.insert(5,0) = -tmp1;
    J.insert(5,1) = 0.5*x(5)/sqrt(x(1))-0.12985e-2*sqrt(s*x(3)/x(1))-tmp1;
    J.insert(5,2) = -tmp1;
    J.insert(5,3) = -0.12985e-2*sqrt(s*x(1)/x(3))-tmp1;
    J.insert(5,4) = -tmp1;
    J.insert(5,5) = sqrt(x(1))-tmp1;
    J.insert(5,6) = -tmp1;
    J.insert(5,7) = -tmp1;
    J.insert(5,8) = -tmp1;
    J.insert(5,9) = -tmp1;

    real_type tmp2 = 0.1724e-2*sqrt(x(0)*x(3)/s);
    J.insert(6,0) = -0.1724e-2*sqrt(s*x(3)/x(0))-tmp2;
    J.insert(6,1) = -tmp2;
    J.insert(6,2) = -tmp2;
    J.insert(6,3) = 0.5*x(6)/sqrt(x(3))-0.1724e-2*sqrt(s*x(0)/x(3))-tmp2;
    J.insert(6,4) = -tmp2;
    J.insert(6,5) = -tmp2;
    J.insert(6,6) = sqrt(x(3))-tmp2;
    J.insert(6,7) = -tmp2;
    J.insert(6,8) = -tmp2;
    J.insert(6,9) = -tmp2;

    real_type tmp3 = 0.1799e-4*x(1);
    J.insert(7,0) = -tmp3;
    J.insert(7,1) = -0.1799e-4*(s+x(1));
    J.insert(7,2) = -tmp3;
    J.insert(7,3) = x(7)-tmp3;
    J.insert(7,4) = -tmp3;
    J.insert(7,5) = -tmp3;
    J.insert(7,6) = -tmp3;
    J.insert(7,7) = x(3)-tmp3;
    J.insert(7,8) = -tmp3;
    J.insert(7,9) = -tmp3;

    real_type tmp4 = 0.10775e-3*x(0)*sqrt(x(2)/s);
    J.insert(8,0) = -0.2155e-3*sqrt(x(2)*s)-tmp4;
    J.insert(8,1) = -tmp4;
    J.insert(8,2) = -0.10775e-3*x(0)*sqrt(s/x(2))-tmp4;
    J.insert(8,3) = x(8)-tmp4;
    J.insert(8,4) = -tmp4;
    J.insert(8,5) = -tmp4;
    J.insert(8,6) = -tmp4;
    J.insert(8,7) = -tmp4;
    J.insert(8,8) = x(3)-tmp4;
    J.insert(8,9) = -tmp4;
    
    real_type tmp5 = 0.3846e-4*x(3)*x(3);
    J.insert(9,0) = -tmp5;
    J.insert(9,1) = -tmp5;
    J.insert(9,2) = -tmp5;
    J.insert(9,3) = -0.7692e-4*x(3)*s-tmp5+2*x(3)*x(9);
    J.insert(9,4) = -tmp5;
    J.insert(9,5) = -tmp5;
    J.insert(9,6) = -tmp5;
    J.insert(9,7) = -tmp5;
    J.insert(9,8) = -tmp5;
    J.insert(9,9) = 0.99996154*x(3)*x(3);

    J.makeCompressed();
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(4);
    auto & x0 { x_vec[0] };
    auto & x1 { x_vec[1] };
    auto & x2 { x_vec[2] };
    auto & x3 { x_vec[3] };
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    x3.resize(n);
    x0 << 1, 1, 10, 1,  1, 1, 0, 0, 0, 0;
    x1 << 2, 2, 10, 1,  1, 2, 0, 0, 0, 0;
    x2 << 2, 5, 40, 10, 0, 0, 0, 0, 0, 0;
    x3 << 2, 1, 20, 1,  0, 0, 0, 0, 0, 0;
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    //for (  i = 0; i < n; ++i )
    //  UTILS_ASSERT( std::abs(x(i)) < 1000, "Bad range" );
    real_type s = x(0)+x(1)+x(2)+x(3)+x(4)+x(5)+x(6)+x(7)+x(8)+x(9);
    UTILS_ASSERT( x(0) >= 0, "Bad range" );
    UTILS_ASSERT( x(1) >= 0, "Bad range" );
    UTILS_ASSERT( x(2) >= 0, "Bad range" );
    UTILS_ASSERT( x(3) >= 0, "Bad range" );
    UTILS_ASSERT( s    >= 0, "Bad range" );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    for ( integer i = 0; i < n; ++i )
      { U[i] = real_max; L[i] = -real_max; }
    L[0] = L[1] = L[2] = L[3] = 0;
  }

};

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

#define SHACHAM1_BIBTEX \
"@article{Mordechai:10.1002/nme.1620230805,\n" \
"  author  = {Shacham Mordechai},\n" \
"  title   = {Numerical solution of constrained non-linear algebraic equations},\n" \
"  journal = {International Journal for Numerical Methods in Engineering},\n" \
"  year    = {1986},\n" \
"  volume  = {23},\n" \
"  number  = {8},\n" \
"  pages   = {1455-1481},\n" \
"  doi     = {10.1002/nme.1620230805},\n" \
"}\n"

class FractionalConversionInAchemicalReactor : public NonlinearSystem {
public:

  FractionalConversionInAchemicalReactor()
  : NonlinearSystem(
      "Fractional conversion in a chemical reactor",
      SHACHAM1_BIBTEX,
      1
    )
  {}

  bool
  check_x( real_type x ) const {
    return x >= 0 && x < 0.8;
  }

  virtual
  void
  evaluate( Vector const & x_in, Vector & f ) const override {
    real_type x = x_in[0];
    if ( check_x(x) )
      f(0) = x/(1-x)-5*log(0.4*(1-x)/(0.4-0.5*x))+4.45977;
    else
      f(0) = nan("FractionalConversionInAchemicalReactor");
  }

  virtual
  void
  jacobian( Vector const & x_in, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    real_type x = x_in[0];
    if ( check_x(x) ) J.insert(0,0) = 0.1/((0.5*x-0.4)*power2(x-1));
    else              J.insert(0,0) = nan("FractionalConversionInAchemicalReactor");
    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0 << 0.7573962462537538794596412979291452934280;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(8);
    for ( integer i{0}; i < 8; ++i ) {
      x_vec[i].resize(n);
      x_vec[i].fill( i * 0.1 );
    }
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    UTILS_ASSERT( x(0) >= 0 && x(0) < 0.8, "x(0) = {} must be in [0,0.8)", x(0) );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U[0] = 0.8; L[0] = 0;
  }

};

class FractionalConversionInAchemicalReactor2 : public NonlinearSystem {
public:

  FractionalConversionInAchemicalReactor2()
  : NonlinearSystem(
      "Fractional conversion in a chemical reactor (ver 2)",
      SHACHAM1_BIBTEX,
      3
    )
  {}

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    real_type x1 = x(0);
    real_type x2 = x(1);
    real_type x3 = x(2);
    if ( x2 <= 0 || x3 <= 0 ) {
      f(0) = f(1) = f(2) = nan("FractionalConversionInAchemicalReactor2");
    } else {
      f(0) = x1/x2-5*log(0.4*x2/x3)+4.45977;
      f(1) = x2+x1-1;
      f(2) = x3+0.5*x1-0.4;
    }
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();
    
    real_type x1 = x(0);
    real_type x2 = x(1);
    real_type x3 = x(2);

    J.insert(0,0) = 1/x2;
    J.insert(0,1) = -(x1/x2+5)/x2;
    J.insert(0,2) = 5/x3;

    J.insert(1,0) = 1;
    J.insert(1,1) = 1;
    J.insert(1,2) = 0;

    J.insert(2,0) = 0.5;
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
    x0 << 0.7573962462537538794596412979291452934280,
          0.2426037537462461205403587020708547065720,
          0.02130187687312306027017935103542735328602;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(8);
    for ( integer ini{0}; ini < 8; ++ini ) {
      auto & x { x_vec[ini] };
      x(0) = ini*0.1;
      x(1) = 1-x(0);
      x(2) = 0.4-0.5*x(0);
      if ( x(2) <= 0 ) x(2) = 0.1;
    }
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    real_type x2 = x(1);
    real_type x3 = x(2);
    UTILS_ASSERT( x2 > 0, "FractionalConversionInAchemicalReactor2, x2 = {} must be > 0", x2 );
    UTILS_ASSERT( x3 > 0, "FractionalConversionInAchemicalReactor2, x3 = {} must be > 0", x3 );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U.fill(real_max);
    L.fill(-real_max);
    L[1] = L[2] = 0;
  }

};

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

#define SHACHAM2_BIBTEX \
"@article{Mordechai,\n" \
"  author  = {Shacham Mordechai},\n" \
"  title   = {Decomposition of systems of nonlinear algebraic equations},\n" \
"  journal = {AIChE Journal},\n" \
"  year    = {1984},\n" \
"  volume  = {30},\n" \
"  number  = {1},\n" \
"  pages   = {92--99},\n" \
"  doi     = {10.1002/aic.690300114},\n" \
"}\n"

class PipelineNetworkProblem : public NonlinearSystem {
public:

  PipelineNetworkProblem()
  : NonlinearSystem(
      "Pipeline Network Problem (partial oxydation of methane)",
      SHACHAM2_BIBTEX,
      7
    )
  {}

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    real_type x1 = x(0);
    real_type x2 = x(1);
    real_type x3 = x(2);
    real_type x4 = x(3);
    real_type x5 = x(4);
    real_type x6 = x(5);
    real_type x7 = x(6);
    f(0) = 0.5*x1 + x2 + 0.5*x3 - x6/x7;
    f(1) = x3 + x4 + 2*x5 - 2/x7;
    f(2) = x1 + x2 + x5 - 1/x7;
    f(3) = -28837*x1 - 139009 * x2 - 78213 * x3 + 18927 * x4 + 8427 * x5 + 13492/x7 - 10690*x6/x7;
    f(4) = x1 + x2 + x3 + x4 + x5 - 1;
    f(5) = 400*x1*power3(x4) - 1.7837e5*x3*x5;
    f(6) = x1*x3 - 2.6058*x2*x4;
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    real_type x1 = x(0);
    real_type x2 = x(1);
    real_type x3 = x(2);
    real_type x4 = x(3);
    real_type x5 = x(4);
    real_type x6 = x(5);
    real_type x7 = x(6);

    J.insert(1-1,1-1) = 0.5;
    J.insert(1-1,2-1) = 1;
    J.insert(1-1,3-1) = 0.5;
    J.insert(1-1,6-1) = -1/x7;
    J.insert(1-1,7-1) = x6/power2(x7);

    J.insert(2-1,3-1) = 1;
    J.insert(2-1,4-1) = 1;
    J.insert(2-1,5-1) = 2;
    J.insert(2-1,7-1) = 2/power2(x7);

    J.insert(3-1,1-1) = 1;
    J.insert(3-1,2-1) = 1;
    J.insert(3-1,5-1) = 1;
    J.insert(3-1,7-1) = 1/power2(x7);

    J.insert(4-1,1-1) = -28837;
    J.insert(4-1,2-1) = -139009;
    J.insert(4-1,3-1) = -78213;
    J.insert(4-1,4-1) = 18927;
    J.insert(4-1,5-1) = 8427;
    J.insert(4-1,6-1) = -10690/x7;
    J.insert(4-1,7-1) = (10690*x6-13492)/power2(x7);

    J.insert(5-1,1-1) = 1;
    J.insert(5-1,2-1) = 1;
    J.insert(5-1,3-1) = 1;
    J.insert(5-1,4-1) = 1;
    J.insert(5-1,5-1) = 1;

    J.insert(6-1,1-1) = 400*power3(x4);
    J.insert(6-1,3-1) = -1.7837e5*x5;
    J.insert(6-1,4-1) = 1200*x1*power2(x4);
    J.insert(6-1,5-1) = -1.7837e5*x3;

    J.insert(7-1,1-1) = x3;
    J.insert(7-1,2-1) = -2.6058*x4;
    J.insert(7-1,3-1) = x1;
    J.insert(7-1,4-1) = -2.6058*x2;
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0 << 0.32287083947654068257, 0.92235435391875035348e-2,
          0.46017090960632262350e-1, 0.61817167507082410985,
          0.37168509528154416956e-2, 0.57671539593554916672,
          2.9778634507911453048;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0 << 0.208, 0.042, 0.048, 0.452, 0.250, 0.340, 2;
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    UTILS_ASSERT( x(6) > 0, "Bad range" );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U.fill(real_max);
    L.fill(-real_max);
    L[6] = 0;
  }

};

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class PipelineNetworkProblem2 : public NonlinearSystem {
public:

  PipelineNetworkProblem2()
  : NonlinearSystem(
      "Pipeline Network Problem version 2 (partial oxydation of methane)",
      SHACHAM2_BIBTEX,
      9
    )
  {}

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    real_type x1 = x(0);
    real_type x2 = x(1);
    real_type x3 = x(2);
    real_type x4 = x(3);
    real_type x5 = x(4);
    real_type x6 = x(5);
    real_type x7 = x(6);
    real_type x8 = x(7);
    real_type x9 = x(8);
    f(0) = 0.5*x1 + x2 + 0.5*x3 - x9;
    f(1) = x3 + x4 + 2*x5 - 2*x8;
    f(2) = x1 + x2 + x5 - x8;
    f(3) = -28837*x1 - 139009 * x2 - 78213 * x3 + 18927 * x4
         + 8427 * x5 + 13492*x8 - 10690*x9;
    f(4) = x1 + x2 + x3 + x4 + x5 - 1;
    f(5) = 400*x1*power3(x4) - 1.7837e5*x3*x5;
    f(6) = x1*x3 - 2.6058*x2*x4;
    f(7) = x8 - 1/x7;
    f(8) = x9 - x6/x7;
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    J.resize(n,n);
    J.setZero();

    real_type x1 = x(0);
    real_type x2 = x(1);
    real_type x3 = x(2);
    real_type x4 = x(3);
    real_type x5 = x(4);
    real_type x6 = x(5);
    real_type x7 = x(6);

    J.insert(1-1,1-1) = 0.5;
    J.insert(1-1,2-1) = 1;
    J.insert(1-1,3-1) = 0.5;
    J.insert(1-1,9-1) = -1;

    J.insert(2-1,3-1) = 1;
    J.insert(2-1,4-1) = 1;
    J.insert(2-1,5-1) = 2;
    J.insert(2-1,8-1) = -2;

    J.insert(3-1,1-1) = 1;
    J.insert(3-1,2-1) = 1;
    J.insert(3-1,5-1) = 1;
    J.insert(3-1,8-1) = -1;

    J.insert(4-1,1-1) = -28837;
    J.insert(4-1,2-1) = -139009;
    J.insert(4-1,3-1) = -78213;
    J.insert(4-1,4-1) = 18927;
    J.insert(4-1,5-1) = 8427;
    J.insert(4-1,8-1) = 13492;
    J.insert(4-1,9-1) = -10690;

    J.insert(5-1,1-1) = 1;
    J.insert(5-1,2-1) = 1;
    J.insert(5-1,3-1) = 1;
    J.insert(5-1,4-1) = 1;
    J.insert(5-1,5-1) = 1;

    J.insert(6-1,1-1) = 400*power3(x4);
    J.insert(6-1,3-1) = -1.7837e5*x5;
    J.insert(6-1,4-1) = 1200*x1*power2(x4);
    J.insert(6-1,5-1) = -1.7837e5*x3;

    J.insert(7-1,1-1) = x3;
    J.insert(7-1,2-1) = -2.6058*x4;
    J.insert(7-1,3-1) = x1;
    J.insert(7-1,4-1) = -2.6058*x2;

    J.insert(8-1,7-1) = 1/power2(x7);
    J.insert(8-1,8-1) = 1;

    J.insert(9-1,6-1) = -1/x7;
    J.insert(9-1,7-1) = x6/power2(x7);
    J.insert(9-1,9-1) = 1;

    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0(0) = 0.32287083947654068257;
    x0(1) = 0.92235435391875035348e-2;
    x0(2) = 0.46017090960632262350e-1;
    x0(3) = 0.61817167507082410985;
    x0(4) = 0.37168509528154416956e-2;
    x0(5) = 0.57671539593554916672;
    x0(6) = 2.9778634507911453048;
    x0(7) = 1/x0(6);
    x0(8) = x0(5)/x0(6);
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0 << 0.208, 0.042, 0.048, 0.452, 0.250, 0.340, 2, 0, 0;
    x0(7) = 1/x0(6);
    x0(8) = x0(5)/x0(6);
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    UTILS_ASSERT( x(6) > 0, "Bad range" );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U.fill(real_max);
    L.fill(-real_max);
    L[6] = 0;
  }

};

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class ModelEquationsForTheCSTR : public NonlinearSystem {
  real_type R, V, vo, CAO, CBO;
public:

  ModelEquationsForTheCSTR()
  : NonlinearSystem(
      "Model equations for the CSTR",
      "@article{Shacham:2002,\n"
      "  author  = {Mordechai Shacham and Neima Brauner},\n"
      "  title   = {Numerical solution of non-linear algebraic\n"
      "             equations with discontinuities},\n"
      "  journal = {Computers \\& Chemical Engineering},\n"
      "  volume  = {26},\n"
      "  number  = {10},\n"
      "  pages   = {1449--1457},\n"
      "  year    = {2002},\n"
      "  doi     = {10.1016/S0098-1354(02)00122-9}\n"
      "}\n",
      15
    )
  {
    R   = 1.987;
    V   = 500;
    vo  = 75/3.3;
    CAO = 25/vo;
    CBO = 50/vo;
  }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    real_type T   = x(0);
    real_type SRH = x(1);
    real_type CA  = x(2);
    real_type CB  = x(3);
    real_type CC  = x(4);
    real_type CD  = x(5);
    real_type CE  = x(6);
    real_type rA  = x(7);
    real_type rB  = x(8);
    real_type rC  = x(9);
    real_type rD  = x(10);
    real_type rE  = x(11);
    real_type k1B = x(12);
    real_type k2C = x(13);
    real_type k3E = x(14);

    f(0)  = V*rA + vo*(CAO-CA);
    f(1)  = V*rB + vo*(CBO-CB);
    f(2)  = V*rC - vo*CC;
    f(3)  = V*rD - vo*CD;
    f(4)  = V*rE - vo*CE;
    f(5)  = SRH*V-6500*T+2200000; // 5000*(350-T)-25*(20+40)*(T-300)+V*SRH;
    f(6)  = rA + 2*k1B*CA*CB;
    f(7)  = rB + (k1B*CA+2*k2C*CC*CB)*CB;
    f(8)  = rC - (3*k1B*CA-k2C*CC*CB)*CB;
    f(9)  = rD + (k3E*CD-k2C*CC*CB*CB);
    f(10) = rE - k3E*CD;
    //f(11) = k1B - 0.4*exp((20000/R)*(1/300.9-1/T));
    //f(12) = k2C - 10*exp((5000/R)*(1/310.0-1/T));
    //f(13) = k3E - 10*exp((10000/R)*(1/320.9-1/T));
    
    f(11) = k1B - 0.4*exp((200/3.009-20000/T)/R);
    f(12) = k2C - 10*exp((50/3.1-5000/T)/R);
    f(13) = k3E - 10*exp((100/3.209-10000/T)/R);

    f(14) = SRH - 40000*k1B*CA*CB - 20000*k2C*CC*CB*CB + 5000*k3E*CD;
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {
    real_type T   = x(0);
    //real_type SRH = x(1);
    real_type CA  = x(2);
    real_type CB  = x(3);
    real_type CC  = x(4);
    real_type CD  = x(5);
    //real_type rB  = x(8);
    //real_type rD  = x(10);
    real_type k1B = x(12);
    real_type k2C = x(13);
    real_type k3E = x(14);

    J.resize(n,n);
    J.setZero();

    J.insert(0,2) = -vo;
    J.insert(0,7) = V;

    J.insert(1,3) = -vo;
    J.insert(1,8) = V;

    J.insert(2,4) = -vo;
    J.insert(2,9) = V;

    J.insert(3,5)  = -vo;
    J.insert(3,10) = V;

    J.insert(4,6)  = -vo;
    J.insert(4,11) = V;

    J.insert(5,0)  = -6500;
    J.insert(5,1)  = V;

    J.insert(6,2)  = 2*k1B*CB;
    J.insert(6,3)  = 2*k1B*CA;
    J.insert(6,7)  = 1;
    J.insert(6,12) = 2*CA*CB;

    J.insert(7,2)  = CB*k1B;
    J.insert(7,3)  = 4*CB*CC*k2C+CA*k1B;
    J.insert(7,4)  = 2*CB*CB*k2C;
    J.insert(7,8)  = 1;
    J.insert(7,12) = CA*CB;
    J.insert(7,13) = 2*CC*CB*CB;

    J.insert(8,2)  = -3*CB*k1B;
    J.insert(8,3)  = 2*CB*CC*k2C-3*CA*k1B;
    J.insert(8,4)  = CB*CB*k2C;
    J.insert(8,9)  = 1;
    J.insert(8,12) = -3*CA*CB;
    J.insert(8,13) = CB*CB*CC;

    J.insert(9,3)  = -2*k2C*CC*CB;
    J.insert(9,4)  = -CB*CB*k2C;
    J.insert(9,5)  = k3E;
    J.insert(9,10) = 1;
    J.insert(9,13) = -CC*CB*CB;
    J.insert(9,14) = CD;

    J.insert(10,5)  = -k3E;
    J.insert(10,11) = 1;
    J.insert(10,14) = -CD;

    J.insert(11,0)  = -8000*exp((66.4672648720505151213027583915-20000.0/T)/R)/(R*T*T);
    J.insert(11,12) = 1;

    J.insert(12,0)  = -50000.0*exp((16.1290322580645161290322580645-5000.0/T)/R)/(R*T*T);
    J.insert(12,13) = 1;

    J.insert(13,0)  = -100000.0*exp((31.1623558741040822686195076348-10000.0/T)/R)/(R*T*T);
    J.insert(13,14) = 1;

    J.insert(14,1)  = 1;
    J.insert(14,2)  = -40000*k1B*CB;
    J.insert(14,3)  = -40000*CB*CC*k2C-40000*CA*k1B;
    J.insert(14,4)  = -20000*CB*CB*k2C;
    J.insert(14,5)  = 5000*k3E;
    J.insert(14,12) = -40000*CA*CB;
    J.insert(14,13) = -20000*CC*CB*CB;
    J.insert(14,14) = 5000*CD;

    J.makeCompressed();
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    real_type & T   = x0(0);
    real_type & SRH = x0(1);
    real_type & CA  = x0(2);
    real_type & CB  = x0(3);
    real_type & CC  = x0(4);
    real_type & CD  = x0(5);
    real_type & CE  = x0(6);
    real_type & rA  = x0(7);
    real_type & rB  = x0(8);
    real_type & rC  = x0(9);
    real_type & rD  = x0(10);
    real_type & rE  = x0(11);
    real_type & k1B = x0(12);
    real_type & k2C = x0(13);
    real_type & k3E = x0(14);

    T   = 458.18172725396032261;
    SRH = 1556.3624543014841939;
    CA  = 0.35883693347107439649e-4;
    CB  = 0.17948231578524555024e-1;
    CC  = 0.83391131932590483949;
    CD  = 0.33768297695385023131e-4;
    CE  = 0.81600108683637911432;
    rA  = -0.49998368923029676935e-1;
    rB  = -0.99184171291885247499e-1;
    rC  = 0.37905059969359310886e-1;
    rD  = 0.15349226225175010514e-5;
    rE  = 0.37090958492562687015e-1;
    k1B = 38815.665805380501262;
    k2C = 138.07747308025211743;
    k3E = 1098.3958631006667095;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    
    real_type & T   = x0(0);
    real_type & SRH = x0(1);
    real_type & CA  = x0(2);
    real_type & CB  = x0(3);
    real_type & CC  = x0(4);
    real_type & CD  = x0(5);
    real_type & CE  = x0(6);
    real_type & rA  = x0(7);
    real_type & rB  = x0(8);
    real_type & rC  = x0(9);
    real_type & rD  = x0(10);
    real_type & rE  = x0(11);
    real_type & k1B = x0(12);
    real_type & k2C = x0(13);
    real_type & k3E = x0(14);

    T   = 420;
    SRH = 1164944;
    CA  = 0.5;
    CB  = 0.01;
    CC  = 1;
    CD  = 0.0001;
    CE  = 1;
    rA  = -58.245;
    rB  = -29.1393;
    rC  = 87.35913;
    rD  = -0.03391;
    rE  = 0.042291;
    k1B = 5824.501;
    k2C = 83.80888;
    k3E = 422.9115;
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    real_type T   = x(0);
    //real_type SRH = x(1);
    real_type CA  = x(2);
    real_type CB  = x(3);
    real_type CC  = x(4);
    real_type CD  = x(5);
    real_type CE  = x(6);
    //real_type rA  = x(7);
    //real_type rB  = x(8);
    //real_type rC  = x(9);
    //real_type rD  = x(10);
    //real_type rE  = x(11);
    real_type k1B = x(12);
    real_type k2C = x(13);
    real_type k3E = x(14);
    UTILS_ASSERT(
      T   > 0 &&
      CA  > 0 &&
      CB  > 0 &&
      CC  > 0 &&
      CD  > 0 &&
      CE  > 0 &&
      k1B > 0 &&
      k2C > 0 &&
      k3E > 0,
      "non positive"
    );
    //ASSERT( rA < 0, "T non positive" );
    //ASSERT( rB < 0, "T non positive" );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U.fill(real_max);
    L.fill(-real_max);
    L[0] = L[2] = L[3] = L[4] = L[5] = L[6] =
    L[12] = L[13] = L[14] = 0;
  }

};

/*
 *  K.L. Hiebert:
 *  An evaluation of mathematical software that
 *  solves systems of nonlinear equations.
 *  ACM Trans. Math. Soft. 8, 5-20, 1982
 */

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

static
inline
string
ini_msg_ModelEquationsForCombustionOfPropane( real_type R_set ) {
  return fmt::format( "Model equations for combustion of propane, R={}", R_set );
}

class ModelEquationsForCombustionOfPropane : public NonlinearSystem {
  real_type K5, K6, K7, K8, K9, K10, p, R;
public:

  ModelEquationsForCombustionOfPropane( real_type R_in )
  : NonlinearSystem(
      ini_msg_ModelEquationsForCombustionOfPropane(R_in),
      "@article{Shacham:2002,\n"
      "  author  = {Mordechai Shacham and Neima Brauner},\n"
      "  title   = {Numerical solution of non-linear algebraic\n"
      "             equations with discontinuities},\n"
      "  journal = {Computers \\& Chemical Engineering},\n"
      "  volume  = {26},\n"
      "  number  = {10},\n"
      "  pages   = {1449--1457},\n"
      "  year    = {2002},\n"
      "  doi     = {10.1016/S0098-1354(02)00122-9}\n"
      "}\n\n"
      "@article{Hiebert:1982,\n"
      "  author  = {Hiebert, K. L.},\n"
      "  title   = {An Evaluation of Mathematical Software That\n"
      "             Solves Systems of Nonlinear Equations},\n"
      "  journal = {ACM Trans. Math. Softw.},\n"
      "  year    = {1982},\n"
      "  volume  = {8},\n"
      "  number  = {1},\n"
      "  pages   = {5--20},\n"
      "  doi     = {10.1145/355984.355986},\n"
      "}\n",
      10
    )
  , R(R_in)
  {
    K5  = 0.193;
    K6  = 0.002597;
    K7  = 0.003448;
    K8  = 1.799e-5;
    K9  = 2.155e-4;
    K10 = 3.846e-5;
    p   = 40;
  }

  virtual
  void
  evaluate( Vector const & x, Vector & f ) const override {
    for ( integer i = 0; i < n; ++i ) {
      if ( x(i) <  0 ) {
        f(0) = f(1) = f(2) = f(3) = f(4) = f(5) =
        f(6) = f(7) = f(8) = f(9) = nan("ModelEquationsForCombustionOfPropane");
        return;
      }
    }

    real_type n1  = x(0);
    real_type n2  = x(1);
    real_type n3  = x(2);
    real_type n4  = x(3);
    real_type n5  = x(4);
    real_type n6  = x(5);
    real_type n7  = x(6);
    real_type n8  = x(7);
    real_type n9  = x(8);
    real_type n10 = x(9);

    real_type nT = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10;

    f(0) = n1+n4-3;
    f(1) = 2*n1+n2+n4+n7+n8+n9+2*n10-R;
    f(2) = 2*n2+2*n5+n6+n7-8;
    f(3) = 2*n3+2*n9-4*R;
    f(4) = K5*n2*n4-n1*n5;
    f(5) = K6*sqrt(n1*n4)-sqrt(n1)*n6*sqrt(p/nT);
    f(6) = K7*sqrt(n1*n2)-sqrt(n4)*n7*sqrt(p/nT);
    f(7) = K8*n1-n4*n8*(p/nT);
    f(8) = K9*sqrt(n1*n3)-n4*n9*sqrt(p/nT);
    f(9) = K10*n1*n1-n4*n4*n10*(p/nT);
  }

  virtual
  void
  jacobian( Vector const & x, SparseMatrix & J ) const override {

    real_type n1  = x(0);
    real_type n2  = x(1);
    real_type n3  = x(2);
    real_type n4  = x(3);
    real_type n5  = x(4);
    real_type n6  = x(5);
    real_type n7  = x(6);
    real_type n8  = x(7);
    real_type n9  = x(8);
    real_type n10 = x(9);

    J.resize(n,n);
    J.setZero();

    J.insert(1-1,1-1) = 1;
    J.insert(1-1,4-1) = 1;

    J.insert(2-1,1-1)  = 2;
    J.insert(2-1,2-1)  = 1;
    J.insert(2-1,4-1)  = 1;
    J.insert(2-1,7-1)  = 1;
    J.insert(2-1,8-1)  = 1;
    J.insert(2-1,9-1)  = 1;
    J.insert(2-1,10-1) = 2;

    J.insert(3-1,2-1) = 2;
    J.insert(3-1,5-1) = 2;
    J.insert(3-1,6-1) = 1;
    J.insert(3-1,7-1) = 1;

    J.insert(4-1,3-1) = 2;
    J.insert(4-1,9-1) = 2;

    J.insert(5-1,1-1) = -n5;
    J.insert(5-1,2-1) = K5*n4;
    J.insert(5-1,4-1) = K5*n2;
    J.insert(5-1,5-1) = -n1;

    {
      //real_type tmp1 = sqrt(n1)*n6/sqrt(p/nT)*p/(nT*nT)/2.0;
      real_type t2 = sqrt(n1*n4);
      real_type t4 = K6/t2;
      real_type t6 = sqrt(n1);
      real_type t9 = sqrt(p);
      real_type t10 = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10;
      real_type t11 = sqrt(t10);
      real_type t12 = 1/t11;
      real_type t19 = t6*n6*t9/t11/t10;
      real_type b1 = t4*n4/2.0-1/t6*n6*t9*t12/2.0+t19/2.0;
      real_type b2 = t19/2.0;
      real_type b3 = b2;
      real_type b4 = t4*n1/2.0+t19/2.0;
      real_type b5 = b3;
      real_type b6 = -t6*t9*t12+b5;
      real_type b7 = b5;
      real_type b8 = b7;
      real_type b9 = b8;
      real_type b10 = b9;
      J.insert(6-1,1-1)  = b1;
      J.insert(6-1,2-1)  = b2;
      J.insert(6-1,3-1)  = b3;
      J.insert(6-1,4-1)  = b4;
      J.insert(6-1,5-1)  = b5;
      J.insert(6-1,6-1)  = b6;
      J.insert(6-1,7-1)  = b7;
      J.insert(6-1,8-1)  = b8;
      J.insert(6-1,9-1)  = b9;
      J.insert(6-1,10-1) = b10;
    }
    
    {
      //real_type tmp2 = sqrt(n4)*n7/sqrt(p/nT)*p/(nT*nT)/2.0;
      real_type t2 = sqrt(n1*n2);
      real_type t4 = K7/t2;
      real_type t6 = sqrt(n4);
      real_type t8 = sqrt(p);
      real_type t9 = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10;
      real_type t10 = sqrt(t9);
      real_type t14 = t6*n7*t8/t10/t9;
      real_type b1 = t4*n2/2.0+t14/2.0;
      real_type b2 = t4*n1/2.0+t14/2.0;
      real_type b3 = t14/2.0;
      real_type t20 = 1/t10;
      real_type b4 = -1/t6*n7*t8*t20/2.0+t14/2.0;
      real_type b5 = b3;
      real_type b6 = b5;
      real_type b7 = -t6*t8*t20+b6;
      real_type b8 = b6;
      real_type b9 = b8;
      real_type b10 = b9;
      J.insert(7-1,1-1)  = b1;
      J.insert(7-1,2-1)  = b2;
      J.insert(7-1,3-1)  = b3;
      J.insert(7-1,4-1)  = b4;
      J.insert(7-1,5-1)  = b5;
      J.insert(7-1,6-1)  = b6;
      J.insert(7-1,7-1)  = b7;
      J.insert(7-1,8-1)  = b8;
      J.insert(7-1,9-1)  = b9;
      J.insert(7-1,10-1) = b10;
    }

    {
      //real_type tmp3 = n4*n8*p/(nT*nT);
      real_type t2 = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10;
      real_type t3 = t2*t2;
      real_type t6 = n4*n8*p/t3;
      real_type b1 = K8+t6;
      real_type b2 = t6;
      real_type b3 = b2;
      real_type t8 = 1/t2;
      real_type b4 = -n8*p*t8+b3;
      real_type b5 = b3;
      real_type b6 = b5;
      real_type b7 = b6;
      real_type b8 = -n4*p*t8+b7;
      real_type b9 = b7;
      real_type b10 = b9;
      J.insert(8-1,1-1)  = b1;
      J.insert(8-1,2-1)  = b2;
      J.insert(8-1,3-1)  = b3;
      J.insert(8-1,4-1)  = b4;
      J.insert(8-1,5-1)  = b5;
      J.insert(8-1,6-1)  = b6;
      J.insert(8-1,7-1)  = b7;
      J.insert(8-1,8-1)  = b8;
      J.insert(8-1,9-1)  = b9;
      J.insert(8-1,10-1) = b10;
    }
    
    {
      // real_type tmp4 = n4*n9/sqrt(p/nT)*p/(nT*nT)/2.0;
      real_type t2 = sqrt(n1*n3);
      real_type t4 = K9/t2;
      real_type t7 = sqrt(p);
      real_type t8 = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10;
      real_type t9 = sqrt(t8);
      real_type t13 = n4*n9*t7/t9/t8;
      real_type b1 = t4*n3/2.0+t13/2.0;
      real_type b2 = t13/2.0;
      real_type b3 = t4*n1/2.0+t13/2.0;
      real_type t18 = 1/t9;
      real_type b4 = -n9*t7*t18+b2;
      real_type b5 = b2;
      real_type b6 = b5;
      real_type b7 = b6;
      real_type b8 = b7;
      real_type b9 = -n4*t7*t18+b8;
      real_type b10 = b8;
      J.insert(9-1,1-1)  = b1;
      J.insert(9-1,2-1)  = b2;
      J.insert(9-1,3-1)  = b3;
      J.insert(9-1,4-1)  = b4;
      J.insert(9-1,5-1)  = b5;
      J.insert(9-1,6-1)  = b6;
      J.insert(9-1,7-1)  = b7;
      J.insert(9-1,8-1)  = b8;
      J.insert(9-1,9-1)  = b9;
      J.insert(9-1,10-1) = b10;
    }

    {
      //real_type tmp5 = n4*n4*n10*p/(nT*nT);
      real_type t3 = n4*n4;
      real_type t5 = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10;
      real_type t6 = t5*t5;
      real_type t9 = t3*n10*p/t6;
      real_type b1 = 2.0*K10*n1+t9;
      real_type b2 = t9;
      real_type b3 = b2;
      real_type t11 = 1/t5;
      real_type b4 = -2.0*n4*n10*p*t11+b3;
      real_type b5 = b3;
      real_type b6 = b5;
      real_type b7 = b6;
      real_type b8 = b7;
      real_type b9 = b8;
      real_type b10 = -t3*p*t11+b9;
      J.insert(10-1,1-1)  = b1;
      J.insert(10-1,2-1)  = b2;
      J.insert(10-1,3-1)  = b3;
      J.insert(10-1,4-1)  = b4;
      J.insert(10-1,5-1)  = b5;
      J.insert(10-1,6-1)  = b6;
      J.insert(10-1,7-1)  = b7;
      J.insert(10-1,8-1)  = b8;
      J.insert(10-1,9-1)  = b9;
      J.insert(10-1,10-1) = b10;
    }
    J.makeCompressed();
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(3);
    auto & x0{ x_vec[0] };
    auto & x1{ x_vec[1] };
    auto & x2{ x_vec[2] };
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    real_type xhh = 1;
    x0.fill(xhh);
    x1.fill(log(xhh));
    x2.fill(sqrt(xhh));
  }

  virtual
  void
  check_if_admissible( Vector const & x ) const override {
    UTILS_ASSERT( x(0) >= 0, "Bad range" );
    UTILS_ASSERT( x(1) >= 0, "Bad range" );
    UTILS_ASSERT( x(2) >= 0, "Bad range" );
  }

  virtual
  void
  bounding_box( Vector & L, Vector & U ) const override {
    U.fill(real_max);
    L.fill(-real_max);
    L[0] = L[1] = L[2] = 0;
  }

};

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class ModelEquationsForCombustionOfPropane1 : public ModelEquationsForCombustionOfPropane {
public:

  ModelEquationsForCombustionOfPropane1()
  : ModelEquationsForCombustionOfPropane(10)
  {
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0 << 2.915725423895220, 3.960942810808880, 19.986291646551500,
          0.084274576104777, 0.022095601769893, 0.000722766590884,
          0.033200408251574, 0.000421099693392, 0.027416706896918,
          0.031146775227006;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0 << 1.5, 2, 35, 0.5, 0.05, 0.005, 0.04, 0.003, 0.02, 5;
  }

};

/*\
 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
\*/

class ModelEquationsForCombustionOfPropane2 : public ModelEquationsForCombustionOfPropane {
public:

  ModelEquationsForCombustionOfPropane2()
  : ModelEquationsForCombustionOfPropane(5)
  {
  }

  virtual
  void
  exact_solution( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0 { x_vec[0] };
    x0.resize(n);
    x0 << 2.915725423895220, 3.960942810808880, 19.986291646551500,
          0.084274576104777, 0.022095601769893, 0.000722766590884,
          0.033200408251574, 0.000421099693392, 0.027416706896918,
          0.031146775227006;
  }

  virtual
  void
  initial_points( vector<Vector> & x_vec ) const override {
    x_vec.resize(1);
    auto & x0{ x_vec[0] };
    x0.resize(n);
    x0 << 1.5, 2, 35, 0.5, 0.05, 0.005, 0.04, 0.003, 0.02, 5;
  }

};
