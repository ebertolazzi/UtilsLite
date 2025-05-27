/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022                                                      |
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

#include "Utils.hh"
#include "Utils_autodiff.hh"

using namespace std;
using namespace autodiff;
using namespace Utils;


// The single-variable function for which derivatives are needed
dual f(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

// Specialize isArithmetic for complex to make it compatible with dual
namespace autodiff::detail {

template<typename T>
struct ArithmeticTraits<complex<T>> : ArithmeticTraits<T> {};

} // autodiff::detail

using cxdual = Dual<complex<double>, complex<double>>;

// The single-variable function for which derivatives are needed
cxdual f2(cxdual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

// The multi-variable function for which derivatives are needed
dual f3(dual x, dual y, dual z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

// A type defining parameters for a function of interest
struct Params
{
    dual a;
    dual b;
    dual c;
};

// The function that depends on parameters for which derivatives are needed
dual f4(dual x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

// Define functions A, Ax, Ay using double; analytical derivatives are available.
double  A(double x, double y) { return x*y; }
double Ax(double x, double y) { return y; }
double Ay(double x, double y) { return x; }

// Define functions B, Bx, By using double; analytical derivatives are available.
double  B(double x, double y) { return x + y; }
double Bx(double x, double y) { return 1.0; }
double By(double x, double y) { return 1.0; }

// Wrap A into Adual function so that it can be used within autodiff-enabled codes.
dual Adual(dual const& x, dual const& y)
{
    dual res = A(x.val, y.val);

    if(x.grad != 0.0)
        res.grad += x.grad * Ax(x.val, y.val);

    if(y.grad != 0.0)
        res.grad += y.grad * Ay(x.val, y.val);

    return res;
}

// Wrap B into Bdual function so that it can be used within autodiff-enabled codes.
dual Bdual(dual const& x, dual const& y)
{
    dual res = B(x.val, y.val);

    if(x.grad != 0.0)
        res.grad += x.grad * Bx(x.val, y.val);

    if(y.grad != 0.0)
        res.grad += y.grad * By(x.val, y.val);

    return res;
}

// Define autodiff-enabled C function that relies on Adual and Bdual
dual C(dual const& x, dual const& y)
{
    const auto A = Adual(x, y);
    const auto B = Bdual(x, y);
    return A*A + B;
}

int
main() {
  {
    dual x = 2.0;                                 // the input variable x
    dual u = f(x);                                // the output variable u

    double dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx

    std::cout << "u = " << u << std::endl;        // print the evaluated output u
    std::cout << "du/dx = " << dudx << std::endl; // print the evaluated derivative du/dx
  }
  
  {
    cxdual x = 2.0;   // the input variable x
    cxdual u = f2(x);  // the output variable u

    cxdual dudx = derivative(f2, wrt(x), at(x));  // evaluate the derivative du/dx

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
  }
  
  {
    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    dual u = f3(x, y, z);

    double dudx = derivative(f3, wrt(x), at(x, y, z));
    double dudy = derivative(f3, wrt(y), at(x, y, z));
    double dudz = derivative(f3, wrt(z), at(x, y, z));

    std::cout << "u = " << u << std::endl;         // print the evaluated output u = f(x, y, z)
    std::cout << "du/dx = " << dudx << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/dy = " << dudy << std::endl;  // print the evaluated derivative du/dy
    std::cout << "du/dz = " << dudz << std::endl;  // print the evaluated derivative du/dz
  }
  {
    Params params;   // initialize the parameter variables
    params.a = 1.0;  // the parameter a of type dual, not double!
    params.b = 2.0;  // the parameter b of type dual, not double!
    params.c = 3.0;  // the parameter c of type dual, not double!

    dual x = 0.5;  // the input variable x

    dual u = f4(x, params);  // the output variable u

    double dudx = derivative(f4, wrt(x), at(x, params));        // evaluate the derivative du/dx
    double duda = derivative(f4, wrt(params.a), at(x, params)); // evaluate the derivative du/da
    double dudb = derivative(f4, wrt(params.b), at(x, params)); // evaluate the derivative du/db
    double dudc = derivative(f4, wrt(params.c), at(x, params)); // evaluate the derivative du/dc

    std::cout << "u = " << u << std::endl;         // print the evaluated output u
    std::cout << "du/dx = " << dudx << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/da = " << duda << std::endl;  // print the evaluated derivative du/da
    std::cout << "du/db = " << dudb << std::endl;  // print the evaluated derivative du/db
    std::cout << "du/dc = " << dudc << std::endl;  // print the evaluated derivative du/dc
  }
  {
    dual x = 1.0;
    dual y = 2.0;

    auto C0 = C(x, y);

    // Compute derivatives of C with respect to x and y using autodiff!
    auto Cx = derivative(C, wrt(x), at(x, y));
    auto Cy = derivative(C, wrt(y), at(x, y));

    // Compute expected analytical derivatives of C with respect to x and y
    auto x0 = x.val;
    auto y0 = y.val;
    auto expectedCx = 2.0*A(x0, y0)*Ax(x0, y0) + Bx(x0, y0);
    auto expectedCy = 2.0*A(x0, y0)*Ay(x0, y0) + By(x0, y0);

    std::cout << "C0 = " << C0 << "\n";

    std::cout << "Cx(computed) = " << Cx << "\n";
    std::cout << "Cx(expected) = " << expectedCx << "\n";

    std::cout << "Cy(computed) = " << Cy << "\n";
    std::cout << "Cy(expected) = " << expectedCy << "\n";
  }

  {
    dual x = 1.0;
    dual y = 2.0;

    auto fun = []( dual x, dual y) -> dual {
      return hypot( x, x+y, x-y ) * x * y;
    };

    auto Cx = derivative(fun, wrt(x), at(x, y));
    auto Cy = derivative(fun, wrt(y), at(x, y));

    std::cout << "Cx(computed) = " << Cx << "\n";
    std::cout << "Cy(computed) = " << Cy << "\n";

  }

  {
    dual2nd x = 1.0;
    dual2nd y = 2.0;

    auto fun = []( dual2nd x, dual2nd y) -> dual2nd {
      return hypot( x, x+y, x-y ) * x * y;
    };

    {
      auto [C, Cx, Cxx] = derivatives(fun, wrt(x, x), at(x, y));
      std::cout << "C (computed) = " << C << "\n";
      std::cout << "Cx (computed) = " << Cx << "\n";
      std::cout << "Cxx (computed) = " << Cxx << "\n";
    }
    {
      auto [C, Cx, Cxy] = derivatives(fun, wrt(x, y), at(x, y));
      std::cout << "C (computed) = " << C << "\n";
      std::cout << "Cx (computed) = " << Cx << "\n";
      std::cout << "Cxy (computed) = " << Cxy << "\n";
    }
    {
      auto [C, Cy, Cyy] = derivatives(fun, wrt(y, y), at(x, y));
      std::cout << "C (computed) = " << C << "\n";
      std::cout << "Cy (computed) = " << Cy << "\n";
      std::cout << "Cyy (computed) = " << Cyy << "\n";
    }
  }

  {
    dual2nd x = 1.0;
    dual2nd y = 2.0;

    auto fun = []( dual2nd x, dual2nd y) -> dual2nd {
      return log( 1 + x + y*y ) - log1p( x + y*y );
    };

    {
      auto [C, Cx, Cxx] = derivatives(fun, wrt(x, x), at(x, y));
      std::cout << "C (computed) = " << C << "\n";
      std::cout << "Cx (computed) = " << Cx << "\n";
      std::cout << "Cxx (computed) = " << Cxx << "\n";
    }
    {
      auto [C, Cx, Cxy] = derivatives(fun, wrt(x, y), at(x, y));
      std::cout << "C (computed) = " << C << "\n";
      std::cout << "Cx (computed) = " << Cx << "\n";
      std::cout << "Cxy (computed) = " << Cxy << "\n";
    }
    {
      auto [C, Cy, Cyy] = derivatives(fun, wrt(y, y), at(x, y));
      std::cout << "C (computed) = " << C << "\n";
      std::cout << "Cy (computed) = " << Cy << "\n";
      std::cout << "Cyy (computed) = " << Cyy << "\n";
    }
  }

  {
    dual2nd x = 1.1;
    double z = {4.5};
    double pz = power6(z);

    auto fun = []( dual2nd x) -> dual2nd {
      return power6(log1p( x )*x);
    };

    {
      auto [C, Cx, Cxx] = derivatives(fun, wrt(x, x), at(x));
      std::cout << "x*log1p(x)   (computed) = " << C << "\n";
      std::cout << "x*log1p(x)'  (computed) = " << Cx << "\n";
      std::cout << "x*log1p(x)'' (computed) = " << Cxx << "\n";
    }
  }
  {
    dual2nd x = 1.1;

    auto fun = []( dual2nd x) -> dual2nd {
      return power3(cbrt(x));
      //return power4(sqrt(sqrt(x)));
    };
    auto fun2 = []( dual2nd x) -> dual2nd {
      return erf(x)+erfc(x);
    };

    {
      auto [C, Cx, Cxx] = derivatives(fun, wrt(x, x), at(x));
      std::cout << "cbrt(x)^3     (computed) = " << C   << "\n";
      std::cout << "(cbrt(x)^3)'  (computed) = " << Cx  << "\n";
      std::cout << "(cbrt(x)^3)'' (computed) = " << Cxx << "\n";
    }
    {
      auto [C, Cx, Cxx] = derivatives(fun2, wrt(x, x), at(x));
      std::cout << "(erfc(x)+erf(x)) (computed) = " << C   << "\n";
      std::cout << "(erfc(x)+erf(x))' (computed) = " << Cx  << "\n";
      std::cout << "(erfc(x)+erf(x))'' (computed) = " << Cxx << "\n";
    }
  }

}
