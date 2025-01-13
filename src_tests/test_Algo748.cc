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
 |      Universita` degli Studi di Trento                                   |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Utils_Algo748.hh"
#include "Utils_fmt.hh"

using namespace std;

using Utils::Algo748;

#include "1D_fun.cxx"

static int ntest{0};
static int nfuneval{0};

template <typename FUN>
void
do_solve( string const & name, real_type a, real_type b, FUN f ) {
  Algo748<real_type> solver;
  real_type res  = solver.eval2( a, b, f );
  real_type fres = f(res);
  ++ntest;
  nfuneval += solver.num_fun_eval();
  fmt::print(
    "#{:<3} it:{:<3} #f:{:<3} {} x = {:12} f(x) = {:15} b-a={:10} [{}]\n",
    ntest, solver.used_iter(), solver.num_fun_eval(), solver.converged() ? "YES" : "NO ",
    fmt::format("{:.6}",res),
    fmt::format("{:.3}",fres),
    fmt::format("{:.4}",solver.b()-solver.a()),
    name
  );
}

template <typename FUN>
void
do_solve2( real_type a, real_type b, real_type amin, real_type bmax, FUN f ) {
  Algo748<real_type> solver;
  real_type res = solver.eval2( a, b, amin, bmax, f );
  ++ntest;
  nfuneval += solver.num_fun_eval();
  fmt::print(
    "#{:<3} iter = {:<3} #nfun = {:<3} {} x = {:12} f(x) = {:15} b-a={}\n",
    ntest, solver.used_iter(), solver.num_fun_eval(), solver.converged() ? "YES" : "NO ",
    fmt::format("{:.6}",res),
    fmt::format("{:.3}",f(res)),
    fmt::format("{:.6}",solver.b()-solver.a())
  );
}


int
main() {

  std::vector<std::unique_ptr<fun1D>> f_list;

  build_1dfun_list( f_list );
  
  for ( auto & f : f_list )
    do_solve( f->info(), f->a0(), f->b0(), f->function() );

  //do_solve( "fun_penalty(x,0)",              -1.0, 1.0,    [] ( real_type x ) { return fun_penalty(x,0); } );
  //do_solve( "fun_penalty(x,-10)",            -1.0, 1.0,    [] ( real_type x ) { return fun_penalty(x,-10); } );
  //do_solve2( -1, 1.1498547501802843, -100, 100, [] ( real_type x ) { return fun_penalty(x,-229.970950036057); } );

  fmt::print( "nfuneval {}\n", nfuneval );

  cout << "\nAll Done Folks!\n";

  return 0;
}

