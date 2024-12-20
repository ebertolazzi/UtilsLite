/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2017                                                      |
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

#include "Utils.hh"
#include "Utils_fmt.hh"

using std::cout;

static std::atomic<unsigned> accumulator;

class Counter {
  Utils::BinarySearch<int> bs;
public:
  Counter() {
    bool ok ;
    int * pdata = bs.search( std::this_thread::get_id(), ok );
    *pdata = 0;
  }

  void
  inc() {
    bool ok;
    int * pdata = bs.search( std::this_thread::get_id(), ok );
    if ( !ok ) fmt::print("Counter::inc failed thread\n");
    ++(*pdata);
  }

  int
  get() {
    bool ok;
    int * pdata = bs.search( std::this_thread::get_id(), ok );
    if ( !ok ) fmt::print("Counter::inc failed thread\n");
    return *pdata;
  }

  void
  print() {
    bool ok;
    int * pdata = bs.search( std::this_thread::get_id(), ok );
    if ( !ok ) fmt::print("Counter::inc failed thread\n");
    fmt::print(
      "thread {}, counter = {}\n",
      std::this_thread::get_id(),
      *pdata
    );
  }
};

static
void
do_test( int n, int sz ) {
  Counter c;
  int nn = 1+((n*14)%sz);
  //int nn = 40;
  for ( int i = 0; i < nn; ++i ) {
    //Utils::sleep_for_nanoseconds(10);
    c.inc();
  }
  accumulator += c.get();
  //c.print();
}



template <class TP>
void
test_TP( int NN, int nt, int sz, char const * name ) {
  Utils::TicToc tm;

  accumulator = 0;
  TP pool(nt);

  tm.tic();
  for ( int i = 0; i < NN; ++i) pool.run( do_test, i, sz );
  tm.toc();
  double t_launch = tm.elapsed_ms();

  tm.tic();
  pool.wait();
  tm.toc();
  double t_wait = tm.elapsed_ms();

  tm.tic();
  pool.join();
  tm.toc();
  double t_join = tm.elapsed_ms();

  fmt::print(
    "\n[{}] result {} [LAUNCH: {}ms, WAIT {}ms, JOIN {}ms] {}ms\n",
    name, accumulator.load(), t_launch, t_wait, t_join, t_launch+t_wait
  );

  pool.info(cout);
}

int
main( int argc, char *argv[] ) {
  Utils::TicToc tm;

  int NN = 10000;
  int nt = 64;
  int sz = 200;
  int zz = 0;

  if ( argc >= 2 ) nt = atoi( argv[1] );
  if ( argc >= 3 ) sz = atoi( argv[2] );
  if ( argc >= 4 ) NN = atoi( argv[3] );
  if ( argc == 5 ) zz = atoi( argv[3] );

  fmt::print( "NT = {}\n", nt );

  accumulator = 0;
  tm.tic();
  for ( int i = 0; i < NN; ++i) do_test(i,sz);
  tm.toc();
  fmt::print(
    "[No Thread]   result {} [{:.6} ms, AVE = {:.6} mus]\n",
    accumulator.load(), tm.elapsed_ms(), (1000*tm.elapsed_ms())/NN
  );

  test_TP<Utils::ThreadPool0>( NN, nt, sz, "ThreadPool0 [fake]");

  test_TP<Utils::ThreadPool1>( NN, nt, sz, "ThreadPool1");

  test_TP<Utils::ThreadPool2>( NN, nt, sz, "ThreadPool2");

  test_TP<Utils::ThreadPool3>( NN, nt, sz, "ThreadPool3");

  test_TP<Utils::ThreadPool4>( NN, nt, sz, "ThreadPool4");

  test_TP<Utils::ThreadPool5>( NN, nt, sz, "ThreadPool5");

  fmt::print("All done folks!\n\n");

  if ( zz > 0 ) {
    fmt::print("ThreadPool1\n");
    Utils::ThreadPool1 TP1(16); // 0%
    Utils::sleep_for_seconds(4);

    fmt::print("ThreadPool2\n");
    Utils::ThreadPool2 TP2(16); // 0%
    Utils::sleep_for_seconds(4);

    fmt::print("ThreadPool3\n");
    Utils::ThreadPool3 TP3(16); // 100%
    Utils::sleep_for_seconds(4);

    fmt::print("ThreadPool4\n");
    Utils::ThreadPool4 TP4(16); // 100%
    Utils::sleep_for_seconds(4);

    fmt::print("ThreadPool5\n");
    Utils::ThreadPool5 TP5(16); // 0%
    Utils::sleep_for_seconds(4);
  }

  cout << "All done folks!\n\n";

  return 0;
}
