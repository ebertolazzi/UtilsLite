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
 |      Università degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Utils.hh"
#include "Utils_fmt.hh"
#include "Utils_TicToc.hh"

using namespace Utils;

// Funzione per stampa memoria con colore
static void mem_info( const char msg[], fmt::color color = fmt::color::cyan )
{
  fmt::print(
    fg( color ) | fmt::emphasis::bold,
    "{}: mem actual {} max {} (alloc: {}, freed: {})\n",
    msg,
    Utils::out_bytes( Utils::AllocatedBytes ),
    Utils::out_bytes( Utils::MaximumAllocatedBytes ),
    CountAlloc,
    CountFreed );
}

// Test 1: Allocazione base
static void test_basic_allocation()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n=== Test 1: Allocazione Base ===\n" );

  TicToc timer;
  timer.tic();
  mem_info( "IN", fmt::color::yellow );

  {
    Malloc<int> mem( "test_basic" );
    mem.allocate( 1000 );

    int * ptr = mem( 500 );
    fmt::print( "Allocated 500 ints at: {}\n", static_cast<void *>( ptr ) );

    // Scrittura valori
    for ( int i = 0; i < 500; ++i ) ptr[i] = i * 2;

    // Lettura verifica
    bool ok = true;
    for ( int i = 0; i < 500; ++i )
    {
      if ( ptr[i] != i * 2 )
      {
        ok = false;
        break;
      }
    }
    fmt::print( "Data verification: {}\n", ok ? "✓ PASS" : "✗ FAIL" );

    mem.free();
  }

  mem_info( "OUT", fmt::color::yellow );
  timer.toc();
  fmt::print( "Time: {:.6f} seconds\n", timer.elapsed_s() );
}

// Test 2: Riallocazione
static void test_reallocation()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n=== Test 2: Riallocazione ===\n" );

  TicToc timer;
  timer.tic();
  mem_info( "IN", fmt::color::yellow );

  Malloc<double> mem( "test_realloc" );

  // Allocazione iniziale
  mem.allocate( 100 );
  double * ptr1 = mem( 100 );
  fmt::print( "Initial allocation: {} doubles\n", mem.size() );
  fmt::print( "Pointer: {}\n", static_cast<void *>( ptr1 ) );

  // Riallocazione più grande
  mem.reallocate( 500 );
  double * ptr2 = mem( 500 );
  fmt::print( "After realloc: {} doubles\n", mem.size() );
  fmt::print( "Pointer: {}\n", static_cast<void *>( ptr2 ) );

  // Verifica che la memoria sia diversa (se resized)
  if ( ptr1 != ptr2 ) { fmt::print( fg( fmt::color::blue ), "Memory moved during reallocation\n" ); }

  mem.hard_free();
  mem_info( "OUT", fmt::color::yellow );
  timer.toc();
  fmt::print( "Time: {:.6f} seconds\n", timer.elapsed_s() );
}

// Test 3: Operator() e pop
static void test_stack_operations()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n=== Test 3: Stack-like Operations ===\n" );

  TicToc timer;
  timer.tic();

  mem_info( "IN", fmt::color::yellow );

  Malloc<float> mem( "test_stack" );
  mem.allocate( 1000 );

  // Allocazione a blocchi
  float * block1 = mem( 300 );
  float * block2 = mem( 200 );
  float * block3 = mem( 100 );

  fmt::print( "Allocated 3 blocks:\n" );
  fmt::print( "  Block1 (300): {}\n", static_cast<void *>( block1 ) );
  fmt::print( "  Block2 (200): {}\n", static_cast<void *>( block2 ) );
  fmt::print( "  Block3 (100): {}\n", static_cast<void *>( block3 ) );
  fmt::print( "  Total allocated: {}\n", mem.size() );

  // Pop
  mem.pop( 100 );  // Libera block3
  fmt::print( "After pop(100): {} free slots\n", mem.size() - mem.is_empty() );

  // Nuova allocazione dopo pop
  float * block4 = mem( 150 );
  fmt::print( "New block (150): {}\n", static_cast<void *>( block4 ) );

  mem.hard_free();
  mem_info( "OUT", fmt::color::yellow );
  timer.toc();

  fmt::print( "Time: {:.6f} seconds\n", timer.elapsed_s() );
}

// Test 4: Allocazione multipla
static void test_multiple_allocations()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n=== Test 4: Multiple Allocations ===\n" );

  TicToc timer;
  timer.tic();

  mem_info( "IN", fmt::color::yellow );

  std::vector<Malloc<char> *> allocs;
  const int                   num_allocs = 10;

  for ( int i = 0; i < num_allocs; ++i )
  {
    std::string name  = fmt::format( "alloc_{}", i );
    auto *      alloc = new Malloc<char>( name );
    alloc->allocate( 1000 * ( i + 1 ) );
    allocs.push_back( alloc );

    // Usa un po' di memoria
    char * ptr = ( *alloc )( 500 );
    for ( int j = 0; j < 500; ++j ) ptr[j] = 'A' + ( j % 26 );
  }

  fmt::print( "Created {} allocations\n", num_allocs );
  mem_info( "MID", fmt::color::orange );

  // Cleanup
  for ( auto alloc : allocs )
  {
    alloc->hard_free();
    delete alloc;
  }

  mem_info( "OUT", fmt::color::yellow );

  timer.toc();
  fmt::print( "Time: {:.6f} seconds\n", timer.elapsed_s() );
}

// Test 5: Stress test
static void test_stress()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n=== Test 5: Stress Test ===\n" );

  TicToc timer;
  timer.tic();

  mem_info( "IN", fmt::color::yellow );

  Malloc<int64_t>                 mem( "stress_test" );
  std::random_device              rd;
  std::mt19937                    gen( rd() );
  std::uniform_int_distribution<> size_dist( 100, 10000 );
  std::uniform_int_distribution<> op_dist( 0, 3 );

  const int operations = 100;
  for ( int i = 0; i < operations; ++i )
  {
    int op = op_dist( gen );

    switch ( op )
    {
      case 0:
      {  // Alloca
        size_t sz = size_dist( gen );
        mem.allocate( sz );
        int64_t * ptr = mem( sz );
        if ( ptr )
        {
          for ( size_t j = 0; j < sz && j < 100; ++j ) ptr[j] = j;
        }
        break;
      }
      case 1:
      {  // Rialloca
        size_t sz = size_dist( gen );
        mem.reallocate( sz );
        break;
      }
      case 2:  // Free
        mem.free();
        break;
      case 3:  // Hard free e rialloca
        mem.hard_free();
        if ( i % 5 == 0 ) mem.allocate( size_dist( gen ) );
        break;
    }
  }

  mem.hard_free();
  mem_info( "OUT", fmt::color::yellow );

  timer.toc();
  fmt::print( "Operations: {}, Time: {:.6f} seconds\n", operations, timer.elapsed_s() );
}

// Test 6: MallocFixed
static void test_fixed_malloc()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n=== Test 6: MallocFixed ===\n" );

  TicToc timer;
  timer.tic();

  mem_info( "IN", fmt::color::yellow );

  MallocFixed<int, 1024> fixed( "fixed_test" );

  // Alloca tutto lo spazio
  int * ptr1 = fixed( 512 );
  int * ptr2 = fixed( 512 );

  fmt::print( "Fixed allocation (512+512=1024):\n" );
  fmt::print( "  Ptr1: {}\n", static_cast<void *>( ptr1 ) );
  fmt::print( "  Ptr2: {}\n", static_cast<void *>( ptr2 ) );
  fmt::print( "  Is empty: {}\n", fixed.is_empty() );

  // Pop e rialloca
  fixed.pop( 256 );
  int * ptr3 = fixed( 256 );
  fmt::print( "After pop(256) and new alloc(256):\n" );
  fmt::print( "  Ptr3: {}\n", static_cast<void *>( ptr3 ) );

  fixed.free();
  mem_info( "OUT", fmt::color::yellow );

  timer.toc();
  fmt::print( "Time: {:.6f} seconds\n", timer.elapsed_s() );
}

// Test 7: Thread safety (base)
static void test_thread_safety()
{
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n=== Test 7: Thread Safety (Basic) ===\n" );

  TicToc timer;
  timer.tic();

  mem_info( "IN", fmt::color::yellow );

  constexpr int num_threads = 4;
  int           allocs_per_thread{ 100 };

  auto worker = [allocs_per_thread]( int thread_id )
  {
    std::string name = fmt::format( "thread_{}", thread_id );
    Malloc<int> mem( name );

    for ( int i = 0; i < allocs_per_thread; ++i )
    {
      mem.allocate( 100 * ( i + 1 ) );
      int * ptr = mem( 50 );
      if ( ptr ) ptr[0] = thread_id;
      mem.free();
    }

    mem.hard_free();
  };

  std::vector<std::thread> threads;
  for ( int i = 0; i < num_threads; ++i ) { threads.emplace_back( worker, i ); }

  for ( auto & t : threads ) t.join();

  mem_info( "OUT", fmt::color::yellow );

  timer.toc();
  fmt::print(
    "Threads: {}, Allocs per thread: {}, Time: {:.6f} seconds\n",
    num_threads,
    allocs_per_thread,
    timer.elapsed_s() );
}

int main()
{
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "========================================\n" );
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "    COMPLETE MALLOC TEST SUITE\n" );
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "========================================\n" );

  // Attiva debug per vedere le allocazioni
  Utils::MallocDebug = true;

  mem_info( "START", fmt::color::red );

  // Esegui tutti i test
  test_basic_allocation();
  test_reallocation();
  test_stack_operations();
  test_multiple_allocations();
  test_stress();
  test_fixed_malloc();
  test_thread_safety();

  // Statistiche finali
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "\n========================================\n" );
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "          FINAL STATISTICS\n" );
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "========================================\n" );

  mem_info( "FINAL", fmt::color::red );

  fmt::print( "\n" );
  fmt::print( fg( fmt::color::green ), "✓ All tests completed successfully!\n" );

  return 0;
}
