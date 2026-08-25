/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2025                                                      |
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

// Includes the minimal utility headers and the Pareto Front implementation

#include "Utils.hh"
#include "Utils_Pareto.hh"
#include "Utils_fmt.hh"

namespace Test
{

  template <typename T, size_t N> using Point_type = typename Utils::ParetoFront<T, N>::Point_type;

  // Placeholder function to format points for printing
  template <typename T, size_t N> std::string fmt_point( std::array<T, N> const & p )
  {
    std::string s = "{";
    for ( size_t i = 0; i < N; ++i )
    {
      // Using std::to_string, but a custom fmt function would be better for
      // precision
      s += std::to_string( p[i] );
      if ( i < N - 1 ) s += ", ";
    }
    s += "}";
    return s;
  }

  template <typename T> inline void check_test( std::string const & name, T expected, T actual )
  {
    auto const & GREEN{ fmt::fg( fmt::color::green ) };
    auto const & RED{ fmt::fg( fmt::color::red ) };
    if ( expected == actual )
    {
      // Print GREEN using fmt color API
      fmt::print( GREEN, "[ OK ] {} - Expected: {}, Actual: {}\n", name, expected, actual );
    }
    else
    {
      // Print RED using fmt color API
      fmt::print( RED, "[ FAIL ] {} - Expected: {}, Actual: {}\n", name, expected, actual );
    }
  }

  void check_test_bool( std::string const & name, bool expected, bool actual )
  {
    // More readable formatting for boolean
    std::string  s_expected{ expected ? "true" : "false" };
    std::string  s_actual{ actual ? "true" : "false" };
    auto const & GREEN{ fmt::fg( fmt::color::green ) };
    auto const & RED{ fmt::fg( fmt::color::red ) };

    if ( expected == actual )
    {
      // Print GREEN using fmt color API
      fmt::print( GREEN, "[ OK ] {} - Expected: {}, Actual: {}\n", name, s_expected, s_actual );
    }
    else
    {
      // Print RED using fmt color API
      fmt::print( RED, "[ FAIL ] {} - Expected: {}, Actual: {}\n", name, s_expected, s_actual );
    }
  }

  using Real = double;

  // Alias for the ParetoFront class with various template parameters
  template <std::size_t N, typename Payload>
  using PF = Utils::ParetoFront<Real, N, Payload, 8>;  // RebuildThreshold = 8
                                                       // for easier testing

  /**
   * \brief Helper function to print the front's state.
   */
  template <std::size_t N, typename Payload>
  inline void print_front_state( std::string const & title, PF<N, Payload> const & pf )
  {
    fmt::print(
      "\n--- {} ---\n"
      "Front Size: {}, Raw Size (incl. tombstones): {}, Empty: {}\n"
      "Live Points:\n",
      title,
      pf.size(),
      pf.raw_size(),
      pf.empty() );
    int count = 0;
    pf.for_each_alive(
      [&]( auto const & e )
      {
        fmt::print( "  ID={} P={}", e.id, fmt_point( e.p ) );
        // Payload printing, handles both monostate (void) and custom type
        if constexpr ( !std::is_same<Payload, void>::value ) { fmt::print( " Payload={}", e.payload ); }
        fmt::print( "\n" );
        ++count;
      } );
    fmt::print( "Total Live Count: {}\n", count );
  }

  /**
   * \brief Function to run the 2D minimization test.
   */
  void test_2d_minimization()
  {
    fmt::print(
      "\n"
      "======================================================\n"
      "      TEST 1: 2D MINIMIZATION (Payload=void)          \n"
      "======================================================\n" );

    PF<2, void> pf;

    std::map<std::size_t, Point_type<Real, 2>> points;

    // P1: {1.0, 5.0} -> Front: {P1}
    auto r1 = pf.insert( { 1.0, 5.0 } );
    check_test_bool( "1. Insert P1 (1.0, 5.0)", true, r1.first );
    points[r1.second] = { 1.0, 5.0 };

    // P2: {5.0, 1.0} -> Front: {P1, P2}
    auto r2 = pf.insert( { 5.0, 1.0 } );
    check_test_bool( "2. Insert P2 (5.0, 1.0)", true, r2.first );
    points[r2.second] = { 5.0, 1.0 };

    // P3: {3.0, 3.0} -> Non dominated (x between P1/P2, y between P2/P1)
    // Front: {P1, P2, P3}
    auto r3 = pf.insert( { 3.0, 3.0 } );
    check_test_bool( "3. Insert P3 (3.0, 3.0)", true, r3.first );
    points[r3.second] = { 3.0, 3.0 };

    // P4: {4.0, 4.0} -> Dominated. Front: {P1, P2, P3, P4}
    auto r4 = pf.insert( { 4.0, 4.0 } );
    check_test_bool( "4. Insert P4 (4.0, 4.0)", false, r4.first );
    points[r4.second] = { 4.0, 4.0 };

    // P5: {6.0, 6.0} -> Dominated by P1, P2, P3, P4. Should be rejected.
    auto r5 = pf.insert( { 6.0, 6.0 } );
    check_test_bool( "5. Insert P5 (6.0, 6.0) (Dominated)", false, r5.first );

    // P6: {0.5, 0.5} -> Should dominate P1, P2, P3, P4. Front: {P6}
    auto r6 = pf.insert( { 0.5, 0.5 } );
    check_test_bool( "6. Insert P6 (0.5, 0.5) (Dominator)", true, r6.first );
    points[r6.second] = { 0.5, 0.5 };

    // Check dominance check
    Point_type<Real, 2> dominated_point = { 10.0, 10.0 };
    check_test_bool( "7. is_dominated_by_front (10, 10)", true, pf.is_dominated_by_front( dominated_point ) );

    Point_type<Real, 2> non_dominated_point = { 0.1, 100.0 };
    check_test_bool( "8. is_dominated_by_front (0.1, 100)", false, pf.is_dominated_by_front( non_dominated_point ) );

    // Final size check
    check_test( "9. Final Front Size (Expected 1)", std::size_t( 1 ), pf.size() );
    print_front_state( "Final 2D Minimization Front", pf );

    // Check nearest neighbor (should be P6)
    auto nearest = pf.find_nearest( { 0.6, 0.6 } );
    check_test_bool( "10. Find Nearest Found", true, nearest.first );
    check_test( "11. Find Nearest ID (Expected r6)", r6.second, nearest.second );

    // Clear and check empty
    pf.clear();
    check_test( "12. After Clear Size (Expected 0)", std::size_t( 0 ), pf.size() );
    check_test_bool( "13. After Clear Empty", true, pf.empty() );
  }

  /**
   * \brief Function to run the 3D minimization test with Payload.
   */
  void test_3d_with_payload()
  {
    fmt::print(
      "\n"
      "======================================================\n"
      "  TEST 2: MINIMIZE 3D (Payload=int) + Rebuild         \n"
      "======================================================\n" );

    // Maximize x, Minimize y, Maximize z
    PF<3, int> pf;

    // Points for the front
    // P1: {10.0, 2.0, 10.0} (ID 1)
    /* auto r1 = */ pf.insert( { -10.0, 2.0, -10.0 }, 100 );
    // P2: {2.0, 1.0, 20.0} (ID 2)
    /* auto r2 = */ pf.insert( { -2.0, 1.0, -20.0 }, 200 );
    // P3: {5.0, 5.0, 5.0} (ID 3) - This will be dominated by P1
    /* auto r3 = */ pf.insert( { -5.0, 5.0, -5.0 }, 300 );

    // P3 is dominated by P1, so the front size should be 2
    check_test( "1. Initial Size (Expected 2)", std::size_t( 2 ), pf.size() );

    // P4: {1.0, 10.0, 1.0} -> Non dominated
    /* auto r4 = */ pf.insert( { -1.0, 10.0, -1.0 }, 400 );

    // P5: {15.0, 1.5, 15.0} -> Dominates P1, P3. P5 does NOT dominate P2.
    // Front: {P2, P5}
    auto r5 = pf.insert( { -15.0, 1.5, -15.0 }, 500 );

    check_test_bool( "2. P5 Inserted (and dominated P1, P3)", true, r5.first );
    check_test( "3. Size after P5 (Expected 2: P2, P5)", std::size_t( 2 ), pf.size() );

    // P6: {15.0, 1.5, 15.0} (Duplicate, should be rejected)
    auto r6 = pf.insert( { -15.0, 1.5, -15.0 }, 600 );
    check_test_bool( "4. P6 Duplicate (Dominated)", false, r6.first );

    // Check domination using the specialized rule (Max, Min, Max)
    Point_type<Real, 3> dominated = { -1.0, 10.0, -1.0 };
    // P5 (15, 1.5, 15) dominates (1, 10, 1). P5[0]>1 (ok), P5[1]<10 (ok),
    // P5[2]>1 (ok). YES.
    check_test_bool( "5. is_dominated_by_front (-1, 10, -1)", true, pf.is_dominated_by_front( dominated ) );

    // P7: {15.0, 1.0, 20.0} -> Dominates P2 (2, 1, 20) and P5 (15, 1.5, 15).
    // Front: {P7}
    auto r7 = pf.insert( { -15.0, 1.0, -20.0 }, 700 );

    // Rebuild is likely triggered here (since RebuildThreshold=8 and we have 7
    // inserts + tombstones)
    check_test( "6. Size after P7 (Expected 1: P7)", std::size_t( 1 ), pf.size() );
    print_front_state( "Intermediate 3D Front (Should be only P7)", pf );

    // Check payload retrieval
    auto payload_check = pf.get_payload_by_id( r7.second );
    check_test_bool( "7. Get P7 Payload Found", true, payload_check.first );
    check_test( "8. Get P7 Payload Value (Expected 700)", 700, payload_check.second );

    // Test nearest neighbor on P7
    auto nearest = pf.find_nearest( { -15.1, 0.9, -20.1 } );
    check_test_bool( "9. Find Nearest Found", true, nearest.first );
    check_test( "10. Find Nearest ID (Expected r7)", r7.second, nearest.second );
  }

  /**
   * \brief Function to run the batch build and ID removal test.
   */
  void test_batch_build_and_erase()
  {
    fmt::print(
      "\n"
      "======================================================\n"
      "      TEST 3: BATCH BUILD & ID ERASE (2D)             \n"
      "======================================================\n" );

    PF<2, int> pf;  // Minimize x, Minimize y

    std::vector<std::pair<Point_type<Real, 2>, int>> batch_pts = {
      { { 10.0, 1.0 }, 1 },   // A
      { { 1.0, 10.0 }, 2 },   // B
      { { 5.0, 5.0 }, 3 },    // C
      { { 11.0, 11.0 }, 4 },  // D (Dominated)
      { { 0.5, 0.5 }, 5 },    // E (Dominator)
      { { 2.0, 2.0 }, 6 },    // F (Dominated)
    };

    pf.batch_build( batch_pts );

    // After batch build, only E should have survived.
    check_test( "1. Batch Build Size (Expected 1)", std::size_t( 1 ), pf.size() );

    // Run another build for a front with more than one point
    std::vector<std::pair<Point_type<Real, 2>, int>> batch_pts_2 = {
      { { 10.0, 1.0 }, 10 },
      { { 1.0, 10.0 }, 20 },
      { { 5.0, 5.0 }, 30 },
    };
    pf.batch_build( batch_pts_2 );

    // All three points are non-dominated relative to each other
    check_test( "2. Batch Build 2 Size (Expected 3)", std::size_t( 3 ), pf.size() );

    // Add points - these might dominate or be dominated by existing points
    /* auto r_a = */ pf.insert( { 0.1, 10.0 }, 10 );
    /* auto r_b = */ pf.insert( { 10.0, 0.1 }, 20 );
    // The final front depends on the dominance relationships
    check_test( "3. Size after inserts", std::size_t( 3 ), pf.size() );

    // Find the ID of a point to remove (e.g., {10.0, 0.1} with payload 20)
    std::size_t id_to_remove = 0;
    pf.for_each_alive(
      [&]( auto const & e )
      {
        if ( e.payload == 20 && std::abs( e.p[0] - 10.0 ) < 1e-9 && std::abs( e.p[1] - 0.1 ) < 1e-9 )
        {
          id_to_remove = e.id;
        }
      } );

    check_test_bool( "4. ID to Remove Found (Expected true)", true, id_to_remove != 0 );

    // Remove the found point
    bool erased = pf.erase_by_id( id_to_remove );
    check_test_bool( "5. Erase by ID (Expected true)", true, erased );
    check_test( "6. Size after Erase", std::size_t( 2 ), pf.size() );

    // Remove a non-existent ID (error check)
    bool erased_fail = pf.erase_by_id( 9999 );
    check_test_bool( "7. Erase non-existent ID (Expected false)", false, erased_fail );

    // Remove the remaining points
    std::vector<std::size_t> ids_to_remove;
    pf.for_each_alive( [&]( auto const & e ) { ids_to_remove.push_back( e.id ); } );

    for ( auto id : ids_to_remove ) { pf.erase_by_id( id ); }
    check_test( "8. Final Size after all erase (Expected 0)", std::size_t( 0 ), pf.size() );

    // Verify that Raw Size contains the tombstones
    check_test( "9. Final Raw Size", std::size_t( 5 ), pf.raw_size() );

    print_front_state( "Final Batch/Erase Test State", pf );
  }

  /**
   * \brief Function to run the forced ND-Tree rebuild test.
   */
  void test_tree_rebuild()
  {
    fmt::print(
      "\n"
      "======================================================\n"
      "      TEST 4: FORCED TREE REBUILD (N=2)               \n"
      "======================================================\n" );

    // RebuildThreshold is set to 8
    PF<2, void> pf;

    std::size_t id_list[10] = { 0 };

    // 1. Insert 8 points (should trigger the first rebuild)
    for ( int i = 0; i < 8; ++i )
    {
      // Non-dominated points
      auto r     = pf.insert( { 10.0 - (Real) i, 1.0 + (Real) i } );
      id_list[i] = r.second;
    }

    check_test( "1. Size after 8 Inserts (Expected 8)", std::size_t( 8 ), pf.size() );
    // The raw size SHOULD be 8, and the tree is rebuilt.
    check_test( "2. Raw Size (Expected 8)", std::size_t( 8 ), pf.raw_size() );

    // 2. Insert 4 dominated points (Do not count for rebuild, but confirm
    // dominance)
    for ( int i = 0; i < 4; ++i ) pf.insert( { 20.0, 20.0 } );
    check_test( "3. Size after 4 Dominated Inserts (Expected 8)", std::size_t( 8 ), pf.size() );

    // 3. Erase 8 points (tombstones) (should force rebuild for tombstones >= 8)
    for ( int i = 0; i < 8; ++i ) pf.erase_by_id( id_list[i] );

    // After 8 removals, the live front is 0, but raw size 8, tombstones 8.
    check_test( "4. Size after 8 Erases (Expected 0)", std::size_t( 0 ), pf.size() );

    // 4. Insert the 9th point (should force rebuild because tombstones == 8 >=
    // threshold)
    /* auto r_new = */ pf.insert( { 1.0, 1.0 } );  // P9

    // After rebuild:
    // Raw Size should be 1 (only P9)
    check_test( "5. Raw Size after Rebuild (Expected 1)", std::size_t( 1 ), pf.raw_size() );
    check_test( "6. Live Size after Rebuild (Expected 1)", std::size_t( 1 ), pf.size() );

    print_front_state( "Final Tree Rebuild State (Should be 1 point)", pf );
  }
}  // namespace Test

/**
 * \brief Main function for executing all tests.
 */
int main()
{
  try
  {
    Test::test_2d_minimization();
    Test::test_3d_with_payload();
    Test::test_batch_build_and_erase();
    Test::test_tree_rebuild();

    // Print the final message in green
    fmt::print(
      fmt::fg( fmt::color::green ),
      "\n"
      "======================================================\n"
      "      All tests completed successfully.               \n"
      "======================================================\n" );
  }
  catch ( const std::exception & e )
  {
    // Print the error message in red
    fmt::print( fmt::fg( fmt::color::red ), "Exception caught: {}\n", e.what() );
    return 1;
  }
  catch ( ... )
  {
    // Print the unknown error message in red
    fmt::print( fmt::fg( fmt::color::red ), "Unknown exception caught.\n" );
    return 1;
  }
  return 0;
}
