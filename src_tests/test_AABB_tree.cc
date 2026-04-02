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

#include "Utils_AABB_tree.hh"
#include "Utils_GG2D.hh"
#include "Utils_fmt.hh"
#include "Utils_TicToc.hh"

using namespace std;
using integer   = int;
using real_type = double;

// ===========================================================================
// Helper Functions
// ===========================================================================

static unsigned     seed1 = 2;
static std::mt19937 generator( seed1 );

using Utils::AABBtree;
using Utils::m_pi;
using Utils::Point2D;
using Utils::Segment2D;
using Utils::TicToc;

static real_type rand_uniform( real_type const xmin, real_type const xmax )
{
  real_type const random{ static_cast<real_type>( generator() ) / generator.max() };
  return xmin + ( xmax - xmin ) * random;
}

// static real_type rand_normal( real_type mean, real_type stddev )
//{
//   std::normal_distribution<real_type> dist( mean, stddev );
//   return dist( generator );
// }

// Brute-force verification functions
template <typename Real> std::set<integer> brute_force_point_intersection(
  Real const pnt[],
  Real const bb_min[],
  Real const bb_max[],
  integer    nbox,
  integer    dim )
{
  std::set<integer> result;
  for ( integer i = 0; i < nbox; ++i )
  {
    bool inside = true;
    for ( integer d = 0; d < dim; ++d )
    {
      Real min_val = bb_min[i * dim + d];
      Real max_val = bb_max[i * dim + d];
      if ( pnt[d] < min_val || pnt[d] > max_val )
      {
        inside = false;
        break;
      }
    }
    if ( inside ) result.insert( i );
  }
  return result;
}

template <typename Real> std::set<integer> brute_force_bbox_intersection(
  Real const bbox_min[],
  Real const bbox_max[],
  Real const bb_min[],
  Real const bb_max[],
  integer    nbox,
  integer    dim )
{
  std::set<integer> result;
  for ( integer i = 0; i < nbox; ++i )
  {
    bool overlap = true;
    for ( integer d = 0; d < dim; ++d )
    {
      Real a_min = bbox_min[d];
      Real a_max = bbox_max[d];
      Real b_min = bb_min[i * dim + d];
      Real b_max = bb_max[i * dim + d];
      if ( a_max < b_min || a_min > b_max )
      {
        overlap = false;
        break;
      }
    }
    if ( overlap ) result.insert( i );
  }
  return result;
}

// Verifica se set1 è superset di set2
template <typename T> bool is_superset( const std::set<T> & set1, const std::set<T> & set2 )
{
  return std::includes( set1.begin(), set1.end(), set2.begin(), set2.end() );
}

// ===========================================================================
// Test Cases
// ===========================================================================

void test_basic_functionality()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 1: Basic Functionality        │\n"
    "└────────────────────────────────────────┘\n" );

  // Test empty tree
  {
    AABBtree<real_type> empty_tree;
    fmt::print( "  • Empty tree test... " );
    std::set<integer> results;
    real_type         pnt[2] = { 0.5, 0.5 };
    empty_tree.intersect_with_one_point( pnt, results );
    if ( results.empty() )
      fmt::print( fg( fmt::color::green ), "✓ PASS\n" );
    else
      fmt::print( fg( fmt::color::red ), "✗ FAIL\n" );
  }

  // Test single bounding box
  {
    constexpr integer nbox               = 1;
    constexpr integer dim                = 2;
    real_type         bb_min[nbox * dim] = { 0, 0 };
    real_type         bb_max[nbox * dim] = { 1, 1 };

    AABBtree<real_type> tree;
    tree.build( bb_min, dim, bb_max, dim, nbox, dim );

    fmt::print( "  • Single bbox tree test... " );

    // Point inside
    std::set<integer> results;
    real_type         pnt_inside[2] = { 0.5, 0.5 };
    tree.intersect_with_one_point( pnt_inside, results );
    bool test1 = ( results.size() == 1 && *results.begin() == 0 );

    // Point outside
    results.clear();
    real_type pnt_outside[2] = { 2, 2 };
    tree.intersect_with_one_point( pnt_outside, results );
    bool test2 = results.empty();

    if ( test1 && test2 )
      fmt::print( fg( fmt::color::green ), "✓ PASS\n" );
    else
      fmt::print( fg( fmt::color::red ), "✗ FAIL\n" );
  }

  // Test tree copy constructor
  {
    constexpr integer nbox = 10;
    constexpr integer dim  = 2;

    real_type bb_min[nbox * dim];
    real_type bb_max[nbox * dim];

    for ( integer i = 0; i < nbox; ++i )
    {
      bb_min[i * dim]     = i;  // Box non sovrapposti per test semplice
      bb_min[i * dim + 1] = 0;
      bb_max[i * dim]     = i + 1;
      bb_max[i * dim + 1] = 1;
    }

    AABBtree<real_type> tree1;
    tree1.build( bb_min, dim, bb_max, dim, nbox, dim );

    AABBtree<real_type> tree2( tree1 );  // Copy constructor

    fmt::print( "  • Copy constructor test... " );

    // Test con punti specifici
    bool all_passed = true;
    for ( integer i = 0; i < nbox; ++i )
    {
      real_type         pnt[2] = { i + 0.5, 0.5 };
      std::set<integer> res1, res2;
      tree1.intersect_with_one_point_and_refine( pnt, res1 );
      tree2.intersect_with_one_point_and_refine( pnt, res2 );

      if ( res1 != res2 || res1.size() != 1 || *res1.begin() != i )
      {
        all_passed = false;
        break;
      }
    }

    if ( all_passed && tree1.num_tree_nodes() == tree2.num_tree_nodes() )
      fmt::print( fg( fmt::color::green ), "✓ PASS\n" );
    else
      fmt::print( fg( fmt::color::red ), "✗ FAIL\n" );
  }
}

void test_parameter_settings()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 2: Parameter Settings         │\n"
    "└────────────────────────────────────────┘\n" );

  constexpr integer nbox = 1000;
  constexpr integer dim  = 2;

  real_type bb_min[nbox * dim];
  real_type bb_max[nbox * dim];

  for ( integer i = 0; i < nbox; ++i )
  {
    bb_min[i * dim]     = rand_uniform( 0, 8 );
    bb_min[i * dim + 1] = rand_uniform( 0, 8 );
    bb_max[i * dim]     = bb_min[i * dim] + rand_uniform( 0.1, 2 );
    bb_max[i * dim + 1] = bb_min[i * dim + 1] + rand_uniform( 0.1, 2 );
  }

  // Test different parameters
  fmt::print( "  Testing different tree configurations:\n" );

  std::vector<std::tuple<integer, real_type, real_type, real_type>> configs = { { 8, 0.7, 0.05, 0.01 },
                                                                                { 16, 0.8, 0.1, 0.0 },
                                                                                { 32, 0.9, 0.2, 0.05 },
                                                                                { 64, 0.6, 0.15, 0.02 } };

  fmt::print( "  ┌─────────┬──────┬─────────┬─────────┬─────────┬────────────┐\n" );
  fmt::print( "  │ Config  │ Max  │ Ratio   │ Overlap │ MinSize │ Build Time │\n" );
  fmt::print( "  │ ID      │ Obj  │         │ Tol     │ Tol     │ (ms)       │\n" );
  fmt::print( "  ├─────────┼──────┼─────────┼─────────┼─────────┼────────────┤\n" );

  TicToc  timer;
  integer config_id = 1;

  for ( auto const & [max_obj, ratio, overlap, min_size] : configs )
  {
    AABBtree<real_type> tree;
    tree.set_max_num_objects_per_node( max_obj );
    tree.set_bbox_long_edge_ratio( ratio );
    tree.set_bbox_overlap_tolerance( overlap );
    tree.set_bbox_min_size_tolerance( min_size );

    timer.tic();
    tree.build( bb_min, dim, bb_max, dim, nbox, dim );
    timer.toc();

    fmt::print(
      "  │ {:7} │ {:4} │ {:7.3} │ {:7.3} │ {:7.3} │ {:10.2} │\n",
      config_id,
      max_obj,
      ratio,
      overlap,
      min_size,
      timer.elapsed_ms() );

    config_id++;
  }

  fmt::print( "  └─────────┴──────┴─────────┴─────────┴─────────┴────────────┘\n" );
}

void test_point_intersection_comprehensive()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 3: Point Intersection         │\n"
    "└────────────────────────────────────────┘\n" );

  constexpr integer nbox = 1000;
  constexpr integer dim  = 2;

  real_type bb_min[nbox * dim];
  real_type bb_max[nbox * dim];

  // Generate random bounding boxes
  for ( integer i = 0; i < nbox; ++i )
  {
    bb_min[i * dim]     = rand_uniform( 0, 8 );
    bb_min[i * dim + 1] = rand_uniform( 0, 8 );
    bb_max[i * dim]     = bb_min[i * dim] + rand_uniform( 0.1, 2 );
    bb_max[i * dim + 1] = bb_min[i * dim + 1] + rand_uniform( 0.1, 2 );
  }

  AABBtree<real_type> tree;
  tree.build( bb_min, dim, bb_max, dim, nbox, dim );

  fmt::print( "  Tree built with {} bounding boxes\n", nbox );

  // Test multiple query points
  std::vector<std::array<real_type, 2>> test_points = { { 1.0, 1.0 }, { 5.0, 5.0 },   { 9.0, 9.0 },
                                                        { 0.0, 0.0 }, { 10.0, 10.0 }, { 3.2, 4.7 } };

  integer passed_basic   = 0;
  integer passed_refined = 0;
  integer total          = static_cast<integer>( test_points.size() );

  TicToc timer;

  // Test basic intersection (candidates should be superset)
  fmt::print( "  ├─ Basic Intersection (candidates) ───────────\n" );
  timer.tic();

  for ( auto const & pnt : test_points )
  {
    // Tree query (basic - returns candidates)
    std::set<integer> tree_results;
    tree.intersect_with_one_point( pnt.data(), tree_results );

    // Brute force verification (exact results)
    std::set<integer> brute_results = brute_force_point_intersection( pnt.data(), bb_min, bb_max, nbox, dim );

    // Basic version returns candidates, so it should be a superset of exact results
    if ( is_superset( tree_results, brute_results ) ) { passed_basic++; }
    else
    {
      fmt::print( fg( fmt::color::yellow ), "    Basic mismatch at point ({}, {})\n", pnt[0], pnt[1] );
      fmt::print( "      Tree found: {} boxes, Brute force: {} boxes\n", tree_results.size(), brute_results.size() );
    }
  }

  timer.toc();
  fmt::print( "  │ Basic: {}/{} tests passed (candidates are supersets)\n", passed_basic, total );
  fmt::print( "  │ Time: {:.2f} ms\n", timer.elapsed_ms() );

  // Test refined intersection (should match exactly)
  fmt::print( "  ├─ Refined Intersection (exact) ──────────────\n" );
  timer.tic();

  for ( auto const & pnt : test_points )
  {
    // Tree query (refined - returns exact results)
    std::set<integer> tree_results;
    tree.intersect_with_one_point_and_refine( pnt.data(), tree_results );

    // Brute force verification (exact results)
    std::set<integer> brute_results = brute_force_point_intersection( pnt.data(), bb_min, bb_max, nbox, dim );

    if ( tree_results == brute_results ) { passed_refined++; }
    else
    {
      fmt::print( fg( fmt::color::yellow ), "    Refined mismatch at point ({}, {})\n", pnt[0], pnt[1] );
      fmt::print( "      Tree found: {} boxes, Brute force: {} boxes\n", tree_results.size(), brute_results.size() );
    }
  }

  timer.toc();

  fmt::print( "  │ Refined: {}/{} tests passed (exact match)\n", passed_refined, total );
  fmt::print( "  │ Total time: {:.2f} ms\n", timer.elapsed_ms() );
  fmt::print( "  │ Avg time per query: {:.3f} ms\n", timer.elapsed_ms() / total );

  if ( passed_refined == total )
    fmt::print( fg( fmt::color::green ), "  ✓ All point intersection tests passed!\n" );
  else
    fmt::print( fg( fmt::color::red ), "  ✗ Some refined tests failed\n" );
}

void test_bbox_intersection_comprehensive()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 4: BBox Intersection          │\n"
    "└────────────────────────────────────────┘\n" );

  constexpr integer nbox = 1000;
  constexpr integer dim  = 2;

  real_type bb_min[nbox * dim];
  real_type bb_max[nbox * dim];

  for ( integer i = 0; i < nbox; ++i )
  {
    bb_min[i * dim]     = rand_uniform( 0, 8 );
    bb_min[i * dim + 1] = rand_uniform( 0, 8 );
    bb_max[i * dim]     = bb_min[i * dim] + rand_uniform( 0.1, 2 );
    bb_max[i * dim + 1] = bb_min[i * dim + 1] + rand_uniform( 0.1, 2 );
  }

  AABBtree<real_type> tree;
  tree.build( bb_min, dim, bb_max, dim, nbox, dim );

  // Test multiple query bounding boxes
  std::vector<std::pair<std::array<real_type, 2>, std::array<real_type, 2>>> test_bboxes = {
    { { 0, 0 }, { 1, 1 } },   { { 2, 2 }, { 4, 4 } },         { { 5, 5 }, { 7, 7 } },
    { { 8, 8 }, { 10, 10 } }, { { 1.5, 1.5 }, { 3.5, 3.5 } }, { { 6, 2 }, { 8, 4 } }
  };

  integer passed_basic   = 0;
  integer passed_refined = 0;
  integer total          = static_cast<integer>( test_bboxes.size() );

  TicToc timer;

  // Test basic intersection (candidates should be superset)
  fmt::print( "  ├─ Basic Intersection (candidates) ───────────\n" );
  timer.tic();

  for ( auto const & [bbox_min, bbox_max] : test_bboxes )
  {
    // Tree query (basic - returns candidates)
    std::set<integer> tree_results;
    tree.intersect_with_one_bbox( bbox_min.data(), tree_results );

    // Brute force verification (exact results)
    std::set<integer> brute_results =
      brute_force_bbox_intersection( bbox_min.data(), bbox_max.data(), bb_min, bb_max, nbox, dim );

    // Basic version returns candidates, so it should be a superset of exact results
    if ( is_superset( tree_results, brute_results ) ) { passed_basic++; }
    else
    {
      fmt::print(
        fg( fmt::color::yellow ),
        "    Basic mismatch for bbox [{},{}]-[{},{}]\n",
        bbox_min[0],
        bbox_min[1],
        bbox_max[0],
        bbox_max[1] );
    }
  }

  timer.toc();
  fmt::print( "  │ Basic: {}/{} tests passed (candidates are supersets)\n", passed_basic, total );
  fmt::print( "  │ Time: {:.2f} ms\n", timer.elapsed_ms() );

  // Test refined intersection (should match exactly)
  fmt::print( "  ├─ Refined Intersection (exact) ──────────────\n" );
  timer.tic();

  for ( auto const & [bbox_min, bbox_max] : test_bboxes )
  {
    std::set<integer> tree_results;
    tree.intersect_with_one_bbox_and_refine( bbox_min.data(), tree_results );

    std::set<integer> brute_results =
      brute_force_bbox_intersection( bbox_min.data(), bbox_max.data(), bb_min, bb_max, nbox, dim );

    if ( tree_results == brute_results ) passed_refined++;
  }

  timer.toc();

  fmt::print( "  │ Refined: {}/{} passed (exact match), Time: {:.2f} ms\n", passed_refined, total, timer.elapsed_ms() );

  if ( passed_refined == total ) fmt::print( fg( fmt::color::green ), "  ✓ All bbox intersection tests passed!\n" );
}

void test_tree_intersection_comprehensive()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 5: Tree-Tree Intersection     │\n"
    "└────────────────────────────────────────┘\n" );

  constexpr integer nbox1 = 500;
  constexpr integer nbox2 = 500;
  constexpr integer dim   = 2;

  // Generate two sets of bounding boxes
  real_type bb_min1[nbox1 * dim];
  real_type bb_max1[nbox1 * dim];
  real_type bb_min2[nbox2 * dim];
  real_type bb_max2[nbox2 * dim];

  // First tree: boxes in [0,5] x [0,5]
  for ( integer i = 0; i < nbox1; ++i )
  {
    bb_min1[i * dim]     = rand_uniform( 0, 5 );
    bb_min1[i * dim + 1] = rand_uniform( 0, 5 );
    bb_max1[i * dim]     = bb_min1[i * dim] + rand_uniform( 0.1, 1.5 );
    bb_max1[i * dim + 1] = bb_min1[i * dim + 1] + rand_uniform( 0.1, 1.5 );
  }

  // Second tree: boxes in [3,8] x [3,8] (overlap with first tree)
  for ( integer i = 0; i < nbox2; ++i )
  {
    bb_min2[i * dim]     = rand_uniform( 3, 8 );
    bb_min2[i * dim + 1] = rand_uniform( 3, 8 );
    bb_max2[i * dim]     = bb_min2[i * dim] + rand_uniform( 0.1, 1.5 );
    bb_max2[i * dim + 1] = bb_min2[i * dim + 1] + rand_uniform( 0.1, 1.5 );
  }

  AABBtree<real_type> tree1, tree2;
  tree1.build( bb_min1, dim, bb_max1, dim, nbox1, dim );
  tree2.build( bb_min2, dim, bb_max2, dim, nbox2, dim );

  fmt::print( "  Built two trees with {} and {} bounding boxes\n", nbox1, nbox2 );

  TicToc timer;

  // Test basic intersection (returns candidates) -> AABB_MAP keys are object indices
  fmt::print( "  ├─ Basic Intersection (candidates) ───────────\n" );
  timer.tic();
  std::map<integer, std::set<integer>> candidate_results;  // object in tree1 -> set of candidate objects in tree2
  tree1.intersect( tree2, candidate_results );
  timer.toc();

  integer total_candidate_pairs = 0;
  for ( auto const & [k, v] : candidate_results ) total_candidate_pairs += static_cast<integer>( v.size() );

  fmt::print( "  │ Found {} candidate pairs\n", total_candidate_pairs );
  fmt::print( "  │ Time: {:.2f} ms\n", timer.elapsed_ms() );
  fmt::print( "  │ Number of checks: {}\n", tree1.num_check() );

  // Test refined intersection (returns exact results)
  fmt::print( "  ├─ Refined Intersection (exact) ──────────────\n" );
  timer.tic();
  std::map<integer, std::set<integer>> refined_results;
  tree1.intersect_and_refine( tree2, refined_results );
  timer.toc();

  integer total_exact_pairs = 0;
  for ( auto const & [k, v] : refined_results ) total_exact_pairs += static_cast<integer>( v.size() );

  fmt::print( "  │ Found {} exact intersecting pairs\n", total_exact_pairs );
  fmt::print( "  │ Time: {:.2f} ms\n", timer.elapsed_ms() );

  // VERIFICATION: Compute exact intersections using brute force
  fmt::print( "  ├─ Brute Force Verification ──────────────────\n" );
  timer.tic();

  // Compute exact intersections using brute force
  std::map<integer, std::set<integer>> brute_results;
  integer                              brute_pairs = 0;

  for ( integer i = 0; i < nbox1; ++i )
  {
    real_type const * bbox_i_min = bb_min1 + i * dim;
    real_type const * bbox_i_max = bb_max1 + i * dim;

    for ( integer j = 0; j < nbox2; ++j )
    {
      real_type const * bbox_j_min = bb_min2 + j * dim;
      real_type const * bbox_j_max = bb_max2 + j * dim;

      bool overlap = true;
      for ( integer d = 0; d < dim; ++d )
      {
        if ( bbox_i_max[d] < bbox_j_min[d] || bbox_i_min[d] > bbox_j_max[d] )
        {
          overlap = false;
          break;
        }
      }

      if ( overlap )
      {
        brute_results[i].insert( j );
        brute_pairs++;
      }
    }
  }
  timer.toc();

  fmt::print( "  │ Brute force found {} pairs\n", brute_pairs );
  fmt::print( "  │ Brute force time: {:.2f} ms\n", timer.elapsed_ms() );


  // Convert both results to sets of pairs for order-independent comparison
  std::set<std::pair<integer, integer>> refined_pairs;
  for ( const auto & [i, js] : refined_results )
  {
    for ( integer j : js ) { refined_pairs.insert( { i, j } ); }
  }

  std::set<std::pair<integer, integer>> brute_pairs_set;
  for ( const auto & [i, js] : brute_results )
  {
    for ( integer j : js ) { brute_pairs_set.insert( { i, j } ); }
  }

  // Check if refined_pairs matches brute_pairs_set exactly
  bool exact_match = ( refined_pairs == brute_pairs_set );

  if ( exact_match ) { fmt::print( fg( fmt::color::green ), "  │ ✓ Refined results match brute force exactly\n" ); }
  else
  {
    fmt::print( fg( fmt::color::red ), "  │ ✗ Refined results DO NOT match brute force!\n" );

    // Find differences
    std::vector<std::pair<integer, integer>> missing_in_refined;
    std::vector<std::pair<integer, integer>> extra_in_refined;

    // Find pairs in brute but not in refined
    std::set_difference(
      brute_pairs_set.begin(),
      brute_pairs_set.end(),
      refined_pairs.begin(),
      refined_pairs.end(),
      std::back_inserter( missing_in_refined ) );

    // Find pairs in refined but not in brute
    std::set_difference(
      refined_pairs.begin(),
      refined_pairs.end(),
      brute_pairs_set.begin(),
      brute_pairs_set.end(),
      std::back_inserter( extra_in_refined ) );

    if ( !missing_in_refined.empty() )
    {
      fmt::print( "  │ Missing {} pairs in refined results\n", missing_in_refined.size() );
      // Print first few missing pairs
      integer count = 0;
      for ( const auto & p : missing_in_refined )
      {
        if ( count++ < 5 ) { fmt::print( "  │   ({}, {})\n", p.first, p.second ); }
      }
      if ( missing_in_refined.size() > 5 ) { fmt::print( "  │   ... and {} more\n", missing_in_refined.size() - 5 ); }
    }

    if ( !extra_in_refined.empty() )
    {
      fmt::print( "  │ Extra {} pairs in refined results\n", extra_in_refined.size() );
      // Print first few extra pairs
      integer count = 0;
      for ( const auto & p : extra_in_refined )
      {
        if ( count++ < 5 ) { fmt::print( "  │   ({}, {})\n", p.first, p.second ); }
      }
      if ( extra_in_refined.size() > 5 ) { fmt::print( "  │   ... and {} more\n", extra_in_refined.size() - 5 ); }
    }
  }

  // Check that candidates are superset of exact results
  std::set<std::pair<integer, integer>> candidate_pairs;
  for ( const auto & [i, js] : candidate_results )
  {
    for ( integer j : js ) { candidate_pairs.insert( { i, j } ); }
  }

  bool is_superset =
    std::includes( candidate_pairs.begin(), candidate_pairs.end(), refined_pairs.begin(), refined_pairs.end() );

  if ( is_superset ) { fmt::print( fg( fmt::color::green ), "  │ ✓ Candidates are superset of exact results\n" ); }
  else
  {
    // Find which exact pairs are missing from candidates
    std::vector<std::pair<integer, integer>> missing_from_candidates;
    std::set_difference(
      refined_pairs.begin(),
      refined_pairs.end(),
      candidate_pairs.begin(),
      candidate_pairs.end(),
      std::back_inserter( missing_from_candidates ) );

    fmt::print(
      fg( fmt::color::red ),
      "  │ ✗ Candidates are NOT superset! Missing {} pairs\n",
      missing_from_candidates.size() );

    // Print first few missing pairs
    integer count = 0;
    for ( const auto & p : missing_from_candidates )
    {
      if ( count++ < 5 ) { fmt::print( "  │   ({}, {})\n", p.first, p.second ); }
    }
  }

  if ( exact_match && is_superset )
    fmt::print( fg( fmt::color::green ), "  ✓ All tree intersection tests passed!\n" );
  else
    fmt::print( fg( fmt::color::red ), "  ✗ Tree intersection tests failed!\n" );
}


void test_simple_tree_intersection()
{
  fmt::print( fg( fmt::color::cyan ), "\n┌──────────────────────────────────────────┐\n" );
  fmt::print( fg( fmt::color::cyan ), "│  Simple Tree Intersection (Known Answer) │\n" );
  fmt::print( fg( fmt::color::cyan ), "└──────────────────────────────────────────┘\n" );

  constexpr integer nbox1 = 4;
  constexpr integer nbox2 = 4;
  constexpr integer dim   = 2;

  // Tree 1: 4 non-overlapping boxes
  real_type bb_min1[nbox1 * dim] = {
    0.0, 0.0,  // Box 0
    2.0, 0.0,  // Box 1
    0.0, 2.0,  // Box 2
    2.0, 2.0   // Box 3
  };

  real_type bb_max1[nbox1 * dim] = {
    1.0, 1.0,  // Box 0
    3.0, 1.0,  // Box 1
    1.0, 3.0,  // Box 2
    3.0, 3.0   // Box 3
  };

  // Tree 2: 4 boxes that partially overlap with tree1
  real_type bb_min2[nbox2 * dim] = {
    0.5, 0.5,  // Box 0 - overlaps with box 0 of tree1
    2.5, 0.5,  // Box 1 - overlaps with box 1 of tree1
    0.5, 2.5,  // Box 2 - overlaps with box 2 of tree1
    2.5, 2.5   // Box 3 - overlaps with box 3 of tree1
  };

  real_type bb_max2[nbox2 * dim] = {
    1.5, 1.5,  // Box 0
    3.5, 1.5,  // Box 1
    1.5, 3.5,  // Box 2
    3.5, 3.5   // Box 3
  };

  AABBtree<real_type> tree1, tree2;
  tree1.build( bb_min1, dim, bb_max1, dim, nbox1, dim );
  tree2.build( bb_min2, dim, bb_max2, dim, nbox2, dim );

  fmt::print( "  Tree 1: 4 boxes in a 2x2 grid\n" );
  fmt::print( "  Tree 2: 4 boxes offset by (0.5, 0.5)\n" );

  // Expected intersections:
  // Box 0 of tree1 intersects with box 0 of tree2
  // Box 1 of tree1 intersects with box 1 of tree2
  // Box 2 of tree1 intersects with box 2 of tree2
  // Box 3 of tree1 intersects with box 3 of tree2

  // Test refined intersection
  std::map<integer, std::set<integer>> refined_results;
  tree1.intersect_and_refine( tree2, refined_results );

  fmt::print( "  ├─ Refined Intersection Results ──────────────\n" );

  bool all_correct = true;
  for ( integer i = 0; i < nbox1; ++i )
  {
    auto it = refined_results.find( i );
    if ( it != refined_results.end() )
    {
      if ( it->second.size() == 1 && *it->second.begin() == i )
      {
        fmt::print( fg( fmt::color::green ), "  │   Box {} -> Box {} ✓\n", i, i );
      }
      else
      {
        fmt::print( fg( fmt::color::red ), "  │   Box {} -> {{", i );
        for ( auto j : it->second ) fmt::print( "{} ", j );
        fmt::print( "}} ✗ (expected {{ {} }})\n", i );
        all_correct = false;
      }
    }
    else
    {
      fmt::print( fg( fmt::color::red ), "  │   Box {} -> no intersections ✗ (expected {{ {} }})\n", i, i );
      all_correct = false;
    }
  }

  if ( all_correct ) { fmt::print( fg( fmt::color::green ), "  ✓ All simple intersections correct!\n" ); }
  else
  {
    fmt::print( fg( fmt::color::red ), "  ✗ Simple intersection test failed!\n" );
  }

  // Also test with overlapping boxes
  fmt::print( "\n  ├─ Testing with overlapping boxes ────────────\n" );

  // Create two boxes that definitely overlap
  real_type overlap_min1[2] = { 0.0, 0.0 };
  real_type overlap_max1[2] = { 2.0, 2.0 };
  real_type overlap_min2[2] = { 1.0, 1.0 };
  real_type overlap_max2[2] = { 3.0, 3.0 };

  AABBtree<real_type> tree3, tree4;
  tree3.build( overlap_min1, dim, overlap_max1, dim, 1, dim );
  tree4.build( overlap_min2, dim, overlap_max2, dim, 1, dim );

  std::map<integer, std::set<integer>> overlap_results;
  tree3.intersect_and_refine( tree4, overlap_results );

  if ( overlap_results.size() == 1 && overlap_results[0].size() == 1 && *overlap_results[0].begin() == 0 )
  {
    fmt::print( fg( fmt::color::green ), "  │   Overlapping boxes correctly detected ✓\n" );
  }
  else
  {
    fmt::print( fg( fmt::color::red ), "  │   Failed to detect overlapping boxes ✗\n" );
    all_correct = false;
  }
}

void test_min_distance()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 6: Minimum Distance           │\n"
    "└────────────────────────────────────────┘\n" );

  constexpr integer nbox = 500;
  constexpr integer dim  = 2;

  real_type bb_min[nbox * dim];
  real_type bb_max[nbox * dim];

  for ( integer i = 0; i < nbox; ++i )
  {
    bb_min[i * dim]     = rand_uniform( 0, 10 );
    bb_min[i * dim + 1] = rand_uniform( 0, 10 );
    bb_max[i * dim]     = bb_min[i * dim] + rand_uniform( 0.1, 2 );
    bb_max[i * dim + 1] = bb_min[i * dim + 1] + rand_uniform( 0.1, 2 );
  }

  AABBtree<real_type> tree;
  tree.build( bb_min, dim, bb_max, dim, nbox, dim );

  // Test multiple query points
  std::vector<std::array<real_type, 2>> test_points = { { 0, 0 },   { 5, 5 },   { 10, 10 },
                                                        { -1, -1 }, { 11, 11 }, { 2.5, 7.5 } };

  fmt::print( "  Testing minimum distance candidates...\n" );

  integer passed = 0;
  integer total  = static_cast<integer>( test_points.size() );

  for ( auto const & pnt : test_points )
  {
    std::set<integer> candidates;
    tree.min_distance_candidates( pnt.data(), candidates );

    // Basic validation: candidates should not be empty for points inside domain
    bool valid = true;

    // Point inside [0,10] domain should have candidates
    if ( pnt[0] >= 0 && pnt[0] <= 10 && pnt[1] >= 0 && pnt[1] <= 10 )
    {
      if ( candidates.empty() ) valid = false;
    }

    if ( valid )
      passed++;
    else
      fmt::print( fg( fmt::color::yellow ), "    Unexpected empty candidates for point ({}, {})\n", pnt[0], pnt[1] );
  }

  fmt::print( "  ├─────────────────────────────────────────────\n" );
  fmt::print( "  │ Results: {}/{} tests passed\n", passed, total );

  if ( passed == total )
    fmt::print( fg( fmt::color::green ), "  ✓ Minimum distance tests passed!\n" );
  else
    fmt::print( fg( fmt::color::red ), "  ✗ Some minimum distance tests failed\n" );
}

void test_tree_info_and_queries()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 7: Tree Info & Queries        │\n"
    "└────────────────────────────────────────┘\n" );

  constexpr integer nbox = 100;
  constexpr integer dim  = 2;

  real_type bb_min[nbox * dim];
  real_type bb_max[nbox * dim];

  for ( integer i = 0; i < nbox; ++i )
  {
    bb_min[i * dim]     = rand_uniform( 0, 8 );
    bb_min[i * dim + 1] = rand_uniform( 0, 8 );
    bb_max[i * dim]     = bb_min[i * dim] + rand_uniform( 0.1, 1 );
    bb_max[i * dim + 1] = bb_min[i * dim + 1] + rand_uniform( 0.1, 1 );
  }

  AABBtree<real_type> tree;
  tree.set_max_num_objects_per_node( 16 );
  tree.build( bb_min, dim, bb_max, dim, nbox, dim );

  fmt::print( "  Tree Information:\n" );
  fmt::print( "{}\n", tree.info( "  " ) );

  // Test get_bbox_indexes_of_a_node
  fmt::print( "  ├─ Testing node queries:\n" );

  integer num_nodes    = tree.num_tree_nodes();
  integer tested_nodes = 0;

  for ( integer i = 0; i < num_nodes && i < 5; ++i )  // Test first 5 nodes
  {
    std::set<integer> node_bboxes;
    tree.get_bbox_indexes_of_a_node( i, node_bboxes );

    if ( !node_bboxes.empty() )
    {
      fmt::print( "  │ Node {} contains {} bboxes\n", i, node_bboxes.size() );
      tested_nodes++;
    }
  }

  if ( tested_nodes > 0 )
    fmt::print( fg( fmt::color::green ), "  ✓ Node query tests passed!\n" );
  else
    fmt::print( fg( fmt::color::yellow ), "  ⚠ No nodes with bboxes found for testing\n" );

  // Test get_root_bbox
  real_type root_min[dim], root_max[dim];
  tree.get_root_bbox( root_min, root_max );
  fmt::print(
    "  ├─ Root bbox: [{:.2f}, {:.2f}] - [{:.2f}, {:.2f}]\n",
    root_min[0],
    root_min[1],
    root_max[0],
    root_max[1] );
}

void test_performance_scaling()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌────────────────────────────────────────┐\n"
    "│     Test 8: Performance Scaling        │\n"
    "└────────────────────────────────────────┘\n" );

  constexpr integer    dim   = 2;
  std::vector<integer> sizes = { 100, 500, 1000, 5000, 10000 };

  fmt::print( "  ┌──────────┬────────────┬────────────┬────────────┐\n" );
  fmt::print( "  │ N Boxes  │ Build (ms) │ Point Q    │ Tree-Tree  │\n" );
  fmt::print( "  │          │            │ (ms)       │ (ms)       │\n" );
  fmt::print( "  ├──────────┼────────────┼────────────┼────────────┤\n" );

  for ( integer nbox : sizes )
  {
    // Generate data
    std::vector<real_type> bb_min( nbox * dim );
    std::vector<real_type> bb_max( nbox * dim );

    for ( integer i = 0; i < nbox; ++i )
    {
      bb_min[i * dim]     = rand_uniform( 0, 10 );
      bb_min[i * dim + 1] = rand_uniform( 0, 10 );
      bb_max[i * dim]     = bb_min[i * dim] + rand_uniform( 0.1, 2 );
      bb_max[i * dim + 1] = bb_min[i * dim + 1] + rand_uniform( 0.1, 2 );
    }

    // Build tree
    TicToc              timer;
    AABBtree<real_type> tree;

    timer.tic();
    tree.build( bb_min.data(), dim, bb_max.data(), dim, nbox, dim );
    timer.toc();
    real_type build_time = timer.elapsed_ms();

    // Point query
    real_type         pnt[2] = { 5, 5 };
    std::set<integer> results;

    timer.tic();
    tree.intersect_with_one_point( pnt, results );
    timer.toc();
    real_type point_query_time = timer.elapsed_ms();

    // Tree-tree query (self-intersection)
    std::map<integer, std::set<integer>> intersection_results;

    timer.tic();
    tree.intersect( tree, intersection_results );
    timer.toc();
    real_type tree_query_time = timer.elapsed_ms();

    fmt::print( "  │ {:8} │ {:10.2f} │ {:10.3f} │ {:10.2f} │\n", nbox, build_time, point_query_time, tree_query_time );
  }

  fmt::print( "  └──────────┴────────────┴────────────┴────────────┘\n" );
}

void test_tree_structure()
{
  fmt::print(
    fg( fmt::color::cyan ),
    "\n"
    "┌───────────────────────────────────────┐\n"
    "│     Test 9: Tree Structure            │\n"
    "└───────────────────────────────────────┘\n" );

  constexpr integer nbox = 10;
  constexpr integer dim  = 2;

  real_type bb_min[nbox * dim];
  real_type bb_max[nbox * dim];

  // Crea box non sovrapposti per test semplice
  for ( integer i = 0; i < nbox; ++i )
  {
    bb_min[i * dim]     = i;
    bb_min[i * dim + 1] = 0;
    bb_max[i * dim]     = i + 1;
    bb_max[i * dim + 1] = 1;
  }

  AABBtree<real_type> tree;
  tree.build( bb_min, dim, bb_max, dim, nbox, dim );

  fmt::print( "  Tree info for non-overlapping boxes:\n" );
  fmt::print( "{}\n", tree.info( "  " ) );

  // Test che ogni punto trovi esattamente 1 box
  fmt::print( "  Testing point queries on non-overlapping boxes:\n" );
  integer passed = 0;

  for ( integer i = 0; i < nbox; ++i )
  {
    real_type         pnt[2] = { i + 0.5, 0.5 };
    std::set<integer> results;
    tree.intersect_with_one_point_and_refine( pnt, results );

    if ( results.size() == 1 && *results.begin() == i )
    {
      passed++;
      fmt::print( "    Point ({}, {}) correctly found box {}\n", pnt[0], pnt[1], i );
    }
    else
    {
      fmt::print(
        fg( fmt::color::yellow ),
        "    Point ({}, {}) found {} boxes, expected box {}\n",
        pnt[0],
        pnt[1],
        results.size(),
        i );
    }
  }

  if ( passed == nbox )
    fmt::print( fg( fmt::color::green ), "  ✓ All point queries correct!\n" );
  else
    fmt::print( fg( fmt::color::red ), "  ✗ {} of {} queries failed\n", nbox - passed, nbox );
}

// ===========================================================================
// Main Test Runner
// ===========================================================================

int main()
{
  fmt::print(
    fg( fmt::color::magenta ),
    "\n"
    "╔══════════════════════════════════════════════╗\n"
    "║     AABBTree Comprehensive Test Suite        ║\n"
    "╚══════════════════════════════════════════════╝\n" );

  TicToc total_timer;
  total_timer.tic();

  try
  {
    test_basic_functionality();
    test_parameter_settings();
    test_point_intersection_comprehensive();
    test_bbox_intersection_comprehensive();
    test_simple_tree_intersection();  // Aggiunto
    test_tree_intersection_comprehensive();
    test_min_distance();
    test_tree_info_and_queries();
    test_performance_scaling();
    test_tree_structure();

    total_timer.toc();

    fmt::print( fg( fmt::color::green ), "\n═══════════════════════════════════════════════\n" );
    fmt::print( fg( fmt::color::green ), "All tests completed successfully!\n" );
    fmt::print( "Total execution time: {:.2f} ms\n", total_timer.elapsed_ms() );
    fmt::print( fg( fmt::color::green ), "═══════════════════════════════════════════════\n" );
  }
  catch ( std::exception const & e )
  {
    total_timer.toc();
    fmt::print( fg( fmt::color::red ), "\n═══════════════════════════════════════════════\n" );
    fmt::print( "Test failed with exception: {}\n", e.what() );
    fmt::print( "Total execution time: {:.2f} ms\n", total_timer.elapsed_ms() );
    fmt::print( fg( fmt::color::red ), "═══════════════════════════════════════════════\n" );
    return 1;
  }

  return 0;
}
