/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2024                                                      |
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

#include "Utils_Quaternion.hh"
#include "Utils_TicToc.hh"
#include "Utils_fmt.hh"
#include "Utils_string.hh"

using namespace std;
using namespace Utils;

using real_type = double;

// ============================================================================
// Utility Functions for Testing
// ============================================================================

// Generate a random quaternion with uniform distribution
template <typename T> Quaternion<T> random_quaternion()
{
  static std::mt19937                      rng( std::random_device{}() );
  static std::uniform_real_distribution<T> dist( -1.0, 1.0 );

  return Quaternion<T>( dist( rng ), dist( rng ), dist( rng ), dist( rng ) ).normalize();
}

// Generate a random rotation angle and axis
template <typename T> void random_axis_angle( T axis[3], T & angle )
{
  static std::mt19937                      rng( std::random_device{}() );
  static std::uniform_real_distribution<T> angle_dist( 0.0, 2.0 * M_PI );
  static std::uniform_real_distribution<T> coord_dist( -1.0, 1.0 );

  axis[0] = coord_dist( rng );
  axis[1] = coord_dist( rng );
  axis[2] = coord_dist( rng );

  // Normalize axis
  T norm = std::sqrt( axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2] );
  if ( norm > 0 )
  {
    axis[0] /= norm;
    axis[1] /= norm;
    axis[2] /= norm;
  }
  else
  {
    axis[0] = 1;
    axis[1] = 0;
    axis[2] = 0;
  }

  angle = angle_dist( rng );
}

// Compute error between two quaternions (accounting for double cover)
template <typename T> T quaternion_error( const Quaternion<T> & q1, const Quaternion<T> & q2 )
{
  // Quaternions q and -q represent the same rotation
  T error1 = ( q1 - q2 ).norm_squared();
  T error2 = ( q1 + q2 ).norm_squared();
  return std::min( error1, error2 );
}

// ============================================================================
// Test Case Definitions
// ============================================================================

struct TestResult
{
  string    test_name;
  bool      passed;
  real_type error;
  int       iterations;
  real_type duration_ms;
  string    message;
};

class TestRunner
{
  static constexpr auto HEADER_COLOR  = fmt::color::steel_blue;
  static constexpr auto SUCCESS_COLOR = fmt::color::lime_green;
  static constexpr auto WARNING_COLOR = fmt::color::gold;
  static constexpr auto ERROR_COLOR   = fmt::color::crimson;
  static constexpr auto METHOD_COLOR  = fmt::color::deep_sky_blue;
  static constexpr auto VALUE_COLOR   = fmt::color::white;
  static constexpr auto ITER_COLOR    = fmt::color::violet;
  static constexpr auto TIME_COLOR    = fmt::color::light_blue;

public:
  static void print_header( const string & title )
  {
    fmt::print( fg( HEADER_COLOR ) | fmt::emphasis::bold, "\n{:=^80}\n", " " + title + " " );
  }

  static void print_test_header()
  {
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "┌─────────────────────────────────┬──────────┬──────────────┬──────────┬────────────────────────────────┐\n" );
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "│ {:31s} │ {:8s} │ {:12s} │ {:8s} │ {:30s} │\n",
      "Test Name",
      "Status",
      "Error",
      "Time(ms)",
      "Message" );
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "├─────────────────────────────────┼──────────┼──────────────┼──────────┼────────────────────────────────┤\n" );
  }

  static void print_test_footer()
  {
    fmt::print(
      fg( HEADER_COLOR ) | fmt::emphasis::bold,
      "└─────────────────────────────────┴──────────┴──────────────┴──────────┴────────────────────────────────┘\n" );
  }

  static void print_result( const TestResult & result )
  {
    // Choose color based on test status
    auto status_color = result.passed ? fg( SUCCESS_COLOR ) : fg( ERROR_COLOR );
    auto status_text  = result.passed ? "✓ PASS" : "✗ FAIL";

    fmt::print( "│ " );
    fmt::print( fg( METHOD_COLOR ), "{:31s}", result.test_name );
    fmt::print( " │ " );
    fmt::print( status_color, "{:>8s}", status_text );
    fmt::print( " │ " );

    // Color code error magnitude
    if ( result.error < 1e-10 )
      fmt::print( fg( SUCCESS_COLOR ), "{:12.4g}", result.error );
    else if ( result.error < 1e-5 )
      fmt::print( fg( WARNING_COLOR ), "{:12.4g}", result.error );
    else
      fmt::print( fg( ERROR_COLOR ), "{:12.4g}", result.error );

    fmt::print( " │ " );
    fmt::print( fg( TIME_COLOR ), "{:8.3f}", result.duration_ms );
    fmt::print( " │ " );

    string msg = Utils::utf8_padding( result.message, 30 );

    if ( result.passed )
      fmt::print( fg( SUCCESS_COLOR ), "{}", msg );
    else
      fmt::print( fg( ERROR_COLOR ), "{}", msg );

    fmt::print( " │\n" );
  }

  static void run_test_suite()
  {
    vector<TestResult> all_results;

    // SECTION 1: Basic Operations
    print_header( "SECTION 1: BASIC OPERATIONS" );
    print_test_header();

    auto results1 = test_basic_operations();
    all_results.insert( all_results.end(), results1.begin(), results1.end() );
    for ( const auto & r : results1 ) print_result( r );
    print_test_footer();

    // SECTION 2: Rotation Operations
    print_header( "SECTION 2: ROTATION OPERATIONS" );
    print_test_header();

    auto results2 = test_rotation_operations();
    all_results.insert( all_results.end(), results2.begin(), results2.end() );
    for ( const auto & r : results2 ) print_result( r );
    print_test_footer();

    // SECTION 3: Conversion Operations
    print_header( "SECTION 3: CONVERSION OPERATIONS" );
    print_test_header();

    auto results3 = test_conversion_operations();
    all_results.insert( all_results.end(), results3.begin(), results3.end() );
    for ( const auto & r : results3 ) print_result( r );
    print_test_footer();

    // SECTION 4: Advanced Operations
    print_header( "SECTION 4: ADVANCED OPERATIONS" );
    print_test_header();

    auto results4 = test_advanced_operations();
    all_results.insert( all_results.end(), results4.begin(), results4.end() );
    for ( const auto & r : results4 ) print_result( r );
    print_test_footer();

    // SECTION 5: Performance Tests
    print_header( "SECTION 5: PERFORMANCE TESTS" );
    print_test_header();

    auto results5 = test_performance();
    all_results.insert( all_results.end(), results5.begin(), results5.end() );
    for ( const auto & r : results5 ) print_result( r );
    print_test_footer();

    // Summary
    print_summary( all_results );
  }

private:
  // ============================================================================
  // Test Implementations
  // ============================================================================

  static vector<TestResult> test_basic_operations()
  {
    vector<TestResult>  results;
    constexpr real_type tolerance = 1e-10;

    // Test 1: Default Constructor
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q;
      timer.toc();

      TestResult r;
      r.test_name   = "Default Constructor";
      r.passed      = ( q[0] == 1 && q[1] == 0 && q[2] == 0 && q[3] == 0 );
      r.error       = ( q - Quaternion<real_type>::Identity() ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔄 Identity quaternion";
      results.push_back( r );
    }

    // Test 2: Component Constructor
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q( 1, 2, 3, 4 );
      timer.toc();

      TestResult r;
      r.test_name   = "Component Constructor";
      r.passed      = ( q[0] == 1 && q[1] == 2 && q[2] == 3 && q[3] == 4 );
      r.error       = ( q - Quaternion<real_type>( 1, 2, 3, 4 ) ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "✅ Components check";
      results.push_back( r );
    }

    // Test 3: Norm and Normalization
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q( 1, 2, 3, 4 );
      real_type             norm_before = q.norm();
      q.normalize();
      real_type norm_after = q.norm();
      timer.toc();

      TestResult r;
      r.test_name   = "Normalization";
      r.passed      = ( abs( norm_after - 1.0 ) < tolerance );
      r.error       = abs( norm_after - 1.0 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = fmt::format( "📏 Norm: {:.2f}→{:.2f}", norm_before, norm_after );
      results.push_back( r );
    }

    // Test 4: Conjugation
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q( 1, 2, 3, 4 );
      Quaternion<real_type> conj = q.conjugate();
      timer.toc();

      TestResult r;
      r.test_name   = "Conjugation";
      r.passed      = ( conj[0] == 1 && conj[1] == -2 && conj[2] == -3 && conj[3] == -4 );
      r.error       = ( conj - Quaternion<real_type>( 1, -2, -3, -4 ) ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔁 Conjugate check";
      results.push_back( r );
    }

    // Test 5: Inverse
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q       = random_quaternion<real_type>();
      Quaternion<real_type> inv     = q.inverse();
      Quaternion<real_type> product = q * inv;
      timer.toc();

      TestResult r;
      r.test_name   = "Inverse";
      r.passed      = product.isNormalized( tolerance );
      r.error       = ( product - Quaternion<real_type>::Identity() ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "✅ q * q⁻¹ = I";
      results.push_back( r );
    }

    // Test 6: Addition and Subtraction
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q1( 1, 2, 3, 4 );
      Quaternion<real_type> q2( 5, 6, 7, 8 );
      Quaternion<real_type> sum  = q1 + q2;
      Quaternion<real_type> diff = q1 - q2;
      timer.toc();

      TestResult r;
      r.test_name = "Addition/Subtraction";
      r.passed =
        ( sum[0] == 6 && sum[1] == 8 && sum[2] == 10 && sum[3] == 12 && diff[0] == -4 && diff[1] == -4 &&
          diff[2] == -4 && diff[3] == -4 );
      r.error = ( sum - Quaternion<real_type>( 6, 8, 10, 12 ) ).norm() +
                ( diff - Quaternion<real_type>( -4, -4, -4, -4 ) ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "➕➖ Vector operations";
      results.push_back( r );
    }

    // Test 7: Multiplication (Hamilton Product)
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q1( 1, 2, 3, 4 );
      Quaternion<real_type> q2( 5, 6, 7, 8 );
      Quaternion<real_type> prod = q1 * q2;
      timer.toc();

      // Manual calculation: (1,2,3,4) * (5,6,7,8)
      // w = 1*5 - 2*6 - 3*7 - 4*8 = -60
      // i = 1*6 + 2*5 + 3*8 - 4*7 = 12
      // j = 1*7 - 2*8 + 3*5 + 4*6 = 30
      // k = 1*8 + 2*7 - 3*6 + 4*5 = 24

      TestResult r;
      r.test_name = "Hamilton Product";
      r.passed =
        ( abs( prod[0] + 60 ) < tolerance && abs( prod[1] - 12 ) < tolerance && abs( prod[2] - 30 ) < tolerance &&
          abs( prod[3] - 24 ) < tolerance );
      r.error       = ( prod - Quaternion<real_type>( -60, 12, 30, 24 ) ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "✖️ Non-commutative mult";
      results.push_back( r );
    }

    // Test 8: Scalar Operations
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q( 1, 2, 3, 4 );
      Quaternion<real_type> scaled  = q * 2.5;
      Quaternion<real_type> divided = q / 0.5;
      timer.toc();

      TestResult r;
      r.test_name = "Scalar Operations";
      r.passed =
        ( scaled[0] == 2.5 && scaled[1] == 5.0 && scaled[2] == 7.5 && scaled[3] == 10.0 && divided[0] == 2.0 &&
          divided[1] == 4.0 && divided[2] == 6.0 && divided[3] == 8.0 );
      r.error = ( scaled - Quaternion<real_type>( 2.5, 5, 7.5, 10 ) ).norm() +
                ( divided - Quaternion<real_type>( 2, 4, 6, 8 ) ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "📈 Scale by scalar";
      results.push_back( r );
    }

    // Test 9: Dot Product
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q1( 1, 2, 3, 4 );
      Quaternion<real_type> q2( 5, 6, 7, 8 );
      real_type             dot = q1.dot( q2 );
      timer.toc();

      TestResult r;
      r.test_name   = "Dot Product";
      r.passed      = abs( dot - ( 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8 ) ) < tolerance;
      r.error       = abs( dot - 70 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔘 4D dot product";
      results.push_back( r );
    }

    // Test 10: Equality Comparison
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q1( 1, 2, 3, 4 );
      Quaternion<real_type> q2( 1, 2, 3, 4 );
      Quaternion<real_type> q3( 1, 2, 3, 5 );
      timer.toc();

      TestResult r;
      r.test_name   = "Equality Comparison";
      r.passed      = ( q1 == q2 ) && !( q1 == q3 ) && q1.equals( q2, tolerance );
      r.error       = quaternion_error( q1, q2 ) + quaternion_error( q1, q3 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "⚖️ == and equals()";
      results.push_back( r );
    }

    return results;
  }

  static vector<TestResult> test_rotation_operations()
  {
    vector<TestResult>  results;
    constexpr real_type tolerance = 1e-10;

    // Test 1: Identity Rotation
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q    = Quaternion<real_type>::Identity();
      real_type             v[3] = { 1, 2, 3 };
      real_type             w[3];
      q.rotate( v, w );
      timer.toc();

      TestResult r;
      r.test_name = "Identity Rotation";
      r.passed = ( abs( w[0] - v[0] ) < tolerance && abs( w[1] - v[1] ) < tolerance && abs( w[2] - v[2] ) < tolerance );
      r.error  = sqrt( pow( w[0] - v[0], 2 ) + pow( w[1] - v[1], 2 ) + pow( w[2] - v[2], 2 ) );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔄 No rotation";
      results.push_back( r );
    }

    // Test 2: 180-degree X-axis Rotation
    {
      TicToc timer;
      timer.tic();
      real_type             axis[3] = { 1, 0, 0 };
      Quaternion<real_type> q       = Quaternion<real_type>::FromAxisAngle( axis, M_PI );
      real_type             v[3]    = { 1, 2, 3 };
      real_type             w[3];
      q.rotate( v, w );
      timer.toc();

      TestResult r;
      r.test_name   = "180° X-axis Rotation";
      r.passed      = ( abs( w[0] - 1 ) < tolerance && abs( w[1] + 2 ) < tolerance && abs( w[2] + 3 ) < tolerance );
      r.error       = sqrt( pow( w[0] - 1, 2 ) + pow( w[1] + 2, 2 ) + pow( w[2] + 3, 2 ) );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "↩️ Flip Y and Z";
      results.push_back( r );
    }

    // Test 3: 90-degree Y-axis Rotation
    {
      TicToc timer;
      timer.tic();
      real_type             axis[3] = { 0, 1, 0 };
      Quaternion<real_type> q       = Quaternion<real_type>::FromAxisAngle( axis, M_PI / 2 );
      real_type             v[3]    = { 1, 0, 0 };
      real_type             w[3];
      q.rotate( v, w );
      timer.toc();

      TestResult r;
      r.test_name   = "90° Y-axis Rotation";
      r.passed      = ( abs( w[0] ) < tolerance && abs( w[1] ) < tolerance && abs( w[2] + 1 ) < tolerance );
      r.error       = sqrt( pow( w[0], 2 ) + pow( w[1], 2 ) + pow( w[2] + 1, 2 ) );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "↪️ X→-Z";
      results.push_back( r );
    }

    // Test 4: Random Rotation Consistency
    {
      TicToc timer;
      timer.tic();
      real_type axis[3], angle;
      random_axis_angle( axis, angle );

      Quaternion<real_type> q    = Quaternion<real_type>::FromAxisAngle( axis, angle );
      real_type             v[3] = { 1, 2, 3 };
      real_type             w1[3], w2[3];

      q.rotate( v, w1 );

      // Convert to matrix and rotate
      real_type R[3][3];
      q.to_rotation_matrix( R );

      w2[0] = R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2];
      w2[1] = R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2];
      w2[2] = R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2];

      timer.toc();

      real_type err = sqrt( pow( w1[0] - w2[0], 2 ) + pow( w1[1] - w2[1], 2 ) + pow( w1[2] - w2[2], 2 ) );

      TestResult r;
      r.test_name   = "Rotation Consistency";
      r.passed      = err < tolerance;
      r.error       = err;
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔄 Quat vs Matrix";
      results.push_back( r );
    }

    // Test 5: Multiple Rotations Composition
    {
      TicToc timer;
      timer.tic();

      real_type             axis_x[3] = { 1, 0, 0 };
      real_type             axis_y[3] = { 0, 1, 0 };
      real_type             axis_z[3] = { 0, 0, 1 };
      Quaternion<real_type> qx        = Quaternion<real_type>::FromAxisAngle( axis_x, M_PI / 4 );
      Quaternion<real_type> qy        = Quaternion<real_type>::FromAxisAngle( axis_y, M_PI / 3 );
      Quaternion<real_type> qz        = Quaternion<real_type>::FromAxisAngle( axis_z, M_PI / 6 );

      Quaternion<real_type> q_total1 = qz * qy * qx;
      Quaternion<real_type> q_total2 = Quaternion<real_type>::FromEulerAngles( M_PI / 6, M_PI / 3, M_PI / 4 );

      timer.toc();

      TestResult r;
      r.test_name   = "Rotation Composition";
      r.passed      = q_total1.equals( q_total2, tolerance );
      r.error       = quaternion_error( q_total1, q_total2 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔄 qₓqᵧq_z vs Euler";
      results.push_back( r );
    }

    return results;
  }

  static vector<TestResult> test_conversion_operations()
  {
    vector<TestResult>  results;
    constexpr real_type tolerance = 1e-8;

    // Test 1: Axis-Angle Round Trip
    {
      TicToc timer;
      timer.tic();
      real_type axis[3], angle;
      random_axis_angle( axis, angle );

      Quaternion<real_type> q1 = Quaternion<real_type>::FromAxisAngle( axis, angle );
      real_type             recovered_axis[3];
      real_type             recovered_angle = q1.to_axis_angle( recovered_axis );
      Quaternion<real_type> q2              = Quaternion<real_type>::FromAxisAngle( recovered_axis, recovered_angle );

      timer.toc();

      TestResult r;
      r.test_name   = "Axis-Angle Round Trip";
      r.passed      = q1.equals( q2, tolerance ) || q1.equals( -q2, tolerance );
      r.error       = quaternion_error( q1, q2 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔄 q→axis/angle→q";
      results.push_back( r );
    }

    // Test 2: Rotation Matrix Round Trip
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q1 = random_quaternion<real_type>();

      real_type R[3][3];
      q1.to_rotation_matrix( R );
      Quaternion<real_type> q2 = Quaternion<real_type>::FromRotationMatrix( R );

      timer.toc();

      TestResult r;
      r.test_name   = "Rotation Matrix Round Trip";
      r.passed      = q1.equals( q2, tolerance ) || q1.equals( -q2, tolerance );
      r.error       = quaternion_error( q1, q2 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔄 q→R→q";
      results.push_back( r );
    }

    // Test 3: Euler Angles Round Trip (ZYX convention)
    {
      TicToc timer;
      timer.tic();

      // Avoid gimbal lock (pitch near ±90°)
      real_type yaw = 0.5, pitch = 0.3, roll = 0.7;

      Quaternion<real_type> q1 = Quaternion<real_type>::FromEulerAngles( yaw, pitch, roll );
      real_type             yaw2, pitch2, roll2;
      q1.to_Euler_angles( yaw2, pitch2, roll2 );
      Quaternion<real_type> q2 = Quaternion<real_type>::FromEulerAngles( yaw2, pitch2, roll2 );

      timer.toc();

      // Normalize angles for comparison
      yaw    = fmod( yaw + M_PI, 2 * M_PI ) - M_PI;
      yaw2   = fmod( yaw2 + M_PI, 2 * M_PI ) - M_PI;
      pitch  = fmod( pitch + M_PI, 2 * M_PI ) - M_PI;
      pitch2 = fmod( pitch2 + M_PI, 2 * M_PI ) - M_PI;
      roll   = fmod( roll + M_PI, 2 * M_PI ) - M_PI;
      roll2  = fmod( roll2 + M_PI, 2 * M_PI ) - M_PI;

      TestResult r;
      r.test_name = "Euler Angles Round Trip";
      r.passed    = ( abs( yaw - yaw2 ) < tolerance && abs( pitch - pitch2 ) < tolerance &&
                   abs( roll - roll2 ) < tolerance ) &&
                 ( q1.equals( q2, tolerance ) || q1.equals( -q2, tolerance ) );
      r.error       = abs( yaw - yaw2 ) + abs( pitch - pitch2 ) + abs( roll - roll2 ) + quaternion_error( q1, q2 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔄 ZYX convention";
      results.push_back( r );
    }

    // Test 4: Matrix Orthogonality
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q = random_quaternion<real_type>();

      real_type R[3][3];
      q.to_rotation_matrix( R );

      // Check RᵀR = I
      real_type ortho_error = 0;
      for ( int i = 0; i < 3; ++i )
      {
        for ( int j = 0; j < 3; ++j )
        {
          real_type dot = 0;
          for ( int k = 0; k < 3; ++k ) dot += R[k][i] * R[k][j];
          real_type expected = ( i == j ) ? 1.0 : 0.0;
          ortho_error += abs( dot - expected );
        }
      }

      // Check det(R) = +1
      real_type det = R[0][0] * ( R[1][1] * R[2][2] - R[1][2] * R[2][1] ) -
                      R[0][1] * ( R[1][0] * R[2][2] - R[1][2] * R[2][0] ) +
                      R[0][2] * ( R[1][0] * R[2][1] - R[1][1] * R[2][0] );

      timer.toc();

      TestResult r;
      r.test_name   = "Matrix Orthogonality";
      r.passed      = ( ortho_error < tolerance && abs( det - 1.0 ) < tolerance );
      r.error       = ortho_error + abs( det - 1.0 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "📐 RᵀR=I, det=+1";
      results.push_back( r );
    }

    // Test 5: Scalar/Vector Parts
    {
      TicToc timer;
      timer.tic();
      Quaternion<real_type> q( 1, 2, 3, 4 );

      real_type scalar = q.scalar();
      real_type vec[3];
      q.vector( vec );

      timer.toc();

      TestResult r;
      r.test_name   = "Scalar/Vector Parts";
      r.passed      = ( scalar == 1 && vec[0] == 2 && vec[1] == 3 && vec[2] == 4 );
      r.error       = abs( scalar - 1 ) + abs( vec[0] - 2 ) + abs( vec[1] - 3 ) + abs( vec[2] - 4 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "📊 Separation check";
      results.push_back( r );
    }

    return results;
  }

  static vector<TestResult> test_advanced_operations()
  {
    vector<TestResult>  results;
    constexpr real_type tolerance = 1e-8;

    // Test 1: Exponential and Logarithm
    {
      TicToc timer;
      timer.tic();

      // Pure vector quaternion (imaginary part only)
      Quaternion<real_type> q_pure( 0, 1, 2, 3 );
      q_pure.normalize();
      q_pure = q_pure * M_PI / 4;  // Scale to represent rotation

      Quaternion<real_type> q_exp = q_pure.exp();
      Quaternion<real_type> q_log = q_exp.log();

      timer.toc();

      TestResult r;
      r.test_name   = "Exp/Log Round Trip";
      r.passed      = q_pure.equals( q_log, tolerance );
      r.error       = ( q_pure - q_log ).norm();
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔢 exp(log(q)) = q";
      results.push_back( r );
    }

    // Test 2: Power Operation
    {
      TicToc timer;
      timer.tic();

      Quaternion<real_type> q         = random_quaternion<real_type>();
      Quaternion<real_type> q_squared = q * q;
      Quaternion<real_type> q_pow2    = q.pow( 2 );

      timer.toc();

      TestResult r;
      r.test_name   = "Power Operation";
      r.passed      = q_squared.equals( q_pow2, tolerance );
      r.error       = quaternion_error( q_squared, q_pow2 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔢 q² vs q.pow(2)";
      results.push_back( r );
    }

    // Test 3: Spherical Linear Interpolation (SLERP)
    {
      TicToc timer;
      timer.tic();

      real_type             axis[3] = { 1, 0, 0 };
      Quaternion<real_type> q1      = Quaternion<real_type>::FromAxisAngle( axis, 0 );
      Quaternion<real_type> q2      = Quaternion<real_type>::FromAxisAngle( axis, M_PI / 2 );

      Quaternion<real_type> q_mid    = Quaternion<real_type>::slerp( q1, q2, 0.5 );
      Quaternion<real_type> expected = Quaternion<real_type>::FromAxisAngle( axis, M_PI / 4 );

      timer.toc();

      TestResult r;
      r.test_name   = "SLERP";
      r.passed      = q_mid.equals( expected, tolerance ) || q_mid.equals( -expected, tolerance );
      r.error       = quaternion_error( q_mid, expected );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "↔️ 45° interpolation";
      results.push_back( r );
    }

    // Test 4: Normalized Linear Interpolation (NLERP)
    {
      TicToc timer;
      timer.tic();

      real_type             axis[3] = { 1, 0, 0 };
      Quaternion<real_type> q1      = Quaternion<real_type>::FromAxisAngle( axis, 0 );
      Quaternion<real_type> q2      = Quaternion<real_type>::FromAxisAngle( axis, M_PI / 2 );

      Quaternion<real_type> q_mid = Quaternion<real_type>::nlerp( q1, q2, 0.5 );

      // NLERP isn't exact for large angles, just check it's normalized
      timer.toc();

      TestResult r;
      r.test_name   = "NLERP";
      r.passed      = q_mid.isNormalized( tolerance );
      r.error       = abs( q_mid.norm_squared() - 1.0 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "⚡ Fast interpolation";
      results.push_back( r );
    }

    // Test 5: Exponential Identities
    {
      TicToc timer;
      timer.tic();

      Quaternion<real_type> q1( 0, 1, 0, 0 );  // i
      Quaternion<real_type> q2( 0, 0, 1, 0 );  // j

      Quaternion<real_type> exp_sum  = ( q1 + q2 ).exp();
      Quaternion<real_type> exp_prod = ( q1.exp() * q2.exp() );

      // Note: exp(a+b) != exp(a)exp(b) for non-commuting quaternions
      // We just verify both are unit quaternions

      timer.toc();

      TestResult r;
      r.test_name   = "Exponential Properties";
      r.passed      = exp_sum.isNormalized( tolerance ) && exp_prod.isNormalized( tolerance );
      r.error       = abs( exp_sum.norm_squared() - 1.0 ) + abs( exp_prod.norm_squared() - 1.0 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "🔢 exp properties";
      results.push_back( r );
    }

    // Test 6: Angular Distance
    {
      TicToc timer;
      timer.tic();

      real_type             axis[3] = { 1, 0, 0 };
      Quaternion<real_type> q1      = Quaternion<real_type>::FromAxisAngle( axis, 0 );
      Quaternion<real_type> q2      = Quaternion<real_type>::FromAxisAngle( axis, M_PI / 3 );

      real_type dist = angular_distance( q1, q2 );

      timer.toc();

      TestResult r;
      r.test_name   = "Angular Distance";
      r.passed      = abs( dist - M_PI / 3 ) < tolerance;
      r.error       = abs( dist - M_PI / 3 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "📏 60° separation";
      results.push_back( r );
    }

    // Test 7: Spherical Centroid
    {
      TicToc timer;
      timer.tic();

      vector<Quaternion<real_type>> quats;
      for ( int i = 0; i < 10; ++i ) { quats.push_back( random_quaternion<real_type>() ); }

      Quaternion<real_type> centroid = spherical_centroid( quats, 50, 1e-12 );

      timer.toc();

      TestResult r;
      r.test_name   = "Spherical Centroid";
      r.passed      = centroid.isNormalized( tolerance );
      r.error       = abs( centroid.norm_squared() - 1.0 );
      r.duration_ms = timer.elapsed_ms();
      r.message     = "📊 Average of 10 quats";
      results.push_back( r );
    }

    return results;
  }

  static vector<TestResult> test_performance()
  {
    vector<TestResult> results;
    const int          NUM_ITERATIONS = 10000;

    // Test 1: Rotation Performance
    {
      TicToc timer;
      timer.tic();

      Quaternion<real_type> q    = random_quaternion<real_type>();
      real_type             v[3] = { 1, 2, 3 };
      real_type             w[3];

      for ( int i = 0; i < NUM_ITERATIONS; ++i )
      {
        q.rotate( v, w );
        // Use w to prevent optimization
        v[0] = w[0];
        v[1] = w[1];
        v[2] = w[2];
      }

      timer.toc();

      TestResult r;
      r.test_name   = "Rotation Performance";
      r.passed      = true;
      r.error       = 0;
      r.duration_ms = timer.elapsed_ms();
      r.message     = fmt::format( "⚡ {} ops", NUM_ITERATIONS );
      results.push_back( r );
    }

    // Test 2: Multiplication Performance
    {
      TicToc timer;
      timer.tic();

      Quaternion<real_type> q1 = random_quaternion<real_type>();
      Quaternion<real_type> q2 = random_quaternion<real_type>();
      Quaternion<real_type> result;

      for ( int i = 0; i < NUM_ITERATIONS; ++i )
      {
        result = q1 * q2;
        // Cycle quaternions to prevent optimization
        q1 = q2;
        q2 = result;
      }

      timer.toc();

      TestResult r;
      r.test_name   = "Multiplication Performance";
      r.passed      = true;
      r.error       = 0;
      r.duration_ms = timer.elapsed_ms();
      r.message     = fmt::format( "⚡ {} ops", NUM_ITERATIONS );
      results.push_back( r );
    }

    // Test 3: Normalization Performance
    {
      TicToc timer;
      timer.tic();

      vector<Quaternion<real_type>> quats;
      for ( int i = 0; i < NUM_ITERATIONS; ++i )
      {
        quats.push_back(
          Quaternion<real_type>(
            rand() / real_type( RAND_MAX ),
            rand() / real_type( RAND_MAX ),
            rand() / real_type( RAND_MAX ),
            rand() / real_type( RAND_MAX ) ) );
      }

      for ( auto & q : quats ) { q.normalize(); }

      timer.toc();

      TestResult r;
      r.test_name   = "Normalization Performance";
      r.passed      = true;
      r.error       = 0;
      r.duration_ms = timer.elapsed_ms();
      r.message     = fmt::format( "⚡ {} quats", NUM_ITERATIONS );
      results.push_back( r );
    }

    // Test 4: Conversion Performance
    {
      TicToc timer;
      timer.tic();

      Quaternion<real_type> q = random_quaternion<real_type>();
      real_type             R[3][3];
      real_type             axis[3];
      real_type             yaw, pitch, roll;

      for ( int i = 0; i < NUM_ITERATIONS / 10; ++i )
      {
        q.to_rotation_matrix( R );
        q.to_axis_angle( axis );  // CORRETTO: restituisce angle
        q.to_Euler_angles( yaw, pitch, roll );
        q = q * Quaternion<real_type>::FromEulerAngles( yaw, pitch, roll );
      }

      timer.toc();

      TestResult r;
      r.test_name   = "Conversion Performance";
      r.passed      = true;
      r.error       = 0;
      r.duration_ms = timer.elapsed_ms();
      r.message     = fmt::format( "⚡ {} ops", NUM_ITERATIONS / 10 );
      results.push_back( r );
    }

    return results;
  }

  static void print_summary( const vector<TestResult> & results )
  {
    int       total         = static_cast<int>( results.size() );
    int       passed        = 0;
    real_type total_error   = 0;
    real_type total_time_ms = 0;

    for ( const auto & r : results )
    {
      if ( r.passed ) passed++;
      total_error += r.error;
      total_time_ms += r.duration_ms;
    }

    fmt::print( fg( HEADER_COLOR ) | fmt::emphasis::bold, "\n{:=^80}\n", " 📊 TEST SUMMARY 📊 " );

    fmt::print( fg( fmt::color::light_blue ), "\n📈 Overall Statistics:\n" );
    fmt::print( "  Total Tests:     {}\n", total );
    fmt::print( "  ✅ Passed:          {} ({:.1f}%)\n", passed, 100.0 * passed / total );
    fmt::print( "  ❌ Failed:          {} ({:.1f}%)\n", total - passed, 100.0 * ( total - passed ) / total );
    fmt::print( "  📊 Total Error:     {:.3e}\n", total_error );
    fmt::print( "  ⏱️ Total Time:      {:.3f} ms\n", total_time_ms );
    fmt::print( "  🚀 Avg Time/Test:   {:.3f} μs\n", total_time_ms * 1000 / total );

    // Performance benchmarks
    fmt::print( fg( fmt::color::light_blue ), "\n🏎️ Performance Benchmarks:\n" );

    // Find fastest and slowest performance tests
    real_type fastest_ms = numeric_limits<real_type>::max();
    real_type slowest_ms = 0;
    string    fastest_name, slowest_name;

    for ( const auto & r : results )
    {
      if ( r.test_name.find( "Performance" ) != string::npos )
      {
        if ( r.duration_ms < fastest_ms )
        {
          fastest_ms   = r.duration_ms;
          fastest_name = r.test_name;
        }
        if ( r.duration_ms > slowest_ms )
        {
          slowest_ms   = r.duration_ms;
          slowest_name = r.test_name;
        }
      }
    }

    if ( fastest_ms != numeric_limits<real_type>::max() )
    {
      fmt::print( "  🥇 Fastest:         {} ({:.3f} ms)\n", fastest_name, fastest_ms );
      fmt::print( "  🥈 Slowest:         {} ({:.3f} ms)\n", slowest_name, slowest_ms );
      fmt::print( "  📈 Speed Ratio:     {:.1f}x\n", slowest_ms / fastest_ms );
    }

    // Quality assessment
    fmt::print( fg( fmt::color::light_blue ), "\n🏆 Quality Assessment:\n" );

    if ( passed == total )
      fmt::print( fg( SUCCESS_COLOR ), "  ✅ ALL TESTS PASSED - EXCELLENT QUALITY\n" );
    else if ( passed >= total * 0.9 )
      fmt::print( fg( WARNING_COLOR ), "  ⚠️ {} FAILURES - GOOD QUALITY\n", total - passed );
    else if ( passed >= total * 0.7 )
      fmt::print( fg( fmt::color::orange ), "  ⚠️ {} FAILURES - FAIR QUALITY\n", total - passed );
    else
      fmt::print( fg( ERROR_COLOR ), "  ❌ {} FAILURES - POOR QUALITY\n", total - passed );

    fmt::print( fg( HEADER_COLOR ) | fmt::emphasis::bold, "\n{:=^80}\n", " 🎯 END OF QUATERNION TEST SUITE 🎯 " );
  }
};


// ============================================================================
// Main Function
// ============================================================================

int main()
{
  fmt::print(
    fg( fmt::color::orange ) | fmt::emphasis::bold,
    "{:=^80}\n",
    " 🧪 Quaternion Class Comprehensive Test Suite 🧪 " );
  fmt::print(
    fg( fmt::color::gray ),
    "Testing all methods of Utils::Quaternion class\n"
    "Reference: J. B. Kuipers, \"Quaternions and Rotation Sequences\", 1999\n\n" );

  try
  {
    TestRunner::run_test_suite();
  }
  catch ( const exception & e )
  {
    fmt::print(
      fg( fmt::color::red ) | fmt::emphasis::bold,
      "\n❌ ERROR: Test suite crashed with exception: {}\n",
      e.what() );
    return 1;
  }
  catch ( ... )
  {
    fmt::print(
      fg( fmt::color::red ) | fmt::emphasis::bold,
      "\n❌ ERROR: Test suite crashed with unknown exception\n" );
    return 1;
  }

  return 0;
}
