/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2023-2024                                                 |
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

#include "Utils_TVD.hh"
#include "Utils_fmt.hh"
#include "Utils_eigen.hh"
#include "Utils_TicToc.hh"

namespace fs = std::filesystem;

using integer   = int;
using real_type = double;
using dmat_t    = Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic>;
using dvec_t    = Eigen::Matrix<real_type, Eigen::Dynamic, 1>;

//==============================================================================
// Unicode table characters
//==============================================================================
namespace unicode
{
  constexpr const char * CHECK = "✓";  // Check mark
  constexpr const char * CROSS = "✗";  // Cross mark
}  // namespace unicode

//==============================================================================
// Color theme
//==============================================================================
namespace colors
{
  auto title     = fg( fmt::color::cyan ) | fmt::emphasis::bold;
  auto header    = fg( fmt::color::yellow ) | fmt::emphasis::bold;
  auto value     = fg( fmt::color::white );
  auto good      = fg( fmt::color::green );
  auto warn      = fg( fmt::color::yellow );
  auto error     = fg( fmt::color::red );
  auto highlight = fg( fmt::color::magenta ) | fmt::emphasis::bold;
  auto metric    = fg( fmt::color::cyan );
  auto time      = fg( fmt::color::blue );
}  // namespace colors

//==============================================================================
// Test signal generators
//==============================================================================

dvec_t generate_step_signal( integer N, real_type noise_std = 0.1 )
{
  dvec_t  y( N );
  integer step1 = N / 4;
  integer step2 = 2 * N / 4;
  integer step3 = 3 * N / 4;

  std::random_device                  rd;
  std::mt19937                        gen( rd() );
  std::normal_distribution<real_type> dist( 0, noise_std );

  for ( integer i = 0; i < N; ++i )
  {
    real_type value;
    if ( i < step1 )
      value = 1.0;
    else if ( i < step2 )
      value = 0.5;
    else if ( i < step3 )
      value = 1.5;
    else
      value = 0.0;

    y( i ) = value + dist( gen );
  }
  return y;
}

dvec_t generate_smooth_signal( integer N, real_type noise_std = 0.1 )
{
  dvec_t                              t( N ), y( N );
  std::random_device                  rd;
  std::mt19937                        gen( rd() );
  std::normal_distribution<real_type> dist( 0, noise_std );

  for ( integer i = 0; i < N; ++i )
  {
    t( i ) = ( i * 4 * M_PI ) / N;
    y( i ) = sin( t( i ) ) + 0.5 * sin( 3 * t( i ) ) + dist( gen );
  }
  return y;
}

dvec_t generate_piecewise_linear( integer N, real_type noise_std = 0.1 )
{
  dvec_t                              y( N );
  std::random_device                  rd;
  std::mt19937                        gen( rd() );
  std::normal_distribution<real_type> dist( 0, noise_std );

  for ( integer i = 0; i < N; ++i )
  {
    if ( i < N / 3 )
      y( i ) = i * 0.01;
    else if ( i < 2 * N / 3 )
      y( i ) = 0.5 - ( i - N / 3 ) * 0.005;
    else
      y( i ) = 0.2 + ( i - 2 * N / 3 ) * 0.008;

    y( i ) += dist( gen );
  }
  return y;
}

dvec_t generate_spike_signal( integer N, real_type noise_std = 0.05 )
{
  dvec_t                              y = dvec_t::Zero( N );
  std::random_device                  rd;
  std::mt19937                        gen( rd() );
  std::normal_distribution<real_type> noise_dist( 0, noise_std );

  // Add spikes at specific positions
  std::vector<integer>   spike_positions  = { N / 5, 2 * N / 5, 3 * N / 5, 4 * N / 5 };
  std::vector<real_type> spike_amplitudes = { 2.0, -1.5, 1.8, -2.2 };

  integer NS = static_cast<integer>( spike_positions.size() );
  for ( integer i = 0; i < NS; ++i ) { y( spike_positions[i] ) = spike_amplitudes[i]; }

  // Add background noise
  for ( integer i = 0; i < N; ++i ) { y( i ) += noise_dist( gen ); }

  return y;
}

//==============================================================================
// Performance metrics structure
//==============================================================================

struct TVDMetrics
{
  real_type rmse{ 0 };             // Root Mean Square Error
  real_type psnr{ 0 };             // Peak Signal-to-Noise Ratio (dB)
  real_type tv_reduction{ 0 };     // Total Variation reduction (%)
  real_type mae{ 0 };              // Mean Absolute Error
  real_type max_error{ 0 };        // Maximum absolute error
  real_type snr_improvement{ 0 };  // SNR improvement (dB)
  real_type execution_ms{ 0 };     // Execution time (ms)
  integer   segments{ 0 };         // Number of constant segments

  // Method to compute all metrics
  void compute( const dvec_t & original, const dvec_t & noisy, const dvec_t & denoised, real_type exec_time = 0 )
  {
    integer N = static_cast<integer>( original.size() );

    // Compute errors
    real_type sum_sq = 0, sum_abs = 0;
    real_type max_err     = 0;
    real_type noisy_power = 0, denoised_power = 0;

    for ( integer i = 0; i < N; ++i )
    {
      real_type err = original( i ) - denoised( i );
      sum_sq += err * err;
      sum_abs += abs( err );
      max_err = std::max( max_err, abs( err ) );

      noisy_power += noisy( i ) * noisy( i );
      denoised_power += denoised( i ) * denoised( i );
    }

    rmse      = sqrt( sum_sq / N );
    mae       = sum_abs / N;
    max_error = max_err;

    // PSNR
    real_type max_val = original.maxCoeff() - original.minCoeff();
    if ( max_val > 0 && rmse > 0 ) { psnr = 20 * log10( max_val / rmse ); }

    // SNR improvement
    real_type snr_noisy    = 10 * log10( noisy_power / ( sum_sq + 1e-10 ) );
    real_type snr_denoised = 10 * log10( denoised_power / ( sum_sq + 1e-10 ) );
    snr_improvement        = snr_denoised - snr_noisy;

    // TV reduction
    real_type tv_noisy = 0, tv_denoised = 0;
    for ( integer i = 0; i < N - 1; ++i )
    {
      tv_noisy += abs( noisy( i + 1 ) - noisy( i ) );
      tv_denoised += abs( denoised( i + 1 ) - denoised( i ) );
    }
    tv_reduction = 100.0 * ( tv_noisy - tv_denoised ) / tv_noisy;

    // Count segments (constant regions)
    segments            = 1;
    real_type threshold = ( denoised.maxCoeff() - denoised.minCoeff() ) * 1e-3;
    for ( integer i = 0; i < N - 1; ++i )
    {
      if ( abs( denoised( i + 1 ) - denoised( i ) ) > threshold ) { segments++; }
    }

    execution_ms = exec_time;
  }

  // Method to print as a table row
  void print_row( real_type lambda ) const
  {
    fmt::print( colors::value, "{:10.6f}", lambda );
    fmt::print( " │ " );
    fmt::print( colors::metric, "{:10.6f}", rmse );
    fmt::print( " │ " );
    fmt::print( colors::metric, "{:7.2f} dB", psnr );
    fmt::print( " │ " );
    fmt::print( colors::metric, "{:10.4f}", mae );
    fmt::print( " │ " );
    fmt::print( colors::metric, "{:9.2f}%", tv_reduction );
    fmt::print( " │ " );
    fmt::print( colors::time, "{:7.3f} ms", execution_ms );
    fmt::print( " │ " );
    fmt::print( colors::highlight, "{:6d}", segments );
    fmt::print( " │\n" );
  }
};

//==============================================================================
// Table printing utilities
//==============================================================================

void print_table_header( std::vector<std::string> const & headers, std::vector<int> const & widths )
{
  // Top border
  fmt::print( "┌" );
  integer NH = static_cast<integer>( headers.size() );
  for ( integer i = 0; i < NH; ++i )
  {
    fmt::print( "{:─^{}}", "", widths[i] + 2 );
    if ( i < NH - 1 ) fmt::print( "┬" );
  }
  fmt::print( "┐\n" );

  // Header row
  fmt::print( "│" );
  for ( integer i = 0; i < NH; ++i )
  {
    fmt::print( " {:^{}} ", headers[i], widths[i] );
    fmt::print( "│" );
  }
  fmt::print( "\n" );

  // Separator
  fmt::print( "├" );
  for ( integer i = 0; i < NH; ++i )
  {
    fmt::print( "{:─^{}}", "", widths[i] + 2 );
    if ( i < NH - 1 ) fmt::print( "┼" );
  }
  fmt::print( "┤\n" );
}

void print_table_footer( std::vector<int> const & widths )
{
  fmt::print( "└" );
  integer WS = static_cast<integer>( widths.size() );
  for ( integer i = 0; i < WS; ++i )
  {
    fmt::print( "{:─^{}}", "", widths[i] + 2 );
    if ( i < WS - 1 ) fmt::print( "┴" );
  }
  fmt::print( "┘\n" );
}

//==============================================================================
// Test case structure
//==============================================================================

struct TestCase
{
  std::string            name;
  std::string            description;
  dvec_t                 original;
  dvec_t                 noisy;
  std::vector<real_type> lambdas;
  std::string            filename_prefix;

  TestCase(
    std::string            n,
    std::string            desc,
    integer                N,
    std::vector<real_type> lams = { 0.01, 0.05, 0.1, 0.2, 0.5, 1.0 } )
    : name( std::move( n ) ), description( std::move( desc ) ), lambdas( std::move( lams ) )
  {
    original.resize( N );
    noisy.resize( N );
  }
};

//==============================================================================
// TVD Test Runner
//==============================================================================

class TVDTestRunner
{
private:
  std::vector<TestCase> test_cases;
  std::string           output_dir;

public:
  TVDTestRunner( std::string dir = "test_results" ) : output_dir( std::move( dir ) )
  {
    // Create output directory
    fs::create_directories( output_dir );
  }

  void add_test_case( TestCase && test_case ) { test_cases.emplace_back( std::move( test_case ) ); }

  void run_all_tests()
  {
    fmt::print( colors::title | fmt::emphasis::bold, "\n{:=^80}\n", " TOTAL VARIATION DENOISING TEST SUITE " );
    fmt::print( colors::title, "Algorithm: Condat's O(N) TVD\n" );

    std::vector<std::tuple<std::string, real_type, real_type, real_type>> best_results;

    for ( const auto & test_case : test_cases )
    {
      auto result = run_single_test( test_case );
      best_results.push_back( result );
    }

    print_best_results_summary( best_results );
    run_edge_case_tests();
    run_performance_benchmark();
  }

private:
  std::tuple<std::string, real_type, real_type, real_type> run_single_test( TestCase const & test_case )
  {
    integer N = static_cast<integer>( test_case.original.size() );
    dvec_t  denoised( N );

    fmt::print( colors::header, "\n{:─^80}\n", " TEST: " + test_case.name + " " );
    fmt::print( colors::value, "{}\n\n", test_case.description );

    // Print signal statistics
    fmt::print( colors::value, "Signal Statistics:\n" );
    fmt::print( "  Length: {} samples\n", N );
    fmt::print( "  Original range: [{:.4f}, {:.4f}]\n", test_case.original.minCoeff(), test_case.original.maxCoeff() );
    fmt::print( "  Noise σ: {:.4f}\n", sqrt( ( test_case.noisy - test_case.original ).array().square().mean() ) );

    // Table headers
    std::vector<std::string> headers = { "λ", "RMSE", "PSNR", "MAE", "TV Red%", "Time", "Seg" };
    std::vector<int>         widths  = { 9, 10, 10, 10, 10, 10, 6 };

    print_table_header( headers, widths );

    // Test each lambda
    std::vector<TVDMetrics> metrics_list;
    real_type               best_rmse   = std::numeric_limits<real_type>::max();
    real_type               best_lambda = 0;
    real_type               best_psnr   = 0;

    for ( real_type lambda : test_case.lambdas )
    {
      Utils::TicToc tm;
      tm.tic();
      Utils::TVD<real_type>::denoise( N, test_case.noisy.data(), lambda, denoised.data() );
      tm.toc();
      real_type exec_time = tm.elapsed_ms();

      TVDMetrics metrics;
      metrics.compute( test_case.original, test_case.noisy, denoised, exec_time );
      metrics_list.push_back( metrics );

      // Print row
      fmt::print( "│" );
      metrics.print_row( lambda );

      // Update best
      if ( metrics.rmse < best_rmse )
      {
        best_rmse   = metrics.rmse;
        best_lambda = lambda;
        best_psnr   = metrics.psnr;
      }

      // Save results
      if ( !test_case.filename_prefix.empty() )
      {
        std::string   filename = fmt::format( "{}/{}_{:.3f}.csv", output_dir, test_case.filename_prefix, lambda );
        std::ofstream file( filename );
        file << "index,original,noisy,denoised\n";
        for ( integer i = 0; i < N; ++i )
        {
          file << fmt::format(
            "{},{:.6f},{:.6f},{:.6f}\n",
            i,
            test_case.original( i ),
            test_case.noisy( i ),
            denoised( i ) );
        }
        file.close();
      }
    }

    print_table_footer( widths );

    // Print best result
    fmt::print( colors::good, "\n{} Best result: ", unicode::CHECK );
    fmt::print( "λ = {:.4f}, RMSE = {:.6f}, PSNR = {:.2f} dB\n", best_lambda, best_rmse, best_psnr );

    return make_tuple( test_case.name, best_lambda, best_rmse, best_psnr );
  }

  void print_best_results_summary(
    std::vector<std::tuple<std::string, real_type, real_type, real_type>> const & results )
  {
    print( colors::title, "\n{:=^80}\n", " SUMMARY OF BEST RESULTS " );

    std::vector<std::string> headers = { "Test Case", "Best λ", "RMSE", "PSNR (dB)", "Quality" };
    std::vector<int>         widths  = { 20, 10, 12, 12, 15 };

    print_table_header( headers, widths );

    for ( const auto & [name, lambda, rmse, psnr] : results )
    {
      fmt::print( "│" );
      fmt::print( colors::value, " {:^{}} ", name, widths[0] );
      fmt::print( "│" );
      fmt::print( colors::highlight, " {:^{}.4f} ", lambda, widths[1] );
      fmt::print( "│" );
      fmt::print( colors::metric, " {:^{}.6f} ", rmse, widths[2] );
      fmt::print( "│" );
      fmt::print( colors::metric, " {:^{}.2f} ", psnr, widths[3] );
      fmt::print( "│" );

      // Quality indicator
      std::string quality;
      if ( psnr > 30 ) { fmt::print( colors::good, " {:^{}} ", "Excellent", widths[4] ); }
      else if ( psnr > 20 ) { fmt::print( colors::warn, " {:^{}} ", "Good", widths[4] ); }
      else
      {
        fmt::print( colors::error, " {:^{}} ", "Poor", widths[4] );
      }
      fmt::print( "│\n" );
    }

    print_table_footer( widths );
  }

  void run_edge_case_tests()
  {
    fmt::print( colors::title, "\n{:=^80}\n", " EDGE CASE TESTS " );

    std::vector<std::pair<std::string, bool>> results;

    // Test 1: Constant signal
    {
      integer   N      = 100;
      dvec_t    signal = dvec_t::Constant( N, 1.0 );
      dvec_t    denoised( N );
      real_type lambda = 0.1;

      Utils::TVD<real_type>::denoise( N, signal.data(), lambda, denoised.data() );

      real_type max_diff = ( signal - denoised ).array().abs().maxCoeff();
      bool      passed   = max_diff < 1e-10;
      results.emplace_back( "Constant signal", passed );

      if ( passed ) { print( colors::good, "{}: max deviation = {:.2e}\n", unicode::CHECK, max_diff ); }
      else
      {
        print( colors::error, "{}: max deviation = {:.2e} FAILED\n", unicode::CROSS, max_diff );
      }
    }

    // Test 2: Single spike preservation
    {
      integer N       = 200;
      dvec_t  signal  = dvec_t::Zero( N );
      signal( N / 2 ) = 5.0;

      dvec_t    denoised( N );
      real_type lambda = 0.5;

      Utils::TVD<real_type>::denoise( N, signal.data(), lambda, denoised.data() );

      real_type spike_value = denoised( N / 2 );
      bool      passed      = spike_value > 2.5;  // Spike should be preserved
      results.emplace_back( "Spike preservation", passed );

      if ( passed ) { print( colors::good, "{}: spike preserved = {:.2f}\n", unicode::CHECK, spike_value ); }
      else
      {
        print( colors::error, "{}: spike preserved = {:.2f} FAILED\n", unicode::CROSS, spike_value );
      }
    }

    // Test 3: Large lambda (over-smoothing)
    {
      integer N      = 500;
      dvec_t  signal = dvec_t::Random( N );

      dvec_t    denoised( N );
      real_type lambda = 10.0;  // Very large

      Utils::TVD<real_type>::denoise( N, signal.data(), lambda, denoised.data() );

      real_type std_dev = sqrt( denoised.array().square().mean() );
      bool      passed  = std_dev < 0.5;  // Should be heavily smoothed
      results.emplace_back( "Large λ smoothing", passed );

      if ( passed ) { print( colors::good, "{}: std dev after smoothing = {:.4f}\n", unicode::CHECK, std_dev ); }
      else
      {
        print( colors::error, "{}: std dev after smoothing = {:.4f} FAILED\n", unicode::CROSS, std_dev );
      }
    }

    // Summary
    integer passed_count =
      static_cast<integer>( count_if( results.begin(), results.end(), []( const auto & p ) { return p.second; } ) );

    print( colors::header, "\nEdge Cases: {}/{} passed\n", passed_count, results.size() );
  }

  void run_performance_benchmark()
  {
    print( colors::title, "\n{:=^80}\n", " PERFORMANCE BENCHMARK " );

    std::vector<integer> sizes  = { 100, 1000, 10000, 100000, 1000000 };
    real_type            lambda = 0.1;

    std::vector<std::string> headers = { "Size", "Time (ms)", "Time/point (µs)", "Rate (kS/s)" };
    std::vector<int>         widths  = { 10, 12, 16, 15 };

    print_table_header( headers, widths );

    for ( integer size : sizes )
    {
      if ( size > 1000000 ) continue;  // Limit for quick testing

      dvec_t signal = dvec_t::Random( size );
      dvec_t denoised( size );

      // Warm-up
      Utils::TVD<real_type>::denoise( std::min( size, 1000 ), signal.data(), lambda, denoised.data() );

      // Measure
      Utils::TicToc tm;
      tm.tic();
      Utils::TVD<real_type>::denoise( size, signal.data(), lambda, denoised.data() );
      tm.toc();

      real_type time_ms        = tm.elapsed_ms();
      real_type time_per_point = time_ms * 1000 / size;  // µs per point
      real_type rate           = size / time_ms;         // kS/s

      fmt::print( "│" );
      fmt::print( colors::value, " {:^{}} ", size, widths[0] );
      fmt::print( "│" );
      fmt::print( colors::time, " {:^{}.3f} ", time_ms, widths[1] );
      fmt::print( "│" );
      fmt::print( colors::time, " {:^{}.3f} ", time_per_point, widths[2] );
      fmt::print( "│" );
      fmt::print( colors::time, " {:^{}.1f} ", rate / 1000, widths[3] );
      fmt::print( "│\n" );
    }

    print_table_footer( widths );
  }
};

//==============================================================================
// Main function
//==============================================================================

int main()
{
  try
  {
    // Create test runner
    TVDTestRunner runner( "tvd_test_results" );

    constexpr integer N = 1000;

    // Create test cases
    {
      // Step function test
      TestCase tc( "Step Function", "Piecewise constant signal with additive Gaussian noise", N );
      tc.original        = generate_step_signal( N, 0.0 );
      tc.noisy           = generate_step_signal( N, 0.1 );
      tc.lambdas         = { 0.01, 0.05, 0.1, 0.2, 0.5 };
      tc.filename_prefix = "step";
      runner.add_test_case( std::move( tc ) );
    }

    {
      // Smooth signal test
      TestCase tc( "Smooth Signal", "Sinusoidal signal with multiple frequencies", N );
      tc.original        = generate_smooth_signal( N, 0.0 );
      tc.noisy           = generate_smooth_signal( N, 0.1 );
      tc.lambdas         = { 0.01, 0.05, 0.1, 0.2, 0.3 };
      tc.filename_prefix = "smooth";
      runner.add_test_case( std::move( tc ) );
    }

    {
      // Piecewise linear test
      TestCase tc( "Piecewise Linear", "Signal with piecewise linear segments", N );
      tc.original        = generate_piecewise_linear( N, 0.0 );
      tc.noisy           = generate_piecewise_linear( N, 0.1 );
      tc.lambdas         = { 0.05, 0.1, 0.2, 0.3, 0.5 };
      tc.filename_prefix = "piecewise";
      runner.add_test_case( std::move( tc ) );
    }

    {
      // Spike signal test
      TestCase tc( "Spike Signal", "Signal with sparse spikes in noise", N );
      tc.original        = dvec_t::Zero( N );  // For spikes, original is zero
      tc.noisy           = generate_spike_signal( N, 0.1 );
      tc.lambdas         = { 0.1, 0.2, 0.5, 1.0, 2.0 };
      tc.filename_prefix = "spike";
      runner.add_test_case( std::move( tc ) );
    }

    // Run all tests
    runner.run_all_tests();

    print( colors::title | fmt::emphasis::bold, "\n{:=^80}\n", " TEST SUITE COMPLETED SUCCESSFULLY " );
    print( colors::value, "Results saved to: tvd_test_results/\n" );
  }
  catch ( const std::exception & e )
  {
    fmt::print( colors::error | fmt::emphasis::bold, "\nError: {}\n", e.what() );
    return 1;
  }

  return 0;
}
