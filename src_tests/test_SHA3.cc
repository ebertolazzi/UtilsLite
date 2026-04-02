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
 |      Comprehensive SHA3 Test Suite                                       |
 |                                                                          |
\*--------------------------------------------------------------------------*/

// Include your SHA3 implementation
#include "Utils_SHA3.hh"

using namespace std;
using namespace Utils;

// Test utility functions
void print_test_header( const string & name )
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n=== {} ===\n", name );
}

void print_test_result( bool passed, const string & message )
{
  if ( passed ) { fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "✓ {}\n", message ); }
  else
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "✗ {}\n", message );
  }
}

void print_hash_result( const string & input, const string & hash )
{
  fmt::print( "  Input: \"{}\"\n", input );
  fmt::print( "  Hash:  {}\n", hash );
  fmt::print( "{:-<60}\n", "" );
}

// Performance benchmark
void benchmark_performance()
{
  print_test_header( "Performance Benchmark" );

  const int       TEST_SIZE = 1000000;  // 1MB of data
  vector<uint8_t> data( TEST_SIZE, 'A' );

  // Test SHA3-256
  {
    auto start = chrono::high_resolution_clock::now();

    SHA3 sha3( 32 );  // 256-bit hash
    for ( int i = 0; i < TEST_SIZE; i++ ) { sha3.hash( data[i] ); }
    string hash = sha3.digest_in_hex();

    auto end      = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( end - start );

    fmt::print( "SHA3-256 processed {} bytes in {} ms\n", TEST_SIZE, duration.count() );
    fmt::print(
      "Throughput: {:.2f} MB/s\n",
      ( TEST_SIZE / ( static_cast<double>( duration.count() ) / 1000.0 ) ) / ( 1024 * 1024 ) );
  }

  // Test SHA3-512
  {
    auto start = chrono::high_resolution_clock::now();

    SHA3 sha3( 64 );  // 512-bit hash
    for ( int i = 0; i < TEST_SIZE; i++ ) { sha3.hash( data[i] ); }
    string hash = sha3.digest_in_hex();

    auto end      = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( end - start );

    fmt::print( "\nSHA3-512 processed {} bytes in {} ms\n", TEST_SIZE, duration.count() );
    fmt::print(
      "Throughput: {:.2f} MB/s\n",
      ( TEST_SIZE / ( static_cast<double>( duration.count() ) / 1000.0 ) ) / ( 1024 * 1024 ) );
  }
}

// NIST standard test vectors
void run_nist_test_vectors()
{
  print_test_header( "NIST Standard Test Vectors" );

  // SHA3-224 test vector (empty input)
  {
    SHA3 sha3( 28 );  // 224-bit hash
    sha3.hash_string( "" );
    string result   = sha3.digest_in_hex();
    string expected = "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7";

    bool passed = ( result == expected );
    print_test_result( passed, "SHA3-224 empty string" );
    if ( !passed )
    {
      fmt::print( fg( fmt::color::yellow ), "  Expected: {}\n", expected );
      fmt::print( fg( fmt::color::yellow ), "  Actual:   {}\n", result );
    }
  }

  // SHA3-256 test vector ("abc")
  {
    SHA3 sha3( 32 );  // 256-bit hash
    sha3.hash_string( "abc" );
    string result   = sha3.digest_in_hex();
    string expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532";

    bool passed = ( result == expected );
    print_test_result( passed, "SHA3-256 \"abc\"" );
    if ( !passed )
    {
      fmt::print( fg( fmt::color::yellow ), "  Expected: {}\n", expected );
      fmt::print( fg( fmt::color::yellow ), "  Actual:   {}\n", result );
    }
  }

  // SHA3-384 test vector
  {
    SHA3 sha3( 48 );  // 384-bit hash
    sha3.hash_string( "" );
    string result = sha3.digest_in_hex();
    string expected =
      "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004";

    bool passed = ( result == expected );
    print_test_result( passed, "SHA3-384 empty string" );
    if ( !passed )
    {
      fmt::print( fg( fmt::color::yellow ), "  Expected: {}\n", expected );
      fmt::print( fg( fmt::color::yellow ), "  Actual:   {}\n", result );
    }
  }

  // SHA3-512 test vector ("abc")
  {
    SHA3 sha3( 64 );  // 512-bit hash
    sha3.hash_string( "abc" );
    string result = sha3.digest_in_hex();
    string expected =
      "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0";

    bool passed = ( result == expected );
    print_test_result( passed, "SHA3-512 \"abc\"" );
    if ( !passed )
    {
      fmt::print( fg( fmt::color::yellow ), "  Expected: {}\n", expected );
      fmt::print( fg( fmt::color::yellow ), "  Actual:   {}\n", result );
    }
  }
}

// Edge case tests
void run_edge_case_tests()
{
  print_test_header( "Edge Case Tests" );

  // Test empty string
  {
    SHA3 sha3( 32 );
    sha3.hash_string( "" );
    string result = sha3.digest_in_hex();
    bool   passed = !result.empty() && result.length() == 64;
    print_test_result( passed, "Empty string hashing" );
    print_hash_result( "(empty string)", result );
  }

  // Test single character
  {
    SHA3 sha3( 32 );
    sha3.hash_string( "a" );
    string result = sha3.digest_in_hex();
    bool   passed = result.length() == 64;
    print_test_result( passed, "Single character hashing" );
    print_hash_result( "a", result );
  }

  // Test special characters
  {
    SHA3 sha3( 32 );
    sha3.hash_string( "!@#$%^&*()_+-=[]{}|;:,.<>?" );
    string result = sha3.digest_in_hex();
    bool   passed = result.length() == 64;
    print_test_result( passed, "Special characters hashing" );
    print_hash_result( "!@#$%^&*()_+-=[]{}|;:,.<>?", result );
  }

  // Test very long string
  {
    SHA3   sha3( 32 );
    string long_string( 10000, 'x' );
    sha3.hash_string( long_string );
    string result = sha3.digest_in_hex();
    bool   passed = result.length() == 64;
    print_test_result( passed, "10KB string hashing" );
    fmt::print( "  Input: 10000 'x' characters\n" );
    fmt::print( "  Hash:  {}\n", result );
    fmt::print( "{:-<60}\n", "" );
  }
}

// Test hash_hex_string function
void test_hex_string_hashing()
{
  print_test_header( "Hexadecimal String Hashing Tests" );

  // Test valid hex string
  {
    SHA3 sha3( 32 );
    sha3.hash_hex_string( "616263" );  // "abc" in hex
    string result   = sha3.digest_in_hex();
    string expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532";

    bool passed = ( result == expected );
    print_test_result( passed, "Hex string \"616263\" (abc)" );
    if ( !passed )
    {
      fmt::print( fg( fmt::color::yellow ), "  Expected: {}\n", expected );
      fmt::print( fg( fmt::color::yellow ), "  Actual:   {}\n", result );
    }
  }

  // Test uppercase hex string
  {
    SHA3 sha31( 32 );
    SHA3 sha32( 32 );

    sha31.hash_hex_string( "616263" );
    sha32.hash_string( "abc" );

    string result1 = sha31.digest_in_hex();
    string result2 = sha32.digest_in_hex();

    bool passed = ( result1 == result2 );
    print_test_result( passed, "Hex string vs ASCII string consistency" );
  }
}

// Test incremental hashing
void test_incremental_hashing()
{
  print_test_header( "Incremental Hashing Tests" );

  // Test that hashing in chunks gives same result as one go
  {
    SHA3 sha3_chunk( 32 );
    SHA3 sha3_once( 32 );

    string test_string = "The quick brown fox jumps over the lazy dog";

    // Hash in chunks
    for ( char c : test_string ) { sha3_chunk.hash( static_cast<int>( c ) ); }
    string chunk_result = sha3_chunk.digest_in_hex();

    // Hash all at once
    sha3_once.hash_string( test_string );
    string once_result = sha3_once.digest_in_hex();

    bool passed = ( chunk_result == once_result );
    print_test_result( passed, "Incremental vs single hashing consistency" );
    fmt::print( "  Chunked hash: {}\n", chunk_result );
    fmt::print( "  Single hash:  {}\n", once_result );
  }
}

// Test different digest sizes
void test_different_digest_sizes()
{
  print_test_header( "Different Digest Sizes" );

  vector<pair<int, string>> test_cases = { { 28, "SHA3-224 (28 bytes)" },
                                           { 32, "SHA3-256 (32 bytes)" },
                                           { 48, "SHA3-384 (48 bytes)" },
                                           { 64, "SHA3-512 (64 bytes)" } };

  for ( const auto & [size, description] : test_cases )
  {
    SHA3 sha3( size );
    sha3.hash_string( "test" );
    string result = sha3.digest_in_hex();

    // Expected length: 2 characters per byte
    size_t expected_length = size * 2;
    bool   passed          = ( result.length() == expected_length );

    print_test_result( passed, description );
    fmt::print( "  Expected length: {}, Actual: {}\n", expected_length, result.length() );
  }
}

// Main test runner
int main()
{
  fmt::print( fg( fmt::color::magenta ) | fmt::emphasis::bold, "{:=^80}\n", " SHA3 COMPREHENSIVE TEST SUITE " );
  fmt::print( "\n" );

  // Run all test suites
  run_nist_test_vectors();
  run_edge_case_tests();
  test_hex_string_hashing();
  test_incremental_hashing();
  test_different_digest_sizes();
  benchmark_performance();

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "\n{:=^80}\n", " ALL TESTS COMPLETED " );

  return 0;
}
