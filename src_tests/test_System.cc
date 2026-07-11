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

// Includi il nuovo header
#include "Utils.hh"
#include "Utils_System.hh"
#include "Utils_fmt.hh"

#include <filesystem>

using namespace Utils;
using namespace std;
namespace fs = std::filesystem;

// Funzioni di utilità per la formattazione
void print_section( const string & title )
{
  fmt::print(
    fg( fmt::color::deep_sky_blue ) | fmt::emphasis::bold,
    "\n┌─────────────────────────────────────────────────────────┐\n" );
  fmt::print( fg( fmt::color::deep_sky_blue ) | fmt::emphasis::bold, "│ {:^55} │\n", title );
  fmt::print(
    fg( fmt::color::deep_sky_blue ) | fmt::emphasis::bold,
    "└─────────────────────────────────────────────────────────┘\n" );
}

void print_success( const string & message )
{
  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, "  ✓ {}\n", message );
}

void print_info( const string & message )
{
  fmt::print( fg( fmt::color::light_blue ), "  ℹ {}\n", message );
}

void print_warning( const string & message )
{
  fmt::print( fg( fmt::color::gold ) | fmt::emphasis::bold, "  ⚠ {}\n", message );
}

void print_error( const string & message )
{
  fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "  ✗ {}\n", message );
}

void print_result( const string & label, const string & value, bool success = true )
{
  if ( success )
  {
    fmt::print( fg( fmt::color::white ), "    {:<25}: ", label );
    fmt::print( fg( fmt::color::light_green ), "{}\n", value );
  }
  else
  {
    fmt::print( fg( fmt::color::white ), "    {:<25}: ", label );
    fmt::print( fg( fmt::color::light_salmon ), "{}\n", value );
  }
}

// Test della classe Architecture
void test_architecture_class()
{
  print_section( "TEST ARCHITECTURE CLASS" );

  try
  {
    auto   arch     = Architecture::get_architecture();
    string arch_str = Architecture::get_architecture_string();

    print_result( "Architecture", arch_str );
    print_result( "Vendor", Architecture::get_cpu_vendor() );
    print_result( "Model", Architecture::get_cpu_model() );
    print_result( "Core Count", to_string( Architecture::get_cpu_count() ) );
    print_result( "Full CPU Info", Architecture::get_cpu_info_string() );

    // Test features specifiche per architettura
    print_info( "Architecture-Specific Features:" );

    switch ( arch )
    {
      case Architecture::ArchType::X86:
      case Architecture::ArchType::X64:
#ifdef UTILS_CPU_X86
      {
        auto features = Architecture::get_features();
        print_result( "MMX", features.mmx ? "✅ Supported" : "❌ Not supported" );
        print_result( "SSE", features.sse ? "✅ Supported" : "❌ Not supported" );
        print_result( "SSE2", features.sse2 ? "✅ Supported" : "❌ Not supported" );
        print_result( "SSE3", features.sse3 ? "✅ Supported" : "❌ Not supported" );
        print_result( "AVX", features.avx ? "✅ Supported" : "❌ Not supported" );
        print_result( "AVX2", features.avx2 ? "✅ Supported" : "❌ Not supported" );
        print_result( "AES", features.aes ? "✅ Supported" : "❌ Not supported" );
        print_result( "FMA", features.fma ? "✅ Supported" : "❌ Not supported" );
      }
#else
        print_info( "x86 features require UTILS_CPU_X86 define" );
#endif
      break;

      case Architecture::ArchType::ARM32:
      case Architecture::ArchType::ARM64:
#ifdef UTILS_CPU_ARM
        print_info( "ARM features detected through /proc/cpuinfo" );
        // Note: These functions need UTILS_CPU_ARM defined
        print_result( "NEON", "Detection requires UTILS_CPU_ARM define" );
        print_result( "ARM Crypto", "Detection requires UTILS_CPU_ARM define" );
#else
        print_info( "ARM features require UTILS_CPU_ARM define" );
#endif
        break;

      case Architecture::ArchType::POWERPC:
      case Architecture::ArchType::POWERPC64:
#ifdef UTILS_CPU_PPC
        print_info( "PowerPC features" );
        print_result( "AltiVec", "Detection requires UTILS_CPU_PPC define" );
        print_result( "VSX", "Detection requires UTILS_CPU_PPC define" );
#else
        print_info( "PowerPC features require UTILS_CPU_PPC define" );
#endif
        break;

      case Architecture::ArchType::MIPS:
      case Architecture::ArchType::MIPS64:
#ifdef UTILS_CPU_MIPS
        print_info( "MIPS features" );
        print_result( "MSA", "Detection requires UTILS_CPU_MIPS define" );
#else
        print_info( "MIPS features require UTILS_CPU_MIPS define" );
#endif
        break;

      case Architecture::ArchType::RISCV32:
      case Architecture::ArchType::RISCV64:
#ifdef UTILS_CPU_RISCV
        print_info( "RISC-V features" );
        print_result( "Vector Ext", "Detection requires UTILS_CPU_RISCV define" );
#else
        print_info( "RISC-V features require UTILS_CPU_RISCV define" );
#endif
        break;

      default: print_info( "Unknown architecture - feature detection not available" ); break;
    }

    print_success( "Architecture class test completed" );
  }
  catch ( const exception & e )
  {
    print_error( "Architecture class test failed: " + string( e.what() ) );
  }
}

// Test delle funzioni di sistema
void test_system_functions()
{
  print_section( "TEST SYSTEM FUNCTIONS" );

  try
  {
    // 1. Nome host
    string hostname = get_host_name();
    print_result( "Host Name", hostname, !hostname.empty() );

    // 2. Nome utente
    string username = get_user_name();
    print_result( "User Name", username, !username.empty() );

    // 3. Home directory
    string home = get_home_directory();
    print_result( "Home Directory", home, !home.empty() );

    // 4. Percorso eseguibile
    string exe_path = get_executable_path_name();
    print_result( "Executable Path", exe_path, !exe_path.empty() );

    // 5. Variabili d'ambiente
    string path_value;
    bool   got_env = get_environment( "PATH", path_value );
    print_result(
      "PATH Environment",
      got_env ? "Retrieved (" + to_string( path_value.length() ) + " chars)" : "Not found",
      got_env );

    // 6. Test set environment
    set_environment( "TEST_VAR_UTILS", "Hello from SystemUtils!", true );
    string test_var;
    bool   test_got = get_environment( "TEST_VAR_UTILS", test_var );
    print_result(
      "Test Environment Var",
      test_got ? "✓ Set and retrieved: " + test_var : "✗ Failed",
      test_got && test_var == "Hello from SystemUtils!" );

    print_success( "System functions test completed" );
  }
  catch ( const exception & e )
  {
    print_error( "System functions test failed: " + string( e.what() ) );
  }
}

// Test delle funzioni di rete
void test_network_functions()
{
  print_section( "TEST NETWORK FUNCTIONS" );

  try
  {
    // 1. MAC Address
    map<string, string> mac_addr;
    get_MAC_address( mac_addr );

    print_info( fmt::format( "Found {} network interface(s):", mac_addr.size() ) );

    for ( const auto & [iface, addr] : mac_addr )
    {
      fmt::print( fg( fmt::color::cyan ), "    Interface: {:<15} → ", iface );
      fmt::print( fg( fmt::color::light_green ), "{}\n", addr );
    }

    if ( mac_addr.empty() )
    {
      print_warning( "No MAC addresses found. This is normal on some systems (VMs, containers)." );
    }

    // 2. IP Address
    vector<string> ip_addr;
    get_IP_address( ip_addr );

    print_info( fmt::format( "Found {} IP address(es):", ip_addr.size() ) );

    for ( size_t i = 0; i < ip_addr.size(); ++i )
    {
      fmt::print( fg( fmt::color::light_cyan ), "    [{:2}] {}\n", i, ip_addr[i] );
    }

    if ( ip_addr.empty() ) { print_warning( "No IP addresses found. Network may be disconnected." ); }

    print_success( "Network functions test completed" );
  }
  catch ( const exception & e )
  {
    print_error( "Network functions test failed: " + string( e.what() ) );
  }
}

// Test delle funzioni file system
void test_filesystem_functions()
{
  print_section( "TEST FILESYSTEM FUNCTIONS" );

  try
  {
    // Percorsi di test
    vector<pair<string, string>> test_paths = { { "/tmp", "Temp directory (Unix)" },
                                                { "/usr/bin", "System binary directory" },
                                                { ".", "Current directory" },
                                                { __FILE__, "This test file" } };

#if defined( _WIN32 ) || defined( _WIN64 )
    test_paths.push_back( { "C:\\Windows", "Windows directory" } );
    test_paths.push_back( { "C:\\Windows\\System32", "System32 directory" } );
#endif

    print_info( "Testing path existence:" );

    for ( const auto & [path, description] : test_paths )
    {
      bool is_dir  = check_if_dir_exists( path );
      bool is_file = check_if_file_exists( path );

      fmt::print( fg( fmt::color::white ), "    {:<40} → ", description );

      if ( is_dir && !is_file ) { fmt::print( fg( fmt::color::dodger_blue ), "📁 Directory\n" ); }
      else if ( !is_dir && is_file ) { fmt::print( fg( fmt::color::light_green ), "📄 File\n" ); }
      else if ( is_dir && is_file ) { fmt::print( fg( fmt::color::orange ), "⚠ Both (unexpected)\n" ); }
      else
      {
        fmt::print( fg( fmt::color::light_gray ), "❌ Not found\n" );
      }
    }

    // Test funzioni di percorso
    print_info( "Testing path manipulation functions:" );

    struct PathTest
    {
      string path;
      string expected_basename;
      string expected_filename;
      string expected_extension;
    };

    vector<PathTest> path_tests = {
      { "/home/user/file.txt", "/home/user", "file.txt", ".txt" },
      { "/usr/local/bin/program", "/usr/local/bin", "program", "" },
      { "C:\\Windows\\System32\\kernel32.dll", "C:\\Windows\\System32", "kernel32.dll", ".dll" },
      { "relative/path/to/file.jpg", "relative/path/to", "file.jpg", ".jpg" }
    };

    for ( const auto & test : path_tests )
    {
      string basename  = get_basename( test.path );
      string filename  = get_filename( test.path );
      string extension = get_extension( test.path );

      fmt::print( fg( fmt::color::white ), "\n    Path: {}\n", test.path );
      fmt::print( fg( fmt::color::cyan ), "      Basename:  {}\n", basename );
      fmt::print( fg( fmt::color::cyan ), "      Filename:  {}\n", filename );
      fmt::print( fg( fmt::color::cyan ), "      Extension: {}\n", extension );
    }

    // Test creazione directory
    print_info( "Testing directory creation:" );

    string test_dir = "./test_directory_" + to_string( time( nullptr ) );

    if ( make_directory( test_dir ) )
    {
      fmt::print( fg( fmt::color::lime_green ), "    ✓ Created directory: {}\n", test_dir );

      // Verifica che esista
      if ( check_if_dir_exists( test_dir ) )
      {
        fmt::print( fg( fmt::color::lime_green ), "    ✓ Directory verified\n" );
      }

      std::error_code ec;
      if ( fs::remove( test_dir, ec ) )
      {
        fmt::print( fg( fmt::color::lime_green ), "    ✓ Removed directory: {}\n", test_dir );
      }
      else
      {
        print_warning( "Test directory created at: " + test_dir + " (cleanup failed: " + ec.message() + ")" );
      }
    }
    else
    {
      fmt::print( fg( fmt::color::red ), "    ✗ Failed to create directory: {}\n", test_dir );
    }

    print_success( "Filesystem functions test completed" );
  }
  catch ( const exception & e )
  {
    print_error( "Filesystem functions test failed: " + string( e.what() ) );
  }
}

// Test delle funzioni data/ora
void test_datetime_functions()
{
  print_section( "TEST DATE/TIME FUNCTIONS" );

  try
  {
    // 1. Data corrente
    string date = get_date();
    print_result( "Current Date", date, !date.empty() );

    // 2. Ora corrente
    string time = get_day_time();
    print_result( "Current Time", time, !time.empty() );

    // 3. Data e ora combinate
    string datetime = get_day_time_and_date();
    print_result( "Date & Time", datetime, !datetime.empty() );

    // 4. Data/ora per log
    string log_datetime = get_log_date_time();
    print_result( "Log Timestamp", log_datetime, !log_datetime.empty() );

    // Formattazione bella con Unicode
    fmt::print( fg( fmt::color::light_blue ) | fmt::emphasis::bold, "\n  🗓️  {}  🕒 {}\n", date, time );

    print_success( "Datetime functions test completed" );
  }
  catch ( const exception & e )
  {
    print_error( "Datetime functions test failed: " + string( e.what() ) );
  }
}

// Test cross-platform
void test_cross_platform()
{
  print_section( "CROSS-PLATFORM TESTS" );

  try
  {
#if defined( _WIN32 ) || defined( _WIN64 )
    print_info( "Platform: Windows" );
    print_result( "Windows Build", "✅ Detected" );
#elif defined( __linux__ )
    print_info( "Platform: Linux" );
    print_result( "Linux Build", "✅ Detected" );
#elif defined( __APPLE__ )
    print_info( "Platform: macOS" );
    print_result( "macOS Build", "✅ Detected" );
#else
    print_warning( "Platform: Unknown" );
#endif

    // Test dipendenze
    print_info( "System Information:" );

    // Dimensione dei tipi fondamentali
    print_result( "sizeof(int)", to_string( sizeof( int ) ) );
    print_result( "sizeof(long)", to_string( sizeof( long ) ) );
    print_result( "sizeof(void*)", to_string( sizeof( void * ) ) + " bytes" );

    // Thread support
    unsigned int threads = thread::hardware_concurrency();
    print_result( "CPU Threads", threads > 0 ? to_string( threads ) : "Unknown" );

    print_success( "Cross-platform tests completed" );
  }
  catch ( const exception & e )
  {
    print_error( "Cross-platform tests failed: " + string( e.what() ) );
  }
}

// Test di performance (base)
void test_performance()
{
  print_section( "PERFORMANCE TESTS" );

  try
  {
    auto start = chrono::high_resolution_clock::now();

    // Test rapido di alcune funzioni
    int iterations = 1000;

    // Test a simple function instead of has_SSE2()
    volatile string test = Architecture::get_architecture_string();
    (void) test;

    for ( int i = 0; i < iterations; ++i )
    {
      volatile auto arch = Architecture::get_architecture();
      (void) arch;
    }

    auto end      = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>( end - start );

    print_result(
      "Architecture Check",
      fmt::format(
        "{} iterations: {} µs ({:.2f} µs/iter)",
        iterations,
        duration.count(),
        double( duration.count() ) / iterations ) );

    // Test velocità funzioni di sistema
    start = chrono::high_resolution_clock::now();

    for ( int i = 0; i < 100; ++i )
    {
      volatile string host = get_host_name();
      (void) host;
    }

    end      = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>( end - start );

    print_result( "Hostname Lookup", fmt::format( "100 iterations: {} µs", duration.count() ) );

    print_success( "Performance tests completed" );
  }
  catch ( const exception & e )
  {
    print_error( "Performance tests failed: " + string( e.what() ) );
  }
}

void test_environment_edge_cases()
{
  print_section( "ENVIRONMENT EDGE CASES" );

  string value;

  // Variabile inesistente
  bool exists = get_environment( "THIS_VAR_DOES_NOT_EXIST", value );
  print_result( "Non-existing var", exists ? "Unexpected" : "Correct", !exists );

  // Set + overwrite
  set_environment( "UTILS_ENV_TEST", "first", true );
  get_environment( "UTILS_ENV_TEST", value );
  print_result( "Initial set", value, value == "first" );

  set_environment( "UTILS_ENV_TEST", "second", true );
  get_environment( "UTILS_ENV_TEST", value );
  print_result( "Overwrite", value, value == "second" );

  // Simula unset: set a empty
  set_environment( "UTILS_ENV_TEST", "", true );
  get_environment( "UTILS_ENV_TEST", value );
  print_result( "Clear value", value.empty() ? "Cleared" : "Not cleared", value.empty() );

  print_success( "Environment edge cases passed" );
}

void test_disk_space_edge_cases()
{
  print_section( "DISK SPACE EDGE CASES" );

  DiskSpaceInfo root     = get_disk_space( "/" );
  string        root_str = root.to_string();

  print_result( "Root disk info", root_str, !root_str.empty() );

  DiskSpaceInfo invalid     = get_disk_space( "/this/path/does/not/exist" );
  string        invalid_str = invalid.to_string();

  print_result( "Invalid path", invalid_str, invalid_str.find( "error" ) != string::npos || invalid_str.empty() );

  print_success( "Disk space edge cases completed" );
}


void test_uptime_and_load_sanity()
{
  print_section( "UPTIME & LOAD SANITY" );

  uint64_t u1 = get_system_uptime();
  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
  uint64_t u2 = get_system_uptime();

  print_result( "Uptime monotonic", u2 >= u1 ? "OK" : "ERROR", u2 >= u1 );

  SystemLoadInfo load     = get_system_load();
  string         load_str = load.to_string();

  print_result( "Load info", load_str, !load_str.empty() );

  print_success( "Uptime and load sanity passed" );
}


void test_memory_consistency()
{
  print_section( "MEMORY CONSISTENCY" );

  MemoryInfo mem     = get_memory_info();
  string     mem_str = mem.to_string();

  print_result( "Memory info", mem_str, !mem_str.empty() );

  print_success( "Memory consistency checks passed" );
}


void test_identity_edge_cases()
{
  print_section( "IDENTITY EDGE CASES" );

  string host = get_host_name();
  print_result( "Hostname length", to_string( host.size() ), host.size() < 256 );

  string user = get_user_name();
  print_result( "Username ASCII", user, user.find( ' ' ) == string::npos );

  string home = get_home_directory();
  print_result( "Home exists", home, check_if_dir_exists( home ) );

  print_success( "Identity edge cases passed" );
}

void test_executable_path()
{
  print_section( "EXECUTABLE PATH" );

  string exe = get_executable_path_name();
  print_result( "Executable path not empty", exe, !exe.empty() );
  print_result( "Executable exists", exe, check_if_file_exists( exe ) );

  print_success( "Executable path test passed" );
}

void test_time_formatting()
{
  print_section( "TIME FORMAT VALIDATION" );

  string date = get_date();      // YYYY-MM-DD
  string time = get_day_time();  // HH:MM:SS

  print_result( "Date format", date, date.size() == 10 );
  print_result( "Time format", time, time.size() == 8 );

  print_success( "Time formatting validated" );
}

int main()
{
  fmt::print(
    fg( fmt::color::light_sea_green ) | fmt::emphasis::bold,
    "\n╔══════════════════════════════════════════════════════════════╗\n" );
  fmt::print(
    fg( fmt::color::light_sea_green ) | fmt::emphasis::bold,
    "║               SYSTEM UTILS COMPREHENSIVE TEST               ║\n" );
  fmt::print(
    fg( fmt::color::light_sea_green ) | fmt::emphasis::bold,
    "╚══════════════════════════════════════════════════════════════╝\n\n" );

  int passed = 0;
  int total  = 0;

  auto run_test = [&]( const string & name, auto test_func )
  {
    total++;
    try
    {
      test_func();
      passed++;
      fmt::print( fg( fmt::color::lime_green ), "  [PASS] {}\n", name );
    }
    catch ( const exception & e )
    {
      fmt::print( fg( fmt::color::red ), "  [FAIL] {}: {}\n", name, e.what() );
    }
  };

  // Esecuzione dei test
  run_test( "Architecture Class", test_architecture_class );
  run_test( "System Functions", test_system_functions );
  run_test( "Network Functions", test_network_functions );
  run_test( "Filesystem Functions", test_filesystem_functions );
  run_test( "Datetime Functions", test_datetime_functions );
  run_test( "Cross-Platform", test_cross_platform );
  run_test( "Performance", test_performance );
  run_test( "Environment Edge Cases", test_environment_edge_cases );
  run_test( "Disk Space Edge Cases", test_disk_space_edge_cases );
  run_test( "Uptime & Load", test_uptime_and_load_sanity );
  run_test( "Memory Consistency", test_memory_consistency );
  run_test( "Identity Edge Cases", test_identity_edge_cases );
  run_test( "Executable Path", test_executable_path );
  run_test( "Time Formatting", test_time_formatting );

  // Riepilogo
  print_section( "TEST SUMMARY" );

  fmt::print( fg( fmt::color::white ), "  Tests Passed: " );
  if ( passed == total )
  {
    fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, "{} / {} ✅\n", passed, total );
  }
  else
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "{} / {} ❌\n", passed, total );
  }

  double percentage = ( total > 0 ) ? ( passed * 100.0 / total ) : 0;

  // Barra di progresso
  fmt::print( fg( fmt::color::white ), "  Progress: [" );

  int bar_width = 30;
  int filled    = static_cast<int>( percentage * bar_width / 100 );

  for ( int i = 0; i < filled; ++i ) fmt::print( fg( fmt::color::lime_green ), "█" );

  for ( int i = filled; i < bar_width; ++i ) fmt::print( fg( fmt::color::silver ), "░" );

  fmt::print( fg( fmt::color::white ), "] {:.1f}%\n", percentage );

  // Messaggio finale
  fmt::print( "\n" );
  if ( passed == total )
  {
    fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, "🎉 All tests passed successfully! 🎉\n" );
  }
  else
  {
    fmt::print( fg( fmt::color::orange ) | fmt::emphasis::bold, "⚠ Some tests failed. Check the output above.\n" );
  }

  fmt::print( fg( fmt::color::light_blue ), "\nTest completed at: {} {}\n", get_date(), get_day_time() );

  return ( passed == total ) ? 0 : 1;
}
