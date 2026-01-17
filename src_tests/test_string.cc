/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022-2026                                                 |
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

#include "Utils_string.hh"
#include <vector>
#include <array>

// Test runner class with colored output
class TestRunner
{
private:
  int                                              passed_tests = 0;
  int                                              failed_tests = 0;
  int                                              total_tests  = 0;
  std::vector<std::pair<std::string, std::string>> failures;  // test name, error message

public:
  void startSection( const std::string & section_name )
  {
    fmt::print( "\n" );
    fmt::print(
      fmt::emphasis::bold | fg( fmt::color::cyan ),
      "┌{0:─^{2}}┐\n"
      "│{1: ^{2}}│\n"
      "└{0:─^{2}}┘\n",
      "",
      section_name,
      70 );
  }

  template <typename Func> bool runTest( const std::string & test_name, Func test_func )
  {
    total_tests++;
    fmt::print( "  [{:03d}] {:<60}", total_tests, test_name );

    try
    {
      test_func();
      passed_tests++;
      fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "✓ PASS\n" );
      return true;
    }
    catch ( const std::exception & e )
    {
      failed_tests++;
      failures.emplace_back( test_name, e.what() );
      fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "✗ FAIL\n" );
      fmt::print( fg( fmt::color::yellow ), "    Error: {}\n", e.what() );
      return false;
    }
    catch ( ... )
    {
      failed_tests++;
      failures.emplace_back( test_name, "Unknown error" );
      fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "✗ FAIL\n" );
      fmt::print( fg( fmt::color::yellow ), "    Unknown error\n" );
      return false;
    }
  }

  void printDetailedResults()
  {
    if ( !failures.empty() )
    {
      fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n" );
      fmt::print(
        fg( fmt::color::red ) | fmt::emphasis::bold,
        "╔══════════════════════════════════════════════════════════════════════╗\n" );
      fmt::print(
        fg( fmt::color::red ) | fmt::emphasis::bold,
        "║                        FAILURE DETAILS                              ║\n" );
      fmt::print(
        fg( fmt::color::red ) | fmt::emphasis::bold,
        "╠══════════════════════════════════════════════════════════════════════╣\n" );

      for ( size_t i = 0; i < failures.size(); ++i )
      {
        fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "║ {:2d}. {:<65} ║\n", i + 1, failures[i].first );
        fmt::print( fg( fmt::color::yellow ), "║     {:<65} ║\n", failures[i].second );
        if ( i < failures.size() - 1 )
          fmt::print(
            fg( fmt::color::red ) | fmt::emphasis::bold,
            "║                                                                      ║\n" );
      }

      fmt::print(
        fg( fmt::color::red ) | fmt::emphasis::bold,
        "╚══════════════════════════════════════════════════════════════════════╝\n" );
    }
  }

  void printResults()
  {
    fmt::print(
      fg( fmt::color::cyan ) | fmt::emphasis::bold,
      "\n"
      "╔══════════════════════════════════════════════════════════════════════╗\n"
      "║                         TEST SUMMARY                                 ║\n"
      "╠══════════════════════════════════════════════════════════════════════╣\n" );

    fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║  " );
    fmt::print( fg( fmt::color::white ), "Total Tests: {:<58}", total_tests );
    fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║\n" );

    if ( passed_tests > 0 )
    {
      fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║  " );
      fmt::print( fg( fmt::color::green ), "Passed: {} ✓", passed_tests );
      fmt::print( "{:59}", "" );
      fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║\n" );
    }

    if ( failed_tests > 0 )
    {
      fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║  " );
      fmt::print( fg( fmt::color::red ), "Failed: {} ✗", failed_tests );
      fmt::print( "{:59}", "" );
      fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║\n" );
    }

    double percentage       = total_tests > 0 ? ( 100.0 * passed_tests / total_tests ) : 0.0;
    auto   percentage_color = ( percentage == 100.0 )  ? fg( fmt::color::green )
                              : ( percentage >= 90.0 ) ? fg( fmt::color::light_green )
                              : ( percentage >= 80.0 ) ? fg( fmt::color::yellow )
                                                       : fg( fmt::color::red );

    fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║  " );
    fmt::print( percentage_color | fmt::emphasis::bold, "Success Rate: {:6.2f}%", percentage );
    fmt::print( "{:51}", "" );
    fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "║\n" );

    fmt::print(
      fg( fmt::color::cyan ) | fmt::emphasis::bold,
      "╚══════════════════════════════════════════════════════════════════════╝\n" );

    printDetailedResults();
  }

  int getExitCode() const { return failed_tests > 0 ? 1 : 0; }
};

// Helper function to display UTF-8 characters with their codes
void print_utf8_info( const std::string & label, const std::string & chars )
{
  fmt::print( fg( fmt::color::light_gray ), "    {}: ", label );
  for ( unsigned char c : chars ) { fmt::print( fg( fmt::color::magenta ), "\\x{:02x}", c ); }
  fmt::print( fg( fmt::color::white ), " → '{}'", chars );
  fmt::print( "\n" );
}

// Helper function to display UTF-8 codepoint info
void print_codepoint_info( int cp, const std::string & description )
{
  fmt::print(
    fg( fmt::color::light_gray ),
    "    U+{:04X} {} → width={}\n",
    cp,
    description,
    Utils::utf8_character_width( cp ) );
}

// Test functions
void test_case_conversion( TestRunner & runner )
{
  runner.startSection( "STRING CASE CONVERSION TESTS" );

  runner.runTest(
    "to_upper converts to uppercase",
    []()
    {
      std::string str = "Hello World 123";
      Utils::to_upper( str );
      assert( str == "HELLO WORLD 123" );
      fmt::print( fg( fmt::color::light_gray ), "\n    'Hello World 123' → '{}' ", str );
    } );

  runner.runTest(
    "to_lower converts to lowercase",
    []()
    {
      std::string str = "HELLO WORLD 123";
      Utils::to_lower( str );
      assert( str == "hello world 123" );
      fmt::print( fg( fmt::color::light_gray ), "\n    'HELLO WORLD 123' → '{}' ", str );
    } );

  runner.runTest(
    "is_lower detects lowercase strings",
    []()
    {
      assert( Utils::is_lower( "hello" ) == true );
      assert( Utils::is_lower( "Hello" ) == false );
      assert( Utils::is_lower( "123" ) == false );  // digits are not lower case
      fmt::print(
        fg( fmt::color::light_gray ),
        "\n    Results: 'hello'={}, 'Hello'={}, '123'={} ",
        Utils::is_lower( "hello" ),
        Utils::is_lower( "Hello" ),
        Utils::is_lower( "123" ) );
    } );

  runner.runTest(
    "is_upper detects uppercase strings",
    []()
    {
      assert( Utils::is_upper( "HELLO" ) == true );
      assert( Utils::is_upper( "Hello" ) == false );
      assert( Utils::is_upper( "123" ) == false );
      fmt::print(
        fg( fmt::color::light_gray ),
        "\n    Results: 'HELLO'={}, 'Hello'={}, '123'={} ",
        Utils::is_upper( "HELLO" ),
        Utils::is_upper( "Hello" ),
        Utils::is_upper( "123" ) );
    } );
}

void test_character_classification( TestRunner & runner )
{
  runner.startSection( "CHARACTER CLASSIFICATION TESTS" );

  runner.runTest(
    "is_alpha detects alphabetic strings",
    []()
    {
      assert( Utils::is_alpha( "Hello" ) == true );
      assert( Utils::is_alpha( "Hello123" ) == false );
      assert( Utils::is_alpha( "Hello World" ) == false );
      fmt::print(
        fg( fmt::color::light_gray ),
        "\n    Results: 'Hello'={}, 'Hello123'={}, 'Hello World'={} ",
        Utils::is_alpha( "Hello" ),
        Utils::is_alpha( "Hello123" ),
        Utils::is_alpha( "Hello World" ) );
    } );

  runner.runTest(
    "is_alphanum detects alphanumeric strings",
    []()
    {
      assert( Utils::is_alphanum( "Hello123" ) == true );
      assert( Utils::is_alphanum( "Hello 123" ) == false );
      assert( Utils::is_alphanum( "Hello!" ) == false );
      fmt::print(
        fg( fmt::color::light_gray ),
        "\n    Results: 'Hello123'={}, 'Hello 123'={}, 'Hello!'={} ",
        Utils::is_alphanum( "Hello123" ),
        Utils::is_alphanum( "Hello 123" ),
        Utils::is_alphanum( "Hello!" ) );
    } );

  runner.runTest(
    "is_digits detects digit-only strings",
    []()
    {
      assert( Utils::is_digits( "12345" ) == true );
      assert( Utils::is_digits( "123.45" ) == false );
      assert( Utils::is_digits( "123 456" ) == false );
      fmt::print(
        fg( fmt::color::light_gray ),
        "\n    Results: '12345'={}, '123.45'={}, '123 456'={} ",
        Utils::is_digits( "12345" ),
        Utils::is_digits( "123.45" ),
        Utils::is_digits( "123 456" ) );
    } );

  runner.runTest(
    "is_xdigits detects hexadecimal strings",
    []()
    {
      assert( Utils::is_xdigits( "123ABCdef" ) == true );
      assert( Utils::is_xdigits( "123G" ) == false );
      assert( Utils::is_xdigits( "0x123" ) == false );
      fmt::print(
        fg( fmt::color::light_gray ),
        "\n    Results: '123ABCdef'={}, '123G'={}, '0x123'={} ",
        Utils::is_xdigits( "123ABCdef" ),
        Utils::is_xdigits( "123G" ),
        Utils::is_xdigits( "0x123" ) );
    } );
}

void test_utf8_core_functions( TestRunner & runner )
{
  runner.startSection( "UTF-8 CORE FUNCTION TESTS" );

  runner.runTest(
    "utf8_char_length correctly identifies UTF-8 byte lengths",
    []()
    {
      assert( Utils::utf8_char_length( 'A' ) == 1 );   // ASCII
      assert( Utils::utf8_char_length( 0xC3 ) == 2 );  // 2-byte UTF-8 start
      assert( Utils::utf8_char_length( 0xE2 ) == 3 );  // 3-byte UTF-8 start
      assert( Utils::utf8_char_length( 0xF0 ) == 4 );  // 4-byte UTF-8 start
      assert( Utils::utf8_char_length( 0x80 ) == 0 );  // Invalid: continuation byte
      assert( Utils::utf8_char_length( 0xFF ) == 0 );  // Invalid byte
      fmt::print( fg( fmt::color::light_gray ), "\n    All byte length tests passed " );
    } );

  runner.runTest(
    "utf8_is_continuation detects continuation bytes",
    []()
    {
      assert( Utils::utf8_is_continuation( 0x80 ) == true );   // 10xxxxxx
      assert( Utils::utf8_is_continuation( 0xBF ) == true );   // 10xxxxxx
      assert( Utils::utf8_is_continuation( 0x7F ) == false );  // 01111111
      assert( Utils::utf8_is_continuation( 0xC0 ) == false );  // 11000000
      fmt::print( fg( fmt::color::light_gray ), "\n    Continuation byte detection correct " );
    } );

  runner.runTest(
    "utf8_next correctly decodes UTF-8 sequences",
    []()
    {
      std::string test = "Aα€𝄞";  // A (U+0041), α (U+03B1), € (U+20AC), 𝄞 (U+1D11E)
      int         pos  = 0;

      int cp1 = Utils::utf8_next( test, pos );
      assert( cp1 == 0x41 );  // 'A'
      assert( pos == 1 );

      int cp2 = Utils::utf8_next( test, pos );
      assert( cp2 == 0x3B1 );  // 'α'
      assert( pos == 3 );      // 2-byte character

      int cp3 = Utils::utf8_next( test, pos );
      assert( cp3 == 0x20AC );  // '€'
      assert( pos == 6 );       // 3-byte character

      int cp4 = Utils::utf8_next( test, pos );
      assert( cp4 == 0x1D11E );  // '𝄞'
      assert( pos == 10 );       // 4-byte character

      fmt::print(
        fg( fmt::color::light_gray ),
        "\n    Decoded: U+{:04X}, U+{:04X}, U+{:04X}, U+{:04X} ",
        cp1,
        cp2,
        cp3,
        cp4 );
    } );

  runner.runTest(
    "utf8_next handles invalid UTF-8 gracefully",
    []()
    {
      std::string invalid = "\xC3";  // Start byte without continuation
      int         pos     = 0;

      int cp = Utils::utf8_next( invalid, pos );
      assert( cp == -1 );  // Should return -1 for error
      assert( pos == 1 );  // Should skip invalid byte

      fmt::print( fg( fmt::color::light_gray ), "\n    Invalid UTF-8 handled correctly " );
    } );
}

void test_unicode_width_calculation( TestRunner & runner )
{
  runner.startSection( "UNICODE WIDTH CALCULATION TESTS" );

  runner.runTest(
    "utf8_character_width for ASCII characters",
    []()
    {
      assert( Utils::utf8_character_width( 'A' ) == 1 );
      assert( Utils::utf8_character_width( ' ' ) == 1 );
      assert( Utils::utf8_character_width( '\n' ) == 0 );  // Control character
      assert( Utils::utf8_character_width( 0x7F ) == 0 );  // DEL character
      fmt::print( fg( fmt::color::light_gray ), "\n    ASCII width calculation correct " );
    } );

  runner.runTest(
    "utf8_character_width for specific symbols and emoji",
    []()
    {
      fmt::print( fg( fmt::color::light_gray ), "\n    Testing specific Unicode characters:\n" );

      // Test specific characters mentioned in the table
      print_codepoint_info( 0x2705, "✅ White Heavy Check Mark" );
      assert( Utils::utf8_character_width( 0x2705 ) == 2 );

      print_codepoint_info( 0x274C, "❌ Cross Mark" );
      assert( Utils::utf8_character_width( 0x274C ) == 2 );  // Should be 2

      print_codepoint_info( 0x2795, "➕ Heavy Plus Sign" );
      assert( Utils::utf8_character_width( 0x2795 ) == 2 );

      print_codepoint_info( 0x2796, "➖ Heavy Minus Sign" );
      assert( Utils::utf8_character_width( 0x2796 ) == 2 );

      print_codepoint_info( 0x26A1, "⚡ High Voltage Sign" );
      assert( Utils::utf8_character_width( 0x26A1 ) == 2 );

      print_codepoint_info( 0x2B50, "⭐ White Medium Star" );
      assert( Utils::utf8_character_width( 0x2B50 ) == 2 );

      print_codepoint_info( 0x2B55, "⭕ Heavy Large Circle" );
      assert( Utils::utf8_character_width( 0x2B55 ) == 2 );

      print_codepoint_info( 0x21A9, "↩ Leftwards Arrow with Hook" );
      assert( Utils::utf8_character_width( 0x21A9 ) == 1 );

      print_codepoint_info( 0x21AA, "↪ Rightwards Arrow with Hook" );
      assert( Utils::utf8_character_width( 0x21AA ) == 1 );
    } );

  runner.runTest(
    "utf8_character_width for CJK characters",
    []()
    {
      assert( Utils::utf8_character_width( 0x4E00 ) == 2 );  // 一 (CJK Unified Ideograph)
      assert( Utils::utf8_character_width( 0x9FFF ) == 2 );  // Another CJK character
      assert( Utils::utf8_character_width( 0x3041 ) == 2 );  // ぁ (Hiragana)
      assert( Utils::utf8_character_width( 0x30A1 ) == 2 );  // ァ (Katakana)
      fmt::print( fg( fmt::color::light_gray ), "\n    CJK width calculation correct " );
    } );

  runner.runTest(
    "utf8_character_width for modern emoji",
    []()
    {
      assert( Utils::utf8_character_width( 0x1F600 ) == 2 );  // 😀 Grinning Face
      assert( Utils::utf8_character_width( 0x1F680 ) == 2 );  // 🚀 Rocket
      assert( Utils::utf8_character_width( 0x1F44D ) == 2 );  // 👍 Thumbs Up
      fmt::print( fg( fmt::color::light_gray ), "\n    Emoji width calculation correct " );
    } );

  runner.runTest(
    "utf8_display_width for complete strings",
    []()
    {
      struct TestCase
      {
        std::string str;
        int         expected;
      };
      std::vector<TestCase> tests = {
        { "Hello", 5 },
        { "Hello 世界", 5 + 1 + 2 * 2 },      // 5 ASCII + 1 space + 2 CJK chars * 2 width = 10
        { "✅➕➖⚡", 2 * 4 },                // 4 symbols * 2 width each = 8
        { "Test 😀 🚀", 4 + 1 + 2 + 1 + 2 },  // "Test"=4, space=1, 😀=2, space=1, 🚀=2 = 10
        { "↩️ ", 3 },                          // ↩ + variation selector (treated as 2)
        { "↪️ ", 3 },                          // ↪ + variation selector (treated as 2)
        { "A\nB", 2 },                        // Newline has width 0
      };

      for ( const auto & test : tests )
      {
        int width = Utils::utf8_display_width( test.str );
        fmt::print(
          fg( fmt::color::light_gray ),
          "\n    '{}' → width={} (expected={}) {} ",
          test.str,
          width,
          test.expected,
          width == test.expected ? "✓" : "✗" );
        assert( width == test.expected );
      }
    } );
}

void test_utf8_string_operations( TestRunner & runner )
{
  runner.startSection( "UTF-8 STRING OPERATION TESTS" );

  runner.runTest(
    "utf8_truncate truncates strings correctly",
    []()
    {
      struct TestCase
      {
        std::string input;
        int         width;
        std::string expected;
      };
      std::vector<TestCase> tests = {
        { "Hello World", 5, "Hello" },
        { "Hello 世界", 8, "Hello 世" },  // 5 + 1 + 2 = 8 > 7, so only "Hello 世" fits
        { "✅➕➖⚡", 4, "✅➕" },        // Each is width 2, so 2+2=4
        { "Test 😀  ", 7, "Test 😀" },    // Exact fit: 4 + 2 = 6
        { "Test 😀  ", 5, "Test " }       // Only space fits after "Test"
      };

      for ( const auto & test : tests )
      {
        std::string result = Utils::utf8_truncate( test.input, test.width );
        fmt::print(
          fg( fmt::color::light_gray ),
          "\n    truncate('{}', {}) → '{}' (expected='{}') {} ",
          test.input,
          test.width,
          result,
          test.expected,
          result == test.expected ? "✓" : "✗" );
        assert( result == test.expected );
      }
    } );

  runner.runTest(
    "utf8_truncate with ellipsis",
    []()
    {
      std::string result = Utils::utf8_truncate( "Hello World", 8, "..." );
      assert( result == "Hello..." );
      fmt::print( fg( fmt::color::light_gray ), "\n    truncate('Hello World', 8, '...') → '{}' ", result );

      result = Utils::utf8_truncate( "✅➕➖⚡", 5, "…" );
      assert( result == "✅➕…" );  // ✅(2) + ➕(2) + …(1) = 5
      fmt::print( fg( fmt::color::light_gray ), "\n    truncate('✅➕➖⚡', 5, '…') → '{}' ", result );
    } );

  runner.runTest(
    "utf8_padding pads strings correctly",
    []()
    {
      std::string result = Utils::utf8_padding( "Hello", 10 );
      assert( result == "Hello     " );  // 5 spaces added
      fmt::print( fg( fmt::color::light_gray ), "\n    padding('Hello', 10) → '{}' ", result );

      result = Utils::utf8_padding( "✅", 5 );
      // ✅ is width 2, needs 3 more width, spaces are width 1 each
      assert( result == "✅   " );
      fmt::print( fg( fmt::color::light_gray ), "\n    padding('✅', 5) → '{}' ", result );

      result = Utils::utf8_padding( "Test", 2 );  // Already wider
      assert( result == "Test" );
      fmt::print( fg( fmt::color::light_gray ), "\n    padding('Test', 2) → '{}' (no change) ", result );

      result = Utils::utf8_padding( "Hi", 5, "." );
      assert( result == "Hi..." );  // 2 + 3 = 5
      fmt::print( fg( fmt::color::light_gray ), "\n    padding('Hi', 5, '.') → '{}' ", result );
    } );

  runner.runTest(
    "repeat function works correctly",
    []()
    {
      assert( Utils::repeat( "abc", 3 ) == "abcabcabc" );
      fmt::print( fg( fmt::color::light_gray ), "\n    repeat('abc', 3) = '{}' ", Utils::repeat( "abc", 3 ) );

      assert( Utils::repeat( "", 5 ) == "" );
      fmt::print( fg( fmt::color::light_gray ), "\n    repeat('', 5) = '{}' ", Utils::repeat( "", 5 ) );

      assert( Utils::repeat( "x", 0 ) == "" );
      fmt::print( fg( fmt::color::light_gray ), "\n    repeat('x', 0) = '{}' ", Utils::repeat( "x", 0 ) );

      assert( Utils::repeat( "✅", 3 ) == "✅✅✅" );
      fmt::print( fg( fmt::color::light_gray ), "\n    repeat('✅', 3) = '{}' ", Utils::repeat( "✅", 3 ) );
    } );
}

void test_border_characters( TestRunner & runner )
{
  runner.startSection( "BORDER CHARACTER TESTS" );

  runner.runTest(
    "All border constants are defined and valid UTF-8",
    []()
    {
      // Test that all border constants are non-empty and have valid width
      std::vector<std::pair<std::string, const char *>> borders = {
        { "border_top", Utils::border_top },
        { "border_top_left", Utils::border_top_left },
        { "border_top_right", Utils::border_top_right },
        { "border_bottom", Utils::border_bottom },
        { "border_bottom_left", Utils::border_bottom_left },
        { "border_bottom_right", Utils::border_bottom_right },
        { "border_left", Utils::border_left },
        { "border_right", Utils::border_right },
        { "border_middle", Utils::border_middle },
      };

      for ( const auto & [name, border] : borders )
      {
        assert( border != nullptr && strlen( border ) > 0 );
        int width = Utils::utf8_display_width( border );
        assert( width == 1 );  // All border chars should be single width
        fmt::print( fg( fmt::color::light_gray ), "\n    {} = '{}' (width={}) ", name, border, width );
      }
    } );

  runner.runTest(
    "Draw boxes with different border styles",
    []()
    {
      auto draw_box = [](
                        const char *        top_left,
                        const char *        top_right,
                        const char *        bottom_left,
                        const char *        bottom_right,
                        const char *        horizontal,
                        const char *        vertical,
                        const std::string & title )
      {
        std::string top_line = fmt::format( "{}{}{}", top_left, Utils::repeat( horizontal, 20 ), top_right );

        std::string middle_line = fmt::format( "{}{: ^20}{}", vertical, title, vertical );

        std::string bottom_line = fmt::format( "{}{}{}", bottom_left, Utils::repeat( horizontal, 20 ), bottom_right );

        return fmt::format( "{}\n{}\n{}", top_line, middle_line, bottom_line );
      };

      fmt::print( fg( fmt::color::cyan ), "\n    Single Line Box:\n" );
      fmt::print(
        "{}\n",
        draw_box(
          Utils::border_top_left,
          Utils::border_top_right,
          Utils::border_bottom_left,
          Utils::border_bottom_right,
          Utils::border_top,
          Utils::border_left,
          "Hello" ) );

      fmt::print( fg( fmt::color::cyan ), "\n    Bold Line Box:\n" );
      fmt::print(
        "{}\n",
        draw_box(
          Utils::border_top_left_bold,
          Utils::border_top_right_bold,
          Utils::border_bottom_left_bold,
          Utils::border_bottom_right_bold,
          Utils::border_top_bold,
          Utils::border_left_bold,
          "Bold" ) );

      fmt::print( fg( fmt::color::cyan ), "\n    Double Line Box:\n" );
      fmt::print(
        "{}\n",
        draw_box(
          Utils::border_top_left_double,
          Utils::border_top_right_double,
          Utils::border_bottom_left_double,
          Utils::border_bottom_right_double,
          Utils::border_top_double,
          Utils::border_left_double,
          "Double" ) );
    } );
}

void print_banner()
{
  fmt::print(
    fg( fmt::color::magenta ) | fmt::emphasis::bold,
    "\n"
    "╔════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                       UTF-8 STRING UTILITIES TEST SUITE                        ║\n"
    "╠════════════════════════════════════════════════════════════════════════════════╣\n"
    "║                                                                                ║\n"
    "║  ██╗   ██╗████████╗███████╗    ███████╗████████╗██████╗ ██╗███╗   ██╗ ██████╗  ║\n"
    "║  ██║   ██║╚══██╔══╝██╔════╝    ██╔════╝╚══██╔══╝██╔══██╗██║████╗  ██║██╔════╝  ║\n"
    "║  ██║   ██║   ██║   █████╗      ███████╗   ██║   ██████╔╝██║██╔██╗ ██║██║  ███╗ ║\n"
    "║  ██║   ██║   ██║   ██╔══╝      ╚════██║   ██║   ██╔══██╗██║██║╚██╗██║██║   ██║ ║\n"
    "║  ╚██████╔╝   ██║   ███████╗    ███████║   ██║   ██║  ██║██║██║ ╚████║╚██████╔╝ ║\n"
    "║   ╚═════╝    ╚═╝   ╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝  ║\n"
    "║                                                                                ║\n"
    "║           Comprehensive test suite for Utils_string.hh                         ║\n"
    "║           Testing UTF-8 handling, width calculation,                           ║\n"
    "║           border characters, and string utilities                              ║\n"
    "║                                                                                ║\n"
    "╚════════════════════════════════════════════════════════════════════════════════╝\n\n" );

  fmt::print( fg( fmt::color::light_gray ), "Starting tests...\n\n" );
}

int main()
{
  print_banner();

  TestRunner runner;

  try
  {
    // Run all test categories
    test_case_conversion( runner );
    test_character_classification( runner );
    test_utf8_core_functions( runner );
    test_unicode_width_calculation( runner );
    test_utf8_string_operations( runner );
    test_border_characters( runner );

    // Print results
    runner.printResults();

    return runner.getExitCode();
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n❌ FATAL ERROR in test suite: {}\n", e.what() );
    return 1;
  }
  catch ( ... )
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n❌ UNKNOWN FATAL ERROR in test suite\n" );
    return 1;
  }
}
