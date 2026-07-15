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
 |      Universita degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "Utils.hh"
#include "Utils_Console.hh"

using namespace std;

namespace
{

  void clear_stream( std::ostringstream & s )
  {
    s.str( "" );
    s.clear();
  }

  bool contains( std::string const & str, std::string_view expected )
  { return str.find( std::string( expected ) ) != std::string::npos; }

  std::string strip_ansi( std::string_view text )
  {
    std::string plain;
    plain.reserve( text.size() );

    for ( size_t i = 0; i < text.size(); )
    {
      if ( text[i] == '\x1b' && i + 1 < text.size() && text[i + 1] == '[' )
      {
        i += 2;
        while ( i < text.size() )
        {
          unsigned char const c = static_cast<unsigned char>( text[i++] );
          if ( c >= 0x40 && c <= 0x7e ) { break; }
        }
      }
      else
      {
        plain.push_back( text[i++] );
      }
    }
    return plain;
  }

  template <typename Fun> void expect_contains(
    Utils::Console const & C,
    std::ostringstream &   out,
    std::string_view       label,
    std::string_view       expected,
    Fun &&                 fun )
  {
    clear_stream( out );
    std::forward<Fun>( fun )();
    C.flush();

    std::string const got       = out.str();
    std::string const got_plain = strip_ansi( got );
    if ( !contains( got_plain, expected ) )
    {
      std::cerr << "FAILED: " << label << "\n"
                << "expected substring: [" << expected << "]\n"
                << "actual output:       [" << got << "]\n";
      throw std::runtime_error( std::string( "Console test failed: " ) + std::string( label ) );
    }
  }

  template <typename Fun>
  void expect_empty( Utils::Console const & C, std::ostringstream & out, std::string_view label, Fun && fun )
  {
    clear_stream( out );
    std::forward<Fun>( fun )();
    C.flush();

    std::string const got = out.str();
    if ( !got.empty() )
    {
      std::cerr << "FAILED: " << label << "\n"
                << "expected empty output\n"
                << "actual output: [" << got << "]\n";
      throw std::runtime_error( std::string( "Console test failed: " ) + std::string( label ) );
    }
  }

  template <typename Exception, typename Fun> void expect_throws( std::string_view label, Fun && fun )
  {
    try
    {
      std::forward<Fun>( fun )();
    }
    catch ( Exception const & )
    {
      return;
    }
    catch ( ... )
    {
      std::cerr << "FAILED: " << label << "\nexpected a different exception type\n";
      throw std::runtime_error( std::string( "Console test failed: " ) + std::string( label ) );
    }

    std::cerr << "FAILED: " << label << "\nexpected an exception\n";
    throw std::runtime_error( std::string( "Console test failed: " ) + std::string( label ) );
  }

  template <typename Fun> void expect_reset_before_each_newline(
    Utils::Console const & C,
    std::ostringstream &   out,
    std::string_view       label,
    Fun &&                 fun )
  {
    clear_stream( out );
    std::forward<Fun>( fun )();
    C.flush();

    constexpr std::string_view reset{ "\x1b[0m" };
    std::string const          got           = out.str();
    size_t                     search_from   = 0;
    unsigned                   newline_count = 0;

    while ( true )
    {
      size_t const newline = got.find( '\n', search_from );
      if ( newline == std::string::npos ) { break; }

      bool const reset_precedes_newline = newline >= reset.size() &&
                                          got.compare( newline - reset.size(), reset.size(), reset ) == 0;
      if ( !reset_precedes_newline )
      {
        std::cerr << "FAILED: " << label << "\nANSI reset does not precede newline\n";
        throw std::runtime_error( std::string( "Console test failed: " ) + std::string( label ) );
      }

      ++newline_count;
      search_from = newline + 1;
    }

    if ( newline_count == 0 )
    {
      std::cerr << "FAILED: " << label << "\ntest output contains no newline\n";
      throw std::runtime_error( std::string( "Console test failed: " ) + std::string( label ) );
    }
  }

  void test_stream_and_level_methods()
  {
    std::ostringstream out1;
    std::ostringstream out2;
    Utils::Console     C( &out1, 4 );

    expect_throws<std::invalid_argument>( "null stream in constructor", []() { Utils::Console invalid( nullptr ); } );
    expect_throws<std::out_of_range>( "constructor level below range", [&]() { Utils::Console invalid( &out1, -2 ); } );
    expect_throws<std::out_of_range>( "constructor level above range", [&]() { Utils::Console invalid( &out1, 5 ); } );

    assert( C.get_level() == 4 );
    assert( C.getLevel() == 4 );
    assert( C.get_stream() == &out1 );
    assert( C.getStream() == &out1 );

    C.change_level( 2 );
    assert( C.get_level() == 2 );

    expect_throws<std::out_of_range>( "change_level below range", [&]() { C.change_level( -2 ); } );
    expect_throws<std::out_of_range>( "change_level above range", [&]() { C.change_level( 5 ); } );
    assert( C.get_level() == 2 );

    C.changeLevel( 4 );
    assert( C.getLevel() == 4 );

    C.change_stream( &out2 );
    assert( C.get_stream() == &out2 );

    expect_throws<std::invalid_argument>( "null change_stream", [&]() { C.change_stream( nullptr ); } );
    assert( C.get_stream() == &out2 );

    expect_contains( C, out2, "change_stream", "stream 2\n", [&]() { C.message( "stream 2\n", 0 ); } );
    assert( out1.str().empty() );

    C.changeStream( &out1 );
    assert( C.getStream() == &out1 );

    expect_contains( C, out1, "changeStream", "stream 1\n", [&]() { C.message( "stream 1\n", 0 ); } );
  }

  void test_style_methods()
  {
    std::ostringstream            out;
    Utils::Console                C( &out, 4 );
    Utils::Console::Console_style legacy_style;
    legacy_style.ts = fmt::fg( fmt::color::green );
    (void) legacy_style;

    assert( C.color_enabled() );
    C.set_color_enabled( false );
    assert( !C.color_enabled() );
    C.set_color_enabled( true );
    assert( C.color_enabled() );

    C.set_message_style( fmt::emphasis::italic, fmt::color::green, fmt::color::black );
    expect_contains( C, out, "set_message_style", "message style\n", [&]() { C.message( "message style\n", 0 ); } );
    expect_reset_before_each_newline(
      C,
      out,
      "background reset for multiline message",
      [&]() { C.message( "first background line\nsecond background line\n", 0 ); } );
    expect_reset_before_each_newline(
      C,
      out,
      "background reset for formatted message",
      [&]() { C.message( 0, "formatted background line {}\n", 3 ); } );

    C.setMessageStyle( fmt::emphasis::underline, fmt::color::cyan, fmt::color::black );
    expect_contains(
      C,
      out,
      "setMessageStyle",
      "message style deprecated\n",
      [&]() { C.message( "message style deprecated\n", 0 ); } );

    C.set_warning_style( fmt::emphasis::italic, fmt::color::yellow, fmt::color::black );
    expect_contains( C, out, "set_warning_style", "warning style\n", [&]() { C.warning( "warning style\n" ); } );

    C.setWarningStyle( fmt::emphasis::underline, fmt::color::yellow, fmt::color::black );
    expect_contains(
      C,
      out,
      "setWarningStyle",
      "warning style deprecated\n",
      [&]() { C.warning( "warning style deprecated\n" ); } );

    C.set_error_style( fmt::emphasis::italic, fmt::color::red, fmt::color::black );
    expect_contains( C, out, "set_error_style", "error style\n", [&]() { C.error( "error style\n" ); } );

    C.setErrorStyle( fmt::emphasis::underline, fmt::color::red, fmt::color::black );
    expect_contains(
      C,
      out,
      "setErrorStyle",
      "error style deprecated\n",
      [&]() { C.error( "error style deprecated\n" ); } );

    C.set_fatal_style( fmt::emphasis::italic, fmt::color::red, fmt::color::black );
    expect_contains( C, out, "set_fatal_style", "fatal style\n", [&]() { C.fatal( "fatal style\n" ); } );

    C.setFatalStyle( fmt::emphasis::underline, fmt::color::red, fmt::color::black );
    expect_contains(
      C,
      out,
      "setFatalStyle",
      "fatal style deprecated\n",
      [&]() { C.fatal( "fatal style deprecated\n" ); } );

    C.set_off();
    assert( !C.color_enabled() );
    clear_stream( out );
    C.message( "plain after set_off\n", 0 );
    assert( out.str() == "plain after set_off\n" );

    C.set_auto();
    assert( C.color_enabled() );
    clear_stream( out );
    C.message( "styled after set_auto\n", 0 );
    assert( contains( out.str(), "\x1b[" ) );

    C.setOff();
    assert( !C.color_enabled() );
    clear_stream( out );
    C.warning( "plain after setOff\n" );
    assert( out.str() == "plain after setOff\n" );

    C.setAuto();
    assert( C.color_enabled() );
    clear_stream( out );
    C.warning( "styled after setAuto\n" );
    assert( contains( out.str(), "\x1b[" ) );
  }

  void test_legacy_message_methods()
  {
    std::ostringstream out;
    Utils::Console     C( &out, 4 );

    expect_contains( C, out, "message old", "message legacy\n", [&]() { C.message( "message legacy\n" ); } );
    expect_contains(
      C,
      out,
      "message old level",
      "message legacy level\n",
      [&]() { C.message( "message legacy level\n", 2 ); } );

    expect_contains(
      C,
      out,
      "semaphore old 0",
      "semaphore legacy 0\n",
      [&]() { C.semaphore( 0, "semaphore legacy 0\n" ); } );
    expect_contains(
      C,
      out,
      "semaphore old 1",
      "semaphore legacy 1\n",
      [&]() { C.semaphore( 1, "semaphore legacy 1\n" ); } );
    expect_contains(
      C,
      out,
      "semaphore old 2",
      "semaphore legacy 2\n",
      [&]() { C.semaphore( 2, "semaphore legacy 2\n" ); } );
    expect_contains(
      C,
      out,
      "semaphore old level",
      "semaphore legacy level\n",
      [&]() { C.semaphore( 2, "semaphore legacy level\n", 2 ); } );

    for ( unsigned c = 0; c < 5; ++c )
    {
      std::string const msg = std::string( "colors legacy " ) + std::to_string( c ) + "\n";
      expect_contains( C, out, "colors old", msg, [&]() { C.colors( c, msg ); } );
    }
    expect_contains(
      C,
      out,
      "colors old level",
      "colors legacy level\n",
      [&]() { C.colors( 3, "colors legacy level\n", 2 ); } );

    // Indices intentionally wrap: semaphore modulo 3, colors modulo 5.
    expect_contains(
      C,
      out,
      "semaphore wrap",
      "semaphore wrapped 3 -> 0\n",
      [&]() { C.semaphore( 3, "semaphore wrapped 3 -> 0\n" ); } );
    expect_contains(
      C,
      out,
      "colors wrap",
      "colors wrapped 5 -> 0\n",
      [&]() { C.colors( 5, "colors wrapped 5 -> 0\n" ); } );

    expect_contains( C, out, "warning old", "warning legacy\n", [&]() { C.warning( "warning legacy\n" ); } );
    expect_contains( C, out, "error old", "error legacy\n", [&]() { C.error( "error legacy\n" ); } );
    expect_contains( C, out, "fatal old", "fatal legacy\n", [&]() { C.fatal( "fatal legacy\n" ); } );

    expect_contains( C, out, "black old", "black legacy\n", [&]() { C.black( "black legacy\n" ); } );
    expect_contains( C, out, "red old", "red legacy\n", [&]() { C.red( "red legacy\n" ); } );
    expect_contains( C, out, "green old", "green legacy\n", [&]() { C.green( "green legacy\n" ); } );
    expect_contains( C, out, "yellow old", "yellow legacy\n", [&]() { C.yellow( "yellow legacy\n" ); } );
    expect_contains( C, out, "blue old", "blue legacy\n", [&]() { C.blue( "blue legacy\n" ); } );
    expect_contains( C, out, "magenta old", "magenta legacy\n", [&]() { C.magenta( "magenta legacy\n" ); } );
    expect_contains( C, out, "cyan old", "cyan legacy\n", [&]() { C.cyan( "cyan legacy\n" ); } );
    expect_contains( C, out, "gray old", "gray legacy\n", [&]() { C.gray( "gray legacy\n" ); } );

    expect_contains(
      C,
      out,
      "black_reversed old",
      "black_reversed legacy\n",
      [&]() { C.black_reversed( "black_reversed legacy\n" ); } );
    expect_contains(
      C,
      out,
      "red_reversed old",
      "red_reversed legacy\n",
      [&]() { C.red_reversed( "red_reversed legacy\n" ); } );
    expect_contains(
      C,
      out,
      "green_reversed old",
      "green_reversed legacy\n",
      [&]() { C.green_reversed( "green_reversed legacy\n" ); } );
    expect_contains(
      C,
      out,
      "yellow_reversed old",
      "yellow_reversed legacy\n",
      [&]() { C.yellow_reversed( "yellow_reversed legacy\n" ); } );
    expect_contains(
      C,
      out,
      "blue_reversed old",
      "blue_reversed legacy\n",
      [&]() { C.blue_reversed( "blue_reversed legacy\n" ); } );
    expect_contains(
      C,
      out,
      "magenta_reversed old",
      "magenta_reversed legacy\n",
      [&]() { C.magenta_reversed( "magenta_reversed legacy\n" ); } );
    expect_contains(
      C,
      out,
      "cyan_reversed old",
      "cyan_reversed legacy\n",
      [&]() { C.cyan_reversed( "cyan_reversed legacy\n" ); } );
    expect_contains(
      C,
      out,
      "gray_reversed old",
      "gray_reversed legacy\n",
      [&]() { C.gray_reversed( "gray_reversed legacy\n" ); } );
  }

  void test_formatted_message_methods()
  {
    std::ostringstream out;
    Utils::Console     C( &out, 4 );

    expect_contains(
      C,
      out,
      "message formatted",
      "message formatted: alpha 7 3.14\n",
      [&]() { C.message( 3, "message formatted: {} {} {:.2f}\n", "alpha", 7, 3.14159 ); } );

    expect_contains(
      C,
      out,
      "semaphore formatted 0",
      "semaphore formatted 0: red 001\n",
      [&]() { C.semaphore( 0, 0, "semaphore formatted {}: {} {:0>3}\n", 0, "red", 1 ); } );
    expect_contains(
      C,
      out,
      "semaphore formatted 1",
      "semaphore formatted 1: yellow 002\n",
      [&]() { C.semaphore( 0, 1, "semaphore formatted {}: {} {:0>3}\n", 1, "yellow", 2 ); } );
    expect_contains(
      C,
      out,
      "semaphore formatted 2",
      "semaphore formatted 2: green 003\n",
      [&]() { C.semaphore( 0, 2, "semaphore formatted {}: {} {:0>3}\n", 2, "green", 3 ); } );

    for ( unsigned c = 0; c < 5; ++c )
    {
      std::string const expected = std::string( "colors formatted " ) + std::to_string( c ) + ": value " +
                                   std::to_string( 10 + c ) + "\n";
      expect_contains(
        C,
        out,
        "colors formatted",
        expected,
        [&]() { C.colors( 0, c, "colors formatted {}: {} {}\n", c, "value", 10 + c ); } );
    }

    expect_contains(
      C,
      out,
      "warning formatted",
      "warning formatted: tol=1.23e-04 iter=5\n",
      [&]() { C.warning( "warning formatted: tol={:.2e} iter={}\n", 1.234e-4, 5 ); } );
    expect_contains(
      C,
      out,
      "error formatted",
      "error formatted: file.dat line=77\n",
      [&]() { C.error( "error formatted: {} line={}\n", "file.dat", 77 ); } );
    expect_contains(
      C,
      out,
      "fatal formatted",
      "fatal formatted: code=-9 reason=stop\n",
      [&]() { C.fatal( "fatal formatted: code={} reason={}\n", -9, "stop" ); } );

    expect_contains(
      C,
      out,
      "black formatted",
      "black formatted: A 1\n",
      [&]() { C.black( 0, "black formatted: {} {}\n", 'A', 1 ); } );
    expect_contains(
      C,
      out,
      "red formatted",
      "red formatted: B 2\n",
      [&]() { C.red( 0, "red formatted: {} {}\n", 'B', 2 ); } );
    expect_contains(
      C,
      out,
      "green formatted",
      "green formatted: C 3\n",
      [&]() { C.green( 0, "green formatted: {} {}\n", 'C', 3 ); } );
    expect_contains(
      C,
      out,
      "yellow formatted",
      "yellow formatted: D 4\n",
      [&]() { C.yellow( 0, "yellow formatted: {} {}\n", 'D', 4 ); } );
    expect_contains(
      C,
      out,
      "blue formatted",
      "blue formatted: E 5\n",
      [&]() { C.blue( 0, "blue formatted: {} {}\n", 'E', 5 ); } );
    expect_contains(
      C,
      out,
      "magenta formatted",
      "magenta formatted: F 6\n",
      [&]() { C.magenta( 0, "magenta formatted: {} {}\n", 'F', 6 ); } );
    expect_contains(
      C,
      out,
      "cyan formatted",
      "cyan formatted: G 7\n",
      [&]() { C.cyan( 0, "cyan formatted: {} {}\n", 'G', 7 ); } );
    expect_contains(
      C,
      out,
      "gray formatted",
      "gray formatted: H 8\n",
      [&]() { C.gray( 0, "gray formatted: {} {}\n", 'H', 8 ); } );

    expect_contains(
      C,
      out,
      "black_reversed formatted",
      "black_reversed formatted: A 11\n",
      [&]() { C.black_reversed( 0, "black_reversed formatted: {} {}\n", 'A', 11 ); } );
    expect_contains(
      C,
      out,
      "red_reversed formatted",
      "red_reversed formatted: B 12\n",
      [&]() { C.red_reversed( 0, "red_reversed formatted: {} {}\n", 'B', 12 ); } );
    expect_contains(
      C,
      out,
      "green_reversed formatted",
      "green_reversed formatted: C 13\n",
      [&]() { C.green_reversed( 0, "green_reversed formatted: {} {}\n", 'C', 13 ); } );
    expect_contains(
      C,
      out,
      "yellow_reversed formatted",
      "yellow_reversed formatted: D 14\n",
      [&]() { C.yellow_reversed( 0, "yellow_reversed formatted: {} {}\n", 'D', 14 ); } );
    expect_contains(
      C,
      out,
      "blue_reversed formatted",
      "blue_reversed formatted: E 15\n",
      [&]() { C.blue_reversed( 0, "blue_reversed formatted: {} {}\n", 'E', 15 ); } );
    expect_contains(
      C,
      out,
      "magenta_reversed formatted",
      "magenta_reversed formatted: F 16\n",
      [&]() { C.magenta_reversed( 0, "magenta_reversed formatted: {} {}\n", 'F', 16 ); } );
    expect_contains(
      C,
      out,
      "cyan_reversed formatted",
      "cyan_reversed formatted: G 17\n",
      [&]() { C.cyan_reversed( 0, "cyan_reversed formatted: {} {}\n", 'G', 17 ); } );
    expect_contains(
      C,
      out,
      "gray_reversed formatted",
      "gray_reversed formatted: H 18\n",
      [&]() { C.gray_reversed( 0, "gray_reversed formatted: {} {}\n", 'H', 18 ); } );
  }

  void test_level_filtering()
  {
    std::ostringstream out;
    Utils::Console     C( &out, 2 );

    expect_contains(
      C,
      out,
      "message visible at level",
      "visible message level 2\n",
      [&]() { C.message( "visible message level 2\n", 2 ); } );
    expect_empty( C, out, "message hidden at level", [&]() { C.message( "hidden message level 3\n", 3 ); } );
    expect_empty(
      C,
      out,
      "message default hidden at level 2",
      [&]() { C.message( "hidden default message level 4\n" ); } );

    expect_contains(
      C,
      out,
      "formatted message visible at level",
      "visible formatted 2\n",
      [&]() { C.message( 2, "visible formatted {}\n", 2 ); } );
    expect_empty( C, out, "formatted message hidden at level", [&]() { C.message( 3, "hidden formatted {}\n", 3 ); } );

    expect_contains(
      C,
      out,
      "color visible at level",
      "visible red level 2\n",
      [&]() { C.red( "visible red level 2\n", 2 ); } );
    expect_empty( C, out, "color hidden at level", [&]() { C.red( "hidden red level 3\n", 3 ); } );

    expect_contains(
      C,
      out,
      "formatted color visible at level",
      "visible red formatted 2\n",
      [&]() { C.red( 2, "visible red formatted {}\n", 2 ); } );
    expect_empty( C, out, "formatted color hidden at level", [&]() { C.red( 3, "hidden red formatted {}\n", 3 ); } );

    expect_contains(
      C,
      out,
      "semaphore visible at level",
      "visible semaphore level 2\n",
      [&]() { C.semaphore( 0, "visible semaphore level 2\n", 2 ); } );
    expect_empty( C, out, "semaphore hidden at level", [&]() { C.semaphore( 0, "hidden semaphore level 3\n", 3 ); } );

    expect_contains(
      C,
      out,
      "formatted semaphore visible at level",
      "visible semaphore formatted 2\n",
      [&]() { C.semaphore( 2, 0, "visible semaphore formatted {}\n", 2 ); } );
    expect_empty(
      C,
      out,
      "formatted semaphore hidden at level",
      [&]() { C.semaphore( 3, 0, "hidden semaphore formatted {}\n", 3 ); } );

    expect_contains(
      C,
      out,
      "colors visible at level",
      "visible colors level 2\n",
      [&]() { C.colors( 0, "visible colors level 2\n", 2 ); } );
    expect_empty( C, out, "colors hidden at level", [&]() { C.colors( 0, "hidden colors level 3\n", 3 ); } );

    expect_contains(
      C,
      out,
      "formatted colors visible at level",
      "visible colors formatted 2\n",
      [&]() { C.colors( 2, 0, "visible colors formatted {}\n", 2 ); } );
    expect_empty(
      C,
      out,
      "formatted colors hidden at level",
      [&]() { C.colors( 3, 0, "hidden colors formatted {}\n", 3 ); } );

    C.change_level( 1 );
    expect_empty( C, out, "warning hidden at level 1", [&]() { C.warning( "hidden warning\n" ); } );
    expect_empty( C, out, "formatted warning hidden at level 1", [&]() { C.warning( "hidden warning {}\n", 1 ); } );
    expect_contains( C, out, "error visible at level 1", "visible error\n", [&]() { C.error( "visible error\n" ); } );
    expect_contains(
      C,
      out,
      "formatted error visible at level 1",
      "visible error 1\n",
      [&]() { C.error( "visible error {}\n", 1 ); } );

    C.change_level( 0 );
    expect_empty( C, out, "error hidden at level 0", [&]() { C.error( "hidden error\n" ); } );
    expect_empty( C, out, "formatted error hidden at level 0", [&]() { C.error( "hidden error {}\n", 0 ); } );

    C.change_level( 0 );
    expect_contains(
      C,
      out,
      "fatal visible at level 0",
      "visible fatal at 0\n",
      [&]() { C.fatal( "visible fatal at {}\n", 0 ); } );

    C.change_level( -1 );
    expect_empty( C, out, "normal output hidden at level -1", [&]() { C.green( "hidden green\n", 0 ); } );
    expect_empty(
      C,
      out,
      "negative message level hidden at level -1",
      [&]() { C.message( "hidden message at -1\n", -1 ); } );
    expect_empty(
      C,
      out,
      "negative formatted level hidden at level -1",
      [&]() { C.red( -2, "hidden formatted at {}\n", -2 ); } );
    expect_empty( C, out, "fatal hidden at level -1", [&]() { C.fatal( "hidden fatal at {}\n", -1 ); } );
  }

  void print_visual_examples()
  {
    Utils::Console C( &std::cout, 4 );

    std::cout << "\n========== Console: visual examples ==========\n";

    C.message( "message(string_view): plain message\n", 0 );
    C.message( 0, "message(format): integer={}, real={:.3f}, text={}\n", 42, 3.141592, "hello" );

    std::cout << "\n-- semaphore (red / yellow / green) --\n";
    C.semaphore( 0, "STOP  - red\n" );
    C.semaphore( 1, "WAIT  - yellow\n" );
    C.semaphore( 2, "GO    - green\n" );
    C.semaphore( 0, 2, "formatted semaphore: {} = {}\n", "progress", "100%" );

    std::cout << "\n-- colors (red / magenta / yellow / cyan / green) --\n";
    C.colors( 0, "colors(0): red\n" );
    C.colors( 1, "colors(1): magenta\n" );
    C.colors( 2, "colors(2): yellow\n" );
    C.colors( 3, "colors(3): cyan\n" );
    C.colors( 4, "colors(4): green\n" );
    C.colors( 0, 3, "formatted color: {} {:.1f}%\n", "load", 87.5 );

    std::cout << "\n-- warning / error / fatal --\n";
    C.warning( "warning(string_view): check this value\n" );
    C.warning( "warning(format): iteration {} of {}\n", 3, 10 );
    C.error( "error(string_view): recoverable error\n" );
    C.error( "error(format): code={}, file={}\n", -2, "data.txt" );
    C.fatal( "fatal(string_view): fatal example (the program continues)\n" );
    C.fatal( "fatal(format): code={}, reason={}\n", -9, "demo only" );

    std::cout << "\n-- named colors --\n";
    C.black( "black\n" );
    C.red( "red\n" );
    C.green( "green\n" );
    C.yellow( "yellow\n" );
    C.blue( "blue\n" );
    C.magenta( "magenta\n" );
    C.cyan( "cyan\n" );
    C.gray( 0, "gray formatted: {}\n", 123 );

    std::cout << "\n-- named colors, reversed --\n";
    C.black_reversed( " black reversed \n" );
    C.red_reversed( " red reversed \n" );
    C.green_reversed( " green reversed \n" );
    C.yellow_reversed( " yellow reversed \n" );
    C.blue_reversed( " blue reversed \n" );
    C.magenta_reversed( " magenta reversed \n" );
    C.cyan_reversed( " cyan reversed \n" );
    C.gray_reversed( 0, " gray reversed formatted: {} \n", 456 );

    std::cout << "\n-- custom styles --\n";
    C.set_message_style( fmt::emphasis::bold, fmt::color::white, fmt::color::blue );
    C.message( " bold white on blue message \n", 0 );
    C.set_warning_style( fmt::emphasis::underline, fmt::color::black, fmt::color::yellow );
    C.warning( " underlined black on yellow warning \n" );
    C.set_error_style( fmt::emphasis::italic, fmt::color::white, fmt::color::red );
    C.error( " italic white on red error \n" );
    C.set_fatal_style( fmt::emphasis::bold, fmt::color::yellow, fmt::color::red );
    C.fatal( " bold yellow on red fatal \n" );

    std::cout << "\n-- color switch and level filtering --\n";
    C.set_off();
    C.red( "set_off(): this line has no ANSI styling\n" );
    C.set_auto();
    C.red( "set_auto(): styling is enabled again\n" );
    C.change_level( 1 );
    C.warning( "THIS WARNING MUST BE HIDDEN (level 2 > threshold 1)\n" );
    C.error( "this error is visible (level 1 <= threshold 1)\n" );
    C.change_level( 4 );
    C.flush();

    std::cout << "========== End visual examples ==========\n";
  }

}  // namespace

int main()
{
  test_stream_and_level_methods();
  test_style_methods();
  test_legacy_message_methods();
  test_formatted_message_methods();
  test_level_filtering();
  print_visual_examples();

  std::cout << "\nAll Console tests passed\n";
  return 0;
}
