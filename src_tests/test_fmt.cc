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

#include <chrono>
#include <iostream>
#include <string>

#include "Utils.hh"
#include "Utils_fmt.hh"

using namespace std;
using integer   = int;
using real_type = double;

int main()
{
  fmt::print( "Don't {}\n\n", "panic" );
  fmt::print( "I'd rather be {1} than {0}.\n\n", "right", "happy" );
  fmt::print(
    "Hello, {name}! The answer is {number}. Goodbye, {name}.\n\n",
    fmt::arg( "name", "World" ),
    fmt::arg( "number", 42 ) );
  using namespace fmt::literals;
  fmt::print( "Hello, {name}! The answer is {number}. Goodbye, {name}.\n\n", "name"_a = "World", "number"_a = 42 );

  std::string const message{ std::format( "The answer is {}.", 42 ) };
  std::cout << message << "\n\n";

  //  run time error
  // std::string s = std::format(FMT_STRING("{:d}"), "foo");
  std::time_t const t{ std::time( nullptr ) };

  // Prints "The date is 2020-11-07." (with the current date):
  fmt::print( "The date is {:%Y-%m-%d}.\n", fmt::gmtime( t ) );

  // C++14
  using namespace std::literals::chrono_literals;

  // Prints "Default format: 42s 100ms":
  fmt::print( "Default format: {} {}\n", 42s, 100ms );

  // Prints "strftime-like format: 03:15:30":
  fmt::print( "strftime-like format: {:%H:%M:%S}\n", 3h + 15min + 30s );

  fmt::print( fmt::emphasis::bold | fg( fmt::color::red ), "Elapsed time: {0:.2f} seconds\n\n", 1.23 );

  fmt::print(
    "Elapsed time: {0:.2f} seconds\n\n",
    fmt::styled( 1.23, fmt::fg( fmt::color::green ) | fmt::bg( fmt::color::blue ) ) );

  fmt::printf( "Elapsed time: %.2f seconds\n\n", 1.23 );

  std::cout
    << Utils::fmt_table_top_row( 50, " PIPPO " ) << Utils::fmt_table_row( 50 ) << Utils::fmt_table_row( 50 )
    << Utils::fmt_table_middle_row( 50 )
    << Utils::fmt_table_row( 50, "├", "┬", "┤\n", "─", std::initializer_list<string_view>{ "", "PLUTO", "" }, "^" )
    << Utils::fmt_table_row( 50, { "A", "B", "C" } ) << Utils::fmt_table_row( 50, "├", "┴", "┤\n", "─", 3 )
    << Utils::fmt_table_row( 50 ) << Utils::fmt_table_row( 50 ) << Utils::fmt_table_bottom_row( 50 )
    << Utils::fmt_table_top_row( 50, 4 )
    << Utils::fmt_table_row( 50, std::initializer_list<string_view>{ "pippo", "plto", "paperino", "paperino" } )
    << Utils::fmt_table_middle_row( 50, 4 )
    << Utils::fmt_table_row( 50, std::initializer_list<string_view>{ "pippo", "pluto", "paino", "paperino" } )
    << Utils::fmt_table_middle_row( 50, 4 )
    << Utils::fmt_table_row( 50, std::initializer_list<string_view>{ "pippo", "pluto", "paperino", "erino" } )
    << Utils::fmt_table_bottom_row( 50, 4 ) << Utils::fmt_table_top_row( 50 )
    << Utils::fmt_table_row( 50, std::initializer_list<string_view>{ "unico" } ) << Utils::fmt_table_bottom_row( 50 );

  fmt::print( "\n\nAll done!\n" );
  return 0;
}
