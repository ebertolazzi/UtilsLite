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
#include "Utils_trace.hh"

using std::cout;

static void fun1( int i );
static void fun2( int i );
static void fun3( int i );
static void fun4( int i );

static void fun5( char const str[] )
{
  cout << "fun3: " << str << '\n';
  Utils::print_trace( __LINE__, __FILE__, "in fun3", std::cerr );
}

static void fun1( int const i )
{
  cout << "in fun1\n";
  if ( i > 0 ) { fun2( i - 1 ); }
  else
  {
    std::string const str{ std::format( "format {}", i ) };
    fun5( str.data() );
  }
}

static void fun2( int i )
{
  cout << "in fun2\n";
  if ( i > 0 ) { fun3( i - 1 ); }
  else
  {
    std::string const str{ std::format( "format {}", i ) };
    fun5( str.data() );
  }
}

static void fun3( int i )
{
  cout << "in fun3\n";
  if ( i > 0 ) { fun4( i - 1 ); }
  else
  {
    std::string const str{ std::format( "format {}", i ) };
    fun5( str.data() );
  }
}

static void fun4( int i )
{
  cout << "in fun4\n";
  if ( i > 0 ) { fun1( i - 1 ); }
  else
  {
    std::string const str{ std::format( "format {}", i ) };
    fun5( str.data() );
  }
}

int main()
{
  try
  {
    constexpr int i{ 6 };
    cout << "call fun1\n";
    fun1( i );
  }
  catch ( std::exception const & exc )
  {
    cout << "Error: " << exc.what() << '\n';
  }
  catch ( ... )
  {
    cout << "Unknown error\n";
  }
  cout << "All done folks\n\n";
  return 0;
}
