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

#include "Utils_SHA3.hh"
#include "Utils_fmt.hh"

#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using integer   = int;
using real_type = double;

int
main() {
  Utils::SHA3 sha3_1(16), sha3_2(16);

  string a{"pippo"};
  string b{"pluto"};
  sha3_1.hash_string( a );
  sha3_2.hash_string( b );
  fmt::print( "{} -> {}\n", a, sha3_1.digest_in_hex() );
  fmt::print( "{} -> {}\n", b, sha3_2.digest_in_hex() );
  sha3_1.hash_string( b );
  sha3_2.hash_string( a );
  fmt::print( "{} -> {}\n", b, sha3_1.digest_in_hex() );
  fmt::print( "{} -> {}\n", a, sha3_2.digest_in_hex() );

  return 0;
}
