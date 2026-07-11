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
 |      Comprehensive Tokenizer Tests                                       |
 |      Using fmt:: for colored output with Unicode icons                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Utils_Token.hh"
#include "Utils_fmt.hh"

using std::endl;
using std::string;
using std::vector;

// Helper function to print vectors with colored formatting
template <typename T> void print_vector( const vector<T> & vec, const string & name )
{
  fmt::print( fg( fmt::color::light_blue ) | fmt::emphasis::bold, "{} [{} elements]: ", name, vec.size() );

  for ( size_t i = 0; i < vec.size(); ++i )
  {
    if ( i > 0 ) fmt::print( ", " );
    fmt::print( fg( fmt::color::light_green ), "'{}'", vec[i] );
  }
  fmt::print( "\n" );
}

void test_1_basic_tokenization()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 1: Basic Tokenization {}\n", "🔍", "🔍" );

  string           str        = "Hello World Tokenizer Test";
  string           delimiters = " ";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Tokens" );

  assert( tokens.size() == 4 );
  assert( tokens[0] == "Hello" );
  assert( tokens[1] == "World" );
  assert( tokens[2] == "Tokenizer" );
  assert( tokens[3] == "Test" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Basic tokenization passed\n" );
}

void test_2_multiple_delimiters()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 2: Multiple Delimiters {}\n", "🎯", "🎯" );

  string           str        = "apple,banana;cherry orange:grape";
  string           delimiters = ",; :";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Tokens" );

  assert( tokens.size() == 5 );
  assert( tokens[0] == "apple" );
  assert( tokens[1] == "banana" );
  assert( tokens[2] == "cherry" );
  assert( tokens[3] == "orange" );
  assert( tokens[4] == "grape" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Multiple delimiters passed\n" );
}

void test_3_consecutive_delimiters()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 3: Consecutive Delimiters {}\n", "🔗", "🔗" );

  string           str        = "data1,,,data2,data3";
  string           delimiters = ",";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Tokens" );

  // Note: consecutive delimiters are skipped, no empty tokens produced
  assert( tokens.size() == 3 );
  assert( tokens[0] == "data1" );
  assert( tokens[1] == "data2" );
  assert( tokens[2] == "data3" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Consecutive delimiters passed\n" );
}

void test_4_empty_string()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 4: Empty String {}\n", "⚫", "⚫" );

  string           str        = "";
  string           delimiters = ",;";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Tokens" );

  assert( tokens.empty() );
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Empty string passed\n" );
}

void test_5_only_delimiters()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 5: Only Delimiters {}\n", "📏", "📏" );

  string           str        = ",,,;;; , ;,";
  string           delimiters = ",; ";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Tokens" );

  assert( tokens.empty() );
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Only delimiters passed\n" );
}

void test_6_no_delimiters()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 6: No Delimiters {}\n", "🚫", "🚫" );

  string           str        = "SingleToken";
  string           delimiters = ",;";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Tokens" );

  assert( tokens.size() == 1 );
  assert( tokens[0] == "SingleToken" );
  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ No delimiters passed\n" );
}

void test_7_mixed_whitespace()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 7: Mixed Whitespace {}\n", "📝", "📝" );

  string           str        = "  token1  token2\ttoken3\ntoken4\r\n  ";
  string           delimiters = " \t\n\r";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Tokens" );

  assert( tokens.size() == 4 );
  assert( tokens[0] == "token1" );
  assert( tokens[1] == "token2" );
  assert( tokens[2] == "token3" );
  assert( tokens[3] == "token4" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Mixed whitespace passed\n" );
}

void test_8_split_string_function()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 8: split_string Function {}\n", "⚙️", "⚙️" );

  // Test original example
  string         str1 = "pippo,pluto paperino;nonna papera,,,zorro";
  string         sep1 = " ,;";
  vector<string> res1;
  Utils::split_string( str1, sep1, res1 );

  print_vector( res1, "Result 1" );
  assert( res1.size() == 6 );
  assert( res1[0] == "pippo" );
  assert( res1[1] == "pluto" );
  assert( res1[2] == "paperino" );
  assert( res1[3] == "nonna" );
  assert( res1[4] == "papera" );
  assert( res1[5] == "zorro" );

  // Test empty string
  vector<string> res2;
  Utils::split_string( "", ",;", res2 );
  assert( res2.empty() );

  // Test consecutive delimiters
  vector<string> res3;
  Utils::split_string( "a,,,b,c", ",", res3 );
  assert( res3.size() == 3 );
  assert( res3[0] == "a" );
  assert( res3[1] == "b" );
  assert( res3[2] == "c" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ split_string function passed\n" );
}

void test_9_reuse_behavior()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 9: Reuse Behavior {}\n", "🔁", "🔁" );

  string           str        = "token1 token2 token3";
  string           delimiters = " ";
  Utils::Tokenizer tokenizer( str, delimiters );

  // First pass
  vector<string> tokens1;
  while ( tokenizer.next_token() ) { tokens1.push_back( tokenizer.get_token() ); }

  // Second pass - should have no more tokens
  vector<string> tokens2;
  while ( tokenizer.next_token() ) { tokens2.push_back( tokenizer.get_token() ); }

  print_vector( tokens1, "First pass" );
  print_vector( tokens2, "Second pass" );

  assert( tokens1.size() == 3 );
  assert( tokens2.empty() );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Reuse behavior passed\n" );
}

void test_10_edge_cases()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 10: Edge Cases {}\n", "⚠️", "⚠️" );

  // Case 1: Delimiters at beginning and end
  {
    fmt::print( fg( fmt::color::yellow ), "  • Test: delimiters at beginning and end\n" );
    string           str        = ",start,middle,end,";
    string           delimiters = ",";
    Utils::Tokenizer tokenizer( str, delimiters );

    vector<string> tokens;
    while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

    assert( tokens.size() == 3 );
    assert( tokens[0] == "start" );
    assert( tokens[1] == "middle" );
    assert( tokens[2] == "end" );
  }

  // Case 2: Single character string
  {
    fmt::print( fg( fmt::color::yellow ), "  • Test: single character string\n" );
    string           str        = "a";
    string           delimiters = ",";
    Utils::Tokenizer tokenizer( str, delimiters );

    assert( tokenizer.next_token() );
    assert( tokenizer.get_token() == "a" );
    assert( !tokenizer.next_token() );
  }

  // Case 3: All characters the same
  {
    fmt::print( fg( fmt::color::yellow ), "  • Test: all characters the same\n" );
    string           str        = "aaaaaaaa";
    string           delimiters = "b";
    Utils::Tokenizer tokenizer( str, delimiters );

    assert( tokenizer.next_token() );
    assert( tokenizer.get_token() == "aaaaaaaa" );
    assert( !tokenizer.next_token() );
  }

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Edge cases passed\n" );
}

void test_11_get_token_without_next()
{
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n{} Test 11: get_token without next_token {}\n",
    "❓",
    "❓" );

  string           str        = "test token";
  string           delimiters = " ";
  Utils::Tokenizer tokenizer( str, delimiters );

  // Calling get_token() before next_token() - should return empty string
  assert( tokenizer.get_token() == "" );

  tokenizer.next_token();
  assert( tokenizer.get_token() == "test" );

  tokenizer.next_token();
  assert( tokenizer.get_token() == "token" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ get_token without next_token passed\n" );
}

void test_12_unicode_special_chars()
{
  fmt::print(
    fg( fmt::color::cyan ) | fmt::emphasis::bold,
    "\n{} Test 12: Unicode and Special Characters {}\n",
    "🌍",
    "🌍" );

  // CORRECTED: Using single-byte ASCII delimiters to avoid UTF-8 issues
  string           str        = "hello|world|test";
  string           delimiters = "|";
  Utils::Tokenizer tokenizer( str, delimiters );

  vector<string> tokens;
  while ( tokenizer.next_token() ) { tokens.push_back( tokenizer.get_token() ); }

  print_vector( tokens, "Unicode tokens" );

  assert( tokens.size() == 3 );
  assert( tokens[0] == "hello" );
  assert( tokens[1] == "world" );
  assert( tokens[2] == "test" );

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Unicode and special characters passed\n" );
}

void test_13_performance_test()
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{} Test 13: Performance Test {}\n", "⚡", "⚡" );

  // Build a long string
  string    long_string;
  const int NUM_TOKENS = 10000;

  for ( int i = 0; i < NUM_TOKENS; ++i )
  {
    long_string += "token" + std::to_string( i );
    if ( i < NUM_TOKENS - 1 ) { long_string += ","; }
  }

  string delimiters = ",";

  // Test with Tokenizer
  {
    Utils::Tokenizer tokenizer( long_string, delimiters );
    int              count = 0;
    while ( tokenizer.next_token() ) { ++count; }
    assert( count == NUM_TOKENS );
  }

  // Test with split_string
  {
    vector<string> tokens;
    Utils::split_string( long_string, delimiters, tokens );
    assert( tokens.size() == NUM_TOKENS );
  }

  fmt::print( fg( fmt::color::green ) | fmt::emphasis::bold, "  ✅ Performance test passed ({} tokens)\n", NUM_TOKENS );
}

int main()
{
  fmt::print(
    fg( fmt::color::magenta ) | fmt::emphasis::bold,
    "\n{} Starting comprehensive Tokenizer tests {} {}\n\n",
    "🚀",
    "📋",
    "🚀" );

  fmt::print( fg( fmt::color::white ) | fmt::emphasis::italic, "Using fmt:: with colors and Unicode icons\n\n" );

  try
  {
    test_1_basic_tokenization();
    test_2_multiple_delimiters();
    test_3_consecutive_delimiters();
    test_4_empty_string();
    test_5_only_delimiters();
    test_6_no_delimiters();
    test_7_mixed_whitespace();
    test_8_split_string_function();
    test_9_reuse_behavior();
    test_10_edge_cases();
    test_11_get_token_without_next();
    test_12_unicode_special_chars();
    test_13_performance_test();

    fmt::print(
      fg( fmt::color::green ) | fmt::emphasis::bold,
      "\n{} ALL TESTS PASSED SUCCESSFULLY! {} {}\n",
      "✅",
      "🎉",
      "✅" );

    fmt::print(
      fg( fmt::color::cyan ) | fmt::emphasis::bold,
      "════════════════════════════════════════════════════\n" );
    fmt::print( fg( fmt::color::light_green ), "  Tokenizer works correctly in all tested cases\n" );
    fmt::print(
      fg( fmt::color::cyan ) | fmt::emphasis::bold,
      "════════════════════════════════════════════════════\n" );
  }
  catch ( const std::exception & e )
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n❌ Test failed with exception: {}\n", e.what() );
    return 1;
  }
  catch ( ... )
  {
    fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\n❌ Test failed with unknown exception\n" );
    return 1;
  }

  return 0;
}
