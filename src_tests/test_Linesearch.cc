/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Comprehensive Line Search Testing Framework - Enhanced Version          |
 |  Test file: test_Linesearch_enhanced.cc                                  |
 |                                                                          |
 |  This program thoroughly tests all line search algorithms implemented    |
 |  in Utils_nonlinear_linesearch.hh                                        |
 |  Uses fmt::print for all output with proper Unicode and colors           |
 |                                                                          |
 |  Author: Enrico Bertolazzi                                               |
 |  Version: 3.5                                                            |
 |  Date: 2025                                                              |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Utils_eigen.hh"
#include "Utils_Linesearch.hh"
#include "Utils_TicToc.hh"

namespace fmt
{
  template <typename Scalar, int Rows> struct formatter<Eigen::Matrix<Scalar, Rows, 1>>
  {
    constexpr auto parse( format_parse_context & ctx ) { return ctx.begin(); }

    template <typename FormatContext> auto format( const Eigen::Matrix<Scalar, Rows, 1> & vec, FormatContext & ctx )
    {
      auto out = ctx.out();
      *out++   = '[';
      for ( int i = 0; i < vec.size(); ++i )
      {
        if ( i > 0 )
        {
          *out++ = ',';
          *out++ = ' ';
        }
        out = fmt::format_to( out, "{:.6g}", vec[i] );
      }
      *out++ = ']';
      return out;
    }
  };
}  // namespace fmt

using namespace Utils;
using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// ===========================================================================
// Utility Functions
// ===========================================================================

inline std::string join_strings( const std::vector<std::string> & strings, const std::string & delimiter )
{
  if ( strings.empty() ) return "";

  std::string result;
  for ( size_t i = 0; i < strings.size(); ++i )
  {
    if ( i > 0 ) result += delimiter;
    result += strings[i];
  }
  return result;
}

inline std::string repeat( int n, std::string const & c )
{
  std::string res;
  for ( int k = 0; k < n; ++k ) res += c;
  return res;
}

// ===========================================================================
// Unicode Symbols and Table Constants
// ===========================================================================

namespace Unicode
{
  const std::string CHECK       = "✓";
  const std::string CROSS       = "✗";
  const std::string WARNING     = "⚠";
  const std::string INFO        = "ℹ";
  const std::string CORNER_TL   = "┌";
  const std::string CORNER_TR   = "┐";
  const std::string CORNER_BL   = "└";
  const std::string CORNER_BR   = "┘";
  const std::string HORIZONTAL  = "─";
  const std::string VERTICAL    = "│";
  const std::string T_UP        = "┴";
  const std::string T_DOWN      = "┬";
  const std::string T_LEFT      = "┤";
  const std::string T_RIGHT     = "├";
  const std::string CROSS_T     = "┼";
  const std::string ARROW_UP    = "↑";
  const std::string ARROW_DOWN  = "↓";
  const std::string ARROW_RIGHT = "→";
  const std::string ARROW_LEFT  = "←";
}  // namespace Unicode

// ===========================================================================
// Style and Color Definitions
// ===========================================================================

namespace Style
{
  const fmt::text_style NONE              = fmt::text_style();
  const fmt::text_style BOLD              = fmt::emphasis::bold;
  const fmt::text_style DIM               = fmt::emphasis::faint;
  const fmt::text_style RED               = fg( fmt::color::red );
  const fmt::text_style GREEN             = fg( fmt::color::green );
  const fmt::text_style YELLOW            = fg( fmt::color::yellow );
  const fmt::text_style MAGENTA           = fg( fmt::color::magenta );
  const fmt::text_style CYAN              = fg( fmt::color::cyan );
  const fmt::text_style GRAY              = fg( fmt::color::gray );
  const fmt::text_style SUCCESS           = BOLD | GREEN;
  const fmt::text_style WARNING           = BOLD | YELLOW;
  const fmt::text_style ERROR             = BOLD | RED;
  const fmt::text_style HEADER            = BOLD | CYAN;
  const fmt::text_style TITLE             = BOLD | MAGENTA;
  const fmt::text_style COMMENT           = DIM | GRAY;
  const fmt::text_style ARMIDO_COLOR      = fg( fmt::color::light_green );
  const fmt::text_style WOLFE_COLOR       = fg( fmt::color::light_blue );
  const fmt::text_style STRONGWOLFE_COLOR = fg( fmt::color::dodger_blue );
  const fmt::text_style GOLDSTEIN_COLOR   = fg( fmt::color::gold );
  const fmt::text_style HAGERZHANG_COLOR  = fg( fmt::color::violet );
  const fmt::text_style MORETHUENTE_COLOR = fg( fmt::color::light_sea_green );
}  // namespace Style

// ===========================================================================
// Table Printer
// ===========================================================================

class TablePrinter
{
private:
  std::vector<int>             column_widths;
  std::vector<std::string>     headers;
  std::vector<fmt::text_style> header_styles;
  std::vector<fmt::text_style> column_styles;

public:
  TablePrinter(
    const std::vector<std::pair<std::string, int>> & cols,
    const std::vector<fmt::text_style> &             styles = {} )
  {
    for ( const auto & [header, width] : cols )
    {
      headers.push_back( header );
      column_widths.push_back( width );
    }

    if ( styles.empty() )
    {
      header_styles = std::vector<fmt::text_style>( headers.size(), Style::HEADER );
      column_styles = std::vector<fmt::text_style>( headers.size(), Style::NONE );
    }
    else
    {
      header_styles = styles;
      column_styles = styles;
    }
  }

  void print_header()
  {
    fmt::print( "{}", Unicode::CORNER_TL );
    for ( size_t i = 0; i < column_widths.size(); ++i )
    {
      fmt::print(
        "{}{}",
        repeat( column_widths[i], Unicode::HORIZONTAL ),
        ( i < column_widths.size() - 1 ) ? Unicode::T_DOWN : Unicode::CORNER_TR );
    }
    fmt::print( "\n" );

    fmt::print( "{}", Unicode::VERTICAL );
    for ( size_t i = 0; i < headers.size(); ++i )
    {
      fmt::print( header_styles[i], "{:^{}}", headers[i], column_widths[i] );
      fmt::print( "{}", Unicode::VERTICAL );
    }
    fmt::print( "\n" );

    fmt::print( "{}", Unicode::T_RIGHT );
    for ( size_t i = 0; i < column_widths.size(); ++i )
    {
      fmt::print(
        "{}{}",
        repeat( column_widths[i], Unicode::HORIZONTAL ),
        ( i < column_widths.size() - 1 ) ? Unicode::CROSS_T : Unicode::T_LEFT );
    }
    fmt::print( "\n" );
  }

  void print_row( const std::vector<std::string> & cells, const std::vector<fmt::text_style> & cell_styles = {} )
  {
    fmt::print( "{}", Unicode::VERTICAL );
    for ( size_t i = 0; i < cells.size(); ++i )
    {
      fmt::text_style style = ( i < cell_styles.size() ) ? cell_styles[i] : column_styles[i];
      fmt::print( style, "{:<{}}", cells[i], column_widths[i] );
      fmt::print( "{}", Unicode::VERTICAL );
    }
    fmt::print( "\n" );
  }

  void print_footer()
  {
    fmt::print( "{}", Unicode::CORNER_BL );
    for ( size_t i = 0; i < column_widths.size(); ++i )
    {
      fmt::print(
        "{}{}",
        repeat( column_widths[i], Unicode::HORIZONTAL ),
        ( i < column_widths.size() - 1 ) ? Unicode::T_UP : Unicode::CORNER_BR );
    }
    fmt::print( "\n" );
  }

  void print_separator()
  {
    fmt::print( "{}", Unicode::T_RIGHT );
    for ( size_t i = 0; i < column_widths.size(); ++i )
    {
      fmt::print(
        "{}{}",
        repeat( column_widths[i], Unicode::HORIZONTAL ),
        ( i < column_widths.size() - 1 ) ? Unicode::CROSS_T : Unicode::T_LEFT );
    }
    fmt::print( "\n" );
  }
};

// ===========================================================================
// Test Problem Interface and Factory
// ===========================================================================

class TestFunction
{
public:
  virtual ~TestFunction()                                                       = default;
  virtual Scalar              value( const Vector & x ) const                   = 0;
  virtual void                gradient( const Vector & x, Vector & grad ) const = 0;
  virtual std::string         name() const                                      = 0;
  virtual std::string         description() const                               = 0;
  virtual int                 dimension() const                                 = 0;
  virtual Vector              solution() const                                  = 0;
  virtual std::vector<Vector> test_points() const                               = 0;
  virtual std::vector<Vector> test_directions( const Vector & x ) const         = 0;
};

class Quadratic1D : public TestFunction
{
public:
  std::string name() const override { return "Quadratic1D"; }
  std::string description() const override { return "f(x) = (x - 2)²"; }
  int         dimension() const override { return 1; }
  Vector      solution() const override { return Vector::Constant( 1, 2.0 ); }

  Scalar value( const Vector & x ) const override { return ( x[0] - 2.0 ) * ( x[0] - 2.0 ); }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 1 );
    grad[0] = 2.0 * ( x[0] - 2.0 );
  }

  std::vector<Vector> test_points() const override
  {
    return { Vector::Constant( 1, 0.0 ),
             Vector::Constant( 1, 5.0 ),
             Vector::Constant( 1, -3.0 ),
             Vector::Constant( 1, 10.0 ) };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 1 );
    gradient( x, grad );
    return { -grad, Vector::Constant( 1, ( x[0] > 2.0 ) ? -1.0 : 1.0 ), -0.1 * grad, -5.0 * grad };
  }
};

class Rosenbrock2D : public TestFunction
{
public:
  std::string name() const override { return "Rosenbrock2D"; }
  std::string description() const override { return "f(x,y) = (1-x)² + 100(y-x²)²"; }
  int         dimension() const override { return 2; }
  Vector      solution() const override
  {
    Vector sol( 2 );
    sol << 1.0, 1.0;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar a = 1.0 - x[0];
    Scalar b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 2 );
    Scalar a = 1.0 - x[0];
    Scalar b = x[1] - x[0] * x[0];
    grad[0]  = -2.0 * a - 400.0 * x[0] * b;
    grad[1]  = 200.0 * b;
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 2 ) << -1.2, 1.0 ).finished(),
             ( Vector( 2 ) << 0.0, 0.0 ).finished(),
             ( Vector( 2 ) << 2.0, 2.0 ).finished(),
             ( Vector( 2 ) << -0.5, 1.5 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 2 );
    gradient( x, grad );

    std::vector<Vector> dirs = { -grad };

    Scalar h11 = 2.0 + 1200.0 * x[0] * x[0] - 400.0 * x[1];
    Scalar h12 = -400.0 * x[0];
    Scalar h22 = 200.0;
    Scalar det = h11 * h22 - h12 * h12;

    if ( std::abs( det ) > 1e-10 )
    {
      Vector newton_dir( 2 );
      newton_dir[0] = -( h22 * grad[0] - h12 * grad[1] ) / det;
      newton_dir[1] = -( -h12 * grad[0] + h11 * grad[1] ) / det;
      dirs.push_back( newton_dir );
    }

    dirs.push_back( ( Vector( 2 ) << 0.5, -0.5 ).finished().normalized() );
    dirs.push_back( ( Vector( 2 ) << -1.0, 0.2 ).finished().normalized() );

    return dirs;
  }
};

class BealeFunction : public TestFunction
{
public:
  std::string name() const override { return "Beale2D"; }
  std::string description() const override { return "f(x,y) = Σ(1.5-x+xy)²"; }
  int         dimension() const override { return 2; }
  Vector      solution() const override
  {
    Vector sol( 2 );
    sol << 3.0, 0.5;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar x1 = x[0], x2 = x[1];
    Scalar t1 = 1.5 - x1 + x1 * x2;
    Scalar t2 = 2.25 - x1 + x1 * x2 * x2;
    Scalar t3 = 2.625 - x1 + x1 * x2 * x2 * x2;
    return t1 * t1 + t2 * t2 + t3 * t3;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    Scalar x1 = x[0], x2 = x[1];
    Scalar t1 = 1.5 - x1 + x1 * x2;
    Scalar t2 = 2.25 - x1 + x1 * x2 * x2;
    Scalar t3 = 2.625 - x1 + x1 * x2 * x2 * x2;

    Scalar dt1_dx1 = -1.0 + x2;
    Scalar dt1_dx2 = x1;
    Scalar dt2_dx1 = -1.0 + x2 * x2;
    Scalar dt2_dx2 = 2.0 * x1 * x2;
    Scalar dt3_dx1 = -1.0 + x2 * x2 * x2;
    Scalar dt3_dx2 = 3.0 * x1 * x2 * x2;

    grad.resize( 2 );
    grad[0] = 2.0 * t1 * dt1_dx1 + 2.0 * t2 * dt2_dx1 + 2.0 * t3 * dt3_dx1;
    grad[1] = 2.0 * t1 * dt1_dx2 + 2.0 * t2 * dt2_dx2 + 2.0 * t3 * dt3_dx2;
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 2 ) << 1.0, 1.0 ).finished(),
             ( Vector( 2 ) << 2.0, 0.0 ).finished(),
             ( Vector( 2 ) << 4.0, 0.5 ).finished(),
             ( Vector( 2 ) << 0.5, 2.0 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 2 );
    gradient( x, grad );
    return { -grad, -0.5 * grad, -2.0 * grad, ( Vector( 2 ) << -1.0, 0.5 ).finished().normalized() };
  }
};

class Exponential1D : public TestFunction
{
public:
  std::string name() const override { return "Exponential1D"; }
  std::string description() const override { return "f(x) = exp(x) - x"; }
  int         dimension() const override { return 1; }
  Vector      solution() const override { return Vector::Constant( 1, 0.0 ); }

  Scalar value( const Vector & x ) const override { return std::exp( x[0] ) - x[0]; }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 1 );
    grad[0] = std::exp( x[0] ) - 1.0;
  }

  std::vector<Vector> test_points() const override
  {
    return { Vector::Constant( 1, -2.0 ),
             Vector::Constant( 1, -1.0 ),
             Vector::Constant( 1, 0.5 ),
             Vector::Constant( 1, 2.0 ),
             Vector::Constant( 1, 5.0 ) };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 1 );
    gradient( x, grad );
    return { -grad, Vector::Constant( 1, ( x[0] > 0 ) ? -0.1 : 0.1 ), -0.01 * grad, -10.0 * grad };
  }
};

// ===========================================================================
// New Difficult Test Functions (8 additional functions)
// ===========================================================================

class PowellSingular4D : public TestFunction
{
public:
  std::string name() const override { return "PowellSingular4D"; }
  std::string description() const override { return "Powell Singular function (badly scaled)"; }
  int         dimension() const override { return 4; }
  Vector      solution() const override
  {
    Vector sol( 4 );
    sol << 0.0, 0.0, 0.0, 0.0;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar f1 = x[0] + 10.0 * x[1];
    Scalar f2 = std::sqrt( 5.0 ) * ( x[2] - x[3] );
    Scalar f3 = ( x[1] - 2.0 * x[2] ) * ( x[1] - 2.0 * x[2] );
    Scalar f4 = std::sqrt( 10.0 ) * ( x[0] - x[3] ) * ( x[0] - x[3] );
    return f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 4 );
    Scalar f1 = x[0] + 10.0 * x[1];
    Scalar f2 = std::sqrt( 5.0 ) * ( x[2] - x[3] );
    // Scalar f3 = (x[1] - 2.0 * x[2]) * (x[1] - 2.0 * x[2]);
    // Scalar f4 = std::sqrt(10.0) * (x[0] - x[3]) * (x[0] - x[3]);

    grad[0] = 2.0 * f1 + 2.0 * std::sqrt( 10.0 ) * ( x[0] - x[3] );
    grad[1] = 20.0 * f1 + 2.0 * ( x[1] - 2.0 * x[2] );
    grad[2] = 2.0 * std::sqrt( 5.0 ) * f2 - 4.0 * ( x[1] - 2.0 * x[2] );
    grad[3] = -2.0 * std::sqrt( 5.0 ) * f2 - 2.0 * std::sqrt( 10.0 ) * ( x[0] - x[3] );
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 4 ) << 3.0, -1.0, 0.0, 1.0 ).finished(),
             ( Vector( 4 ) << 1.0, 2.0, 3.0, 4.0 ).finished(),
             ( Vector( 4 ) << -1.0, -2.0, -3.0, -4.0 ).finished(),
             ( Vector( 4 ) << 10.0, 0.0, 0.0, 10.0 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 4 );
    gradient( x, grad );
    return { -grad, -0.5 * grad, -0.1 * grad, ( Vector( 4 ) << 1.0, -2.0, 1.0, -2.0 ).finished().normalized() };
  }
};

class TrigonometricFunction : public TestFunction
{
public:
  std::string name() const override { return "Trigonometric2D"; }
  std::string description() const override { return "Trigonometric function (oscillatory)"; }
  int         dimension() const override { return 2; }
  Vector      solution() const override
  {
    Vector sol( 2 );
    sol << 0.1, 0.1;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar sum = 0.0;
    for ( int i = 0; i < 2; ++i )
    {
      Scalar term = 2 + 2 * i - std::cos( x[0] ) - std::cos( x[1] );
      sum += term * term;
    }
    return sum;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 2 );
    Scalar g1 = 2 + 0 - std::cos( x[0] ) - std::cos( x[1] );
    Scalar g2 = 2 + 2 - std::cos( x[0] ) - std::cos( x[1] );

    grad[0] = 2.0 * g1 * std::sin( x[0] ) + 2.0 * g2 * std::sin( x[0] );
    grad[1] = 2.0 * g1 * std::sin( x[1] ) + 2.0 * g2 * std::sin( x[1] );
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 2 ) << 0.5, 0.5 ).finished(),
             ( Vector( 2 ) << 2.0, 2.0 ).finished(),
             ( Vector( 2 ) << -1.0, 1.0 ).finished(),
             ( Vector( 2 ) << 3.0, -2.0 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 2 );
    gradient( x, grad );
    return { -grad,
             -0.3 * grad,
             ( Vector( 2 ) << std::cos( x[0] ), std::sin( x[1] ) ).finished().normalized(),
             ( Vector( 2 ) << -1.0, 2.0 ).finished().normalized() };
  }
};

class WoodFunction : public TestFunction
{
public:
  std::string name() const override { return "WoodFunction4D"; }
  std::string description() const override { return "Wood function (4D, curved valley)"; }
  int         dimension() const override { return 4; }
  Vector      solution() const override
  {
    Vector sol( 4 );
    sol << 1.0, 1.0, 1.0, 1.0;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar f1 = 100.0 * ( x[1] - x[0] * x[0] ) * ( x[1] - x[0] * x[0] );
    Scalar f2 = ( 1.0 - x[0] ) * ( 1.0 - x[0] );
    Scalar f3 = 90.0 * ( x[3] - x[2] * x[2] ) * ( x[3] - x[2] * x[2] );
    Scalar f4 = ( 1.0 - x[2] ) * ( 1.0 - x[2] );
    Scalar f5 = 10.1 * ( ( x[1] - 1.0 ) * ( x[1] - 1.0 ) + ( x[3] - 1.0 ) * ( x[3] - 1.0 ) );
    Scalar f6 = 19.8 * ( x[1] - 1.0 ) * ( x[3] - 1.0 );
    return f1 + f2 + f3 + f4 + f5 + f6;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 4 );

    grad[0] = -400.0 * x[0] * ( x[1] - x[0] * x[0] ) - 2.0 * ( 1.0 - x[0] );
    grad[1] = 200.0 * ( x[1] - x[0] * x[0] ) + 20.2 * ( x[1] - 1.0 ) + 19.8 * ( x[3] - 1.0 );
    grad[2] = -360.0 * x[2] * ( x[3] - x[2] * x[2] ) - 2.0 * ( 1.0 - x[2] );
    grad[3] = 180.0 * ( x[3] - x[2] * x[2] ) + 20.2 * ( x[3] - 1.0 ) + 19.8 * ( x[1] - 1.0 );
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 4 ) << -3.0, -1.0, -3.0, -1.0 ).finished(),
             ( Vector( 4 ) << 0.0, 0.0, 0.0, 0.0 ).finished(),
             ( Vector( 4 ) << 2.0, 2.0, 2.0, 2.0 ).finished(),
             ( Vector( 4 ) << -1.0, 2.0, -2.0, 1.0 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 4 );
    gradient( x, grad );
    return { -grad,
             -0.2 * grad,
             ( Vector( 4 ) << 1.0, -1.0, 1.0, -1.0 ).finished().normalized(),
             ( Vector( 4 ) << x[0], x[1], -x[2], -x[3] ).finished().normalized() };
  }
};

class ExtendedRosenbrock : public TestFunction
{
  int m_dim;

public:
  ExtendedRosenbrock( int dim = 4 ) : m_dim( dim ) {}

  std::string name() const override { return "ExtendedRosenbrock" + std::to_string( m_dim ) + "D"; }
  std::string description() const override { return "Extended Rosenbrock (chain of valleys)"; }
  int         dimension() const override { return m_dim; }
  Vector      solution() const override
  {
    Vector sol( m_dim );
    for ( int i = 0; i < m_dim; i++ ) sol[i] = 1.0;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar sum = 0.0;
    for ( int i = 0; i < m_dim / 2; i++ )
    {
      int    idx1 = 2 * i;
      int    idx2 = 2 * i + 1;
      Scalar t1   = 1.0 - x[idx1];
      Scalar t2   = 10.0 * ( x[idx2] - x[idx1] * x[idx1] );
      sum += t1 * t1 + t2 * t2;
    }
    return sum;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( m_dim );
    grad.setZero();

    for ( int i = 0; i < m_dim / 2; i++ )
    {
      int idx1 = 2 * i;
      int idx2 = 2 * i + 1;

      Scalar t1 = 1.0 - x[idx1];
      Scalar t2 = 10.0 * ( x[idx2] - x[idx1] * x[idx1] );

      grad[idx1] += -2.0 * t1 - 400.0 * x[idx1] * t2;
      grad[idx2] += 200.0 * t2;
    }
  }

  std::vector<Vector> test_points() const override
  {
    std::vector<Vector> points;
    if ( m_dim == 4 )
    {
      points = { ( Vector( 4 ) << -1.2, 1.0, -1.2, 1.0 ).finished(),
                 ( Vector( 4 ) << 0.0, 0.0, 0.0, 0.0 ).finished(),
                 ( Vector( 4 ) << 2.0, 4.0, 2.0, 4.0 ).finished(),
                 ( Vector( 4 ) << -0.5, 1.5, -0.5, 1.5 ).finished() };
    }
    else if ( m_dim == 6 )
    {
      points = { Vector::Constant( 6, -1.2 ),
                 Vector::Constant( 6, 0.0 ),
                 Vector::Constant( 6, 1.0 ),
                 ( Vector( 6 ) << -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 ).finished() };
    }
    return points;
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( m_dim );
    gradient( x, grad );

    std::vector<Vector> dirs = { -grad, -0.5 * grad };

    // Add some alternating directions
    Vector alt_dir( m_dim );
    for ( int i = 0; i < m_dim; i++ ) { alt_dir[i] = ( i % 2 == 0 ) ? 1.0 : -1.0; }
    dirs.push_back( alt_dir.normalized() );

    // Add a direction with small components
    dirs.push_back( Vector::Constant( m_dim, 0.1 ).normalized() );

    return dirs;
  }
};

class BoothFunction : public TestFunction
{
public:
  std::string name() const override { return "Booth2D"; }
  std::string description() const override { return "Booth function (2D, bowl-shaped)"; }
  int         dimension() const override { return 2; }
  Vector      solution() const override
  {
    Vector sol( 2 );
    sol << 1.0, 3.0;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar t1 = x[0] + 2.0 * x[1] - 7.0;
    Scalar t2 = 2.0 * x[0] + x[1] - 5.0;
    return t1 * t1 + t2 * t2;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 2 );
    Scalar t1 = x[0] + 2.0 * x[1] - 7.0;
    Scalar t2 = 2.0 * x[0] + x[1] - 5.0;

    grad[0] = 2.0 * t1 + 4.0 * t2;
    grad[1] = 4.0 * t1 + 2.0 * t2;
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 2 ) << 0.0, 0.0 ).finished(),
             ( Vector( 2 ) << 5.0, 5.0 ).finished(),
             ( Vector( 2 ) << -2.0, 3.0 ).finished(),
             ( Vector( 2 ) << 3.0, -1.0 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 2 );
    gradient( x, grad );
    return { -grad,
             -0.3 * grad,
             ( Vector( 2 ) << -1.0, 0.5 ).finished().normalized(),
             ( Vector( 2 ) << 0.5, -1.0 ).finished().normalized() };
  }
};

class MatyasFunction : public TestFunction
{
public:
  std::string name() const override { return "Matyas2D"; }
  std::string description() const override { return "Matyas function (2D, gentle slope)"; }
  int         dimension() const override { return 2; }
  Vector      solution() const override
  {
    Vector sol( 2 );
    sol << 0.0, 0.0;
    return sol;
  }

  Scalar value( const Vector & x ) const override { return 0.26 * ( x[0] * x[0] + x[1] * x[1] ) - 0.48 * x[0] * x[1]; }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 2 );
    grad[0] = 0.52 * x[0] - 0.48 * x[1];
    grad[1] = 0.52 * x[1] - 0.48 * x[0];
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 2 ) << 5.0, 5.0 ).finished(),
             ( Vector( 2 ) << -3.0, 2.0 ).finished(),
             ( Vector( 2 ) << 1.0, -4.0 ).finished(),
             ( Vector( 2 ) << 2.5, 2.5 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 2 );
    gradient( x, grad );
    return { -grad,
             -0.5 * grad,
             ( Vector( 2 ) << 1.0, 1.0 ).finished().normalized(),
             ( Vector( 2 ) << 1.0, -1.0 ).finished().normalized() };
  }
};

class ThreeHumpCamel : public TestFunction
{
public:
  std::string name() const override { return "ThreeHumpCamel2D"; }
  std::string description() const override { return "Three-hump camel function (3 local minima)"; }
  int         dimension() const override { return 2; }
  Vector      solution() const override
  {
    Vector sol( 2 );
    sol << 0.0, 0.0;
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    return 2.0 * x[0] * x[0] - 1.05 * x[0] * x[0] * x[0] * x[0] + ( x[0] * x[0] * x[0] * x[0] * x[0] * x[0] ) / 6.0 +
           x[0] * x[1] + x[1] * x[1];
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 2 );
    grad[0] = 4.0 * x[0] - 4.2 * x[0] * x[0] * x[0] + x[0] * x[0] * x[0] * x[0] * x[0] + x[1];
    grad[1] = x[0] + 2.0 * x[1];
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 2 ) << 1.0, 1.0 ).finished(),
             ( Vector( 2 ) << -1.0, -1.0 ).finished(),
             ( Vector( 2 ) << 1.5, 0.0 ).finished(),
             ( Vector( 2 ) << 0.0, 1.5 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 2 );
    gradient( x, grad );
    return { -grad,
             -0.3 * grad,
             ( Vector( 2 ) << std::cos( x[0] ), std::sin( x[1] ) ).finished().normalized(),
             ( Vector( 2 ) << -1.0, 0.5 ).finished().normalized() };
  }
};

class DixonPriceFunction : public TestFunction
{
public:
  std::string name() const override { return "DixonPrice2D"; }
  std::string description() const override { return "Dixon-Price function (bowl-shaped)"; }
  int         dimension() const override { return 2; }
  Vector      solution() const override
  {
    Vector sol( 2 );
    sol[0] = std::pow( 2.0, -( std::pow( 2.0, 2 ) - 2.0 ) / std::pow( 2.0, 2 ) );
    sol[1] = sol[0] * sol[0];
    return sol;
  }

  Scalar value( const Vector & x ) const override
  {
    Scalar t1 = ( x[0] - 1.0 ) * ( x[0] - 1.0 );
    Scalar t2 = 2.0 * ( 2.0 * x[1] * x[1] - x[0] ) * ( 2.0 * x[1] * x[1] - x[0] );
    return t1 + t2;
  }

  void gradient( const Vector & x, Vector & grad ) const override
  {
    grad.resize( 2 );
    grad[0] = 2.0 * ( x[0] - 1.0 ) - 4.0 * ( 2.0 * x[1] * x[1] - x[0] );
    grad[1] = 8.0 * x[1] * ( 2.0 * x[1] * x[1] - x[0] );
  }

  std::vector<Vector> test_points() const override
  {
    return { ( Vector( 2 ) << 0.0, 0.0 ).finished(),
             ( Vector( 2 ) << 1.0, 1.0 ).finished(),
             ( Vector( 2 ) << -1.0, 0.5 ).finished(),
             ( Vector( 2 ) << 2.0, -1.0 ).finished() };
  }

  std::vector<Vector> test_directions( const Vector & x ) const override
  {
    Vector grad( 2 );
    gradient( x, grad );
    return { -grad,
             -0.4 * grad,
             ( Vector( 2 ) << x[1], -x[0] ).finished().normalized(),
             ( Vector( 2 ) << 1.0, 2.0 ).finished().normalized() };
  }
};

// ===========================================================================
// Test Problem Factory
// ===========================================================================

class TestProblemFactory
{
public:
  static std::unique_ptr<TestFunction> create( const std::string & name )
  {
    if ( name == "Quadratic1D" ) return std::make_unique<Quadratic1D>();
    if ( name == "Rosenbrock2D" ) return std::make_unique<Rosenbrock2D>();
    if ( name == "Beale2D" ) return std::make_unique<BealeFunction>();
    if ( name == "Exponential1D" ) return std::make_unique<Exponential1D>();
    if ( name == "PowellSingular4D" ) return std::make_unique<PowellSingular4D>();
    if ( name == "Trigonometric2D" ) return std::make_unique<TrigonometricFunction>();
    if ( name == "WoodFunction4D" ) return std::make_unique<WoodFunction>();
    if ( name == "ExtendedRosenbrock4D" ) return std::make_unique<ExtendedRosenbrock>( 4 );
    if ( name == "ExtendedRosenbrock6D" ) return std::make_unique<ExtendedRosenbrock>( 6 );
    if ( name == "Booth2D" ) return std::make_unique<BoothFunction>();
    if ( name == "Matyas2D" ) return std::make_unique<MatyasFunction>();
    if ( name == "ThreeHumpCamel2D" ) return std::make_unique<ThreeHumpCamel>();
    if ( name == "DixonPrice2D" ) return std::make_unique<DixonPriceFunction>();
    return nullptr;
  }

  static std::vector<std::string> available_problems()
  {
    return { "Quadratic1D",          "Rosenbrock2D",    "Beale2D",        "Exponential1D",
             "PowellSingular4D",     "Trigonometric2D", "WoodFunction4D", "ExtendedRosenbrock4D",
             "ExtendedRosenbrock6D", "Booth2D",         "Matyas2D",       "ThreeHumpCamel2D",
             "DixonPrice2D" };
  }
};

// ===========================================================================
// Algorithm Configuration and Management
// ===========================================================================

struct AlgorithmParams
{
  Scalar c1                   = 1e-4;
  Scalar c2                   = 0.9;
  Scalar step_reduce          = 0.5;
  Scalar step_expand          = 1.2;
  size_t max_iterations       = 50;
  Scalar alpha_max            = 10.0;
  Scalar alpha_min            = 1e-12;
  bool   use_quadratic_interp = true;
  Scalar extrapolation_factor = 2.2;
  Scalar delta                = 0.1;
  Scalar sigma                = 0.9;
};

struct LineSearchConfig
{
  std::string name;
  std::function<std::optional<std::tuple<Scalar, size_t>>(
    Scalar,
    Scalar,
    const Vector &,
    const Vector &,
    std::function<Scalar( const Vector &, Vector * )>,
    Scalar )>
                  algorithm;
  fmt::text_style color;
  std::string     description;
  AlgorithmParams params;
};

class AlgorithmManager
{
private:
  std::vector<LineSearchConfig> algorithms;

  LineSearchConfig create_armijo_config( const AlgorithmParams & /*p*/ )
  {
    return { "Armijo",
             [](
               Scalar                                            f0,
               Scalar                                            Df0,
               const Vector &                                    x,
               const Vector &                                    d,
               std::function<Scalar( const Vector &, Vector * )> callback,
               Scalar                                            alpha0 )
             {
               ArmijoLineSearch<Scalar> ls;
               return ls( f0, Df0, x, d, callback, alpha0 );
             },
             Style::ARMIDO_COLOR,
             "Simple Armijo backtracking line search",
             AlgorithmParams{} };
  }

  LineSearchConfig create_weak_wolfe_config( const AlgorithmParams & /*p*/ )
  {
    return { "WeakWolfe",
             [](
               Scalar                                            f0,
               Scalar                                            Df0,
               const Vector &                                    x,
               const Vector &                                    d,
               std::function<Scalar( const Vector &, Vector * )> callback,
               Scalar                                            alpha0 )
             {
               WeakWolfeLineSearch<Scalar> ls;
               return ls( f0, Df0, x, d, callback, alpha0 );
             },
             Style::WOLFE_COLOR,
             "Weak Wolfe conditions line search",
             AlgorithmParams{} };
  }

  LineSearchConfig create_strong_wolfe_config( const AlgorithmParams & /*p*/ )
  {
    return { "StrongWolfe",
             [](
               Scalar                                            f0,
               Scalar                                            Df0,
               const Vector &                                    x,
               const Vector &                                    d,
               std::function<Scalar( const Vector &, Vector * )> callback,
               Scalar                                            alpha0 )
             {
               StrongWolfeLineSearch<Scalar> ls;
               return ls( f0, Df0, x, d, callback, alpha0 );
             },
             Style::STRONGWOLFE_COLOR,
             "Strong Wolfe conditions line search",
             AlgorithmParams{} };
  }

  LineSearchConfig create_goldstein_config( const AlgorithmParams & /*p*/ )
  {
    return { "Goldstein",
             [](
               Scalar                                            f0,
               Scalar                                            Df0,
               const Vector &                                    x,
               const Vector &                                    d,
               std::function<Scalar( const Vector &, Vector * )> callback,
               Scalar                                            alpha0 )
             {
               GoldsteinLineSearch<Scalar> ls;
               return ls( f0, Df0, x, d, callback, alpha0 );
             },
             Style::GOLDSTEIN_COLOR,
             "Goldstein conditions line search",
             AlgorithmParams{} };
  }

  LineSearchConfig create_hager_zhang_config( const AlgorithmParams & /*p*/ )
  {
    return { "HagerZhang",
             [](
               Scalar                                            f0,
               Scalar                                            Df0,
               const Vector &                                    x,
               const Vector &                                    d,
               std::function<Scalar( const Vector &, Vector * )> callback,
               Scalar                                            alpha0 )
             {
               HagerZhangLineSearch<Scalar> ls;
               return ls( f0, Df0, x, d, callback, alpha0 );
             },
             Style::HAGERZHANG_COLOR,
             "Hager-Zhang robust line search",
             AlgorithmParams{} };
  }

  LineSearchConfig create_more_thuente_config( const AlgorithmParams & /*p*/ )
  {
    return { "MoreThuente",
             [](
               Scalar                                            f0,
               Scalar                                            Df0,
               const Vector &                                    x,
               const Vector &                                    d,
               std::function<Scalar( const Vector &, Vector * )> callback,
               Scalar                                            alpha0 )
             {
               MoreThuenteLineSearch<Scalar> ls;
               return ls( f0, Df0, x, d, callback, alpha0 );
             },
             Style::MORETHUENTE_COLOR,
             "More-Thuente high accuracy line search",
             AlgorithmParams{} };
  }

public:
  AlgorithmManager()
  {
    AlgorithmParams default_params;

    algorithms.push_back( create_armijo_config( default_params ) );
    algorithms.push_back( create_weak_wolfe_config( default_params ) );
    algorithms.push_back( create_strong_wolfe_config( default_params ) );
    algorithms.push_back( create_goldstein_config( default_params ) );
    algorithms.push_back( create_hager_zhang_config( default_params ) );
    algorithms.push_back( create_more_thuente_config( default_params ) );
  }

  const std::vector<LineSearchConfig> & get_algorithms() const { return algorithms; }

  std::vector<LineSearchConfig> get_algorithms( const std::vector<std::string> & names ) const
  {
    std::vector<LineSearchConfig> selected;
    for ( const auto & name : names )
    {
      auto it = std::find_if(
        algorithms.begin(),
        algorithms.end(),
        [&]( const LineSearchConfig & cfg ) { return cfg.name == name; } );
      if ( it != algorithms.end() ) selected.push_back( *it );
    }
    return selected;
  }

  std::vector<std::string> available_algorithms() const
  {
    std::vector<std::string> names;
    for ( const auto & algo : algorithms ) names.push_back( algo.name );
    return names;
  }
};

// ===========================================================================
// Test Result Structure and Statistics
// ===========================================================================

struct TestResult
{
  std::string algorithm;
  std::string test_function;
  std::string status;  // "SUCCESS", "FAILED", "WARNING"
  Scalar      initial_f;
  Scalar      final_f;
  Scalar      step_length;
  size_t      iterations;
  size_t      function_evals;
  size_t      gradient_evals;
  double      cpu_time_mus;

  std::string start_point_str;
  std::string direction_str;

  Scalar reduction() const
  {
    if ( std::abs( initial_f ) < 1e-15 ) return 0.0;
    return ( initial_f - final_f ) / std::abs( initial_f );
  }

  std::string status_symbol() const
  {
    if ( status == "SUCCESS" ) return Unicode::CHECK;
    if ( status == "WARNING" ) return Unicode::WARNING;
    return Unicode::CROSS;
  }

  fmt::text_style status_style() const
  {
    if ( status == "SUCCESS" ) return Style::SUCCESS;
    if ( status == "WARNING" ) return Style::WARNING;
    return Style::ERROR;
  }
};

class Statistics
{
private:
  const std::vector<TestResult> *       m_results             = nullptr;
  const std::vector<LineSearchConfig> * m_selected_algorithms = nullptr;

public:
  size_t total_tests = 0;
  size_t successful  = 0;
  size_t warnings    = 0;
  size_t failures    = 0;

  struct PerAlgorithmStats
  {
    std::vector<Scalar> step_lengths;
    std::vector<size_t> function_evals;
    std::vector<double> execution_times;
    std::vector<Scalar> reductions;
    std::vector<Scalar> final_values;
  };

  std::map<std::string, PerAlgorithmStats> per_algorithm;
  std::map<std::string, PerAlgorithmStats> per_function;

  void analyze( const std::vector<TestResult> & results, const std::vector<LineSearchConfig> & selected_algorithms )
  {
    m_results             = &results;
    m_selected_algorithms = &selected_algorithms;

    total_tests = results.size();
    successful =
      std::count_if( results.begin(), results.end(), []( const TestResult & r ) { return r.status == "SUCCESS"; } );
    warnings =
      std::count_if( results.begin(), results.end(), []( const TestResult & r ) { return r.status == "WARNING"; } );
    failures = total_tests - successful - warnings;

    for ( const auto & result : results )
    {
      per_algorithm[result.algorithm].step_lengths.push_back( result.step_length );
      per_algorithm[result.algorithm].function_evals.push_back( result.function_evals + result.gradient_evals );
      per_algorithm[result.algorithm].execution_times.push_back( result.cpu_time_mus );
      per_algorithm[result.algorithm].reductions.push_back( result.reduction() );
      per_algorithm[result.algorithm].final_values.push_back( result.final_f );

      per_function[result.test_function].step_lengths.push_back( result.step_length );
      per_function[result.test_function].function_evals.push_back( result.function_evals + result.gradient_evals );
      per_function[result.test_function].execution_times.push_back( result.cpu_time_mus );
      per_function[result.test_function].reductions.push_back( result.reduction() );
    }
  }

  void print_detailed_report() const
  {
    if ( !m_results || !m_selected_algorithms )
    {
      fmt::print( Style::ERROR, "Statistics not properly initialized!\n" );
      return;
    }

    const auto & results             = *m_results;
    const auto & selected_algorithms = *m_selected_algorithms;

    fmt::print( "\n" );
    fmt::print( Style::TITLE, "╔══════════════════════════════════════════════════════════╗\n" );
    fmt::print( Style::TITLE, "║           DETAILED STATISTICAL ANALYSIS                  ║\n" );
    fmt::print( Style::TITLE, "╚══════════════════════════════════════════════════════════╝\n\n" );

    // ===========================================================================
    // 1. PER ALGORITHM STATISTICS
    // ===========================================================================
    fmt::print( Style::HEADER, "┌─────────────────────────────────────────────────────────────┐\n" );
    fmt::print( Style::HEADER, "│                  PER ALGORITHM STATISTICS                   │\n" );
    fmt::print( Style::HEADER, "└─────────────────────────────────────────────────────────────┘\n\n" );

    // Trova i migliori valori per ogni metrica
    std::map<std::string, double> algo_avg_reductions;
    std::map<std::string, double> algo_avg_evals;
    std::map<std::string, double> algo_avg_times;

    for ( const auto & [algo, stats] : per_algorithm )
    {
      if ( stats.reductions.empty() ) continue;
      algo_avg_reductions[algo] = std::accumulate( stats.reductions.begin(), stats.reductions.end(), 0.0 ) /
                                  static_cast<double>( stats.reductions.size() );
      algo_avg_evals[algo] = std::accumulate( stats.function_evals.begin(), stats.function_evals.end(), 0.0 ) /
                             static_cast<double>( stats.function_evals.size() );
      algo_avg_times[algo] = std::accumulate( stats.execution_times.begin(), stats.execution_times.end(), 0.0 ) /
                             static_cast<double>( stats.execution_times.size() );
    }

    // Trova i migliori (massimo per riduzione, minimo per eval e time)
    std::string best_reduction_algo, best_eval_algo, best_time_algo;
    double      max_reduction = -1.0, min_eval = 1e9, min_time = 1e9;

    for ( const auto & [algo, reduction] : algo_avg_reductions )
    {
      if ( reduction > max_reduction )
      {
        max_reduction       = reduction;
        best_reduction_algo = algo;
      }
      if ( algo_avg_evals[algo] < min_eval )
      {
        min_eval       = algo_avg_evals[algo];
        best_eval_algo = algo;
      }
      if ( algo_avg_times[algo] < min_time )
      {
        min_time       = algo_avg_times[algo];
        best_time_algo = algo;
      }
    }

    // Tabella principale per algoritmi
    TablePrinter algo_table(
      { { "Algorithm", 12 }, { "Step Length", 40 }, { "Evaluations", 15 }, { "Time", 12 }, { "Reduction %", 15 } },
      { Style::HEADER, Style::HEADER, Style::HEADER, Style::HEADER, Style::HEADER } );

    algo_table.print_header();

    for ( const auto & [algo, stats] : per_algorithm )
    {
      if ( stats.step_lengths.empty() ) continue;

      auto avg_step = std::accumulate( stats.step_lengths.begin(), stats.step_lengths.end(), 0.0 ) /
                      static_cast<double>( stats.step_lengths.size() );
      auto min_step = *std::min_element( stats.step_lengths.begin(), stats.step_lengths.end() );
      auto max_step = *std::max_element( stats.step_lengths.begin(), stats.step_lengths.end() );

      auto avg_evals     = algo_avg_evals[algo];
      auto avg_time      = algo_avg_times[algo];
      auto avg_reduction = algo_avg_reductions[algo];

      // Determina stili per ogni colonna
      std::vector<fmt::text_style> row_styles( 5, Style::NONE );

      // Colonna 0: Nome algoritmo con colore specifico
      for ( const auto & algo_config : selected_algorithms )
      {
        if ( algo_config.name == algo )
        {
          row_styles[0] = algo_config.color;
          break;
        }
      }

      // Colonna 1: Step length (verde per step vicini a 1, giallo per estremi)
      if ( avg_step > 0.8 && avg_step < 1.2 )
        row_styles[1] = Style::SUCCESS;
      else if ( avg_step > 2.0 || avg_step < 0.2 )
        row_styles[1] = Style::WARNING;

      // Colonna 2: Valutazioni (verde per poche, rosso per molte)
      row_styles[2] = ( algo == best_eval_algo ) ? Style::SUCCESS
                      : ( avg_evals > 1000 )     ? Style::ERROR
                      : ( avg_evals > 500 )      ? Style::WARNING
                                                 : Style::NONE;

      // Colonna 3: Tempo (verde per veloce, rosso per lento)
      row_styles[3] = ( algo == best_time_algo ) ? Style::SUCCESS
                      : ( avg_time > 0.2 )       ? Style::ERROR
                      : ( avg_time > 0.1 )       ? Style::WARNING
                                                 : Style::NONE;

      // Colonna 4: Riduzione (verde per alta, rosso per bassa)
      row_styles[4] = ( algo == best_reduction_algo ) ? Style::SUCCESS
                      : ( avg_reduction > 0.5 )       ? Style::SUCCESS
                      : ( avg_reduction > 0.3 )       ? Style::WARNING
                                                      : Style::ERROR;

      // Formatta la riga
      std::string step_str = fmt::format( "min:{:.3f} avg:{:.3f} max:{:.3f}", min_step, avg_step, max_step );

      algo_table.print_row(
        { algo,
          step_str,
          fmt::format( "{:.2g}", avg_evals ),
          fmt::format( "{:.3g} mus", avg_time ),
          fmt::format( "{:.2g}%", 100 * avg_reduction ) },
        row_styles );
    }

    algo_table.print_footer();

    // Riepilogo dei migliori algoritmi
    fmt::print( Style::BOLD, "\n{} BEST PERFORMERS SUMMARY:\n", Unicode::ARROW_RIGHT );
    fmt::print( "  {} ", Unicode::ARROW_UP );
    fmt::print( Style::SUCCESS, "Highest Reduction: {} ({:.1f}%)\n", best_reduction_algo, 100 * max_reduction );

    fmt::print( "  {} ", Unicode::ARROW_DOWN );
    fmt::print( Style::CYAN, "Fewest Evaluations: {} ({:.1f} evals)\n", best_eval_algo, min_eval );

    fmt::print( "  {} ", Unicode::ARROW_DOWN );
    fmt::print( Style::CYAN, "Fastest Execution: {} ({:.2f} ms)\n", best_time_algo, min_time );

    // ===========================================================================
    // 2. PER FUNCTION STATISTICS
    // ===========================================================================
    fmt::print( "\n\n" );
    fmt::print( Style::HEADER, "┌─────────────────────────────────────────────────────────────┐\n" );
    fmt::print( Style::HEADER, "│                PER FUNCTION DIFFICULTITY                    │\n" );
    fmt::print( Style::HEADER, "└─────────────────────────────────────────────────────────────┘\n\n" );

    // Trova le funzioni più facili/difficili
    std::vector<std::pair<std::string, double>> func_reductions;
    std::vector<std::pair<std::string, double>> func_evals;

    for ( const auto & [func, stats] : per_function )
    {
      if ( stats.reductions.empty() ) continue;
      auto avg_reduction = std::accumulate( stats.reductions.begin(), stats.reductions.end(), 0.0 ) /
                           static_cast<double>( stats.reductions.size() );
      auto avg_eval = std::accumulate( stats.function_evals.begin(), stats.function_evals.end(), 0.0 ) /
                      static_cast<double>( stats.function_evals.size() );
      func_reductions.emplace_back( func, avg_reduction );
      func_evals.emplace_back( func, avg_eval );
    }

    // Ordina per difficoltà (riduzione bassa = difficile)
    std::sort(
      func_reductions.begin(),
      func_reductions.end(),
      []( const auto & a, const auto & b ) { return a.second < b.second; } );
    std::sort(
      func_evals.begin(),
      func_evals.end(),
      []( const auto & a, const auto & b ) { return a.second > b.second; } );

    // Tabella per funzioni
    TablePrinter func_table(
      { { "Function", 22 }, { "Reduction %", 40 }, { "Avg Evals", 15 }, { "Difficulty", 15 } },
      { Style::HEADER, Style::HEADER, Style::HEADER, Style::HEADER } );

    func_table.print_header();

    for ( const auto & [func, stats] : per_function )
    {
      if ( stats.reductions.empty() ) continue;

      auto avg_reduction = std::accumulate( stats.reductions.begin(), stats.reductions.end(), 0.0 ) /
                           static_cast<double>( stats.reductions.size() );
      auto min_reduction = *std::min_element( stats.reductions.begin(), stats.reductions.end() );
      auto max_reduction = *std::max_element( stats.reductions.begin(), stats.reductions.end() );
      auto avg_evals     = std::accumulate( stats.function_evals.begin(), stats.function_evals.end(), 0.0 ) /
                       static_cast<double>( stats.function_evals.size() );

      // Determina difficoltà
      std::string     difficulty;
      fmt::text_style diff_style = Style::NONE;

      if ( avg_reduction > 0.7 )
      {
        difficulty = "EASY";
        diff_style = Style::SUCCESS;
      }
      else if ( avg_reduction > 0.4 )
      {
        difficulty = "MEDIUM";
        diff_style = Style::WARNING;
      }
      else if ( avg_reduction > 0.2 )
      {
        difficulty = "HARD";
        diff_style = Style::ERROR;
      }
      else
      {
        difficulty = "VERY HARD";
        diff_style = Style::ERROR | fmt::emphasis::bold;
      }

      // Stile per riduzione (colore basato sulla percentuale)
      fmt::text_style reduction_style = Style::NONE;
      if ( avg_reduction > 0.6 )
        reduction_style = Style::SUCCESS;
      else if ( avg_reduction > 0.3 )
        reduction_style = Style::WARNING;
      else
        reduction_style = Style::ERROR;

      // Stile per valutazioni (colore basato sul numero)
      fmt::text_style eval_style = Style::NONE;
      if ( avg_evals < 400 )
        eval_style = Style::SUCCESS;
      else if ( avg_evals < 800 )
        eval_style = Style::WARNING;
      else
        eval_style = Style::ERROR;

      std::string reduction_str = fmt::format(
        "min:{:.1f}% avg:{:.1f}% max:{:.1f}%",
        100 * min_reduction,
        100 * avg_reduction,
        100 * max_reduction );

      func_table.print_row(
        { func, reduction_str, fmt::format( "{:.1f}", avg_evals ), difficulty },
        { Style::NONE, reduction_style, eval_style, diff_style } );
    }

    func_table.print_footer();

    // Riepilogo difficoltà funzioni
    fmt::print( Style::BOLD, "\n{} FUNCTION DIFFICULTY CLASSIFICATION:\n", Unicode::ARROW_RIGHT );

    if ( !func_reductions.empty() )
    {
      fmt::print( "  {} ", Unicode::ARROW_UP );
      fmt::print(
        Style::SUCCESS,
        "Easiest: {} ({:.1f}% avg reduction)\n",
        func_reductions.back().first,
        100 * func_reductions.back().second );

      fmt::print( "  {} ", Unicode::ARROW_DOWN );
      fmt::print(
        Style::ERROR,
        "Hardest: {} ({:.1f}% avg reduction)\n",
        func_reductions.front().first,
        100 * func_reductions.front().second );
    }

    if ( !func_evals.empty() )
    {
      fmt::print( "  {} ", Unicode::ARROW_DOWN );
      fmt::print(
        Style::CYAN,
        "Most Expensive: {} ({:.1f} avg evals)\n",
        func_evals.front().first,
        func_evals.front().second );

      fmt::print( "  {} ", Unicode::ARROW_UP );
      fmt::print( Style::CYAN, "Cheapest: {} ({:.1f} avg evals)\n", func_evals.back().first, func_evals.back().second );
    }

    // ===========================================================================
    // 3. PERFORMANCE HEATMAP (matrice algoritmo x funzione)
    // ===========================================================================
    fmt::print( "\n\n" );
    fmt::print( Style::HEADER, "┌─────────────────────────────────────────────────────────────┐\n" );
    fmt::print( Style::HEADER, "│             ALGORITHM × FUNCTION PERFORMANCE                │\n" );
    fmt::print( Style::HEADER, "└─────────────────────────────────────────────────────────────┘\n\n" );

    // Raccoglie dati per heatmap
    std::vector<std::string>                              algorithms;
    std::vector<std::string>                              functions;
    std::map<std::pair<std::string, std::string>, double> reduction_map;

    for ( const auto & [algo, _] : per_algorithm ) algorithms.push_back( algo );
    for ( const auto & [func, _] : per_function ) functions.push_back( func );

    // Popola la mappa con dati dai risultati
    for ( const auto & result : results )
    {
      auto key = std::make_pair( result.algorithm, result.test_function );
      if ( reduction_map.find( key ) == reduction_map.end() )
        reduction_map[key] = result.reduction();
      else
        reduction_map[key] = std::max( reduction_map[key], result.reduction() );
    }

    // Crea intestazione della tabella
    fmt::print( "             " );
    for ( const auto & func : functions )
    {
      // Abbrevia nomi lunghi
      std::string display_name = func;
      if ( display_name.length() > 13 ) display_name = display_name.substr( 0, 13 ) + ".";
      fmt::print( "{:^15}", display_name );
    }
    fmt::print( "\n" );

    fmt::print( "             " );
    for ( size_t i = 0; i < functions.size(); ++i ) fmt::print( "{:─^{}}", "", 15 );
    fmt::print( "\n" );

    // Stampa righe per ogni algoritmo
    for ( const auto & algo : algorithms )
    {
      // Trova il colore per questo algoritmo
      fmt::text_style algo_style = Style::NONE;
      for ( const auto & algo_config : selected_algorithms )
      {
        if ( algo_config.name == algo )
        {
          algo_style = algo_config.color;
          break;
        }
      }

      // Abbrevia nome algoritmo se troppo lungo
      std::string algo_display = algo;
      if ( algo_display.length() > 13 ) algo_display = algo_display.substr( 0, 13 ) + ".";

      fmt::print( algo_style, "{:<13}", algo_display );

      for ( const auto & func : functions )
      {
        auto   key       = std::make_pair( algo, func );
        double reduction = ( reduction_map.find( key ) != reduction_map.end() ) ? reduction_map[key] : 0.0;

        // Determina colore in base alla riduzione
        fmt::text_style cell_style = Style::NONE;
        if ( reduction > 0.7 )
          cell_style = Style::SUCCESS | fmt::emphasis::bold;
        else if ( reduction > 0.4 )
          cell_style = Style::WARNING;
        else if ( reduction > 0.1 )
          cell_style = Style::ERROR;
        else
          cell_style = Style::DIM;

        fmt::print( cell_style, "{:^15.2f}", 100 * reduction );
      }
      fmt::print( "\n" );
    }

    // Legenda per la heatmap
    fmt::print( Style::DIM, "\n{} Heatmap Legend: ", Unicode::INFO );
    fmt::print( Style::SUCCESS, "█>70% " );
    fmt::print( Style::WARNING, "█40-70% " );
    fmt::print( Style::ERROR, "█10-40% " );
    fmt::print( Style::DIM, "█<10%\n" );
  }
};

// ===========================================================================
// Reduction Comparison Table
// ===========================================================================

class ReductionComparison
{
private:
  struct ReductionStats
  {
    std::string algorithm;
    std::string function;
    Scalar      avg_reduction;
    Scalar      min_reduction;
    Scalar      max_reduction;
    size_t      success_count;
    size_t      total_count;
  };

  std::vector<ReductionStats> stats;

public:
  void analyze( const std::vector<TestResult> & results )
  {
    std::map<std::pair<std::string, std::string>, std::vector<Scalar>> reductions;
    std::map<std::pair<std::string, std::string>, size_t>              success_counts;
    std::map<std::pair<std::string, std::string>, size_t>              total_counts;

    for ( const auto & result : results )
    {
      auto key = std::make_pair( result.algorithm, result.test_function );
      reductions[key].push_back( result.reduction() );
      total_counts[key]++;
      if ( result.status == "SUCCESS" ) success_counts[key]++;
    }

    for ( const auto & [key, reds] : reductions )
    {
      if ( reds.empty() ) continue;

      Scalar avg = std::accumulate( reds.begin(), reds.end(), 0.0 ) / static_cast<Scalar>( reds.size() );
      Scalar min = *std::min_element( reds.begin(), reds.end() );
      Scalar max = *std::max_element( reds.begin(), reds.end() );

      stats.push_back(
        { key.first,   // algorithm
          key.second,  // function
          avg,
          min,
          max,
          success_counts[key],
          total_counts[key] } );
    }

    // Sort by average reduction descending
    std::sort(
      stats.begin(),
      stats.end(),
      []( const ReductionStats & a, const ReductionStats & b ) { return a.avg_reduction > b.avg_reduction; } );
  }

  void print_comparison() const
  {
    fmt::print( Style::TITLE, "\nReduction Comparison Across Algorithms\n" );
    fmt::print( Style::DIM, "{:─^{}}\n", "", 80 );

    TablePrinter comparison_table(
      { { "Algorithm", 12 }, { "Function", 20 }, { "Avg Reduction", 15 }, { "Min-Max", 15 }, { "Success Rate", 15 } } );

    comparison_table.print_header();

    for ( const auto & stat : stats )
    {
      Scalar success_rate =
        100.0 * static_cast<Scalar>( stat.success_count ) / static_cast<Scalar>( stat.total_count );

      fmt::text_style reduction_style = Style::NONE;
      if ( stat.avg_reduction > 0.8 )
        reduction_style = Style::SUCCESS;
      else if ( stat.avg_reduction > 0.5 )
        reduction_style = Style::WARNING;
      else
        reduction_style = Style::ERROR;

      fmt::text_style success_style = Style::NONE;
      if ( success_rate > 90 )
        success_style = Style::SUCCESS;
      else if ( success_rate > 70 )
        success_style = Style::WARNING;
      else
        success_style = Style::ERROR;

      comparison_table.print_row(
        { stat.algorithm,
          stat.function,
          fmt::format( "{:.1f}%", 100 * stat.avg_reduction ),
          fmt::format( "[{:.1f}%, {:.1f}%]", 100 * stat.min_reduction, 100 * stat.max_reduction ),
          fmt::format( "{:.1f}%", success_rate ) },
        { Style::NONE, Style::NONE, reduction_style, Style::NONE, success_style } );
    }

    comparison_table.print_footer();

    // Print summary
    fmt::print( Style::BOLD, "\nTop Performers by Average Reduction:\n" );

    std::map<std::string, std::vector<Scalar>> algo_reductions;
    for ( const auto & stat : stats ) { algo_reductions[stat.algorithm].push_back( stat.avg_reduction ); }

    std::vector<std::pair<std::string, Scalar>> avg_by_algo;
    for ( const auto & [algo, reds] : algo_reductions )
    {
      Scalar avg = std::accumulate( reds.begin(), reds.end(), 0.0 ) / static_cast<Scalar>( reds.size() );
      avg_by_algo.emplace_back( algo, avg );
    }

    std::sort(
      avg_by_algo.begin(),
      avg_by_algo.end(),
      []( const auto & a, const auto & b ) { return a.second > b.second; } );

    for ( size_t i = 0; i < std::min( size_t( 3 ), avg_by_algo.size() ); ++i )
    {
      fmt::print(
        "  {} {}: {:.1f}% average reduction\n",
        Unicode::ARROW_RIGHT,
        avg_by_algo[i].first,
        100 * avg_by_algo[i].second );
    }
  }
};

// ===========================================================================
// Test Runner with Enhanced Features
// ===========================================================================

struct TestParameters
{
  std::vector<std::string> problems;
  std::vector<std::string> algorithms;
  bool                     quick_mode            = false;
  bool                     verbose               = false;
  bool                     export_csv            = true;
  std::string              csv_filename          = "line_search_results.csv";
  size_t                   max_tests_per_problem = 100;
};

class LineSearchTester
{
private:
  AlgorithmManager    algo_manager;
  ReductionComparison reduction_comp;
  TestParameters      params;

  std::vector<TestResult> results;
  Statistics              stats;

  std::vector<LineSearchConfig>              selected_algorithms;
  std::vector<std::unique_ptr<TestFunction>> test_functions;

public:
  LineSearchTester( const TestParameters & p = {} ) : params( p ) { initialize(); }

  void initialize()
  {
    if ( params.problems.empty() )
    {
      auto all_problems = TestProblemFactory::available_problems();
      for ( const auto & name : all_problems ) test_functions.push_back( TestProblemFactory::create( name ) );
    }
    else
    {
      for ( const auto & name : params.problems )
      {
        auto func = TestProblemFactory::create( name );
        if ( func ) test_functions.push_back( std::move( func ) );
      }
    }

    if ( params.algorithms.empty() ) { selected_algorithms = algo_manager.get_algorithms(); }
    else
    {
      selected_algorithms = algo_manager.get_algorithms( params.algorithms );
    }
  }

  void run_all_tests()
  {
    print_header();

    auto total_start = std::chrono::high_resolution_clock::now();

    size_t test_counter = 0;
    for ( const auto & func : test_functions )
    {
      if ( test_counter >= params.max_tests_per_problem ) break;
      run_tests_for_function( *func, test_counter );
    }

    auto   total_end  = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>( total_end - total_start ).count();

    stats.analyze( results, selected_algorithms );
    reduction_comp.analyze( results );
    print_summary( total_time );
    print_detailed_results();
    print_algorithm_comparison();
    reduction_comp.print_comparison();
    stats.print_detailed_report();

    if ( params.export_csv ) export_results_to_csv( params.csv_filename );
  }

  void run_quick_test()
  {
    fmt::print( Style::HEADER, "Running Quick Test...\n" );

    Vector x( 2 );
    x << 1.0, 2.0;
    Vector d( 2 );
    d << -1.0, -1.0;

    auto quadratic = []( const Vector & x, Vector * grad ) -> Scalar
    {
      Scalar f = ( x[0] - 1.0 ) * ( x[0] - 1.0 ) + ( x[1] - 2.0 ) * ( x[1] - 2.0 );
      if ( grad )
      {
        grad->resize( 2 );
        ( *grad )[0] = 2.0 * ( x[0] - 1.0 );
        ( *grad )[1] = 2.0 * ( x[1] - 2.0 );
      }
      return f;
    };

    Scalar f0 = quadratic( x, nullptr );
    Vector grad0( 2 );
    quadratic( x, &grad0 );
    Scalar Df0 = grad0.dot( d );

    fmt::print( "Testing from point: [{:.1f}, {:.1f}]\n", x[0], x[1] );
    fmt::print( "Initial f: {:.4f}, Directional derivative: {:.4f}\n\n", f0, Df0 );

    for ( const auto & algo_config : selected_algorithms )
    {
      auto result = algo_config.algorithm( f0, Df0, x, d, quadratic, 1.0 );
      if ( result )
      {
        auto [alpha, iters] = result.value();
        fmt::print( algo_config.color, "{}: α = {:.4f}, iterations = {}\n", algo_config.name, alpha, iters );
      }
      else
      {
        fmt::print( Style::ERROR, "{} failed\n", algo_config.name );
      }
    }
  }

  const std::vector<TestResult> & get_results() const { return results; }
  const Statistics &              get_statistics() const { return stats; }

private:
  void print_header()
  {
    fmt::print(
      Style::TITLE,
      "┌{0:─^{2}}┐\n"
      "│{1:^{2}}│\n"
      "└{0:─^{2}}┘\n\n",
      "",
      "LINE SEARCH ALGORITHMS - ENHANCED TEST SUITE v3.5",
      70 );

    fmt::print(
      Style::COMMENT,
      "Testing {} functions with {} algorithms\n",
      test_functions.size(),
      selected_algorithms.size() );
    fmt::print( Style::COMMENT, "{:═^{}}\n\n", "", 80 );
  }

  void run_tests_for_function( const TestFunction & func, size_t & test_counter )
  {
    if ( params.verbose ) fmt::print( Style::HEADER, "▶ {} - {}\n", func.name(), func.description() );

    auto test_points = func.test_points();
    for ( const auto & start_point : test_points )
    {
      if ( test_counter >= params.max_tests_per_problem ) return;

      auto directions = func.test_directions( start_point );
      for ( const auto & direction : directions )
      {
        if ( test_counter >= params.max_tests_per_problem ) return;

        run_single_test( func, start_point, direction );
        test_counter++;

        if ( params.verbose && test_counter % 10 == 0 )
          fmt::print( Style::DIM, "  Completed {} tests...\n", test_counter );
      }
    }
  }

  void run_single_test( const TestFunction & func, const Vector & start_point, const Vector & direction )
  {
    size_t func_evals = 0;
    size_t grad_evals = 0;

    auto callback = [&]( const Vector & x, Vector * grad ) -> Scalar
    {
      if ( grad )
      {
        grad_evals++;
        func.gradient( x, *grad );
      }
      func_evals++;
      return func.value( x );
    };

    Scalar f0 = func.value( start_point );
    Vector grad0( func.dimension() );
    func.gradient( start_point, grad0 );
    Scalar Df0 = grad0.dot( direction );

    if ( Df0 >= -1e-12 ) return;

    for ( const auto & algo_config : selected_algorithms )
    {
      size_t start_func_evals = func_evals;
      size_t start_grad_evals = grad_evals;

      TicToc tm;
      tm.tic();

      // Fix: remove unused variable warning by not declaring result
      for ( int i = 0; i < 99; ++i )
      {
        volatile auto dummy_result = algo_config.algorithm( f0, Df0, start_point, direction, callback, 1.0 );
        (void) dummy_result;
      }
      auto result = algo_config.algorithm( f0, Df0, start_point, direction, callback, 1.0 );

      tm.toc();

      TestResult test_result;
      test_result.algorithm      = algo_config.name;
      test_result.test_function  = func.name();
      test_result.function_evals = func_evals - start_func_evals;
      test_result.gradient_evals = grad_evals - start_grad_evals;
      test_result.cpu_time_mus   = tm.elapsed_mus() / 100;
      test_result.initial_f      = f0;

      std::stringstream ss;
      ss << std::fixed << std::setprecision( 3 );
      ss << "[";
      for ( int i = 0; i < start_point.size(); ++i )
      {
        if ( i > 0 ) ss << ", ";
        ss << start_point[i];
      }
      ss << "]";
      test_result.start_point_str = ss.str();

      ss.str( "" );
      ss << "[";
      for ( int i = 0; i < direction.size(); ++i )
      {
        if ( i > 0 ) ss << ", ";
        ss << direction[i];
      }
      ss << "]";
      test_result.direction_str = ss.str();

      if ( result.has_value() )
      {
        auto [alpha, iters] = result.value();
        Vector new_point    = start_point + alpha * direction;

        test_result.step_length = alpha;
        test_result.iterations  = iters;
        test_result.final_f     = func.value( new_point );

        if ( test_result.final_f <= f0 - 1e-10 * std::abs( f0 ) ) { test_result.status = "SUCCESS"; }
        else
        {
          test_result.status = "WARNING";
        }
      }
      else
      {
        test_result.status      = "FAILED";
        test_result.step_length = 0.0;
        test_result.iterations  = 0;
        test_result.final_f     = f0;
      }

      results.push_back( test_result );
    }
  }

  void print_summary( double total_time )
  {
    fmt::print(
      Style::TITLE,
      "┌{0:─^{2}}┐\n"
      "│{1:^{2}}│\n"
      "└{0:─^{2}}┘\n\n",
      "",
      "TEST SUMMARY",
      60 );

    fmt::print( Style::BOLD, "Overall Statistics:\n" );
    fmt::print( "  Total tests run:      {}\n", stats.total_tests );
    fmt::print(
      Style::SUCCESS,
      "  Successful tests:     {} ({:.1f}%)\n",
      stats.successful,
      100.0 * static_cast<double>( stats.successful ) / static_cast<double>( stats.total_tests ) );
    fmt::print(
      Style::WARNING,
      "  Warning tests:        {} ({:.1f}%)\n",
      stats.warnings,
      100.0 * static_cast<double>( stats.warnings ) / static_cast<double>( stats.total_tests ) );
    fmt::print(
      Style::ERROR,
      "  Failed tests:         {} ({:.1f}%)\n",
      stats.failures,
      100.0 * static_cast<double>( stats.failures ) / static_cast<double>( stats.total_tests ) );
    fmt::print( "  Total execution time: {:.3f} seconds\n\n", total_time );

    fmt::print( Style::BOLD, "Algorithm Performance:\n" );

    TablePrinter summary_table(
      { { "Algorithm", 12 },
        { "Tests", 8 },
        { "Success", 12 },
        { "Success %", 10 },
        { "Avg Evals", 12 },
        { "Avg Time", 12 },
        { "Avg Reduction", 15 } },
      { Style::HEADER, Style::HEADER, Style::HEADER, Style::HEADER, Style::HEADER, Style::HEADER, Style::HEADER } );

    summary_table.print_header();

    for ( const auto & [algo_name, algo_stats] : stats.per_algorithm )
    {
      size_t total = algo_stats.step_lengths.size();
      if ( total == 0 ) continue;

      size_t success = 0;
      size_t warning = 0;
      for ( const auto & res : results )
      {
        if ( res.algorithm == algo_name )
        {
          if ( res.status == "SUCCESS" ) success++;
          if ( res.status == "WARNING" ) warning++;
        }
      }

      double total_count = static_cast<double>( total );
      double avg_evals = std::accumulate( algo_stats.function_evals.begin(), algo_stats.function_evals.end(), 0.0 ) /
                         total_count;
      double avg_time = std::accumulate( algo_stats.execution_times.begin(), algo_stats.execution_times.end(), 0.0 ) /
                        total_count;
      double avg_reduction =
        std::accumulate( algo_stats.reductions.begin(), algo_stats.reductions.end(), 0.0 ) / total_count;

      double success_rate =
        100.0 * ( static_cast<double>( success ) + 0.5 * static_cast<double>( warning ) ) / total_count;

      std::string success_str = fmt::format( "{}{}", success, Unicode::CHECK );
      if ( warning > 0 ) success_str += fmt::format( " {}{}", warning, Unicode::WARNING );

      std::vector<fmt::text_style> row_styles( 7, Style::NONE );

      for ( const auto & algo_config : selected_algorithms )
      {
        if ( algo_config.name == algo_name )
        {
          row_styles[0] = algo_config.color;
          break;
        }
      }

      if ( success_rate >= 90 )
        row_styles[3] = Style::SUCCESS;
      else if ( success_rate >= 70 )
        row_styles[3] = Style::WARNING;
      else
        row_styles[3] = Style::ERROR;

      if ( avg_reduction >= 0.7 )
        row_styles[6] = Style::SUCCESS;
      else if ( avg_reduction >= 0.4 )
        row_styles[6] = Style::WARNING;
      else
        row_styles[6] = Style::ERROR;

      summary_table.print_row(
        { algo_name,
          fmt::format( "{}", total ),
          success_str,
          fmt::format( "{:.2g}%", success_rate ),
          fmt::format( "{:.2g}", avg_evals ),
          fmt::format( "{:.3g} mus", avg_time ),
          fmt::format( "{:.2g}%", 100 * avg_reduction ) },
        row_styles );
    }

    summary_table.print_footer();
    fmt::print( "\n" );
  }

  void print_detailed_results()
  {
    fmt::print(
      Style::TITLE,
      "┌{0:─^{2}}┐\n"
      "│{1:^{2}}│\n"
      "└{0:─^{2}}┘\n\n",
      "",
      "DETAILED RESULTS",
      60 );

    std::map<std::string, std::vector<TestResult>> func_results;
    for ( const auto & res : results ) func_results[res.test_function].push_back( res );

    for ( const auto & [func_name, func_res] : func_results )
    {
      fmt::print( Style::HEADER, "Function: {}\n", func_name );
      fmt::print( Style::DIM, "{:─^{}}\n", "", 60 );

      TablePrinter detail_table(
        { { "Algorithm", 12 },
          { "Status", 10 },
          { "Step", 10 },
          { "Reduction", 12 },
          { "Evals", 10 },
          { "Time", 10 } } );

      detail_table.print_header();

      std::map<std::string, std::vector<TestResult>> algo_groups;
      for ( const auto & res : func_res ) algo_groups[res.algorithm].push_back( res );

      for ( const auto & [algo_name, algo_res] : algo_groups )
      {
        size_t success = std::count_if(
          algo_res.begin(),
          algo_res.end(),
          []( const TestResult & r ) { return r.status == "SUCCESS"; } );
        size_t total = algo_res.size();
        double total_count = static_cast<double>( total );

        double avg_step = 0.0, avg_reduction = 0.0, avg_evals = 0.0, avg_time = 0.0;
        for ( const auto & res : algo_res )
        {
          avg_step += res.step_length;
          avg_reduction += res.reduction();
          avg_evals += static_cast<double>( res.function_evals ) + static_cast<double>( res.gradient_evals );
          avg_time += res.cpu_time_mus;
        }
        avg_step /= total_count;
        avg_reduction /= total_count;
        avg_evals /= total_count;
        avg_time /= total_count;

        std::string     status_str;
        fmt::text_style status_style;
        if ( success == total )
        {
          status_str   = fmt::format( "{}{}/{}", Unicode::CHECK, success, total );
          status_style = Style::SUCCESS;
        }
        else if ( success == 0 )
        {
          status_str   = fmt::format( "{}{}/{}", Unicode::CROSS, success, total );
          status_style = Style::ERROR;
        }
        else
        {
          status_str   = fmt::format( "{}{}/{}", Unicode::WARNING, success, total );
          status_style = Style::WARNING;
        }

        std::vector<fmt::text_style> row_styles( 6, Style::NONE );

        for ( const auto & algo_config : selected_algorithms )
        {
          if ( algo_config.name == algo_name )
          {
            row_styles[0] = algo_config.color;
            break;
          }
        }

        row_styles[1] = status_style;
        row_styles[3] = ( avg_reduction > 0.5 )   ? Style::SUCCESS
                        : ( avg_reduction > 0.2 ) ? Style::WARNING
                                                  : Style::ERROR;

        detail_table.print_row(
          { algo_name,
            status_str,
            fmt::format( "{:.3g}", avg_step ),
            fmt::format( "{:.2g}%", 100.0 * avg_reduction ),
            fmt::format( "{:.2g}", avg_evals ),
            fmt::format( "{:.3g} mus", avg_time ) },
          row_styles );
      }

      detail_table.print_footer();
      fmt::print( "\n" );
    }
  }

  void print_algorithm_comparison()
  {
    fmt::print(
      Style::TITLE,
      "┌{0:─^{2}}┐\n"
      "│{1:^{2}}│\n"
      "└{0:─^{2}}┘\n\n",
      "",
      "ALGORITHM COMPARISON & RECOMMENDATIONS",
      70 );

    TablePrinter comparison_table(
      { { "Algorithm", 15 }, { "Best For", 20 }, { "Advantages", 40 }, { "Typical Use", 35 } } );

    comparison_table.print_header();

    std::vector<std::tuple<std::string, std::string, std::string, std::string, fmt::text_style>> algo_info = {
      { "Armijo",
        "Newton/Simple",
        "Fast, simple, no trial gradients",
        "Initial iterations, simple problems",
        Style::ARMIDO_COLOR },
      { "WeakWolfe",
        "Nonlinear CG",
        "Prevents too-short steps, efficient",
        "Conjugate gradient (c₂=0.1-0.4)",
        Style::WOLFE_COLOR },
      { "StrongWolfe",
        "L-BFGS/Newton",
        "Strong convergence, robust",
        "Quasi-Newton methods (c₂=0.9)",
        Style::STRONGWOLFE_COLOR },
      { "Goldstein",
        "Gradient-only",
        "No trial gradients, bounded steps",
        "When gradients are expensive",
        Style::GOLDSTEIN_COLOR },
      { "HagerZhang",
        "General purpose",
        "Most robust, adaptive safeguards",
        "Difficult problems, production",
        Style::HAGERZHANG_COLOR },
      { "MoreThuente",
        "High accuracy",
        "Most accurate, precise interpolation",
        "High-precision requirements",
        Style::MORETHUENTE_COLOR }
    };

    for ( const auto & [name, best_for, advantages, typical_use, style] : algo_info )
    {
      comparison_table.print_row(
        { name, best_for, advantages, typical_use },
        { style, Style::NONE, Style::NONE, Style::NONE } );
    }

    comparison_table.print_footer();

    fmt::print( Style::BOLD, "\nRECOMMENDATIONS:\n" );
    fmt::print( "  {} ", Unicode::ARROW_RIGHT );
    fmt::print( Style::ARMIDO_COLOR, "For Newton methods: Armijo" );
    fmt::print( " (fastest)\n" );

    fmt::print( "  {} ", Unicode::ARROW_RIGHT );
    fmt::print( Style::WOLFE_COLOR, "For conjugate gradient: Weak Wolfe" );
    fmt::print( " with c₂=0.1-0.4\n" );

    fmt::print( "  {} ", Unicode::ARROW_RIGHT );
    fmt::print( Style::STRONGWOLFE_COLOR, "For L-BFGS: Strong Wolfe" );
    fmt::print( " with c₂=0.9\n" );

    fmt::print( "  {} ", Unicode::ARROW_RIGHT );
    fmt::print( Style::HAGERZHANG_COLOR, "For general purpose: Hager-Zhang" );
    fmt::print( " (most robust)\n" );

    fmt::print( "  {} ", Unicode::ARROW_RIGHT );
    fmt::print( Style::MORETHUENTE_COLOR, "For high accuracy: More-Thuente" );
    fmt::print( "\n" );

    fmt::print( "  {} ", Unicode::ARROW_RIGHT );
    fmt::print( Style::GOLDSTEIN_COLOR, "When gradients are expensive: Goldstein" );
    fmt::print( "\n\n" );
  }

  void export_results_to_csv( const std::string & filename )
  {
    std::ofstream file( filename );
    if ( !file.is_open() )
    {
      fmt::print( Style::ERROR, "Failed to open file: {}\n", filename );
      return;
    }

    file << "Algorithm,TestFunction,Status,InitialF,FinalF,StepLength,"
         << "Iterations,FunctionEvals,GradientEvals,Reduction,TimeMus,"
         << "StartPoint,Direction\n";

    for ( const auto & res : results )
    {
      file << res.algorithm << "," << res.test_function << "," << res.status << "," << res.initial_f << ","
           << res.final_f << "," << res.step_length << "," << res.iterations << "," << res.function_evals << ","
           << res.gradient_evals << "," << res.reduction() << "," << res.cpu_time_mus << ","
           << "\"" << res.start_point_str << "\",\"" << res.direction_str << "\"\n";
    }

    file.close();
    fmt::print( Style::SUCCESS, "{} Results exported to {}\n\n", Unicode::CHECK, filename );
  }
};

// ===========================================================================
// Command Line Interface
// ===========================================================================

struct CLIOptions
{
  bool                     quick_mode = false;
  bool                     full_mode  = false;
  bool                     verbose    = false;
  bool                     help       = false;
  std::vector<std::string> algorithms;
  std::vector<std::string> problems;
  bool                     export_csv   = true;
  std::string              csv_filename = "line_search_results.csv";
  size_t                   max_tests    = 100;

  static CLIOptions parse( int argc, char ** argv )
  {
    CLIOptions       opts;
    AlgorithmManager algo_manager;
    auto             all_algorithms = algo_manager.available_algorithms();
    auto             all_problems   = TestProblemFactory::available_problems();

    for ( int i = 1; i < argc; ++i )
    {
      std::string arg = argv[i];

      if ( arg == "--quick" || arg == "-q" ) { opts.quick_mode = true; }
      else if ( arg == "--full" || arg == "-f" ) { opts.full_mode = true; }
      else if ( arg == "--verbose" || arg == "-v" ) { opts.verbose = true; }
      else if ( arg == "--help" || arg == "-h" ) { opts.help = true; }
      else if ( arg.find( "--algorithms=" ) == 0 )
      {
        std::string        list = arg.substr( 13 );
        std::istringstream iss( list );
        std::string        algo;
        while ( std::getline( iss, algo, ',' ) )
        {
          if ( std::find( all_algorithms.begin(), all_algorithms.end(), algo ) != all_algorithms.end() )
            opts.algorithms.push_back( algo );
        }
      }
      else if ( arg.find( "--problems=" ) == 0 )
      {
        std::string        list = arg.substr( 11 );
        std::istringstream iss( list );
        std::string        prob;
        while ( std::getline( iss, prob, ',' ) )
        {
          if ( std::find( all_problems.begin(), all_problems.end(), prob ) != all_problems.end() )
            opts.problems.push_back( prob );
        }
      }
      else if ( arg.find( "--max-tests=" ) == 0 ) { opts.max_tests = std::stoul( arg.substr( 12 ) ); }
      else if ( arg.find( "--output=" ) == 0 ) { opts.csv_filename = arg.substr( 9 ); }
      else if ( arg == "--no-csv" ) { opts.export_csv = false; }
    }

    return opts;
  }

  static void print_help()
  {
    fmt::print( Style::TITLE, "Line Search Test Framework v3.5\n" );
    fmt::print( Style::DIM, "{:─^{}}\n", "", 60 );
    fmt::print( "\nUsage: test_Linesearch [options]\n\n" );
    fmt::print( "Options:\n" );
    fmt::print( "  -q, --quick           Run quick test only\n" );
    fmt::print( "  -f, --full            Run full test suite\n" );
    fmt::print( "  -v, --verbose         Enable verbose output\n" );
    fmt::print( "  -h, --help            Show this help message\n" );
    fmt::print( "  --algorithms=LIST     Comma-separated list of algorithms to test\n" );
    fmt::print( "  --problems=LIST       Comma-separated list of problems to test\n" );
    fmt::print( "  --max-tests=N         Maximum number of tests to run (default: 100)\n" );
    fmt::print( "  --output=FILENAME     CSV output filename (default: line_search_results.csv)\n" );
    fmt::print( "  --no-csv              Disable CSV export\n\n" );

    AlgorithmManager algo_manager;
    auto             algorithms = algo_manager.available_algorithms();
    fmt::print( "Available algorithms: {}\n\n", join_strings( algorithms, ", " ) );

    auto problems = TestProblemFactory::available_problems();
    fmt::print( "Available problems: {}\n\n", join_strings( problems, ", " ) );
  }
};

// ===========================================================================
// Main Function
// ===========================================================================

int main( int argc, char ** argv )
{
  std::locale::global( std::locale( "en_US.UTF-8" ) );

  CLIOptions opts = CLIOptions::parse( argc, argv );

  if ( opts.help )
  {
    CLIOptions::print_help();
    return 0;
  }

  try
  {
    if ( opts.quick_mode )
    {
      TestParameters params;
      params.quick_mode = true;
      LineSearchTester tester( params );
      tester.run_quick_test();
    }
    else
    {
      TestParameters params;
      params.problems              = opts.problems;
      params.algorithms            = opts.algorithms;
      params.verbose               = opts.verbose;
      params.export_csv            = opts.export_csv;
      params.csv_filename          = opts.csv_filename;
      params.max_tests_per_problem = opts.max_tests;

      LineSearchTester tester( params );
      tester.run_all_tests();
      fmt::print( Style::SUCCESS, "{} All tests completed successfully!\n", Unicode::CHECK );
    }
  }
  catch ( const std::exception & e )
  {
    fmt::print( Style::ERROR, "{} Error during testing: {}\n", Unicode::CROSS, e.what() );
    return 1;
  }

  return 0;
}
