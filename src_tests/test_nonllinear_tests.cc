// main.cpp
/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2025                                                      |
 |                                                                          |
 |      Enrico Bertolazzi                                                   |
 |      Dipartimento di Ingegneria Industriale                              |
 |      Università degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Utils_nonlinear_system.hh"
#include "Utils_fmt.hh"

void test_exact_solutions() {

  using NS = Utils::NonlinearSystem;
  const double tol = 1e-8;

  // ------------------------------------------------------------
  // Determina larghezza colonne dinamicamente
  // ------------------------------------------------------------
  size_t Wname = 10;
  for (auto *sys : Utils::nonlinear_system_tests) {
    if (!sys) continue;
    Wname = std::max(Wname, sys->title().size());
  }
  Wname += 2;

  const size_t Wsol  = 6;
  const size_t Wres  = 16;
  const size_t Wstat = 12;

  auto rep = []( size_t n, std::string const & s ) {
    std::string res;
    for ( size_t k{0}; k < n; ++k ) res += s;
    return res;
  };

  fmt::print("\n");

  // ------------------------------------------------------------
  // Header con bordo esterno doppio e interno singolo
  // ------------------------------------------------------------
  fmt::print("╔{}╦{}╦{}╦{}╗\n",
             rep(Wname+2, "═"),
             rep(Wsol+2,  "═"),
             rep(Wres+2,  "═"),
             rep(Wstat+2, "═"));

  // Header
  fmt::print("║ {:^{}} ║ {:^{}} ║ {:^{}} ║ {:^{}} ║\n",
             "Test",    Wname,
             "Sol#",    Wsol,
             "Residuo", Wres,
             "Status",  Wstat);

  fmt::print("╠{}╬{}╬{}╬{}╣\n",
             rep(Wname+2, "═"),
             rep(Wsol+2,  "═"),
             rep(Wres+2,  "═"),
             rep(Wstat+2, "═"));

  // ------------------------------------------------------------
  // Corpo tabella
  // ------------------------------------------------------------
  for (auto *sys : Utils::nonlinear_system_tests) {

    if (!sys) continue;

    std::vector<NS::Vector> sols;
    sys->exact_solution(sols);

    // Nessuna soluzione
    if (sols.empty()) {
      fmt::print("║ {:<{}} ║ {:>{}} ║ {:>{}} ║ {:^{}} ║\n",
                 sys->title(), Wname,
                 "-",   Wsol,
                 "-",   Wres,
                 "n/a", Wstat);
      continue;
    }

    // Soluzioni esatte
    for (size_t i = 0; i < sols.size(); ++i) {

      const auto& x = sols[i];
      NS::Vector f(sys->num_equations());
      double res = 0.0;
      bool ok = false;

      try {
        sys->evaluate(x, f);
        res = f.norm();
        ok  = (res <= tol);
      }
      catch (...) {
        fmt::print("║ {:<{}} ║ {:>{}} ║ {:>{}} ║ ",
                   sys->title(), Wname,
                   int(i+1),  Wsol,
                   "eval-err", Wres);
        
        fmt::print(fg(fmt::color::orange_red), "{:^{}}", "ERR", Wstat);
        fmt::print(" ║\n");
        continue;
      }

      // Prepara le stringhe con colori
      std::string res_str = fmt::format("{:.8e}", res);
      std::string status_str = ok ? "✓  OK" : "✗  NO";

      fmt::print("║ {:<{}} ║ {:>{}} ║ ",
                 sys->title(), Wname,
                 int(i+1), Wsol);

      // Colore per il residuo
      if (ok) {
        fmt::print(fg(fmt::color::green), "{:>{}}", res_str, Wres);
      } else {
        fmt::print(fg(fmt::color::red), "{:>{}}", res_str, Wres);
      }

      fmt::print(" ║ ");

      // Colore per lo status
      if (ok) {
        fmt::print(fg(fmt::color::green), "{:^{}}", status_str, Wstat);
      } else {
        fmt::print(fg(fmt::color::red), "{:^{}}", status_str, Wstat);
      }

      fmt::print(" ║\n");
    }
  }

  // Footer
  fmt::print("╚{}╩{}╩{}╩{}╝\n",
             rep(Wname+2, "═"),
             rep(Wsol+2,  "═"),
             rep(Wres+2,  "═"),
             rep(Wstat+2, "═"));

  fmt::print("\n");
}

int main() {
  Utils::init_nonlinear_system_tests();
  test_exact_solutions();
  fmt::print("\nAll Done Folks!\n");
  return 0;
}
