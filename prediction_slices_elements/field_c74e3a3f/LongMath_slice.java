// Source-based slice around line 1056
// Method: com.google.common.math.LongMath.millerRabinBaseSets

  }

  /*
   * If n <= millerRabinBases[i][0], then testing n against bases millerRabinBases[i][1..] suffices
   * to prove its primality. Values from miller-rabin.appspot.com.
   *
   * NOTE: We could get slightly better bases that would be treated as unsigned, but benchmarks
   * showed negligible performance improvements.
   */
  private static final long[][] millerRabinBaseSets = {
    {291830, 126401071349994536L},
    {885594168, 725270293939359937L, 3569819667048198375L},
    {273919523040L, 15, 7363882082L, 992620450144556L},
    {47636622961200L, 2, 2570940, 211991001, 3749873356L},
    {
      7999252175582850L,
      2,
      4130806001517L,
      149795463772692060L,
      186635894390467037L,
