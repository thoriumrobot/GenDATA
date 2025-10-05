// Source-based slice around line 44
// Method: <com.google.common.base.PatternCompiler: boolean isPcreLike()>

  CommonPattern compile(String pattern);

  /**
   * Returns {@code true} if the regex implementation behaves like Perl -- notably, by supporting
   * possessive quantifiers but also being susceptible to catastrophic backtracking.
   */
  @RestrictedApi(
      explanation = "PatternCompiler is an implementation detail of com.google.common.base",
      allowedOnPath = ".*/com/google/common/base/.*")
  boolean isPcreLike();
}
