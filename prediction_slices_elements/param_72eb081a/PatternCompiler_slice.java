// Source-based slice around line 35
// Method: <com.google.common.base.PatternCompiler: CommonPattern compile(String)>

interface PatternCompiler {
  /**
   * Compiles the given pattern.
   *
   * @throws IllegalArgumentException if the pattern is invalid
   */
  @RestrictedApi(
      explanation = "PatternCompiler is an implementation detail of com.google.common.base",
      allowedOnPath = ".*/com/google/common/base/.*")
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
