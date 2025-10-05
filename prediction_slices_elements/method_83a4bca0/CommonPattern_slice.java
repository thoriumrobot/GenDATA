// Source-based slice around line 34
// Method: <com.google.common.base.CommonPattern: String toString()>

abstract class CommonPattern {
  public abstract CommonMatcher matcher(CharSequence t);

  public abstract String pattern();

  public abstract int flags();

  // Re-declare this as abstract to force subclasses to override.
  @Override
  public abstract String toString();

  public static CommonPattern compile(String pattern) {
    return Platform.compilePattern(pattern);
  }

  public static boolean isPcreLike() {
    return Platform.patternCompilerIsPcreLike();
  }
}
