// Source-based slice around line 40
// Method: <com.google.common.base.CommonPattern: boolean isPcreLike()>


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
