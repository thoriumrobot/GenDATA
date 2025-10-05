// Source-based slice around line 405
// Method: com.google.common.base.CharMatcher.DISTINCT_CHARS

   *
   * <p>This method has no effect (returns {@code this}) when called in GWT: it's unclear whether a
   * precomputed matcher is faster, but it certainly would consume more memory (which doesn't seem
   * like a worthwhile tradeoff in a browser).
   */
  public CharMatcher precomputed() {
    return Platform.precomputeCharMatcher(this);
  }

  private static final int DISTINCT_CHARS = Character.MAX_VALUE - Character.MIN_VALUE + 1;

  /**
   * This is the actual implementation of {@link #precomputed}, but we bounce calls through a method
   * on {@link Platform} so that we can have different behavior in GWT.
   *
   * <p>This implementation tries to be smart in a number of ways. It recognizes cases where the
   * negation is cheaper to precompute than the matcher itself; it tries to build small hash tables
   * for matchers that only match a few characters, and so on. In the worst-case scenario, it
   * constructs an eight-kilobyte bit array and queries that. In many situations this produces a
   * matcher which is faster to query than the original.
