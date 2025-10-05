// Source-based slice around line 401
// Method: <com.google.common.base.CharMatcher: CharMatcher precomputed()>

  /**
   * Returns a {@code char} matcher functionally equivalent to this one, but which may be faster to
   * query than the original; your mileage may vary. Precomputation takes time and requires more
   * memory, so it is only likely to be worthwhile if the precomputed matcher is queried very often.
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
