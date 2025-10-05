// Source-based slice around line 388
// Method: <com.google.common.base.CharMatcher: CharMatcher or(CharMatcher)>

   * Returns a matcher that matches any character matched by both this matcher and {@code other}.
   */
  public CharMatcher and(CharMatcher other) {
    return new And(this, other);
  }

  /**
   * Returns a matcher that matches any character matched by either this matcher or {@code other}.
   */
  public CharMatcher or(CharMatcher other) {
    return new Or(this, other);
  }

  /**
   * Returns a {@code char} matcher functionally equivalent to this one, but which may be faster to
   * query than the original; your mileage may vary. Precomputation takes time and requires more
   * memory, so it is only likely to be worthwhile if the precomputed matcher is queried very often.
   *
   * <p>This method has no effect (returns {@code this}) when called in GWT: it's unclear whether a
   * precomputed matcher is faster, but it certainly would consume more memory (which doesn't seem
