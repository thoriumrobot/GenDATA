// Source-based slice around line 381
// Method: <com.google.common.base.CharMatcher: CharMatcher and(CharMatcher)>

  // This is not an override in java7, where Guava's Predicate does not extend the JDK's Predicate.
  @SuppressWarnings("MissingOverride")
  public CharMatcher negate() {
    return new Negated(this);
  }

  /**
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

