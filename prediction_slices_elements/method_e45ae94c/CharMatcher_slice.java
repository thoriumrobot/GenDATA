// Source-based slice around line 367
// Method: <com.google.common.base.CharMatcher: boolean matches(char)>

  /**
   * Constructor for use by subclasses. When subclassing, you may want to override {@code
   * toString()} to provide a useful description.
   */
  protected CharMatcher() {}

  // Abstract methods

  /** Determines a true or false value for the given character. */
  public abstract boolean matches(char c);

  // Non-static factories

  /** Returns a matcher that matches any character not matched by this matcher. */
  // This is not an override in java7, where Guava's Predicate does not extend the JDK's Predicate.
  @SuppressWarnings("MissingOverride")
  public CharMatcher negate() {
    return new Negated(this);
  }

