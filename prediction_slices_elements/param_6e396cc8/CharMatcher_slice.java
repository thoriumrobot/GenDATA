// Source-based slice around line 352
// Method: <com.google.common.base.CharMatcher: CharMatcher forPredicate(Predicate)>

   */
  public static CharMatcher inRange(char startInclusive, char endInclusive) {
    return new InRange(startInclusive, endInclusive);
  }

  /**
   * Returns a matcher with identical behavior to the given {@link Character}-based predicate, but
   * which operates on primitive {@code char} instances instead.
   */
  public static CharMatcher forPredicate(Predicate<? super Character> predicate) {
    return predicate instanceof CharMatcher ? (CharMatcher) predicate : new ForPredicate(predicate);
  }

  // Constructors

  /**
   * Constructor for use by subclasses. When subclassing, you may want to override {@code
   * toString()} to provide a useful description.
   */
  protected CharMatcher() {}
