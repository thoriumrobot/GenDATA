// Source-based slice around line 344
// Method: <com.google.common.base.CharMatcher: CharMatcher inRange(char,char)>

  }

  /**
   * Returns a {@code char} matcher that matches any character in a given BMP range (both endpoints
   * are inclusive). For example, to match any lowercase letter of the English alphabet, use {@code
   * CharMatcher.inRange('a', 'z')}.
   *
   * @throws IllegalArgumentException if {@code endInclusive < startInclusive}
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
