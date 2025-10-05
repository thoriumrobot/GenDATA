// Source-based slice around line 333
// Method: <com.google.common.base.CharMatcher: CharMatcher noneOf(CharSequence)>

        // matcher?
        return new AnyOf(sequence);
    }
  }

  /**
   * Returns a {@code char} matcher that matches any BMP character not present in the given
   * character sequence. Returns a bogus matcher if the sequence contains supplementary characters.
   */
  public static CharMatcher noneOf(CharSequence sequence) {
    return anyOf(sequence).negate();
  }

  /**
   * Returns a {@code char} matcher that matches any character in a given BMP range (both endpoints
   * are inclusive). For example, to match any lowercase letter of the English alphabet, use {@code
   * CharMatcher.inRange('a', 'z')}.
   *
   * @throws IllegalArgumentException if {@code endInclusive < startInclusive}
   */
