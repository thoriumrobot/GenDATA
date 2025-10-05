// Source-based slice around line 126
// Method: <com.google.common.base.Splitter: Splitter on(char)>

  }

  /**
   * Returns a splitter that uses the given single-character separator. For example, {@code
   * Splitter.on(',').split("foo,,bar")} returns an iterable containing {@code ["foo", "", "bar"]}.
   *
   * @param separator the character to recognize as a separator
   * @return a splitter, with default settings, that recognizes that separator
   */
  public static Splitter on(char separator) {
    return on(CharMatcher.is(separator));
  }

  /**
   * Returns a splitter that considers any single character matched by the given {@code CharMatcher}
   * to be a separator. For example, {@code
   * Splitter.on(CharMatcher.anyOf(";,")).split("foo,;bar,quux")} returns an iterable containing
   * {@code ["foo", "", "bar", "quux"]}.
   *
   * @param separatorMatcher a {@link CharMatcher} that determines whether a character is a
