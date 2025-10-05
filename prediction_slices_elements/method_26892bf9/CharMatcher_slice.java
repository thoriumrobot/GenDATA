// Source-based slice around line 657
// Method: <com.google.common.base.CharMatcher: String retainFrom(CharSequence)>

   * Returns a string containing all matching BMP characters of a character sequence, in order. For
   * example:
   *
   * {@snippet :
   * CharMatcher.is('a').retainFrom("bazaar")
   * }
   *
   * ... returns {@code "aaa"}.
   */
  public String retainFrom(CharSequence sequence) {
    return negate().removeFrom(sequence);
  }

  /**
   * Returns a string copy of the input character sequence, with each matching BMP character
   * replaced by a given replacement character. For example:
   *
   * {@snippet :
   * CharMatcher.is('a').replaceFrom("radar", 'o')
   * }
