// Source-based slice around line 296
// Method: com.google.common.net.InternetDomainName.PART_CHAR_MATCHER

  }

  private static final CharMatcher DASH_MATCHER = CharMatcher.anyOf("-_");

  private static final CharMatcher DIGIT_MATCHER = CharMatcher.inRange('0', '9');

  private static final CharMatcher LETTER_MATCHER =
      CharMatcher.inRange('a', 'z').or(CharMatcher.inRange('A', 'Z'));

  private static final CharMatcher PART_CHAR_MATCHER =
      DIGIT_MATCHER.or(LETTER_MATCHER).or(DASH_MATCHER);

  /**
   * Helper method for {@link #validateSyntax(List)}. Validates that one part of a domain name is
   * valid.
   *
   * @param part The domain name part to be validated
   * @param isFinalPart Is this the final (rightmost) domain part?
   * @return Whether the part is valid
   */
