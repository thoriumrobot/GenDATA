// Source-based slice around line 293
// Method: com.google.common.net.InternetDomainName.LETTER_MATCHER

    }

    return true;
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
