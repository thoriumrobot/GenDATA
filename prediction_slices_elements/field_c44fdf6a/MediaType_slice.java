// Source-based slice around line 88
// Method: com.google.common.net.MediaType.QUOTED_TEXT_MATCHER

      ImmutableListMultimap.of(CHARSET_ATTRIBUTE, Ascii.toLowerCase(UTF_8.name()));

  /** Matcher for type, subtype and attributes. */
  private static final CharMatcher TOKEN_MATCHER =
      ascii()
          .and(javaIsoControl().negate())
          .and(CharMatcher.isNot(' '))
          .and(CharMatcher.noneOf("()<>@,;:\\\"/[]?="));

  private static final CharMatcher QUOTED_TEXT_MATCHER = ascii().and(CharMatcher.noneOf("\"\\\r"));

  /*
   * This matches the same characters as linear-white-space from RFC 822, but we make no effort to
   * enforce any particular rules with regards to line folding as stated in the class docs.
   */
  private static final CharMatcher LINEAR_WHITE_SPACE = CharMatcher.anyOf(" \t\r\n");

  // TODO(gak): make these public?
  private static final String APPLICATION_TYPE = "application";
  private static final String AUDIO_TYPE = "audio";
