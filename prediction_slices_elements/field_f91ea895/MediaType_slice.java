// Source-based slice around line 97
// Method: com.google.common.net.MediaType.APPLICATION_TYPE

  private static final CharMatcher QUOTED_TEXT_MATCHER = ascii().and(CharMatcher.noneOf("\"\\\r"));

  /*
   * This matches the same characters as linear-white-space from RFC 822, but we make no effort to
   * enforce any particular rules with regards to line folding as stated in the class docs.
   */
  private static final CharMatcher LINEAR_WHITE_SPACE = CharMatcher.anyOf(" \t\r\n");

  // TODO(gak): make these public?
  private static final String APPLICATION_TYPE = "application";
  private static final String AUDIO_TYPE = "audio";
  private static final String IMAGE_TYPE = "image";
  private static final String TEXT_TYPE = "text";
  private static final String VIDEO_TYPE = "video";
  private static final String FONT_TYPE = "font";

  private static final String WILDCARD = "*";

  private static final Map<MediaType, MediaType> knownTypes = new HashMap<>();

