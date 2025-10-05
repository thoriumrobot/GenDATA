// Source-based slice around line 1083
// Method: <com.google.common.net.MediaType: String normalizeToken(String)>

  /**
   * Creates a media type with the "video" type and the given subtype.
   *
   * @throws IllegalArgumentException if subtype is invalid
   */
  static MediaType createVideoType(String subtype) {
    return create(VIDEO_TYPE, subtype);
  }

  private static String normalizeToken(String token) {
    checkArgument(TOKEN_MATCHER.matchesAllOf(token));
    checkArgument(!token.isEmpty());
    return Ascii.toLowerCase(token);
  }

  private static String normalizeParameterValue(String attribute, String value) {
    checkNotNull(value); // for GWT
    checkArgument(ascii().matchesAllOf(value), "parameter values must be ASCII: %s", value);
    return attribute.equals(CHARSET_ATTRIBUTE) ? Ascii.toLowerCase(value) : value;
  }
