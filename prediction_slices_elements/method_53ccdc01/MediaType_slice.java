// Source-based slice around line 1079
// Method: <com.google.common.net.MediaType: MediaType createVideoType(String)>

  static MediaType createTextType(String subtype) {
    return create(TEXT_TYPE, subtype);
  }

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
