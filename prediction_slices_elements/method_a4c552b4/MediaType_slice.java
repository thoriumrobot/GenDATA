// Source-based slice around line 1089
// Method: <com.google.common.net.MediaType: String normalizeParameterValue(String,String)>

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

  /**
   * Parses a media type from its string representation.
   *
   * @throws IllegalArgumentException if the input is not parsable
   */
