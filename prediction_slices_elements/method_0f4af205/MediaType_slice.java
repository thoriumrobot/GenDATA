// Source-based slice around line 1101
// Method: <com.google.common.net.MediaType: MediaType parse(String)>

    return attribute.equals(CHARSET_ATTRIBUTE) ? Ascii.toLowerCase(value) : value;
  }

  /**
   * Parses a media type from its string representation.
   *
   * @throws IllegalArgumentException if the input is not parsable
   */
  @CanIgnoreReturnValue // TODO(b/219820829): consider removing
  public static MediaType parse(String input) {
    checkNotNull(input);
    Tokenizer tokenizer = new Tokenizer(input);
    try {
      String type = tokenizer.consumeToken(TOKEN_MATCHER);
      consumeSeparator(tokenizer, '/');
      String subtype = tokenizer.consumeToken(TOKEN_MATCHER);
      ImmutableListMultimap.Builder<String, String> parameters = ImmutableListMultimap.builder();
      while (tokenizer.hasMore()) {
        consumeSeparator(tokenizer, ';');
        String attribute = tokenizer.consumeToken(TOKEN_MATCHER);
