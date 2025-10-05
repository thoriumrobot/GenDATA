// Source-based slice around line 1001
// Method: <com.google.common.net.MediaType: MediaType create(String,String)>

        && this.parameters.entries().containsAll(mediaTypeRange.parameters.entries());
  }

  /**
   * Creates a new media type with the given type and subtype.
   *
   * @throws IllegalArgumentException if type or subtype is invalid or if a wildcard is used for the
   *     type, but not the subtype.
   */
  public static MediaType create(String type, String subtype) {
    MediaType mediaType = create(type, subtype, ImmutableListMultimap.<String, String>of());
    mediaType.parsedCharset = Optional.absent();
    return mediaType;
  }

  private static MediaType create(
      String type, String subtype, Multimap<String, String> parameters) {
    checkNotNull(type);
    checkNotNull(subtype);
    checkNotNull(parameters);
