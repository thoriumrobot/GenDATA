// Source-based slice around line 1007
// Method: <com.google.common.net.MediaType: MediaType create(String,String,Multimap)>

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
    String normalizedType = normalizeToken(type);
    String normalizedSubtype = normalizeToken(subtype);
    checkArgument(
        !normalizedType.equals(WILDCARD) || normalizedSubtype.equals(WILDCARD),
        "A wildcard type cannot be used with a non-wildcard subtype");
    ImmutableListMultimap.Builder<String, String> builder = ImmutableListMultimap.builder();
