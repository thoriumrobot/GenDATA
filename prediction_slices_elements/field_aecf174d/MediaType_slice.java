// Source-based slice around line 102
// Method: com.google.common.net.MediaType.FONT_TYPE

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

  private static MediaType createConstant(String type, String subtype) {
    MediaType mediaType =
        addKnownType(new MediaType(type, subtype, ImmutableListMultimap.<String, String>of()));
    mediaType.parsedCharset = Optional.absent();
    return mediaType;
