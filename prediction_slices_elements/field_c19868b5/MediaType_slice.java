// Source-based slice around line 106
// Method: com.google.common.net.MediaType.knownTypes

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
  }

  private static MediaType createConstantUtf8(String type, String subtype) {
    MediaType mediaType = addKnownType(new MediaType(type, subtype, UTF_8_CONSTANT_PARAMETERS));
