// Source-based slice around line 115
// Method: <com.google.common.net.MediaType: MediaType createConstantUtf8(String,String)>

  private static final Map<MediaType, MediaType> knownTypes = new HashMap<>();

  private static MediaType createConstant(String type, String subtype) {
    MediaType mediaType =
        addKnownType(new MediaType(type, subtype, ImmutableListMultimap.<String, String>of()));
    mediaType.parsedCharset = Optional.absent();
    return mediaType;
  }

  private static MediaType createConstantUtf8(String type, String subtype) {
    MediaType mediaType = addKnownType(new MediaType(type, subtype, UTF_8_CONSTANT_PARAMETERS));
    mediaType.parsedCharset = Optional.of(UTF_8);
    return mediaType;
  }

  @CanIgnoreReturnValue
  private static MediaType addKnownType(MediaType mediaType) {
    knownTypes.put(mediaType, mediaType);
    return mediaType;
  }
