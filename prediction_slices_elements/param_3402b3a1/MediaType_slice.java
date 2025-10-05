// Source-based slice around line 122
// Method: <com.google.common.net.MediaType: MediaType addKnownType(MediaType)>

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

  /*
   * The following constants are grouped by their type and ordered alphabetically by the constant
   * name within that type. The constant name should be a sensible identifier that is closest to the
   * "common name" of the media. This is often, but not necessarily the same as the subtype.
   *
   * Be sure to declare all constants with the type and subtype in all lowercase. For types that
