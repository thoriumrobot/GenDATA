// Source-based slice around line 1034
// Method: <com.google.common.net.MediaType: MediaType createApplicationType(String)>

    MediaType result = firstNonNull(knownTypes.get(mediaType), mediaType);
    return result;
  }

  /**
   * Creates a media type with the "application" type and the given subtype.
   *
   * @throws IllegalArgumentException if subtype is invalid
   */
  static MediaType createApplicationType(String subtype) {
    return create(APPLICATION_TYPE, subtype);
  }

  /**
   * Creates a media type with the "audio" type and the given subtype.
   *
   * @throws IllegalArgumentException if subtype is invalid
   */
  static MediaType createAudioType(String subtype) {
    return create(AUDIO_TYPE, subtype);
