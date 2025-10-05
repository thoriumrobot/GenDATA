// Source-based slice around line 1043
// Method: <com.google.common.net.MediaType: MediaType createAudioType(String)>

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
  }

  /**
   * Creates a media type with the "font" type and the given subtype.
   *
   * @throws IllegalArgumentException if subtype is invalid
   */
  static MediaType createFontType(String subtype) {
    return create(FONT_TYPE, subtype);
