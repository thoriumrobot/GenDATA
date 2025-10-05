// Source-based slice around line 1052
// Method: <com.google.common.net.MediaType: MediaType createFontType(String)>

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
  }

  /**
   * Creates a media type with the "image" type and the given subtype.
   *
   * @throws IllegalArgumentException if subtype is invalid
   */
  static MediaType createImageType(String subtype) {
    return create(IMAGE_TYPE, subtype);
