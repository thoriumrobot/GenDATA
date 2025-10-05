// Source-based slice around line 1061
// Method: <com.google.common.net.MediaType: MediaType createImageType(String)>

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
  }

  /**
   * Creates a media type with the "text" type and the given subtype.
   *
   * @throws IllegalArgumentException if subtype is invalid
   */
  static MediaType createTextType(String subtype) {
    return create(TEXT_TYPE, subtype);
