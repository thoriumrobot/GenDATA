// Source-based slice around line 137
// Method: com.google.common.net.MediaType.ANY_TYPE

   * The following constants are grouped by their type and ordered alphabetically by the constant
   * name within that type. The constant name should be a sensible identifier that is closest to the
   * "common name" of the media. This is often, but not necessarily the same as the subtype.
   *
   * Be sure to declare all constants with the type and subtype in all lowercase. For types that
   * take a charset (e.g. all text/* types), default to UTF-8 and suffix the constant name with
   * "_UTF_8".
   */

  public static final MediaType ANY_TYPE = createConstant(WILDCARD, WILDCARD);
  public static final MediaType ANY_TEXT_TYPE = createConstant(TEXT_TYPE, WILDCARD);
  public static final MediaType ANY_IMAGE_TYPE = createConstant(IMAGE_TYPE, WILDCARD);
  public static final MediaType ANY_AUDIO_TYPE = createConstant(AUDIO_TYPE, WILDCARD);
  public static final MediaType ANY_VIDEO_TYPE = createConstant(VIDEO_TYPE, WILDCARD);
  public static final MediaType ANY_APPLICATION_TYPE = createConstant(APPLICATION_TYPE, WILDCARD);

  /**
   * Wildcard matching any "font" top-level media type.
   *
   * @since 30.0
