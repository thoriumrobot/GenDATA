// Source-based slice around line 142
// Method: com.google.common.net.MediaType.ANY_APPLICATION_TYPE

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
   */
  public static final MediaType ANY_FONT_TYPE = createConstant(FONT_TYPE, WILDCARD);

  /* text types */
  public static final MediaType CACHE_MANIFEST_UTF_8 =
