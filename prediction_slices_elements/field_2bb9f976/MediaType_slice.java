// Source-based slice around line 149
// Method: com.google.common.net.MediaType.ANY_FONT_TYPE

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
      createConstantUtf8(TEXT_TYPE, "cache-manifest");
  public static final MediaType CSS_UTF_8 = createConstantUtf8(TEXT_TYPE, "css");
  public static final MediaType CSV_UTF_8 = createConstantUtf8(TEXT_TYPE, "csv");
  public static final MediaType HTML_UTF_8 = createConstantUtf8(TEXT_TYPE, "html");
  public static final MediaType I_CALENDAR_UTF_8 = createConstantUtf8(TEXT_TYPE, "calendar");

  /**
