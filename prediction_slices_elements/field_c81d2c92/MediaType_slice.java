// Source-based slice around line 251
// Method: com.google.common.net.MediaType.PSD

   * href="http://en.wikipedia.org/wiki/Adobe_Photoshop#File_format">Wikipedia</a>; this is the
   * regular output/input of Photoshop (which can also export to various image formats; note that
   * files with extension "PSB" are in a distinct but related format).
   *
   * <p>This is a more recent replacement for the older, experimental type {@code x-photoshop}: <a
   * href="http://tools.ietf.org/html/rfc2046#section-6">RFC-2046.6</a>.
   *
   * @since 15.0
   */
  public static final MediaType PSD = createConstant(IMAGE_TYPE, "vnd.adobe.photoshop");

  public static final MediaType SVG_UTF_8 = createConstantUtf8(IMAGE_TYPE, "svg+xml");
  public static final MediaType TIFF = createConstant(IMAGE_TYPE, "tiff");

  /**
   * <a href="https://en.wikipedia.org/wiki/AVIF">AVIF image format</a>.
   *
   * @since 33.5.0
   */
  public static final MediaType AVIF = createConstant(IMAGE_TYPE, "avif");
