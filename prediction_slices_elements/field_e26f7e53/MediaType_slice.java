// Source-based slice around line 207
// Method: com.google.common.net.MediaType.VTT_UTF_8

   */
  public static final MediaType XML_UTF_8 = createConstantUtf8(TEXT_TYPE, "xml");

  /**
   * As described in <a href="https://w3c.github.io/webvtt/#iana-text-vtt">the VTT spec</a>, this is
   * used for Web Video Text Tracks (WebVTT) files, used with the HTML5 track element.
   *
   * @since 20.0
   */
  public static final MediaType VTT_UTF_8 = createConstantUtf8(TEXT_TYPE, "vtt");

  /* image types */
  /**
   * <a href="https://en.wikipedia.org/wiki/BMP_file_format">Bitmap file format</a> ({@code bmp}
   * files).
   *
   * @since 13.0
   */
  public static final MediaType BMP = createConstant(IMAGE_TYPE, "bmp");

