// Source-based slice around line 468
// Method: com.google.common.net.MediaType.APPLICATION_BINARY

   * This is a non-standard media type, but is commonly used in serving hosted binary files as it is
   * <a href="http://code.google.com/p/browsersec/wiki/Part2#Survey_of_content_sniffing_behaviors">
   * known not to trigger content sniffing in current browsers</a>. It <i>should not</i> be used in
   * other situations as it is not specified by any RFC and does not appear in the <a
   * href="http://www.iana.org/assignments/media-types">/IANA MIME Media Types</a> list. Consider
   * {@link #OCTET_STREAM} for binary data that is not being served to a browser.
   *
   * @since 14.0
   */
  public static final MediaType APPLICATION_BINARY = createConstant(APPLICATION_TYPE, "binary");

  /**
   * As described in <a href="https://www.rfc-editor.org/rfc/rfc8949.html">RFC 8949</a>, this
   * constant ({@code application/cbor}) is used for the Concise Binary Object Representation (CBOR)
   * data format.
   *
   * @since 33.4.0
   */
  public static final MediaType CBOR = createConstant(APPLICATION_TYPE, "cbor");

