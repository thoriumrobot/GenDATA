// Source-based slice around line 456
// Method: com.google.common.net.MediaType.KEY_ARCHIVE

      createConstant(APPLICATION_TYPE, "x-www-form-urlencoded");

  /**
   * As described in <a href="https://www.rsa.com/rsalabs/node.asp?id=2138">PKCS #12: Personal
   * Information Exchange Syntax Standard</a>, PKCS #12 defines an archive file format for storing
   * many cryptography objects as a single file.
   *
   * @since 15.0
   */
  public static final MediaType KEY_ARCHIVE = createConstant(APPLICATION_TYPE, "pkcs12");

  /**
   * This is a non-standard media type, but is commonly used in serving hosted binary files as it is
   * <a href="http://code.google.com/p/browsersec/wiki/Part2#Survey_of_content_sniffing_behaviors">
   * known not to trigger content sniffing in current browsers</a>. It <i>should not</i> be used in
   * other situations as it is not specified by any RFC and does not appear in the <a
   * href="http://www.iana.org/assignments/media-types">/IANA MIME Media Types</a> list. Consider
   * {@link #OCTET_STREAM} for binary data that is not being served to a browser.
   *
   * @since 14.0
