// Source-based slice around line 350
// Method: <com.google.common.io.BaseEncoding: BaseEncoding base64()>

   * href="http://tools.ietf.org/html/rfc3548#section-3">RFC 3548</a>.)
   *
   * <p>The character {@code '='} is used for padding, but can be {@linkplain #omitPadding()
   * omitted} or {@linkplain #withPadChar(char) replaced}.
   *
   * <p>No line feeds are added by default, as per <a
   * href="http://tools.ietf.org/html/rfc4648#section-3.1">RFC 4648 section 3.1</a>, Line Feeds in
   * Encoded Data. Line feeds may be added using {@link #withSeparator(String, int)}.
   */
  public static BaseEncoding base64() {
    return BASE64;
  }

  private static final BaseEncoding BASE64_URL =
      new Base64Encoding(
          "base64Url()", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_", '=');

  /**
   * The "base64url" encoding specified by <a
   * href="http://tools.ietf.org/html/rfc4648#section-5">RFC 4648 section 5</a>, Base 64 Encoding
