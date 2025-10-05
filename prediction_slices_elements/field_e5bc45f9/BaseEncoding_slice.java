// Source-based slice around line 333
// Method: com.google.common.io.BaseEncoding.BASE64

   * Returns an encoding that behaves equivalently to this encoding, but decodes letters without
   * regard to case.
   *
   * @throws IllegalStateException if the alphabet used by this encoding contains mixed upper- and
   *     lower-case characters
   * @since 32.0.0
   */
  public abstract BaseEncoding ignoreCase();

  private static final BaseEncoding BASE64 =
      new Base64Encoding(
          "base64()", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", '=');

  /**
   * The "base64" base encoding specified by <a
   * href="http://tools.ietf.org/html/rfc4648#section-4">RFC 4648 section 4</a>, Base 64 Encoding.
   * (This is the same as the base 64 encoding from <a
   * href="http://tools.ietf.org/html/rfc3548#section-3">RFC 3548</a>.)
   *
   * <p>The character {@code '='} is used for padding, but can be {@linkplain #omitPadding()
