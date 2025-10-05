// Source-based slice around line 414
// Method: com.google.common.io.BaseEncoding.BASE16

   *
   * <p>No line feeds are added by default, as per <a
   * href="http://tools.ietf.org/html/rfc4648#section-3.1">RFC 4648 section 3.1</a>, Line Feeds in
   * Encoded Data. Line feeds may be added using {@link #withSeparator(String, int)}.
   */
  public static BaseEncoding base32Hex() {
    return BASE32_HEX;
  }

  private static final BaseEncoding BASE16 = new Base16Encoding("base16()", "0123456789ABCDEF");

  /**
   * The "base16" encoding specified by <a href="http://tools.ietf.org/html/rfc4648#section-8">RFC
   * 4648 section 8</a>, Base 16 Encoding. (This is the same as the base 16 encoding from <a
   * href="http://tools.ietf.org/html/rfc3548#section-6">RFC 3548</a>.) This is commonly known as
   * "hexadecimal" format.
   *
   * <p>No padding is necessary in base 16, so {@link #withPadChar(char)} and {@link #omitPadding()}
   * have no effect.
   *
