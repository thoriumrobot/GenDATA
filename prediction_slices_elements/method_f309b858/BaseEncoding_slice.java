// Source-based slice around line 429
// Method: <com.google.common.io.BaseEncoding: BaseEncoding base16()>

   * "hexadecimal" format.
   *
   * <p>No padding is necessary in base 16, so {@link #withPadChar(char)} and {@link #omitPadding()}
   * have no effect.
   *
   * <p>No line feeds are added by default, as per <a
   * href="http://tools.ietf.org/html/rfc4648#section-3.1">RFC 4648 section 3.1</a>, Line Feeds in
   * Encoded Data. Line feeds may be added using {@link #withSeparator(String, int)}.
   */
  public static BaseEncoding base16() {
    return BASE16;
  }

  static final class Alphabet {
    private final String name;
    // this is meant to be immutable -- don't modify it!
    private final char[] chars;
    final int mask;
    final int bitsPerChar;
    final int charsPerChunk;
