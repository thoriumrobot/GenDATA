// Source-based slice around line 44
// Method: com.google.common.base.Charsets.US_ASCII

@GwtCompatible
public final class Charsets {

  /**
   * US-ASCII: seven-bit ASCII, the Basic Latin block of the Unicode character set (ISO646-US).
   *
   * @deprecated Use {@link StandardCharsets#US_ASCII} instead.
   */
  @Deprecated @J2ktIncompatible @GwtIncompatible // Charset not supported by GWT
  public static final Charset US_ASCII = StandardCharsets.US_ASCII;

  /**
   * ISO-8859-1: ISO Latin Alphabet Number 1 (ISO-LATIN-1).
   *
   * @deprecated Use {@link StandardCharsets#ISO_8859_1} instead.
   */
  @Deprecated public static final Charset ISO_8859_1 = StandardCharsets.ISO_8859_1;

  /**
   * UTF-8: eight-bit UCS Transformation Format.
