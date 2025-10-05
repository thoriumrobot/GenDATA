// Source-based slice around line 83
// Method: com.google.common.base.Charsets.UTF_16

  public static final Charset UTF_16LE = StandardCharsets.UTF_16LE;

  /**
   * UTF-16: sixteen-bit UCS Transformation Format, byte order identified by an optional byte-order
   * mark.
   *
   * @deprecated Use {@link StandardCharsets#UTF_16} instead.
   */
  @Deprecated @J2ktIncompatible @GwtIncompatible // Charset not supported by GWT
  public static final Charset UTF_16 = StandardCharsets.UTF_16;

  private Charsets() {}
}
