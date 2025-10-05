// Source-based slice around line 812
// Method: com.google.common.net.MediaType.subtype

   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares this to be the correct
   * media type for SFNT, but {@link #WOFF2 application/font-woff2} may be necessary in certain
   * situations for compatibility.
   *
   * @since 30.0
   */
  public static final MediaType FONT_WOFF2 = createConstant(FONT_TYPE, "woff2");

  private final String type;
  private final String subtype;
  private final ImmutableListMultimap<String, String> parameters;

  @LazyInit private @Nullable String toString;

  @LazyInit private int hashCode;

  // We need to differentiate between "not computed" and "computed to be absent."
  @SuppressWarnings("NullableOptional")
  @LazyInit
  private @Nullable Optional<Charset> parsedCharset;
