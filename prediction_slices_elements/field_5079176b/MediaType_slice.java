// Source-based slice around line 817
// Method: com.google.common.net.MediaType.hashCode

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

  private MediaType(String type, String subtype, ImmutableListMultimap<String, String> parameters) {
    this.type = type;
    this.subtype = subtype;
    this.parameters = parameters;
