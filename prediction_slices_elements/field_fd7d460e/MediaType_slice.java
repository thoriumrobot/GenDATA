// Source-based slice around line 822
// Method: com.google.common.net.MediaType.parsedCharset

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
  }

  /** Returns the top-level media type. For example, {@code "text"} in {@code "text/plain"}. */
  public String type() {
    return type;
