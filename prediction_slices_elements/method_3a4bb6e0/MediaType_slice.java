// Source-based slice around line 831
// Method: <com.google.common.net.MediaType: String type()>

  private @Nullable Optional<Charset> parsedCharset;

  private MediaType(String type, String subtype, ImmutableListMultimap<String, String> parameters) {
    this.type = type;
    this.subtype = subtype;
    this.parameters = parameters;
  }

  /** Returns the top-level media type. For example, {@code "text"} in {@code "text/plain"}. */
  public String type() {
    return type;
  }

  /** Returns the media subtype. For example, {@code "plain"} in {@code "text/plain"}. */
  public String subtype() {
    return subtype;
  }

  /** Returns a multimap containing the parameters of this media type. */
  public ImmutableListMultimap<String, String> parameters() {
