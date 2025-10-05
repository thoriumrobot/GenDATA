// Source-based slice around line 836
// Method: <com.google.common.net.MediaType: String subtype()>

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
    return parameters;
  }

  private Map<String, ImmutableMultiset<String>> parametersAsMap() {
    return Maps.transformValues(parameters.asMap(), ImmutableMultiset::copyOf);
