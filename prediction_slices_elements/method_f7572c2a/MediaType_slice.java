// Source-based slice around line 845
// Method: <com.google.common.net.MediaType: Map parametersAsMap()>

  public String subtype() {
    return subtype;
  }

  /** Returns a multimap containing the parameters of this media type. */
  public ImmutableListMultimap<String, String> parameters() {
    return parameters;
  }

  private Map<String, ImmutableMultiset<String>> parametersAsMap() {
    return Maps.transformValues(parameters.asMap(), ImmutableMultiset::copyOf);
  }

  /**
   * Returns an optional charset for the value of the charset parameter if it is specified.
   *
   * @throws IllegalStateException if multiple charset values have been set for this media type
   * @throws IllegalCharsetNameException if a charset value is present, but illegal
   * @throws UnsupportedCharsetException if a charset value is present, but no support is available
   *     in this instance of the Java virtual machine
