// Source-based slice around line 841
// Method: <com.google.common.net.MediaType: ImmutableListMultimap parameters()>

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
  }

  /**
   * Returns an optional charset for the value of the charset parameter if it is specified.
   *
