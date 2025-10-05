// Source-based slice around line 890
// Method: <com.google.common.net.MediaType: MediaType withParameters(Multimap)>

  public MediaType withoutParameters() {
    return parameters.isEmpty() ? this : create(type, subtype);
  }

  /**
   * <em>Replaces</em> all parameters with the given parameters.
   *
   * @throws IllegalArgumentException if any parameter or value is invalid
   */
  public MediaType withParameters(Multimap<String, String> parameters) {
    return create(type, subtype, parameters);
  }

  /**
   * <em>Replaces</em> all parameters with the given attribute with parameters using the given
   * values. If there are no values, any existing parameters with the given attribute are removed.
   *
   * @throws IllegalArgumentException if either {@code attribute} or {@code values} is invalid
   * @since 24.0
   */
