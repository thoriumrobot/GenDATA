// Source-based slice around line 901
// Method: <com.google.common.net.MediaType: MediaType withParameters(String,Iterable)>

  }

  /**
   * <em>Replaces</em> all parameters with the given attribute with parameters using the given
   * values. If there are no values, any existing parameters with the given attribute are removed.
   *
   * @throws IllegalArgumentException if either {@code attribute} or {@code values} is invalid
   * @since 24.0
   */
  public MediaType withParameters(String attribute, Iterable<String> values) {
    checkNotNull(attribute);
    checkNotNull(values);
    String normalizedAttribute = normalizeToken(attribute);
    ImmutableListMultimap.Builder<String, String> builder = ImmutableListMultimap.builder();
    for (Entry<String, String> entry : parameters.entries()) {
      String key = entry.getKey();
      if (!normalizedAttribute.equals(key)) {
        builder.put(key, entry.getValue());
      }
    }
