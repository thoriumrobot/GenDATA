// Source-based slice around line 881
// Method: <com.google.common.net.MediaType: MediaType withoutParameters()>

      parsedCharset = local;
    }
    return local;
  }

  /**
   * Returns a new instance with the same type and subtype as this instance, but without any
   * parameters.
   */
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
