// Source-based slice around line 934
// Method: <com.google.common.net.MediaType: MediaType withParameter(String,String)>


  /**
   * <em>Replaces</em> all parameters with the given attribute with a single parameter with the
   * given value. If multiple parameters with the same attributes are necessary use {@link
   * #withParameters(String, Iterable)}. Prefer {@link #withCharset} for setting the {@code charset}
   * parameter when using a {@link Charset} object.
   *
   * @throws IllegalArgumentException if either {@code attribute} or {@code value} is invalid
   */
  public MediaType withParameter(String attribute, String value) {
    return withParameters(attribute, ImmutableSet.of(value));
  }

  /**
   * Returns a new instance with the same type and subtype as this instance, with the {@code
   * charset} parameter set to the {@link Charset#name name} of the given charset. Only one {@code
   * charset} parameter will be present on the new instance regardless of the number set on this
   * one.
   *
   * <p>If a charset must be specified that is not supported on this JVM (and thus is not
