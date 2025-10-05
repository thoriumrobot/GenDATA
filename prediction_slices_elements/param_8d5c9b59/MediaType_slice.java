// Source-based slice around line 947
// Method: <com.google.common.net.MediaType: MediaType withCharset(Charset)>

  /**
   * Returns a new instance with the same type and subtype as this instance, with the {@code
   * charset} parameter set to the {@link Charset#name name} of the given charset. Only one {@code
   * charset} parameter will be present on the new instance regardless of the number set on this
   * one.
   *
   * <p>If a charset must be specified that is not supported on this JVM (and thus is not
   * representable as a {@link Charset} instance), use {@link #withParameter}.
   */
  public MediaType withCharset(Charset charset) {
    checkNotNull(charset);
    MediaType withCharset = withParameter(CHARSET_ATTRIBUTE, charset.name());
    // precache the charset so we don't need to parse it
    withCharset.parsedCharset = Optional.of(charset);
    return withCharset;
  }

  /** Returns true if either the type or subtype is the wildcard. */
  public boolean hasWildcard() {
    return type.equals(WILDCARD) || subtype.equals(WILDCARD);
