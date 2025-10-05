// Source-based slice around line 956
// Method: <com.google.common.net.MediaType: boolean hasWildcard()>

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
  }

  /**
   * Returns {@code true} if this instance falls within the range (as defined by <a
   * href="http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html">the HTTP Accept header</a>) given
   * by the argument according to three criteria:
   *
   * <ol>
   *   <li>The type of the argument is the wildcard or equal to the type of this instance.
