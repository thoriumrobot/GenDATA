// Source-based slice around line 989
// Method: <com.google.common.net.MediaType: boolean is(MediaType)>

   * PLAIN_TEXT_UTF_8.withoutParameters().is(ANY_TEXT_TYPE.withCharset(UTF_8)) // false
   * PLAIN_TEXT_UTF_8.is(ANY_TEXT_TYPE.withCharset(UTF_16)) // false
   * }
   *
   * <p>Note that while it is possible to have the same parameter declared multiple times within a
   * media type this method does not consider the number of occurrences of a parameter. For example,
   * {@code "text/plain; charset=UTF-8"} satisfies {@code "text/plain; charset=UTF-8;
   * charset=UTF-8"}.
   */
  public boolean is(MediaType mediaTypeRange) {
    return (mediaTypeRange.type.equals(WILDCARD) || mediaTypeRange.type.equals(this.type))
        && (mediaTypeRange.subtype.equals(WILDCARD) || mediaTypeRange.subtype.equals(this.subtype))
        && this.parameters.entries().containsAll(mediaTypeRange.parameters.entries());
  }

  /**
   * Creates a new media type with the given type and subtype.
   *
   * @throws IllegalArgumentException if type or subtype is invalid or if a wildcard is used for the
   *     type, but not the subtype.
