// Source-based slice around line 1219
// Method: com.google.common.net.MediaType.PARAMETER_JOINER

    // racy single-check idiom
    int h = hashCode;
    if (h == 0) {
      h = hash(type, subtype, parametersAsMap());
      hashCode = h;
    }
    return h;
  }

  private static final MapJoiner PARAMETER_JOINER = Joiner.on("; ").withKeyValueSeparator("=");

  /**
   * Returns the string representation of this media type in the format described in <a
   * href="http://www.ietf.org/rfc/rfc2045.txt">RFC 2045</a>.
   */
  @Override
  public String toString() {
    // racy single-check idiom, safe because String is immutable
    String result = toString;
    if (result == null) {
