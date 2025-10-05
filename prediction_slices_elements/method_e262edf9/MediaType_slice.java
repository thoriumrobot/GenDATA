// Source-based slice around line 857
// Method: <com.google.common.net.MediaType: Optional charset()>


  /**
   * Returns an optional charset for the value of the charset parameter if it is specified.
   *
   * @throws IllegalStateException if multiple charset values have been set for this media type
   * @throws IllegalCharsetNameException if a charset value is present, but illegal
   * @throws UnsupportedCharsetException if a charset value is present, but no support is available
   *     in this instance of the Java virtual machine
   */
  public Optional<Charset> charset() {
    // racy single-check idiom, this is safe because Optional is immutable.
    Optional<Charset> local = parsedCharset;
    if (local == null) {
      String value = null;
      local = Optional.absent();
      for (String currentValue : parameters.get(CHARSET_ATTRIBUTE)) {
        if (value == null) {
          value = currentValue;
          local = Optional.of(Charset.forName(value));
        } else if (!value.equals(currentValue)) {
