// Source-based slice around line 272
// Method: <com.google.common.io.BaseEncoding: CharSequence trimTrailingPadding(CharSequence)>


  abstract int maxEncodedSize(int bytes);

  abstract void encodeTo(Appendable target, byte[] bytes, int off, int len) throws IOException;

  abstract int maxDecodedSize(int chars);

  abstract int decodeTo(byte[] target, CharSequence chars) throws DecodingException;

  CharSequence trimTrailingPadding(CharSequence chars) {
    return checkNotNull(chars);
  }

  // Modified encoding generators

  /**
   * Returns an encoding that behaves equivalently to this encoding, but omits any padding
   * characters as specified by <a href="http://tools.ietf.org/html/rfc4648#section-3.2">RFC 4648
   * section 3.2</a>, Padding of Encoded Data.
   */
