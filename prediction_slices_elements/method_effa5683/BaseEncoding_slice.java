// Source-based slice around line 230
// Method: <com.google.common.io.BaseEncoding: byte[] decodeChecked(CharSequence)>

  }

  /**
   * Decodes the specified character sequence, and returns the resulting {@code byte[]}. This is the
   * inverse operation to {@link #encode(byte[])}.
   *
   * @throws DecodingException if the input is not a valid encoded string according to this
   *     encoding.
   */
  final byte[] decodeChecked(CharSequence chars)
      throws DecodingException {
    chars = trimTrailingPadding(chars);
    byte[] tmp = new byte[maxDecodedSize(chars.length())];
    int len = decodeTo(tmp, chars);
    return extract(tmp, len);
  }

  /**
   * Returns an {@code InputStream} that decodes base-encoded input from the specified {@code
   * Reader}. The returned stream throws a {@link DecodingException} upon decoding-specific errors.
