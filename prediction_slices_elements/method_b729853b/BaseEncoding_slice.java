// Source-based slice around line 215
// Method: <com.google.common.io.BaseEncoding: byte[] decode(CharSequence)>

  public abstract boolean canDecode(CharSequence chars);

  /**
   * Decodes the specified character sequence, and returns the resulting {@code byte[]}. This is the
   * inverse operation to {@link #encode(byte[])}.
   *
   * @throws IllegalArgumentException if the input is not a valid encoded string according to this
   *     encoding.
   */
  public final byte[] decode(CharSequence chars) {
    try {
      return decodeChecked(chars);
    } catch (DecodingException badInput) {
      throw new IllegalArgumentException(badInput);
    }
  }

  /**
   * Decodes the specified character sequence, and returns the resulting {@code byte[]}. This is the
   * inverse operation to {@link #encode(byte[])}.
