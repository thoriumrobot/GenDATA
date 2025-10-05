// Source-based slice around line 206
// Method: <com.google.common.io.BaseEncoding: boolean canDecode(CharSequence)>

    return trunc;
  }

  /**
   * Determines whether the specified character sequence is a valid encoded string according to this
   * encoding.
   *
   * @since 20.0
   */
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
