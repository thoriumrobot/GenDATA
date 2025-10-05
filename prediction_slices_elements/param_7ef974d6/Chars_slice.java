// Source-based slice around line 339
// Method: <com.google.common.primitives.Chars: char fromBytes(byte,byte)>

  }

  /**
   * Returns the {@code char} value whose byte representation is the given 2 bytes, in big-endian
   * order; equivalent to {@code Chars.fromByteArray(new byte[] {b1, b2})}.
   *
   * @since 7.0
   */
  @GwtIncompatible // doesn't work
  public static char fromBytes(byte b1, byte b2) {
    return (char) ((b1 << 8) | (b2 & 0xFF));
  }

  /**
   * Returns an array containing the same values as {@code array}, but guaranteed to be of a
   * specified minimum length. If {@code array} already has a length of at least {@code minLength},
   * it is returned directly. Otherwise, a new array of size {@code minLength + padding} is
   * returned, containing the values of {@code array}, and zeroes in the remaining places.
   *
   * @param array the source array
