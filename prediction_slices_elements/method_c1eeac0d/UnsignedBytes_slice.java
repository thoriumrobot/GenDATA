// Source-based slice around line 180
// Method: <com.google.common.primitives.UnsignedBytes: String toString(byte)>

    }
    return (byte) max;
  }

  /**
   * Returns a string representation of x, where x is treated as unsigned.
   *
   * @since 13.0
   */
  public static String toString(byte x) {
    return toString(x, 10);
  }

  /**
   * Returns a string representation of {@code x} for the given radix, where {@code x} is treated as
   * unsigned.
   *
   * @param x the value to convert to a string.
   * @param radix the radix to use while working with {@code x}
   * @throws IllegalArgumentException if {@code radix} is not between {@link Character#MIN_RADIX}
