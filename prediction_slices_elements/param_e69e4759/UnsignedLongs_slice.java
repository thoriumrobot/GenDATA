// Source-based slice around line 444
// Method: <com.google.common.primitives.UnsignedLongs: String toString(long)>

      return true;
    }
  }

  /**
   * Returns a string representation of x, where x is treated as unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Long#toUnsignedString(long)} instead.
   */
  public static String toString(long x) {
    return toString(x, 10);
  }

  /**
   * Returns a string representation of {@code x} for the given radix, where {@code x} is treated as
   * unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Long#toUnsignedString(long, int)} instead.
   *
   * @param x the value to convert to a string.
