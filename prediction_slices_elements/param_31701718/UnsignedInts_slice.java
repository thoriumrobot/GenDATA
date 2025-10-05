// Source-based slice around line 376
// Method: <com.google.common.primitives.UnsignedInts: String toString(int)>

    }
    return (int) result;
  }

  /**
   * Returns a string representation of x, where x is treated as unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#toUnsignedString(int)} instead.
   */
  public static String toString(int x) {
    return toString(x, 10);
  }

  /**
   * Returns a string representation of {@code x} for the given radix, where {@code x} is treated as
   * unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#toUnsignedString(int, int)} instead.
   *
   * @param x the value to convert to a string.
