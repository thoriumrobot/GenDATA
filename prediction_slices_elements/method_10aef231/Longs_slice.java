// Source-based slice around line 268
// Method: <com.google.common.primitives.Longs: int checkNoOverflow(long)>

    long[] result = new long[checkNoOverflow(length)];
    int pos = 0;
    for (long[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
    return result;
  }

  private static int checkNoOverflow(long result) {
    checkArgument(
        result == (int) result,
        "the total number of elements (%s) in the arrays must fit in an int",
        result);
    return (int) result;
  }

  /**
   * Returns a big-endian representation of {@code value} in an 8-element byte array; equivalent to
   * {@code ByteBuffer.allocate(8).putLong(value).array()}. For example, the input value {@code
