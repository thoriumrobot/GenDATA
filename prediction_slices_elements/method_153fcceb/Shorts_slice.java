// Source-based slice around line 303
// Method: <com.google.common.primitives.Shorts: int checkNoOverflow(long)>

    short[] result = new short[checkNoOverflow(length)];
    int pos = 0;
    for (short[] array : arrays) {
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
   * Returns a big-endian representation of {@code value} in a 2-element byte array; equivalent to
   * {@code ByteBuffer.allocate(2).putShort(value).array()}. For example, the input value {@code
