// Source-based slice around line 313
// Method: <com.google.common.primitives.Ints: int checkNoOverflow(long)>

    int[] result = new int[checkNoOverflow(length)];
    int pos = 0;
    for (int[] array : arrays) {
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
   * Returns a big-endian representation of {@code value} in a 4-element byte array; equivalent to
   * {@code ByteBuffer.allocate(4).putInt(value).array()}. For example, the input value {@code
