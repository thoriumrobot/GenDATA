// Source-based slice around line 250
// Method: <com.google.common.primitives.Booleans: int checkNoOverflow(long)>

    boolean[] result = new boolean[checkNoOverflow(length)];
    int pos = 0;
    for (boolean[] array : arrays) {
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
   * Returns an array containing the same values as {@code array}, but guaranteed to be of a
   * specified minimum length. If {@code array} already has a length of at least {@code minLength},
