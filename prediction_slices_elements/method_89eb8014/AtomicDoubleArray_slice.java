// Source-based slice around line 279
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: String toString()>

    }
  }

  /**
   * Returns the String representation of the current values of array.
   *
   * @return the String representation of the current values of array
   */
  @Override
  public String toString() {
    int iMax = length() - 1;
    if (iMax == -1) {
      return "[]";
    }

    // Double.toString(Math.PI).length() == 17
    StringBuilder b = new StringBuilder((17 + 2) * (iMax + 1));
    b.append('[');
    for (int i = 0; ; i++) {
      b.append(longBitsToDouble(longs.get(i)));
