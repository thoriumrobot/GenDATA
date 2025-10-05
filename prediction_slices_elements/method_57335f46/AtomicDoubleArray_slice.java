// Source-based slice around line 93
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: int length()>

    }
    this.longs = new AtomicLongArray(longArray);
  }

  /**
   * Returns the length of the array.
   *
   * @return the length of the array
   */
  public final int length() {
    return longs.length();
  }

  /**
   * Gets the current value at position {@code i}.
   *
   * @param i the index
   * @return the current value
   */
  public final double get(int i) {
