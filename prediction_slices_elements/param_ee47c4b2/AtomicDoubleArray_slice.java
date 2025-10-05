// Source-based slice around line 124
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: void lazySet(int,double)>

    longs.set(i, next);
  }

  /**
   * Eventually sets the element at position {@code i} to the given value.
   *
   * @param i the index
   * @param newValue the new value
   */
  public final void lazySet(int i, double newValue) {
    long next = doubleToRawLongBits(newValue);
    longs.lazySet(i, next);
  }

  /**
   * Atomically sets the element at position {@code i} to the given value and returns the old value.
   *
   * @param i the index
   * @param newValue the new value
   * @return the previous value
