// Source-based slice around line 110
// Method: <com.google.common.util.concurrent.AtomicDouble: void lazySet(double)>

    long next = doubleToRawLongBits(newValue);
    value = next;
  }

  /**
   * Eventually sets to the given value.
   *
   * @param newValue the new value
   */
  public final void lazySet(double newValue) {
    long next = doubleToRawLongBits(newValue);
    updater.lazySet(this, next);
  }

  /**
   * Atomically sets to the given value and returns the old value.
   *
   * @param newValue the new value
   * @return the previous value
   */
