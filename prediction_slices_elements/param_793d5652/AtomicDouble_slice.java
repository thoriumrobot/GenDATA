// Source-based slice around line 121
// Method: <com.google.common.util.concurrent.AtomicDouble: double getAndSet(double)>

    updater.lazySet(this, next);
  }

  /**
   * Atomically sets to the given value and returns the old value.
   *
   * @param newValue the new value
   * @return the previous value
   */
  public final double getAndSet(double newValue) {
    long next = doubleToRawLongBits(newValue);
    return longBitsToDouble(updater.getAndSet(this, next));
  }

  /**
   * Atomically sets the value to the given updated value if the current value is <a
   * href="#bitEquals">bitwise equal</a> to the expected value.
   *
   * @param expect the expected value
   * @param update the new value
