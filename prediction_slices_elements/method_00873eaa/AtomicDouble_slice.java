// Source-based slice around line 255
// Method: <com.google.common.util.concurrent.AtomicDouble: String toString()>

    }
  }

  /**
   * Returns the String representation of the current value.
   *
   * @return the String representation of the current value
   */
  @Override
  public String toString() {
    return Double.toString(get());
  }

  /**
   * Returns the value of this {@code AtomicDouble} as an {@code int} after a narrowing primitive
   * conversion.
   */
  @Override
  public int intValue() {
    return (int) get();
