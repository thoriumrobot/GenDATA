// Source-based slice around line 288
// Method: <com.google.common.util.concurrent.AtomicDouble: double doubleValue()>

   * conversion.
   */
  @Override
  public float floatValue() {
    return (float) get();
  }

  /** Returns the value of this {@code AtomicDouble} as a {@code double}. */
  @Override
  public double doubleValue() {
    return get();
  }

  /**
   * Saves the state to a stream (that is, serializes it).
   *
   * @serialData The current value is emitted (a {@code double}).
   */
  private void writeObject(ObjectOutputStream s) throws IOException {
    s.defaultWriteObject();
