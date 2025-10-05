// Source-based slice around line 304
// Method: <com.google.common.util.concurrent.AtomicDouble: void readObject(ObjectInputStream)>

   * @serialData The current value is emitted (a {@code double}).
   */
  private void writeObject(ObjectOutputStream s) throws IOException {
    s.defaultWriteObject();

    s.writeDouble(get());
  }

  /** Reconstitutes the instance from a stream (that is, deserializes it). */
  private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
    s.defaultReadObject();

    set(s.readDouble());
  }
}
