// Source-based slice around line 303
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: void writeObject(ObjectOutputStream)>

    }
  }

  /**
   * Saves the state to a stream (that is, serializes it).
   *
   * @serialData The length of the array is emitted (int), followed by all of its elements (each a
   *     {@code double}) in the proper order.
   */
  private void writeObject(ObjectOutputStream s) throws IOException {
    s.defaultWriteObject();

    // Write out array length
    int length = length();
    s.writeInt(length);

    // Write out all elements in the proper order.
    for (int i = 0; i < length; i++) {
      s.writeDouble(get(i));
    }
