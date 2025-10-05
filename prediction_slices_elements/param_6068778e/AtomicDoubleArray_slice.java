// Source-based slice around line 317
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: void readObject(ObjectInputStream)>

    s.writeInt(length);

    // Write out all elements in the proper order.
    for (int i = 0; i < length; i++) {
      s.writeDouble(get(i));
    }
  }

  /** Reconstitutes the instance from a stream (that is, deserializes it). */
  private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
    s.defaultReadObject();

    int length = s.readInt();
    ImmutableLongArray.Builder builder = ImmutableLongArray.builder();
    for (int i = 0; i < length; i++) {
      builder.add(doubleToRawLongBits(s.readDouble()));
    }
    this.longs = new AtomicLongArray(builder.build().toArray());
  }
}
