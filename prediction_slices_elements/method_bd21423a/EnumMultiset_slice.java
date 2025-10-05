// Source-based slice around line 299
// Method: <com.google.common.collect.EnumMultiset: void writeObject(ObjectOutputStream)>

    }
  }

  @Override
  public Iterator<E> iterator() {
    return Multisets.iteratorImpl(this);
  }

  @GwtIncompatible // java.io.ObjectOutputStream
  private void writeObject(ObjectOutputStream stream) throws IOException {
    stream.defaultWriteObject();
    stream.writeObject(type);
    Serialization.writeMultiset(this, stream);
  }

  /**
   * @serialData the {@code Class<E>} for the enum type, the number of distinct elements, the first
   *     element, its count, the second element, its count, and so on
   */
  @GwtIncompatible // java.io.ObjectInputStream
