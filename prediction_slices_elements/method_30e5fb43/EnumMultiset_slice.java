// Source-based slice around line 310
// Method: <com.google.common.collect.EnumMultiset: void readObject(ObjectInputStream)>

    stream.writeObject(type);
    Serialization.writeMultiset(this, stream);
  }

  /**
   * @serialData the {@code Class<E>} for the enum type, the number of distinct elements, the first
   *     element, its count, the second element, its count, and so on
   */
  @GwtIncompatible // java.io.ObjectInputStream
  private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
    stream.defaultReadObject();
    @SuppressWarnings("unchecked") // reading data stored by writeObject
    Class<E> localType = (Class<E>) requireNonNull(stream.readObject());
    type = localType;
    enumConstants = type.getEnumConstants();
    counts = new int[enumConstants.length];
    Serialization.populateMultiset(this, stream);
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
