// Source-based slice around line 320
// Method: com.google.common.collect.EnumMultiset.serialVersionUID

    stream.defaultReadObject();
    @SuppressWarnings("unchecked") // reading data stored by writeObject
    Class<E> localType = (Class<E>) requireNonNull(stream.readObject());
    type = localType;
    enumConstants = type.getEnumConstants();
    counts = new int[enumConstants.length];
    Serialization.populateMultiset(this, stream);
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
