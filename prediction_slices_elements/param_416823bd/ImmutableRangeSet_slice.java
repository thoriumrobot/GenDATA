// Source-based slice around line 887
// Method: <com.google.common.collect.ImmutableRangeSet: void readObject(ObjectInputStream)>

    }
  }

  @J2ktIncompatible // java.io.ObjectInputStream
  Object writeReplace() {
    return new SerializedForm<C>(ranges);
  }

  @J2ktIncompatible // java.io.ObjectInputStream
  private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }
}
