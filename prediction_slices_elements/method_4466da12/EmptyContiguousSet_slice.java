// Source-based slice around line 167
// Method: <com.google.common.collect.EmptyContiguousSet: Object writeReplace()>

      return new EmptyContiguousSet<>(domain);
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  @GwtIncompatible
  @J2ktIncompatible
    @Override
  Object writeReplace() {
    return new SerializedForm<>(domain);
  }

  @GwtIncompatible
  @J2ktIncompatible
    private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  @GwtIncompatible // NavigableSet
