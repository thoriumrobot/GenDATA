// Source-based slice around line 80
// Method: <com.google.common.collect.ImmutableAsList: void readObject(ObjectInputStream)>

    Object readResolve() {
      return collection.asList();
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  @GwtIncompatible
  @J2ktIncompatible
    private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  @GwtIncompatible
  @J2ktIncompatible
    @Override
  Object writeReplace() {
    return new SerializedForm(delegateCollection());
  }
}
