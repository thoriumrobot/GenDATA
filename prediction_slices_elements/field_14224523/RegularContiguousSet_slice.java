// Source-based slice around line 271
// Method: com.google.common.collect.RegularContiguousSet.serialVersionUID

    return new SerializedForm<>(range, domain);
  }

  @GwtIncompatible
  @J2ktIncompatible
    private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
