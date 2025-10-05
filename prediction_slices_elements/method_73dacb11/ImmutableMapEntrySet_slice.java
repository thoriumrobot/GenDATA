// Source-based slice around line 125
// Method: <com.google.common.collect.ImmutableMapEntrySet: int hashCode()>

  }

  @Override
  @GwtIncompatible // not used in GWT
  boolean isHashCodeFast() {
    return map().isHashCodeFast();
  }

  @Override
  public int hashCode() {
    return map().hashCode();
  }

  @GwtIncompatible
  @J2ktIncompatible
    @Override
  Object writeReplace() {
    return new EntrySetSerializedForm<>(map());
  }

