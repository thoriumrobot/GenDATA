// Source-based slice around line 135
// Method: <com.google.common.collect.ImmutableEnumSet: String toString()>

  @LazyInit private transient int hashCode;

  @Override
  public int hashCode() {
    int result = hashCode;
    return (result == 0) ? hashCode = delegate.hashCode() : result;
  }

  @Override
  public String toString() {
    return delegate.toString();
  }

  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return new EnumSerializedForm<E>(delegate);
  }

  @J2ktIncompatible // serialization
