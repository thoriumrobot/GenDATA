// Source-based slice around line 126
// Method: com.google.common.collect.ImmutableEnumSet.hashCode

    }
    return delegate.equals(object);
  }

  @Override
  boolean isHashCodeFast() {
    return true;
  }

  @LazyInit private transient int hashCode;

  @Override
  public int hashCode() {
    int result = hashCode;
    return (result == 0) ? hashCode = delegate.hashCode() : result;
  }

  @Override
  public String toString() {
    return delegate.toString();
