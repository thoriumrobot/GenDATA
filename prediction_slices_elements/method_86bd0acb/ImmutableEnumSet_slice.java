// Source-based slice around line 122
// Method: <com.google.common.collect.ImmutableEnumSet: boolean isHashCodeFast()>

      return true;
    }
    if (object instanceof ImmutableEnumSet) {
      object = ((ImmutableEnumSet<?>) object).delegate;
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
