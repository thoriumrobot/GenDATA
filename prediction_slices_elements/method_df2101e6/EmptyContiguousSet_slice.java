// Source-based slice around line 139
// Method: <com.google.common.collect.EmptyContiguousSet: boolean isHashCodeFast()>

    if (object instanceof Set) {
      Set<?> that = (Set<?>) object;
      return that.isEmpty();
    }
    return false;
  }

  @GwtIncompatible // not used in GWT
  @Override
  boolean isHashCodeFast() {
    return true;
  }

  @Override
  public int hashCode() {
    return 0;
  }

  @GwtIncompatible
  @J2ktIncompatible
