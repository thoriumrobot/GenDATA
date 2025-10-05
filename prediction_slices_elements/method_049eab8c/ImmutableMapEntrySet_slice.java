// Source-based slice around line 120
// Method: <com.google.common.collect.ImmutableMapEntrySet: boolean isHashCodeFast()>

  }

  @Override
  boolean isPartialView() {
    return map().isPartialView();
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
