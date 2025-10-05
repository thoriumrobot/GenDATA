// Source-based slice around line 93
// Method: <com.google.common.collect.Cut: boolean equals(Object)>

    return Boolean.compare(this instanceof AboveValue, that instanceof AboveValue);
  }

  C endpoint() {
    return endpoint;
  }

  @SuppressWarnings("unchecked") // catching CCE
  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj instanceof Cut) {
      // It might not really be a Cut<C>, but we'll catch a CCE if it's not
      Cut<C> that = (Cut<C>) obj;
      try {
        int compareResult = compareTo(that);
        return compareResult == 0;
      } catch (ClassCastException wastNotComparableToOurType) {
        return false;
      }
    }
