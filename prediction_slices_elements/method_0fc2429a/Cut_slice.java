// Source-based slice around line 87
// Method: <com.google.common.collect.Cut: C endpoint()>

    }
    int result = Range.compareOrThrow(endpoint, that.endpoint);
    if (result != 0) {
      return result;
    }
    // same value. below comes before above
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
