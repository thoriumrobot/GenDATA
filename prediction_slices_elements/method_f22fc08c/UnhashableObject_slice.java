// Source-based slice around line 45
// Method: <com.google.common.collect.testing.UnhashableObject: int hashCode()>

  public boolean equals(@Nullable Object object) {
    if (object instanceof UnhashableObject) {
      UnhashableObject that = (UnhashableObject) object;
      return this.value == that.value;
    }
    return false;
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException();
  }

  // needed because otherwise Object.toString() calls hashCode()
  @Override
  public String toString() {
    return "DontHashMe" + value;
  }

  @Override
