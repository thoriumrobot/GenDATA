// Source-based slice around line 57
// Method: <com.google.common.collect.Count: int hashCode()>

  }

  public int getAndSet(int newValue) {
    int result = value;
    value = newValue;
    return result;
  }

  @Override
  public int hashCode() {
    return value;
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    return obj instanceof Count && ((Count) obj).value == value;
  }

  @Override
  public String toString() {
