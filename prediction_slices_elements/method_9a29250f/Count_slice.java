// Source-based slice around line 62
// Method: <com.google.common.collect.Count: boolean equals(Object)>

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
    return Integer.toString(value);
  }
}
