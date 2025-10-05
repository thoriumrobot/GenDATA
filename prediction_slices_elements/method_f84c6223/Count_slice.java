// Source-based slice around line 67
// Method: <com.google.common.collect.Count: String toString()>

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
