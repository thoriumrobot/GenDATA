// Source-based slice around line 231
// Method: <com.google.common.primitives.UnsignedInteger: boolean equals(Object)>

    return compare(value, other.value);
  }

  @Override
  public int hashCode() {
    return value;
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj instanceof UnsignedInteger) {
      UnsignedInteger other = (UnsignedInteger) obj;
      return value == other.value;
    }
    return false;
  }

  /** Returns a string representation of the {@code UnsignedInteger} value, in base 10. */
  @Override
  public String toString() {
