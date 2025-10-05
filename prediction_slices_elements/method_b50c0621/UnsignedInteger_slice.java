// Source-based slice around line 226
// Method: <com.google.common.primitives.UnsignedInteger: int hashCode()>

   * other}.
   */
  @Override
  public int compareTo(UnsignedInteger other) {
    checkNotNull(other);
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
