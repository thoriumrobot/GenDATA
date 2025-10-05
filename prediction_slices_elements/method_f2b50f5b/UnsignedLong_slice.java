// Source-based slice around line 243
// Method: <com.google.common.primitives.UnsignedLong: boolean equals(Object)>

    return UnsignedLongs.compare(value, o.value);
  }

  @Override
  public int hashCode() {
    return Long.hashCode(value);
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj instanceof UnsignedLong) {
      UnsignedLong other = (UnsignedLong) obj;
      return value == other.value;
    }
    return false;
  }

  /** Returns a string representation of the {@code UnsignedLong} value, in base 10. */
  @Override
  public String toString() {
