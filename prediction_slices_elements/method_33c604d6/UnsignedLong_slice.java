// Source-based slice around line 238
// Method: <com.google.common.primitives.UnsignedLong: int hashCode()>

  }

  @Override
  public int compareTo(UnsignedLong o) {
    checkNotNull(o);
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
