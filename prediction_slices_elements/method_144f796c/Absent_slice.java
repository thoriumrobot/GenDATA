// Source-based slice around line 87
// Method: <com.google.common.base.Absent: int hashCode()>

    return Optional.absent();
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    return this == obj;
  }

  @Override
  public int hashCode() {
    return 0x79a31aac;
  }

  @Override
  public String toString() {
    return "Optional.absent()";
  }

  private Object readResolve() {
    return INSTANCE;
