// Source-based slice around line 82
// Method: <com.google.common.base.Absent: boolean equals(Object)>

  }

  @Override
  public <V> Optional<V> transform(Function<? super T, V> function) {
    checkNotNull(function);
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
