// Source-based slice around line 82
// Method: <com.google.common.base.Present: boolean equals(Object)>

  @Override
  public <V> Optional<V> transform(Function<? super T, V> function) {
    return new Present<>(
        checkNotNull(
            function.apply(reference),
            "the Function passed to Optional.transform() must not return null."));
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj instanceof Present) {
      Present<?> other = (Present<?>) obj;
      return reference.equals(other.reference);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return 0x598df91c + reference.hashCode();
