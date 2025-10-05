// Source-based slice around line 74
// Method: <com.google.common.base.Present: Optional transform(Function)>

    return reference;
  }

  @Override
  public Set<T> asSet() {
    return Collections.singleton(reference);
  }

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
