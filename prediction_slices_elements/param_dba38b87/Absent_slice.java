// Source-based slice around line 76
// Method: <com.google.common.base.Absent: Optional transform(Function)>

    return null;
  }

  @Override
  public Set<T> asSet() {
    return Collections.emptySet();
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
