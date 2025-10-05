// Source-based slice around line 64
// Method: <com.google.common.base.Present: T orNull()>

  }

  @Override
  public T or(Supplier<? extends T> supplier) {
    checkNotNull(supplier);
    return reference;
  }

  @Override
  public T orNull() {
    return reference;
  }

  @Override
  public Set<T> asSet() {
    return Collections.singleton(reference);
  }

  @Override
  public <V> Optional<V> transform(Function<? super T, V> function) {
