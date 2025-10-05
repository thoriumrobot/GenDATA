// Source-based slice around line 66
// Method: <com.google.common.base.Absent: T orNull()>

  }

  @Override
  public T or(Supplier<? extends T> supplier) {
    return checkNotNull(
        supplier.get(), "use Optional.orNull() instead of a Supplier that returns null");
  }

  @Override
  public @Nullable T orNull() {
    return null;
  }

  @Override
  public Set<T> asSet() {
    return Collections.emptySet();
  }

  @Override
  public <V> Optional<V> transform(Function<? super T, V> function) {
