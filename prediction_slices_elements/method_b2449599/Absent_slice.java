// Source-based slice around line 71
// Method: <com.google.common.base.Absent: Set asSet()>

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
    checkNotNull(function);
    return Optional.absent();
  }

  @Override
