// Source-based slice around line 69
// Method: <com.google.common.base.Present: Set asSet()>

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
    return new Present<>(
        checkNotNull(
            function.apply(reference),
            "the Function passed to Optional.transform() must not return null."));
  }
