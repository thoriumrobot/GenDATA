// Source-based slice around line 52
// Method: <com.google.common.base.Present: Optional or(Optional)>

  }

  @Override
  public T or(T defaultValue) {
    checkNotNull(defaultValue, "use Optional.orNull() instead of Optional.or(null)");
    return reference;
  }

  @Override
  public Optional<T> or(Optional<? extends T> secondChoice) {
    checkNotNull(secondChoice);
    return this;
  }

  @Override
  public T or(Supplier<? extends T> supplier) {
    checkNotNull(supplier);
    return reference;
  }

