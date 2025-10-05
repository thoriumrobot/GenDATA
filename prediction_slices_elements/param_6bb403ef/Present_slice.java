// Source-based slice around line 58
// Method: <com.google.common.base.Present: T or(Supplier)>

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

  @Override
  public T orNull() {
    return reference;
  }

  @Override
