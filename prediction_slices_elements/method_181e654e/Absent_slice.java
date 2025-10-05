// Source-based slice around line 60
// Method: <com.google.common.base.Absent: T or(Supplier)>

  }

  @SuppressWarnings("unchecked") // safe covariant cast
  @Override
  public Optional<T> or(Optional<? extends T> secondChoice) {
    return (Optional<T>) checkNotNull(secondChoice);
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
