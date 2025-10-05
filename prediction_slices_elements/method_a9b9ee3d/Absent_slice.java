// Source-based slice around line 55
// Method: <com.google.common.base.Absent: Optional or(Optional)>

  }

  @Override
  public T or(T defaultValue) {
    return checkNotNull(defaultValue, "use Optional.orNull() instead of Optional.or(null)");
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
