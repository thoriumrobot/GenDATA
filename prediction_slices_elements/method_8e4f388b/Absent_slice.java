// Source-based slice around line 49
// Method: <com.google.common.base.Absent: T or(T)>

    return false;
  }

  @Override
  public T get() {
    throw new IllegalStateException("Optional.get() cannot be called on an absent value");
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
