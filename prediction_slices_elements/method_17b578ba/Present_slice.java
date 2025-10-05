// Source-based slice around line 46
// Method: <com.google.common.base.Present: T or(T)>

    return true;
  }

  @Override
  public T get() {
    return reference;
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

