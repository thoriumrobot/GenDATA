// Source-based slice around line 41
// Method: <com.google.common.base.Present: T get()>

    this.reference = reference;
  }

  @Override
  public boolean isPresent() {
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
