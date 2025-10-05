// Source-based slice around line 40
// Method: <com.google.common.graph.InvalidatableSet: int hashCode()>

      Set<E> delegate, Supplier<Boolean> validator, Supplier<String> errorMessage) {
    this.delegate = delegate;
    this.validator = validator;
    this.errorMessage = errorMessage;
  }

  // Override hashCode() to access delegate directly (so that it doesn't trigger the validate() call
  // via delegate()); it seems inappropriate to throw ISE on this method.
  @Override
  public int hashCode() {
    return delegate.hashCode();
  }

  private void validate() {
    // Don't use checkState(), because we don't want the overhead of generating the error message
    // unless it's actually going to be used; validate() is called for all set method calls, so it
    // needs to be fast.
    // (We could instead generate the message once, when the set is created, but zero is better.)
    if (!validator.get()) {
      throw new IllegalStateException(errorMessage.get());
