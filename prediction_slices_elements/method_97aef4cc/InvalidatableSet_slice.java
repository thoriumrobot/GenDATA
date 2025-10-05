// Source-based slice around line 25
// Method: <com.google.common.graph.InvalidatableSet: Set delegate()>

  private final Supplier<String> errorMessage;

  static <E> InvalidatableSet<E> of(
      Set<E> delegate, Supplier<Boolean> validator, Supplier<String> errorMessage) {
    return new InvalidatableSet<>(
        checkNotNull(delegate), checkNotNull(validator), checkNotNull(errorMessage));
  }

  @Override
  protected Set<E> delegate() {
    validate();
    return delegate;
  }

  private InvalidatableSet(
      Set<E> delegate, Supplier<Boolean> validator, Supplier<String> errorMessage) {
    this.delegate = delegate;
    this.validator = validator;
    this.errorMessage = errorMessage;
  }
