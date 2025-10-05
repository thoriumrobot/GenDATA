// Source-based slice around line 18
// Method: <com.google.common.graph.InvalidatableSet: InvalidatableSet of(Set,Supplier,Supplier)>

/**
 * A subclass of `ForwardingSet` that throws `IllegalStateException` on invocation of any method
 * (except `hashCode` and `equals`) if the provided `Supplier` returns false.
 */
final class InvalidatableSet<E> extends ForwardingSet<E> {
  private final Supplier<Boolean> validator;
  private final Set<E> delegate;
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
