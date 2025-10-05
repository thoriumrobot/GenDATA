// Source-based slice around line 14
// Method: com.google.common.graph.InvalidatableSet.validator

import com.google.common.base.Supplier;
import com.google.common.collect.ForwardingSet;
import java.util.Set;

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
