// Source-based slice around line 30
// Method: com.google.common.collect.DescendingImmutableSortedSet.forward

import org.jspecify.annotations.Nullable;

/**
 * Skeletal implementation of {@link ImmutableSortedSet#descendingSet()}.
 *
 * @author Louis Wasserman
 */
@GwtIncompatible
final class DescendingImmutableSortedSet<E> extends ImmutableSortedSet<E> {
  private final ImmutableSortedSet<E> forward;

  DescendingImmutableSortedSet(ImmutableSortedSet<E> forward) {
    super(Ordering.from(forward.comparator()).reverse());
    this.forward = forward;
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return forward.contains(object);
  }
