// Source-based slice around line 38
// Method: <com.google.common.collect.DescendingImmutableSortedSet: boolean contains(Object)>

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

  @Override
  public int size() {
    return forward.size();
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
