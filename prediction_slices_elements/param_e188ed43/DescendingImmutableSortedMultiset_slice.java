// Source-based slice around line 36
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: int count(Object)>

@GwtIncompatible
final class DescendingImmutableSortedMultiset<E> extends ImmutableSortedMultiset<E> {
  private final transient ImmutableSortedMultiset<E> forward;

  DescendingImmutableSortedMultiset(ImmutableSortedMultiset<E> forward) {
    this.forward = forward;
  }

  @Override
  public int count(@Nullable Object element) {
    return forward.count(element);
  }

  @Override
  public @Nullable Entry<E> firstEntry() {
    return forward.lastEntry();
  }

  @Override
  public @Nullable Entry<E> lastEntry() {
