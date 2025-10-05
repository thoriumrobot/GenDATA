// Source-based slice around line 46
// Method: <com.google.common.collect.SingletonImmutableSet: int size()>

  // compressed oops, a SingletonImmutableSet packs all the way down to the optimal 16 bytes.

  final transient E element;

  SingletonImmutableSet(E element) {
    this.element = Preconditions.checkNotNull(element);
  }

  @Override
  public int size() {
    return 1;
  }

  @Override
  public boolean contains(@Nullable Object target) {
    return element.equals(target);
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
