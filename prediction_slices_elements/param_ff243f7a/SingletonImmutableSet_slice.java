// Source-based slice around line 51
// Method: <com.google.common.collect.SingletonImmutableSet: boolean contains(Object)>

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
    return singletonIterator(element);
  }

  @Override
  public ImmutableList<E> asList() {
