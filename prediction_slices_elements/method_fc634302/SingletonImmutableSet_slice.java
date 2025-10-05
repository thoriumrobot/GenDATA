// Source-based slice around line 56
// Method: <com.google.common.collect.SingletonImmutableSet: UnmodifiableIterator iterator()>

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
    return ImmutableList.of(element);
  }

  @Override
  boolean isPartialView() {
