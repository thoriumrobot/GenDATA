// Source-based slice around line 61
// Method: <com.google.common.collect.SingletonImmutableSet: ImmutableList asList()>

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
    return false;
  }

  @Override
  int copyIntoArray(@Nullable Object[] dst, int offset) {
