// Source-based slice around line 39
// Method: <com.google.common.collect.JdkBackedImmutableSet: E get(int)>

  private final Set<?> delegate;
  private final ImmutableList<E> delegateList;

  JdkBackedImmutableSet(Set<?> delegate, ImmutableList<E> delegateList) {
    this.delegate = delegate;
    this.delegateList = delegateList;
  }

  @Override
  E get(int index) {
    return delegateList.get(index);
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return delegate.contains(object);
  }

  @Override
  boolean isPartialView() {
