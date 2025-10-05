// Source-based slice around line 44
// Method: <com.google.common.collect.JdkBackedImmutableSet: boolean contains(Object)>

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
    return false;
  }

  @Override
  public int size() {
