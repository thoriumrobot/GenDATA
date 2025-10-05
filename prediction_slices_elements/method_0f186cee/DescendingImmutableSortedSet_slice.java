// Source-based slice around line 97
// Method: <com.google.common.collect.DescendingImmutableSortedSet: E ceiling(E)>

    return forward.higher(element);
  }

  @Override
  public @Nullable E floor(E element) {
    return forward.ceiling(element);
  }

  @Override
  public @Nullable E ceiling(E element) {
    return forward.floor(element);
  }

  @Override
  public @Nullable E higher(E element) {
    return forward.lower(element);
  }

  @Override
  int indexOf(@Nullable Object target) {
