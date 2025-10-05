// Source-based slice around line 102
// Method: <com.google.common.collect.DescendingImmutableSortedSet: E higher(E)>

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
    int index = forward.indexOf(target);
    if (index == -1) {
      return index;
    } else {
      return size() - 1 - index;
