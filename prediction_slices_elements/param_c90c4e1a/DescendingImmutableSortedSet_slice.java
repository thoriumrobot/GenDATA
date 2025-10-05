// Source-based slice around line 92
// Method: <com.google.common.collect.DescendingImmutableSortedSet: E floor(E)>

    throw new AssertionError("should never be called");
  }

  @Override
  public @Nullable E lower(E element) {
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
