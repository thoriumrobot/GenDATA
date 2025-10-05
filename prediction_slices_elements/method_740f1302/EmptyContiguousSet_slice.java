// Source-based slice around line 124
// Method: <com.google.common.collect.EmptyContiguousSet: String toString()>

    return true;
  }

  @Override
  public ImmutableList<C> asList() {
    return ImmutableList.of();
  }

  @Override
  public String toString() {
    return "[]";
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof Set) {
      Set<?> that = (Set<?>) object;
      return that.isEmpty();
    }
    return false;
