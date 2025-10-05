// Source-based slice around line 119
// Method: <com.google.common.collect.EmptyContiguousSet: ImmutableList asList()>

    return false;
  }

  @Override
  public boolean isEmpty() {
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
