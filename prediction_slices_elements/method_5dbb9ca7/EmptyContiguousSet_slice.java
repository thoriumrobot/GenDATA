// Source-based slice around line 114
// Method: <com.google.common.collect.EmptyContiguousSet: boolean isEmpty()>

    return emptyIterator();
  }

  @Override
  boolean isPartialView() {
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
