// Source-based slice around line 51
// Method: <com.google.common.collect.EmptyContiguousSet: int size()>

    throw new NoSuchElementException();
  }

  @Override
  public C last() {
    throw new NoSuchElementException();
  }

  @Override
  public int size() {
    return 0;
  }

  @Override
  public ContiguousSet<C> intersection(ContiguousSet<C> other) {
    return this;
  }

  @Override
  public Range<C> range() {
