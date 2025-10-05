// Source-based slice around line 46
// Method: <com.google.common.collect.EmptyContiguousSet: C last()>

    super(domain);
  }

  @Override
  public C first() {
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
