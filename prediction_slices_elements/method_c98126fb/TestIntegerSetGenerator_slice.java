// Source-based slice around line 48
// Method: <com.google.common.collect.testing.TestIntegerSetGenerator: Set create(Integer[])>

  public Set<Integer> create(Object... elements) {
    Integer[] array = new Integer[elements.length];
    int i = 0;
    for (Object e : elements) {
      array[i++] = (Integer) e;
    }
    return create(array);
  }

  protected abstract Set<Integer> create(Integer[] elements);

  @Override
  public Integer[] createArray(int length) {
    return new Integer[length];
  }

  /**
   * {@inheritDoc}
   *
   * <p>By default, returns the supplied elements in their given order; however, generators for
