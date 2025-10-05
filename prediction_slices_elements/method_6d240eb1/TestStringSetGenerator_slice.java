// Source-based slice around line 48
// Method: <com.google.common.collect.testing.TestStringSetGenerator: Set create(String[])>

  public Set<String> create(Object... elements) {
    String[] array = new String[elements.length];
    int i = 0;
    for (Object e : elements) {
      array[i++] = (String) e;
    }
    return create(array);
  }

  protected abstract Set<String> create(String[] elements);

  @Override
  public String[] createArray(int length) {
    return new String[length];
  }

  /**
   * {@inheritDoc}
   *
   * <p>By default, returns the supplied elements in their given order; however, generators for
