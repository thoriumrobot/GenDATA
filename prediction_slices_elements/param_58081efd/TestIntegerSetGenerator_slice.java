// Source-based slice around line 51
// Method: <com.google.common.collect.testing.TestIntegerSetGenerator: Integer[] createArray(int)>

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
   * containers with a known order other than insertion order must override this method.
   *
   * <p>Note: This default implementation is overkill (but valid) for an unordered container. An
