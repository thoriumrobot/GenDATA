// Source-based slice around line 51
// Method: <com.google.common.collect.testing.TestStringSetGenerator: String[] createArray(int)>

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
   * containers with a known order other than insertion order must override this method.
   *
   * <p>Note: This default implementation is overkill (but valid) for an unordered container. An
