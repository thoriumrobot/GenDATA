// Source-based slice around line 80
// Method: <com.google.common.collect.testing.TestEnumMapGenerator: Iterable order(List)>

  }

  @Override
  public final String[] createValueArray(int length) {
    return new String[length];
  }

  /** Returns the elements sorted in natural order. */
  @Override
  public Iterable<Entry<AnEnum, String>> order(List<Entry<AnEnum, String>> insertionOrder) {
    return orderEntriesByKey(insertionOrder);
  }
}
