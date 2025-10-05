// Source-based slice around line 74
// Method: <com.google.common.collect.testing.TestEnumMapGenerator: String[] createValueArray(int)>

    return (Entry<AnEnum, String>[]) new Entry<?, ?>[length];
  }

  @Override
  public final AnEnum[] createKeyArray(int length) {
    return new AnEnum[length];
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
