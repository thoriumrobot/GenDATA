// Source-based slice around line 69
// Method: <com.google.common.collect.testing.TestEnumMapGenerator: AnEnum[] createKeyArray(int)>

  protected abstract Map<AnEnum, String> create(Entry<AnEnum, String>[] entries);

  @Override
  @SuppressWarnings("unchecked")
  public final Entry<AnEnum, String>[] createArray(int length) {
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
