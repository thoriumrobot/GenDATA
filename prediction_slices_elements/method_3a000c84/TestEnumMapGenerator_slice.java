// Source-based slice around line 60
// Method: <com.google.common.collect.testing.TestEnumMapGenerator: Map create(Entry[])>

    int i = 0;
    for (Object o : entries) {
      @SuppressWarnings("unchecked")
      Entry<AnEnum, String> e = (Entry<AnEnum, String>) o;
      array[i++] = e;
    }
    return create(array);
  }

  protected abstract Map<AnEnum, String> create(Entry<AnEnum, String>[] entries);

  @Override
  @SuppressWarnings("unchecked")
  public final Entry<AnEnum, String>[] createArray(int length) {
    return (Entry<AnEnum, String>[]) new Entry<?, ?>[length];
  }

  @Override
  public final AnEnum[] createKeyArray(int length) {
    return new AnEnum[length];
