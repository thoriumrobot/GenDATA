// Source-based slice around line 48
// Method: <com.google.common.collect.testing.TestEnumMapGenerator: Map create(Object)>

    return new SampleElements<>(
        mapEntry(AnEnum.A, "January"),
        mapEntry(AnEnum.B, "February"),
        mapEntry(AnEnum.C, "March"),
        mapEntry(AnEnum.D, "April"),
        mapEntry(AnEnum.E, "May"));
  }

  @Override
  public final Map<AnEnum, String> create(Object... entries) {
    @SuppressWarnings("unchecked")
    Entry<AnEnum, String>[] array = (Entry<AnEnum, String>[]) new Entry<?, ?>[entries.length];
    int i = 0;
    for (Object o : entries) {
      @SuppressWarnings("unchecked")
      Entry<AnEnum, String> e = (Entry<AnEnum, String>) o;
      array[i++] = e;
    }
    return create(array);
  }
