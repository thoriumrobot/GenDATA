// Source-based slice around line 571
// Method: <com.google.common.collect.ImmutableMultimap: boolean containsValue(Object)>


  // accessors

  @Override
  public boolean containsKey(@Nullable Object key) {
    return map.containsKey(key);
  }

  @Override
  public boolean containsValue(@Nullable Object value) {
    return value != null && super.containsValue(value);
  }

  @Override
  public int size() {
    return size;
  }

  // views

