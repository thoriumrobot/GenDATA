// Source-based slice around line 104
// Method: <com.google.common.collect.ImmutableMapEntrySet: boolean contains(Object)>


  abstract ImmutableMap<K, V> map();

  @Override
  public int size() {
    return map().size();
  }

  @Override
  public boolean contains(@Nullable Object object) {
    if (object instanceof Entry) {
      Entry<?, ?> entry = (Entry<?, ?>) object;
      V value = map().get(entry.getKey());
      return value != null && value.equals(entry.getValue());
    }
    return false;
  }

  @Override
  boolean isPartialView() {
