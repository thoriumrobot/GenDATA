// Source-based slice around line 566
// Method: <com.google.common.collect.ImmutableMultimap: boolean containsKey(Object)>

   * memory leaks.
   */
  boolean isPartialView() {
    return map.isPartialView();
  }

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
