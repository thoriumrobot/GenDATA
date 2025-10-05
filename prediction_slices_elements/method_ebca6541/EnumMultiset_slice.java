// Source-based slice around line 121
// Method: <com.google.common.collect.EnumMultiset: int distinctElements()>

   */
  private void checkIsE(Object element) {
    checkNotNull(element);
    if (!isActuallyE(element)) {
      throw new ClassCastException("Expected an " + type + " but got " + element);
    }
  }

  @Override
  int distinctElements() {
    return distinctElements;
  }

  @Override
  public int size() {
    return Ints.saturatedCast(size);
  }

  @Override
  public int count(@Nullable Object element) {
