// Source-based slice around line 46
// Method: <com.google.common.collect.RegularImmutableList: int size()>

  static final ImmutableList<Object> EMPTY = new RegularImmutableList<>(new Object[0]);

  @VisibleForTesting final transient Object[] array;

  RegularImmutableList(Object[] array) {
    this.array = array;
  }

  @Override
  public int size() {
    return array.length;
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  Object[] internalArray() {
