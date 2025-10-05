// Source-based slice around line 51
// Method: <com.google.common.collect.RegularImmutableList: boolean isPartialView()>

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
    return array;
  }

  @Override
  int internalArrayStart() {
