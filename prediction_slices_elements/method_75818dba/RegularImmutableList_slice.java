// Source-based slice around line 61
// Method: <com.google.common.collect.RegularImmutableList: int internalArrayStart()>

    return false;
  }

  @Override
  Object[] internalArray() {
    return array;
  }

  @Override
  int internalArrayStart() {
    return 0;
  }

  @Override
  int internalArrayEnd() {
    return array.length;
  }

  @Override
  int copyIntoArray(@Nullable Object[] dst, int dstOff) {
