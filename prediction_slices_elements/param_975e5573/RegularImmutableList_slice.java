// Source-based slice around line 71
// Method: <com.google.common.collect.RegularImmutableList: int copyIntoArray(Object[],int)>

    return 0;
  }

  @Override
  int internalArrayEnd() {
    return array.length;
  }

  @Override
  int copyIntoArray(@Nullable Object[] dst, int dstOff) {
    arraycopy(array, 0, dst, dstOff, array.length);
    return dstOff + array.length;
  }

  // The fake cast to E is safe because the creation methods only allow E's
  @Override
  @SuppressWarnings("unchecked")
  public E get(int index) {
    return (E) array[index];
  }
