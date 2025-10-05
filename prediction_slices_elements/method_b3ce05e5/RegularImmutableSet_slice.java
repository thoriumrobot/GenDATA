// Source-based slice around line 101
// Method: <com.google.common.collect.RegularImmutableSet: int internalArrayEnd()>

    return elements;
  }

  @Override
  int internalArrayStart() {
    return 0;
  }

  @Override
  int internalArrayEnd() {
    return elements.length;
  }

  @Override
  int copyIntoArray(@Nullable Object[] dst, int offset) {
    arraycopy(elements, 0, dst, offset, elements.length);
    return offset + elements.length;
  }

  @Override
