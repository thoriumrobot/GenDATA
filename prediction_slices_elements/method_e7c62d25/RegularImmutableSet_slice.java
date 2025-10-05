// Source-based slice around line 96
// Method: <com.google.common.collect.RegularImmutableSet: int internalArrayStart()>

    return Spliterators.spliterator(elements, SPLITERATOR_CHARACTERISTICS);
  }

  @Override
  Object[] internalArray() {
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
