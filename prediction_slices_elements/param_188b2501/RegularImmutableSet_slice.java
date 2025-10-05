// Source-based slice around line 106
// Method: <com.google.common.collect.RegularImmutableSet: int copyIntoArray(Object[],int)>

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
  ImmutableList<E> createAsList() {
    return (table.length == 0) ? ImmutableList.of() : new RegularImmutableAsList<>(this, elements);
  }

  @Override
