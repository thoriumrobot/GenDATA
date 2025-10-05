// Source-based slice around line 112
// Method: <com.google.common.collect.RegularImmutableSet: ImmutableList createAsList()>

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
  boolean isPartialView() {
    return false;
  }

  @Override
  public int hashCode() {
