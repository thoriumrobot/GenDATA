// Source-based slice around line 117
// Method: <com.google.common.collect.RegularImmutableSet: boolean isPartialView()>

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
    return hashCode;
  }

  @Override
  boolean isHashCodeFast() {
