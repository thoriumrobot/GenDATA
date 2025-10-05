// Source-based slice around line 72
// Method: <com.google.common.collect.SingletonImmutableList: String toString()>

  }

  @Override
  public ImmutableList<E> subList(int fromIndex, int toIndex) {
    Preconditions.checkPositionIndexes(fromIndex, toIndex, 1);
    return (fromIndex == toIndex) ? ImmutableList.of() : this;
  }

  @Override
  public String toString() {
    return '[' + element.toString() + ']';
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
