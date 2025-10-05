// Source-based slice around line 85
// Method: <com.google.common.collect.JdkBackedImmutableMultiset: boolean isPartialView()>

    return (result == null) ? elementSet = new ElementSet<>(entries, this) : result;
  }

  @Override
  Entry<E> getEntry(int index) {
    return entries.get(index);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public int size() {
    return Ints.saturatedCast(size);
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
