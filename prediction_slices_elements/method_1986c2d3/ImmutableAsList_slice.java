// Source-based slice around line 52
// Method: <com.google.common.collect.ImmutableAsList: boolean isEmpty()>

    return delegateCollection().contains(target);
  }

  @Override
  public int size() {
    return delegateCollection().size();
  }

  @Override
  public boolean isEmpty() {
    return delegateCollection().isEmpty();
  }

  @Override
  boolean isPartialView() {
    return delegateCollection().isPartialView();
  }

  /** Serialized form that leads to the same performance as the original list. */
  @GwtIncompatible
