// Source-based slice around line 49
// Method: <com.google.common.collect.JdkBackedImmutableSet: boolean isPartialView()>

    return delegateList.get(index);
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return delegate.contains(object);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public int size() {
    return delegateList.size();
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
