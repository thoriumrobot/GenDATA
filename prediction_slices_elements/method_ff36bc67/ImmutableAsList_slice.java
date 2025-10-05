// Source-based slice around line 47
// Method: <com.google.common.collect.ImmutableAsList: int size()>


  @Override
  public boolean contains(@Nullable Object target) {
    // The collection's contains() is at least as fast as ImmutableList's
    // and is often faster.
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
