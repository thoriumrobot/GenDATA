// Source-based slice around line 57
// Method: <com.google.common.collect.ImmutableAsList: boolean isPartialView()>

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
  @J2ktIncompatible
  private static final class SerializedForm implements Serializable {
    final ImmutableCollection<?> collection;

    SerializedForm(ImmutableCollection<?> collection) {
