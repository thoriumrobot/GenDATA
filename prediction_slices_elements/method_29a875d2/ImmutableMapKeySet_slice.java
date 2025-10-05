// Source-based slice around line 84
// Method: <com.google.common.collect.ImmutableMapKeySet: Object writeReplace()>

  boolean isPartialView() {
    return true;
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
    return super.writeReplace();
  }

  // No longer used for new writes, but kept so that old data can still be read.
  @GwtIncompatible
  @J2ktIncompatible
  @SuppressWarnings("unused")
  private static final class KeySetSerializedForm<K> implements Serializable {
    final ImmutableMap<K, ?> map;

