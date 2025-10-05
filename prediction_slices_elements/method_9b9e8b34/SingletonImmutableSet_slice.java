// Source-based slice around line 82
// Method: <com.google.common.collect.SingletonImmutableSet: String toString()>

    return offset + 1;
  }

  @Override
  public final int hashCode() {
    return element.hashCode();
  }

  @Override
  public String toString() {
    return '[' + element.toString() + ']';
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
    return super.writeReplace();
