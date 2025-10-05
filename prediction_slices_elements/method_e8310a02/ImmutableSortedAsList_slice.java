// Source-based slice around line 102
// Method: <com.google.common.collect.ImmutableSortedAsList: Object writeReplace()>

        delegateList()::get,
        comparator());
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
    return super.writeReplace();
  }
}
