// Source-based slice around line 87
// Method: <com.google.common.collect.RegularImmutableTable: ImmutableCollection createValues()>

    @GwtIncompatible
        Object writeReplace() {
      return super.writeReplace();
    }
  }

  abstract V getValue(int iterationIndex);

  @Override
  final ImmutableCollection<V> createValues() {
    return isEmpty() ? ImmutableList.of() : new Values();
  }

  @WeakOuter
  private final class Values extends ImmutableList<V> {
    @Override
    public int size() {
      return RegularImmutableTable.this.size();
    }

