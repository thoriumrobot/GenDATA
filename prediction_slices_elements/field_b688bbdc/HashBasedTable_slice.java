// Source-based slice around line 106
// Method: com.google.common.collect.HashBasedTable.serialVersionUID

    HashBasedTable<R, C, V> result = create();
    result.putAll(table);
    return result;
  }

  HashBasedTable(Map<R, Map<C, V>> backingMap, Factory<C, V> factory) {
    super(backingMap, factory);
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
