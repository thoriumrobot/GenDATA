// Source-based slice around line 145
// Method: com.google.common.collect.StandardRowSortedTable.serialVersionUID


    @Override
    public SortedMap<R, Map<C, V>> tailMap(R fromKey) {
      checkNotNull(fromKey);
      return new StandardRowSortedTable<R, C, V>(sortedBackingMap().tailMap(fromKey), factory)
          .rowMap();
    }
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
