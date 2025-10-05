// Source-based slice around line 69
// Method: <com.google.common.collect.HashBasedTable: HashBasedTable create()>

    @Override
    public Map<C, V> get() {
      return Maps.newLinkedHashMapWithExpectedSize(expectedSize);
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  /** Creates an empty {@code HashBasedTable}. */
  public static <R, C, V> HashBasedTable<R, C, V> create() {
    return new HashBasedTable<>(new LinkedHashMap<R, Map<C, V>>(), new Factory<C, V>(0));
  }

  /**
   * Creates an empty {@code HashBasedTable} with the specified map sizes.
   *
   * @param expectedRows the expected number of distinct row keys
   * @param expectedCellsPerRow the expected number of column key / value mappings in each row
   * @throws IllegalArgumentException if {@code expectedRows} or {@code expectedCellsPerRow} is
   *     negative
