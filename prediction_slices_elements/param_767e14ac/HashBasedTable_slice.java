// Source-based slice around line 96
// Method: <com.google.common.collect.HashBasedTable: HashBasedTable create(Table)>


  /**
   * Creates a {@code HashBasedTable} with the same mappings as the specified table.
   *
   * @param table the table to copy
   * @throws NullPointerException if any of the row keys, column keys, or values in {@code table} is
   *     null
   */
  public static <R, C, V> HashBasedTable<R, C, V> create(
      Table<? extends R, ? extends C, ? extends V> table) {
    HashBasedTable<R, C, V> result = create();
    result.putAll(table);
    return result;
  }

  HashBasedTable(Map<R, Map<C, V>> backingMap, Factory<C, V> factory) {
    super(backingMap, factory);
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
