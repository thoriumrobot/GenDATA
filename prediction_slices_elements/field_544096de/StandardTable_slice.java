// Source-based slice around line 670
// Method: com.google.common.collect.StandardTable.columnKeySet

      }
    }
  }

  @Override
  public Set<R> rowKeySet() {
    return rowMap().keySet();
  }

  @LazyInit private transient @Nullable Set<C> columnKeySet;

  /**
   * {@inheritDoc}
   *
   * <p>The returned set has an iterator that does not support {@code remove()}.
   *
   * <p>The set's iterator traverses the columns of the first row, the columns of the second row,
   * etc., skipping any columns that have appeared previously.
   */
  @Override
