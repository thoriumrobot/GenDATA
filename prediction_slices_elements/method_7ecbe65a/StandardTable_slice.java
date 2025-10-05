// Source-based slice around line 666
// Method: <com.google.common.collect.StandardTable: Set rowKeySet()>


      @Override
      public boolean retainAll(Collection<?> c) {
        return removeFromColumnIf(Maps.valuePredicateOnEntries(not(in(c))));
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
