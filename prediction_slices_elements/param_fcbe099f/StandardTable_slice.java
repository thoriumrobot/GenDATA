// Source-based slice around line 455
// Method: <com.google.common.collect.StandardTable: Map column(C)>

    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>The returned map's views have iterators that don't support {@code remove()}.
   */
  @Override
  public Map<R, V> column(C columnKey) {
    return new Column(columnKey);
  }

  private final class Column extends ViewCachingAbstractMap<R, V> {
    final C columnKey;

    Column(C columnKey) {
      this.columnKey = checkNotNull(columnKey);
    }

