// Source-based slice around line 35
// Method: com.google.common.collect.SingletonImmutableTable.singleValue

/**
 * An implementation of {@link ImmutableTable} that holds a single cell.
 *
 * @author Gregory Kick
 */
@GwtCompatible
final class SingletonImmutableTable<R, C, V> extends ImmutableTable<R, C, V> {
  final R singleRowKey;
  final C singleColumnKey;
  final V singleValue;

  SingletonImmutableTable(R rowKey, C columnKey, V value) {
    this.singleRowKey = checkNotNull(rowKey);
    this.singleColumnKey = checkNotNull(columnKey);
    this.singleValue = checkNotNull(value);
  }

  SingletonImmutableTable(Cell<R, C, V> cell) {
    this(cell.getRowKey(), cell.getColumnKey(), cell.getValue());
  }
