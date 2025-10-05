// Source-based slice around line 77
// Method: com.google.common.collect.StandardTable.factory

 *
 * <p>Note that this implementation is not synchronized. If multiple threads access this table
 * concurrently and one of the threads modifies the table, it must be synchronized externally.
 *
 * @author Jared Levy
 */
@GwtCompatible
class StandardTable<R, C, V> extends AbstractTable<R, C, V> implements Serializable {
  final Map<R, Map<C, V>> backingMap;
  final Supplier<? extends Map<C, V>> factory;

  StandardTable(Map<R, Map<C, V>> backingMap, Supplier<? extends Map<C, V>> factory) {
    this.backingMap = backingMap;
    this.factory = factory;
  }

  // Accessors

  @Override
  public boolean contains(@Nullable Object rowKey, @Nullable Object columnKey) {
