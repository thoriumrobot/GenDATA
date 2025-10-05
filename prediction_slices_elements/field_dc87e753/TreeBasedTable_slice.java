// Source-based slice around line 75
// Method: com.google.common.collect.TreeBasedTable.columnComparator

 * <p>See the Guava User Guide article on <a href=
 * "https://github.com/google/guava/wiki/NewCollectionTypesExplained#table">{@code Table}</a>.
 *
 * @author Jared Levy
 * @author Louis Wasserman
 * @since 7.0
 */
@GwtCompatible
public class TreeBasedTable<R, C, V> extends StandardRowSortedTable<R, C, V> {
  private final Comparator<? super C> columnComparator;

  private static final class Factory<C, V> implements Supplier<Map<C, V>>, Serializable {
    final Comparator<? super C> comparator;

    Factory(Comparator<? super C> comparator) {
      this.comparator = comparator;
    }

    @Override
    public Map<C, V> get() {
