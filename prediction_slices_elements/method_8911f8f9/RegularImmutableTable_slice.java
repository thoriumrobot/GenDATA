// Source-based slice around line 40
// Method: <com.google.common.collect.RegularImmutableTable: Cell getCell(int)>

/**
 * An implementation of {@link ImmutableTable} holding an arbitrary number of cells.
 *
 * @author Gregory Kick
 */
@GwtCompatible
abstract class RegularImmutableTable<R, C, V> extends ImmutableTable<R, C, V> {
  RegularImmutableTable() {}

  abstract Cell<R, C, V> getCell(int iterationIndex);

  @Override
  final ImmutableSet<Cell<R, C, V>> createCellSet() {
    return isEmpty() ? ImmutableSet.of() : new CellSet();
  }

  @WeakOuter
  private final class CellSet extends IndexedImmutableSet<Cell<R, C, V>> {
    @Override
    public int size() {
