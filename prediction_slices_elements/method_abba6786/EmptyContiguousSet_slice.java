// Source-based slice around line 41
// Method: <com.google.common.collect.EmptyContiguousSet: C first()>

 */
@GwtCompatible
@SuppressWarnings("rawtypes") // allow ungenerified Comparable types
final class EmptyContiguousSet<C extends Comparable> extends ContiguousSet<C> {
  EmptyContiguousSet(DiscreteDomain<C> domain) {
    super(domain);
  }

  @Override
  public C first() {
    throw new NoSuchElementException();
  }

  @Override
  public C last() {
    throw new NoSuchElementException();
  }

  @Override
  public int size() {
