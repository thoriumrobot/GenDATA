// Source-based slice around line 37
// Method: <com.google.common.collect.ImmutableAsList: ImmutableCollection delegateCollection()>

 * List returned by {@link ImmutableCollection#asList} that delegates {@code contains} checks to the
 * backing collection.
 *
 * @author Jared Levy
 * @author Louis Wasserman
 */
@GwtCompatible
@SuppressWarnings("serial")
abstract class ImmutableAsList<E> extends ImmutableList<E> {
  abstract ImmutableCollection<E> delegateCollection();

  @Override
  public boolean contains(@Nullable Object target) {
    // The collection's contains() is at least as fast as ImmutableList's
    // and is often faster.
    return delegateCollection().contains(target);
  }

  @Override
  public int size() {
