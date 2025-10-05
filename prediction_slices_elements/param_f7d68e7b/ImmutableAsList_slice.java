// Source-based slice around line 40
// Method: <com.google.common.collect.ImmutableAsList: boolean contains(Object)>

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
    return delegateCollection().size();
  }

