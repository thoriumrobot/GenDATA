// Source-based slice around line 30
// Method: com.google.common.collect.JdkBackedImmutableSet.delegate


/**
 * ImmutableSet implementation backed by a JDK HashSet, used to defend against apparent hash
 * flooding. This implementation is never used on the GWT client side.
 *
 * @author Louis Wasserman
 */
@GwtIncompatible
final class JdkBackedImmutableSet<E> extends IndexedImmutableSet<E> {
  private final Set<?> delegate;
  private final ImmutableList<E> delegateList;

  JdkBackedImmutableSet(Set<?> delegate, ImmutableList<E> delegateList) {
    this.delegate = delegate;
    this.delegateList = delegateList;
  }

  @Override
  E get(int index) {
    return delegateList.get(index);
