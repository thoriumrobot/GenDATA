// Source-based slice around line 38
// Method: com.google.common.collect.SingletonImmutableList.element

/**
 * Implementation of {@link ImmutableList} with exactly one element.
 *
 * @author Hayward Chan
 */
@GwtCompatible
@SuppressWarnings("serial") // uses writeReplace(), not default serialization
final class SingletonImmutableList<E> extends ImmutableList<E> {

  final transient E element;

  SingletonImmutableList(E element) {
    this.element = checkNotNull(element);
  }

  @Override
  public E get(int index) {
    Preconditions.checkElementIndex(index, 1);
    return element;
  }
