// Source-based slice around line 45
// Method: <com.google.common.collect.SingletonImmutableList: E get(int)>

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

  @Override
  public UnmodifiableIterator<E> iterator() {
    return singletonIterator(element);
  }

  @Override
