// Source-based slice around line 38
// Method: <com.google.common.collect.IndexedImmutableSet: Spliterator spliterator()>

abstract class IndexedImmutableSet<E> extends ImmutableSet.CachingAsList<E> {
  abstract E get(int index);

  @Override
  public UnmodifiableIterator<E> iterator() {
    return asList().iterator();
  }

  @Override
  public Spliterator<E> spliterator() {
    return CollectSpliterators.indexed(size(), SPLITERATOR_CHARACTERISTICS, this::get);
  }

  @Override
  public void forEach(Consumer<? super E> consumer) {
    checkNotNull(consumer);
    int n = size();
    for (int i = 0; i < n; i++) {
      consumer.accept(get(i));
    }
