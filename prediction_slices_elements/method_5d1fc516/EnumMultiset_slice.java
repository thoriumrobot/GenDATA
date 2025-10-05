// Source-based slice around line 264
// Method: <com.google.common.collect.EnumMultiset: Iterator entryIterator()>

    return new Itr<E>() {
      @Override
      E output(int index) {
        return enumConstants[index];
      }
    };
  }

  @Override
  Iterator<Entry<E>> entryIterator() {
    return new Itr<Entry<E>>() {
      @Override
      Entry<E> output(int index) {
        return new Multisets.AbstractEntry<E>() {
          @Override
          public E getElement() {
            return enumConstants[index];
          }

          @Override
