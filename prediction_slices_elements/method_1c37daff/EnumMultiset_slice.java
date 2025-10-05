// Source-based slice around line 254
// Method: <com.google.common.collect.EnumMultiset: Iterator elementIterator()>

        distinctElements--;
        size -= counts[toRemove];
        counts[toRemove] = 0;
      }
      toRemove = -1;
    }
  }

  @Override
  Iterator<E> elementIterator() {
    return new Itr<E>() {
      @Override
      E output(int index) {
        return enumConstants[index];
      }
    };
  }

  @Override
  Iterator<Entry<E>> entryIterator() {
