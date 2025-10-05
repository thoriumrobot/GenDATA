// Source-based slice around line 73
// Method: <com.google.common.collect.RegularImmutableSet: int size()>

      if (candidate == null) {
        return false;
      } else if (candidate.equals(target)) {
        return true;
      }
    }
  }

  @Override
  public int size() {
    return elements.length;
  }

  // We're careful to put only E instances into the array in the mainline.
  // (In the backport, we don't need this suppression, but we keep it to minimize diffs.)
  @SuppressWarnings("unchecked")
  @Override
  public UnmodifiableIterator<E> iterator() {
    return (UnmodifiableIterator<E>) Iterators.forArray(elements);
  }
