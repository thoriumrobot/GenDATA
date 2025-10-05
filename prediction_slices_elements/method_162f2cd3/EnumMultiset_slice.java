// Source-based slice around line 208
// Method: <com.google.common.collect.EnumMultiset: void clear()>

    if (oldCount == 0 && count > 0) {
      distinctElements++;
    } else if (oldCount > 0 && count == 0) {
      distinctElements--;
    }
    return oldCount;
  }

  @Override
  public void clear() {
    Arrays.fill(counts, 0);
    size = 0;
    distinctElements = 0;
  }

  abstract class Itr<T> implements Iterator<T> {
    int index = 0;
    int toRemove = -1;

    abstract T output(int index);
