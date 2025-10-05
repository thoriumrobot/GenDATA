// Source-based slice around line 443
// Method: <com.google.common.collect.MinMaxPriorityQueue: MoveDesc fillHole(int,E)>

        // The trickled element is back before index, but the replaced element
        // has now been moved after index.
        return new MoveDesc<>(actualLastElement, changes.replaced);
      }
    }
    // Trickled element was after index to begin with, no adjustment needed.
    return changes;
  }

  private @Nullable MoveDesc<E> fillHole(int index, E toTrickle) {
    Heap heap = heapForIndex(index);
    // We consider elementData(index) a "hole", and we want to fill it
    // with the last element of the heap, toTrickle.
    // Since the last element of the heap is from the bottom level, we
    // optimistically fill index position with elements from lower levels,
    // moving the hole down. In most cases this reduces the number of
    // comparisons with toTrickle, but in some cases we will need to bubble it
    // all the way up again.
    int vacated = heap.fillHoleAt(index);
    // Try to see if toTrickle can be bubbled up min levels.
