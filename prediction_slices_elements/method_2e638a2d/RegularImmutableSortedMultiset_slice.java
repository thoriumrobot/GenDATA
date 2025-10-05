// Source-based slice around line 131
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: boolean isPartialView()>

      return this;
    } else {
      RegularImmutableSortedSet<E> subElementSet = elementSet.getSubSet(from, to);
      return new RegularImmutableSortedMultiset<>(
          subElementSet, cumulativeCounts, offset + from, to - from);
    }
  }

  @Override
  boolean isPartialView() {
    return offset > 0 || length < cumulativeCounts.length - 1;
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return super.writeReplace();
  }
