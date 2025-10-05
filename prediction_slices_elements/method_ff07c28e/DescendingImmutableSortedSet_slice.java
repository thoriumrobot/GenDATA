// Source-based slice around line 117
// Method: <com.google.common.collect.DescendingImmutableSortedSet: boolean isPartialView()>

    int index = forward.indexOf(target);
    if (index == -1) {
      return index;
    } else {
      return size() - 1 - index;
    }
  }

  @Override
  boolean isPartialView() {
    return forward.isPartialView();
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return super.writeReplace();
  }
