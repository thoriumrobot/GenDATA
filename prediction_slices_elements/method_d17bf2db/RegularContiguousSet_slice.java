// Source-based slice around line 120
// Method: <com.google.common.collect.RegularContiguousSet: boolean isPartialView()>

      }
    };
  }

  private static boolean equalsOrThrow(Comparable<?> left, @Nullable Comparable<?> right) {
    return right != null && Range.compareOrThrow(left, right) == 0;
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public C first() {
    // requireNonNull is safe because we checked the range is not empty in ContiguousSet.create.
    return requireNonNull(range.lowerBound.leastValueAbove(domain));
  }

  @Override
