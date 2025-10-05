// Source-based slice around line 115
// Method: <com.google.common.collect.RegularContiguousSet: boolean equalsOrThrow(Comparable,Comparable)>

      final C first = first();

      @Override
      protected @Nullable C computeNext(C previous) {
        return equalsOrThrow(previous, first) ? null : domain.previous(previous);
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
