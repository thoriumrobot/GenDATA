// Source-based slice around line 113
// Method: <com.google.common.collect.EnumMultiset: void checkIsE(Object)>

      return index < enumConstants.length && enumConstants[index] == e;
    }
    return false;
  }

  /**
   * Returns {@code element} cast to {@code E}, if it actually is a nonnull E. Otherwise, throws
   * either a NullPointerException or a ClassCastException as appropriate.
   */
  private void checkIsE(Object element) {
    checkNotNull(element);
    if (!isActuallyE(element)) {
      throw new ClassCastException("Expected an " + type + " but got " + element);
    }
  }

  @Override
  int distinctElements() {
    return distinctElements;
  }
