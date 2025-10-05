// Source-based slice around line 62
// Method: <com.google.common.collect.CollectPreconditions: void checkRemove(boolean)>

    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive but was: " + value);
    }
  }

  /**
   * Precondition tester for {@code Iterator.remove()} that throws an exception with a consistent
   * error message.
   */
  static void checkRemove(boolean canRemove) {
    checkState(canRemove, "no calls to next() since the last call to remove()");
  }

  private CollectPreconditions() {}
}
