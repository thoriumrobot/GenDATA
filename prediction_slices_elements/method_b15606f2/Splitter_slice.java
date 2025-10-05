// Source-based slice around line 387
// Method: <com.google.common.base.Splitter: Iterator splittingIterator(CharSequence)>

      public String toString() {
        return Joiner.on(", ")
            .appendTo(new StringBuilder().append('['), this)
            .append(']')
            .toString();
      }
    };
  }

  private Iterator<String> splittingIterator(CharSequence sequence) {
    return strategy.iterator(this, sequence);
  }

  /**
   * Splits {@code sequence} into string components and returns them as an immutable list. If you
   * want an {@link Iterable} which may be lazily evaluated, use {@link #split(CharSequence)}.
   *
   * @param sequence the sequence of characters to split
   * @return an immutable list of the segments split from the parameter
   * @since 15.0
