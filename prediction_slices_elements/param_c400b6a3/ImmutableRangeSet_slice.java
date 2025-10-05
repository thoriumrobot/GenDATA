// Source-based slice around line 531
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet subRangeSet(Range)>

                Object writeReplace() {
          return super.writeReplace();
        }
      };
    }
  }

  /** Returns a view of the intersection of this range set with the given range. */
  @Override
  public ImmutableRangeSet<C> subRangeSet(Range<C> range) {
    if (!isEmpty()) {
      Range<C> span = span();
      if (range.encloses(span)) {
        return this;
      } else if (range.isConnected(span)) {
        return new ImmutableRangeSet<>(intersectRanges(range));
      }
    }
    return of();
  }
