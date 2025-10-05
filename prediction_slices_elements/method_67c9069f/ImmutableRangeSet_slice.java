// Source-based slice around line 882
// Method: <com.google.common.collect.ImmutableRangeSet: Object writeReplace()>

      } else if (ranges.equals(ImmutableList.of(Range.all()))) {
        return all();
      } else {
        return new ImmutableRangeSet<C>(ranges);
      }
    }
  }

  @J2ktIncompatible // java.io.ObjectInputStream
  Object writeReplace() {
    return new SerializedForm<C>(ranges);
  }

  @J2ktIncompatible // java.io.ObjectInputStream
  private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }
}
