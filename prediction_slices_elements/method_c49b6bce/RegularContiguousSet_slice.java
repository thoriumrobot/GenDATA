// Source-based slice around line 166
// Method: <com.google.common.collect.RegularContiguousSet: int size()>

          return super.writeReplace();
        }
      };
    } else {
      return super.createAsList();
    }
  }

  @Override
  public int size() {
    long distance = domain.distance(first(), last());
    return (distance >= Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) distance + 1;
  }

  @Override
  public boolean contains(@Nullable Object object) {
    if (object == null) {
      return false;
    }
    try {
