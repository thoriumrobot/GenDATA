// Source-based slice around line 186
// Method: <com.google.common.collect.RegularContiguousSet: boolean containsAll(Collection)>

      @SuppressWarnings("unchecked") // The worst case is usually CCE, which we catch.
      C c = (C) object;
      return range.contains(c);
    } catch (ClassCastException e) {
      return false;
    }
  }

  @Override
  public boolean containsAll(Collection<?> targets) {
    return Collections2.containsAllImpl(this, targets);
  }

  @Override
  public boolean isEmpty() {
    return false;
  }

  @Override
  @SuppressWarnings("unchecked") // TODO(cpovirk): Use a shared unsafeCompare method.
