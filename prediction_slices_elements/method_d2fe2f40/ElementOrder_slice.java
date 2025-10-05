// Source-based slice around line 203
// Method: <com.google.common.graph.ElementOrder: ElementOrder cast()>

      case STABLE:
        return Maps.newLinkedHashMapWithExpectedSize(expectedSize);
      case SORTED:
        return Maps.newTreeMap(comparator());
    }
    throw new AssertionError();
  }

  @SuppressWarnings("unchecked")
  <T1 extends T> ElementOrder<T1> cast() {
    return (ElementOrder<T1>) this;
  }
}
