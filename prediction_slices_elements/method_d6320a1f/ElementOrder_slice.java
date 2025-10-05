// Source-based slice around line 175
// Method: <com.google.common.graph.ElementOrder: int hashCode()>

    if (!(obj instanceof ElementOrder)) {
      return false;
    }

    ElementOrder<?> other = (ElementOrder<?>) obj;
    return (type == other.type) && Objects.equals(comparator, other.comparator);
  }

  @Override
  public int hashCode() {
    return Objects.hash(type, comparator);
  }

  @Override
  public String toString() {
    ToStringHelper helper = MoreObjects.toStringHelper(this).add("type", type);
    if (comparator != null) {
      helper.add("comparator", comparator);
    }
    return helper.toString();
