// Source-based slice around line 389
// Method: <com.google.common.testing.EqualsTesterTest: NamedObject named(String)>

      return o != null;
    }

    @Override
    public int hashCode() {
      return 0;
    }
  }

  private static NamedObject named(String name) {
    return new NamedObject(name);
  }

  private static class NamedObject {
    private final Set<String> peerNames = new HashSet<>();

    private final String name;

    NamedObject(String name) {
      this.name = Preconditions.checkNotNull(name);
