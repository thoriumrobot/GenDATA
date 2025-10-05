    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SameLenTripleThreat {
    @Positive
  public void foo(String[] vars) {
    @Positive
    String[] qrets = new String[vars.length];
    @Positive
    String @SameLen("vars") [] y = qrets;
    @Positive
    String[] indices = new String[vars.length];
    @Positive
    String @SameLen("qrets") [] x = indices;
    @Positive
  }

    @Positive
  String[] indices;

    @Positive
  public void foo2(String... vars) {
    @Positive
    String[] qrets = new String[vars.length];
    @Positive
    indices = new String[vars.length];
    @Positive
    String[] indicesLocal = new String[vars.length];
    @Positive
    for (int i = 0; i < qrets.length; i++) {
    @Positive
      indices[i] = "hello";
    @Positive
      indicesLocal[i] = "hello";
    @Positive
    }
    @Positive
  }

    @Positive
  public void foo3(String... vars) {
    @Positive
    String[] qrets = new String[vars.length];
    @Positive
    String[] indicesLocal = new String[vars.length];
    @Positive
    indices = new String[vars.length];
    @Positive
    for (int i = 0; i < qrets.length; i++) {
    @Positive
      indices[i] = "hello";
    @Positive
      indicesLocal[i] = "hello";
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
