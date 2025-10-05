    @Positive
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

    @Positive
public class RepeatLTLengthOf {

    @Positive
  protected String value1;
    @Positive
  protected String value2;
    @Positive
  protected String value3;
    @Positive
  protected int v1;
    @Positive
  protected int v2;
    @Positive
  protected int v3;

    @Positive
  public void func1() {
    @Positive
    v1 = value1.length() - 3;
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
  }

    @Positive
  public boolean func2() {
    @Positive
    v1 = value1.length() - 3;
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
    return true;
    @Positive
  }

    @Positive
  public void client1() {
    @Positive
    withpostconditionsfunc1();
    @Positive
  }

    @Positive
  public boolean client2() {
    @Positive
    return withcondpostconditionsfunc2();
    @Positive
  }

    @Positive
  })
    @Positive
  public void client3() {
    @Positive
    withpostconditionfunc1();
    @Positive
  }

    @Positive
  })
    @Positive
  public boolean client4() {
    @Positive
    return withcondpostconditionfunc2();
    @Positive
  }

    @Positive
  public void withpostconditionsfunc1() {
    @Positive
    v1 = value1.length() - 3;
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
  }

    @Positive
  public boolean withcondpostconditionsfunc2() {
    @Positive
    v1 = value1.length() - 3;
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
    return true;
    @Positive
  }

    @Positive
  })
    @Positive
  public void withpostconditionfunc1() {
    @Positive
    v1 = value1.length() - 3;
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
  }

    @Positive
  })
    @Positive
  public boolean withcondpostconditionfunc2() {
    @Positive
    v1 = value1.length() - 3;
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
    return true;
    @Positive
  }
    @Positive
}
