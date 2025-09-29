import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForVarargs {

    void m() {
        get(1);
        get(1, "a", "b");
        get(2, "abc");
        String[] stringArg1 = new String[] { "a", "b" };
        String[] stringArg2 = new String[] { "c", "d", "e" };
        String[] stringArg3 = new String[] { "a", "b", "c" };
        method(1, stringArg1, stringArg2);
        method(2, stringArg3);
        get(1, stringArg1);
        get(3, stringArg2);
    }
}
