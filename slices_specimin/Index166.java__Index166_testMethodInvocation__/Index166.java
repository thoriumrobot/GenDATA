import org.checkerframework.checker.index.qual.IndexFor;

public class Index166 {

    public void testMethodInvocation() {
        requiresIndex("012345", 5);
        requiresIndex("012345", 6);
    }
}
